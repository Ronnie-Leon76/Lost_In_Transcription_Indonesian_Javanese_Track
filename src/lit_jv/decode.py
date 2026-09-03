"""Decoding: batched Qwen3-ASR inference with confidence-triggered fallbacks.

Backends, in order of preference:

1. **vLLM offline** (``vllm==0.23.0`` is in the runtime and supports
   ``Qwen3ASRForConditionalGeneration`` natively). Continuous batching turns
   14.6 hours of test audio into a few minutes of wall clock, which is what buys
   the budget for a multi-pass ensemble.
2. **transformers** via the ``qwen_asr`` package, as a fallback and for local
   debugging on machines without a working vLLM.

The interesting part is not the batching, it is the failure handling. An
LLM-decoder ASR model does not degrade gracefully on a 35-second conversational
turn: it truncates, or it loops. Either failure costs tens of word errors on a
single clip — far more than any orthographic detail. So every hypothesis is
checked against two signals, and suspicious clips are re-decoded:

* **average log-probability**, the criterion Whisper uses for its temperature
  fallback (Radford et al., 2022, §4.5);
* **a words-per-second prior** fitted on training transcripts, which catches
  truncation that the model is confidently wrong about — the case average
  log-probability misses.

The re-decode ladder tries a different prompt, then sampling with MBR over
several draws, then splitting the clip at a pause and decoding both halves.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .audio import SAMPLE_RATE, load_audio_16k, split_at_quietest
from .combine import LengthPrior, mbr_select
from .prompts import PromptConfig, build_prompt, parse_generation


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class DecodeConfig:
    """Everything that affects a decoding pass."""

    prompt: PromptConfig = field(default_factory=PromptConfig)
    #: ``max_new_tokens = tokens_per_second * duration + token_headroom``.
    #: Fitted from training data: ~1.9 words/s and ~2.5 subword tokens/word,
    #: with generous headroom because truncation is far more expensive than a
    #: few wasted KV-cache slots.
    tokens_per_second: float = 8.0
    token_headroom: int = 80
    max_tokens_cap: int = 640
    temperature: float = 0.0
    #: Batch size for the transformers backend only; vLLM schedules its own.
    batch_size: int = 16
    name: str = "pass"

    def max_tokens(self, duration_s: float | None) -> int:
        if duration_s is None or duration_s != duration_s or duration_s <= 0:
            return self.max_tokens_cap
        return int(min(self.max_tokens_cap, self.tokens_per_second * duration_s + self.token_headroom))


@dataclass
class Hypothesis:
    audio_filename: str
    text: str
    avg_logprob: float = float("nan")
    finish_reason: str = ""
    n_tokens: int = 0
    pass_name: str = ""
    rescued_by: str = ""


@dataclass
class FallbackConfig:
    """When to re-decode, and what to try."""

    enabled: bool = True
    #: Below this average log-probability the hypothesis is treated as unsafe.
    min_avg_logprob: float = -0.85
    #: Absolute z-score of the words-per-second prior beyond which we re-decode.
    max_length_z: float = 3.0
    #: A hypothesis that stopped because it ran out of tokens is always suspect.
    rescue_on_length_finish: bool = True
    sampling_draws: int = 4
    sampling_temperature: float = 0.4
    split_long_clips: bool = True
    max_rescue_fraction: float = 0.15


# --------------------------------------------------------------------------
# GPU / AutoConfig hygiene
# --------------------------------------------------------------------------


def release_cuda() -> None:
    """Drop cached CUDA blocks so the next engine can see real free memory.

    Training and a failed vLLM start both leave the GPU allocated. vLLM then
    asks for ``gpu_memory_utilization * total`` of *currently free* memory and
    dies if that number is larger than what is left.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass


def fit_vllm_gpu_memory_utilization(
    desired: float = 0.85,
    floor: float = 0.20,
    reserve_gib: float = 1.5,
) -> float:
    """Cap vLLM's reservation at what is actually free right now."""
    try:
        import torch
    except ImportError:
        return desired
    if not torch.cuda.is_available():
        return desired
    free_b, total_b = torch.cuda.mem_get_info()
    free_gib, total_gib = free_b / 1024**3, total_b / 1024**3
    usable = max(0.0, (free_gib - reserve_gib) / total_gib)
    fitted = min(float(desired), max(floor, usable))
    return round(fitted, 3)


def reclaim_qwen3_asr_transformers_classes() -> None:
    """Undo vLLM's AutoConfig hijack of the ``qwen3_asr`` model type.

    Importing vLLM registers ``vllm.transformers_utils.configs.qwen3_asr.Qwen3ASRConfig``.
    ``qwen_asr.Qwen3ASRModel.from_pretrained`` then loads that class, and
    ``AutoModel`` refuses it ("Unrecognized configuration class ... Qwen3ASRConfig").
    Put the transformers / qwen_asr class back.
    """
    import importlib

    try:
        from transformers import AutoConfig, AutoModel
    except ImportError:
        return

    candidates = (
        ("qwen_asr.core.transformers_backend", "Qwen3ASRConfig", "Qwen3ASRForConditionalGeneration"),
        ("transformers.models.qwen3_asr.configuration_qwen3_asr", "Qwen3ASRConfig", None),
        ("transformers.models.qwen3_asr.modeling_qwen3_asr", None, "Qwen3ASRForConditionalGeneration"),
    )
    config_cls = model_cls = None
    for module_name, config_name, model_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if config_name and config_cls is None:
            cls = getattr(module, config_name, None)
            if cls is not None and "vllm" not in getattr(cls, "__module__", ""):
                config_cls = cls
        if model_name and model_cls is None:
            cls = getattr(module, model_name, None)
            if cls is not None and "vllm" not in getattr(cls, "__module__", ""):
                model_cls = cls
    if config_cls is None:
        return
    try:
        AutoConfig.register("qwen3_asr", config_cls, exist_ok=True)
    except TypeError:
        AutoConfig.register("qwen3_asr", config_cls)
    if model_cls is not None:
        try:
            AutoModel.register(config_cls, model_cls, exist_ok=True)
        except TypeError:
            try:
                AutoModel.register(config_cls, model_cls)
            except Exception:
                pass


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class Qwen3AsrBackend:
    """Common interface: prompts + waveforms in, raw generations out."""

    def generate(
        self,
        prompts: Sequence[str],
        waveforms: Sequence[Any],
        max_tokens: Sequence[int],
        temperature: float = 0.0,
        n: int = 1,
        seed: int | None = None,
    ) -> list[list[dict]]:
        raise NotImplementedError


class VllmBackend(Qwen3AsrBackend):
    """vLLM offline engine.

    ``multi_modal_data`` for audio changed shape across vLLM releases (bare
    array vs ``(array, sample_rate)`` tuple), so the payload form is probed once
    on the first call and then reused. Getting this wrong is a hard crash on the
    first batch of a two-hour job, which is not the place to find out.
    """

    def __init__(
        self,
        model_path: str | Path,
        gpu_memory_utilization: float = 0.86,
        max_model_len: int = 2048,
        max_num_seqs: int = 32,
        dtype: str = "bfloat16",
        limit_mm_per_prompt: dict | None = None,
        **engine_kwargs: Any,
    ):
        from vllm import LLM

        release_cuda()
        fitted = fit_vllm_gpu_memory_utilization(gpu_memory_utilization)
        if fitted + 1e-6 < gpu_memory_utilization:
            print(
                f"vLLM gpu_memory_utilization {gpu_memory_utilization:.2f} -> {fitted:.2f} "
                "(capped to currently free GPU memory)"
            )
        kwargs: dict[str, Any] = dict(
            model=str(model_path),
            gpu_memory_utilization=fitted,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            trust_remote_code=True,
            limit_mm_per_prompt=limit_mm_per_prompt or {"audio": 1},
        )
        kwargs.update(engine_kwargs)
        self.llm = LLM(**kwargs)
        self._payload_form: str | None = None

    def _payload(self, wav, form: str) -> dict:
        if form == "array":
            return {"audio": [wav]}
        if form == "tuple":
            return {"audio": [(wav, SAMPLE_RATE)]}
        raise ValueError(form)

    def generate(
        self,
        prompts: Sequence[str],
        waveforms: Sequence[Any],
        max_tokens: Sequence[int],
        temperature: float = 0.0,
        n: int = 1,
        seed: int | None = None,
    ) -> list[list[dict]]:
        from vllm import SamplingParams

        sampling = [
            SamplingParams(
                temperature=temperature,
                top_p=1.0 if temperature == 0 else 0.95,
                max_tokens=int(mt),
                n=n,
                seed=seed,
                logprobs=None,
            )
            for mt in max_tokens
        ]

        forms = [self._payload_form] if self._payload_form else ["array", "tuple"]
        last_error: Exception | None = None
        for form in forms:
            try:
                requests = [
                    {"prompt": p, "multi_modal_data": self._payload(w, form)}
                    for p, w in zip(prompts, waveforms)
                ]
                outputs = self.llm.generate(requests, sampling_params=sampling, use_tqdm=False)
                self._payload_form = form
                break
            except Exception as exc:  # pragma: no cover - depends on vLLM build
                last_error = exc
                continue
        else:
            raise RuntimeError(
                "vLLM rejected both audio payload forms for Qwen3-ASR"
            ) from last_error

        results: list[list[dict]] = []
        for output in outputs:
            candidates = []
            for completion in output.outputs:
                n_tokens = len(getattr(completion, "token_ids", ()) or ())
                cumulative = getattr(completion, "cumulative_logprob", None)
                avg = (
                    float(cumulative) / n_tokens
                    if cumulative is not None and n_tokens
                    else float("nan")
                )
                candidates.append(
                    {
                        "text": completion.text,
                        "avg_logprob": avg,
                        "finish_reason": getattr(completion, "finish_reason", "") or "",
                        "n_tokens": n_tokens,
                    }
                )
            results.append(candidates)
        return results

    def close(self) -> None:
        self.llm = None
        release_cuda()


class TransformersBackend(Qwen3AsrBackend):
    """transformers fallback built on the ``qwen_asr`` model classes.

    Prompts are passed as raw strings rather than going through
    ``Qwen3ASRModel.transcribe`` so that this backend and the vLLM backend send
    byte-identical text to the model.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ):
        import torch
        from qwen_asr.core.transformers_backend import (
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRProcessor,
        )

        reclaim_qwen3_asr_transformers_classes()
        release_cuda()
        torch_dtype = getattr(torch, dtype)
        kwargs: dict[str, Any] = {"dtype": torch_dtype}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            str(model_path), **kwargs
        ).to(device)
        self.model.eval()
        self.processor = Qwen3ASRProcessor.from_pretrained(str(model_path), fix_mistral_regex=True)
        self.device = device
        self.torch = torch

    def generate(
        self,
        prompts: Sequence[str],
        waveforms: Sequence[Any],
        max_tokens: Sequence[int],
        temperature: float = 0.0,
        n: int = 1,
        seed: int | None = None,
    ) -> list[list[dict]]:
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.processor(
            text=list(prompts), audio=list(waveforms), return_tensors="pt", padding=True
        )
        inputs = inputs.to(self.device).to(self.model.dtype)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max(max_tokens)),
            "num_return_sequences": n,
        }
        if temperature and temperature > 0:
            generate_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            generate_kwargs.update(do_sample=False)
        with torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)
        sequences = getattr(output, "sequences", output)
        prompt_len = inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(
            sequences[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        results: list[list[dict]] = []
        for i in range(len(prompts)):
            group = decoded[i * n : (i + 1) * n]
            results.append(
                [
                    {"text": t, "avg_logprob": float("nan"), "finish_reason": "", "n_tokens": 0}
                    for t in group
                ]
            )
        return results

    def close(self) -> None:
        self.model = None
        self.processor = None
        release_cuda()


def build_backend(
    model_path: str | Path,
    prefer: str = "vllm",
    vllm_kwargs: dict | None = None,
    transformers_kwargs: dict | None = None,
    log: Callable[[str], None] = print,
) -> Qwen3AsrBackend:
    """Instantiate the best available backend, reporting which one was used."""
    reclaim_qwen3_asr_transformers_classes()
    release_cuda()
    order = ["vllm", "transformers"] if prefer == "vllm" else ["transformers", "vllm"]
    errors: list[str] = []
    for name in order:
        try:
            if name == "vllm":
                backend = VllmBackend(model_path, **(vllm_kwargs or {}))
            else:
                backend = TransformersBackend(model_path, **(transformers_kwargs or {}))
            log(f"decoder backend: {name}")
            return backend
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no usable decoding backend:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------


class Qwen3AsrDecoder:
    """Runs decoding passes over a manifest and applies the rescue ladder."""

    def __init__(
        self,
        backend: Qwen3AsrBackend,
        length_prior: LengthPrior | None = None,
        fallback: FallbackConfig | None = None,
        log: Callable[[str], None] = print,
    ):
        self.backend = backend
        self.length_prior = length_prior or LengthPrior()
        self.fallback = fallback or FallbackConfig()
        self.log = log

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()
        else:
            release_cuda()

    # -- audio ------------------------------------------------------------

    @staticmethod
    def load_batch(rows: Sequence[dict]) -> list[Any]:
        return [load_audio_16k(row["audio_path"]) for row in rows]

    # -- one pass ---------------------------------------------------------

    def decode_pass(
        self,
        rows: Sequence[dict],
        config: DecodeConfig,
        waveforms: Sequence[Any] | None = None,
        chunk_size: int = 64,
        progress_every: int = 512,
    ) -> list[Hypothesis]:
        """Greedy decode of every row with one prompt configuration."""
        hypotheses: list[Hypothesis] = []
        started = time.time()
        for start in range(0, len(rows), chunk_size):
            batch = rows[start : start + chunk_size]
            wavs = (
                list(waveforms[start : start + chunk_size])
                if waveforms is not None
                else self.load_batch(batch)
            )
            prompts = [
                build_prompt(config.prompt, language_field=row.get("language"))
                for row in batch
            ]
            durations = [self._duration(row, wav) for row, wav in zip(batch, wavs)]
            max_tokens = [config.max_tokens(d) for d in durations]
            outputs = self.backend.generate(
                prompts, wavs, max_tokens, temperature=config.temperature, n=1
            )
            for row, out in zip(batch, outputs):
                best = out[0]
                hypotheses.append(
                    Hypothesis(
                        audio_filename=row["audio_filename"],
                        text=parse_generation(best["text"], config.prompt.force_language),
                        avg_logprob=best["avg_logprob"],
                        finish_reason=best["finish_reason"],
                        n_tokens=best["n_tokens"],
                        pass_name=config.name,
                    )
                )
            done = start + len(batch)
            if progress_every and done % progress_every < chunk_size:
                rate = done / max(1e-6, time.time() - started)
                self.log(f"  [{config.name}] {done}/{len(rows)} clips ({rate:.1f} clips/s)")
        self.log(f"  [{config.name}] done in {time.time() - started:.0f}s")
        return hypotheses

    @staticmethod
    def _duration(row: dict, wav: Any) -> float:
        duration = row.get("duration", float("nan"))
        if duration is None or duration != duration:
            try:
                return len(wav) / SAMPLE_RATE
            except TypeError:
                return float("nan")
        return float(duration)

    # -- rescue ladder ----------------------------------------------------

    def is_suspicious(self, hypothesis: Hypothesis, duration_s: float) -> str:
        """Return a reason string when the hypothesis should be re-decoded."""
        fb = self.fallback
        if not fb.enabled:
            return ""
        if not hypothesis.text.strip():
            return "empty"
        if fb.rescue_on_length_finish and hypothesis.finish_reason == "length":
            return "hit-token-limit"
        z = self.length_prior.z_score(hypothesis.text, duration_s)
        if abs(z) > fb.max_length_z:
            return f"length-z={z:+.1f}"
        if (
            hypothesis.avg_logprob == hypothesis.avg_logprob
            and hypothesis.avg_logprob < fb.min_avg_logprob
        ):
            return f"logprob={hypothesis.avg_logprob:.2f}"
        return ""

    def rescue(
        self,
        rows: Sequence[dict],
        hypotheses: Sequence[Hypothesis],
        config: DecodeConfig,
        waveforms: Sequence[Any] | None = None,
    ) -> list[Hypothesis]:
        """Re-decode the hypotheses that failed the confidence checks."""
        if not self.fallback.enabled:
            return list(hypotheses)

        by_name = {row["audio_filename"]: (i, row) for i, row in enumerate(rows)}
        out = list(hypotheses)
        flagged: list[tuple[int, dict, Hypothesis, str]] = []
        for slot, hypothesis in enumerate(out):
            index, row = by_name[hypothesis.audio_filename]
            wav = waveforms[index] if waveforms is not None else None
            duration = self._duration(row, wav)
            reason = self.is_suspicious(hypothesis, duration)
            if reason:
                flagged.append((slot, row, hypothesis, reason))

        cap = int(self.fallback.max_rescue_fraction * len(out))
        if len(flagged) > cap:
            # A large flagged set means the thresholds are wrong for this data,
            # not that hundreds of clips failed. Re-decoding everything would
            # blow the time budget, so rescue the worst offenders only.
            self.log(
                f"  [rescue] {len(flagged)} flagged (> {cap} cap); "
                "rescuing the least confident only"
            )
            flagged.sort(
                key=lambda item: (
                    item[2].avg_logprob if item[2].avg_logprob == item[2].avg_logprob else -9.9
                )
            )
            flagged = flagged[:cap]
        if not flagged:
            self.log("  [rescue] nothing flagged")
            return out

        self.log(f"  [rescue] re-decoding {len(flagged)} clips")
        for slot, row, hypothesis, reason in flagged:
            index, _ = by_name[row["audio_filename"]]
            wav = (
                waveforms[index]
                if waveforms is not None
                else load_audio_16k(row["audio_path"])
            )
            duration = self._duration(row, wav)
            candidates: list[tuple[str, str]] = [(hypothesis.text, "original")]

            # Rung 1: free language identification instead of a forced prefix.
            alt_prompt = PromptConfig(**{**config.prompt.__dict__, "force_language": None})
            candidates.append(
                (self._single(alt_prompt, wav, config.max_tokens(duration)), "free-lid")
            )

            # Rung 2: sample a few draws and take the MBR centroid.
            if self.fallback.sampling_draws > 1:
                draws = self._multi(
                    config.prompt,
                    wav,
                    config.max_tokens(duration),
                    n=self.fallback.sampling_draws,
                    temperature=self.fallback.sampling_temperature,
                )
                draws = [d for d in draws if d.strip()]
                if draws:
                    candidates.append((mbr_select(draws).text, "sampled"))

            # Rung 3: cut the clip at a pause and decode both halves.
            if self.fallback.split_long_clips and duration == duration and duration > 12:
                halves = split_at_quietest(wav)
                if len(halves) == 2:
                    parts = [
                        self._single(config.prompt, half, config.max_tokens(len(half) / SAMPLE_RATE))
                        for half in halves
                    ]
                    joined = " ".join(p for p in parts if p.strip())
                    candidates.append((joined, "split"))

            scored = [
                (abs(self.length_prior.z_score(text, duration)), text, tag)
                for text, tag in candidates
                if text.strip()
            ]
            if not scored:
                continue
            _, best_text, best_tag = min(scored, key=lambda item: item[0])
            if best_tag != "original":
                out[slot] = Hypothesis(
                    audio_filename=hypothesis.audio_filename,
                    text=best_text,
                    avg_logprob=hypothesis.avg_logprob,
                    finish_reason=hypothesis.finish_reason,
                    n_tokens=hypothesis.n_tokens,
                    pass_name=hypothesis.pass_name,
                    rescued_by=f"{best_tag}({reason})",
                )
        return out

    def _single(self, prompt: PromptConfig, wav: Any, max_tokens: int) -> str:
        outputs = self.backend.generate(
            [build_prompt(prompt)], [wav], [max_tokens], temperature=0.0, n=1
        )
        return parse_generation(outputs[0][0]["text"], prompt.force_language)

    def _multi(
        self, prompt: PromptConfig, wav: Any, max_tokens: int, n: int, temperature: float
    ) -> list[str]:
        outputs = self.backend.generate(
            [build_prompt(prompt)], [wav], [max_tokens], temperature=temperature, n=n, seed=1234
        )
        return [
            parse_generation(candidate["text"], prompt.force_language)
            for candidate in outputs[0]
        ]


# --------------------------------------------------------------------------
# Timing helper
# --------------------------------------------------------------------------


def estimate_budget(
    n_clips: int, audio_hours: float, clips_per_second: float, load_seconds: float = 240.0
) -> dict:
    """Project wall-clock time for a multi-pass plan against the 2-hour limit."""
    decode_seconds = n_clips / max(clips_per_second, 1e-6)
    return {
        "clips": n_clips,
        "audio_hours": round(audio_hours, 2),
        "model_load_s": load_seconds,
        "decode_s_per_pass": round(decode_seconds),
        "passes_within_2h": max(0, math.floor((7200 - load_seconds) / max(decode_seconds, 1e-6))),
    }
