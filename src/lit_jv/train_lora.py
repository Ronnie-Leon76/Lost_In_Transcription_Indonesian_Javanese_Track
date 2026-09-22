"""LoRA fine-tuning of Qwen3-ASR on Indonesian-Javanese code-switched speech.

The model was pre-trained on 30 languages including Indonesian, but not on
Javanese and certainly not on this corpus's transcription conventions. Two
things have to be taught, and they want different data:

*Lexicon and acoustics* — Javanese words, Jember/Pandhalungan pronunciation,
WhatsApp voice-note channel. This wants every hour of Javanese speech we can
find, whatever its transcription style.

*Output convention* — scoring-space orthography, reduplication hyphens,
``e...`` disfluencies, ``-é`` suffixes, acronym casing. This only exists in the
2.5-hour development set, and it is fragile: mixing in a corpus with different
conventions degrades it.

Trying to satisfy both with one undifferentiated pile of data is what makes
low-resource fine-tuning disappointing. Instead:

1. Every corpus carries a **style tag** in its system prompt
   (:mod:`lit_jv.prompts`), so conflicting conventions stay separable and the
   inference-time tag selects the one we want. This is prompt-space domain
   conditioning, in the spirit of the language adapters of MMS (Pratap et al.,
   2024) and the code-switching adapters of Kulkarni et al. (arXiv:2310.07423),
   but without extra parameters or a routing decision.
2. Training runs as a **curriculum**: a high-LR stage on the large mismatched
   corpora, then a low-LR stage dominated by in-style data. Joint monolingual
   plus code-switched fine-tuning is what Hillah et al. (2024) found necessary
   for Spanish-English code-switching, and the ordering matters here because the
   in-style corpus is 4× smaller than the out-of-style one.
3. Optional **per-language adapters plus weighted merging**, the direct
   adaptation of Kulkarni et al.'s adapter fusion: train one LoRA on
   Javanese-dominant data and one on in-style data, then interpolate them with a
   weight chosen on held-out speakers.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .audio import load_audio_16k
from .prompts import (
    DEFAULT_FORCE_LANGUAGE,
    PromptConfig,
    build_context,
    render_chat_prompt,
    render_target,
)

# Files that must sit next to the weights for the checkpoint to be loadable by
# qwen_asr / vLLM. The Trainer only writes model weights.
HF_ASSET_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "chat_template.json",
    "merges.txt",
    "vocab.json",
)


def sanitize_generation_config(config):
    """Make a GenerationConfig saveable under transformers 4.57+ strict checks.

    Qwen3-ASR ships ``temperature=1e-06`` with ``do_sample=False`` — greedy
    decoding written as "temperature almost zero". Newer transformers refuse to
    serialise that combination. Inference here is greedy, so drop the sampling
    knobs rather than turning sampling on.
    """
    if config is None:
        return None
    if not bool(getattr(config, "do_sample", False)):
        for key in ("temperature", "top_p", "top_k", "typical_p", "epsilon_cutoff", "eta_cutoff"):
            if hasattr(config, key):
                setattr(config, key, None)
    try:
        config.validate(strict=True)
    except (TypeError, ValueError):
        config.do_sample = False
        for key in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(config, key):
                setattr(config, key, None)
    return config


def write_generation_config(directory: str | Path, config=None) -> Path:
    """Write a greedy-decode ``generation_config.json`` that transformers will reload."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "generation_config.json"
    payload = {
        "do_sample": False,
        "transformers_version": "4.57.6",
    }
    if config is not None:
        for key in ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id"):
            value = getattr(config, key, None)
            if value is not None:
                payload[key] = value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def generation_sequences(output):
    """Extract token IDs from every ``generate`` return shape we support.

    Transformers normally returns a tensor for ``return_dict_in_generate=False``.
    Qwen3-ASR builds observed in Runpod can instead return a tuple whose first
    item is that tensor. Slicing the tuple as if it were a tensor breaks the
    in-training WER probe after training has already finished, leaving no
    usable checkpoint-selection metric.
    """
    sequences = getattr(output, "sequences", output)
    if isinstance(sequences, tuple):
        if not sequences:
            raise RuntimeError("model.generate returned an empty tuple")
        sequences = sequences[0]
    if not hasattr(sequences, "ndim") or sequences.ndim != 2:
        raise TypeError(
            "model.generate did not return a 2-D token tensor "
            f"(got {type(sequences).__name__})"
        )
    return sequences


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    #: Adapt the text decoder by default. The audio encoder is optional because
    #: it doubles the trainable parameters for a much smaller share of the gain
    #: on 12 hours of data — measure it before paying for it.
    adapt_text_decoder: bool = True
    adapt_audio_encoder: bool = False

    def target_pattern(self) -> str:
        parts: list[str] = []
        if self.adapt_text_decoder:
            parts.append(
                r"thinker\.model\.layers\.\d+\."
                r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
            )
        if self.adapt_audio_encoder:
            parts.append(
                r"thinker\.audio_tower\.layers\.\d+\."
                r"(?:self_attn\.(?:q|k|v|out)_proj|fc[12])"
            )
        if not parts:
            raise ValueError("LoraSpec adapts nothing")
        return "|".join(f"(?:{p})" for p in parts)


@dataclass
class AugmentSpec:
    """Domain-matching augmentation.

    The test clips are 48 kHz mono WhatsApp voice notes at 64 kbit/s; the extra
    training corpora are studio-ish 44.1 kHz sessions and read TTS speech.
    Speed perturbation (Ko et al., 2015) and SpecAugment (Park et al., 2019) are
    the two augmentations that reliably help at this data scale, and gain jitter
    keeps the model from keying on the loudness normalisation.
    """

    enabled: bool = True
    #: Applied only to corpora in this set; the in-style data is left alone so
    #: the model sees its target convention on clean audio.
    corpora: tuple[str, ...] = ("jember", "mdc")
    speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1)
    gain_db_range: tuple[float, float] = (-4.0, 4.0)
    spec_augment: bool = True
    time_mask_fraction: float = 0.05
    n_time_masks: int = 2
    n_freq_masks: int = 2
    freq_mask_width: int = 12


@dataclass
class StageSpec:
    """One curriculum stage."""

    name: str
    #: corpus name -> number of times its rows are repeated in this stage.
    corpus_weights: dict[str, float]
    epochs: float = 1.0
    learning_rate: float = 1e-4
    batch_size: int = 8
    grad_accumulation: int = 4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"


@dataclass
class TrainSpec:
    model_path: str = "Qwen/Qwen3-ASR-1.7B"
    output_dir: str = "artifacts/qwen3asr-lora"
    lora: LoraSpec = field(default_factory=LoraSpec)
    augment: AugmentSpec = field(default_factory=AugmentSpec)
    stages: tuple[StageSpec, ...] = (
        StageSpec(
            name="s1-lexicon",
            corpus_weights={"jember": 1.0, "mdc": 0.5, "dev": 1.0},
            epochs=2.0,
            learning_rate=1.2e-4,
        ),
        StageSpec(
            name="s2-style",
            corpus_weights={"dev": 1.0, "jember": 0.15},
            epochs=4.0,
            learning_rate=3.5e-5,
        ),
    )
    prompt: PromptConfig = field(default_factory=PromptConfig)
    #: Assistant-side language label used in the supervised Qwen ASR target.
    #: Keep this aligned with the track, rather than inheriting the historic
    #: Indonesian-Javanese default from :mod:`lit_jv.prompts`.  Qwen supports
    #: both Indonesian and Spanish, so Spanish-Nahuatl training must use
    #: ``"Spanish"`` here even though many emitted words are Nahuatl.
    target_language: str = DEFAULT_FORCE_LANGUAGE
    #: Keep the style tag in the training prompt. Turning this off is the
    #: ablation that shows what corpus conditioning is worth.
    use_style_tags: bool = True
    #: Include the mined glossary during training so the model learns to *use*
    #: the context field rather than ignoring it.
    train_with_glossary: bool = True
    glossary_dropout: float = 0.3
    max_audio_seconds: float = 40.0
    seed: int = 3407
    #: Ask torch for deterministic kernels. Slower, and some fused attention
    #: kernels have no deterministic path, so this is for verifying a result
    #: rather than for producing every result. See :mod:`lit_jv.repro`.
    deterministic: bool = False
    gradient_checkpointing: bool = True
    bf16: bool = True
    num_workers: int = 4
    logging_steps: int = 10
    save_steps: int = 0  # 0 = save at the end of each stage only

    # -- metrics ----------------------------------------------------------
    #: Where TensorBoard event files go. Defaults to ``<output_dir>/runs``.
    tensorboard_dir: str | None = None
    report_to: tuple[str, ...] = ("tensorboard",)
    run_name: str = "qwen3asr-lora"
    #: Teacher-forced eval loss cadence, in optimiser steps. 0 disables it.
    eval_steps: int = 50
    #: Generation-based WER cadence, in optimiser steps. 0 disables it. Much more
    #: expensive than eval loss, and much more informative: eval loss can improve
    #: while WER gets worse, because loss also rewards punctuation the scorer
    #: deletes.
    wer_eval_steps: int = 200
    #: How many held-out clips the WER probe decodes. Kept small on purpose - it
    #: runs inside the training loop.
    wer_eval_clips: int = 48
    wer_eval_batch_size: int = 8

    # -- publishing -------------------------------------------------------
    #: Set to a :class:`lit_jv.hub.HubSpec` to publish adapters, logs and the run
    #: manifest after training. ``None`` keeps everything local.
    hub: Any = None
    push_each_stage: bool = False


# --------------------------------------------------------------------------
# Prompt/target construction for training
# --------------------------------------------------------------------------


def training_prompt(
    row: dict, spec: TrainSpec, rng=None
) -> tuple[str, str]:
    """Return ``(prefix_prompt, target)`` for one manifest row.

    The prefix is built by exactly the same code as inference. ``glossary_dropout``
    randomly empties the glossary so the model stays accurate when the context is
    absent, which keeps the zero-context ablation honest and prevents the model
    from treating the glossary as a required cue.
    """
    config = spec.prompt
    if spec.train_with_glossary and config.glossary and rng is not None:
        if rng.random() < spec.glossary_dropout:
            config = PromptConfig(**{**config.__dict__, "glossary": ()})
    elif not spec.train_with_glossary:
        config = PromptConfig(**{**config.__dict__, "glossary": ()})

    style_tag = row.get("style_tag") if spec.use_style_tags else ""
    context = build_context(
        config, language_field=row.get("language"), style_tag=style_tag
    )
    prefix = render_chat_prompt(context, force_language=None)
    target = render_target(row["text"], language=spec.target_language)
    return prefix, target


def expand_stage_rows(rows: Sequence[dict], stage: StageSpec, seed: int = 0) -> list[dict]:
    """Apply the stage's per-corpus weights by repeating / subsampling rows."""
    import random

    rng = random.Random(seed)
    out: list[dict] = []
    for row in rows:
        weight = stage.corpus_weights.get(row.get("corpus", ""), 0.0)
        weight *= float(row.get("weight", 1.0) or 1.0)
        if weight <= 0:
            continue
        whole = int(weight)
        out.extend([row] * whole)
        if rng.random() < weight - whole:
            out.append(row)
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------
# Collator
# --------------------------------------------------------------------------


@dataclass
class Qwen3AsrCollator:
    """Build a padded batch and mask the prompt tokens out of the loss.

    Two calls to the processor are needed: one for ``prefix + target`` and one
    for ``prefix`` alone, so that the prefix length is known exactly. Computing
    it from token counts instead would break on the audio placeholder, which
    expands to a variable number of positions.
    """

    processor: Any
    spec: TrainSpec
    seed: int = 0

    def __post_init__(self):
        import random

        self._rng = random.Random(self.seed)

    def _load(self, row: dict):
        import numpy as np

        wav = load_audio_16k(row["audio_path"])
        aug = self.spec.augment
        if aug.enabled and row.get("corpus") in aug.corpora:
            factor = self._rng.choice(aug.speed_factors)
            if factor != 1.0:
                import librosa

                wav = librosa.resample(
                    wav, orig_sr=16000, target_sr=int(16000 / factor)
                ).astype("float32")
            lo, hi = aug.gain_db_range
            gain_db = self._rng.uniform(lo, hi)
            wav = (wav * (10.0 ** (gain_db / 20.0))).astype("float32")
            peak = float(np.max(np.abs(wav))) if wav.size else 0.0
            if peak > 1.0:
                wav = (wav / peak).astype("float32")
        limit = int(self.spec.max_audio_seconds * 16000)
        if len(wav) > limit:
            wav = wav[:limit]
        return wav

    def _spec_augment(self, features):
        aug = self.spec.augment
        if not (aug.enabled and aug.spec_augment):
            return features
        batch, n_mels, n_frames = features.shape
        for b in range(batch):
            for _ in range(aug.n_freq_masks):
                width = self._rng.randint(0, aug.freq_mask_width)
                if width <= 0:
                    continue
                start = self._rng.randint(0, max(0, n_mels - width))
                features[b, start : start + width, :] = 0.0
            span = int(aug.time_mask_fraction * n_frames)
            for _ in range(aug.n_time_masks):
                width = self._rng.randint(0, max(0, span))
                if width <= 0:
                    continue
                start = self._rng.randint(0, max(0, n_frames - width))
                features[b, :, start : start + width] = 0.0
        return features

    def __call__(self, rows: Sequence[dict]) -> dict:
        prompts, targets, waveforms = [], [], []
        for row in rows:
            prefix, target = training_prompt(row, self.spec, rng=self._rng)
            prompts.append(prefix)
            targets.append(target)
            waveforms.append(self._load(row))

        eos = self.processor.tokenizer.eos_token or ""
        full = [p + t + eos for p, t in zip(prompts, targets)]

        batch = self.processor(
            text=full, audio=waveforms, return_tensors="pt", padding=True, truncation=False
        )
        prefix_batch = self.processor(
            text=prompts, audio=waveforms, return_tensors="pt", padding=True, truncation=False
        )
        prefix_lengths = prefix_batch["attention_mask"].sum(dim=1).tolist()

        labels = batch["input_ids"].clone()
        for i, length in enumerate(prefix_lengths):
            labels[i, : int(length)] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        # Padding is right-aligned, so anything outside the attention mask is
        # padding regardless of its token id.
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        if "input_features" in batch:
            batch["input_features"] = self._spec_augment(batch["input_features"])
        return batch


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------


def patch_outer_forward(model) -> None:
    """Route ``model.forward`` to ``model.thinker.forward``.

    ``Qwen3ASRForConditionalGeneration`` exposes generation but not a training
    forward; the official fine-tuning script patches the class the same way.
    """
    cls = model.__class__
    if getattr(cls, "_lit_forward_patched", False):
        return
    if not hasattr(model, "thinker"):
        raise RuntimeError("model has no .thinker; incompatible Qwen3-ASR build")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._lit_forward_patched = True


def describe_lora_targets(model, spec: LoraSpec) -> dict:
    """Report which modules the LoRA pattern will actually wrap.

    PEFT fails silently-ish when a regex matches nothing (it raises, but only
    after a long load), and matching *too much* — for instance catching the
    audio encoder's ``q_proj`` when only the decoder was intended — is worse
    because training still runs. Always look at this before launching.
    """
    import torch.nn as nn

    pattern = re.compile(spec.target_pattern())
    matched: list[str] = []
    total_linear = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_linear += 1
            if pattern.fullmatch(name) or pattern.search(name):
                matched.append(name)
    prefixes = sorted({re.sub(r"\.\d+\.", ".*.", n) for n in matched})
    return {
        "n_linear_total": total_linear,
        "n_matched": len(matched),
        "matched_shapes": prefixes,
        "example": matched[:4],
    }


def build_model(spec: TrainSpec, adapter_path: str | Path | None = None):
    """Load Qwen3-ASR and attach (or restore) a LoRA adapter."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from qwen_asr import Qwen3ASRModel

    from .decode import reclaim_qwen3_asr_transformers_classes, release_cuda

    reclaim_qwen3_asr_transformers_classes()
    release_cuda()
    wrapper = Qwen3ASRModel.from_pretrained(
        spec.model_path,
        dtype=torch.bfloat16 if spec.bf16 else torch.float16,
        device_map=None,
    )
    model, processor = wrapper.model, wrapper.processor
    patch_outer_forward(model)

    # Trainer enables gradient checkpointing below. Keeping the KV cache on at
    # that point produces a noisy warning and is ignored by Transformers anyway.
    model.config.use_cache = not spec.gradient_checkpointing

    from transformers import GenerationConfig

    model.generation_config = sanitize_generation_config(
        GenerationConfig.from_model_config(model.config)
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        info = describe_lora_targets(model, spec.lora)
        print(
            f"LoRA targets: {info['n_matched']}/{info['n_linear_total']} Linear modules\n  "
            + "\n  ".join(info["matched_shapes"])
        )
        if info["n_matched"] == 0:
            raise RuntimeError("LoRA target pattern matched no modules")
        lora_config = LoraConfig(
            r=spec.lora.r,
            lora_alpha=spec.lora.alpha,
            lora_dropout=spec.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=spec.lora.target_pattern(),
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


class ManifestDataset:
    """List-backed training set. Does not import Hugging Face ``datasets``.

    The competition runtime ships transformers / peft / vLLM but not ``datasets``,
    and the only thing we needed from it was ``Dataset.from_list``. Items stay
    plain dicts so the collator can load audio itself.

    ``lengths`` is the integer cost LengthGroupedSampler wants: audio duration
    dominates sequence length here, not the transcript. Trainer only auto-infers
    lengths from ``input_ids``, which do not exist until the collator runs.
    """

    def __init__(self, rows: Sequence[dict]):
        self.rows = [
            {
                "audio_path": r["audio_path"],
                "text": r["text"],
                "language": r.get("language", "javind"),
                "style_tag": r.get("style_tag", ""),
                "corpus": r.get("corpus", ""),
                "weight": float(r.get("weight", 1.0) or 1.0),
                "length": int(max(1.0, float(r.get("duration", 10.0) or 10.0)) * 50),
            }
            for r in rows
        ]
        self.lengths = [row["length"] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def _as_hf_dataset(rows: Sequence[dict]) -> ManifestDataset:
    """Back-compat name: used to return ``datasets.Dataset``."""
    return ManifestDataset(rows)


class _CastInputsTrainer:
    """Mixin factory: Trainer subclass that casts float inputs to model dtype."""

    @staticmethod
    def build():
        from transformers import Trainer

        class CastFloatInputsTrainer(Trainer):
            def _prepare_inputs(self, inputs):
                inputs = super()._prepare_inputs(inputs)
                dtype = getattr(self.model, "dtype", None)
                if dtype is not None:
                    for key, value in list(inputs.items()):
                        if hasattr(value, "is_floating_point") and value.is_floating_point():
                            inputs[key] = value.to(dtype=dtype)
                return inputs

            def _get_train_sampler(self, *args, **kwargs):
                # transformers 4.57+ may pass dataset=; older calls take no args.
                dataset = kwargs.get("dataset", self.train_dataset)
                if args:
                    dataset = args[0]
                if self.args.group_by_length and hasattr(dataset, "lengths"):
                    from torch.utils.data import RandomSampler
                    from transformers.trainer_pt_utils import LengthGroupedSampler

                    if not len(dataset):
                        return RandomSampler(dataset)
                    return LengthGroupedSampler(
                        self.args.train_batch_size * self.args.gradient_accumulation_steps,
                        dataset=dataset,
                        lengths=list(dataset.lengths),
                    )
                try:
                    return super()._get_train_sampler(*args, **kwargs)
                except TypeError:
                    return super()._get_train_sampler()

            def _get_eval_sampler(self, eval_dataset):
                """Keep lazy audio manifests in deterministic order at evaluation.

                ``group_by_length`` is useful for the training batches and is
                handled above from the manifest's explicit ``lengths``.  Recent
                Transformers versions also try to length-group evaluation and
                assume each raw item already has ``input_ids``; ours only gains
                those in the collator.  Evaluation must therefore be sequential.
                """
                from torch.utils.data import SequentialSampler

                return SequentialSampler(eval_dataset)

        return CastFloatInputsTrainer


def generate_transcripts(
    model,
    processor,
    rows: Sequence[dict],
    spec: TrainSpec,
    batch_size: int = 8,
    max_new_tokens: int | None = None,
) -> list[str]:
    """Greedy-decode ``rows`` with the live training model.

    Used by the in-training WER probe, so it deliberately mirrors inference: same
    prompt builder, same audio loader, glossary present, no augmentation, no
    sampling. The only difference from :mod:`lit_jv.decode` is that the weights
    come from memory instead of a checkpoint.
    """
    import torch

    from .decode import DecodeConfig
    from .prompts import build_prompt, parse_generation

    # Borrow the inference token budget rather than restating it, so the probe cannot
    # drift away from what the container will actually do.
    budget_for = DecodeConfig(prompt=spec.prompt).max_tokens
    was_training = model.training
    model.eval()

    # Training turns the KV cache off, because gradient checkpointing is incompatible
    # with it. Generating in that state re-runs the whole prefix for every token, which
    # turns a two-minute probe into a twenty-minute one, so both are flipped for the
    # duration of the probe and restored afterwards.
    config = getattr(model, "config", None)
    cache_was = getattr(config, "use_cache", None) if config is not None else None
    checkpointing_was = bool(getattr(model, "is_gradient_checkpointing", False))
    if config is not None:
        config.use_cache = True
    if checkpointing_was:
        model.gradient_checkpointing_disable()

    out: list[str] = []
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            waveforms = [load_audio_16k(row["audio_path"]) for row in batch]
            prompts = [
                build_prompt(spec.prompt, language_field=row.get("language"))
                for row in batch
            ]
            inputs = processor(
                text=prompts,
                audio=waveforms,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            device = next(model.parameters()).device
            dtype = getattr(model, "dtype", None)
            tensors = {}
            for key, value in inputs.items():
                if not hasattr(value, "to"):
                    continue
                value = value.to(device)
                if dtype is not None and value.is_floating_point():
                    value = value.to(dtype)
                tensors[key] = value

            budget = max_new_tokens or budget_for(
                max(float(row.get("duration", 20.0) or 20.0) for row in batch)
            )
            with torch.no_grad():
                generated = model.generate(
                    **tensors,
                    max_new_tokens=budget,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
                )
            sequences = generation_sequences(generated)
            prompt_len = tensors["input_ids"].shape[1]
            for sequence in sequences[:, prompt_len:]:
                text = processor.tokenizer.decode(sequence, skip_special_tokens=True)
                out.append(parse_generation(text, spec.prompt.force_language))
    finally:
        if checkpointing_was:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if config is not None and cache_was is not None:
            config.use_cache = cache_was
        if was_training:
            model.train()
    return out


def _wer_callback_class():
    """Build the callback lazily: transformers is a GPU-side dependency."""
    from transformers import TrainerCallback

    class WerProbeCallback(TrainerCallback):
        """Log held-out WER, computed the way the leaderboard computes it.

        Eval loss is a proxy and an imperfect one: it rewards getting punctuation
        and casing right in places the scorer deletes, and it is blind to the
        truncation failure that costs the most. This decodes a fixed subset and
        scores it with the real normaliser, so the curve on TensorBoard is the
        quantity being optimised rather than a correlate of it.
        """

        def __init__(self, processor, rows, spec, every_steps, batch_size, stage_name, log=print):
            self.processor = processor
            self.rows = list(rows)
            self.spec = spec
            self.every_steps = every_steps
            self.batch_size = batch_size
            self.stage_name = stage_name
            self.log = log
            self.history: list[dict] = []
            self.best: dict | None = None
            self._references = [row["text"] for row in self.rows]
            self._writer = None
            self._last_step = -1
            # The Trainer's last step is not necessarily the best WER step.  Keep
            # an in-memory CPU copy of the small LoRA state so the next stage (and
            # the adapter ultimately saved) starts from the selected checkpoint.
            # Copying the 1.7B base model would be prohibitive; PEFT's state dict
            # contains adapters only.
            self._best_adapter_state = None

        def _summary_writer(self, args):
            # Writing into the Trainer's own logging_dir means TensorBoard merges
            # these scalars into the same run as the losses.
            if self._writer is None and "tensorboard" in (args.report_to or []):
                try:
                    from torch.utils.tensorboard import SummaryWriter

                    self._writer = SummaryWriter(log_dir=args.logging_dir)
                except ImportError:
                    self.log("  [wer probe] tensorboard not installed; logging to stdout only")
                    self._writer = False
            return self._writer or None

        def _probe(self, args, state, model, tag):
            from .scoring import corpus_wer

            if not self.rows or model is None or state.global_step == self._last_step:
                return
            self._last_step = state.global_step
            started = time.time()
            try:
                hypotheses = generate_transcripts(
                    model, self.processor, self.rows, self.spec, batch_size=self.batch_size
                )
                wer = corpus_wer(self._references, hypotheses)
            except Exception as exc:  # a probe must never kill a training run
                self.log(f"  [wer] probe failed at step {state.global_step}: {exc}")
                return
            empty = sum(1 for h in hypotheses if not h.strip())
            entry = {
                "step": state.global_step,
                "epoch": round(float(state.epoch or 0.0), 3),
                "eval_wer": round(wer, 5),
                "empty_outputs": empty,
                "clips": len(self.rows),
                "seconds": round(time.time() - started, 1),
                "tag": tag,
            }
            self.history.append(entry)
            if self.best is None or wer < self.best["eval_wer"]:
                self.best = entry
                try:
                    from peft import get_peft_model_state_dict

                    self._best_adapter_state = {
                        key: value.detach().cpu().clone()
                        for key, value in get_peft_model_state_dict(model).items()
                    }
                except Exception as exc:
                    # Retain the metric even if a future PEFT version changes its
                    # helper API; training must not be interrupted by bookkeeping.
                    self.log(f"  [wer] could not snapshot best adapter: {exc}")
            self.log(
                f"  [wer] step {state.global_step} ({tag}) WER={wer:.4f} "
                f"empty={empty} in {entry['seconds']:.0f}s"
            )
            writer = self._summary_writer(args)
            if writer is not None:
                writer.add_scalar("eval/wer", wer, state.global_step)
                writer.add_scalar("eval/empty_outputs", empty, state.global_step)
                writer.flush()

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if self.every_steps and state.global_step and state.global_step % self.every_steps == 0:
                self._probe(args, state, model, "mid")

        def on_train_end(self, args, state, control, model=None, **kwargs):
            self._probe(args, state, model, "final")
            if self._best_adapter_state is not None and model is not None:
                try:
                    from peft import set_peft_model_state_dict

                    set_peft_model_state_dict(model, self._best_adapter_state)
                    self.log(
                        f"  [wer] restored best adapter from step {self.best['step']} "
                        f"(WER={self.best['eval_wer']:.4f})"
                    )
                except Exception as exc:
                    self.log(f"  [wer] could not restore best adapter: {exc}")
            if self._writer:
                self._writer.close()

    return WerProbeCallback


def run_stage(
    model,
    processor,
    rows: Sequence[dict],
    stage: StageSpec,
    spec: TrainSpec,
    output_dir: str | Path,
    eval_rows: Sequence[dict] | None = None,
) -> dict:
    """Train one curriculum stage; returns the log history."""
    from transformers import TrainingArguments

    stage_rows = expand_stage_rows(rows, stage, seed=spec.seed)
    if not stage_rows:
        raise ValueError(f"stage {stage.name} selected no rows; check corpus_weights")
    hours = sum(float(r.get("duration", 0.0) or 0.0) for r in stage_rows) / 3600
    print(f"[{stage.name}] {len(stage_rows)} examples ({hours:.2f} h) lr={stage.learning_rate}")

    dataset = _as_hf_dataset(stage_rows)
    collator = Qwen3AsrCollator(processor=processor, spec=spec, seed=spec.seed)

    eval_rows = list(eval_rows or [])
    eval_dataset = _as_hf_dataset(eval_rows) if eval_rows and spec.eval_steps else None
    if eval_rows:
        overlap = {r["audio_path"] for r in eval_rows} & {r["audio_path"] for r in stage_rows}
        if overlap:
            raise ValueError(
                f"{len(overlap)} eval clips also appear in stage {stage.name}'s training "
                "data; split by conversation or speaker, not at random"
            )
        print(f"[{stage.name}] eval on {len(eval_rows)} held-out clips")

    output_dir = Path(output_dir) / stage.name
    logging_dir = Path(spec.tensorboard_dir or Path(spec.output_dir) / "runs") / stage.name

    # Asking for a reporter that is not installed makes the Trainer raise at construction
    # time, which would waste a model load. Drop it and say so instead.
    reporters = list(spec.report_to)
    if "tensorboard" in reporters:
        from transformers.integrations import is_tensorboard_available

        if not is_tensorboard_available():
            reporters.remove("tensorboard")
            print(
                "[warn] tensorboard is not installed; metrics will only reach stdout. "
                "`pip install tensorboard` to get the curves."
            )

    args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        run_name=f"{spec.run_name}-{stage.name}",
        per_device_train_batch_size=stage.batch_size,
        gradient_accumulation_steps=stage.grad_accumulation,
        learning_rate=stage.learning_rate,
        num_train_epochs=stage.epochs,
        warmup_ratio=stage.warmup_ratio,
        weight_decay=stage.weight_decay,
        max_grad_norm=stage.max_grad_norm,
        lr_scheduler_type=stage.lr_scheduler,
        logging_steps=spec.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=spec.eval_steps or None,
        per_device_eval_batch_size=max(1, stage.batch_size // 2),
        eval_accumulation_steps=4,
        save_strategy="steps" if spec.save_steps else "no",
        save_steps=spec.save_steps or 500,
        save_total_limit=2,
        bf16=spec.bf16,
        fp16=not spec.bf16,
        gradient_checkpointing=spec.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=spec.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=spec.num_workers > 0,
        group_by_length=True,
        length_column_name="length",
        remove_unused_columns=False,
        report_to=reporters or "none",
        seed=spec.seed,
        # Separate from `seed` on purpose: it drives data order only, so the same
        # model init can be replayed against a different shuffle.
        data_seed=spec.seed,
        full_determinism=spec.deterministic,
        ddp_find_unused_parameters=False,
    )
    trainer_cls = _CastInputsTrainer.build()
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    probe = None
    if eval_rows and spec.wer_eval_steps:
        probe = _wer_callback_class()(
            processor=processor,
            rows=eval_rows[: spec.wer_eval_clips],
            spec=spec,
            every_steps=spec.wer_eval_steps,
            batch_size=spec.wer_eval_batch_size,
            stage_name=stage.name,
        )
        trainer.add_callback(probe)

    trainer.train()
    result = {
        "stage": stage.name,
        "examples": len(stage_rows),
        "hours": round(hours, 3),
        "logging_dir": str(logging_dir),
        "log_history": trainer.state.log_history,
    }
    if probe is not None:
        result["wer_history"] = probe.history
        result["best_wer"] = probe.best
    return result


def train(
    rows: Sequence[dict],
    spec: TrainSpec,
    stages: Sequence[StageSpec] | None = None,
    save_adapter_as: str = "adapter",
    eval_rows: Sequence[dict] | None = None,
) -> dict:
    """Run the full curriculum, save the adapter, record provenance, publish.

    ``eval_rows`` must be disjoint from ``rows`` by conversation or speaker — the
    test split is, so anything less overstates the result. ``run_stage`` refuses
    an overlapping split rather than reporting a flattering number.
    """
    from . import repro

    repro.seed_everything(spec.seed, strict=spec.deterministic)
    stages = list(stages or spec.stages)

    model, processor = build_model(spec)
    history = []
    adapter_root = Path(spec.output_dir)
    for stage in stages:
        history.append(
            run_stage(model, processor, rows, stage, spec, spec.output_dir, eval_rows=eval_rows)
        )

    adapter_dir = adapter_root / save_adapter_as
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    with open(adapter_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)

    manifest_path = repro.write_run_manifest(
        adapter_dir / "run_manifest.json",
        seed=spec.seed,
        deterministic=spec.deterministic,
        config_fingerprint=repro.config_fingerprint(spec),
        config=spec,
        data=repro.manifest_fingerprint(rows),
        eval_data=repro.manifest_fingerprint(eval_rows or []),
        stages=[s.name for s in stages],
        best_wer=[h.get("best_wer") for h in history if h.get("best_wer")],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"saved adapter to {adapter_dir}")
    print("run: " + repro.summarise(manifest))

    tensorboard_dir = Path(spec.tensorboard_dir or adapter_root / "runs")
    result = {
        "adapter_dir": str(adapter_dir),
        "manifest_path": str(manifest_path),
        "tensorboard_dir": str(tensorboard_dir),
        "history": history,
        "best_wer": min(
            (h["best_wer"] for h in history if h.get("best_wer")),
            key=lambda e: e["eval_wer"],
            default=None,
        ),
    }

    if spec.hub is not None:
        from . import hub as hub_module

        result["published"] = hub_module.publish_run(
            spec.hub,
            adapters={save_adapter_as: adapter_dir},
            tensorboard_dir=tensorboard_dir,
            manifest_path=manifest_path,
            metrics=[
                {"system": h["stage"], "eval WER": (h.get("best_wer") or {}).get("eval_wer", "")}
                for h in history
            ],
            stages=[
                {
                    "name": s.name,
                    "corpus_weights": s.corpus_weights,
                    "epochs": s.epochs,
                    "learning_rate": s.learning_rate,
                }
                for s in stages
            ],
        ).url()
    from .decode import release_cuda

    del model, processor
    release_cuda()
    return result


# --------------------------------------------------------------------------
# Adapter fusion and merging
# --------------------------------------------------------------------------


def fuse_adapters(
    spec: TrainSpec,
    adapters: dict[str, str | Path],
    weights: dict[str, float],
    out_dir: str | Path,
) -> Path:
    """Interpolate several LoRA adapters into one (Kulkarni-style fusion).

    Train one adapter per data regime — Javanese-dominant, in-style
    voice-note — then pick the interpolation weight on held-out speakers. This
    gives a knob between "knows more Javanese" and "writes the right
    orthography" that a single joint fine-tune does not expose.
    """
    import torch
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    from .decode import reclaim_qwen3_asr_transformers_classes, release_cuda

    reclaim_qwen3_asr_transformers_classes()
    release_cuda()
    wrapper = Qwen3ASRModel.from_pretrained(
        spec.model_path, dtype=torch.bfloat16 if spec.bf16 else torch.float16, device_map=None
    )
    model = wrapper.model
    patch_outer_forward(model)

    names = list(adapters)
    peft_model = PeftModel.from_pretrained(model, str(adapters[names[0]]), adapter_name=names[0])
    for name in names[1:]:
        peft_model.load_adapter(str(adapters[name]), adapter_name=name)
    peft_model.add_weighted_adapter(
        adapters=names,
        weights=[float(weights[n]) for n in names],
        adapter_name="fused",
        combination_type="linear",
    )
    peft_model.set_adapter("fused")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir), selected_adapters=["fused"])
    return out_dir / "fused"


def merge_and_save(
    spec: TrainSpec,
    adapter_dir: str | Path,
    out_dir: str | Path,
    adapter_name: str | None = None,
) -> Path:
    """Merge LoRA into the base weights and write a self-contained checkpoint.

    vLLM loads a plain checkpoint far more reliably than a base model plus
    adapter for multimodal architectures, and the submission has to ship weights
    anyway, so merging removes a moving part from the container.
    """
    import torch
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    from .decode import reclaim_qwen3_asr_transformers_classes, release_cuda

    reclaim_qwen3_asr_transformers_classes()
    release_cuda()
    wrapper = Qwen3ASRModel.from_pretrained(
        spec.model_path, dtype=torch.bfloat16 if spec.bf16 else torch.float16, device_map=None
    )
    model, processor = wrapper.model, wrapper.processor
    patch_outer_forward(model)

    peft_model = PeftModel.from_pretrained(
        model, str(adapter_dir), adapter_name=adapter_name or "default"
    )
    merged = peft_model.merge_and_unload()
    sanitize_generation_config(getattr(merged, "generation_config", None))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        merged.save_pretrained(str(out_dir), safe_serialization=True)
    except ValueError as exc:
        if "GenerationConfig" not in str(exc):
            raise
        # Detach the invalid config, write weights, then drop a greedy one.
        merged.generation_config = None
        merged.save_pretrained(str(out_dir), safe_serialization=True)
    write_generation_config(out_dir, getattr(merged, "generation_config", None))
    try:
        processor.save_pretrained(str(out_dir))
    except Exception:
        pass
    copy_hf_assets(spec.model_path, out_dir)

    # Keep provenance attached to the weights: a merged checkpoint is what gets
    # shipped and published, and it is the artefact most likely to outlive the box
    # it was built on.
    source_manifest = Path(adapter_dir) / "run_manifest.json"
    if source_manifest.is_file():
        shutil.copyfile(source_manifest, out_dir / "run_manifest.json")
    else:
        from . import repro

        repro.write_run_manifest(
            out_dir / "run_manifest.json",
            seed=spec.seed,
            config_fingerprint=repro.config_fingerprint(spec),
            merged_from=str(adapter_dir),
        )
    print(f"merged checkpoint written to {out_dir}")
    return out_dir


def copy_hf_assets(source: str | Path, destination: str | Path) -> list[str]:
    """Copy tokenizer / processor / template files next to merged weights.

    ``source`` may be a local directory or a Hub id; in the latter case the
    files are pulled from the local Hub cache via ``huggingface_hub``.
    """
    destination = Path(destination)
    copied: list[str] = []
    source_path = Path(source)
    if source_path.is_dir():
        for name in HF_ASSET_FILES:
            candidate = source_path / name
            if candidate.exists() and not (destination / name).exists():
                shutil.copy2(candidate, destination / name)
                copied.append(name)
        return copied
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return copied
    for name in HF_ASSET_FILES:
        if (destination / name).exists():
            continue
        try:
            path = hf_hub_download(repo_id=str(source), filename=name)
        except Exception:
            continue
        shutil.copy2(path, destination / name)
        copied.append(name)
    return copied
