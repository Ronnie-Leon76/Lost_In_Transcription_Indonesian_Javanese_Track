"""Competition entrypoint: transcribe /code_execution/data/clips -> submission.csv.

Runs inside the official container: Python 3.12, one A100 80 GB, 24 vCPU, no
network, two hours wall clock, read-only ``data/``.

Design rules this file follows, in priority order:

1. **Never fail to produce a submission.** A valid ``submission.csv`` with one
   row per clip is written before the model is even loaded, and rewritten as
   results arrive. Every stage is wrapped so that a crash in an optional stage
   degrades the output instead of destroying it.
2. **Never exceed the clock.** Extra decoding passes are scheduled only if the
   measured cost of the first pass says they fit, with margin. A system that
   times out scores nothing.
3. **Never log anything about the test data.** Counts, timings and stage names
   only — no filenames, no transcripts. The rules make logging test content
   grounds for disqualification.
4. **Use the same code as validation.** Prompts, post-processing and combination
   all come from the ``lit_jv`` package that the notebook validated, so the
   leaderboard system is the measured system.

Only the standard library is used for I/O; pandas/polars are avoided so a
version difference in the container cannot break the submission.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

START_TIME = time.time()

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

WORK_DIR = Path("/code_execution")
if not WORK_DIR.exists():  # local dry-run outside the container
    WORK_DIR = CODE_DIR.parent
DATA_DIR = Path(os.environ.get("LIT_DATA_DIR", WORK_DIR / "data"))
SUBMISSION_PATH = Path(
    os.environ.get("LIT_SUBMISSION_PATH", WORK_DIR / "submission" / "submission.csv")
)
CONFIG_PATH = Path(os.environ.get("LIT_CONFIG_PATH", CODE_DIR / "submission_config.json"))

#: Hard ceiling for the whole script. The platform allows 7200 s; stopping at
#: 6300 s leaves room for container teardown and for the final CSV write.
DEFAULT_TIME_BUDGET_S = 6300.0


def log(message: str) -> None:
    """Timestamped progress line. Never called with test-set content."""
    print(f"[{time.time() - START_TIME:7.1f}s] {message}", flush=True)


def write_submission(path: Path, filenames, transcripts) -> None:
    """Write ``submission.csv`` atomically: one row per clip, always complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(["audio_filename", "transcript"])
        for name, text in zip(filenames, transcripts):
            writer.writerow([name, text])
    os.replace(tmp, path)


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    log("no submission_config.json; using built-in defaults")
    return {}


def main() -> int:
    config = load_config()
    budget = float(config.get("time_budget_s", DEFAULT_TIME_BUDGET_S))
    deadline = START_TIME + budget

    from lit_jv import data as lit_data

    rows = lit_data.load_test_manifest(DATA_DIR)
    filenames = [row["audio_filename"] for row in rows]
    log(f"manifest: {len(rows)} clips")

    # Stage 0 — a complete, valid submission exists from this point on.
    transcripts = [""] * len(rows)
    write_submission(SUBMISSION_PATH, filenames, transcripts)
    log("placeholder submission written")

    # ``run_inference`` writes a checkpoint CSV after every completed pass.  Do
    # not replace that file with the all-empty Stage 0 placeholder when a later
    # optional pass fails: the checkpoint is deliberately the best recoverable
    # answer in that case.
    inference_completed = False
    try:
        transcripts = run_inference(rows, config, deadline, filenames)
        inference_completed = True
    except Exception:
        log("inference failed; preserving the latest checkpoint submission")
        traceback.print_exc()

    if inference_completed:
        write_submission(SUBMISSION_PATH, filenames, transcripts)
        empty = sum(1 for t in transcripts if not t.strip())
        log(f"final submission written; {len(rows) - empty}/{len(rows)} clips non-empty")
    else:
        log("final submission is the latest successful checkpoint")
    return 0


def run_inference(rows, config, deadline, filenames):
    from lit_jv import postprocess as lit_post
    from lit_jv.combine import LengthPrior, mbr_select, rover_combine
    from lit_jv.decode import DecodeConfig, FallbackConfig, Qwen3AsrDecoder, build_backend
    from lit_jv.prompts import PromptConfig

    # ---------------- post-processing ----------------
    pipeline = None
    post_path = CODE_DIR / config.get("postprocess", "postprocess.json")
    try:
        if post_path.exists():
            pipeline = lit_post.RulePipeline.load(post_path)
            log(f"post-processing rules: {list(pipeline.rules)}")
    except Exception:
        log("could not load post-processing rules; continuing without them")
        traceback.print_exc()
    if pipeline is None:
        pipeline = lit_post.RulePipeline(lexicon=lit_post.StyleLexicon(), rules=())

    prior_config = config.get("length_prior", {})
    length_prior = LengthPrior(
        mean=float(prior_config.get("mean", 1.886)),
        std=float(prior_config.get("std", 0.264)),
    )

    def choose_combiner(n_passes: int) -> str:
        """Pick the combiner for the number of passes that actually finished.

        The clock can cut a pass, and the right combiner depends on how many
        survived: ROVER needs a majority to vote, so with two hypotheses it is
        reduced to arbitrary tie-breaking and can score *worse* than its better
        input. ``combiner_by_n`` is measured in the notebook; ``combiner`` is the
        fallback for counts it does not mention.
        """
        if n_passes < 2:
            return "none"
        by_n = config.get("combiner_by_n") or {}
        return by_n.get(str(n_passes)) or by_n.get(n_passes) or config.get("combiner", "mbr")

    def finish(hypotheses_by_pass):
        """Combine whatever passes have completed and post-process."""
        texts = []
        weights = config.get("weights") or [1.0] * len(hypotheses_by_pass)
        combiner = choose_combiner(len(hypotheses_by_pass))
        rover_config = config.get("rover", {})
        for i, row in enumerate(rows):
            candidates = [hyp[i].text for hyp in hypotheses_by_pass]
            active = [(t, w) for t, w in zip(candidates, weights) if t.strip()]
            if not active:
                texts.append("")
                continue
            hyps = [t for t, _ in active]
            ws = [w for _, w in active]
            # A per-clip guard on top of choose_combiner: other passes may have
            # returned nothing for *this* clip, leaving too few voters for ROVER.
            if combiner == "rover" and len(hyps) >= 3:
                merged = rover_combine(
                    hyps,
                    ws,
                    null_penalty=float(rover_config.get("null_penalty", 1.0)),
                    insertion_threshold=float(rover_config.get("insertion_threshold", 0.5)),
                ).text
            elif combiner in ("rover", "mbr") and len(hyps) > 1:
                merged = mbr_select(hyps, ws).text
            else:
                merged = hyps[0]
            texts.append(pipeline.apply(merged, duration_s=row.get("duration")))
        return texts

    # ---------------- model ----------------
    model_dir = CODE_DIR / config.get("model_dir", "model")
    if not model_dir.exists():
        model_dir = config.get("model_dir", "Qwen/Qwen3-ASR-1.7B")
    vllm_kwargs = dict(config.get("vllm", {}))
    log("loading model")
    load_started = time.time()
    backend = build_backend(model_dir, prefer=config.get("backend", "vllm"), vllm_kwargs=vllm_kwargs, log=log)
    log(f"model ready in {time.time() - load_started:.0f}s")

    fallback_config = config.get("fallback", {})
    decoder = Qwen3AsrDecoder(
        backend=backend,
        length_prior=length_prior,
        fallback=FallbackConfig(
            enabled=bool(fallback_config.get("enabled", True)),
            min_avg_logprob=float(fallback_config.get("min_avg_logprob", -0.85)),
            max_length_z=float(fallback_config.get("max_length_z", 3.0)),
            sampling_draws=int(fallback_config.get("sampling_draws", 4)),
            sampling_temperature=float(fallback_config.get("sampling_temperature", 0.4)),
            split_long_clips=bool(fallback_config.get("split_long_clips", True)),
            max_rescue_fraction=float(fallback_config.get("max_rescue_fraction", 0.15)),
        ),
        log=log,
    )

    glossary = tuple(config.get("glossary", ()))
    pass_specs = config.get("passes") or [{"name": "p1", "prose": "codeswitch_style", "glossary": True}]

    def make_config(spec):
        prompt = PromptConfig(
            prose=spec.get("prose", "codeswitch_style"),
            use_language_field=bool(spec.get("use_language_field", True)),
            style_tag=spec.get("style_tag", PromptConfig.style_tag),
            include_style_tag=bool(spec.get("include_style_tag", True)),
            glossary=glossary if spec.get("glossary", True) else (),
            max_glossary_terms=int(spec.get("max_glossary_terms", 96)),
            force_language=spec.get("force_language", "Indonesian"),
        )
        return DecodeConfig(
            prompt=prompt,
            temperature=float(spec.get("temperature", 0.0)),
            tokens_per_second=float(spec.get("tokens_per_second", 8.0)),
            token_headroom=int(spec.get("token_headroom", 80)),
            max_tokens_cap=int(spec.get("max_tokens_cap", 640)),
            name=spec.get("name", "pass"),
        )

    # ---------------- decoding passes ----------------
    hypotheses_by_pass = []
    chunk_size = int(config.get("chunk_size", 64))
    pass_cost = 0.0
    rescue_reserve = 180.0
    for index, spec in enumerate(pass_specs):
        decode_config = make_config(spec)
        if index > 0:
            remaining = deadline - time.time()
            projected = pass_cost * 1.15 + rescue_reserve
            if remaining < projected:
                log(
                    f"skipping remaining passes: {remaining:.0f}s left, "
                    f"{projected:.0f}s needed"
                )
                break
        started = time.time()
        hypotheses = decoder.decode_pass(rows, decode_config, chunk_size=chunk_size)
        pass_cost = time.time() - started
        hypotheses_by_pass.append(hypotheses)
        # Reserve enough time to rescue the flagged clips of one pass.
        rescue_reserve = max(120.0, 0.35 * pass_cost)
        log(f"pass {decode_config.name} took {pass_cost:.0f}s")
        try:
            write_submission(SUBMISSION_PATH, filenames, finish(hypotheses_by_pass))
            log(f"checkpoint submission written after {len(hypotheses_by_pass)} pass(es)")
        except Exception:
            log("checkpoint write failed")
            traceback.print_exc()

    if not hypotheses_by_pass:
        raise RuntimeError("no decoding pass completed")

    # ---------------- rescue ----------------
    if deadline - time.time() > rescue_reserve:
        try:
            hypotheses_by_pass[0] = decoder.rescue(
                rows, hypotheses_by_pass[0], make_config(pass_specs[0])
            )
            rescued = sum(1 for h in hypotheses_by_pass[0] if h.rescued_by)
            log(f"rescue changed {rescued} clips")
        except Exception:
            log("rescue stage failed; keeping unrescued hypotheses")
            traceback.print_exc()
    else:
        log("no time for the rescue stage")

    return finish(hypotheses_by_pass)


if __name__ == "__main__":
    raise SystemExit(main())
