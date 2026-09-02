"""CPU tests for seeding, provenance and Hub packaging logic.

Network is never touched: the Hub tests exercise card rendering, spec handling and
path resolution, which is where the mistakes that matter live. Uploading is a thin
wrapper over ``huggingface_hub`` and is verified by actually pushing, once, from
the notebook.

    python tests/test_repro_and_hub.py
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lit_jv import hub, repro  # noqa: E402
from lit_jv import data as lit_data  # noqa: E402
from lit_jv.train_lora import LoraSpec, TrainSpec  # noqa: E402

DEV_DIR = lit_data.resolve_dev_dir()


class Skip(Exception):
    """Optional local corpus is absent. Not a failure."""


def _require_dev() -> None:
    if not lit_data.dev_dir_ready(DEV_DIR):
        raise Skip(
            f"no development set at {DEV_DIR} "
            "(copy indonesian_dev/ here, or set LIT_DEV_DIR)"
        )


def test_seeding_is_reproducible_across_processes():
    # PYTHONHASHSEED is set for the *next* process, so the check here is on the
    # generators this process actually uses.
    repro.seed_everything(1234)
    first = [random.random() for _ in range(5)]
    repro.seed_everything(1234)
    assert first == [random.random() for _ in range(5)]

    repro.seed_everything(1235)
    assert first != [random.random() for _ in range(5)]

    seed = repro.seed_everything(repro.DEFAULT_SEED)
    assert seed == repro.DEFAULT_SEED


def test_config_fingerprint_ignores_key_order_but_not_values():
    a = TrainSpec(seed=7, lora=LoraSpec(r=16))
    b = TrainSpec(seed=7, lora=LoraSpec(r=16))
    c = TrainSpec(seed=7, lora=LoraSpec(r=32))
    assert repro.config_fingerprint(a) == repro.config_fingerprint(b)
    assert repro.config_fingerprint(a) != repro.config_fingerprint(c)
    # Dict ordering must not leak into the hash, or every rerun looks different.
    assert repro.config_fingerprint({"x": 1, "y": 2}) == repro.config_fingerprint({"y": 2, "x": 1})


def test_manifest_fingerprint_tracks_content_not_order():
    _require_dev()
    rows = lit_data.load_dev_manifest(DEV_DIR, with_durations=False)[:20]
    shuffled = list(rows)
    random.Random(0).shuffle(shuffled)
    assert repro.manifest_fingerprint(rows) == repro.manifest_fingerprint(shuffled)

    edited = [dict(r) for r in rows]
    edited[3]["text"] = edited[3]["text"] + " extra"
    assert repro.manifest_fingerprint(edited) != repro.manifest_fingerprint(rows)

    summary = repro.manifest_fingerprint(rows)
    assert summary["rows"] == 20
    assert summary["by_corpus"] == {"dev": 20}


def test_run_manifest_is_written_and_summarisable():
    with tempfile.TemporaryDirectory() as tmp:
        path = repro.write_run_manifest(
            Path(tmp) / "run_manifest.json",
            seed=99,
            config_fingerprint="deadbeef",
            data={"sha256_16": "cafe"},
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 99
    assert "environment" in payload and "packages" in payload["environment"]
    line = repro.summarise(payload)
    assert "seed=99" in line and "config=deadbeef" in line


def test_environment_fingerprint_reports_absent_packages_as_none():
    env = repro.environment_fingerprint()
    assert env["python"]
    assert set(env["packages"]) == set(repro.TRACKED_PACKAGES)
    assert all(v is None or isinstance(v, str) for v in env["packages"].values())


def test_model_card_documents_the_run():
    spec = hub.HubSpec(repo_id="someone/lit-jv-qwen3asr-lora", private=False)
    manifest = {
        "seed": 3407,
        "config_fingerprint": "abc123",
        "data": {"rows": 5000, "hours": 12.5, "sha256_16": "feed01"},
        "environment": {
            "packages": {"torch": "2.11.0", "peft": "0.20.0"},
            "git": {"available": True, "commit": "0123456789abcdef", "dirty": False},
            "torch": {"devices": ["NVIDIA A100-SXM4-80GB"]},
        },
    }
    card = hub.render_model_card(
        spec,
        metrics=[{"system": "s2-style + rules", "WER": 0.231}],
        manifest=manifest,
        stages=[{"name": "s1-lexicon", "corpus_weights": {"jember": 1.0}, "epochs": 2, "learning_rate": 1.2e-4}],
        notes="Held out conversation 5.",
    )
    assert card.startswith("---\n")
    assert "license: mpl-2.0" in card          # winners must ship an open licence
    assert "base_model: Qwen/Qwen3-ASR-1.7B" in card
    assert "s2-style + rules" in card and "0.231" in card
    assert "seed: `3407`" in card
    assert "feed01" in card and "0123456789ab" in card
    assert "NVIDIA A100-SXM4-80GB" in card
    assert "Held out conversation 5." in card
    # The card must not leak a token even if one is in the environment.
    assert "HF_TOKEN" not in card


def test_hub_spec_reads_token_from_the_environment_only():
    import os

    spec = hub.HubSpec(repo_id="a/b", token_env="LIT_TEST_TOKEN_ENV")
    os.environ.pop("LIT_TEST_TOKEN_ENV", None)
    assert spec.token() is None
    os.environ["LIT_TEST_TOKEN_ENV"] = "hf_dummy"
    try:
        assert spec.token() == "hf_dummy"
        # The token must not be part of the serialised config, or it lands in a
        # run manifest and then in a public repo.
        assert "hf_dummy" not in json.dumps(repro._canonical(spec))
    finally:
        os.environ.pop("LIT_TEST_TOKEN_ENV", None)


def test_ignore_patterns_exclude_optimiser_state_and_checkpoints():
    # Optimiser shards are gigabytes and useless to a reader; intermediate
    # checkpoints make a repo unnavigable.
    patterns = hub.HubSpec(repo_id="a/b").ignore_patterns
    assert any("checkpoint-" in p for p in patterns)
    assert "optimizer.pt" in patterns


def test_resolve_checkpoint_returns_local_directories_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        assert hub.resolve_checkpoint(tmp) == Path(tmp)


def test_train_spec_exposes_metrics_and_hub_controls():
    spec = TrainSpec()
    assert spec.report_to == ("tensorboard",)
    assert spec.wer_eval_steps > 0 and spec.eval_steps > 0
    assert spec.hub is None and spec.push_each_stage is False
    assert spec.deterministic is False
    # tensorboard_dir defaults to <output_dir>/runs, resolved at run_stage time.
    assert spec.tensorboard_dir is None


def run() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Skip as exc:
            print(f"skip {name}  ({exc})")
        except Exception:
            import traceback

            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures} tests passed" if not failures else f"\n{failures} failing test(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
