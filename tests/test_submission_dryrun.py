"""End-to-end dry run of the packaged submission, with a fake decoder.

Exercises the code path that runs on the leaderboard — manifest reading,
multi-pass decoding, ROVER combination, post-processing, atomic CSV writes,
the time budget and the failure guards — without a GPU. The fake backend
returns style-damaged references, so the assertions also confirm that the
post-processor is wired in and improving WER rather than merely running.

    python tests/test_submission_dryrun.py
"""

from __future__ import annotations

import csv
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "submission_src"))

from lit_jv import data as lit_data  # noqa: E402
from lit_jv import decode as lit_decode  # noqa: E402
from lit_jv import postprocess as lit_post  # noqa: E402
from lit_jv import scoring  # noqa: E402

DEV_DIR = lit_data.resolve_dev_dir()
N_CLIPS = 12


#: One corruption per pass, each wrong in a different place, so combination and the
#: rule pipeline both have real work to do. Index i is used by pass i.
CORRUPTIONS = (
    lambda t: t.lower(),
    lambda t: t.replace("-", " "),
    lambda t: t.replace("...", " \u2026 "),
    lambda t: t.replace("é", "e"),
)


class FakeBackend(lit_decode.Qwen3AsrBackend):
    """Returns a per-pass corruption of the reference for each clip."""

    def __init__(self, references: dict[str, str]):
        self.references = references
        self.calls = 0
        self.pass_index = 0

    def generate(self, prompts, waveforms, max_tokens, temperature=0.0, n=1, seed=None):
        self.calls += 1
        corrupt = CORRUPTIONS[self.pass_index % len(CORRUPTIONS)]
        out = []
        for key in waveforms:  # the fake receives filenames instead of arrays
            text = corrupt(self.references[key])
            out.append(
                [
                    {
                        "text": text,
                        "avg_logprob": -0.2,
                        "finish_reason": "stop",
                        "n_tokens": len(text.split()) * 2,
                    }
                ]
                * n
            )
        return out


def _build_fixture(tmp: Path):
    rows = lit_data.load_dev_manifest(DEV_DIR, with_durations=True)[:N_CLIPS]
    data_dir = tmp / "data"
    (data_dir / "clips").mkdir(parents=True)
    for row in rows:
        shutil.copyfile(row["audio_path"], data_dir / "clips" / row["audio_filename"])
    with open(data_dir / "submission_format.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["audio_filename", "transcript"])
        for row in rows:
            writer.writerow([row["audio_filename"], "hello world"])
    with open(data_dir / "test_metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["audio_filename", "file_duration_seconds", "language"])
        for row in rows:
            writer.writerow([row["audio_filename"], round(row["duration"]), row["language"]])
    return rows, data_dir


def scenario(label: str, pipeline, config_override: dict | None = None) -> None:
    """Run main.py end to end against a fake decoder and check the output."""
    import os

    print(f"\n--- {label} " + "-" * (60 - len(label)))
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        rows, data_dir = _build_fixture(tmp)
        references = {r["audio_filename"]: r["text"] for r in rows}
        submission_path = tmp / "submission" / "submission.csv"

        # Run against a copy of the shipped config, with the rules redirected into the
        # temporary directory. The archive's own postprocess.json is fitted by the
        # notebook against real hypotheses and must never be clobbered by a test.
        config = json.loads(
            (ROOT / "submission_src" / "submission_config.json").read_text("utf-8")
        )
        config.update(config_override or {})
        config["postprocess"] = str(pipeline.save(tmp / "postprocess.json"))
        config_path = tmp / "submission_config.json"
        config_path.write_text(json.dumps(config), "utf-8")
        os.environ["LIT_CONFIG_PATH"] = str(config_path)

        fake = FakeBackend(references)

        def fake_build_backend(model_path, prefer="vllm", vllm_kwargs=None, transformers_kwargs=None, log=print):
            log("decoder backend: fake (dry run)")
            return fake

        # The fake consumes filenames, so bypass audio decoding entirely.
        def fake_load_batch(batch_rows):
            return [r["audio_filename"] for r in batch_rows]

        original_build, original_load = lit_decode.build_backend, lit_decode.Qwen3AsrDecoder.load_batch
        lit_decode.build_backend = fake_build_backend
        lit_decode.Qwen3AsrDecoder.load_batch = staticmethod(fake_load_batch)

        # Each pass must corrupt differently; the pass index advances per pass.
        original_decode_pass = lit_decode.Qwen3AsrDecoder.decode_pass

        def counting_decode_pass(self, rows_, config, **kwargs):
            result = original_decode_pass(self, rows_, config, **kwargs)
            fake.pass_index += 1
            return result

        lit_decode.Qwen3AsrDecoder.decode_pass = counting_decode_pass

        os.environ["LIT_DATA_DIR"] = str(data_dir)
        os.environ["LIT_SUBMISSION_PATH"] = str(submission_path)
        try:
            main_module = importlib.import_module("main")
            importlib.reload(main_module)
            exit_code = main_module.main()
        finally:
            lit_decode.build_backend = original_build
            lit_decode.Qwen3AsrDecoder.load_batch = original_load
            lit_decode.Qwen3AsrDecoder.decode_pass = original_decode_pass
            os.environ.pop("LIT_DATA_DIR", None)
            os.environ.pop("LIT_SUBMISSION_PATH", None)
            os.environ.pop("LIT_CONFIG_PATH", None)

        assert exit_code == 0
        assert submission_path.exists(), "no submission.csv produced"
        with open(submission_path, encoding="utf-8", newline="") as f:
            written = list(csv.DictReader(f))

        assert [r["audio_filename"] for r in written] == [r["audio_filename"] for r in rows]
        assert list(written[0]) == ["audio_filename", "transcript"]
        assert all(r["transcript"].strip() for r in written), "empty transcripts written"

        refs = [references[r["audio_filename"]] for r in written]
        hyps = [r["transcript"] for r in written]
        combined_wer = scoring.corpus_wer(refs, hyps)

        n_passes = len(config["passes"])
        assert fake.calls >= n_passes, "not every configured pass ran"

        # Each corrupted pass, post-processed the same way, is the bar the
        # combined output has to clear.
        singles = {
            index: scoring.corpus_wer(refs, [pipeline.apply(CORRUPTIONS[index](r)) for r in refs])
            for index in range(n_passes)
        }
        # Mirrors main.choose_combiner: one pass means nothing to combine.
        if n_passes < 2:
            combiner = "none"
        else:
            combiner = (config.get("combiner_by_n") or {}).get(str(n_passes)) or config.get(
                "combiner"
            )
        print(f"  passes {n_passes}, combiner {combiner}")
        print(f"  combined WER          {combined_wer:.4f}")
        for index, wer in singles.items():
            print(f"  single pass {index} WER   {wer:.4f}")

        if n_passes >= 3 and combiner == "rover":
            # ROVER fixes each slot independently, so it must beat every input.
            assert combined_wer <= min(singles.values()) + 1e-9, (combined_wer, singles)
        else:
            # Fewer than three voters means no majority to vote with, so the only
            # contract is that combining never does worse than the first pass -
            # which is why the passes are ordered strongest-first.
            assert combined_wer <= singles[0] + 1e-9, (combined_wer, singles)


def run() -> int:
    if not lit_data.dev_dir_ready(DEV_DIR):
        print(
            f"skip submission dry run: no development set at {DEV_DIR}\n"
            "copy indonesian_dev/ onto this machine, or set LIT_DEV_DIR"
        )
        return 0
    all_rows = lit_data.load_dev_manifest(DEV_DIR, with_durations=False)
    lexicon = lit_post.StyleLexicon.fit(r["text"] for r in all_rows)
    pipeline = lit_post.RulePipeline(lexicon=lexicon)

    scenario("shipped configuration", pipeline)
    # What happens when the clock cuts the plan down: two passes, no majority, so
    # combination must fall back rather than vote.
    shipped = json.loads((ROOT / "submission_src" / "submission_config.json").read_text("utf-8"))
    scenario(
        "clock-reduced plan (2 passes, combiner disabled)",
        pipeline,
        {"passes": shipped["passes"][:2], "combiner": "none", "combiner_by_n": {"2": "none"}},
    )
    scenario(
        "single pass (worst case)",
        pipeline,
        {"passes": shipped["passes"][:1]},
    )
    print("\nsubmission dry run passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
