"""Regression test for preserving a completed-pass checkpoint on later failure."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "submission_src")]


def run() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        output = tmp / "submission" / "submission.csv"
        old_data = os.environ.get("LIT_DATA_DIR")
        old_output = os.environ.get("LIT_SUBMISSION_PATH")
        os.environ["LIT_DATA_DIR"] = str(tmp / "data")
        os.environ["LIT_SUBMISSION_PATH"] = str(output)
        try:
            main = importlib.import_module("main")
            main = importlib.reload(main)
            from lit_jv import data as lit_data

            original_manifest = lit_data.load_test_manifest
            original_inference = main.run_inference
            lit_data.load_test_manifest = lambda _: [
                {"audio_filename": "one.mp3", "audio_path": "unused", "duration": 1.0}
            ]

            def fail_after_checkpoint(rows, config, deadline, filenames):
                main.write_submission(main.SUBMISSION_PATH, filenames, ["checkpoint text"])
                raise RuntimeError("optional second pass failed")

            main.run_inference = fail_after_checkpoint
            try:
                assert main.main() == 0
            finally:
                lit_data.load_test_manifest = original_manifest
                main.run_inference = original_inference
        finally:
            if old_data is None:
                os.environ.pop("LIT_DATA_DIR", None)
            else:
                os.environ["LIT_DATA_DIR"] = old_data
            if old_output is None:
                os.environ.pop("LIT_SUBMISSION_PATH", None)
            else:
                os.environ["LIT_SUBMISSION_PATH"] = old_output

        written = output.read_text(encoding="utf-8")
        assert "checkpoint text" in written
        assert "one.mp3" in written
    print("submission checkpoint-preservation test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
