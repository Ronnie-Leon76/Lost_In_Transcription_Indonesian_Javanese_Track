"""Turn the development set into a local stand-in for the competition container's data.

    python scripts/make_local_test_data.py --out runtime/data
    python scripts/make_local_test_data.py --out runtime/data --limit 8   # smoke test

Writes, under ``--out``:

    clips/…                  copies (or symlinks) of the dev mp3s
    test_metadata.csv        audio_filename, file_duration_seconds, language
    submission_format.csv    audio_filename, transcript (placeholder)
    ground_truth.csv         audio_filename, transcript (references, for score.py)

``ground_truth.csv`` is written outside the container's ``data/`` view by default so a
local run cannot accidentally read the answers. Score a finished run with:

    python score.py runtime/ground_truth.csv --predicted-path runtime/submission/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lit_jv import data as lit_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "runtime" / "data")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Where to write the reference CSV. Defaults to the parent of --out.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N clips.")
    parser.add_argument(
        "--link",
        action="store_true",
        help="Symlink the audio instead of copying it (needs developer mode on Windows).",
    )
    args = parser.parse_args()

    rows = lit_data.load_dev_manifest(args.dev, with_durations=True)
    if args.limit:
        rows = rows[: args.limit]

    clips = args.out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = clips / row["audio_filename"]
        if target.exists():
            continue
        if args.link:
            target.symlink_to(Path(row["audio_path"]).resolve())
        else:
            shutil.copyfile(row["audio_path"], target)

    def write(path: Path, header, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(records)

    write(
        args.out / "test_metadata.csv",
        ["audio_filename", "file_duration_seconds", "language"],
        [(r["audio_filename"], round(r["duration"]), r["language"]) for r in rows],
    )
    write(
        args.out / "submission_format.csv",
        ["audio_filename", "transcript"],
        [(r["audio_filename"], "") for r in rows],
    )
    ground_truth = args.ground_truth or args.out.parent / "ground_truth.csv"
    write(
        ground_truth,
        ["audio_filename", "transcript"],
        [(r["audio_filename"], r["raw_text"]) for r in rows],
    )

    print(f"{len(rows)} clips -> {args.out}")
    print(f"references -> {ground_truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
