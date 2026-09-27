"""Stage Tetelancingo Nahuatl with the official MDC Python SDK."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lit_jv.data import load_dotenv  # noqa: E402

TETELANCINGO_DATASET_ID = "cmhkl8z2a007rnr07p9bm5kmz"
DEFAULT_OUT = ROOT / "data" / "spanish_nahuatl" / "tetelancingo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true", help="Check MDC metadata; never download.")
    parser.add_argument("--download", action="store_true", help="Start or resume the SDK download.")
    args = parser.parse_args()
    if args.check_only == args.download:
        parser.error("choose exactly one of --check-only or --download")
    load_dotenv(ROOT / ".env")
    if not (os.environ.get("MDC_API_KEY") or os.environ.get("LIT_MDC_API_KEY")):
        raise SystemExit("MDC_API_KEY is absent; set it in .env or as a RunPod secret.")
    try:
        from datacollective import download_dataset, get_dataset_details
    except ImportError as exc:
        raise SystemExit("Install the official SDK: pip install 'datacollective>=0.5.5'") from exc
    target = args.out.resolve()
    if target.is_dir() and any(not p.name.startswith(".") for p in target.iterdir()):
        print(f"Tetelancingo already staged at {target}; nothing to do.")
        return 0
    if args.check_only:
        get_dataset_details(TETELANCINGO_DATASET_ID)
        print("MDC SDK metadata check succeeded; no download was started.")
        return 0
    downloaded = Path(download_dataset(TETELANCINGO_DATASET_ID, enable_logging=True)).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if downloaded != target:
        if target.exists():
            # Placeholders such as .gitkeep do not count as a staged corpus.
            if any(not p.name.startswith(".") for p in target.iterdir()):
                raise SystemExit(f"destination is not empty: {target}")
            shutil.rmtree(target)
        shutil.move(str(downloaded), str(target))
    print(f"Tetelancingo ready: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
