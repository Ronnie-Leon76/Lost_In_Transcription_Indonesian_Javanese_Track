"""Use the official MDC SDK to check Bangor Miami metadata without downloading."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lit_jv.data import load_dotenv  # noqa: E402

BANGOR_MIAMI_DATASET_ID = "cmmfulo4r018bnz07py4q9t09"


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not (os.environ.get("MDC_API_KEY") or os.environ.get("LIT_MDC_API_KEY")):
        print("MDC_API_KEY is absent. Add it to a local .env or RunPod secret.", file=sys.stderr)
        return 2
    try:
        from datacollective import get_dataset_details
    except ImportError:
        print("Install the official SDK first: pip install 'datacollective>=0.5.5'", file=sys.stderr)
        return 2
    try:
        details = get_dataset_details(BANGOR_MIAMI_DATASET_ID)
    except Exception as exc:
        print(f"MDC SDK metadata check failed: {exc}", file=sys.stderr)
        return 1
    name = getattr(details, "name", None) or getattr(details, "title", None) or "Bangor Miami"
    print(f"MDC SDK metadata check succeeded: {name}. No download was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
