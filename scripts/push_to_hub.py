"""Push adapters, logs, the run manifest, and optionally the merged checkpoint.

    source scripts/setup_env.sh
    python scripts/push_to_hub.py
    python scripts/push_to_hub.py --merged artifacts/merged-qwen3asr --upload-merged

Requires ``LIT_HF_REPO`` and ``HF_TOKEN`` (in the environment or in ``.env``).
Adapters are tens of megabytes. The merged 1.7B checkpoint is gigabytes and is
only uploaded with ``--upload-merged`` or ``LIT_HF_UPLOAD_MERGED=1``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lit_jv import hub  # noqa: E402
from lit_jv.data import load_dotenv  # noqa: E402


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None, help="Override LIT_HF_REPO.")
    parser.add_argument("--adapters", type=Path, default=ROOT / "artifacts" / "qwen3asr-lora")
    parser.add_argument("--merged", type=Path, default=ROOT / "artifacts" / "merged-qwen3asr")
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument(
        "--upload-merged",
        action="store_true",
        help="Also push the merged 1.7B weights (gigabytes).",
    )
    parser.add_argument("--private", action="store_true", default=None)
    args = parser.parse_args()

    spec = hub.HubSpec.from_env()
    if args.repo:
        spec = spec or hub.HubSpec(repo_id=args.repo)
        spec.repo_id = args.repo
    if spec is None:
        print(
            "LIT_HF_REPO is not set. Put it in .env or pass --repo your-name/lit-jv-qwen3asr-lora",
            file=sys.stderr,
        )
        return 2
    if not spec.token():
        print("HF_TOKEN is not set. Add a write token to .env or the environment.", file=sys.stderr)
        return 2
    if args.upload_merged:
        spec.upload_merged = True
    if args.private is True:
        spec.private = True

    adapters = {}
    if args.adapters.is_dir():
        for child in sorted(args.adapters.iterdir()):
            if child.is_dir() and (child / "adapter_config.json").exists():
                adapters[child.name] = child
    if not adapters and not (args.merged.is_dir() and spec.upload_merged):
        print(f"nothing to push under {args.adapters} or {args.merged}", file=sys.stderr)
        return 2

    manifest = None
    for path in list(adapters.values()) + ([args.merged] if args.merged.is_dir() else []):
        candidate = Path(path) / "run_manifest.json"
        if candidate.is_file():
            manifest = candidate
            break

    extra = {}
    for rel in (
        ROOT / "submission_src" / "submission_config.json",
        ROOT / "submission_src" / "postprocess.json",
        ROOT / "artifacts" / "ablations.csv",
    ):
        if rel.is_file():
            extra[rel.name] = rel

    result = hub.publish_run(
        spec,
        adapters=adapters,
        merged_dir=args.merged if args.merged.is_dir() else None,
        tensorboard_dir=args.runs if args.runs.is_dir() else None,
        manifest_path=manifest,
        extra_files=extra,
    )
    print(result.url())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
