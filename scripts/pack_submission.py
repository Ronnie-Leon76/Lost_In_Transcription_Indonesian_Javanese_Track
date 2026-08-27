"""Assemble and verify ``submission.zip``.

    python scripts/pack_submission.py --model artifacts/merged-qwen3asr-1.7b

Layout inside the archive (``main.py`` must be at the root, not nested — the
single most common way to get a submission rejected):

    main.py
    submission_config.json
    postprocess.json
    postprocess.lexicon.json
    lit_jv/...
    model/...            <- merged Qwen3-ASR weights, tokenizer, chat template

The archive is verified after writing: entry points present, no ``__pycache__``,
no stray checkpoints, and the model directory actually contains weights. The
container has no network, so a missing asset is a failed run, not a slow one.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MODEL_FILES = ("config.json", "preprocessor_config.json", "chat_template.json")
EXCLUDE_PATTERNS = ("__pycache__", ".ipynb_checkpoints", ".DS_Store", ".pyc", ".git")


def should_skip(path: Path) -> bool:
    return any(pattern in str(path) for pattern in EXCLUDE_PATTERNS)


def add_tree(archive: zipfile.ZipFile, source: Path, prefix: str) -> tuple[int, int]:
    files = 0
    total = 0
    for path in sorted(source.rglob("*")):
        if path.is_dir() or should_skip(path):
            continue
        archive.write(path, f"{prefix}/{path.relative_to(source).as_posix()}")
        files += 1
        total += path.stat().st_size
    return files, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Merged checkpoint to embed as model/. Either a local directory or a "
        "Hugging Face repo id, which is downloaded first — the container has no "
        "network, so the weights must be in the archive. Omit for a code-only "
        "archive (useful for a wiring-only smoke test).",
    )
    parser.add_argument(
        "--model-subfolder",
        default="merged",
        help="Subfolder to pull when --model is a Hub repo id.",
    )
    parser.add_argument("--model-revision", default=None, help="Hub branch, tag or commit.")
    parser.add_argument("--src", type=Path, default=ROOT / "submission_src")
    parser.add_argument("--package", type=Path, default=ROOT / "src" / "lit_jv")
    parser.add_argument("--out", type=Path, default=ROOT / "submission" / "submission.zip")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    main_py = args.src / "main.py"
    if not main_py.exists():
        raise SystemExit(f"missing entrypoint: {main_py}")

    config_path = args.src / "submission_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    model_dir = None
    if args.model is not None:
        model_dir = Path(args.model)
        if not model_dir.is_dir():
            sys.path.insert(0, str(ROOT / "src"))
            from lit_jv import hub

            model_dir = hub.resolve_checkpoint(
                args.model, subfolder=args.model_subfolder, revision=args.model_revision
            )
        missing = [f for f in REQUIRED_MODEL_FILES if not (model_dir / f).exists()]
        if missing:
            raise SystemExit(
                f"{model_dir} is missing {missing}. Run lit_jv.train_lora.merge_and_save "
                "(it copies the tokenizer and chat template next to the weights)."
            )
        weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        if not weights:
            raise SystemExit(f"no model weights found in {model_dir}")
        if not (model_dir / "run_manifest.json").exists():
            print(
                f"warning: {model_dir} has no run_manifest.json, so the archive will not "
                "record which run produced these weights"
            )

    summary: dict[str, tuple[int, int]] = {}
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(args.src.iterdir()):
            if path.is_file() and not should_skip(path):
                archive.write(path, path.name)
        summary["submission_src"] = (
            sum(1 for p in args.src.iterdir() if p.is_file()),
            sum(p.stat().st_size for p in args.src.iterdir() if p.is_file()),
        )
        summary["lit_jv"] = add_tree(archive, args.package, "lit_jv")
        if model_dir is not None:
            # Weights are already compressed; storing them saves packing time.
            for path in sorted(model_dir.rglob("*")):
                if path.is_dir() or should_skip(path):
                    continue
                compress = (
                    zipfile.ZIP_STORED
                    if path.suffix in (".safetensors", ".bin")
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(
                    path,
                    f"model/{path.relative_to(model_dir).as_posix()}",
                    compress_type=compress,
                )
            summary["model"] = (
                sum(1 for p in model_dir.rglob("*") if p.is_file()),
                sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file()),
            )

    verify(args.out, expect_model=model_dir is not None, config=config)

    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1e9:.2f} GB on disk)")
    for name, (files, size) in summary.items():
        print(f"  {name:16s} {files:5d} files  {size / 1e6:9.1f} MB uncompressed")
    return 0


def verify(archive_path: Path, expect_model: bool, config: dict) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    problems: list[str] = []
    if "main.py" not in names:
        problems.append("main.py is not at the archive root")
    if not any(n.startswith("lit_jv/") for n in names):
        problems.append("lit_jv package missing")
    if "submission_config.json" not in names:
        problems.append("submission_config.json missing")
    if config.get("postprocess", "postprocess.json") not in names:
        problems.append(f"{config.get('postprocess', 'postprocess.json')} missing")
    if expect_model and not any(n.startswith("model/") for n in names):
        problems.append("model/ missing")
    if expect_model and not any(n.endswith(".safetensors") for n in names):
        problems.append("no safetensors weights in model/")
    bad = [n for n in names if should_skip(Path(n))]
    if bad:
        problems.append(f"{len(bad)} excluded files leaked in (e.g. {bad[0]})")
    if problems:
        raise SystemExit("archive verification failed:\n  - " + "\n  - ".join(problems))
    print("archive verified: entrypoint, package, config, rules" + (", weights" if expect_model else ""))


if __name__ == "__main__":
    raise SystemExit(main())
