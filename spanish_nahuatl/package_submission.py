"""Build an offline Spanish-Nahuatl ``submission.zip`` from a merged model.

Run this only after selecting the model by held-out-conversation WER.  It copies
the generic runtime mechanics, but hard-codes Spanish-Nahuatl prompts and never
includes development audio, MDC credentials, notebooks, or other-track assets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="Merged model directory")
    parser.add_argument(
        "--glossary",
        required=True,
        type=Path,
        help="JSON glossary emitted by the selected training run",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "spnh" / "submission.zip")
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"merged model not found: {args.model}")
    try:
        glossary = json.loads(args.glossary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not load glossary JSON: {args.glossary}") from exc
    if not isinstance(glossary, list) or not all(isinstance(item, str) for item in glossary):
        raise SystemExit("glossary must be a JSON list of strings")

    staging = args.out.with_suffix("")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(ROOT / "submission_src" / "main.py", staging / "main.py")
    shutil.copytree(ROOT / "src" / "lit_jv", staging / "lit_jv")
    shutil.copytree(args.model, staging / "model")

    config = {
        "model_dir": "model",
        "backend": "vllm",
        "time_budget_s": 6300,
        "chunk_size": 48,
        "glossary": glossary,
        "passes": [{
            "name": "spnh-selected",
            "prose": "codeswitch_spnh",
            "use_language_field": True,
            "style_tag": "wa-voicenote-sp-nh",
            "include_style_tag": True,
            "glossary": True,
            "max_glossary_terms": len(glossary),
            "force_language": "Spanish",
            "temperature": 0.0,
        }],
        "combiner": "none",
        "combiner_by_n": {},
        "fallback": {"enabled": True, "split_long_clips": True},
    }
    (staging / "submission_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging))
    print(f"created {args.out}")
    print("archive root contains main.py:", (staging / "main.py").is_file())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
