"""Build an offline Spanish-English competition archive from merged weights.

``--postprocess`` embeds the rule pipeline fitted in the notebook (plus its
lexicon), and ``--decode-config`` embeds the measured pass list, combiner and
length prior. Without them the archive falls back to one greedy pass and no
post-processing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PASS = {
    "name": "spen-selected", "prose": "codeswitch_spen", "use_language_field": True,
    "style_tag": "wa-voicenote-sp-en", "include_style_tag": True, "glossary": True,
    "max_glossary_terms": 128, "force_language": "Spanish", "temperature": 0.0,
}
#: Keys ``--decode-config`` may set; anything else is rejected so a typo cannot
#: silently leave a default in place.
DECODE_KEYS = {"passes", "combiner", "combiner_by_n", "length_prior", "fallback", "rover", "chunk_size", "time_budget_s"}


def copy_postprocess(source: Path, staging: Path) -> str:
    payload = json.loads(source.read_text(encoding="utf-8"))
    lexicon = source.parent / payload["lexicon"]
    if not lexicon.is_file():
        raise SystemExit(f"post-processing lexicon not found: {lexicon}")
    shutil.copy2(source, staging / "postprocess.json")
    shutil.copy2(lexicon, staging / payload["lexicon"])
    print(f"post-processing rules: {payload.get('rules', [])}")
    return "postprocess.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--glossary", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, help="postprocess.json saved by RulePipeline.save")
    parser.add_argument("--decode-config", type=Path, help="JSON with passes/combiner/combiner_by_n/length_prior")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "spen" / "submission.zip")
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"merged model not found: {args.model}")
    glossary = json.loads(args.glossary.read_text(encoding="utf-8"))
    if not isinstance(glossary, list) or not all(isinstance(item, str) for item in glossary):
        raise SystemExit("glossary must be a JSON list of strings")
    decode = json.loads(args.decode_config.read_text(encoding="utf-8")) if args.decode_config else {}
    unknown = set(decode) - DECODE_KEYS
    if unknown:
        raise SystemExit(f"unknown decode-config keys: {sorted(unknown)}")
    if "passes" in decode and (not decode["passes"] or not all(isinstance(p, dict) and p.get("name") for p in decode["passes"])):
        raise SystemExit("decode-config 'passes' must be a non-empty list of objects with a 'name'")

    staging = args.out.with_suffix("")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(ROOT / "submission_src" / "main.py", staging / "main.py")
    shutil.copytree(ROOT / "src" / "lit_jv", staging / "lit_jv", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(args.model, staging / "model")
    config = {
        "model_dir": "model", "backend": "vllm", "time_budget_s": 6300, "chunk_size": 48,
        "glossary": glossary, "combiner": "none", "combiner_by_n": {},
        "passes": [{**DEFAULT_PASS, "max_glossary_terms": len(glossary)}],
        "fallback": {"enabled": True, "split_long_clips": True},
        **decode,
    }
    if args.postprocess:
        config["postprocess"] = copy_postprocess(args.postprocess, staging)
    (staging / "submission_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging))

    with zipfile.ZipFile(args.out) as archive:
        names = set(archive.namelist())
    required = {"main.py", "submission_config.json", "lit_jv/__init__.py", "model/config.json"}
    if args.postprocess:
        required.add("postprocess.json")
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"archive is missing {missing}")
    print(f"passes: {[p['name'] for p in config['passes']]} combiner={config['combiner']!r}")
    print(f"created {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
