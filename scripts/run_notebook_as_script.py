"""Execute every code cell of the notebook as one flat script.

A notebook that has not been run end to end in a fresh kernel is a notebook with a
`NameError` in it, and the failure always surfaces at the worst moment. This
concatenates the code cells in order and executes them in a single namespace, which
is exactly the "restart and run all" contract without needing a kernel or a browser.

CPU-only by design: the GPU sections skip themselves when CUDA is absent, so what
this checks is the wiring — imports, names, shapes of the dataframes, the
simulated-hypothesis fallbacks, and the packaging and dry-run subprocesses at the
end. Run it after any edit to the notebook.

    python scripts/run_notebook_as_script.py [--out artifacts/_flat_notebook.py]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "lost_in_transcription_id_jv.ipynb"


def flatten(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    chunks = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        body = "".join(cell["source"])
        # IPython magics (%tensorboard, %load_ext) are not valid Python.
        if body.lstrip().startswith("%"):
            continue
        chunks.append(f"# ---- cell {index} ----\n{body}")
    return "\n\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "_flat_notebook.py")
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="Flatten without executing, for inspecting the script by hand.",
    )
    args = parser.parse_args()

    script = flatten(args.notebook)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(script, encoding="utf-8")
    print(f"flattened {args.notebook.name} -> {args.out} ({script.count(chr(10))} lines)")
    if args.write_only:
        return 0

    sys.path.insert(0, str(ROOT / "src"))
    from lit_jv.data import load_dotenv, resolve_dev_dir, dev_dir_ready

    load_dotenv(ROOT / ".env")
    dev_dir = resolve_dev_dir()
    if not dev_dir_ready(dev_dir):
        print(
            f"development set not found at {dev_dir / 'metadata.tsv'}\n"
            "Source the env script, then make sure indonesian_dev/ is on this machine:\n"
            "  source scripts/setup_env.sh\n"
            "On a private repo: git add indonesian_dev && git commit && git push, then git pull here.",
            file=sys.stderr,
        )
        return 2

    try:
        import matplotlib

        matplotlib.use("Agg")  # no display, and plt.show() must not block
    except ImportError:
        pass

    sys.argv = [str(args.out)]
    namespace = {"__name__": "__main__", "__file__": str(args.out)}
    exec(compile(script, str(args.out), "exec"), namespace)
    print("\n== notebook executed cleanly ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
