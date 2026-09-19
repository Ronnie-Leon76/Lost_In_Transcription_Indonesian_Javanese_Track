"""Fail fast if a Runpod environment cannot run the packaged ASR submission.

Run this immediately after installing ``requirements-runpod.txt`` and before
starting the notebook or a long decode.  It intentionally checks imports and
CUDA only; it neither downloads a model nor touches competition data.
"""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import sys


EXPECTED = {
    "qwen-asr": "0.0.6",
    "transformers": "4.57.6",
    "vllm": "0.14.0",
}


def installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Skip the CUDA-device requirement (only useful for import diagnostics).",
    )
    args = parser.parse_args()

    failures: list[str] = []
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 12):
        failures.append("use a fresh Python 3.12 Runpod image")

    for package, expected in EXPECTED.items():
        actual = installed(package)
        print(f"{package}={actual or 'missing'} (expected {expected})")
        # A CUDA wheel may report a PEP 440 local suffix (for example
        # ``0.14.0+cu129``).  The upstream pin applies to its public version.
        if actual is None or actual.split("+", 1)[0] != expected:
            failures.append(f"{package} must be {expected}, found {actual or 'missing'}")

    # These imports catch the common partial-install and CUDA-extension errors.
    for module in ("qwen_asr", "vllm", "soundfile", "librosa"):
        try:
            importlib.import_module(module)
            print(f"import {module}: OK")
        except Exception as exc:
            failures.append(f"cannot import {module}: {type(exc).__name__}: {exc}")

    try:
        import torch

        print(f"torch={torch.__version__}; CUDA build={torch.version.cuda}")
        if not args.allow_cpu and not torch.cuda.is_available():
            failures.append("CUDA is not available to PyTorch; choose a GPU pod or fix its driver")
        elif torch.cuda.is_available():
            print(f"GPU={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        failures.append(f"cannot import torch: {type(exc).__name__}: {exc}")

    if failures:
        print("\nRunpod preflight FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nRunpod preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
