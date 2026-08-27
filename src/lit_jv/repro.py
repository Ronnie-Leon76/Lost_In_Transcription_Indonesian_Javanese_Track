"""Seeding, determinism and run provenance.

A leaderboard number is only useful if the run behind it can be rebuilt, and the
prize documentation asks for exactly that. Two separate jobs live here:

**Seeding** (:func:`seed_everything`) pins every generator the pipeline touches —
Python's ``random``, NumPy, torch CPU and CUDA, the hash seed, and the dataloader
workers, which are the usual culprit because each worker forks with its own RNG
state. Note the honest limit: identical seeds give identical results on the *same*
software and hardware. Change the GPU, the CUDA version or the batch size and the
numbers move slightly, because reduction order changes. That is why
:func:`environment_fingerprint` records what the run actually ran on.

**Provenance** (:func:`write_run_manifest`) writes a JSON record next to every
artefact: seeds, config hash, data hash, package versions, git commit, GPU. When
two runs disagree, the diff between their manifests is the answer, and there is no
archaeology to do.

``strict=True`` additionally asks torch for deterministic kernels. It is off by
default because it is slower and because some fused attention kernels have no
deterministic implementation — turning it on can raise at the first backward pass.
Use it to verify a result, not to produce every result.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Packages whose version can move a WER by more than rounding.
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "vllm",
    "qwen_asr",
    "librosa",
    "soundfile",
    "numpy",
    "jiwer",
)

DEFAULT_SEED = 3407


def seed_everything(seed: int = DEFAULT_SEED, strict: bool = False) -> int:
    """Seed every generator in the pipeline. Returns the seed, for logging.

    Call before building a model or a dataloader, not after.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        # cuBLAS needs this set *before* the first CUDA context to make GEMMs
        # deterministic; setting it afterwards is silently ignored.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass
    else:
        torch.backends.cudnn.benchmark = True
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give each dataloader worker a distinct, reproducible seed.

    Pass to ``DataLoader(worker_init_fn=...)``. The HF ``Trainer`` derives worker
    seeds from ``args.seed`` already; this is for the loaders we build ourselves.
    """
    import torch

    base = torch.initial_seed() % 2**31
    seed = (base + worker_id) % 2**31
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def generator(seed: int = DEFAULT_SEED):
    """A seeded ``torch.Generator`` for shuffling, for reproducible epoch order."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    from importlib import metadata

    for candidate in (name, name.replace("_", "-")):
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def _git_state(root: Path | None = None) -> dict:
    root = Path(root or Path(__file__).resolve().parents[2])

    def git(*args):
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = git("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit does not describe the code that ran.
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def environment_fingerprint() -> dict:
    """Everything that can change a number without the code changing."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in TRACKED_PACKAGES},
        "git": _git_state(),
        "env": {
            key: os.environ.get(key)
            for key in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")
            if os.environ.get(key)
        },
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
    except ImportError:
        info["torch"] = None
    return info


def _canonical(obj: Any) -> Any:
    """Convert dataclasses / Paths / sets into something JSON can hash stably."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _canonical(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, set):
        return sorted(str(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def config_fingerprint(config: Any) -> str:
    """Stable short hash of a config object, insensitive to key order."""
    payload = json.dumps(_canonical(config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def manifest_fingerprint(rows: Sequence[dict]) -> dict:
    """Identify a training set by content, not by path.

    Hashes the (filename, target) pairs rather than the audio, which is enough to
    catch the failures that matter — a re-segmentation that moved boundaries, a
    corpus that silently lost rows, targets built in the wrong space — without
    reading gigabytes.
    """
    digest = hashlib.sha256()
    hours = 0.0
    corpora: dict[str, int] = {}
    for row in sorted(rows, key=lambda r: (str(r.get("audio_path", "")), str(r.get("text", "")))):
        digest.update(Path(str(row.get("audio_path", ""))).name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(row.get("text", "")).encode("utf-8"))
        digest.update(b"\x00")
        duration = row.get("duration", 0.0)
        if duration == duration and duration:
            hours += float(duration) / 3600
        corpus = str(row.get("corpus", "?"))
        corpora[corpus] = corpora.get(corpus, 0) + 1
    return {
        "rows": len(rows),
        "hours": round(hours, 3),
        "by_corpus": dict(sorted(corpora.items())),
        "sha256_16": digest.hexdigest()[:16],
    }


def file_fingerprints(paths: Iterable[str | Path]) -> dict:
    """Size and hash per file, for artefacts that are shipped or published."""
    out = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        out[path.name] = {"bytes": path.stat().st_size, "sha256_16": digest.hexdigest()[:16]}
    return out


def write_run_manifest(path: str | Path, **sections: Any) -> Path:
    """Write a provenance record. Always includes the environment fingerprint."""
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment_fingerprint(),
        **{key: _canonical(value) for key, value in sections.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path


def summarise(manifest: dict) -> str:
    """One-line human summary of a run manifest, for notebook output."""
    env = manifest.get("environment", {})
    git = env.get("git", {})
    torch_info = env.get("torch") or {}
    device = (torch_info.get("devices") or ["cpu"])[0]
    commit = (git.get("commit") or "no-git")[:8]
    dirty = " +dirty" if git.get("dirty") else ""
    return (
        f"seed={manifest.get('seed')} config={manifest.get('config_fingerprint')} "
        f"data={(manifest.get('data') or {}).get('sha256_16')} "
        f"commit={commit}{dirty} device={device} torch={torch_info.get('version')}"
    )
