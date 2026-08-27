"""Hugging Face Hub publishing and retrieval for checkpoints and logs.

Why bother, when the submission has to carry its own weights anyway:

* **Reference.** ``artifacts/merged-qwen3asr/`` on a rented GPU box disappears with
  the box. A Hub revision does not, and it is addressable from the notebook, the
  packaging script and the prize documentation by one string.
* **Provenance.** Every push carries the run manifest from :mod:`lit_jv.repro`, so
  a checkpoint is never separated from the seed, config hash, data hash, package
  versions and commit that produced it.
* **Metrics.** TensorBoard event files pushed under ``runs/`` are rendered by the
  Hub's Training-metrics tab, so the loss and held-out WER curves live next to the
  weights they explain.
* **Rules.** Winning solutions must be reproducible and openly licensed. A public
  repo with a model card, an MPL-2.0 licence and a manifest satisfies that by
  construction rather than by scramble.

**The container never talks to the Hub.** It has no network. Everything here runs
on the development box; :func:`resolve_checkpoint` is the bridge, pulling a
revision to a local directory *before* ``pack_submission.py`` embeds it.

Authentication uses the standard ``HF_TOKEN`` environment variable or a cached
``huggingface-cli login``. No token is ever written to an artefact or a log.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

#: Never upload these, whatever the folder contains: optimiser state is large and
#: useless to a reader, and caches/checkpoints are noise.
DEFAULT_IGNORE = (
    "*.pt",
    "optimizer.bin",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*.pth",
    "**/checkpoint-*/**",
    "**/__pycache__/**",
    "*.lock",
    ".DS_Store",
)


@dataclass
class HubSpec:
    """Where and how to publish.

    ``repo_id`` is ``"<user-or-org>/<name>"``. Set ``private=True`` while iterating
    and flip it to public before submitting, since the rules require a winning
    solution to be independently verifiable.
    """

    repo_id: str
    private: bool = True
    #: Environment variable holding the write token. Read at call time, never stored.
    token_env: str = "HF_TOKEN"
    #: Branch to push to. A per-experiment branch keeps ``main`` meaning "the one
    #: that was submitted".
    revision: str | None = None
    license: str = "mpl-2.0"
    base_model: str = "Qwen/Qwen3-ASR-1.7B"
    tags: tuple[str, ...] = (
        "automatic-speech-recognition",
        "code-switching",
        "indonesian",
        "javanese",
        "lora",
        "qwen3-asr",
    )
    ignore_patterns: tuple[str, ...] = DEFAULT_IGNORE
    #: Uploading a merged 1.7B checkpoint is minutes of bandwidth. Off by default
    #: so an accidental cell run does not start one.
    upload_merged: bool = False
    extra_card_notes: str = ""

    def token(self) -> str | None:
        return os.environ.get(self.token_env) or None


def _api(spec: HubSpec):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "huggingface_hub is required for Hub operations. "
            "Install it with `pip install huggingface_hub` on the development box "
            "(the competition container neither needs nor has network access)."
        ) from exc
    return HfApi(token=spec.token())


def ensure_repo(spec: HubSpec, repo_type: str = "model") -> str:
    """Create the repo (and branch) if missing. Idempotent."""
    api = _api(spec)
    api.create_repo(
        repo_id=spec.repo_id, private=spec.private, repo_type=repo_type, exist_ok=True
    )
    if spec.revision:
        try:
            api.create_branch(
                repo_id=spec.repo_id, branch=spec.revision, repo_type=repo_type, exist_ok=True
            )
        except TypeError:  # older hub versions have no exist_ok
            try:
                api.create_branch(
                    repo_id=spec.repo_id, branch=spec.revision, repo_type=repo_type
                )
            except Exception:
                pass
    return spec.repo_id


def push_folder(
    local_dir: str | Path,
    spec: HubSpec,
    path_in_repo: str,
    commit_message: str,
    repo_type: str = "model",
    allow_patterns: Sequence[str] | None = None,
) -> str:
    """Upload a directory to ``path_in_repo``. Returns the commit URL."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"nothing to push: {local_dir}")
    ensure_repo(spec, repo_type=repo_type)
    api = _api(spec)
    result = api.upload_folder(
        repo_id=spec.repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        revision=spec.revision,
        ignore_patterns=list(spec.ignore_patterns),
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )
    return getattr(result, "commit_url", "") or str(result)


def push_file(
    local_path: str | Path, spec: HubSpec, path_in_repo: str, commit_message: str
) -> str:
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    ensure_repo(spec)
    api = _api(spec)
    result = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=spec.repo_id,
        commit_message=commit_message,
        revision=spec.revision,
    )
    return getattr(result, "commit_url", "") or str(result)


# --------------------------------------------------------------------------
# Model card
# --------------------------------------------------------------------------


def render_model_card(
    spec: HubSpec,
    metrics: Sequence[dict] = (),
    manifest: dict | None = None,
    stages: Sequence[dict] = (),
    notes: str = "",
) -> str:
    """Build a model card that documents the run rather than advertising it."""
    manifest = manifest or {}
    environment = manifest.get("environment", {})
    packages = {k: v for k, v in (environment.get("packages") or {}).items() if v}
    git = environment.get("git", {}) or {}

    front_matter = {
        "license": spec.license,
        "base_model": spec.base_model,
        "tags": list(spec.tags),
        "language": ["id", "jv"],
        "library_name": "peft",
        "pipeline_tag": "automatic-speech-recognition",
    }
    lines = ["---"]
    for key, value in front_matter.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"- {v}" for v in value)
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {spec.repo_id.split('/')[-1]}")
    lines.append("")
    lines.append(
        "LoRA adaptation of "
        f"[{spec.base_model}](https://huggingface.co/{spec.base_model}) for "
        "code-switched Indonesian-Javanese conversational speech, trained for the "
        "[Lost in Transcription](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/) "
        "Indonesian-Javanese track."
    )
    lines.append("")

    if metrics:
        keys = [k for k in metrics[0] if k != "system"]
        lines.append("## Results")
        lines.append("")
        lines.append("| system | " + " | ".join(keys) + " |")
        lines.append("|---" * (len(keys) + 1) + "|")
        for row in metrics:
            lines.append(
                f"| {row.get('system', '?')} | "
                + " | ".join(str(row.get(k, "")) for k in keys)
                + " |"
            )
        lines.append("")
        lines.append(
            "WER is computed with the competition's own normaliser "
            "(`lit_jv.scoring`, a verbatim port of `score.py`) on a held-out "
            "conversation: the development set shares no speaker or conversation "
            "with the test set, so a random split would overstate every row."
        )
        lines.append("")

    if stages:
        lines.append("## Training curriculum")
        lines.append("")
        lines.append("| stage | data | epochs | lr |")
        lines.append("|---|---|---|---|")
        for stage in stages:
            lines.append(
                f"| {stage.get('name')} | {json.dumps(stage.get('corpus_weights', {}))} "
                f"| {stage.get('epochs')} | {stage.get('learning_rate')} |"
            )
        lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    if manifest:
        lines.append(f"* seed: `{manifest.get('seed')}`")
        lines.append(f"* config fingerprint: `{manifest.get('config_fingerprint')}`")
        data = manifest.get("data") or {}
        if data:
            lines.append(
                f"* training data: {data.get('rows')} rows, {data.get('hours')} h, "
                f"content hash `{data.get('sha256_16')}`"
            )
        if git.get("available"):
            lines.append(
                f"* commit: `{git.get('commit', '')[:12]}`"
                + (" (working tree dirty at run time)" if git.get("dirty") else "")
            )
        devices = (environment.get("torch") or {}).get("devices") or []
        if devices:
            lines.append(f"* hardware: {devices[0]}")
        if packages:
            lines.append(
                "* versions: "
                + ", ".join(f"`{name}=={version}`" for name, version in sorted(packages.items()))
            )
    lines.append(
        "* `run_manifest.json` in this repository carries the full record. Identical "
        "seeds reproduce identical results on the same software and hardware; a "
        "different GPU or batch size changes reduction order and moves WER slightly."
    )
    lines.append("")
    lines.append("## Usage")
    lines.append("")
    lines.append("```python")
    lines.append("from lit_jv import hub")
    lines.append("")
    lines.append("# pull the merged checkpoint to a local directory (needs network)")
    lines.append(f'path = hub.resolve_checkpoint("{spec.repo_id}", subfolder="merged")')
    lines.append("")
    lines.append("from lit_jv import decode")
    lines.append('backend = decode.build_backend(path, prefer="vllm")')
    lines.append("```")
    lines.append("")
    lines.append(
        "Training metrics (loss and held-out WER per step) are in `runs/` and render "
        "under the Training metrics tab."
    )
    if notes or spec.extra_card_notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append(notes or spec.extra_card_notes)
    lines.append("")
    return "\n".join(lines)


def push_model_card(spec: HubSpec, card: str, commit_message: str = "Update model card") -> str:
    ensure_repo(spec)
    api = _api(spec)
    result = api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=spec.repo_id,
        commit_message=commit_message,
        revision=spec.revision,
    )
    return getattr(result, "commit_url", "") or str(result)


# --------------------------------------------------------------------------
# High-level publishing
# --------------------------------------------------------------------------


@dataclass
class PublishResult:
    repo_id: str
    revision: str | None
    uploaded: dict = field(default_factory=dict)

    def url(self) -> str:
        base = f"https://huggingface.co/{self.repo_id}"
        return f"{base}/tree/{self.revision}" if self.revision else base


def publish_run(
    spec: HubSpec,
    adapters: dict[str, str | Path] | None = None,
    merged_dir: str | Path | None = None,
    tensorboard_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    extra_files: dict[str, str | Path] | None = None,
    metrics: Sequence[dict] = (),
    stages: Sequence[dict] = (),
    notes: str = "",
    log=print,
) -> PublishResult:
    """Publish a training run: adapters, logs, manifest, card, optionally weights.

    Adapters are tens of megabytes and always pushed; the merged checkpoint is
    gigabytes and only pushed when ``spec.upload_merged`` is set.
    """
    result = PublishResult(repo_id=spec.repo_id, revision=spec.revision)
    ensure_repo(spec)

    for name, path in (adapters or {}).items():
        target = f"adapters/{name}"
        log(f"pushing adapter {name} -> {target}")
        result.uploaded[target] = push_folder(
            path, spec, target, f"Add LoRA adapter: {name}"
        )

    if merged_dir is not None:
        if spec.upload_merged:
            log("pushing merged checkpoint (this is gigabytes)")
            result.uploaded["merged"] = push_folder(
                merged_dir, spec, "merged", "Add merged checkpoint"
            )
        else:
            log("skipping merged checkpoint: set HubSpec.upload_merged=True to include it")

    if tensorboard_dir is not None and Path(tensorboard_dir).is_dir():
        log("pushing TensorBoard event files -> runs/")
        result.uploaded["runs"] = push_folder(
            tensorboard_dir, spec, "runs", "Add training metrics"
        )

    manifest = {}
    if manifest_path is not None and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        result.uploaded["run_manifest.json"] = push_file(
            manifest_path, spec, "run_manifest.json", "Add run manifest"
        )

    for path_in_repo, local in (extra_files or {}).items():
        if Path(local).is_file():
            result.uploaded[path_in_repo] = push_file(
                local, spec, path_in_repo, f"Add {path_in_repo}"
            )

    card = render_model_card(spec, metrics=metrics, manifest=manifest, stages=stages, notes=notes)
    result.uploaded["README.md"] = push_model_card(spec, card)
    log(f"published: {result.url()}")
    return result


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def resolve_checkpoint(
    repo_or_path: str | Path,
    subfolder: str | None = None,
    revision: str | None = None,
    local_dir: str | Path | None = None,
    token_env: str = "HF_TOKEN",
    log=print,
) -> Path:
    """Return a local directory for a checkpoint, downloading it if necessary.

    Accepts a local path (returned unchanged) or a Hub repo id. Use this before
    packaging so the archive embeds real files: the container cannot download.
    """
    candidate = Path(repo_or_path)
    if candidate.is_dir():
        return candidate

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("huggingface_hub is required to download a checkpoint") from exc

    patterns = [f"{subfolder}/**"] if subfolder else None
    log(f"downloading {repo_or_path}" + (f" [{subfolder}]" if subfolder else ""))
    path = Path(
        snapshot_download(
            repo_id=str(repo_or_path),
            revision=revision,
            allow_patterns=patterns,
            local_dir=str(local_dir) if local_dir else None,
            token=os.environ.get(token_env) or None,
        )
    )
    resolved = path / subfolder if subfolder else path
    if not resolved.is_dir():
        raise FileNotFoundError(f"{subfolder!r} not found in {repo_or_path}")
    return resolved
