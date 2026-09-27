"""Manifests, corpus ingestion and the validation split.

A manifest row is a flat dict with a fixed schema so that every corpus — the
provided development set, the Jember session recordings, MDC read-speech
corpora — reaches the trainer through the same code path:

    audio_path            absolute path to the clip
    text                  target transcript, already in scoring space
    raw_text              transcript as distributed (kept for auditing)
    language              javind | jav | ind
    style_tag             prompts.STYLE_* — which transcription convention
    speaker, convo_id     grouping keys for cross-validation
    duration              seconds
    corpus                short corpus name
    weight                sampling weight (repeats) for the trainer

The validation design is the part that is easy to get wrong. The development
set is two conversations and four speakers, and the test set shares *no*
speaker or conversation with it. A random utterance-level split would let the
model memorise a speaker's idiolect and would report a WER that the leaderboard
will not reproduce. Splits are therefore grouped by speaker, and the notebook
reports both the grouped-CV WER and the single held-out-conversation WER.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .prompts import STYLE_JEMBER, STYLE_READ, STYLE_VOICE_NOTE
from .scoring import to_scoring_space

MANIFEST_FIELDS = (
    "audio_path",
    "text",
    "raw_text",
    "language",
    "style_tag",
    "speaker",
    "convo_id",
    "duration",
    "corpus",
    "weight",
)


# --------------------------------------------------------------------------
# Audio duration without a heavyweight dependency
# --------------------------------------------------------------------------


def probe_duration(path: str | Path) -> float:
    """Duration in seconds, using mutagen when available and soundfile after.

    ``mutagen`` is in the competition runtime and reads MP3 headers without
    decoding, which matters when probing thousands of clips.
    """
    path = Path(path)
    try:
        from mutagen import File as MutagenFile

        info = MutagenFile(str(path))
        if info is not None and getattr(info, "info", None) is not None:
            return float(info.info.length)
    except Exception:
        pass
    try:
        import soundfile as sf

        with sf.SoundFile(str(path)) as f:
            return float(len(f) / f.samplerate)
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------
# Development set (also the shape of the test set)
# --------------------------------------------------------------------------


def load_dotenv(path: str | Path | None = None) -> Path | None:
    """Fill ``os.environ`` from a ``.env`` file without overwriting existing vars.

    Already-exported values win, so a RunPod secret or a shell ``export`` still
    overrides a file. Missing files are a no-op. Returns the path that was read,
    or ``None``.
    """
    path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
    return path


def resolve_dev_dir(explicit: str | Path | None = None) -> Path:
    """Locate the development set.

    Order: an explicit path, then ``LIT_DEV_DIR``, then the canonical
    ``<repo>/data/indonesian_javanese/development`` directory.
    """
    if explicit is not None:
        return Path(explicit)
    load_dotenv()
    env = os.environ.get("LIT_DEV_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "indonesian_javanese" / "development"


def dev_dir_ready(dev_dir: str | Path | None = None, metadata_name: str = "metadata.tsv") -> bool:
    """True when ``metadata.tsv`` is present. Clips can be probed later."""
    return (resolve_dev_dir(dev_dir) / metadata_name).is_file()


def load_dev_manifest(
    dev_dir: str | Path | None = None,
    metadata_name: str = "metadata.tsv",
    clips_dirname: str = "clips",
    with_durations: bool = True,
) -> list[dict]:
    """Read the Indonesian–Javanese development directory into manifest rows.

    Targets are converted to scoring space here and nowhere else, so a single
    flag flip in :func:`build_training_manifest` switches the whole pipeline
    between raw-text and scoring-space supervision.
    """
    dev_dir = resolve_dev_dir(dev_dir)
    metadata = dev_dir / metadata_name
    if not metadata.is_file():
        raise FileNotFoundError(
            f"development set not found at {metadata}. "
            "The folder must contain metadata.tsv and clips/. "
            "Set LIT_DEV_DIR, or place the data at "
            "data/indonesian_javanese/development/."
        )
    clips = dev_dir / clips_dirname
    rows: list[dict] = []
    with open(metadata, encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            audio = clips / rec["audio_filename"]
            rows.append(
                {
                    "audio_filename": rec["audio_filename"],
                    "audio_path": str(audio),
                    "text": to_scoring_space(rec["transcript"]),
                    "raw_text": rec["transcript"],
                    "language": rec.get("language", "javind"),
                    "style_tag": STYLE_VOICE_NOTE,
                    "speaker": str(rec.get("speaker", "")),
                    "convo_id": str(rec.get("convo_id", "")),
                    "duration": probe_duration(audio) if with_durations else float("nan"),
                    "corpus": "dev",
                    "weight": 1.0,
                }
            )
    return rows


def load_test_manifest(data_dir: str | Path) -> list[dict]:
    """Read the container's ``data/`` directory.

    Prefers ``test_metadata.csv`` (it carries ``language`` and
    ``file_duration_seconds``, both of which the decoder uses) and falls back to
    ``submission_format.csv``. The returned order is the order the submission
    must be written in.
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "test_metadata.csv"
    fmt_path = data_dir / "submission_format.csv"
    clips = data_dir / "clips"

    by_name: dict[str, dict] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8", newline="") as f:
            for rec in csv.DictReader(f):
                name = rec["audio_filename"]
                duration = rec.get("file_duration_seconds") or ""
                try:
                    duration = float(duration)
                except (TypeError, ValueError):
                    duration = float("nan")
                by_name[name] = {
                    "audio_filename": name,
                    "audio_path": str(clips / name),
                    "language": (rec.get("language") or "javind").strip() or "javind",
                    "duration": duration,
                }

    order: list[str] = []
    if fmt_path.exists():
        with open(fmt_path, encoding="utf-8", newline="") as f:
            order = [rec["audio_filename"] for rec in csv.DictReader(f)]
    elif by_name:
        order = list(by_name)
    else:
        raise FileNotFoundError(
            f"neither test_metadata.csv nor submission_format.csv found in {data_dir}"
        )

    rows: list[dict] = []
    for name in order:
        row = by_name.get(name)
        if row is None:
            row = {
                "audio_filename": name,
                "audio_path": str(clips / name),
                "language": "javind",
                "duration": float("nan"),
            }
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Jember Javanese Spontaneous Speech Corpus
# --------------------------------------------------------------------------

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")


def parse_timestamp(value: str) -> float:
    """Parse ``H:MM:SS`` / ``MM:SS`` / ``SS`` into seconds."""
    s = str(value).strip()
    if not s:
        return float("nan")
    m = _TIME_RE.match(s)
    if m:
        hours = float(m.group(1) or 0)
        return hours * 3600 + float(m.group(2)) * 60 + float(m.group(3))
    try:
        return float(s)
    except ValueError:
        return float("nan")


@dataclass
class SegmentationConfig:
    """How to cut session recordings into training clips.

    The Jember timestamps are given at one-second resolution, which clips word
    onsets and offsets. ``pad_s`` restores the margin and ``snap_window_s`` then
    moves each boundary to the quietest point nearby, so segments start and end
    in silence rather than mid-phoneme. Both are cheap and remove a systematic
    source of truncated targets.
    """

    pad_s: float = 0.35
    snap_window_s: float = 0.45
    min_duration_s: float = 1.0
    max_duration_s: float = 40.0
    #: Reject segments whose speaking rate is implausible: a mismatch means the
    #: timestamps and the text disagree, and training on those teaches the model
    #: to hallucinate or to truncate.
    min_words_per_second: float = 0.6
    max_words_per_second: float = 5.0
    target_sample_rate: int = 16000


def _snap_to_quiet(wav, sr: int, sample: int, window: int) -> int:
    """Move ``sample`` to the local energy minimum within ``±window``."""
    import numpy as np

    lo = max(0, sample - window)
    hi = min(len(wav), sample + window)
    if hi - lo < 32:
        return int(min(max(sample, 0), len(wav)))
    seg = np.abs(wav[lo:hi])
    win = max(8, int(0.01 * sr))
    if len(seg) <= win:
        return int(lo + int(np.argmin(seg)))
    energy = np.convolve(seg, np.ones(win, dtype=seg.dtype) / win, mode="valid")
    return int(lo + int(np.argmin(energy)) + win // 2)


def segment_session_corpus(
    audio_dir: str | Path,
    tsv_path: str | Path,
    out_dir: str | Path,
    config: SegmentationConfig | None = None,
    style_tag: str = STYLE_JEMBER,
    corpus: str = "jember",
    language: str = "jav",
    audio_column: str = "Audio file name",
    start_column: str = "start",
    end_column: str = "end",
    text_column: str = "text",
    audio_suffixes: Sequence[str] = (".mp3", ".wav", ".m4a", ".flac", ".ogg"),
    limit_sessions: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Cut session recordings into utterance clips and return manifest rows.

    Writes 16 kHz mono WAV files to ``out_dir`` (the trainer reads WAV; the
    competition clips are MP3 but the model resamples either way). Column names
    are parameters because the MDC corpora do not agree on them.
    """
    import numpy as np
    import soundfile as sf

    config = config or SegmentationConfig()
    audio_dir, out_dir = Path(audio_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(tsv_path, encoding="utf-8", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
        records = list(csv.DictReader(f, delimiter=delimiter))
    if not records:
        return []

    header = list(records[0])
    resolved = _resolve_columns(
        header,
        {
            "audio": audio_column,
            "start": start_column,
            "end": end_column,
            "text": text_column,
        },
    )

    by_session: dict[str, list[dict]] = {}
    for rec in records:
        by_session.setdefault(str(rec[resolved["audio"]]).strip(), []).append(rec)

    index = _index_audio_files(audio_dir, audio_suffixes)
    sessions = sorted(by_session)
    if limit_sessions:
        sessions = sessions[:limit_sessions]

    rows: list[dict] = []
    skipped = {"no_audio": 0, "bad_time": 0, "too_short": 0, "too_long": 0, "bad_rate": 0}

    for session in sessions:
        path = index.get(session) or index.get(Path(session).stem)
        if path is None:
            skipped["no_audio"] += len(by_session[session])
            continue
        try:
            wav, sr = _load_mono(path, config.target_sample_rate)
        except Exception:
            skipped["no_audio"] += len(by_session[session])
            continue

        pad = int(config.pad_s * sr)
        snap = int(config.snap_window_s * sr)
        for i, rec in enumerate(by_session[session]):
            text_raw = (rec.get(resolved["text"]) or "").strip()
            start = parse_timestamp(rec.get(resolved["start"], ""))
            end = parse_timestamp(rec.get(resolved["end"], ""))
            if not text_raw or math.isnan(start) or math.isnan(end) or end <= start:
                skipped["bad_time"] += 1
                continue

            a = _snap_to_quiet(wav, sr, max(0, int(start * sr) - pad), snap)
            b = _snap_to_quiet(wav, sr, min(len(wav), int(end * sr) + pad), snap)
            if b <= a:
                skipped["bad_time"] += 1
                continue
            clip = wav[a:b]
            duration = len(clip) / sr
            if duration < config.min_duration_s:
                skipped["too_short"] += 1
                continue
            if duration > config.max_duration_s:
                skipped["too_long"] += 1
                continue

            text = to_scoring_space(text_raw)
            rate = len(text.split()) / duration if duration > 0 else 0.0
            if not config.min_words_per_second <= rate <= config.max_words_per_second:
                skipped["bad_rate"] += 1
                continue

            peak = float(np.max(np.abs(clip))) if clip.size else 0.0
            if peak > 0:
                clip = (clip / peak * 0.95).astype("float32")
            out_path = out_dir / f"{corpus}_{Path(str(session)).stem}_{i:05d}.wav"
            sf.write(out_path, clip, sr)
            rows.append(
                {
                    "audio_filename": out_path.name,
                    "audio_path": str(out_path),
                    "text": text,
                    "raw_text": text_raw,
                    "language": language,
                    "style_tag": style_tag,
                    "speaker": f"{corpus}_{Path(str(session)).stem}",
                    "convo_id": f"{corpus}_{Path(str(session)).stem}",
                    "duration": duration,
                    "corpus": corpus,
                    "weight": 1.0,
                }
            )

    if verbose:
        total = sum(len(v) for v in by_session.values())
        hours = sum(r["duration"] for r in rows) / 3600
        print(
            f"[{corpus}] kept {len(rows)}/{total} utterances ({hours:.2f} h); "
            f"skipped {skipped}"
        )
    return rows


def _resolve_columns(header: Sequence[str], wanted: dict[str, str]) -> dict[str, str]:
    """Match requested column names case/space-insensitively, with fallbacks."""
    lookup = {re.sub(r"[^a-z0-9]", "", h.lower()): h for h in header}
    aliases = {
        "audio": ["audiofilename", "audio", "filename", "file", "session", "audiofile"],
        "start": ["start", "starttime", "begin", "startsec"],
        "end": ["end", "endtime", "stop", "endsec"],
        "text": ["text", "transcript", "transcription", "sentence"],
    }
    resolved: dict[str, str] = {}
    for key, requested in wanted.items():
        norm = re.sub(r"[^a-z0-9]", "", requested.lower())
        if norm in lookup:
            resolved[key] = lookup[norm]
            continue
        for alias in aliases.get(key, []):
            if alias in lookup:
                resolved[key] = lookup[alias]
                break
        else:
            raise KeyError(
                f"could not find a '{key}' column (asked for {requested!r}) in {list(header)}"
            )
    return resolved


def _index_audio_files(audio_dir: Path, suffixes: Sequence[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(audio_dir.rglob("*")):
        if path.suffix.lower() in suffixes:
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
    return index


def _load_mono(path: Path, target_sr: int):
    import librosa

    wav, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return wav.astype("float32"), target_sr


def load_flat_corpus(
    root: str | Path,
    metadata_name: str = "metadata.tsv",
    clips_dirname: str = "clips",
    style_tag: str = STYLE_READ,
    corpus: str = "mdc",
    language: str = "jav",
    text_column: str = "transcript",
    audio_column: str = "audio_filename",
    speaker_column: str | None = "speaker",
    weight: float = 1.0,
    max_items: int | None = None,
) -> list[dict]:
    """Ingest an MDC corpus that is already cut into clips."""
    root = Path(root)
    clips = root / clips_dirname
    path = root / metadata_name
    with open(path, encoding="utf-8", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
        records = list(csv.DictReader(f, delimiter=delimiter))
    if not records:
        return []
    resolved = _resolve_columns(list(records[0]), {"audio": audio_column, "text": text_column})
    rows: list[dict] = []
    for i, rec in enumerate(records):
        if max_items and len(rows) >= max_items:
            break
        text_raw = (rec.get(resolved["text"]) or "").strip()
        if not text_raw:
            continue
        name = str(rec[resolved["audio"]]).strip()
        audio = clips / name if (clips / name).exists() else root / name
        if not audio.exists():
            continue
        speaker = str(rec.get(speaker_column, "")) if speaker_column else ""
        rows.append(
            {
                "audio_filename": name,
                "audio_path": str(audio),
                "text": to_scoring_space(text_raw),
                "raw_text": text_raw,
                "language": language,
                "style_tag": style_tag,
                "speaker": speaker or f"{corpus}_{i}",
                "convo_id": f"{corpus}",
                "duration": probe_duration(audio),
                "corpus": corpus,
                "weight": weight,
            }
        )
    return rows


def extract_archives(root: str | Path) -> Path:
    """Unpack MDC SDK downloads in place and return the directory to read.

    ``datacollective.download_dataset`` returns the path of an archive rather
    than an extracted tree, and the download scripts move that path to the
    track's data folder. Handle ``root`` itself being the archive and ``root``
    holding archives, detected by content because MDC files may carry no
    extension, including archives nested inside archives. Each unpacked archive
    is recorded in ``.extracted`` so repeat calls do no work.
    """
    import shutil
    import tarfile
    import zipfile

    root = Path(root)
    skip = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".cha", ".tsv", ".csv", ".txt", ".json", ".jsonl", ".md", ".xml",
            ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".epub", ".jar", ".pdf"}  # zip-based documents, not archives

    def is_archive(path: Path) -> bool:
        if not path.is_file() or path.suffix.lower() in skip:
            return False
        try:
            return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)
        except OSError:
            return False

    def unpack(archive: Path, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
            return
        with tarfile.open(archive) as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(dest, filter="data")
            else:
                tf.extractall(dest)

    marker = root / ".extracted"
    if is_archive(root):
        archive = root.with_name(root.name + ".archive")
        shutil.move(str(root), str(archive))
        unpack(archive, root)
        marker.write_text(f"../{archive.name}\n", encoding="utf-8")
    if not root.is_dir():
        return root

    done = set(marker.read_text(encoding="utf-8").split("\n")) if marker.is_file() else set()
    while True:
        pending = [
            p for p in sorted(root.rglob("*"))
            if p.name != marker.name and str(p.relative_to(root)) not in done and is_archive(p)
        ]
        if not pending:
            return root
        for archive in pending:
            print(f"[extract_archives] unpacking {archive.relative_to(root)}")
            unpack(archive, archive.parent)
            done.add(str(archive.relative_to(root)))
        marker.write_text("\n".join(sorted(done)) + "\n", encoding="utf-8")


def describe_tree(root: str | Path, limit: int = 25) -> str:
    """Short listing of files under ``root`` (name and size) for error messages."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    lines = [f"{p.relative_to(root)} ({p.stat().st_size:,} B)" for p in files[:limit]]
    if len(files) > limit:
        lines.append(f"... and {len(files) - limit} more")
    return "\n  ".join(lines) or "(empty)"


def find_corpus_root(path: str | Path) -> Path:
    """Locate ``metadata.tsv`` / ``clips/`` after a tarball extracts a nested folder."""
    path = Path(path)
    if (path / "metadata.tsv").is_file() or (path / "clips").is_dir():
        return path
    for meta in sorted(path.rglob("metadata.tsv")):
        return meta.parent
    for clips in sorted(path.rglob("clips")):
        if clips.is_dir():
            return clips.parent
    return path


def load_extracted_mdc_corpus(
    root: str | Path,
    *,
    corpus: str = "mdc",
    language: str = "nah",
    style_tag: str | None = None,
    weight: float = 1.0,
) -> list[dict]:
    """Load a Mozilla Data Collective clip corpus (Indonesian-Javanese or Nahuatl)."""
    from .prompts import STYLE_VOICE_NOTE_SP_NH, STYLE_READ

    root = find_corpus_root(root)
    tag = style_tag or (
        STYLE_VOICE_NOTE_SP_NH if language in {"nah", "nhe", "nci", "nahuatl", "spnh"} else STYLE_READ
    )
    meta = "metadata.tsv" if (root / "metadata.tsv").is_file() else "metadata.csv"
    clips = "clips" if (root / "clips").is_dir() else "."
    return load_flat_corpus(
        root,
        metadata_name=meta,
        clips_dirname=clips,
        style_tag=tag,
        corpus=corpus,
        language=language,
        weight=weight,
    )


# --------------------------------------------------------------------------
# Training manifest assembly
# --------------------------------------------------------------------------


def build_training_manifest(
    parts: Sequence[Sequence[dict]],
    use_scoring_space_targets: bool = True,
    drop_empty: bool = True,
) -> list[dict]:
    """Concatenate corpora and choose the supervision space.

    ``use_scoring_space_targets=True`` trains the model to emit text that the
    scorer leaves untouched: no commas, no sentence-final periods, mid-sentence
    capitals only where the reference keeps them. That removes an entire class of
    error — matching the reference's *punctuation* so that the reference's
    sentence-initial lowercasing lands on the same words as ours — and shortens
    every target. Set it to ``False`` to reproduce the raw-text ablation.
    """
    rows: list[dict] = []
    for part in parts:
        for row in part:
            row = dict(row)
            if not use_scoring_space_targets:
                row["text"] = " ".join(str(row.get("raw_text") or row["text"]).split())
            if drop_empty and not row["text"].strip():
                continue
            rows.append(row)
    return rows


def write_jsonl(rows: Sequence[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def manifest_summary(rows: Sequence[dict]) -> list[dict]:
    """Per-corpus hours / utterances / words, for the notebook's data table."""
    from collections import defaultdict

    agg: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"utterances": 0, "hours": 0.0, "words": 0}
    )
    for row in rows:
        key = (row.get("corpus", "?"), row.get("style_tag", "?"))
        entry = agg[key]
        entry["utterances"] += 1
        duration = row.get("duration", float("nan"))
        if duration == duration:  # not NaN
            entry["hours"] += duration / 3600
        entry["words"] += len(str(row.get("text", "")).split())
    out = []
    for (corpus, tag), entry in sorted(agg.items()):
        out.append(
            {
                "corpus": corpus,
                "style_tag": tag,
                "utterances": entry["utterances"],
                "hours": round(entry["hours"], 2),
                "words": entry["words"],
            }
        )
    return out


# --------------------------------------------------------------------------
# Grouped cross-validation
# --------------------------------------------------------------------------


def speaker_folds(rows: Sequence[dict], n_folds: int = 4, seed: int = 0) -> list[list[int]]:
    """Assign row indices to folds so that no speaker spans two folds.

    Greedy largest-first packing by total duration keeps folds comparable in
    size, which matters because corpus WER is duration-weighted.
    """
    from collections import defaultdict

    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_speaker[str(row.get("speaker", i))].append(i)

    def load(indices: Sequence[int]) -> float:
        return sum(
            rows[i].get("duration", 0.0) if rows[i].get("duration", 0.0) == rows[i].get("duration", 0.0) else 0.0
            for i in indices
        )

    speakers = sorted(by_speaker, key=lambda s: (-load(by_speaker[s]), s))
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    loads = [0.0] * n_folds
    for speaker in speakers:
        target = min(range(n_folds), key=lambda k: (loads[k], k))
        folds[target].extend(by_speaker[speaker])
        loads[target] += load(by_speaker[speaker])
    return [sorted(f) for f in folds]


def iter_grouped_cv(
    rows: Sequence[dict], n_folds: int = 4, seed: int = 0
) -> Iterator[tuple[int, list[int], list[int]]]:
    """Yield ``(fold_index, train_indices, valid_indices)``."""
    folds = speaker_folds(rows, n_folds=n_folds, seed=seed)
    for k, valid in enumerate(folds):
        valid_set = set(valid)
        train = [i for i in range(len(rows)) if i not in valid_set]
        yield k, train, valid


def holdout_by_conversation(rows: Sequence[dict], held_out: str) -> tuple[list[int], list[int]]:
    """Split by conversation id — the closest local analogue of the real split.

    The test set shares no conversation and no speaker with the development set,
    so holding out a whole conversation is the least optimistic estimate we can
    build from the data we have.
    """
    train, valid = [], []
    for i, row in enumerate(rows):
        (valid if str(row.get("convo_id")) == str(held_out) else train).append(i)
    return train, valid


def subset(rows: Sequence[dict], indices: Iterable[int]) -> list[dict]:
    return [rows[i] for i in indices]
