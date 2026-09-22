"""Track-isolated development manifest builder for Spanish-English."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from lit_jv.data import probe_duration
from lit_jv.prompts import STYLE_VOICE_NOTE_SP_EN
from lit_jv.scoring import to_scoring_space


def load_bangor_miami(root: str | Path) -> list[dict]:
    """Load Bangor Miami through MDC's schema-backed SDK after it is staged."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Bangor Miami is not staged: {root}")
    try:
        from datacollective import load_dataset
    except ImportError as exc:
        raise ImportError("Install datacollective>=0.5.5 to load Bangor Miami") from exc
    frame = load_dataset("cmmfulo4r018bnz07py4q9t09", download_directory=str(root))
    columns = set(frame.columns)
    text_key = next((key for key in ("transcript", "transcription", "text", "sentence") if key in columns), None)
    audio_key = next((key for key in ("audio_path", "audio_file", "file_path", "path", "audio") if key in columns), None)
    if not text_key or not audio_key:
        raise ValueError(f"Bangor SDK schema lacks transcript/audio columns: {sorted(columns)}")
    rows: list[dict] = []
    for index, record in frame.iterrows():
        raw = str(record[text_key]).strip()
        value = record[audio_key]
        if isinstance(value, dict):
            value = value.get("path") or value.get("audio_path")
        audio = Path(str(value))
        if not audio.is_absolute():
            audio = root / audio
        if not raw or not audio.is_file():
            continue
        rows.append({"audio_filename": audio.name, "audio_path": str(audio.resolve()), "text": to_scoring_space(raw),
                     "raw_text": raw, "language": "spen", "style_tag": STYLE_VOICE_NOTE_SP_EN,
                     "speaker": str(record.get("speaker", "")), "convo_id": str(record.get("conversation", record.get("session", index))),
                     "duration": probe_duration(audio), "corpus": "bangor_miami", "weight": 1.0})
    if not rows:
        raise ValueError("Bangor SDK returned no usable audio/transcript rows")
    return rows


def load_competition_dev(root: str | Path) -> list[dict]:
    """Load the official Spanish-English ``metadata.tsv`` and ``clips/`` only."""
    root = Path(root)
    metadata = root / "metadata.tsv"
    clips = root / "clips"
    if not metadata.is_file() or not clips.is_dir():
        raise FileNotFoundError(f"expected metadata.tsv and clips/ under {root}")
    with metadata.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    required = {"audio_filename", "transcript", "convo_id"}
    if not records or not required.issubset(records[0]):
        raise ValueError(f"{metadata} must contain: {', '.join(sorted(required))}")
    rows: list[dict] = []
    for record in records:
        audio = clips / record["audio_filename"].strip()
        raw = record["transcript"].strip()
        if not raw or not audio.is_file():
            continue
        rows.append({
            "audio_filename": audio.name,
            "audio_path": str(audio.resolve()),
            "text": to_scoring_space(raw),
            "raw_text": raw,
            "language": (record.get("language") or "spen").strip() or "spen",
            "style_tag": STYLE_VOICE_NOTE_SP_EN,
            "speaker": str(record.get("speaker") or ""),
            "convo_id": str(record.get("convo_id") or ""),
            "duration": probe_duration(audio),
            "corpus": "spen_dev",
            "weight": 1.0,
        })
    if not rows:
        raise ValueError(f"no usable clips found under {root}")
    return rows


def holdout_speakers(rows: Iterable[dict], count: int = 1, seed: int = 2026) -> tuple[list[dict], list[dict]]:
    """Make the speaker- and conversation-disjoint validation split used for selection."""
    rows = list(rows)
    speakers = sorted({str(row.get("speaker") or "") for row in rows if row.get("speaker")})
    if len(speakers) <= count:
        raise ValueError(f"need more than {count} labelled speakers; found {len(speakers)}")
    rng = random.Random(seed)
    rng.shuffle(speakers)
    held = set(speakers[:count])
    evaluation = [row for row in rows if row.get("speaker") in held]
    held_conversations = {row["convo_id"] for row in evaluation if row.get("convo_id")}
    fitting = [
        row for row in rows
        if row.get("speaker") not in held and row.get("convo_id") not in held_conversations
    ]
    if not fitting or not evaluation:
        raise ValueError("speaker-disjoint split unexpectedly produced an empty partition")
    return fitting, evaluation


def glossary(rows: Iterable[dict], minimum_count: int = 2, limit: int = 128) -> list[str]:
    """Mine only this track's repeated spellings for inference-time hotwords."""
    counts = Counter(token for row in rows for token in row["text"].split() if len(token) > 1)
    return [token for token, frequency in counts.most_common() if frequency >= minimum_count][:limit]
