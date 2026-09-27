"""Track-isolated development manifest builder for Spanish-English."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from lit_jv.data import (
    SegmentationConfig,
    describe_tree,
    extract_archives,
    probe_duration,
    read_jsonl,
    segment_session_corpus,
    write_jsonl,
)
from lit_jv.prompts import STYLE_VOICE_NOTE_SP_EN
from lit_jv.scoring import to_scoring_space


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
# CHAT bullets are ``\x15start_end\x15`` in milliseconds; some exports render
# the NAK byte as a middle dot.
_CHAT_BULLET = re.compile(r"[\x15·](\d+)_(\d+)[\x15·]")


def _clean_chat(text: str) -> str:
    """Strip CHAT markup from a main-tier utterance, keeping spoken words."""
    text = re.sub(r"\[[^\]]*\]", " ", text)             # [: repl] [//] [+ gram] [=! laughs] ...
    text = re.sub(r"&[=+~*][^\s]*", " ", text)           # events, fragments
    text = re.sub(r"&-(\S+)", r"\1", text)               # fillers: &-uh -> uh
    text = re.sub(r"(?<!\S)0\S+", " ", text)             # omitted words
    text = re.sub(r"(?<!\S)(xxx|yyy|www)(?!\S)", " ", text)
    text = re.sub(r"\+[^\s\w]*", " ", text)              # +... +/. +// +< terminators/linkers
    text = re.sub(r"@\S+", "", text)                     # word@s:eng, word@l ...
    text = re.sub(r"[<>()‹›“”„‡↑↓≠∬⌈⌉⌊⌋^]", "", text)
    text = re.sub(r"(?<=\w):(?=\w|\s|$)", "", text)      # lengthening no: -> no
    text = text.replace("_", " ")
    text = re.sub(r"\s+([.,?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _chat_utterances(cha: Path) -> Iterable[dict]:
    """Yield timed main-tier utterances from one CHAT transcript."""
    current: list[str] = []

    def flush():
        if not current:
            return None
        line = " ".join(current)
        match = _CHAT_BULLET.search(line)
        speaker, _, body = line[1:].partition(":")
        if not match:
            return None
        text = _clean_chat(_CHAT_BULLET.sub(" ", body))
        if not re.search(r"\w", text):
            return None
        return {"start": int(match.group(1)) / 1000, "end": int(match.group(2)) / 1000,
                "speaker": speaker.strip(), "text": text}

    for line in cha.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("\t") and current:
            current.append(line.strip())
            continue
        utterance = flush()
        if utterance:
            yield utterance
        current = [line.strip()] if line.startswith("*") else []
    utterance = flush()
    if utterance:
        yield utterance


def load_bangor_miami(root: str | Path, config: SegmentationConfig | None = None) -> list[dict]:
    """Load the staged Bangor Miami corpus from disk (no SDK call).

    ``download_bangor_miami.py`` stages the MDC release. It ships whole session
    recordings paired with CHAT ``.cha`` transcripts, so the timed utterances
    are cut into clips once and cached as ``_segments/manifest.jsonl``.
    """
    root = extract_archives(root)
    if not root.exists():
        raise FileNotFoundError(f"Bangor Miami is not staged: {root}; run spanish_english/download_bangor_miami.py --download")
    cache_dir = root / "_segments"
    manifest = cache_dir / "manifest.jsonl"
    if manifest.is_file():
        rows = [row for row in read_jsonl(manifest) if Path(row["audio_path"]).is_file()]
        if rows:
            return rows

    chats = sorted(p for p in root.rglob("*.cha") if cache_dir not in p.parents)
    if not chats:
        raise FileNotFoundError(
            f"no CHAT .cha transcripts under {root}. Files present:\n  {describe_tree(root)}\n"
            "If this is empty or only .extracted, delete the folder and re-run the download cell."
        )
    audio = {p.stem.lower(): p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES and cache_dir not in p.parents}

    cache_dir.mkdir(parents=True, exist_ok=True)
    table = cache_dir / "utterances.tsv"
    missing = []
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio", "start", "end", "text", "speaker"], delimiter="\t")
        writer.writeheader()
        for cha in chats:
            recording = audio.get(cha.stem.lower())
            if recording is None:
                missing.append(cha.name)
                continue
            for utterance in _chat_utterances(cha):
                writer.writerow({"audio": recording.name, **utterance})
    if missing:
        print(f"[bangor_miami] no audio for {len(missing)} transcripts, e.g. {missing[:3]}")

    rows = segment_session_corpus(
        root, table, cache_dir / "clips", config=config, style_tag=STYLE_VOICE_NOTE_SP_EN,
        corpus="bangor_miami", language="spen", audio_column="audio", text_column="text",
        audio_suffixes=tuple(AUDIO_SUFFIXES),
    )
    if not rows:
        raise ValueError(f"Bangor Miami produced no usable clips from {len(chats)} transcripts under {root}")
    write_jsonl(rows, manifest)
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
    if not fitting:
        # The released Spanish-English dev set is one conversation between two
        # speakers, so conversation-disjointness would leave nothing to fit on.
        # Keep the split speaker-disjoint and accept the shared conversation.
        print(f"[holdout_speakers] every speaker shares a held-out conversation; "
              f"falling back to a speaker-only split (held speakers: {sorted(held)})")
        fitting = [row for row in rows if row.get("speaker") not in held]
    if not fitting or not evaluation:
        raise ValueError("speaker-disjoint split unexpectedly produced an empty partition")
    return fitting, evaluation


def glossary(rows: Iterable[dict], minimum_count: int = 2, limit: int = 128) -> list[str]:
    """Mine only this track's repeated spellings for inference-time hotwords."""
    counts = Counter(token for row in rows for token in row["text"].split() if len(token) > 1)
    return [token for token, frequency in counts.most_common() if frequency >= minimum_count][:limit]
