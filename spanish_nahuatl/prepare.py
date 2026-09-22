"""Data preparation for Lost in Transcription: Spanish-Nahuatl.

The competition development set and the Tetelancingo corpus have different
schemas.  This module turns them into the common manifest consumed by the Qwen
LoRA trainer without ever mixing Indonesian-Javanese material into this track.

Use Tetelancingo's released ``sentence`` field, not its Spanish translation or
``original_sentence``.  ``sentence`` is the normalized Nahuatl transcription
and is the closest available supervision for the competition orthography.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from lit_jv.data import probe_duration
from lit_jv.prompts import STYLE_VOICE_NOTE_SP_NH
from lit_jv.scoring import to_scoring_space


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        probe = handle.read(8192)
        handle.seek(0)
        delimiter = "\t" if probe.count("\t") >= probe.count(",") else ","
        return list(csv.DictReader(handle, delimiter=delimiter))


def _find_table(root: Path, required: set[str]) -> Path:
    for path in sorted(root.rglob("*.tsv")) + sorted(root.rglob("*.csv")):
        rows = _read_table(path)
        if rows and required.issubset(rows[0]):
            return path
    wanted = ", ".join(sorted(required))
    raise FileNotFoundError(f"could not find a table with columns: {wanted} under {root}")


def load_competition_dev(root: str | Path) -> list[dict]:
    """Load ``nahuatl_dev/metadata.tsv`` and its MP3 clips."""
    root = Path(root)
    table = _find_table(root, {"audio_filename", "transcript", "convo_id"})
    clips = table.parent / "clips"
    rows: list[dict] = []
    for rec in _read_table(table):
        name = rec["audio_filename"].strip()
        audio = clips / name
        raw = rec["transcript"].strip()
        if not raw or not audio.is_file():
            continue
        rows.append(
            {
                "audio_filename": name,
                "audio_path": str(audio.resolve()),
                "text": to_scoring_space(raw),
                "raw_text": raw,
                "language": (rec.get("language") or "spnh").strip() or "spnh",
                "style_tag": STYLE_VOICE_NOTE_SP_NH,
                "speaker": str(rec.get("speaker") or ""),
                "convo_id": str(rec.get("convo_id") or ""),
                "duration": probe_duration(audio),
                "corpus": "spnh_dev",
                "weight": 1.0,
            }
        )
    if not rows:
        raise ValueError(f"no usable competition clips found below {root}")
    return rows


def load_tetelancingo(root: str | Path, split: str | None = "train") -> list[dict]:
    """Load released Tetelancingo clips using normalized Nahuatl ``sentence``.

    ``split='train'`` is the default for fitting.  Keep the corpus's own test
    split for an external sanity check; do not tune repeatedly on it.  Pass
    ``None`` only after model selection when making the final training manifest.
    """
    root = Path(root)
    table = _find_table(root, {"audio_file", "sentence"})
    # Index once. Calling ``rglob`` for every row turns a 2,681-clip dataset
    # into millions of directory visits on a network-mounted RunPod volume.
    audio_index = {
        path.name: path
        for path in table.parent.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}
    }
    rows: list[dict] = []
    for index, rec in enumerate(_read_table(table)):
        rec_split = (rec.get("split") or "").strip().lower()
        if split is not None and rec_split != split:
            continue
        raw = (rec.get("sentence") or "").strip()
        if not raw:
            continue
        name = (rec.get("audio_file") or "").strip()
        audio = audio_index.get(Path(name).name)
        if audio is None:
            continue
        rows.append(
            {
                "audio_filename": audio.name,
                "audio_path": str(audio.resolve()),
                "text": to_scoring_space(raw),
                "raw_text": raw,
                "language": "nah",
                "style_tag": "tetelancingo-read-nhi",
                "speaker": str(rec.get("speaker") or f"tet_{index}"),
                "convo_id": f"tetelancingo_{rec.get('speaker') or index}",
                "duration": probe_duration(audio),
                "corpus": "tetelancingo",
                "weight": 1.0,
            }
        )
    if not rows:
        raise ValueError(f"no usable Tetelancingo {split!r} rows found below {root}")
    return rows


def holdout_conversations(rows: Iterable[dict], count: int = 3, seed: int = 2026) -> tuple[list[dict], list[dict]]:
    """Create a conversation-disjoint dev split for model selection.

    The contest test set has no overlapping conversation or speaker, so random
    clip splits are invalidly optimistic.  Hold out whole conversations and
    retain their speakers exclusively in evaluation whenever the metadata
    permits it.
    """
    rows = list(rows)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = row.get("convo_id") or row["audio_filename"]
        groups.setdefault(key, []).append(row)
    if len(groups) <= count:
        raise ValueError(f"need more than {count} conversations; found {len(groups)}")
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    held = set(ordered[:count])
    eval_rows = [row for key, group in groups.items() if key in held for row in group]
    eval_speakers = {row.get("speaker") for row in eval_rows if row.get("speaker")}
    train_rows = [
        row for key, group in groups.items() if key not in held for row in group
        if not row.get("speaker") or row.get("speaker") not in eval_speakers
    ]
    if not train_rows or not eval_rows:
        raise ValueError("grouped split unexpectedly produced an empty partition")
    return train_rows, eval_rows


def holdout_speakers(rows: Iterable[dict], count: int = 1, seed: int = 2026) -> tuple[list[dict], list[dict]]:
    """Return fitting/evaluation rows with no shared speaker.

    This is the primary model-selection split because the official test set has
    no overlapping speaker with development.  Whole conversations associated
    with the held-out speaker go to evaluation as well.
    """
    rows = list(rows)
    speakers = sorted({str(row.get("speaker") or "") for row in rows if row.get("speaker")})
    if len(speakers) <= count:
        raise ValueError(f"need more than {count} labelled speakers; found {len(speakers)}")
    rng = random.Random(seed)
    rng.shuffle(speakers)
    held = set(speakers[:count])
    eval_rows = [row for row in rows if str(row.get("speaker") or "") in held]
    held_convos = {row.get("convo_id") for row in eval_rows if row.get("convo_id")}
    train_rows = [
        row for row in rows
        if str(row.get("speaker") or "") not in held and row.get("convo_id") not in held_convos
    ]
    if not train_rows or not eval_rows:
        raise ValueError("speaker-disjoint split unexpectedly produced an empty partition")
    return train_rows, eval_rows


def glossary(rows: Iterable[dict], minimum_count: int = 2, limit: int = 128) -> list[str]:
    """Stable track-specific hotword list, built only from fitting rows."""
    counts = Counter(
        token
        for row in rows
        for token in str(row["text"]).split()
        if len(token) > 1
    )
    return [token for token, n in counts.most_common() if n >= minimum_count][:limit]


def orthography_audit(rows: Iterable[dict]) -> dict[str, int]:
    """Count spellings that require a human check before training.

    Do *not* blindly rewrite ``c``, ``qu``, ``w`` or ``z``: a bilingual
    transcript can contain standard-Spanish words and adapted Spanish loans.
    The competition's whole-token WER makes a broad replacement worse than the
    occasional Nahuatl mismatch.  Tetelancingo's normalized ``sentence`` is
    expected to be close to the requested convention; this audit makes any
    departure visible before it enters the training targets.
    """
    counts = Counter()
    for row in rows:
        for token in str(row.get("text") or "").split():
            lowered = token.lower()
            if "w" in lowered:
                counts["contains_w"] += 1
            if "qu" in lowered:
                counts["contains_qu"] += 1
            if "c" in lowered:
                counts["contains_c"] += 1
            if "z" in lowered:
                counts["contains_z"] += 1
    return dict(counts)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
