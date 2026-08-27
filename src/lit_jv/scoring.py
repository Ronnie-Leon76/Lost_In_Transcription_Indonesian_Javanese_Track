"""Byte-exact replication of the competition scorer plus analysis helpers.

The normalisation code in :func:`normalize_text` is a verbatim port of
``score.py`` from ``drivendataorg/lost-in-transcription-runtime``. It must not be
"improved": every deviation makes local WER diverge from the leaderboard.

Two facts about the scorer drive the whole solution and are asserted in
:func:`self_check`:

1. Only three non-alphanumeric characters survive normalisation: ``.`` (inside
   ``...``), ``-`` and ``'``. Everything else (commas, periods, ``!?``, quotes,
   em dashes, brackets) is deleted.
2. Casing is compared *as written* apart from sentence-initial letters, and
   sentence-initiality is decided from the *raw* text of each side
   independently. A hypothesis that omits the reference's sentence punctuation
   therefore keeps capitals the reference lost, and vice versa.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# Verbatim port of score.py
# --------------------------------------------------------------------------

_BRACKETED = re.compile(r"\[[^\]]+\]")
_UNINTELLIGIBLE_PAREN = re.compile(r"\(\?+\)")
_WORD_PAREN = re.compile(r"\(([^()]*)\)")
_PUNCTUATION_OTHER = re.compile('[¿¡";:]+')
_COMMA = re.compile(",+")
_SENTENCE_INITIAL = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
_SENTENCE_END = re.compile("[!?]+")
_MULTISPACE = re.compile("  +")


def _lowercase_sentence_initial(match: re.Match) -> str:
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def normalize_text(text: str) -> str:
    """Normalise a transcript exactly as the competition scorer does."""
    text = text.replace("~", "")
    text = re.sub(_BRACKETED, " ", text)
    text = re.sub(_UNINTELLIGIBLE_PAREN, " ", text)
    text = re.sub(_WORD_PAREN, r"\1", text)
    text = text.replace("#x27;", "'")
    text = re.sub(_PUNCTUATION_OTHER, " ", text)
    text = re.sub(_SENTENCE_INITIAL, _lowercase_sentence_initial, text)
    text = text.replace("—", ", ")
    text = re.sub(_COMMA, " ", text)
    text = re.sub(_SENTENCE_END, " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    text = re.sub(_MULTISPACE, " ", text)
    return text


# --------------------------------------------------------------------------
# Scoring space
# --------------------------------------------------------------------------

MAX_NORMALIZE_ITERATIONS = 8


def to_scoring_space(text: str, max_iterations: int = MAX_NORMALIZE_ITERATIONS) -> str:
    """Return the fixed point of :func:`normalize_text`, whitespace-collapsed.

    ``normalize_text`` is *almost* idempotent. It is not, because an ellipsis
    that survives normalisation is still a sentence delimiter for the
    ``_SENTENCE_INITIAL`` rule, so ``"jenengé... Sangiran"`` loses its capital
    on a second pass. Training targets live at the fixed point, which makes the
    model's own output a scorer no-op and removes any dependency on our
    hypothesis reproducing the reference's punctuation.
    """
    if text is None:
        return ""
    current = str(text)
    for _ in range(max_iterations):
        nxt = normalize_text(current)
        if nxt == current:
            break
        current = nxt
    return " ".join(current.split())


def is_normalization_fixed_point(text: str) -> bool:
    """True when ``text`` is unchanged by another round of normalisation."""
    return " ".join(normalize_text(text).split()) == " ".join(str(text).split())


# --------------------------------------------------------------------------
# WER
# --------------------------------------------------------------------------


def _require_jiwer():
    try:
        import jiwer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "jiwer is required for scoring. Install it with `pip install jiwer`."
        ) from exc
    return jiwer


def corpus_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level WER computed the way the platform computes it.

    The scorer normalises both sides then calls ``jiwer.wer`` on the two lists,
    which pools errors over the corpus: ``(S + D + I) / N_reference_words``.
    Long utterances therefore matter more than short ones.
    """
    jiwer = _require_jiwer()
    if len(references) != len(hypotheses):
        raise ValueError(f"length mismatch: {len(references)} refs vs {len(hypotheses)} hyps")
    refs = [normalize_text(t if t is not None else "") for t in references]
    hyps = [normalize_text(t if t is not None else "") for t in hypotheses]
    return float(jiwer.wer(refs, hyps))


@dataclass
class WerReport:
    """Corpus WER together with the counts that produced it."""

    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_words: int
    per_utterance: list[float]

    def as_row(self, name: str) -> dict:
        return {
            "system": name,
            "WER": round(self.wer, 5),
            "sub": self.substitutions,
            "del": self.deletions,
            "ins": self.insertions,
            "ref_words": self.reference_words,
        }


def wer_report(references: Sequence[str], hypotheses: Sequence[str]) -> WerReport:
    """Corpus WER plus the S/D/I breakdown and per-utterance WERs."""
    jiwer = _require_jiwer()
    refs = [normalize_text(t if t is not None else "") for t in references]
    hyps = [normalize_text(t if t is not None else "") for t in hypotheses]
    out = jiwer.process_words(refs, hyps)
    n_ref = out.substitutions + out.deletions + out.hits
    per_utt = []
    for ref, hyp in zip(refs, hyps):
        n = len(ref.split())
        per_utt.append(float(jiwer.wer(ref, hyp)) if n else 0.0)
    return WerReport(
        wer=float(out.wer),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
        hits=int(out.hits),
        reference_words=int(n_ref),
        per_utterance=per_utt,
    )


def token_error_pairs(references: Sequence[str], hypotheses: Sequence[str]) -> list[tuple[str, str]]:
    """Return ``(reference_token, hypothesis_token)`` pairs for every error.

    Deletions are reported as ``(ref, "")`` and insertions as ``("", hyp)``.
    This is the raw material for the error taxonomy in the notebook: it is how
    we discovered that casing, reduplication hyphens and disfluency ellipses
    account for roughly nine WER points on their own.
    """
    jiwer = _require_jiwer()
    refs = [normalize_text(t if t is not None else "") for t in references]
    hyps = [normalize_text(t if t is not None else "") for t in hypotheses]
    out = jiwer.process_words(refs, hyps)
    pairs: list[tuple[str, str]] = []
    for utt_idx, chunks in enumerate(out.alignments):
        ref_words = refs[utt_idx].split()
        hyp_words = hyps[utt_idx].split()
        for chunk in chunks:
            if chunk.type == "equal":
                continue
            r = ref_words[chunk.ref_start_idx : chunk.ref_end_idx]
            h = hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
            if chunk.type == "substitute":
                pairs.extend(zip(r, h))
            elif chunk.type == "delete":
                pairs.extend((w, "") for w in r)
            elif chunk.type == "insert":
                pairs.extend(("", w) for w in h)
    return pairs


# --------------------------------------------------------------------------
# Style ablations: how much WER each orthographic dimension is worth
# --------------------------------------------------------------------------


def style_loss_ablation(references: Iterable[str]) -> dict[str, float]:
    """WER incurred by a system that recognises every word but loses style.

    Each entry degrades the *reference* in one orthographic dimension and scores
    it against the untouched reference, so the number is the WER a
    style-agnostic system pays for that dimension alone.
    """
    refs = [to_scoring_space(t) for t in references]
    degraded = {
        "lowercase everything": [t.lower() for t in refs],
        "split reduplication hyphens": [t.replace("-", " ") for t in refs],
        "drop disfluency ellipses": [t.replace("...", "") for t in refs],
        "strip é diacritic": [t.replace("é", "e").replace("è", "e") for t in refs],
        "drop apostrophes": [t.replace("'", "") for t in refs],
    }
    degraded["all of the above"] = [
        t.lower().replace("-", " ").replace("...", "").replace("é", "e").replace("'", "")
        for t in refs
    ]
    return {
        name: corpus_wer(refs, [" ".join(d.split()) for d in degraded_texts])
        for name, degraded_texts in degraded.items()
    }


# --------------------------------------------------------------------------
# Self check
# --------------------------------------------------------------------------


def self_check() -> None:
    """Assert the scorer behaves the way the whole solution assumes it does.

    Run this before trusting any local WER number.
    """
    # 1. Punctuation that disappears entirely.
    assert normalize_text('Halo, "dunia"; ya!') == "halo dunia ya "
    # 2. Sentence-initial lowercasing, including after a period.
    assert to_scoring_space("Nah, itu di Dieng. Dieng ki adoh.") == "nah itu di Dieng dieng ki adoh"
    # 3. Acronyms keep their capitals (next letter is also uppercase).
    assert to_scoring_space("Aku SMP di Solo.") == "aku SMP di Solo"
    # 4. Attached ellipses survive; free-standing ones are deleted.
    assert to_scoring_space("aku e... lupa") == "aku e... lupa"
    assert to_scoring_space("aku ... lupa") == "aku lupa"
    # 5. Reduplication hyphens and apostrophes survive.
    assert to_scoring_space("waktu-waktu 'kan ho'oh") == "waktu-waktu 'kan ho'oh"
    # 6. The fixed point really is a fixed point.
    tricky = "jenengé... Sangiran. Apa ya... SMP-ku."
    assert is_normalization_fixed_point(to_scoring_space(tricky))
    # 7. A hypothesis already in scoring space scores 0 against its reference.
    assert corpus_wer([tricky], [to_scoring_space(tricky)]) == 0.0
