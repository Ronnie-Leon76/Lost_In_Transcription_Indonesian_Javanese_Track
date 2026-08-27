"""System combination: MBR selection and ROVER voting, in pure Python.

``jiwer`` is *not* in the competition runtime — it is a dependency of the
scoring script, not of the container — so everything here is implemented from
scratch with the standard library. That is also why the edit distance is
hand-rolled instead of imported.

Two combiners, because they fail differently:

*MBR* (minimum Bayes risk) picks the single hypothesis with the lowest expected
WER against the others. It can never invent a word sequence that no system
proposed, so it never breaks agreement between neighbouring words. Its ceiling
is the best hypothesis in the pool.

*ROVER* votes word by word after aligning every hypothesis to a pivot, so it can
beat every individual system, but it can also stitch together an incoherent
sentence, and here it votes on surface forms — ``SMP`` and ``smp`` are different
candidates — which is exactly what the metric rewards.

Which one wins is an empirical question, so the notebook measures both on
held-out speakers and the answer is recorded in the submission config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

NULL = "@"  # ROVER's null-arc symbol; never emitted


# --------------------------------------------------------------------------
# Edit distance and alignment on word sequences
# --------------------------------------------------------------------------


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences."""
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    previous = list(range(m + 1))
    current = [0] * (m + 1)
    for i in range(1, n + 1):
        current[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous, current = current, previous
    return previous[m]


def word_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> list[tuple[str, int, int]]:
    """Levenshtein backtrace as ``(op, ref_index, hyp_index)`` triples.

    ``op`` is one of ``equal``, ``sub``, ``del`` (reference word missing from the
    hypothesis) or ``ins`` (hypothesis word absent from the reference). Indices
    are ``-1`` where they do not apply. Ties prefer substitution over
    insert/delete, which keeps alignments compact and keeps ROVER slots aligned
    with pivot words.
    """
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                ops.append(("equal" if cost == 0 else "sub", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))
            i -= 1
            continue
        ops.append(("ins", -1, j - 1))
        j -= 1
    ops.reverse()
    return ops


# --------------------------------------------------------------------------
# MBR
# --------------------------------------------------------------------------


@dataclass
class CombinationResult:
    text: str
    #: Weighted fraction of systems that agreed with the chosen hypothesis
    #: token-for-token. Used as a cheap confidence signal for the re-decode
    #: guard, not as a probability.
    agreement: float
    #: Index of the source hypothesis for MBR; -1 when ROVER built a new string.
    source: int


def mbr_select(
    hypotheses: Sequence[str],
    weights: Sequence[float] | None = None,
) -> CombinationResult:
    """Pick the hypothesis minimising expected WER against the others."""
    texts = [h if h is not None else "" for h in hypotheses]
    if not texts:
        return CombinationResult("", 0.0, -1)
    if weights is None:
        weights = [1.0] * len(texts)
    if len(weights) != len(texts):
        raise ValueError("weights and hypotheses must have the same length")

    # Collapse duplicates: identical hypotheses are common once post-processing
    # has run, and the risk computation is quadratic.
    unique: dict[str, float] = {}
    first_index: dict[str, int] = {}
    for idx, (text, weight) in enumerate(zip(texts, weights)):
        unique[text] = unique.get(text, 0.0) + weight
        first_index.setdefault(text, idx)
    keys = list(unique)
    tokens = {k: k.split() for k in keys}
    total = sum(unique.values()) or 1.0

    if len(keys) == 1:
        return CombinationResult(keys[0], 1.0, first_index[keys[0]])

    distance: dict[tuple[int, int], float] = {}
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            d = word_error_rate(tokens[keys[b]], tokens[keys[a]])
            e = word_error_rate(tokens[keys[a]], tokens[keys[b]])
            distance[(a, b)] = d
            distance[(b, a)] = e

    best_key, best_risk = keys[0], float("inf")
    for a, key in enumerate(keys):
        risk = sum(
            unique[keys[b]] * (0.0 if a == b else distance[(a, b)]) for b in range(len(keys))
        )
        if risk < best_risk - 1e-12:
            best_key, best_risk = key, risk
    return CombinationResult(
        best_key, unique[best_key] / total, first_index[best_key]
    )


# --------------------------------------------------------------------------
# ROVER
# --------------------------------------------------------------------------


def rover_combine(
    hypotheses: Sequence[str],
    weights: Sequence[float] | None = None,
    pivot: int | None = None,
    null_penalty: float = 1.0,
    insertion_threshold: float = 0.5,
) -> CombinationResult:
    """Word-level voting over a confusion network built around a pivot.

    ``null_penalty`` scales the weight of a "this system had no word here" vote:
    below 1.0 the combiner keeps words that only some systems produced (fewer
    deletions, more insertions), above 1.0 it prunes them.

    ``insertion_threshold`` is the weighted fraction of systems that must agree
    on an inserted word before it is admitted. Insertions are the dangerous
    direction — one system's hallucination becomes everyone's — so the default
    demands a majority.
    """
    texts = [h if h is not None else "" for h in hypotheses]
    if not texts:
        return CombinationResult("", 0.0, -1)
    if weights is None:
        weights = [1.0] * len(texts)
    if len(texts) == 1:
        return CombinationResult(texts[0], 1.0, 0)

    if pivot is None:
        pivot = mbr_select(texts, weights).source
    pivot = max(0, min(pivot, len(texts) - 1))
    pivot_tokens = texts[pivot].split()
    total_weight = sum(weights) or 1.0

    n = len(pivot_tokens)
    slot_votes: list[dict[str, float]] = [dict() for _ in range(n)]
    gap_votes: list[dict[tuple[str, ...], float]] = [dict() for _ in range(n + 1)]

    for idx, (text, weight) in enumerate(zip(texts, weights)):
        hyp_tokens = text.split()
        if idx == pivot:
            for i, word in enumerate(pivot_tokens):
                slot_votes[i][word] = slot_votes[i].get(word, 0.0) + weight
            continue
        ops = align(pivot_tokens, hyp_tokens)
        pending: list[str] = []
        next_slot = 0
        for op, ref_i, hyp_j in ops:
            if op == "ins":
                pending.append(hyp_tokens[hyp_j])
                continue
            if pending:
                key = tuple(pending)
                gap_votes[next_slot][key] = gap_votes[next_slot].get(key, 0.0) + weight
                pending = []
            if op in ("equal", "sub"):
                word = hyp_tokens[hyp_j]
                slot_votes[ref_i][word] = slot_votes[ref_i].get(word, 0.0) + weight
            else:  # del
                slot_votes[ref_i][NULL] = slot_votes[ref_i].get(NULL, 0.0) + weight * null_penalty
            next_slot = ref_i + 1
        if pending:
            key = tuple(pending)
            gap_votes[n][key] = gap_votes[n].get(key, 0.0) + weight

    out: list[str] = []
    agreements: list[float] = []
    for i in range(n + 1):
        best_gap, best_gap_weight = None, 0.0
        for key, w in gap_votes[i].items():
            if w > best_gap_weight:
                best_gap, best_gap_weight = key, w
        if best_gap and best_gap_weight >= insertion_threshold * total_weight:
            out.extend(best_gap)
        if i == n:
            break
        votes = slot_votes[i]
        if not votes:
            continue
        pivot_word = pivot_tokens[i]
        # Ties go to the pivot: it comes from the strongest single system, and
        # switching surface forms on a tie is how style drift creeps in.
        best_word, best_weight = pivot_word, votes.get(pivot_word, 0.0)
        for word, w in votes.items():
            if w > best_weight + 1e-12:
                best_word, best_weight = word, w
        agreements.append(best_weight / total_weight)
        if best_word != NULL:
            out.append(best_word)

    agreement = sum(agreements) / len(agreements) if agreements else 0.0
    return CombinationResult(" ".join(out), agreement, -1)


# --------------------------------------------------------------------------
# Length prior, used as a decoding guard rather than a selection criterion
# --------------------------------------------------------------------------


@dataclass
class LengthPrior:
    """Words-per-second statistics fitted on training transcripts.

    LLM-based ASR fails on long audio by truncating: the decoder emits an EOS
    halfway through a 35-second turn and every remaining word becomes a
    deletion. A single clip losing 30 words costs more than a hundred casing
    errors, so the guard matters more than it looks.
    """

    mean: float = 1.886
    std: float = 0.264
    low_z: float = -3.0
    high_z: float = 3.0

    @classmethod
    def fit(cls, texts: Sequence[str], durations: Sequence[float]) -> "LengthPrior":
        rates = [
            len(str(t).split()) / d
            for t, d in zip(texts, durations)
            if d and d == d and d > 0
        ]
        if not rates:
            return cls()
        mean = sum(rates) / len(rates)
        var = sum((r - mean) ** 2 for r in rates) / max(1, len(rates) - 1)
        return cls(mean=mean, std=var**0.5)

    def z_score(self, text: str, duration_s: float) -> float:
        if not duration_s or duration_s != duration_s or duration_s <= 0 or self.std <= 0:
            return 0.0
        return (len(str(text).split()) / duration_s - self.mean) / self.std

    def is_suspicious(self, text: str, duration_s: float) -> bool:
        z = self.z_score(text, duration_s)
        return z < self.low_z or z > self.high_z
