"""Orthography repair, gated by measured WER on held-out data.

Only three non-alphanumeric characters survive the scorer: the ``.`` inside an
ellipsis, the reduplication ``-``, and ``'``. Degrading the references along
those three axes plus casing costs about nine WER points, so they are worth
explicit modelling on top of whatever the acoustic model learns.

Hand-written rules are how competitions get lost, so none of them is trusted by
default. :class:`RulePipeline` measures each rule's WER delta on held-out
utterances, keeps it only if it helps, and records the numbers for the notebook's
ablation table. A rule that helps on one fold and hurts on another gets dropped.

The lexicon is mined from training transcripts only. It ships with the
submission as a small JSON file; nothing here reads the test set.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .scoring import corpus_wer, to_scoring_space

_TOKEN_RE = re.compile(r"[^\W\d_][\w'\-é]*", re.UNICODE)


def strip_accents(token: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", token) if unicodedata.category(c) != "Mn"
    )


# --------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------


@dataclass
class StyleLexicon:
    """Orthographic evidence mined from training transcripts.

    ``min_count`` and ``dominance`` are deliberately conservative: a rule that
    fires on a genuinely ambiguous word (``Kota`` appears 4 times capitalised and
    27 times not) trades one error for another, so only lopsided cases qualify.
    """

    vocab: dict[str, int] = field(default_factory=dict)
    #: lowercase form -> canonical surface form (acronyms, proper nouns)
    casing: dict[str, str] = field(default_factory=dict)
    #: accent-stripped form -> accented surface form (Javanese ``-é`` suffix)
    accents: dict[str, str] = field(default_factory=dict)
    #: hyphenated surface forms, e.g. ``jalan-jalan``, ``kanca-kancaku``, ``SMA-ku``
    hyphenated: dict[str, int] = field(default_factory=dict)
    #: tokens that legitimately carry a trailing ellipsis, e.g. ``e...``
    ellipsis_tokens: dict[str, int] = field(default_factory=dict)
    min_count: int = 3
    dominance: float = 0.9

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        min_count: int = 3,
        dominance: float = 0.9,
    ) -> "StyleLexicon":
        from collections import Counter, defaultdict

        vocab: Counter[str] = Counter()
        hyphenated: Counter[str] = Counter()
        ellipsis_tokens: Counter[str] = Counter()
        for text in texts:
            for raw in to_scoring_space(text).split():
                if raw.endswith("..."):
                    ellipsis_tokens[raw] += 1
                token = raw[:-3] if raw.endswith("...") else raw
                if not token:
                    continue
                vocab[token] += 1
                if "-" in token and any(c.isalpha() for c in token):
                    hyphenated[token] += 1

        by_lower: dict[str, Counter[str]] = defaultdict(Counter)
        by_plain: dict[str, Counter[str]] = defaultdict(Counter)
        for token, count in vocab.items():
            by_lower[token.lower()][token] += count
            by_plain[strip_accents(token).lower()][token] += count

        casing: dict[str, str] = {}
        for lower, forms in by_lower.items():
            total = sum(forms.values())
            surface, count = forms.most_common(1)[0]
            if total >= min_count and count / total >= dominance and surface != lower:
                casing[lower] = surface

        accents: dict[str, str] = {}
        for plain, forms in by_plain.items():
            total = sum(forms.values())
            surface, count = forms.most_common(1)[0]
            if (
                total >= min_count
                and count / total >= dominance
                and strip_accents(surface).lower() != surface.lower()
            ):
                accents[plain] = surface

        return cls(
            vocab=dict(vocab),
            casing=casing,
            accents=accents,
            hyphenated=dict(hyphenated),
            ellipsis_tokens=dict(ellipsis_tokens),
            min_count=min_count,
            dominance=dominance,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab": self.vocab,
                    "casing": self.casing,
                    "accents": self.accents,
                    "hyphenated": self.hyphenated,
                    "ellipsis_tokens": self.ellipsis_tokens,
                    "min_count": self.min_count,
                    "dominance": self.dominance,
                },
                f,
                ensure_ascii=False,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StyleLexicon":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls(**payload)

    def count(self, token: str) -> int:
        return self.vocab.get(token, 0)


# --------------------------------------------------------------------------
# Individual rules. Each takes and returns a scoring-space string.
# --------------------------------------------------------------------------


#: Characters an LLM decoder likes to emit that the scorer does *not* fold into
#: their ASCII equivalents. A curly apostrophe or a U+2026 ellipsis survives
#: normalisation and turns an otherwise correct word into a mismatch, so this
#: mapping is applied unconditionally rather than being gated.
UNICODE_FOLD = {
    "\u2026": "...",  # … -> ...
    "\u2019": "'",
    "\u2018": "'",
    "\u02bc": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",  # en dash
    "\u2012": "-",
    "\u2011": "-",
    "\u2212": "-",  # minus sign
    "\u00a0": " ",
    "\u200b": "",
}


def rule_canonicalize_unicode(text: str, lex: StyleLexicon) -> str:
    """Fold look-alike punctuation onto the ASCII forms the references use."""
    for source, target in UNICODE_FOLD.items():
        if source in text:
            text = text.replace(source, target)
    # A lone dash surrounded by spaces is a false reduplication hyphen; the
    # scorer keeps it as a token, so drop it before it becomes an insertion.
    text = re.sub(r"\s+-\s+-\s+", " ", text)
    return text


def rule_attach_ellipsis(text: str, lex: StyleLexicon) -> str:
    """``e ...`` -> ``e...`` when the attached form is attested.

    A free-standing ``...`` is deleted by the scorer, so an ellipsis is only
    worth anything when it is glued to the preceding word. Models that emit it
    with a space are throwing the token away.
    """
    words = text.split()
    out: list[str] = []
    for word in words:
        if word == "..." and out and not out[-1].endswith("..."):
            candidate = out[-1] + "..."
            if lex.ellipsis_tokens.get(candidate, 0) >= 1:
                out[-1] = candidate
                continue
        out.append(word)
    return " ".join(out)


def rule_join_hyphen_spacing(text: str, lex: StyleLexicon) -> str:
    """``jalan - jalan`` -> ``jalan-jalan``: a stray hyphen token is never right."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        if 0 < i < len(words) - 1 and words[i] == "-" and out:
            out[-1] = f"{out[-1]}-{words[i + 1]}"
            i += 2
            continue
        out.append(words[i])
        i += 1
    return " ".join(out)


def rule_join_reduplication(text: str, lex: StyleLexicon) -> str:
    """Join adjacent tokens when the hyphenated form is attested in training.

    Covers exact reduplication (``jalan jalan`` -> ``jalan-jalan``), affixed
    reduplication (``kanca kancaku`` -> ``kanca-kancaku``) and clitics
    (``SMA ku`` -> ``SMA-ku``). Guarded by attestation because the risk is
    symmetric: a wrong join costs exactly as much as a missed one.
    """
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            left, right = words[i], words[i + 1]
            suffix = "..." if right.endswith("...") else ""
            bare_right = right[: -len(suffix)] if suffix else right
            candidate = f"{left}-{bare_right}"
            joined_count = lex.hyphenated.get(candidate, 0)
            if joined_count >= 2 and lex.vocab.get(candidate, 0) >= 2:
                separate = min(lex.vocab.get(left, 0), lex.vocab.get(bare_right, 0))
                if joined_count >= separate:
                    out.append(candidate + suffix)
                    i += 2
                    continue
        out.append(words[i])
        i += 1
    return " ".join(out)


def rule_fix_acronym_casing(text: str, lex: StyleLexicon) -> str:
    """Upper-case tokens that the training data always writes as acronyms.

    Restricted to short all-caps forms (``SMP``, ``HP``, ``PR``, ``OSIS``),
    which are unambiguous. General casing is handled by
    :func:`rule_fix_casing` and gated separately because it is riskier.

    Unlike :func:`rule_fix_casing` this rule *does* apply to the first token.
    The scorer only lowercases a sentence-initial letter when the next letter is
    lowercase, so an utterance-initial ``SMP`` survives on the reference side and
    a lowercase ``smp`` will not match it.
    """
    out: list[str] = []
    for word in text.split():
        suffix = "..." if word.endswith("...") else ""
        bare = word[: -len(suffix)] if suffix else word
        head, sep, tail = bare.partition("-")
        canonical = lex.casing.get(head.lower())
        if (
            canonical
            and canonical.isupper()
            and 2 <= len(canonical) <= 5
            and head != canonical
            and head.lower() == canonical.lower()
        ):
            bare = canonical + sep + tail
        out.append(bare + suffix)
    return " ".join(out)


def rule_fix_casing(text: str, lex: StyleLexicon) -> str:
    """Snap every token to its dominant training casing.

    Never touches the first token: for title-cased words the scorer lowercases
    the sentence-initial letter on both sides, so the capital there is a free
    variable and "fixing" it can only introduce noise. Utterance-initial
    acronyms are the exception and are handled by
    :func:`rule_fix_acronym_casing`.
    """
    words = text.split()
    out: list[str] = []
    for idx, word in enumerate(words):
        suffix = "..." if word.endswith("...") else ""
        bare = word[: -len(suffix)] if suffix else word
        if idx > 0:
            canonical = lex.casing.get(bare.lower())
            if canonical and canonical != bare:
                bare = canonical
        out.append(bare + suffix)
    return " ".join(out)


def rule_restore_accents(text: str, lex: StyleLexicon) -> str:
    """``namane`` -> ``namané`` when training data is unanimous about it."""
    out: list[str] = []
    for word in text.split():
        suffix = "..." if word.endswith("...") else ""
        bare = word[: -len(suffix)] if suffix else word
        if lex.vocab.get(bare, 0) == 0:
            canonical = lex.accents.get(strip_accents(bare).lower())
            if canonical and strip_accents(canonical).lower() == strip_accents(bare).lower():
                bare = canonical if bare[:1].islower() else canonical[:1].upper() + canonical[1:]
        out.append(bare + suffix)
    return " ".join(out)


def rule_drop_bare_ellipsis(text: str, lex: StyleLexicon) -> str:
    """Delete free-standing ``...`` tokens: the scorer removes them anyway.

    Keeping them out of the string makes ROVER voting and MBR distances
    reflect real words instead of punctuation the metric ignores.
    """
    return " ".join(w for w in text.split() if w != "...")


ALL_RULES: dict[str, Callable[[str, StyleLexicon], str]] = {
    "canonicalize_unicode": rule_canonicalize_unicode,
    "join_hyphen_spacing": rule_join_hyphen_spacing,
    "attach_ellipsis": rule_attach_ellipsis,
    "drop_bare_ellipsis": rule_drop_bare_ellipsis,
    "join_reduplication": rule_join_reduplication,
    "fix_acronym_casing": rule_fix_acronym_casing,
    "fix_casing": rule_fix_casing,
    "restore_accents": rule_restore_accents,
}

#: Rules that must see the model's *raw* output. Normalisation deletes a
#: free-standing ``...``, so ``e ... lupa`` and ``e lupa`` are indistinguishable
#: once normalised — the attached form can only be recovered before that.
PRE_NORM_RULES = ("canonicalize_unicode", "join_hyphen_spacing", "attach_ellipsis")

#: Rules that operate in scoring space, where tokens are stable.
POST_NORM_RULES = (
    "drop_bare_ellipsis",
    "join_reduplication",
    "fix_acronym_casing",
    "fix_casing",
    "restore_accents",
)

#: Always applied; folding look-alike punctuation can only help, and gating it
#: on development hypotheses that happen to contain none would silently disable
#: it for the test set.
MANDATORY_RULES = ("canonicalize_unicode",)

DEFAULT_RULE_ORDER = PRE_NORM_RULES + POST_NORM_RULES


# --------------------------------------------------------------------------
# Degenerate-output repair
# --------------------------------------------------------------------------

#: Fastest speaker in the development set manages 3.63 words/s; anything past
#: this is a decoder loop, not speech.
DEGENERATE_WORDS_PER_SECOND = 4.5


def words_per_second(text: str, duration_s: float) -> float:
    if not duration_s or duration_s != duration_s or duration_s <= 0:
        return float("nan")
    return len(text.split()) / duration_s


def looks_degenerate(text: str, duration_s: float, threshold: float = DEGENERATE_WORDS_PER_SECOND) -> bool:
    rate = words_per_second(text, duration_s)
    return rate == rate and rate > threshold


def collapse_repeated_ngrams(text: str, max_repeats: int = 3, max_n: int = 6) -> str:
    """Collapse runs of a repeated n-gram down to ``max_repeats`` copies.

    Only called on outputs that already failed the speaking-rate check.
    Unconditional collapsing would be wrong: the references contain
    ``bla bla bla bla bla bla`` and ``tolong tolong tolong tolong``.
    """
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        collapsed = False
        for n in range(1, max_n + 1):
            if i + 2 * n > len(words):
                break
            gram = words[i : i + n]
            repeats = 1
            while words[i + repeats * n : i + (repeats + 1) * n] == gram:
                repeats += 1
            if repeats > max_repeats:
                out.extend(gram * max_repeats)
                i += repeats * n
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def repair_degenerate(text: str, duration_s: float) -> str:
    if looks_degenerate(text, duration_s):
        return collapse_repeated_ngrams(text)
    return text


# --------------------------------------------------------------------------
# Gated pipeline
# --------------------------------------------------------------------------


@dataclass
class RulePipeline:
    """A validated sequence of orthography rules."""

    lexicon: StyleLexicon
    rules: tuple[str, ...] = DEFAULT_RULE_ORDER
    #: Populated by :meth:`fit` — rule name -> WER delta (negative is good).
    deltas: dict[str, float] = field(default_factory=dict)

    def apply(self, text: str, duration_s: float | None = None) -> str:
        """Raw model output -> final, scorer-stable transcript."""
        active = set(self.rules) | set(MANDATORY_RULES)
        out = str(text or "")
        for name in PRE_NORM_RULES:
            if name in active:
                out = ALL_RULES[name](out, self.lexicon)
        out = to_scoring_space(out)
        if duration_s is not None:
            out = repair_degenerate(out, duration_s)
        for name in POST_NORM_RULES:
            if name in active:
                out = ALL_RULES[name](out, self.lexicon)
        return to_scoring_space(out)

    def apply_many(
        self, texts: Sequence[str], durations: Sequence[float] | None = None
    ) -> list[str]:
        if durations is None:
            return [self.apply(t) for t in texts]
        return [self.apply(t, d) for t, d in zip(texts, durations)]

    def fit(
        self,
        references: Sequence[str],
        hypotheses: Sequence[str],
        candidate_rules: Sequence[str] = DEFAULT_RULE_ORDER,
        min_gain: float = 1e-5,
        verbose: bool = True,
    ) -> "RulePipeline":
        """Greedily keep the rules that reduce WER on held-out *raw* hypotheses.

        Forward selection, and every trial re-runs the whole pipeline from the
        raw text rather than chaining onto the previous stage's output. The rules
        interact — ``attach_ellipsis`` changes what ``join_reduplication`` sees,
        and half of them run before normalisation while the rest run after — so
        evaluating a candidate in isolation would measure the wrong thing.

        ``hypotheses`` must be raw decoder output, not normalised text.
        """
        kept: list[str] = []
        deltas: dict[str, float] = {}

        def evaluate(rules: Sequence[str]) -> tuple[float, list[str]]:
            trial = RulePipeline(lexicon=self.lexicon, rules=tuple(rules))
            texts = trial.apply_many(hypotheses)
            return corpus_wer(references, texts), texts

        base, _ = evaluate(())
        if verbose:
            print(f"post-processing gate: baseline WER {base:.5f}")
        remaining = [
            r for r in candidate_rules if r in ALL_RULES and r not in MANDATORY_RULES
        ]
        while remaining:
            scored: list[tuple[float, str]] = []
            for name in remaining:
                order = [r for r in DEFAULT_RULE_ORDER if r in kept or r == name]
                wer, _ = evaluate(order)
                scored.append((wer, name))
            scored.sort(key=lambda x: x[0])
            best_wer, best_name = scored[0]
            gain = base - best_wer
            deltas[best_name] = -gain
            if gain < min_gain:
                if verbose:
                    for wer, name in scored:
                        print(f"  rejected {name:22s} WER {wer:.5f} ({wer - base:+.5f})")
                break
            if verbose:
                print(f"  keep     {best_name:22s} WER {best_wer:.5f} ({-gain:+.5f})")
            kept.append(best_name)
            base = best_wer
            remaining = [r for r in remaining if r != best_name]
        # Canonical order, so pre-normalisation repairs still run first.
        self.rules = tuple(r for r in DEFAULT_RULE_ORDER if r in kept)
        self.deltas = deltas
        return self

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lexicon_path = path.with_suffix(".lexicon.json")
        self.lexicon.save(lexicon_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"rules": list(self.rules), "deltas": self.deltas, "lexicon": lexicon_path.name},
                f,
                ensure_ascii=False,
                indent=1,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RulePipeline":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        lexicon = StyleLexicon.load(path.parent / payload["lexicon"])
        return cls(
            lexicon=lexicon,
            rules=tuple(payload["rules"]),
            deltas=payload.get("deltas", {}),
        )
