"""Prompt construction — the single source of truth for train/inference parity.

Every training example and every inference request is built by the functions in
this module. If the training prefix and the inference prefix drift apart by even
one token, the fine-tune silently degrades, and that class of bug is invisible
in the loss curve.

Three ideas live here:

*Context injection.* Qwen3-ASR takes free-form text in the ``system`` turn and
uses it to bias recognition (the "context enhancement" feature of the Qwen3-ASR
family; see ``qwen_asr.inference.qwen3_asr.Qwen3ASRModel._build_messages``). We
use it for a code-switching declaration, an in-context orthography glossary
mined from the training data, and style exemplars. This is the in-context
adaptation route that Omnilingual ASR (arXiv:2511.09690) uses to reach unseen
languages with an LLM decoder, applied to an unseen *orthography* rather than an
unseen language.

*Style tags.* Training data comes from corpora with incompatible transcription
conventions (spontaneous voice notes with disfluency ellipses, Jember session
recordings, read TTS speech). Rather than throwing away the mismatched corpora
or letting them corrupt the target style, each corpus gets a tag in its system
prompt during training. At inference we always ask for the voice-note tag. This
is domain-adapter conditioning done through the prompt instead of through
weights — the same separation of "shared acoustics" from "per-domain output
convention" that language adapters buy in MMS (Pratap et al., 2024) and in
Kulkarni et al. (arXiv:2310.07423) for code-switching.

*Language conditioning.* ``test_metadata.csv`` ships a per-clip ``language``
field (``javind`` / ``jav`` / ``ind``). That is metadata about the clip being
transcribed, not information leaked from other test clips, so using it is
compliant with the "process each test sample independently" rule. Monolingual
Indonesian clips get a different declaration from code-switched ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# Chat template. Verified byte-for-byte against chat_template.json of
# Qwen/Qwen3-ASR-1.7B so that main.py can build prompts without loading the
# processor (and without depending on the qwen_asr package at inference time).
# --------------------------------------------------------------------------

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
ASR_TEXT_TAG = "<asr_text>"
LANGUAGE_PREFIX = "language "

#: Assistant-side language name. "Javanese" is not one of Qwen3-ASR's 30
#: supported languages, so forcing it would put the decoder off-distribution.
#: The code-switch information goes in the system context instead, and the
#: assistant prefix stays on the in-distribution "Indonesian".
DEFAULT_FORCE_LANGUAGE = "Indonesian"

# Style tags. Keep these stable: they are baked into the fine-tuned weights.
STYLE_VOICE_NOTE = "wa-voicenote-id-jv"
STYLE_JEMBER = "jember-session-jv"
STYLE_READ = "read-tts"
STYLE_VOICE_NOTE_SP_NH = "wa-voicenote-sp-nh"
STYLE_VOICE_NOTE_SP_EN = "wa-voicenote-sp-en"


def render_chat_prompt(context: str, force_language: str | None = DEFAULT_FORCE_LANGUAGE) -> str:
    """Render the full text prompt for one audio clip.

    Equivalent to ``processor.apply_chat_template([{"role": "system", ...},
    {"role": "user", "content": [{"type": "audio", ...}]}],
    add_generation_prompt=True)``, optionally with the assistant turn prefilled
    with the language prefix so the decoder emits transcript tokens only.
    """
    prompt = (
        "<|im_start|>system\n"
        f"{context or ''}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{AUDIO_PLACEHOLDER}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if force_language:
        prompt += f"{LANGUAGE_PREFIX}{force_language}{ASR_TEXT_TAG}"
    return prompt


def render_target(text: str, language: str = DEFAULT_FORCE_LANGUAGE) -> str:
    """Render the supervised target for fine-tuning.

    Matches the format the official SFT script expects
    (``language English<asr_text>This is a test sentence.``). Keeping the
    language prefix in the target means the same checkpoint works both with a
    prefilled prefix and with free language identification.
    """
    return f"{LANGUAGE_PREFIX}{language}{ASR_TEXT_TAG}{text}"


def parse_generation(raw: str, force_language: str | None = DEFAULT_FORCE_LANGUAGE) -> str:
    """Extract the transcript from a raw generation.

    Handles both the prefilled-prefix case (output is already transcript-only)
    and the free case (``language Indonesian<asr_text>...``).
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if ASR_TEXT_TAG in s:
        meta, s = s.split(ASR_TEXT_TAG, 1)
        if "language none" in meta.lower() and not s.strip():
            return ""
    elif force_language is None and s.lower().startswith(LANGUAGE_PREFIX):
        # Truncated generation that produced a language line and nothing else.
        return ""
    for tail in ("<|im_end|>", "<|endoftext|>"):
        s = s.split(tail)[0]
    return s.strip()


# --------------------------------------------------------------------------
# System context
# --------------------------------------------------------------------------

#: Prose blocks are numbered so the notebook can grid-search them on dev and
#: report which one it chose. Written in Indonesian because the model's
#: instruction following is stronger in the target language than in English for
#: this family, and because it keeps the context short.
CONTEXT_PROSE: dict[str, str] = {
    "none": "",
    "minimal": "Percakapan santai bahasa Indonesia bercampur bahasa Jawa.",
    "codeswitch": (
        "Pesan suara percakapan santai antara dua penutur bilingual. "
        "Bahasa Indonesia bercampur bahasa Jawa ngoko dalam satu kalimat. "
        "Tulis setiap kata dalam ejaan bahasanya sendiri, jangan diterjemahkan."
    ),
    "codeswitch_style": (
        "Pesan suara percakapan santai antara dua penutur bilingual. "
        "Bahasa Indonesia bercampur bahasa Jawa ngoko dalam satu kalimat. "
        "Tulis setiap kata dalam ejaan bahasanya sendiri, jangan diterjemahkan. "
        "Salin apa adanya termasuk pengulangan, kata terpotong, dan jeda ragu "
        "yang ditulis dengan tiga titik seperti e... atau mem... membaca. "
        "Tulis kata ulang dengan tanda hubung seperti jalan-jalan atau kanca-kancaku."
    ),
    "monolingual_id": (
        "Pesan suara percakapan santai dalam bahasa Indonesia. "
        "Salin apa adanya termasuk pengulangan, kata terpotong, dan jeda ragu "
        "yang ditulis dengan tiga titik seperti e..."
    ),
    "codeswitch_spnh": (
        "Nota de voz de una conversacion informal entre hablantes bilingues. "
        "Espanol mezclado con nahuatl en la misma oracion. "
        "Escribe cada palabra en la ortografia de su idioma, no traduzcas. "
        "Copia disfluencias, cortes y pausas tal cual, incluso puntos suspensivos."
    ),
    "codeswitch_spen": (
        "Nota de voz de una conversacion informal entre hablantes bilingues. "
        "Espanol e ingles se mezclan en la misma oracion. "
        "Escribe cada palabra en la ortografia de su idioma, no traduzcas. "
        "Copia disfluencias, cortes y pausas tal cual, incluso puntos suspensivos."
    ),
    "monolingual_es": (
        "Nota de voz de una conversacion informal en espanol. "
        "Copia tal cual, incluyendo repeticiones, cortes y pausas."
    ),
}

_LANGUAGE_FIELD_TO_PROSE = {
    "javind": "codeswitch_style",
    "jav": "codeswitch_style",
    "ind": "monolingual_id",
    "spnh": "codeswitch_spnh",
    "nah": "codeswitch_spnh",
    "nhe": "codeswitch_spnh",
    "nci": "codeswitch_spnh",
    "nahuatl": "codeswitch_spnh",
    "spen": "codeswitch_spen",
    "eng": "codeswitch_spen",
    "en": "codeswitch_spen",
    "english": "codeswitch_spen",
    "spa": "monolingual_es",
    "es": "monolingual_es",
    "spanish": "monolingual_es",
}


@dataclass
class PromptConfig:
    """Everything that determines the system context, in one hashable place."""

    prose: str = "codeswitch_style"
    #: Adapt the prose to the per-clip ``language`` field when it is available.
    use_language_field: bool = True
    style_tag: str = STYLE_VOICE_NOTE
    include_style_tag: bool = True
    #: Orthography glossary injected as hotwords (mined from training text).
    glossary: tuple[str, ...] = ()
    max_glossary_terms: int = 96
    #: Short reference sentences that demonstrate the target convention.
    exemplars: tuple[str, ...] = ()
    max_exemplars: int = 0
    force_language: str | None = DEFAULT_FORCE_LANGUAGE

    def with_glossary(self, terms: Iterable[str]) -> "PromptConfig":
        return PromptConfig(**{**self.__dict__, "glossary": tuple(terms)})

    def describe(self) -> str:
        bits = [f"prose={self.prose}"]
        if self.include_style_tag:
            bits.append(f"tag={self.style_tag}")
        if self.glossary:
            bits.append(f"glossary={min(len(self.glossary), self.max_glossary_terms)}")
        if self.exemplars and self.max_exemplars:
            bits.append(f"exemplars={min(len(self.exemplars), self.max_exemplars)}")
        if self.use_language_field:
            bits.append("lang-field")
        return " ".join(bits)


def build_context(
    config: PromptConfig,
    language_field: str | None = None,
    style_tag: str | None = None,
) -> str:
    """Assemble the ``system`` turn for one clip."""
    prose_key = config.prose
    # Only the code-switch variants get specialised per clip; "none" and
    # "minimal" are ablation settings and must stay untouched.
    if config.use_language_field and language_field and prose_key in (
        "codeswitch",
        "codeswitch_style",
        "codeswitch_spnh",
        "codeswitch_spen",
    ):
        mapped = _LANGUAGE_FIELD_TO_PROSE.get(str(language_field).strip().lower())
        if mapped in ("monolingual_id", "monolingual_es", "codeswitch_spnh", "codeswitch_spen"):
            prose_key = mapped
    parts: list[str] = []
    tag = style_tag if style_tag is not None else config.style_tag
    if config.include_style_tag and tag:
        parts.append(f"[gaya: {tag}]")
    prose = CONTEXT_PROSE.get(prose_key, CONTEXT_PROSE["codeswitch_style"])
    if prose:
        parts.append(prose)
    if config.glossary and config.max_glossary_terms:
        terms = list(config.glossary)[: config.max_glossary_terms]
        parts.append("Kosakata: " + ", ".join(terms) + ".")
    if config.exemplars and config.max_exemplars:
        shown = list(config.exemplars)[: config.max_exemplars]
        parts.append("Contoh gaya transkrip: " + " | ".join(shown))
    return " ".join(parts).strip()


def build_prompt(
    config: PromptConfig,
    language_field: str | None = None,
    style_tag: str | None = None,
) -> str:
    """Context plus chat template — what actually goes to the model."""
    context = build_context(config, language_field=language_field, style_tag=style_tag)
    return render_chat_prompt(context, force_language=config.force_language)


# --------------------------------------------------------------------------
# Glossary mining
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\W\d_][\w'\-é]*", re.UNICODE)


def mine_glossary(
    texts: Sequence[str],
    max_terms: int = 96,
    min_count: int = 2,
    background: Sequence[str] | None = None,
) -> list[str]:
    """Pick the terms whose *spelling* the model is most likely to get wrong.

    Scoring favours tokens that carry a convention the base model has no reason
    to produce: an ``é``, a reduplication hyphen, an all-caps acronym, or an
    apostrophe. Plain frequent words are excluded — spending context tokens on
    ``aku`` or ``itu`` buys nothing.

    ``background`` (optional) is a list of texts from a *different* style; terms
    that are common there are down-weighted so the glossary stays specific.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_TOKEN_RE.findall(str(text)))
    bg: Counter[str] = Counter()
    if background:
        for text in background:
            bg.update(_TOKEN_RE.findall(str(text)))

    scored: list[tuple[float, str]] = []
    for term, count in counts.items():
        if count < min_count or len(term) < 2:
            continue
        weight = 0.0
        if "é" in term:
            weight += 3.0
        if "-" in term:
            weight += 1.5 if len(set(term.split("-"))) > 1 else 0.6
        if term.isupper():
            weight += 2.5
        elif term[:1].isupper():
            weight += 1.0
        if "'" in term:
            weight += 2.0
        if weight == 0.0:
            continue
        penalty = 1.0 + 0.5 * (bg.get(term, 0) > count)
        scored.append((weight * (1.0 + count) ** 0.5 / penalty, term))

    scored.sort(key=lambda x: (-x[0], x[1]))
    seen: set[str] = set()
    out: list[str] = []
    for _, term in scored:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= max_terms:
            break
    return out


def mine_exemplars(texts: Sequence[str], n: int = 6, max_chars: int = 140) -> list[str]:
    """Pick short reference snippets that show the most style features at once."""
    scored: list[tuple[float, str]] = []
    for text in texts:
        s = " ".join(str(text).split())
        if not 40 <= len(s) <= max_chars:
            continue
        score = (
            2.0 * s.count("...")
            + 1.0 * s.count("-")
            + 1.0 * s.count("é")
            + 1.5 * sum(1 for w in s.split() if w.isupper() and len(w) > 1)
        )
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:n]]
