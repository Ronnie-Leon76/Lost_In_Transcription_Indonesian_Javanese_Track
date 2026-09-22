"""CPU-only tests for everything that runs without a GPU.

These cover the parts of the system where a silent bug would be invisible in a
loss curve or a leaderboard score: the scorer replica, prompt/target formatting,
the combiners, and the orthography rules. Run them before every submission:

    python -m pytest tests -q          # or: python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lit_jv import combine, data, postprocess, prompts, scoring  # noqa: E402

DEV_DIR = data.resolve_dev_dir()


class Skip(Exception):
    """Optional local corpus is absent. Not a failure."""


def _require_dev() -> None:
    if not data.dev_dir_ready(DEV_DIR):
        raise Skip(
            f"no development set at {DEV_DIR} "
            "(stage data/indonesian_javanese/development/, or set LIT_DEV_DIR)"
        )


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------


def test_scorer_self_check():
    scoring.self_check()


def test_load_dotenv_fills_missing_keys_only(tmp_path=None):
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".env"
        path.write_text(
            "# comment\nLIT_DOTENV_PROBE=from-file\nLIT_DOTENV_KEEP=from-file\n",
            encoding="utf-8",
        )
        os.environ.pop("LIT_DOTENV_PROBE", None)
        os.environ["LIT_DOTENV_KEEP"] = "from-shell"
        try:
            assert data.load_dotenv(path) == path
            assert os.environ["LIT_DOTENV_PROBE"] == "from-file"
            assert os.environ["LIT_DOTENV_KEEP"] == "from-shell"
            assert data.load_dotenv(Path(tmp) / "missing.env") is None
        finally:
            os.environ.pop("LIT_DOTENV_PROBE", None)
            os.environ.pop("LIT_DOTENV_KEEP", None)


def test_load_dev_manifest_explains_a_missing_folder():
    try:
        data.load_dev_manifest("/definitely/missing/lit-dev")
    except FileNotFoundError as exc:
        message = str(exc)
        assert "LIT_DEV_DIR" in message
        assert "metadata.tsv" in message
    else:
        raise AssertionError("expected FileNotFoundError")


def test_resolve_dev_dir_honours_env_and_explicit_path():
    import os

    old = os.environ.get("LIT_DEV_DIR")
    try:
        os.environ["LIT_DEV_DIR"] = "/tmp/lit-dev-from-env"
        assert data.resolve_dev_dir() == Path("/tmp/lit-dev-from-env")
        assert data.resolve_dev_dir("/explicit") == Path("/explicit")
        os.environ.pop("LIT_DEV_DIR")
        assert data.resolve_dev_dir() == ROOT / "data" / "indonesian_javanese" / "development"
    finally:
        if old is None:
            os.environ.pop("LIT_DEV_DIR", None)
        else:
            os.environ["LIT_DEV_DIR"] = old


def test_scoring_space_is_a_fixed_point_on_real_references():
    _require_dev()
    rows = data.load_dev_manifest(DEV_DIR, with_durations=False)
    assert len(rows) == 372
    for row in rows:
        assert scoring.is_normalization_fixed_point(row["text"]), row["text"][:80]
    # A hypothesis already in scoring space is scored as a perfect match.
    assert scoring.corpus_wer([r["raw_text"] for r in rows], [r["text"] for r in rows]) == 0.0


def test_style_dimensions_are_worth_measurable_wer():
    _require_dev()
    rows = data.load_dev_manifest(DEV_DIR, with_durations=False)
    ablation = scoring.style_loss_ablation(r["raw_text"] for r in rows)
    assert ablation["split reduplication hyphens"] > 0.05
    assert ablation["lowercase everything"] > 0.015
    assert ablation["drop disfluency ellipses"] > 0.008
    assert ablation["all of the above"] > 0.08


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def test_prompt_matches_the_published_chat_template():
    rendered = prompts.render_chat_prompt("CTX", force_language="Indonesian")
    assert rendered == (
        "<|im_start|>system\nCTX<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\nlanguage Indonesian<asr_text>"
    )
    assert prompts.render_chat_prompt("", force_language=None).endswith("<|im_start|>assistant\n")


def test_target_and_parse_round_trip():
    target = prompts.render_target("aku e... lupa")
    assert target == "language Indonesian<asr_text>aku e... lupa"
    assert prompts.parse_generation(target, force_language=None) == "aku e... lupa"
    # With a prefilled prefix the model emits the transcript only.
    assert prompts.parse_generation(" aku e... lupa ") == "aku e... lupa"
    assert prompts.parse_generation("language None<asr_text>") == ""


def test_language_field_switches_the_declaration():
    config = prompts.PromptConfig()
    code_switched = prompts.build_context(config, language_field="javind")
    monolingual = prompts.build_context(config, language_field="ind")
    assert "bahasa Jawa" in code_switched
    assert "bahasa Jawa" not in monolingual
    assert prompts.STYLE_VOICE_NOTE in code_switched


def test_nahuatl_language_field_selects_spanish_nahuatl_prose():
    config = prompts.PromptConfig(prose="codeswitch_spnh")
    mixed = prompts.build_context(config, language_field="nah")
    spanish = prompts.build_context(config, language_field="spa")
    assert "nahuatl" in mixed.lower()
    assert "espanol" in spanish.lower()
    assert "nahuatl" not in spanish.lower()


def test_glossary_prefers_marked_spellings():
    _require_dev()
    rows = data.load_dev_manifest(DEV_DIR, with_durations=False)
    glossary = prompts.mine_glossary([r["text"] for r in rows], max_terms=40)
    assert glossary
    assert any(term.isupper() for term in glossary)
    assert any("-" in term for term in glossary)
    assert any("é" in term for term in glossary)
    assert "aku" not in glossary and "itu" not in glossary


# --------------------------------------------------------------------------
# Combiners
# --------------------------------------------------------------------------


def test_edit_distance_and_alignment():
    assert combine.edit_distance("a b c".split(), "a b c".split()) == 0
    assert combine.edit_distance("a b c".split(), "a x c".split()) == 1
    assert combine.edit_distance("a b c".split(), "a c".split()) == 1
    ops = combine.align("a b c".split(), "a x c d".split())
    assert [op for op, _, _ in ops] == ["equal", "sub", "equal", "ins"]


def test_mbr_picks_the_consensus_hypothesis():
    result = combine.mbr_select(
        ["aku mangan sega", "aku mangan sega", "aku ngombe sega", "wis mangan"]
    )
    assert result.text == "aku mangan sega"
    assert result.agreement == 0.5


def test_rover_can_beat_every_input():
    # Every system has one error, in a different place; voting fixes all of them.
    reference = "aku SMP ing Solo saiki"
    hypotheses = [
        "aku SMP ing Solo wingi",
        "aku smp ing Solo saiki",
        "aku SMP ning Solo saiki",
    ]
    assert min(combine.edit_distance(reference.split(), h.split()) for h in hypotheses) == 1
    assert combine.rover_combine(hypotheses).text == reference


def test_rover_requires_a_majority_before_inserting():
    hypotheses = ["a b c", "a b c", "a b XX c"]
    assert combine.rover_combine(hypotheses).text == "a b c"
    hypotheses = ["a b XX c", "a b XX c", "a b c"]
    assert combine.rover_combine(hypotheses).text == "a b XX c"


def test_length_prior_flags_truncation_and_loops():
    _require_dev()
    rows = data.load_dev_manifest(DEV_DIR, with_durations=True)
    prior = combine.LengthPrior.fit(
        [r["text"] for r in rows], [r["duration"] for r in rows]
    )
    assert 1.5 < prior.mean < 2.3
    # Every real reference should look plausible under its own prior.
    outliers = sum(prior.is_suspicious(r["text"], r["duration"]) for r in rows)
    assert outliers <= 3, outliers
    # A truncated hypothesis on a 30 s clip should not.
    assert prior.is_suspicious("aku mangan", 30.0)
    assert prior.is_suspicious(" ".join(["bla"] * 400), 30.0)


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------


def test_lexicon_learns_dominant_forms_only():
    _require_dev()
    rows = data.load_dev_manifest(DEV_DIR, with_durations=False)
    lexicon = postprocess.StyleLexicon.fit(r["text"] for r in rows)
    assert lexicon.casing.get("smp") == "SMP"
    assert lexicon.casing.get("hp") == "HP"
    # "Kota" is 4 capitalised against 27 lowercase: too ambiguous to touch.
    assert lexicon.casing.get("kota") is None
    assert lexicon.hyphenated.get("jalan-jalan", 0) >= 2
    assert lexicon.ellipsis_tokens.get("e...", 0) >= 10


def test_rules_repair_the_style_damage_they_are_meant_to():
    _require_dev()
    rows = data.load_dev_manifest(DEV_DIR, with_durations=False)
    lexicon = postprocess.StyleLexicon.fit(r["text"] for r in rows)
    pipeline = postprocess.RulePipeline(lexicon=lexicon)
    # Pre-normalisation repairs: an ellipsis is only worth anything attached.
    assert pipeline.apply("aku e ... lupa") == "aku e... lupa"
    assert pipeline.apply("aku e\u2026 lupa") == "aku e... lupa"
    assert pipeline.apply("dhewe jalan - jalan") == "dhewe jalan-jalan"
    assert pipeline.apply("kanca\u2019ku ra teka") == "kanca'ku ra teka"
    # Post-normalisation repairs.
    assert pipeline.apply("dhewe jalan jalan") == "dhewe jalan-jalan"
    # Acronyms and unambiguous proper nouns both snap to their dominant form.
    assert pipeline.apply("mbiyen aku smp ing solo") == "mbiyen aku SMP ing Solo"
    # Utterance-initial casing is subtle. The scorer lowercases a sentence's
    # first letter only when the *next* letter is lowercase, so a title-cased
    # first word is a free variable (both sides get lowercased) but an
    # utterance-initial acronym is not: it survives on both sides and has to be
    # produced in upper case to match.
    assert pipeline.apply("smp iku angel") == "SMP iku angel"
    assert pipeline.apply("solo kuwi adoh") == "solo kuwi adoh"
    assert scoring.to_scoring_space("SMP iku angel") == "SMP iku angel"
    assert scoring.to_scoring_space("Solo kuwi adoh") == "solo kuwi adoh"


def test_unicode_folding_is_never_gated_away():
    lexicon = postprocess.StyleLexicon.fit(["dummy text"])
    empty = postprocess.RulePipeline(lexicon=lexicon, rules=())
    assert empty.apply("aku wis\u2026 mangan") == "aku wis... mangan"


def test_degenerate_repair_only_fires_on_impossible_speaking_rates():
    looped = " ".join(["mangan sega"] * 40)
    assert postprocess.repair_degenerate(looped, 10.0) != looped
    # A real reference that repeats a word six times must survive.
    real = "terus bla bla bla bla bla bla ya wis"
    assert postprocess.repair_degenerate(real, 12.0) == real


def test_rule_gate_rejects_rules_that_do_not_help():
    _require_dev()
    rows = data.load_dev_manifest(DEV_DIR, with_durations=False)
    references = [r["text"] for r in rows]
    lexicon = postprocess.StyleLexicon.fit(references)
    # Simulate a style-blind recogniser: correct words, wrong orthography.
    degraded = [
        t.lower().replace("-", " ").replace("...", " ... ").replace("é", "e")
        for t in references
    ]
    degraded = [" ".join(t.split()) for t in degraded]
    before = scoring.corpus_wer(references, degraded)
    pipeline = postprocess.RulePipeline(lexicon=lexicon).fit(
        references, degraded, verbose=False
    )
    after = scoring.corpus_wer(references, pipeline.apply_many(degraded))
    assert after < before, (before, after)
    assert pipeline.rules, "the gate should have kept at least one rule"
    print(f"    style-blind WER {before:.4f} -> {after:.4f} with rules {pipeline.rules}")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def test_grouped_folds_never_split_a_speaker():
    _require_dev()
    rows = data.load_dev_manifest(DEV_DIR, with_durations=True)
    for _, train_idx, valid_idx in data.iter_grouped_cv(rows, n_folds=4):
        train_speakers = {rows[i]["speaker"] for i in train_idx}
        valid_speakers = {rows[i]["speaker"] for i in valid_idx}
        assert not (train_speakers & valid_speakers)
        assert valid_idx


def test_conversation_holdout_is_disjoint():
    _require_dev()
    rows = data.load_dev_manifest(DEV_DIR, with_durations=False)
    train_idx, valid_idx = data.holdout_by_conversation(rows, held_out="2")
    assert len(train_idx) + len(valid_idx) == len(rows)
    assert {rows[i]["convo_id"] for i in valid_idx} == {"2"}
    assert "2" not in {rows[i]["convo_id"] for i in train_idx}


def test_find_corpus_root_walks_a_nested_extract():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        nested = Path(tmp) / "tetelancingo-nahuatl" / "clips"
        nested.mkdir(parents=True)
        (nested.parent / "metadata.tsv").write_text("audio_filename\ttranscript\n", encoding="utf-8")
        found = data.find_corpus_root(tmp)
        assert found == nested.parent


def test_timestamp_parsing():
    assert data.parse_timestamp("0:01:30") == 90.0
    assert data.parse_timestamp("1:02:03") == 3723.0
    assert data.parse_timestamp("02:05") == 125.0
    assert data.parse_timestamp("7.5") == 7.5


def test_test_manifest_reads_the_submission_format(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "clips").mkdir()
        (tmp / "submission_format.csv").write_text(
            "audio_filename,transcript\na.mp3,hello world\nb.mp3,hello world\n", encoding="utf-8"
        )
        (tmp / "test_metadata.csv").write_text(
            "audio_filename,file_duration_seconds,language\n"
            "b.mp3,12,ind\na.mp3,30,javind\n",
            encoding="utf-8",
        )
        rows = data.load_test_manifest(tmp)
    assert [r["audio_filename"] for r in rows] == ["a.mp3", "b.mp3"]
    assert rows[0]["duration"] == 30.0 and rows[0]["language"] == "javind"
    assert rows[1]["language"] == "ind"


def _main() -> int:
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
        except Skip as exc:
            print(f"skip {name}  ({exc})")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print("\nall tests passed" if not failures else f"\n{failures} failing test(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
