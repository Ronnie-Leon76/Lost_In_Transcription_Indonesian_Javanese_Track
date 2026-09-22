# %% [markdown]
# Spanish-Nahuatl track — RunPod cells
#
# Set SPNH_DEV_DIR to the competition development-set folder and TETEL_DIR to
# the extracted Tetelancingo dataset.  Neither directory belongs in the final
# submission archive.

# %%
from pathlib import Path
import sys

ROOT = Path("/workspace/Lost_In_Transcription_Indonesian_Javanese_Track")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from spanish_nahuatl import prepare
from lit_jv import prompts
from lit_jv.train_lora import AugmentSpec, LoraSpec, StageSpec, TrainSpec, train, merge_and_save
from lit_jv.decode import DecodeConfig, FallbackConfig, Qwen3AsrDecoder, build_backend
from lit_jv.scoring import wer_report

SPNH_DEV_DIR = Path("/workspace/nahuatl_dev")       # metadata.tsv + clips/
TETEL_DIR = Path("/workspace/tetelancingo_nahuatl") # extracted MDC archive
ARTIFACTS = ROOT / "artifacts" / "spnh"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# %%
# Fit rows and WER selection rows are disjoint by speaker and conversation.
# Never use random clips here: it gives a misleadingly good score.
dev_rows = prepare.load_competition_dev(SPNH_DEV_DIR)
dev_train, dev_eval = prepare.holdout_speakers(dev_rows, count=1, seed=2026)
tetel_train = prepare.load_tetelancingo(TETEL_DIR, split="train")
tetel_eval = prepare.load_tetelancingo(TETEL_DIR, split="test")

fit_rows = tetel_train + dev_train
terms = prepare.glossary(fit_rows)
(ARTIFACTS / "glossary.json").write_text(__import__("json").dumps(terms, ensure_ascii=False, indent=2), encoding="utf-8")
print({"dev_train": len(dev_train), "dev_eval": len(dev_eval), "tetel_train": len(tetel_train), "tetel_eval": len(tetel_eval), "glossary": len(terms), "orthography_flags": prepare.orthography_audit(tetel_train)})

# %%
# Decision gate 1: record the unadapted model's WER before spending hours on
# LoRA.  A fine-tune that does not beat this number is rejected.
def score_rows(model_path, rows, prompt, name, chunk_size=16):
    backend = build_backend(model_path, prefer="vllm", log=print)
    decoder = Qwen3AsrDecoder(
        backend,
        fallback=FallbackConfig(enabled=False),  # compare greedy systems first
        log=print,
    )
    try:
        hypotheses = decoder.decode_pass(
            rows,
            DecodeConfig(prompt=prompt, name=name, temperature=0.0),
            chunk_size=chunk_size,
        )
        report = wer_report([row["raw_text"] for row in rows], [hyp.text for hyp in hypotheses])
        print(report.as_row(name))
        return report
    finally:
        decoder.close()

baseline_prompt = prompts.PromptConfig(
    prose="codeswitch_spnh",
    style_tag=prompts.STYLE_VOICE_NOTE_SP_NH,
    glossary=tuple(terms),
    max_glossary_terms=128,
    force_language="Spanish",
)
baseline = score_rows("Qwen/Qwen3-ASR-1.7B", dev_eval, baseline_prompt, "qwen-zero-shot")

# %%
# This is the first serious run.  The final in-style stage deliberately gives
# the competition voice-note data more influence than read Tetelancingo speech.
# WER, not loss, selects the checkpoint.
spec = TrainSpec(
    model_path="Qwen/Qwen3-ASR-1.7B",
    output_dir=str(ARTIFACTS / "lora"),
    target_language="Spanish",
    lora=LoraSpec(r=16, alpha=32, adapt_text_decoder=True, adapt_audio_encoder=False),
    augment=AugmentSpec(corpora=("tetelancingo",)),
    prompt=prompts.PromptConfig(
        prose="codeswitch_spnh",
        style_tag=prompts.STYLE_VOICE_NOTE_SP_NH,
        glossary=tuple(terms),
        max_glossary_terms=128,
        force_language="Spanish",
    ),
    # Teacher-forced evaluation is disabled: raw audio rows are not tokenized
    # examples. The generation WER probe below is the real selection measure.
    eval_steps=0,
    wer_eval_steps=10,
    wer_eval_clips=len(dev_eval),
    wer_eval_batch_size=4,
    run_name="spnh-qwen3asr",
)
stages = (
    StageSpec("s1-tetelancingo", {"tetelancingo": 1.0, "spnh_dev": 1.25}, epochs=2.0, learning_rate=1.0e-4, batch_size=4, grad_accumulation=8),
    StageSpec("s2-voice-note", {"spnh_dev": 1.0, "tetelancingo": 0.15}, epochs=5.0, learning_rate=3.0e-5, batch_size=4, grad_accumulation=8),
)
result = train(fit_rows, spec, stages=stages, save_adapter_as="selected", eval_rows=dev_eval)
result["best_wer"]

# %%
# Merge *only the adapter chosen by held-out dev WER*, then package an offline
# archive.  The generated ZIP root has main.py, model/, lit_jv/ and config only.
merged = merge_and_save(spec, ARTIFACTS / "lora" / "selected", ARTIFACTS / "merged")
print(merged)

# Decision gate 2: the merged model must beat the baseline on competition-dev
# WER. Tetelancingo's released test split is one one-time transfer sanity check,
# not a tuning target.
selected_dev = score_rows(merged, dev_eval, spec.prompt, "selected-dev")
selected_tetel = score_rows(merged, tetel_eval, spec.prompt, "selected-tetel")
assert selected_dev.wer < baseline.wer, "reject: selected adapter did not beat zero-shot Qwen"

# %%
# Only after the assertion passes, train the same selected recipe once more on
# every released training example. There is no held-out WER in this final fit:
# all tuning decisions are already frozen above.
final_rows = prepare.load_tetelancingo(TETEL_DIR, split=None) + dev_rows
final_spec = TrainSpec(**{**spec.__dict__, "output_dir": str(ARTIFACTS / "final-lora"), "wer_eval_steps": 0})
final = train(final_rows, final_spec, stages=stages, save_adapter_as="final", eval_rows=None)
final_merged = merge_and_save(final_spec, ARTIFACTS / "final-lora" / "final", ARTIFACTS / "final-merged")
print(final_merged)

# %%
# In a terminal (not the competition container):
# python spanish_nahuatl/package_submission.py --model artifacts/spnh/final-merged --glossary artifacts/spnh/glossary.json
