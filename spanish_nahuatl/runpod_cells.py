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
# Sanity-check transfer to the held-out Tetelancingo split once.  Do not use it
# for repeated tuning; its role is to catch an adapter that forgot Nahuatl.
from lit_jv.decode import DecodeConfig, decode_rows
from lit_jv.scoring import wer_report

# Reuse the same selected adapter/model path in your existing decoder evaluation
# cell, then score tetel_eval with wer_report(reference, hypothesis).  Select
# the run by dev_eval WER, not by Tetelancingo WER.

# %%
# Merge *only the adapter chosen by held-out dev WER*, then package an offline
# archive.  The generated ZIP root has main.py, model/, lit_jv/ and config only.
merged = merge_and_save(spec, ARTIFACTS / "lora" / "selected", ARTIFACTS / "merged")
print(merged)
# In a terminal (not the competition container):
# python spanish_nahuatl/package_submission.py --model artifacts/spnh/merged --glossary artifacts/spnh/glossary.json
