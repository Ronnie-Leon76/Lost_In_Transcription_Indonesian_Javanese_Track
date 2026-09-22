# %% [markdown]
# Spanish-Nahuatl track — RunPod cells
#
# Set SPNH_DEV_DIR to the competition development-set folder and TETEL_DIR to
# the extracted Tetelancingo dataset.  Neither directory belongs in the final
# submission archive.

# %% [markdown]
# ## 0. RunPod setup (run first on a GPU pod)
#
# This installs the complete, compatible environment for Qwen3-ASR, LoRA
# training, packaging, and Hugging Face publication.  Run it on a fresh Python
# 3.12 CUDA pod.  It deliberately runs outside the final competition archive.

# %%
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path("/workspace/Lost_In_Transcription_Indonesian_Javanese_Track")
if not ROOT.is_dir():
    raise FileNotFoundError(f"Update ROOT for this pod: {ROOT}")

# RunPod's base image can contain a Debian-managed Blinker without pip metadata.
# Repair that narrow case before resolving qwen-asr's optional Gradio dependency.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--upgrade", "--ignore-installed", "blinker"],
    check=True,
)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "eager",
        "-r", str(ROOT / "requirements-runpod.txt"),
    ],
    check=True,
)
subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_runpod.py")], check=True)
print("RunPod environment verified. Restart the kernel if pip upgraded an imported package.")

# %%
from pathlib import Path
import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
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
# Package only after the final all-data merge, so the ZIP and separately-published
# model/ directory always describe the same weights.
package = subprocess.run(
    [
        sys.executable,
        str(ROOT / "spanish_nahuatl" / "package_submission.py"),
        "--model", str(final_merged),
        "--glossary", str(ARTIFACTS / "glossary.json"),
    ],
    check=True,
    capture_output=True,
    text=True,
)
print(package.stdout)
SUBMISSION_ZIP = ARTIFACTS / "submission.zip"
assert SUBMISSION_ZIP.is_file() and SUBMISSION_ZIP.stat().st_size > 0
print(f"Ready to publish: {final_merged} and {SUBMISSION_ZIP}")

# %% [markdown]
# ## Publish the final weights and submission archive to Hugging Face
#
# Set `SPNH_HF_REPO` to an existing or new `owner/repository` name to control
# the destination. If omitted, this cell creates a repository under the signed-
# in Hugging Face account. `HF_TOKEN` is requested privately when it is not
# already a RunPod secret; it is never written to the ZIP, model, or notebook.

# %%
from getpass import getpass
from huggingface_hub import HfApi, hf_hub_url

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = getpass("Hugging Face write token (not saved): ")
if not os.environ["HF_TOKEN"].strip():
    raise RuntimeError("HF_TOKEN is required to publish the model and submission archive")

api = HfApi(token=os.environ["HF_TOKEN"])
account = api.whoami()["name"]
REPO_ID = os.environ.get("SPNH_HF_REPO", f"{account}/lost-in-transcription-spanish-nahuatl")
PRIVATE = os.environ.get("SPNH_HF_PRIVATE", "0") == "1"

api.create_repo(repo_id=REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)

# `final-merged` is uploaded as model/ so users can download just the weights;
# submission.zip remains at the repository root, ready for competition upload.
model_commit = api.upload_folder(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=str(final_merged),
    path_in_repo="model",
    commit_message="Upload final Spanish-Nahuatl merged model weights",
    ignore_patterns=["**/__pycache__/**", "*.lock"],
)
archive_commit = api.upload_file(
    path_or_fileobj=str(SUBMISSION_ZIP),
    path_in_repo="submission.zip",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload Spanish-Nahuatl competition submission archive",
)
api.upload_file(
    path_or_fileobj=str(ARTIFACTS / "glossary.json"),
    path_in_repo="assets/glossary.json",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload final glossary provenance",
)

print("Model weights:", getattr(model_commit, "commit_url", model_commit))
print("Submission archive:", getattr(archive_commit, "commit_url", archive_commit))
print("Repository:", f"https://huggingface.co/{REPO_ID}")
print("Archive download:", hf_hub_url(repo_id=REPO_ID, filename="submission.zip", repo_type="model"))
