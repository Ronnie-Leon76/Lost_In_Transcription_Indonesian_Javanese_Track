# Lost in Transcription — Indonesian ↔ Javanese track

A code-switched ASR system for the [Lost in Transcription](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/)
Indonesian–Javanese track: Qwen3-ASR with a two-stage LoRA curriculum, prompt-level style
and glossary conditioning, a confidence-triggered re-decode ladder, multi-pass system
combination, and orthography repair rules that are each gated on held-out WER.

The reasoning, measurements and ablations live in
[`notebooks/lost_in_transcription_id_jv.ipynb`](notebooks/lost_in_transcription_id_jv.ipynb).
Start there. This file covers layout and mechanics.

## The short version

The score is a text normaliser followed by `jiwer.wer`, so the system is built to match
what the normaliser keeps and to spend nothing on what it deletes. Three properties of the
scorer drive the design, and §3 of the notebook prices each one on real data:

* casing is compared **as written** away from a sentence start, so `SMP` vs `smp` costs a
  full substitution;
* a reduplication hyphen makes `jalan-jalan` **one** token, so splitting it costs a
  substitution *and* an insertion — and Javanese is full of reduplication;
* a free-standing `...` is deleted while an attached `wis...` is a scored token, and
  spontaneous speech is full of disfluencies.

None of those is an acoustic problem. Together they are worth several WER points.

## Layout

```
src/lit_jv/            the library; identical code runs in the notebook and the container
  scoring.py           verbatim port of the organisers' normaliser + WER analysis helpers
  prompts.py           chat template, style tags, glossary/exemplar mining
  data.py              manifests, Jember segmentation, grouped CV by speaker/conversation
  audio.py             one 16 kHz loading path shared by training and inference
  decode.py            batched vLLM/Transformers decoding + the rescue ladder
  combine.py           ROVER, MBR, length prior (pure Python, no extra dependencies)
  postprocess.py       style lexicon and WER-gated orthography repair rules
  train_lora.py        LoRA curriculum, adapter fusion, merge-and-save, WER probe
  repro.py             seeding, determinism, environment/data/config fingerprints
  hub.py               publish checkpoints, metrics and manifests to the HF Hub
submission_src/        exactly what gets zipped: main.py + config + fitted rules
scripts/               packaging and local-test helpers
tests/                 CPU-only test suite, including a full container dry run
notebooks/             the analysis
indonesian_dev/        development set (tracked on a private repo so clones have it)
artifacts/             generated: hypotheses, adapters, merged weights, ablation tables
```

## Setup

Local development needs very little, because everything CPU-side is standard library plus
`pandas`/`jiwer`/`matplotlib`:

```bash
pip install -r requirements-dev.txt
python tests/test_pipeline.py
python tests/test_repro_and_hub.py
python tests/test_submission_dryrun.py
python tests/test_submission_resilience.py
python scripts/run_notebook_as_script.py     # executes every notebook code cell
```

On a private repo, `indonesian_dev/` (metadata + clips, ~71 MB) is tracked so a clone
on RunPod already has it. If a box still lacks `metadata.tsv`, either `git pull` or:

```bash
rsync -avP indonesian_dev/ runpod:/workspace/Lost_In_Transcription_Indonesian_Javanese_Track/indonesian_dev/
```

To publish checkpoints to the Hub, set `HF_TOKEN` and `LIT_HF_REPO` in `.env`, then:

```bash
source scripts/setup_env.sh
python scripts/push_to_hub.py                  # adapters + logs + manifest
python scripts/push_to_hub.py --upload-merged  # also the merged 1.7B weights
```

Training pushes adapters automatically when those two variables are set. The notebook
cell §10.3 and the fusion cell do the same after a merge.

Copy [`.env.example`](.env.example) to `.env` if you want Hub publishing or a non-default
data path. Then on the GPU box:

```bash
source scripts/setup_env.sh
```

That exports every `LIT_*` / `HF_*` variable, loads `.env` without clobbering secrets
already in the environment, and refuses to continue if `metadata.tsv` or `clips/` is
missing. Do not execute the file (`bash scripts/setup_env.sh`) — the exports would die
with the process.

Optional extra audio from Mozilla Data Collective (Jember, Tetelancingo Nahuatl) is
downloaded on a **development** machine only. Put `MDC_API_KEY` in `.env`, then:

```bash
python scripts/download_mdc_dataset.py --out nahuatl_dev
export LIT_NAHUATL_DIR=$PWD/nahuatl_dev
```

Do **not** put that key in `submission.zip`. The official container has no network;
test clips are already at `/code_execution/data/clips/`.

For the competition container, use its locked image — never install packages inside
the submitted archive. For a Runpod GPU pod, create a fresh **Python 3.12** environment
and install the repository's complete notebook environment before training or decoding:

```bash
python -m pip install --upgrade --ignore-installed blinker
python -m pip install --upgrade --upgrade-strategy eager -r requirements-runpod.txt
python scripts/verify_runpod.py
```

This deliberately pins `qwen-asr==0.0.6`, whose published compatibility set pins
`transformers==4.57.6` and `vllm==0.14.0`. It does not force a standalone Torch/CUDA
wheel: that is the usual source of Runpod ABI mismatches, and pip must select a build
compatible with the pod's driver. `verify_runpod.py` fails before a costly run if the
resolved versions, imports, or CUDA access are wrong. The official Qwen package metadata
is the source of truth for these pins. The first command is intentional: some Runpod Debian
images ship `blinker` without pip's `RECORD` metadata, and Qwen-ASR's Gradio dependency then
cannot upgrade it normally.

## Workflow

```bash
# 1. analysis, targets, prompts, zero-shot baseline, rule fitting  (the notebook)
jupyter lab notebooks/lost_in_transcription_id_jv.ipynb

# 2. package and verify (--model also accepts a Hub repo id; it is downloaded first,
#    because the container has no network and the weights must be inside the archive)
python scripts/pack_submission.py --model artifacts/merged-qwen3asr

# 3. rehearse the container locally
python scripts/make_local_test_data.py --out runtime/data
cp submission/submission.zip runtime/submission/
make test-submission                     # inside the runtime repository
python score.py runtime/ground_truth.csv --predicted-path runtime/submission/submission.csv
```

`pack_submission.py` verifies the archive after writing it: `main.py` at the root, the
package present, the config, fitted rules and their lexicon present, weights present, and no
`__pycache__` or checkpoint files leaked in.

## Metrics and TensorBoard

Training writes event files to `artifacts/runs/<stage>/`:

```bash
tensorboard --logdir artifacts/runs --port 6006
```

Four scalars, in descending order of usefulness:

| Scalar | What it is |
|---|---|
| `eval/wer` | 48 held-out clips decoded greedily and scored with the competition normaliser, every 200 steps |
| `eval/empty_outputs` | how many of those came back blank — the failure that costs whole clips |
| `eval/loss` | teacher-forced loss on the held-out conversation, every 50 steps |
| `train/loss` | the least informative curve in the project |

`eval/loss` and `eval/wer` disagree often enough that the probe pays for itself. Loss
rewards reproducing punctuation and casing in exactly the places the scorer deletes them,
and it barely registers a truncated hypothesis — a few tokens of loss, a whole clip of WER.
`WerProbeCallback` in `train_lora.py` is the one to trust, and it never raises: a failed
probe logs and training continues.

## Checkpoints on the Hugging Face Hub

Opt-in, and never on the container's path — it has no network. Set `LIT_HF_REPO` and
`HF_TOKEN`, then §10.3 of the notebook pushes adapters, the TensorBoard event files, the
run manifest, the fitted rules and a model card generated from the actual run:

```python
from lit_jv import hub

spec = hub.HubSpec(repo_id="you/lit-jv-qwen3asr-lora", private=True, upload_merged=True)
hub.publish_run(spec, adapters={"adapter-style": "artifacts/qwen3asr-lora/adapter-style"},
                merged_dir="artifacts/merged-qwen3asr", tensorboard_dir="artifacts/runs",
                manifest_path="artifacts/qwen3asr-lora/adapter-style/run_manifest.json")
```

Coming back the other way, `hub.resolve_checkpoint(repo, subfolder="merged")` returns a
local directory, and `pack_submission.py --model <repo-id>` calls it before zipping. The
merged 1.7B checkpoint is only uploaded when `upload_merged=True`, since it is gigabytes;
adapters are tens of megabytes and always go. Tokens are read from the environment at call
time and never written into an artefact — `tests/test_repro_and_hub.py` asserts that.

## Reproducibility

* **Metric parity.** `scoring.self_check()` asserts the behaviours the whole solution
  relies on. `tests/test_pipeline.py` runs it. If it fails, no local number is meaningful.
* **Held-out protocol.** The test set shares no speaker and no conversation with the dev
  set, so every gate uses `holdout_by_conversation` or `speaker_folds`. Random splits
  overstate every result here. `run_stage` refuses an eval split that overlaps training
  rather than reporting the flattering number.
* **Seeds.** `repro.seed_everything(seed)` pins `random`, NumPy, torch CPU and CUDA, the
  hash seed, and each dataloader worker — the last of which forks with its own RNG state
  and is the usual reason "seeded" augmentation still varies. `LIT_SEED` overrides the
  default 3407 everywhere, including the notebook's simulated-hypothesis path.
* **Fingerprints.** Every artefact is written next to a `run_manifest.json` recording the
  seed, a hash of the config, a content hash of the training data, package versions, the
  git commit (and whether the tree was dirty) and the GPU model. When two runs disagree,
  the diff between manifests is the answer.
* **The honest limit.** Identical seeds reproduce identical results on the same software
  and hardware. A different GPU, CUDA version or batch size changes floating-point
  reduction order and moves WER slightly. `LIT_DETERMINISTIC=1` (`TrainSpec.deterministic`)
  closes most of that gap at a real speed cost, and some fused attention kernels have no
  deterministic implementation, so it can refuse to run.
* **Frozen artefacts.** The glossary, style lexicon, rule set, length prior and combiner
  choice are all fitted in the notebook and written into `submission_src/`, so the
  container computes nothing it has not already been measured doing.
* **Inference determinism.** All submitted passes decode greedily (`temperature=0.0`);
  sampling appears only inside the rescue ladder, where draws are seeded and reduced by MBR.

## Rules compliance

Every test clip is decoded independently — no pseudo-labelling, no test-time adaptation,
no statistics pooled across test clips. All conditioning is fitted on the development set
and on MDC corpora, then frozen into the archive. The container needs no network. Nothing
about the test data is ever logged: `main.py` logs stage names, counts and timings only.

## Notes on the failure modes that actually decide this competition

* **Truncation.** LLM-decoder ASR ends long turns early, and half the scored words in the
  dev set sit in the longest 42% of clips. The length prior, the logprob gate and the
  split-and-stitch rescue exist for that one failure.
* **The clock.** Two hours, and a timeout scores zero. A valid `submission.csv` is written
  before the model loads, rewritten after every pass, and no pass starts unless its
  measured cost fits in the remaining budget.
* **Combiner arity.** ROVER needs three voters to have a majority. When the clock cuts a
  pass, `combiner_by_n` (measured in §9.1) selects a combiner appropriate to the number of
  passes that actually finished — including "don't combine".
