# Spanish–Nahuatl: clean RunPod path

This directory is a dedicated solution for the Spanish–Nahuatl track. Do not
run the Indonesian–Javanese notebook or package for this competition.

## Data

1. Download the competition development data into `data/spanish_nahuatl/development/` so it contains
   `metadata.tsv` and `clips/`.
2. On RunPod, set `MDC_API_KEY` as a secret (never in a notebook or zip) and run:

   ```bash
   python spanish_nahuatl/download_tetelancingo.py --download
   ```

3. Open `spanish_nahuatl/lost_in_transcription_sp_nh.ipynb`, set its two
   directory paths, and run it in order. `runpod_cells.py` remains the
   notebook-style source companion.

The Tetelancingo loader trains on its `sentence` column. That is the normalized
Nahuatl transcript; it deliberately does not train against the Spanish
translation or the original-orthography column.

## Selection rule

Choose a model by WER on the speaker- and conversation-disjoint competition-dev holdout,
not by training loss, a random clip split, or Tetelancingo's own test split.
The test set contains speakers and conversations absent from the development
data, so this is the closest local proxy for the leaderboard.

## Submission

After selecting the adapter, merge it, then package it with the glossary
emitted by that same selected run:

```bash
python spanish_nahuatl/package_submission.py \
  --model artifacts/spnh/merged \
  --glossary artifacts/spnh/glossary.json
```

The builder emits `artifacts/spnh/submission.zip`. Inspect it before upload:

```bash
unzip -l artifacts/spnh/submission.zip | head
```

It must show `main.py` at the archive root. The resulting runtime uses the
Spanish-Nahuatl prompt, forces the Qwen ASR prefix to `Spanish`, and writes the
required `submission/submission.csv` without any network access.
