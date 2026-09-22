# Data to download

1. Official **Lost in Transcription Competition Development Data:
   Spanish-English** → `data/spanish_english/development/`.
2. **Bangor Miami Corpus** → `data/spanish_english/bangor_miami/`.
3. Official **Test Submission Format** and **Smoke Test Submission Format** →
   `submission_formats/spanish_english/`.

The Bangor corpus contains session recordings with paired CHAT transcripts and
word-level glosses. Keep the audio and CHAT files together; its loader must be
verified against the extracted files before it is admitted to training.

After MDC confirms this RunPod client is allowed, run these commands from the
repository root. The first never downloads anything; the second downloads and
extracts the corpus into the directory above:

```bash
python scripts/check_mdc_access.py
python spanish_english/download_bangor_miami.py --check-only
python spanish_english/download_bangor_miami.py --download
```
