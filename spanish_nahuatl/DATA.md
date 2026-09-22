# Data to download

1. Official **Lost in Transcription Competition Development Data:
   Spanish-Nahuatl** → `data/spanish_nahuatl/development/`.
2. **Tetelancingo Nahuatl Corpus** → `data/spanish_nahuatl/tetelancingo/`.
3. Official **Submission Format** and **Smoke Test Submission Format** →
   `submission_formats/spanish_nahuatl/`.

The external corpus has a predefined train/test split. Use only its `train`
portion for fitting; reserve its test split as a one-time transfer check rather
than a tuning target.

Use the official MDC SDK after accepting the corpus terms in the MDC web UI:

```bash
python spanish_nahuatl/download_tetelancingo.py --check-only
python spanish_nahuatl/download_tetelancingo.py --download
```
