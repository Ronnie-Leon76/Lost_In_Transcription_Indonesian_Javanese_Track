# Three-track operating guide

Each Lost in Transcription language pair is a separate competition entry.
Never mix development clips, external corpora, fitted glossaries, adapters,
TensorBoard runs, merged weights, or submission archives between rows below.

| Track | Notebook | Development directory | Artifact root | Hub environment variable |
|---|---|---|---|---|
| Indonesian–Javanese | `indonesian_javanese/lost_in_transcription_id_jv.ipynb` | `data/indonesian_javanese/development/` | `artifacts/` | `LIT_HF_REPO` |
| Spanish–Nahuatl | `spanish_nahuatl/lost_in_transcription_sp_nh.ipynb` | `data/spanish_nahuatl/development/` | `artifacts/spnh/` | `SPNH_HF_REPO` |
| Spanish–English | `spanish_english/lost_in_transcription_sp_en.ipynb` | `data/spanish_english/development/` | `artifacts/spen/` | `SPEN_HF_REPO` |

The Jember corpus belongs only to Indonesian–Javanese. Tetelancingo belongs
only to Spanish–Nahuatl. Spanish–English begins with official development data;
add only Spanish–English external data after validating its license and its
effect on a speaker- and conversation-disjoint development split.

See `data/README.md` and each track's `DATA.md` for the exact local destination
of every official download, external corpus, and official submission-format CSV.
