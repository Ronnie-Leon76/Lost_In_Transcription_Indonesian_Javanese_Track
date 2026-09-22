# Track data layout

Download each official development archive and external corpus manually, then
extract it into the matching directory below. Do not put archives, audio, or
official submission-format files into Git.

| Track | Official development data | External training corpus |
|---|---|---|
| Indonesian-Javanese | `data/indonesian_javanese/development/` | `data/indonesian_javanese/jember/` |
| Spanish-English | `data/spanish_english/development/` | `data/spanish_english/bangor_miami/` |
| Spanish-Nahuatl | `data/spanish_nahuatl/development/` | `data/spanish_nahuatl/tetelancingo/` |

The development directory must contain `metadata.tsv` and `clips/` after
extraction. Preserve external corpus metadata next to its audio. The notebooks
validate the layout before training and never treat external data as a test set.
