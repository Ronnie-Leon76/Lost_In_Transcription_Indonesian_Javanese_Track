"""Lost In Transcription — Indonesian-Javanese track.

The modules here are shared by the research notebook and by the packaged
``main.py`` that runs inside the competition container. Keeping one copy of the
prompt builder, the post-processor and the scorer is what guarantees that the
leaderboard system is the system we validated.

Import cost matters: ``main.py`` runs under a two-hour wall clock, so nothing in
this package imports torch, vllm or transformers at module import time.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "audio",
    "combine",
    "data",
    "decode",
    "hub",
    "postprocess",
    "prompts",
    "repro",
    "scoring",
    "train_lora",
]
