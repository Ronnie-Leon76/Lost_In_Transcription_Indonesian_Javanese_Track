"""Audio loading, shared by training and inference.

One function, used everywhere, because a level or resampling mismatch between
training and inference is a silent accuracy leak. The competition clips are
48 kHz mono MP3; the Jember sessions are 44.1 kHz. Everything is resampled to
16 kHz mono and peak-normalised, so the model never sees a systematic loudness
difference between corpora.
"""

from __future__ import annotations

from pathlib import Path

SAMPLE_RATE = 16000
PEAK_TARGET = 0.95


def load_audio_16k(
    path: str | Path,
    peak_normalize: bool = True,
    sample_rate: int = SAMPLE_RATE,
):
    """Return a mono float32 waveform at ``sample_rate``.

    Tries soundfile first (fast, and handles the 48 kHz MP3s in this
    competition) and falls back to librosa, which is more tolerant of odd
    containers. Both are present in the competition runtime.
    """
    import numpy as np

    path = str(path)
    wav = None
    sr = sample_rate
    try:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        wav = None
    if wav is None:
        import librosa

        wav, sr = librosa.load(path, sr=None, mono=False)
        wav = np.asarray(wav, dtype=np.float32)

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        # soundfile gives (samples, channels); librosa gives (channels, samples).
        if wav.shape[0] <= 8 and wav.shape[1] > wav.shape[0]:
            wav = wav.T
        wav = wav.mean(axis=-1).astype(np.float32)

    if int(sr) != int(sample_rate):
        import librosa

        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sample_rate)).astype("float32")

    if peak_normalize and wav.size:
        peak = float(np.max(np.abs(wav)))
        if peak > 0:
            wav = (wav / peak * PEAK_TARGET).astype("float32")
    return np.ascontiguousarray(wav, dtype="float32")


def split_at_quietest(wav, sample_rate: int = SAMPLE_RATE, search_fraction: float = 0.30):
    """Split a waveform near its midpoint at the quietest 100 ms window.

    Used by the re-decode ladder: when a long clip comes back truncated, cutting
    it in a pause and decoding both halves recovers the tail. Searching only the
    middle ``search_fraction`` keeps both halves substantial.
    """
    import numpy as np

    n = len(wav)
    if n < sample_rate:
        return [wav]
    mid = n // 2
    half = int(n * search_fraction / 2)
    lo, hi = max(0, mid - half), min(n, mid + half)
    window = max(8, int(0.1 * sample_rate))
    seg = np.abs(wav[lo:hi])
    if len(seg) <= window:
        cut = mid
    else:
        energy = np.convolve(seg, np.ones(window, dtype="float32") / window, mode="valid")
        cut = lo + int(np.argmin(energy)) + window // 2
    cut = int(min(max(cut, sample_rate // 2), n - sample_rate // 2))
    return [wav[:cut], wav[cut:]]
