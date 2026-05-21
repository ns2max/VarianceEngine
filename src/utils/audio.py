"""
audio.py — Low-level audio loading and feature extraction utilities.

Project: VarianceEngine

All functions operate on a single file at a time and return plain Python/NumPy
objects so they are easy to serialise and parallelise. No side effects.

Design decisions
----------------
- librosa is used for spectral and pitch features because its API surface for
  chroma, onset detection, and f0 estimation is richer than torchaudio at
  analysis time.  torchaudio is reserved for the training pipeline (Step 5+)
  where GPU throughput matters.
- All audio is loaded at its native sample rate first; resampling happens only
  when a specific analysis requires a fixed rate.
- RMS and peak are computed on the raw waveform (no A-weighting) because the
  downstream task cares about the recording level that the model will see, not
  perceptual loudness.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_audio(filepath: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a WAV file and return (waveform, sample_rate).

    Parameters
    ----------
    filepath:
        Path to the WAV file.
    sr:
        Target sample rate.  None keeps the native rate.

    Returns
    -------
    waveform : np.ndarray, shape (T,)
        Mono float32 waveform in [-1, 1].
    sample_rate : int
        Actual sample rate of the returned waveform.

    Notes
    -----
    librosa.load always returns float32 in [-1, 1] and downmixes to mono
    automatically — consistent with the dataset (which is already mono, 16-bit
    PCM at 22 050 Hz) and with MusicGen's expected input format.
    """
    waveform, sample_rate = librosa.load(filepath, sr=sr, mono=True)
    return waveform, sample_rate


# ---------------------------------------------------------------------------
# Time-domain statistics
# ---------------------------------------------------------------------------

def peak_amplitude(waveform: np.ndarray) -> float:
    """Return peak absolute amplitude of the waveform.

    Used in the audit to detect clipped recordings (peak ≈ 1.0 in a 16-bit
    WAV that was recorded hot, or exactly 1.0 after normalisation).
    """
    return float(np.max(np.abs(waveform)))


def rms_amplitude(waveform: np.ndarray) -> float:
    """Return root-mean-square amplitude.

    RMS correlates with perceived loudness better than peak for sustained
    musical content. Used to identify unusually quiet recordings (possible
    silence or near-silence) and to characterise loudness spread across
    artists.
    """
    return float(np.sqrt(np.mean(waveform ** 2)))


def duration_seconds(waveform: np.ndarray, sample_rate: int) -> float:
    """Return duration in seconds."""
    return len(waveform) / sample_rate


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

def compute_chroma(
    waveform: np.ndarray,
    sample_rate: int,
    n_chroma: int = 12,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute chromagram (chroma energy normalised per frame).

    Parameters
    ----------
    n_chroma:
        Number of chroma bins (12 = one per semitone in the octave).
    hop_length:
        STFT hop size in samples.  At 22 050 Hz with hop=512, each frame
        represents ~23 ms — sufficient temporal resolution for harmonic
        analysis of musical passages.

    Returns
    -------
    chroma : np.ndarray, shape (12, T_frames)
        Normalised chroma energy per frame.

    Notes on choice of chroma_stft vs chroma_cqt
    ---------------------------------------------
    chroma_stft is used here because the dataset contains polyphonic content
    with both chordal and melodic elements.  chroma_cqt can produce ringing
    artifacts at lower frequencies for dense chords; chroma_stft is more
    robust to polyphonic densities at the cost of slightly lower frequency
    resolution in the bass register.
    """
    return librosa.feature.chroma_stft(
        y=waveform,
        sr=sample_rate,
        n_chroma=n_chroma,
        hop_length=hop_length,
    )


def chroma_mean(chroma: np.ndarray) -> np.ndarray:
    """Time-averaged chroma vector, shape (12,).

    Represents the aggregate harmonic content of the recording as a pitch-class
    distribution.  Used as the pattern-level harmonic fingerprint.
    """
    return chroma.mean(axis=1)


def chroma_cosine_similarity(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    """Cosine similarity between two time-averaged chroma vectors.

    Range [−1, 1]; values near 1.0 indicate near-identical pitch-class content.
    Used to measure harmonic proximity between ground truth and its variations.

    Notes
    -----
    Cosine similarity is preferred over Euclidean distance here because chroma
    vectors are non-negative and their magnitude encodes energy rather than
    harmonic identity.  Normalising to unit length removes the energy confound,
    leaving only the distribution shape (i.e. which pitch classes are active).
    """
    a = chroma_a / (np.linalg.norm(chroma_a) + 1e-8)
    b = chroma_b / (np.linalg.norm(chroma_b) + 1e-8)
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Pitch estimation
# ---------------------------------------------------------------------------

def estimate_pitch_range(
    waveform: np.ndarray,
    sample_rate: int,
    fmin: float = librosa.note_to_hz("C2"),
    fmax: float = librosa.note_to_hz("C8"),
    threshold: float = 0.2,
) -> dict:
    """Estimate the pitch range of a recording using pYIN f0 tracking.

    pYIN (Mauch & Dixon, 2014) is a probabilistic extension of YIN that
    produces voiced/unvoiced decisions alongside f0 estimates.  Only voiced
    frames (confidence > threshold) are used for range estimation, which
    prevents silence and noise frames from inflating the apparent range.

    Parameters
    ----------
    fmin, fmax:
        Search range for f0.  C2 (65 Hz) to C8 (4186 Hz) covers the full
        practical range of orchestral instruments and voice.
    threshold:
        Minimum voicing probability to include a frame in range estimation.

    Returns
    -------
    dict with keys:
        f0_min_hz    : float  — minimum voiced f0 in Hz
        f0_max_hz    : float  — maximum voiced f0 in Hz
        f0_range_st  : float  — range in semitones (log2 ratio × 12)
        voiced_ratio : float  — proportion of voiced frames

    Notes
    -----
    pYIN is chosen over piptrack (harmonic peak picking) because piptrack
    frequently hallucinates f0 candidates in polyphonic audio.  pYIN's
    probabilistic voicing model is more robust, though it still tracks only
    the dominant pitch — in dense chords, this corresponds to the highest
    audible partial, not the root.  The pitch range estimate is therefore
    approximate for polyphonic content; it serves as a coarse characterisation
    tool rather than a precise harmonic analysis.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        waveform,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
    )

    voiced_f0 = f0[voiced_flag & (voiced_probs > threshold)]

    if len(voiced_f0) == 0:
        return {
            "f0_min_hz": None,
            "f0_max_hz": None,
            "f0_range_st": None,
            "voiced_ratio": 0.0,
        }

    f0_min = float(np.min(voiced_f0))
    f0_max = float(np.max(voiced_f0))
    range_st = float(12 * np.log2(f0_max / f0_min)) if f0_min > 0 else None

    return {
        "f0_min_hz": f0_min,
        "f0_max_hz": f0_max,
        "f0_range_st": range_st,
        "voiced_ratio": float(voiced_flag.mean()),
    }
