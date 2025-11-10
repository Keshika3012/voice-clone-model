#!/usr/bin/env python3
"""
Audio similarity scoring utilities.

Provides PESQ/STOI with preprocessing (mono 16k resample, silence trim,
loudness normalization) and optional cross-correlation alignment to
reduce timing/loudness biases when comparing reference vs generated audio.
"""
from typing import Tuple
import os
import numpy as np
import librosa
from scipy.signal import correlate

try:
    from pesq import pesq  # expects 8k (nb) or 16k (wb)
except Exception:
    pesq = None

try:
    from pystoi import stoi  # works 10–20 kHz, commonly 16 kHz
except Exception:
    stoi = None


def load_mono_resampled(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32 in [-1,1], resampled to target_sr."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    return y, target_sr


def trim_silence(y: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if len(yt) > 0 else y


def normalize_rms(y: np.ndarray, target_dbfs: float = -26.0) -> np.ndarray:
    eps = 1e-9
    rms = np.sqrt(np.mean(np.square(y)) + eps)
    rms_db = 20.0 * np.log10(rms + eps)
    gain_db = target_dbfs - rms_db
    gain = 10.0 ** (gain_db / 20.0)
    y_norm = np.clip(y * gain, -1.0, 1.0)
    return y_norm.astype(np.float32)


def align_by_xcorr(ref: np.ndarray, test: np.ndarray, sr: int, max_shift_sec: float = 0.5):
    max_lag = int(max_shift_sec * sr)
    n = min(len(ref), len(test))
    ref_c = ref[:n]
    test_c = test[:n]

    pad = max_lag
    ref_pad = np.pad(ref_c, (pad, pad), mode='constant')
    test_pad = np.pad(test_c, (pad, pad), mode='constant')

    corr = correlate(test_pad, ref_pad, mode='valid')
    lag_range = np.arange(-max_lag, max_lag + 1)
    center = len(corr) // 2
    window = corr[center - max_lag : center + max_lag + 1]
    best_idx = int(np.argmax(window))
    best_lag = lag_range[best_idx]

    if best_lag > 0:
        test_aligned = test_c[best_lag:]
        ref_aligned = ref_c[:len(test_aligned)]
    elif best_lag < 0:
        ref_aligned = ref_c[-best_lag:]
        test_aligned = test_c[:len(ref_aligned)]
    else:
        ref_aligned, test_aligned = ref_c, test_c

    min_len = min(len(ref_aligned), len(test_aligned))
    return ref_aligned[:min_len], test_aligned[:min_len]


def safe_pesq(sr: int, ref: np.ndarray, deg: np.ndarray) -> float:
    if pesq is None:
        return 0.0
    try:
        return float(pesq(sr, ref, deg, 'wb'))
    except Exception:
        return 0.0


def safe_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    if stoi is None:
        return 0.0
    try:
        return float(stoi(ref, deg, sr, extended=False))
    except Exception:
        return 0.0


def compare_audio(ref_path: str, test_path: str, target_sr: int = 16000, align: bool = True):
    if not os.path.exists(ref_path) or not os.path.exists(test_path):
        raise FileNotFoundError("One or both audio files do not exist.")

    def preprocess(p: str):
        y, _ = load_mono_resampled(p, target_sr=target_sr)
        y = trim_silence(y, top_db=40.0)
        y = normalize_rms(y, target_dbfs=-26.0)
        return y

    ref = preprocess(ref_path)
    test = preprocess(test_path)

    if align:
        ref, test = align_by_xcorr(ref, test, sr=target_sr, max_shift_sec=0.5)

    min_len = min(len(ref), len(test))
    ref = ref[:min_len]
    test = test[:min_len]

    pesq_score = safe_pesq(target_sr, ref, test)
    stoi_score = safe_stoi(ref, test, target_sr)
    pesq_norm = 0.0 if pesq_score == 0.0 else (pesq_score - 1.0) / 3.5
    combined = float(np.mean([pesq_norm, stoi_score]))

    return {
        'pesq': pesq_score,
        'pesq_norm': pesq_norm,
        'stoi': stoi_score,
        'combined': combined,
        'seconds': len(ref)/target_sr,
        'aligned': align,
    }
