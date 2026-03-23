# src/preprocessing.py

import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from typing import Optional
from src.config import CFG


def preprocess_signal(
    signal: np.ndarray,
    fs: int = CFG.fs,
    apply_baseline_removal: bool = CFG.apply_baseline_removal,
    baseline_method: str = CFG.baseline_removal_method,
    apply_bandpass: bool = CFG.apply_bandpass,
    apply_notch: bool = CFG.apply_notch
) -> np.ndarray:
    """
    Apply leakage-safe preprocessing to a raw ECG signal.

    All operations are signal-local (no cross-subject statistics).
    ST morphology is deliberately preserved — NO high-pass above 0.5 Hz.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples, n_leads)

    Returns
    -------
    np.ndarray, shape (n_samples, n_leads), preprocessed
    """
    out = signal.copy().astype(np.float64)

    if apply_baseline_removal:
        out = _remove_baseline(out, fs, method=baseline_method)

    if apply_bandpass:
        # ⚠️  WARNING: bandpass must NOT cut ST-relevant low frequencies
        # Only use if baseline_removal alone is insufficient
        out = _bandpass_filter(out, fs,
                               low=CFG.bandpass_low_hz,
                               high=CFG.bandpass_high_hz)

    if apply_notch:
        out = _notch_filter(out, fs)

    return out


def _remove_baseline(
    signal: np.ndarray,
    fs: int,
    method: str = 'median_filter'
) -> np.ndarray:
    """
    Remove baseline wander without distorting ST segment.

    Method: 'median_filter' — safe for ST preservation
    Kernel size must be > 1 full QRS complex (~600ms at 100Hz = 60 samples)
    Use 600ms kernel -> 61 samples (must be odd)
    """
    if method == 'median_filter':
        kernel_samples = int(CFG.median_filter_kernel_ms * fs / 1000)
        if kernel_samples % 2 == 0:
            kernel_samples += 1           # Must be odd for medfilt

        out = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            baseline = medfilt(signal[:, ch], kernel_size=kernel_samples)
            out[:, ch] = signal[:, ch] - baseline
        return out

    elif method == 'wavelet':
        # Wavelet-based baseline: safer for short records
        # Requires pywavelets
        try:
            import pywt
            out = np.zeros_like(signal)
            for ch in range(signal.shape[1]):
                coeffs = pywt.wavedec(signal[:, ch], 'db4', level=8)
                # Zero out approximation (lowest frequency baseline)
                coeffs[0] = np.zeros_like(coeffs[0])
                out[:, ch] = pywt.waverec(coeffs, 'db4')[:len(signal)]
            return out
        except ImportError:
            # Fallback to median filter silently
            return _remove_baseline(signal, fs, method='median_filter')

    else:
        raise ValueError(f"Unknown baseline method: {method}")


def _bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4
) -> np.ndarray:
    """Butterworth bandpass — use with caution to preserve ST."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    out = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        out[:, ch] = filtfilt(b, a, signal[:, ch])
    return out


def _notch_filter(
    signal: np.ndarray,
    fs: int,
    freq: float = 50.0,
    Q: float = 30.0
) -> np.ndarray:
    """IIR notch filter for powerline interference."""
    from scipy.signal import iirnotch
    b, a = iirnotch(freq / (fs / 2), Q)
    out = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        out[:, ch] = filtfilt(b, a, signal[:, ch])
    return out