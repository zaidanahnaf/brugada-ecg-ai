"""
src/preprocessing.py

Signal preprocessing pipeline for Brugada ECG CNN branch.

Pipeline (applied per fold, fit on training set only):
  1. Baseline wander removal via median filter (~610 ms kernel)
  2. Optional mild Butterworth low-pass filter (40 Hz, order 4, zero-phase)
  3. Per-lead amplitude normalization (zero mean, unit std)
     — statistics computed on training fold only
     — applied identically to validation fold

CRITICAL RULES:
  - Do NOT use high-pass filter above 0.5 Hz (ST morphology distortion)
  - Normalization stats MUST be fit on training fold only
  - Stats are saved to disk for reproducibility and audit
"""

import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

from src.cnn_utils import ensure_dirs, get_logger

logger = get_logger("preprocessing")


# =============================================================================
# STEP 1 — BASELINE WANDER REMOVAL
# =============================================================================

def remove_baseline_wander(
    signals: np.ndarray,
    kernel_samples: int = 61,
) -> np.ndarray:
    """
    Remove baseline wander using a median filter.

    Args:
        signals:        shape (N, C, T) or (C, T) — float32
        kernel_samples: median filter kernel size in samples
                        ~61 samples at 100 Hz ≈ 610 ms (close to 600 ms target)
                        Must be odd for symmetric filtering.

    Returns:
        Baseline-corrected signals, same shape as input.

    Method: subtract median-filtered version from original.
    This preserves ST segment morphology (unlike HP filters).
    """
    if kernel_samples % 2 == 0:
        kernel_samples += 1   # enforce odd

    single = signals.ndim == 2
    if single:
        signals = signals[np.newaxis]   # (1, C, T)

    result = np.empty_like(signals)
    for i in range(signals.shape[0]):
        for c in range(signals.shape[1]):
            baseline = median_filter(signals[i, c], size=kernel_samples, mode="nearest")
            result[i, c] = signals[i, c] - baseline

    if single:
        result = result[0]
    return result


# =============================================================================
# STEP 2 — LOW-PASS FILTER  (optional)
# =============================================================================

def apply_lowpass_filter(
    signals: np.ndarray,
    cutoff_hz: float = 40.0,
    order: int = 4,
    fs: float = 100.0,
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.

    Args:
        signals:    shape (N, C, T) or (C, T)
        cutoff_hz:  cutoff frequency in Hz (40 Hz recommended)
        order:      filter order (4 recommended)
        fs:         sampling rate in Hz

    Returns:
        Filtered signals, same shape as input.

    NOTE: Uses filtfilt (zero-phase) — no phase distortion on ST segment.
    """
    nyq  = fs / 2.0
    norm = cutoff_hz / nyq
    if norm >= 1.0:
        logger.warning(f"LP cutoff {cutoff_hz} Hz >= Nyquist {nyq} Hz — skipping LP filter")
        return signals

    b, a = butter(order, norm, btype="low", analog=False)

    single = signals.ndim == 2
    if single:
        signals = signals[np.newaxis]

    result = np.empty_like(signals)
    for i in range(signals.shape[0]):
        for c in range(signals.shape[1]):
            result[i, c] = filtfilt(b, a, signals[i, c])

    if single:
        result = result[0]
    return result.astype(np.float32)


# =============================================================================
# STEP 3 — AMPLITUDE NORMALIZATION
# =============================================================================

class PerLeadNormalizer:
    """
    Normalize each lead to zero mean, unit std.

    CRITICAL: .fit() must be called ONLY on training fold signals.
              .transform() is then applied to both train and val.

    Stats saved/loaded as .npz for audit trail.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps   = eps
        self.mean_: Optional[np.ndarray] = None   # shape (C,)
        self.std_:  Optional[np.ndarray] = None   # shape (C,)

    def fit(self, signals: np.ndarray) -> "PerLeadNormalizer":
        """
        Fit mean and std from training signals.
        signals: shape (N, C, T)
        """
        # Compute mean/std per channel across all samples and time steps
        # Reshape to (N*T, C) equivalent: mean over axes 0 and 2
        self.mean_ = signals.mean(axis=(0, 2))   # (C,)
        self.std_  = signals.std(axis=(0, 2))    # (C,)

        # Guard against zero-std leads (flat signal)
        zero_std = self.std_ < self.eps
        if zero_std.any():
            logger.warning(f"Zero-std leads detected at indices: {np.where(zero_std)[0].tolist()}")
            self.std_[zero_std] = 1.0

        logger.info(f"Normalizer fit: mean range [{self.mean_.min():.4f}, {self.mean_.max():.4f}], "
                    f"std range [{self.std_.min():.4f}, {self.std_.max():.4f}]")
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply normalization.
        signals: shape (N, C, T)
        Returns: float32 array, same shape.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("PerLeadNormalizer must be fit before transform")
        # Broadcast: mean/std are (C,) → (1, C, 1)
        mean = self.mean_[np.newaxis, :, np.newaxis]
        std  = self.std_[np.newaxis, :, np.newaxis]
        return ((signals - mean) / std).astype(np.float32)

    def fit_transform(self, signals: np.ndarray) -> np.ndarray:
        return self.fit(signals).transform(signals)

    def save(self, path: str) -> None:
        ensure_dirs(os.path.dirname(path))
        np.savez(path, mean=self.mean_, std=self.std_)
        logger.info(f"Normalizer stats saved: {path}")

    @classmethod
    def load(cls, path: str) -> "PerLeadNormalizer":
        data = np.load(path + ".npz" if not path.endswith(".npz") else path)
        obj = cls()
        obj.mean_ = data["mean"]
        obj.std_  = data["std"]
        return obj


# =============================================================================
# FULL PIPELINE
# =============================================================================

def preprocess_fold(
    train_signals: np.ndarray,
    val_signals: np.ndarray,
    kernel_samples: int = 61,
    apply_lp: bool = True,
    lp_cutoff: float = 40.0,
    lp_order: int = 4,
    fs: float = 100.0,
    stats_save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, PerLeadNormalizer]:
    """
    Run the full preprocessing pipeline for one fold.

    Steps:
      1. Baseline wander removal (both splits, no fitting needed)
      2. Optional low-pass filter (both splits, no fitting needed)
      3. Per-lead normalization (fit on train, apply to both)

    Args:
        train_signals:   (N_train, C, T) raw signals for training fold
        val_signals:     (N_val, C, T) raw signals for validation fold
        stats_save_path: if provided, save normalizer stats here (e.g. "preprocessing/fold_stats/fold_0")

    Returns:
        train_proc:  preprocessed training signals
        val_proc:    preprocessed validation signals
        normalizer:  fitted PerLeadNormalizer (for audit / inference)
    """
    logger.info("Preprocessing: Step 1 — baseline wander removal")
    train_proc = remove_baseline_wander(train_signals, kernel_samples=kernel_samples)
    val_proc   = remove_baseline_wander(val_signals,   kernel_samples=kernel_samples)

    if apply_lp:
        logger.info(f"Preprocessing: Step 2 — low-pass filter ({lp_cutoff} Hz, order {lp_order})")
        train_proc = apply_lowpass_filter(train_proc, cutoff_hz=lp_cutoff, order=lp_order, fs=fs)
        val_proc   = apply_lowpass_filter(val_proc,   cutoff_hz=lp_cutoff, order=lp_order, fs=fs)
    else:
        logger.info("Preprocessing: Step 2 — low-pass filter SKIPPED (APPLY_LP_FILTER=False)")

    logger.info("Preprocessing: Step 3 — per-lead normalization (fit on train only)")
    normalizer = PerLeadNormalizer()
    train_proc = normalizer.fit_transform(train_proc)
    val_proc   = normalizer.transform(val_proc)

    if stats_save_path:
        normalizer.save(stats_save_path)

    return train_proc, val_proc, normalizer