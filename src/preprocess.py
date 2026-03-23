"""
preprocess.py — Signal preprocessing pipeline.

Order of operations per signal:
  1. Baseline wander removal (median filter)
  2. Optional low-pass filter (Butterworth 40 Hz)
  3. Per-lead amplitude normalization (zero-mean, unit-std)
     -> statistics MUST be computed on TRAINING fold only, then applied to val.

Key constraint: do NOT distort ST morphology / J-point.
  - No high-pass filter > 0.5 Hz
  - Only gentle baseline removal via median filter
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import median_filter

from src.config import (
    APPLY_LOWPASS,
    BASELINE_KERNEL_SAMPLES,
    LOWPASS_CUTOFF_HZ,
    LOWPASS_ORDER,
    N_LEADS,
    N_SAMPLES,
    SAMPLING_RATE,
)

logger = logging.getLogger(__name__)


# ============================================================
# STEP 1 — BASELINE WANDER REMOVAL
# ============================================================

def remove_baseline_wander(
    ecg: np.ndarray,
    kernel_samples: int = BASELINE_KERNEL_SAMPLES,
) -> np.ndarray:
    """
    Remove baseline wander using a median filter per lead.

    Args:
        ecg: (n_leads, n_samples) float32
        kernel_samples: must be odd. Default: 61 samples (≈600 ms at 100 Hz)

    Returns:
        ecg with baseline removed, same shape.

    Note:
        Median filter at ~600 ms does NOT distort ST morphology because
        it operates at a frequency well below ECG signal components.
        This is preferred over high-pass filtering which distorts J-point.
    """
    assert ecg.ndim == 2
    if kernel_samples % 2 == 0:
        kernel_samples += 1   # enforce odd

    cleaned = np.empty_like(ecg)
    for lead_idx in range(ecg.shape[0]):
        baseline = median_filter(ecg[lead_idx], size=kernel_samples)
        cleaned[lead_idx] = ecg[lead_idx] - baseline
    return cleaned


# ============================================================
# STEP 2 — OPTIONAL LOW-PASS FILTER
# ============================================================

def apply_lowpass_filter(
    ecg: np.ndarray,
    cutoff_hz: float = LOWPASS_CUTOFF_HZ,
    order: int = LOWPASS_ORDER,
    fs: float = SAMPLING_RATE,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth low-pass filter per lead.

    Args:
        ecg: (n_leads, n_samples)
        cutoff_hz: cutoff frequency (default 40 Hz)
        order: filter order (default 4)
        fs: sampling rate

    Returns:
        Filtered ECG, same shape.
    """
    nyq = 0.5 * fs
    b, a = sp_signal.butter(order, cutoff_hz / nyq, btype="low")
    filtered = np.empty_like(ecg)
    for lead_idx in range(ecg.shape[0]):
        filtered[lead_idx] = sp_signal.filtfilt(b, a, ecg[lead_idx])
    return filtered


# ============================================================
# STEP 3 — PER-LEAD NORMALIZATION
# ============================================================

class LeadNormalizer:
    """
    Per-lead zero-mean, unit-std normalization.

    LEAKAGE RULE:
        - Call fit() on training signals ONLY.
        - Call transform() on both train and val.
        - Never call fit() on validation data.
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None   # (n_leads,)
        self.std_:  Optional[np.ndarray] = None   # (n_leads,)
        self._fitted = False

    def fit(self, signals: List[np.ndarray]) -> "LeadNormalizer":
        """
        Compute per-lead mean and std from a list of training signals.

        Args:
            signals: list of (n_leads, n_samples) arrays
        """
        # Stack all training signals: (N, n_leads, n_samples)
        stacked = np.stack([s for s in signals if s is not None], axis=0)
        # Compute statistics over samples and subjects dimensions
        # mean/std shape: (n_leads,)
        self.mean_ = stacked.mean(axis=(0, 2))        # mean over N and timesteps
        self.std_  = stacked.std(axis=(0, 2))
        # Avoid division by zero for flat leads
        self.std_  = np.where(self.std_ < 1e-8, 1.0, self.std_)
        self._fitted = True
        logger.info(
            f"LeadNormalizer fitted on {len(signals)} signals. "
            f"Per-lead std range: [{self.std_.min():.4f}, {self.std_.max():.4f}]"
        )
        return self

    def transform(self, ecg: np.ndarray) -> np.ndarray:
        """
        Apply normalization to a single ECG.

        Args:
            ecg: (n_leads, n_samples)

        Returns:
            Normalized ECG, same shape.
        """
        if not self._fitted:
            raise RuntimeError("LeadNormalizer.fit() must be called before transform().")
        normalized = (ecg - self.mean_[:, None]) / self.std_[:, None]
        return normalized.astype(np.float32)

    def fit_transform(self, signals: List[np.ndarray]) -> List[np.ndarray]:
        """Convenience: fit then transform a list of training signals."""
        self.fit(signals)
        return [self.transform(s) if s is not None else None for s in signals]

    def save(self, path: str) -> None:
        """Save normalizer stats to .npz for reproducibility."""
        np.savez(path, mean=self.mean_, std=self.std_)
        logger.info(f"LeadNormalizer stats saved to {path}")

    def load(self, path: str) -> "LeadNormalizer":
        """Load normalizer stats from .npz."""
        data = np.load(path)
        self.mean_  = data["mean"]
        self.std_   = data["std"]
        self._fitted = True
        return self


# ============================================================
# FULL PREPROCESSING PIPELINE
# ============================================================

def preprocess_signal(
    ecg: np.ndarray,
    normalizer: Optional[LeadNormalizer] = None,
    apply_lowpass: bool = APPLY_LOWPASS,
) -> np.ndarray:
    """
    Apply full preprocessing pipeline to a single ECG signal.

    Steps:
        1. Baseline wander removal (median filter)
        2. Optional low-pass filter
        3. Normalization (if normalizer provided)

    Args:
        ecg: (n_leads, n_samples) raw signal
        normalizer: fitted LeadNormalizer. If None, no normalization is applied.
        apply_lowpass: whether to apply low-pass filter

    Returns:
        Preprocessed ECG, same shape.
    """
    ecg = remove_baseline_wander(ecg)
    if apply_lowpass:
        ecg = apply_lowpass_filter(ecg)
    if normalizer is not None:
        ecg = normalizer.transform(ecg)
    return ecg


def build_preprocessed_dataset(
    train_signals: Dict[str, Optional[np.ndarray]],
    val_signals:   Dict[str, Optional[np.ndarray]],
    fold_id: int,
    save_normalizer_path: Optional[str] = None,
    apply_lowpass: bool = APPLY_LOWPASS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], LeadNormalizer]:
    """
    Fit normalizer on training fold only, then preprocess both splits.

    Args:
        train_signals: {patient_id -> ecg | None}
        val_signals:   {patient_id -> ecg | None}
        fold_id: used for logging only
        save_normalizer_path: if set, saves .npz to this path
        apply_lowpass: whether to apply low-pass filter

    Returns:
        (train_preprocessed, val_preprocessed, normalizer)
        Both dicts: {patient_id -> preprocessed_ecg}
        Failed subjects (None) are excluded from dict.
    """
    # ---------- Step 1: baseline removal + filtering (no stats needed) ----------
    def _partial_preprocess(signals_dict):
        out = {}
        for pid, ecg in signals_dict.items():
            if ecg is None:
                continue
            ecg = remove_baseline_wander(ecg)
            if apply_lowpass:
                ecg = apply_lowpass_filter(ecg)
            out[pid] = ecg
        return out

    train_partial = _partial_preprocess(train_signals)
    val_partial   = _partial_preprocess(val_signals)

    # ---------- Step 2: fit normalizer on TRAINING only ----------
    normalizer = LeadNormalizer()
    normalizer.fit(list(train_partial.values()))

    if save_normalizer_path is not None:
        normalizer.save(save_normalizer_path)

    # ---------- Step 3: apply normalization to both splits ----------
    train_preprocessed = {pid: normalizer.transform(ecg) for pid, ecg in train_partial.items()}
    val_preprocessed   = {pid: normalizer.transform(ecg) for pid, ecg in val_partial.items()}

    logger.info(
        f"[Fold {fold_id}] Preprocessing complete. "
        f"Train: {len(train_preprocessed)}, Val: {len(val_preprocessed)}"
    )
    return train_preprocessed, val_preprocessed, normalizer


# ============================================================
# QUICK SANITY (run as __main__)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic ECG for testing
    np.random.seed(0)
    n_leads, n_samples = 12, 1200
    fake_ecg = np.random.randn(n_leads, n_samples).astype(np.float32)
    # Add artificial baseline wander
    t = np.linspace(0, 12, n_samples)
    for i in range(n_leads):
        fake_ecg[i] += 0.3 * np.sin(2 * np.pi * 0.2 * t)   # 0.2 Hz wander

    print(f"Raw ECG: mean={fake_ecg.mean():.3f}, std={fake_ecg.std():.3f}")

    cleaned = remove_baseline_wander(fake_ecg)
    print(f"After baseline removal: mean={cleaned.mean():.4f}, std={cleaned.std():.3f}")

    filtered = apply_lowpass_filter(cleaned)
    print(f"After low-pass: mean={filtered.mean():.4f}, std={filtered.std():.3f}")

    norm = LeadNormalizer()
    norm.fit([filtered, filtered])
    normed = norm.transform(filtered)
    print(f"After normalization: mean={normed.mean():.4f}, std={normed.std():.3f}")
    print("Preprocessing pipeline OK.")