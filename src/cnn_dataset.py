"""
src/dataset.py

PyTorch Dataset for Brugada ECG signals.
Includes safe augmentation (training only — ST-preserving).
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# AUGMENTATION
# =============================================================================

def augment_signal(
    signal: np.ndarray,
    noise_std: float = 0.005,
    amp_min: float = 0.95,
    amp_max: float = 1.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    ST-preserving augmentation.
    Applied ONLY during training — never during validation/inference.

    Allowed:
      - Small Gaussian noise (noise_std relative to unit-normalized signal)
      - Small amplitude scaling in [amp_min, amp_max]

    Forbidden (NOT implemented here):
      - Time warping
      - Lead permutation
      - Baseline shift
      - Any transformation that distorts J-point or ST morphology

    Args:
        signal: (C, T) float32 — already preprocessed and normalized
        rng:    optional numpy random Generator for seeded reproducibility

    Returns:
        Augmented signal (C, T) float32
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Gaussian noise
    noise = rng.normal(0, noise_std, signal.shape).astype(np.float32)
    signal = signal + noise

    # 2. Amplitude scaling (per-sample scalar, same scale for all leads)
    scale = rng.uniform(amp_min, amp_max)
    signal = signal * scale

    return signal


# =============================================================================
# DATASET
# =============================================================================

class BrugadaECGDataset(Dataset):
    """
    PyTorch Dataset for preprocessed ECG signals.

    Args:
        signals:     np.ndarray (N, C, T) — preprocessed, normalized
        labels:      np.ndarray (N,) — 0/1
        patient_ids: list of str, length N
        augment:     if True, apply ST-preserving augmentation
        seed:        for reproducible augmentation (per-worker RNG)
        noise_std:   Gaussian noise std
        amp_min/max: amplitude scaling range
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        patient_ids: List[str],
        augment: bool = False,
        seed: int = 42,
        noise_std: float = 0.005,
        amp_min: float = 0.95,
        amp_max: float = 1.05,
    ):
        assert len(signals) == len(labels) == len(patient_ids), \
            "signals, labels, patient_ids must have the same length"

        self.signals     = signals.astype(np.float32)
        self.labels      = labels.astype(np.float32)
        self.patient_ids = [str(p) for p in patient_ids]
        self.augment     = augment
        self.seed        = seed
        self.noise_std   = noise_std
        self.amp_min     = amp_min
        self.amp_max     = amp_max

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx].copy()   # (C, T)
        label  = self.labels[idx]

        if self.augment:
            # Per-item seed = global seed + index (prevents identical augmentation)
            rng    = np.random.default_rng(self.seed + idx)
            signal = augment_signal(
                signal,
                noise_std=self.noise_std,
                amp_min=self.amp_min,
                amp_max=self.amp_max,
                rng=rng,
            )

        return (
            torch.from_numpy(signal),            # (C, T)
            torch.tensor(label, dtype=torch.float32),
        )


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def make_dataloader(
    signals: np.ndarray,
    labels: np.ndarray,
    patient_ids: List[str],
    batch_size: int = 16,
    shuffle: bool = True,
    augment: bool = False,
    seed: int = 42,
    noise_std: float = 0.005,
    amp_min: float = 0.95,
    amp_max: float = 1.05,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader from preprocessed signals.

    num_workers=0 is default for CPU-only (avoids multiprocessing overhead
    and ensures deterministic behavior without worker seeding complexity).
    """
    dataset = BrugadaECGDataset(
        signals=signals,
        labels=labels,
        patient_ids=patient_ids,
        augment=augment,
        seed=seed,
        noise_std=noise_std,
        amp_min=amp_min,
        amp_max=amp_max,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,   # CPU-only; no benefit
        drop_last=False,    # never drop last batch — val set is small
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )
    return loader