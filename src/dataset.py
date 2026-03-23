"""
dataset.py — PyTorch Dataset classes for beat-level and recording-level training.

Two datasets:
  1. BeatDataset     — one sample = one beat (for CNN training)
  2. RecordingDataset — one sample = full 12s recording (for ablation)

Key: BeatDataset stores patient_id so we can aggregate predictions per subject.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    AUG_AMPLITUDE_RANGE,
    AUG_BASELINE_SHIFT_STD,
    BEAT_WINDOW_SAMPLES,
    N_SAMPLES,
)

logger = logging.getLogger(__name__)


class BeatDataset(Dataset):
    """
    Dataset where each item is a single beat window.

    Used during training. At inference, predictions are aggregated per patient.

    Args:
        beats_dict:  {patient_id -> (n_beats, n_leads, window_size) | None}
        labels_dict: {patient_id -> int (0 or 1)}
        patient_ids: ordered list of patient IDs to include
        augment: whether to apply training augmentation
    """

    def __init__(
        self,
        beats_dict:  Dict[str, Optional[np.ndarray]],
        labels_dict: Dict[str, int],
        patient_ids: List[str],
        augment: bool = False,
    ):
        self.augment = augment
        self.items: List[Tuple[np.ndarray, int, str]] = []  # (beat, label, pid)

        n_excluded = 0
        for pid in patient_ids:
            pid = str(pid)
            beats = beats_dict.get(pid, None)
            if beats is None:
                logger.debug(f"[{pid}] No beats — excluded from dataset.")
                n_excluded += 1
                continue
            label = labels_dict.get(pid)
            if label is None:
                logger.warning(f"[{pid}] No label found — excluded.")
                n_excluded += 1
                continue
            for beat in beats:
                self.items.append((beat, int(label), pid))

        n_patients = len(patient_ids) - n_excluded
        logger.info(
            f"BeatDataset: {len(self.items)} beats from {n_patients} patients "
            f"({n_excluded} excluded). augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        beat, label, pid = self.items[idx]
        beat = beat.copy()   # avoid modifying original

        if self.augment:
            beat = self._augment(beat)

        x = torch.tensor(beat, dtype=torch.float32)
        y = torch.tensor(float(label), dtype=torch.float32)
        return x, y, pid

    def _augment(self, beat: np.ndarray) -> np.ndarray:
        """
        Allowed augmentations (from Work Package C):
          - Small amplitude scaling (0.9–1.1)
          - Small baseline shift

        Forbidden: time warping, lead permutation, aggressive amplitude changes.
        """
        # Amplitude scaling
        scale = np.random.uniform(*AUG_AMPLITUDE_RANGE)
        beat = beat * scale

        # Baseline shift (additive noise, very small)
        shift = np.random.randn(*beat.shape).astype(np.float32) * AUG_BASELINE_SHIFT_STD
        beat = beat + shift

        return beat

    def get_patient_ids(self) -> List[str]:
        return [pid for _, _, pid in self.items]

    def get_labels(self) -> List[int]:
        return [label for _, label, _ in self.items]


class RecordingDataset(Dataset):
    """
    Dataset where each item is a full 12-second recording.

    Used for the recording-level ablation (Option A).
    Input shape: (n_leads, 1200)

    Args:
        signals_dict: {patient_id -> (n_leads, n_samples) | None}
        labels_dict:  {patient_id -> int}
        patient_ids:  ordered list of patient IDs to include
        augment: whether to apply augmentation
    """

    def __init__(
        self,
        signals_dict: Dict[str, Optional[np.ndarray]],
        labels_dict:  Dict[str, int],
        patient_ids:  List[str],
        augment: bool = False,
    ):
        self.augment = augment
        self.items: List[Tuple[np.ndarray, int, str]] = []

        n_excluded = 0
        for pid in patient_ids:
            pid = str(pid)
            sig = signals_dict.get(pid, None)
            if sig is None:
                n_excluded += 1
                continue
            label = labels_dict.get(pid)
            if label is None:
                n_excluded += 1
                continue
            self.items.append((sig, int(label), pid))

        logger.info(
            f"RecordingDataset: {len(self.items)} recordings "
            f"({n_excluded} excluded). augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sig, label, pid = self.items[idx]
        sig = sig.copy()

        if self.augment:
            sig = self._augment(sig)

        x = torch.tensor(sig, dtype=torch.float32)
        y = torch.tensor(float(label), dtype=torch.float32)
        return x, y, pid

    def _augment(self, sig: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*AUG_AMPLITUDE_RANGE)
        shift = np.random.randn(*sig.shape).astype(np.float32) * AUG_BASELINE_SHIFT_STD
        return sig * scale + shift

    def get_patient_ids(self) -> List[str]:
        return [pid for _, _, pid in self.items]

    def get_labels(self) -> List[int]:
        return [label for _, label, _ in self.items]


# ============================================================
# COLLATE FUNCTION (returns pid alongside tensors)
# ============================================================

def collate_with_pid(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Custom collate_fn that preserves patient_id strings.
    Returns: (x_batch, y_batch, pid_list)
    """
    xs, ys, pids = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(pids)