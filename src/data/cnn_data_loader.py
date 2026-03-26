"""
src/data_loader.py

Loads raw WFDB ECG signals for the Brugada-HUCA dataset.

Responsibilities:
  - Read fold_assignments.csv (single source of truth)
  - Load .dat/.hea files via wfdb
  - Verify lead order and extract V1/V2/V3 indices
  - Handle known failed subjects gracefully
  - Return signals as numpy arrays, patient IDs as strings

CRITICAL: patient_id is ALWAYS treated as str, never int.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:
    raise ImportError("wfdb is required: pip install wfdb")

from src.cnn_utils import get_logger

logger = get_logger("data_loader")

# Expected lead names in order (WFDB standard for 12-lead)
EXPECTED_LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]


# =============================================================================
# FOLD ASSIGNMENT LOADING
# =============================================================================

def load_fold_assignments(fold_csv: str) -> pd.DataFrame:
    """
    Load fold_assignments.csv.
    Returns DataFrame with columns [patient_id (str), fold_id (int), brugada (int)].
    Fails loudly if expected columns are missing.
    """
    df = pd.read_csv(fold_csv, dtype={"patient_id": str})

    required = {"patient_id", "fold_id", "brugada"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fold_assignments.csv missing columns: {missing}\n"
                         f"Found columns: {list(df.columns)}")

    df["fold_id"] = df["fold_id"].astype(int)
    df["brugada"] = df["brugada"].astype(int)
    df["patient_id"] = df["patient_id"].astype(str)

    logger.info(f"Loaded fold assignments: {len(df)} subjects, "
                f"{df['brugada'].sum()} Brugada, {(df['brugada']==0).sum()} Normal")

    # Log per-fold class distribution for stratification verification
    for fold_id in sorted(df["fold_id"].unique()):
        fold_df = df[df["fold_id"] == fold_id]
        n_pos = fold_df["brugada"].sum()
        n_neg = (fold_df["brugada"] == 0).sum()
        logger.info(f"  Fold {fold_id}: {len(fold_df)} subjects | "
                    f"Brugada={n_pos} ({100*n_pos/len(fold_df):.1f}%) | "
                    f"Normal={n_neg}")
    return df


# =============================================================================
# SINGLE SUBJECT LOADER
# =============================================================================

def load_record(
    data_dir: str,
    patient_id: str,
    n_samples: int = 1200,
    expected_leads: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """
    Load a single WFDB record.

    Returns:
        np.ndarray of shape (12, n_samples) in float32, OR
        None if the record cannot be loaded.

    Lead order is VERIFIED against expected_leads if provided.
    Signal is clipped to n_samples (or zero-padded if shorter).
    """
    if expected_leads is None:
        expected_leads = EXPECTED_LEAD_NAMES

    record_path = os.path.join(data_dir, str(patient_id), f"{patient_id}")

    try:
        record = wfdb.rdrecord(record_path)
    except Exception as e:
        logger.warning(f"Cannot load record {patient_id}: {e}")
        return None

    # --- Verify lead names ---
    actual_leads = list(record.sig_name)
    if actual_leads != expected_leads:
        logger.warning(
            f"Lead mismatch for {patient_id}.\n"
            f"  Expected: {expected_leads}\n"
            f"  Actual  : {actual_leads}\n"
            "Will attempt to reorder by name matching."
        )
        # Attempt reorder by name
        try:
            reorder_idx = [actual_leads.index(name) for name in expected_leads]
            signal = record.p_signal[:, reorder_idx]  # (n_samples, 12)
        except ValueError as ve:
            logger.error(f"Cannot reorder leads for {patient_id}: {ve}")
            return None
    else:
        signal = record.p_signal  # (n_samples, 12)

    if signal is None:
        logger.warning(f"Null signal for {patient_id}")
        return None

    # --- Verify sampling rate ---
    if record.fs != 100:
        logger.warning(f"Unexpected fs={record.fs} for {patient_id} (expected 100 Hz)")

    # --- Shape enforcement ---
    actual_samples = signal.shape[0]
    if actual_samples < n_samples:
        # Zero-pad at the end
        pad = np.zeros((n_samples - actual_samples, signal.shape[1]), dtype=np.float32)
        signal = np.vstack([signal, pad])
        logger.warning(f"Zero-padded {patient_id}: {actual_samples} → {n_samples} samples")
    elif actual_samples > n_samples:
        signal = signal[:n_samples, :]

    # --- NaN/Inf check ---
    if not np.isfinite(signal).all():
        n_bad = (~np.isfinite(signal)).sum()
        logger.warning(f"{patient_id}: {n_bad} non-finite values — replacing with 0")
        signal = np.where(np.isfinite(signal), signal, 0.0)

    # Transpose to (12, n_samples) — channel-first for PyTorch Conv1d
    signal = signal.T.astype(np.float32)
    return signal


# =============================================================================
# DATASET BUILDER
# =============================================================================

def load_dataset(
    data_dir: str,
    patient_ids: List[str],
    labels: Dict[str, int],
    failed_subjects: Optional[Dict[str, str]] = None,
    n_samples: int = 1200,
    v_lead_indices: Optional[List[int]] = None,
    use_3_lead: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load signals for a list of patient_ids.

    Args:
        data_dir:        path to WFDB files
        patient_ids:     list of str IDs to load
        labels:          dict {patient_id -> 0/1}
        failed_subjects: dict of known-failed IDs to skip
        n_samples:       expected signal length
        v_lead_indices:  [6,7,8] for V1/V2/V3
        use_3_lead:      if True, slice V1/V2/V3; else return full 12-lead

    Returns:
        signals:    np.ndarray shape (N, C, T)  — loaded subjects only
        y:          np.ndarray shape (N,)
        loaded_ids: list of str patient_ids (successfully loaded, same order as signals/y)

    NOTE: Failed subjects are silently skipped here.
          The caller is responsible for inserting NaN rows in output CSVs.
    """
    if failed_subjects is None:
        failed_subjects = {}
    if v_lead_indices is None:
        v_lead_indices = [6, 7, 8]

    signals_list, y_list, loaded_ids = [], [], []

    for pid in patient_ids:
        pid = str(pid)
        if pid in failed_subjects:
            logger.debug(f"Skipping known-failed subject {pid}")
            continue

        signal = load_record(data_dir, pid, n_samples=n_samples)
        if signal is None:
            logger.warning(f"Failed to load {pid} — will appear as NaN in outputs")
            continue

        if use_3_lead:
            signal = signal[v_lead_indices, :]   # (3, 1200)

        signals_list.append(signal)
        y_list.append(labels[pid])
        loaded_ids.append(pid)

    if len(signals_list) == 0:
        raise RuntimeError("No signals loaded — check data_dir path and patient IDs")

    signals = np.stack(signals_list, axis=0).astype(np.float32)   # (N, C, T)
    y       = np.array(y_list, dtype=np.float32)

    logger.info(f"Loaded {len(loaded_ids)}/{len(patient_ids)} subjects | "
                f"Shape: {signals.shape} | "
                f"Brugada: {int(y.sum())} | Normal: {int((y==0).sum())}")
    return signals, y, loaded_ids


# =============================================================================
# LEAD INDEX VERIFICATION HELPER
# =============================================================================

def verify_lead_indices(data_dir: str, sample_patient_id: str,
                        v_leads: List[str] = None) -> Dict[str, int]:
    """
    Load one record, print its lead names, and return the confirmed V1/V2/V3 indices.
    Call this once before training to confirm lead ordering.
    """
    if v_leads is None:
        v_leads = ["V1", "V2", "V3"]

    record_path = os.path.join(data_dir, sample_patient_id)
    record = wfdb.rdrecord(record_path)
    lead_names = list(record.sig_name)
    logger.info(f"Lead names in {sample_patient_id}: {lead_names}")

    result = {}
    for v in v_leads:
        if v not in lead_names:
            raise ValueError(f"Lead '{v}' not found in record. Available: {lead_names}")
        result[v] = lead_names.index(v)
        logger.info(f"  {v} → index {result[v]}")
    return result