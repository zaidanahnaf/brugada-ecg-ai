"""
load_data.py — WFDB loader, metadata loader, and fold-file integrity check.

Usage:
    from src.load_data import load_metadata, load_signal, verify_fold_sha256
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

from src.config import (
    DATA_FILES_DIR,
    FOLD_ASSIGNMENTS,
    FOLD_SHA256,
    LEAD_NAMES,
    METADATA_CSV,
    N_LEADS,
    N_SAMPLES,
    PROBLEMATIC_SUBJECTS,
    SAMPLING_RATE,
)

logger = logging.getLogger(__name__)


# ============================================================
# FOLD FILE INTEGRITY
# ============================================================

def verify_fold_sha256(path: Path = FOLD_ASSIGNMENTS) -> bool:
    """
    CRITICAL — must be called before any training run.
    Verifies the SHA256 hash of fold_assignments.csv matches the contract.
    Raises RuntimeError on mismatch.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()

    if digest != FOLD_SHA256:
        raise RuntimeError(
            f"fold_assignments.csv SHA256 MISMATCH!\n"
            f"  Expected : {FOLD_SHA256}\n"
            f"  Got      : {digest}\n"
            "Do NOT proceed. The fold file may have been modified."
        )
    logger.info(f"[OK] fold_assignments.csv SHA256 verified: {digest}")
    return True


def load_fold_assignments(path: Path = FOLD_ASSIGNMENTS) -> pd.DataFrame:
    """
    Load fold assignments. Returns DataFrame with columns:
        patient_id (str), fold_id (int), label (int)
    """
    verify_fold_sha256(path)
    df = pd.read_csv(path, dtype={"patient_id": str})
    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    assert "patient_id" in df.columns, "fold_assignments.csv must have patient_id column"
    assert "fold_id"    in df.columns, "fold_assignments.csv must have fold_id column"
    df["patient_id"] = df["patient_id"].astype(str)
    df["fold_id"]    = df["fold_id"].astype(int)
    logger.info(f"Loaded {len(df)} fold assignments across {df['fold_id'].nunique()} folds.")
    return df


# ============================================================
# METADATA
# ============================================================

def load_metadata(path: Path = METADATA_CSV) -> pd.DataFrame:
    """
    Load metadata.csv.
    Ensures patient_id is string and label column 'brugada' exists (0/1 int).
    """
    df = pd.read_csv(path, dtype={"patient_id": str})
    df.columns = [c.strip().lower() for c in df.columns]
    df["patient_id"] = df["patient_id"].astype(str)

    # Detect label column — could be named 'brugada', 'label', 'diagnosis', etc.
    label_candidates = ["brugada", "label", "diagnosis", "class"]
    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(
            f"Cannot find label column in metadata.csv. "
            f"Columns found: {list(df.columns)}"
        )
    if label_col != "brugada":
        df = df.rename(columns={label_col: "brugada"})
    df["brugada"] = df["brugada"].astype(int)

    n_pos = df["brugada"].sum()
    n_neg = len(df) - n_pos
    logger.info(
        f"Metadata loaded: {len(df)} subjects — "
        f"{n_pos} Brugada, {n_neg} Normal"
    )
    return df


# ============================================================
# WFDB SIGNAL LOADER
# ============================================================

def find_record_path(patient_id: str, files_dir: Path = DATA_FILES_DIR) -> Optional[Path]:
    """
    Locate the WFDB record for a given patient_id.
    Expected structure: files_dir / patient_id / patient_id   (no extension)
    """
    record_dir  = files_dir / str(patient_id)
    record_path = record_dir / str(patient_id)
    hea_file    = record_path.with_suffix(".hea")

    if hea_file.exists():
        return record_path
    # fallback: search recursively (handles unexpected nesting)
    matches = list(files_dir.rglob(f"{patient_id}.hea"))
    if matches:
        return matches[0].with_suffix("")
    return None


def load_signal(
    patient_id: str,
    files_dir: Path = DATA_FILES_DIR,
    expected_leads: int = N_LEADS,
    expected_samples: int = N_SAMPLES,
) -> Optional[np.ndarray]:
    """
    Load a single WFDB record and return raw signal array.

    Returns:
        np.ndarray of shape (n_leads, n_samples) — float32
        None if the record cannot be loaded or is known-problematic and unrecoverable.

    Notes:
        - Always returns PHYSICAL units (mV).
        - Handles missing channels by zero-padding.
        - Handles length mismatch by truncate/pad.
    """
    record_path = find_record_path(patient_id, files_dir)
    if record_path is None:
        logger.warning(f"[{patient_id}] Record not found in {files_dir}")
        return None

    try:
        record = wfdb.rdrecord(str(record_path))
    except Exception as e:
        logger.error(f"[{patient_id}] wfdb.rdrecord failed: {e}")
        return None

    signal = record.p_signal   # shape: (n_samples, n_leads)
    if signal is None:
        logger.error(f"[{patient_id}] p_signal is None")
        return None

    signal = signal.T.astype(np.float32)   # -> (n_leads, n_samples)

    # ---------- handle unexpected number of leads ----------
    actual_leads, actual_samples = signal.shape
    if actual_leads != expected_leads:
        logger.warning(
            f"[{patient_id}] Expected {expected_leads} leads, "
            f"got {actual_leads}. Zero-padding extra / truncating."
        )
        if actual_leads < expected_leads:
            pad = np.zeros((expected_leads - actual_leads, actual_samples), dtype=np.float32)
            signal = np.vstack([signal, pad])
        else:
            signal = signal[:expected_leads, :]

    # ---------- handle length mismatch ----------
    if actual_samples != expected_samples:
        logger.warning(
            f"[{patient_id}] Expected {expected_samples} samples, "
            f"got {actual_samples}."
        )
        if actual_samples < expected_samples:
            pad = np.zeros((expected_leads, expected_samples - actual_samples), dtype=np.float32)
            signal = np.hstack([signal, pad])
        else:
            signal = signal[:, :expected_samples]

    # ---------- NaN / Inf check ----------
    if np.any(~np.isfinite(signal)):
        n_bad = np.sum(~np.isfinite(signal))
        logger.warning(
            f"[{patient_id}] {n_bad} non-finite values detected. "
            "Replacing with 0."
        )
        signal = np.where(np.isfinite(signal), signal, 0.0)

    return signal   # (n_leads, n_samples), float32


# ============================================================
# BULK LOADER
# ============================================================

def load_all_signals(
    patient_ids: List[str],
    files_dir: Path = DATA_FILES_DIR,
    skip_problematic: bool = True,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Load signals for a list of patient IDs.

    Returns dict: { patient_id (str) -> np.ndarray | None }
    Problematic subjects are loaded with None if skip_problematic=True.
    """
    results: Dict[str, Optional[np.ndarray]] = {}
    for pid in patient_ids:
        if skip_problematic and pid in PROBLEMATIC_SUBJECTS:
            logger.info(
                f"[{pid}] Skipping — listed in PROBLEMATIC_SUBJECTS. "
                "Will appear as NaN in deliverables."
            )
            results[pid] = None
            continue
        results[pid] = load_signal(pid, files_dir)
        if results[pid] is not None:
            logger.debug(f"[{pid}] Loaded OK — shape {results[pid].shape}")

    n_loaded = sum(1 for v in results.values() if v is not None)
    n_failed = len(results) - n_loaded
    logger.info(
        f"Bulk load complete: {n_loaded} loaded, {n_failed} failed/skipped."
    )
    return results


# ============================================================
# QUICK SANITY CHECK (run as __main__)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Verifying fold file ===")
    verify_fold_sha256()

    print("\n=== Loading metadata ===")
    meta = load_metadata()
    print(meta.head())
    print(f"\nLabel distribution:\n{meta['brugada'].value_counts()}")

    print("\n=== Loading fold assignments ===")
    folds = load_fold_assignments()
    print(folds.groupby("fold_id").size())

    print("\n=== Test-loading first 3 subjects ===")
    test_ids = meta["patient_id"].iloc[:3].tolist()
    signals = load_all_signals(test_ids)
    for pid, sig in signals.items():
        if sig is not None:
            print(f"  {pid}: shape={sig.shape}, dtype={sig.dtype}, "
                  f"min={sig.min():.3f}, max={sig.max():.3f}")
        else:
            print(f"  {pid}: FAILED/SKIPPED")