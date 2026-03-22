# src/data_loader.py

import wfdb
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from src.config import CFG

logger = logging.getLogger(__name__)


def load_record(
    patient_id: str,
    data_dir: str = CFG.data_dir
) -> Optional[Dict]:
    """
    Load a single WFDB record and validate it.

    Returns
    -------
    dict with keys:
        patient_id  : str
        signal      : np.ndarray, shape (n_samples, n_leads)
        fs          : int
        lead_names  : List[str]
        n_samples   : int
        duration_s  : float
        valid       : bool
        failure_reason : str or None
    """
    result = {
        'patient_id': patient_id,
        'signal': None,
        'fs': None,
        'lead_names': None,
        'n_samples': None,
        'duration_s': None,
        'valid': False,
        'failure_reason': None
    }

    record_path = str(Path(data_dir) / patient_id)

    try:
        record = wfdb.rdsamp(record_path)
        signal, fields = record  # signal: (n_samples, n_leads)

    except Exception as e:
        result['failure_reason'] = f"WFDB_READ_ERROR: {e}"
        logger.error(f"[{patient_id}] {result['failure_reason']}")
        return result

    # ── Structural Validation ─────────────────────────────────────
    if signal is None or signal.size == 0:
        result['failure_reason'] = "EMPTY_SIGNAL"
        return result

    if signal.ndim != 2:
        result['failure_reason'] = f"UNEXPECTED_DIMS: {signal.ndim}"
        return result

    fs = fields['fs']
    lead_names = fields['sig_name']
    n_samples, n_leads = signal.shape

    # ── Lead Validation ───────────────────────────────────────────
    missing_priority = [
        l for l in CFG.priority_leads if l not in lead_names
    ]
    if missing_priority:
        result['failure_reason'] = f"MISSING_PRIORITY_LEADS: {missing_priority}"
        logger.warning(f"[{patient_id}] {result['failure_reason']}")
        # Do NOT return — partial failure, mark and continue

    # ── Signal Range Check ────────────────────────────────────────
    if np.any(np.abs(signal) > 10.0):   # >10 mV is physiologically implausible
        logger.warning(f"[{patient_id}] Signal amplitude exceeds 10mV — possible unit mismatch")

    # ── NaN / Inf Check ───────────────────────────────────────────
    nan_mask = ~np.isfinite(signal)
    if nan_mask.any():
        n_bad = nan_mask.sum()
        pct_bad = 100 * n_bad / signal.size
        if pct_bad > 5.0:
            result['failure_reason'] = f"EXCESSIVE_NAN: {pct_bad:.1f}%"
            return result
        else:
            # Interpolate small gaps
            signal = _interpolate_nans(signal)
            logger.warning(f"[{patient_id}] Interpolated {n_bad} NaN samples ({pct_bad:.1f}%)")

    result.update({
        'signal': signal,
        'fs': fs,
        'lead_names': lead_names,
        'n_samples': n_samples,
        'duration_s': n_samples / fs,
        'valid': True,
        'failure_reason': None,
        'missing_priority_leads': missing_priority
    })
    return result


def _interpolate_nans(signal: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN samples per lead."""
    out = signal.copy()
    for ch in range(signal.shape[1]):
        x = signal[:, ch]
        nans = ~np.isfinite(x)
        if nans.any():
            idx = np.arange(len(x))
            out[:, ch] = np.interp(idx, idx[~nans], x[~nans])
    return out


def get_lead_index(lead_names: list, lead: str) -> Optional[int]:
    """Return column index for a named lead, or None if absent."""
    try:
        return lead_names.index(lead)
    except ValueError:
        return None


def load_metadata(path: str = CFG.metadata_path) -> pd.DataFrame:
    """Load and validate metadata CSV."""
    df = pd.read_csv(path)
    required = ['patient_id', 'brugada']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")
    df['patient_id'] = df['patient_id'].astype(str)
    return df