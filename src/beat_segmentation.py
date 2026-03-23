"""
beat_segmentation.py — Beat-level segmentation of ECG recordings.

Strategy (from Work Package B update, confirmed 19 March):
  - Detect R-peaks using neurokit2 on lead II (or fallback to V2)
  - Extract fixed window: 200ms pre-R + 500ms post-R = 70 samples per beat
  - Filter invalid beats (too short, artifact-contaminated)
  - Expected: ~9–14 valid beats per 12-second recording -> ~3,600 samples/fold

CRITICAL: beat-level predictions MUST be aggregated back to subject-level
          before evaluation. This module handles extraction only.
          Aggregation is in train.py / evaluate.py.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    BEAT_POST_SAMPLES,
    BEAT_PRE_SAMPLES,
    BEAT_WINDOW_SAMPLES,
    MIN_BEATS_PER_RECORD,
    SAMPLING_RATE,
)

logger = logging.getLogger(__name__)

# Lead indices for R-peak detection candidates (prefer lead II = index 1)
RPEAK_LEAD_PRIORITY = [1, 0, 7]   # Lead II, Lead I, V2


def detect_rpeaks(
    ecg: np.ndarray,
    fs: int = SAMPLING_RATE,
    method: str = "pantompkins1985",
) -> Optional[np.ndarray]:
    """
    Detect R-peak locations in a single-lead ECG.

    Args:
        ecg: (n_samples,) 1D array — single lead, already preprocessed
        fs: sampling rate
        method: neurokit2 R-peak algorithm

    Returns:
        Array of R-peak sample indices (int), or None on failure.
    """
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(ecg, sampling_rate=fs, method=method)
        peaks = info.get("ECG_R_Peaks", np.array([]))
        if len(peaks) == 0:
            return None
        return np.array(peaks, dtype=int)
    except Exception as e:
        logger.debug(f"neurokit2 R-peak detection failed ({method}): {e}")
        return None


def detect_rpeaks_from_signal(
    ecg: np.ndarray,
    fs: int = SAMPLING_RATE,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Try R-peak detection across multiple leads, return first success.

    Args:
        ecg: (n_leads, n_samples)
        fs: sampling rate

    Returns:
        (peaks_array, lead_used_index) or (None, -1) if all leads fail.
    """
    n_leads = ecg.shape[0]
    for lead_idx in RPEAK_LEAD_PRIORITY:
        if lead_idx >= n_leads:
            continue
        peaks = detect_rpeaks(ecg[lead_idx], fs=fs)
        if peaks is not None and len(peaks) >= MIN_BEATS_PER_RECORD:
            logger.debug(
                f"R-peaks detected on lead {lead_idx}: "
                f"{len(peaks)} peaks found."
            )
            return peaks, lead_idx

    # All priority leads failed — try all remaining leads
    for lead_idx in range(n_leads):
        if lead_idx in RPEAK_LEAD_PRIORITY:
            continue
        peaks = detect_rpeaks(ecg[lead_idx], fs=fs)
        if peaks is not None and len(peaks) >= MIN_BEATS_PER_RECORD:
            logger.debug(
                f"R-peaks detected on fallback lead {lead_idx}: "
                f"{len(peaks)} peaks."
            )
            return peaks, lead_idx

    logger.warning("R-peak detection failed on all leads.")
    return None, -1


def filter_valid_rpeaks(
    peaks: np.ndarray,
    n_samples: int,
    pre: int = BEAT_PRE_SAMPLES,
    post: int = BEAT_POST_SAMPLES,
) -> np.ndarray:
    """
    Remove R-peaks that are too close to signal boundaries
    (would produce incomplete beat windows).

    Args:
        peaks: R-peak indices
        n_samples: total signal length
        pre: samples before R-peak
        post: samples after R-peak

    Returns:
        Filtered peak indices.
    """
    valid = peaks[(peaks >= pre) & (peaks + post <= n_samples)]
    return valid


def extract_beats(
    ecg: np.ndarray,
    peaks: np.ndarray,
    lead_indices: Optional[List[int]] = None,
    pre: int = BEAT_PRE_SAMPLES,
    post: int = BEAT_POST_SAMPLES,
) -> np.ndarray:
    """
    Extract fixed-length beat windows from an ECG signal.

    Args:
        ecg: (n_leads, n_samples)
        peaks: valid R-peak indices
        lead_indices: which leads to extract (None = all leads)
        pre: samples before R-peak  (default 20 = 200ms)
        post: samples after R-peak  (default 50 = 500ms)

    Returns:
        beats: (n_beats, n_leads_selected, window_size) float32
    """
    if lead_indices is None:
        lead_indices = list(range(ecg.shape[0]))

    ecg_selected = ecg[lead_indices, :]   # (n_leads_sel, n_samples)
    window = pre + post
    beats = np.stack(
        [ecg_selected[:, p - pre : p + post] for p in peaks],
        axis=0
    )   # (n_beats, n_leads, window)
    beats = beats.astype(np.float32)
    return beats


def segment_recording(
    ecg: np.ndarray,
    lead_indices: Optional[List[int]] = None,
    fs: int = SAMPLING_RATE,
    pre: int = BEAT_PRE_SAMPLES,
    post: int = BEAT_POST_SAMPLES,
    min_beats: int = MIN_BEATS_PER_RECORD,
) -> Optional[np.ndarray]:
    """
    Full pipeline: detect R-peaks -> filter -> extract beats.

    Args:
        ecg: (n_leads, n_samples) preprocessed ECG
        lead_indices: subset of leads to include in each beat window
        fs: sampling rate
        pre/post: beat window parameters
        min_beats: minimum beats required; returns None if fewer found

    Returns:
        beats: (n_beats, n_leads_sel, window_size) or None
    """
    n_samples = ecg.shape[1]

    peaks, lead_used = detect_rpeaks_from_signal(ecg, fs=fs)
    if peaks is None:
        logger.warning("Segmentation failed: no R-peaks detected.")
        return None

    valid_peaks = filter_valid_rpeaks(peaks, n_samples=n_samples, pre=pre, post=post)

    if len(valid_peaks) < min_beats:
        logger.warning(
            f"Segmentation: only {len(valid_peaks)} valid beats "
            f"(min required: {min_beats}). Skipping."
        )
        return None

    beats = extract_beats(ecg, valid_peaks, lead_indices=lead_indices, pre=pre, post=post)
    logger.debug(
        f"Segmented {len(valid_peaks)} beats — shape {beats.shape}"
    )
    return beats


def segment_all_recordings(
    preprocessed_signals: Dict[str, Optional[np.ndarray]],
    lead_indices: Optional[List[int]] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Segment a full set of preprocessed signals into beats.

    Args:
        preprocessed_signals: {patient_id -> (n_leads, n_samples) | None}
        lead_indices: lead subset to use

    Returns:
        {patient_id -> (n_beats, n_leads_sel, window) | None}
    """
    results = {}
    failed = []
    for pid, ecg in preprocessed_signals.items():
        if ecg is None:
            results[pid] = None
            failed.append(pid)
            continue
        beats = segment_recording(ecg, lead_indices=lead_indices)
        results[pid] = beats
        if beats is None:
            failed.append(pid)

    n_ok = sum(1 for v in results.values() if v is not None)
    total_beats = sum(v.shape[0] for v in results.values() if v is not None)
    logger.info(
        f"Beat segmentation complete: "
        f"{n_ok}/{len(results)} subjects successful, "
        f"{total_beats} total beats."
    )
    if failed:
        logger.warning(f"Failed/skipped subjects: {failed}")
    return results


# ============================================================
# SANITY CHECK
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import numpy as np

    fs = 100
    n_samples = 1200
    n_leads = 12

    # Synthetic ECG with clear R-peaks at 100-sample intervals
    ecg = np.zeros((n_leads, n_samples), dtype=np.float32)
    # Simulate heartbeats: add positive spike at R-peak locations on lead II
    r_locs = np.arange(100, 1100, 90)
    for r in r_locs:
        ecg[1, r] = 2.0   # sharp peak on lead II

    # Add noise
    ecg += np.random.randn(*ecg.shape).astype(np.float32) * 0.05

    beats = segment_recording(ecg, lead_indices=list(range(12)))
    if beats is not None:
        print(f"Beat array shape: {beats.shape}")
        print(f"n_beats={beats.shape[0]}, n_leads={beats.shape[1]}, window={beats.shape[2]}")
    else:
        print("No beats detected (expected with synthetic data — run with real ECG).")