# src/beat_segmentation.py

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from src.config.__init__ import CFG

logger = logging.getLogger(__name__)


def detect_rpeaks(
    signal: np.ndarray,
    fs: int = CFG.data.fs,
    lead_names: list = None,
    method: str = CFG.preprocessing.rpeak_method
) -> Dict:
    """
    Detect R-peaks using the specified method.
    Detection is performed on Lead II preferentially, then V1 as fallback.

    Returns
    -------
    dict:
        rpeaks     : np.ndarray of sample indices
        rr_ms      : np.ndarray of RR intervals in ms
        method_used: str
        failed     : bool
        failure_reason: str or None
    """
    result = {
        'rpeaks': np.array([]),
        'rr_ms': np.array([]),
        'method_used': method,
        'failed': False,
        'failure_reason': None
    }

    # Select detection lead
    detection_lead_idx = _select_detection_lead(signal, lead_names)
    detection_signal = signal[:, detection_lead_idx]

    try:
        if method == 'neurokit':
            rpeaks = _detect_neurokit(detection_signal, fs)
        elif method == 'pantompkins':
            rpeaks = _detect_pantompkins(detection_signal, fs)
        elif method == 'wfdb':
            rpeaks = _detect_wfdb(detection_signal, fs)
        else:
            raise ValueError(f"Unknown R-peak method: {method}")

    except Exception as e:
        # Cascade fallback: try pantompkins if neurokit fails
        logger.warning(f"Primary R-peak detection failed ({e}), trying fallback")
        try:
            rpeaks = _detect_pantompkins(detection_signal, fs)
            result['method_used'] = 'pantompkins_fallback'
        except Exception as e2:
            result['failed'] = True
            result['failure_reason'] = f"ALL_RPEAK_METHODS_FAILED: {e2}"
            return result

    if len(rpeaks) < CFG.preprocessing.min_valid_beats:
        result['failed'] = True
        result['failure_reason'] = f"TOO_FEW_RPEAKS: {len(rpeaks)}"
        return result

    rr_ms = np.diff(rpeaks) * 1000 / fs
    result['rpeaks'] = rpeaks
    result['rr_ms'] = rr_ms
    return result


def _select_detection_lead(signal: np.ndarray, lead_names: list) -> int:
    """Prefer Lead II for R-peak detection; fallback to index 1, then max energy."""
    if lead_names:
        for preferred in ['II', 'I', 'V2']:
            if preferred in lead_names:
                return lead_names.index(preferred)
    # Fallback: lead with highest RMS
    return int(np.argmax(np.sqrt(np.mean(signal**2, axis=0))))


def _detect_neurokit(signal_1d: np.ndarray, fs: int) -> np.ndarray:
    import neurokit2 as nk
    _, info = nk.ecg_peaks(signal_1d, sampling_rate=fs, method='neurokit')
    return np.array(info['ECG_R_Peaks'])


def _detect_pantompkins(signal_1d: np.ndarray, fs: int) -> np.ndarray:
    import neurokit2 as nk
    _, info = nk.ecg_peaks(signal_1d, sampling_rate=fs, method='pantompkins1985')
    return np.array(info['ECG_R_Peaks'])


def _detect_wfdb(signal_1d: np.ndarray, fs: int) -> np.ndarray:
    import wfdb.processing as wp
    rpeaks = wp.qrs_detect(signal_1d, fs=fs)
    return np.array(rpeaks)


def filter_beats(
    rpeaks: np.ndarray,
    rr_ms: np.ndarray,
    signal_length: int,
    fs: int = CFG.data.fs
) -> Dict:
    """
    Filter beats by:
    1. Physiological RR range (40–150 BPM)
    2. RR stability: within ±15% of median RR (reject ectopics/artifacts)
    3. Sufficient signal before and after R-peak for window extraction

    Returns
    -------
    dict:
        valid_rpeaks     : np.ndarray
        rejected_indices : np.ndarray
        rejection_reasons: list of str
        n_valid          : int
        median_rr_ms     : float
        failed           : bool
        failure_reason   : str or None
    """
    if len(rpeaks) < 2:
        return {
            'valid_rpeaks': np.array([]),
            'n_valid': 0,
            'failed': True,
            'failure_reason': 'INSUFFICIENT_RPEAKS_FOR_FILTERING'
        }

    # Append a synthetic last RR (use median) so every beat has an associated RR
    rr_all = np.append(rr_ms, np.median(rr_ms))

    median_rr = np.median(rr_ms)

    pre_samples = int(CFG.preprocessing.beat_window_pre_ms * fs / 1000)
    post_samples = int(CFG.preprocessing.beat_window_post_ms * fs / 1000)

    valid_mask = np.ones(len(rpeaks), dtype=bool)
    rejection_reasons = [''] * len(rpeaks)

    for i, (rpeak, rr) in enumerate(zip(rpeaks, rr_all)):
        reasons = []

        # RR physiological range
        if not (CFG.preprocessing.min_rr_ms <= rr <= CFG.preprocessing.max_rr_ms):
            reasons.append(f"RR_OUT_OF_RANGE:{rr:.0f}ms")

        # RR stability (ectopic detection)
        if abs(rr - median_rr) / median_rr > CFG.preprocessing.rr_stability_threshold:
            reasons.append(f"RR_UNSTABLE:{rr:.0f}ms_vs_median_{median_rr:.0f}ms")

        # Boundary check
        if rpeak - pre_samples < 0:
            reasons.append("BEAT_TOO_CLOSE_TO_START")
        if rpeak + post_samples >= signal_length:
            reasons.append("BEAT_TOO_CLOSE_TO_END")

        if reasons:
            valid_mask[i] = False
            rejection_reasons[i] = '|'.join(reasons)

    valid_rpeaks = rpeaks[valid_mask]
    n_valid = int(valid_mask.sum())

    failed = n_valid < CFG.preprocessing.min_valid_beats
    failure_reason = f"TOO_FEW_VALID_BEATS:{n_valid}" if failed else None

    return {
        'valid_rpeaks': valid_rpeaks,
        'rejected_indices': np.where(~valid_mask)[0],
        'rejection_reasons': rejection_reasons,
        'n_valid': n_valid,
        'median_rr_ms': float(median_rr),
        'failed': failed,
        'failure_reason': failure_reason
    }


def extract_beat_windows(
    signal: np.ndarray,
    valid_rpeaks: np.ndarray,
    fs: int = CFG.data.fs
) -> np.ndarray:
    """
    Extract fixed-width beat windows around each valid R-peak.

    Returns
    -------
    beats : np.ndarray, shape (n_beats, window_samples, n_leads)
    """
    pre = int(CFG.preprocessing.beat_window_pre_ms * fs / 1000)
    post = int(CFG.preprocessing.beat_window_post_ms * fs / 1000)
    window_size = pre + post

    beats = np.zeros((len(valid_rpeaks), window_size, signal.shape[1]))
    for i, rpeak in enumerate(valid_rpeaks):
        beats[i] = signal[rpeak - pre: rpeak + post, :]
    return beats