# src/fiducial_detection.py

import numpy as np
from typing import Dict, Optional, Tuple
from src.config import CFG


def detect_fiducials(
    beat: np.ndarray,          # shape: (window_samples, n_leads)
    lead_idx: int,             # which lead column to analyze
    fs: int = CFG.fs,
    median_rr_ms: float = 800.0
) -> Dict:
    """
    Detect fiducial points for a single beat on a single lead.

    Fiducials:
        r_peak_sample   : int (by construction = pre_samples index)
        s_nadir_sample  : int
        j_point_sample  : int (QRS end proxy)
        st_samples      : dict {offset_ms: sample_index}
        t_start_sample  : int
        t_end_sample    : int
        baseline_samples: slice
        confidence      : dict of per-fiducial confidence flags

    All samples are relative to beat array start (index 0).
    """
    pre_samples = int(CFG.beat_window_pre_ms * fs / 1000)
    r_sample = pre_samples  # By construction

    sig = beat[:, lead_idx]
    n = len(sig)

    confidence = {}

    # ── Baseline Reference ─────────────────────────────────────────
    # PR segment: window before QRS
    bl_start = max(0, r_sample + int(CFG.baseline_ref_start_ms * fs / 1000))
    bl_end = max(0, r_sample + int(CFG.baseline_ref_end_ms * fs / 1000))

    if bl_end > bl_start + 2:
        baseline_mv = float(np.median(sig[bl_start:bl_end]))
        confidence['baseline'] = 'PR_SEGMENT'
    else:
        baseline_mv = float(np.median(sig[:max(1, r_sample - 5)]))
        confidence['baseline'] = 'FALLBACK_PRE_QRS'

    # ── S-Nadir Detection ──────────────────────────────────────────
    # Search window: R+10ms to R+100ms
    s_search_start = r_sample + int(10 * fs / 1000)
    s_search_end = min(n, r_sample + int(100 * fs / 1000))

    if s_search_end > s_search_start:
        s_nadir_rel = int(np.argmin(sig[s_search_start:s_search_end]))
        s_nadir_sample = s_search_start + s_nadir_rel
        confidence['s_nadir'] = 'DETECTED'
    else:
        # Fallback: assume S-nadir at R+40ms
        s_nadir_sample = r_sample + int(40 * fs / 1000)
        confidence['s_nadir'] = 'FALLBACK_R+40MS'

    # ── J-Point Detection (QRS End Proxy) ─────────────────────────
    # Strategy: minimum |dV/dt| after S-nadir
    j_search_start = s_nadir_sample + int(CFG.qrs_end_search_start_ms * fs / 1000)
    j_search_end = min(n, s_nadir_sample + int(CFG.qrs_end_search_end_ms * fs / 1000))

    j_sample, j_confidence = _detect_j_point(
        sig, s_nadir_sample, j_search_start, j_search_end, fs
    )
    confidence['j_point'] = j_confidence

    # ── ST Sample Indices ──────────────────────────────────────────
    st_samples = {}
    for offset_ms in CFG.st_offsets_ms:
        st_idx = j_sample + int(offset_ms * fs / 1000)
        if st_idx < n:
            st_samples[offset_ms] = st_idx
        else:
            st_samples[offset_ms] = None   # Out of bounds

    # ST window end
    st_window_end = min(n, j_sample + int(CFG.st_window_end_ms * fs / 1000))

    # ── T-Wave Window ──────────────────────────────────────────────
    t_start, t_end = _estimate_t_wave_window(
        r_sample, n, fs, median_rr_ms
    )
    confidence['t_wave'] = 'HR_ADJUSTED' if median_rr_ms > 0 else 'FALLBACK_FIXED'

    return {
        'r_sample': r_sample,
        's_nadir_sample': s_nadir_sample,
        'j_point_sample': j_sample,
        'baseline_mv': baseline_mv,
        'st_samples': st_samples,
        'st_window_start': j_sample,
        'st_window_end': st_window_end,
        't_start_sample': t_start,
        't_end_sample': t_end,
        'confidence': confidence
    }


def _detect_j_point(
    sig: np.ndarray,
    s_nadir: int,
    search_start: int,
    search_end: int,
    fs: int
) -> Tuple[int, str]:
    """
    Find J-point as the sample of minimum absolute first derivative
    in the post-S-nadir search window.

    Fallback: use S_nadir + 40ms if search window is too narrow.
    """
    if search_end <= search_start + 2:
        fallback = s_nadir + int(40 * fs / 1000)
        return min(fallback, len(sig) - 1), 'FALLBACK_NARROW_WINDOW'

    window = sig[search_start:search_end]
    deriv = np.abs(np.diff(window))

    if len(deriv) == 0:
        fallback = s_nadir + int(40 * fs / 1000)
        return min(fallback, len(sig) - 1), 'FALLBACK_EMPTY_DERIV'

    j_rel = int(np.argmin(deriv))
    j_sample = search_start + j_rel

    # Confidence check: is the derivative actually small here?
    # If not, we are probably not at a true plateau
    min_deriv_val = float(deriv[j_rel])
    max_in_qrs = float(np.max(np.abs(np.diff(sig[max(0, s_nadir-20):s_nadir+5]))))

    if max_in_qrs > 0 and (min_deriv_val / max_in_qrs) > 0.3:
        # Derivative still elevated — J-point uncertain
        confidence = 'LOW_CONFIDENCE_HIGH_DERIV'
    else:
        confidence = 'OK'

    return j_sample, confidence


def _estimate_t_wave_window(
    r_sample: int,
    signal_length: int,
    fs: int,
    median_rr_ms: float
) -> Tuple[int, int]:
    """
    Estimate T-wave window relative to R-peak.
    Uses HR-adjusted fractions of RR interval.
    Falls back to fixed offsets if RR unavailable.
    """
    if median_rr_ms > 400:
        rr_samples = int(median_rr_ms * fs / 1000)
        t_start = r_sample + int(CFG.t_wave_start_fraction * rr_samples)
        t_end = r_sample + int(CFG.t_wave_end_fraction * rr_samples)
    else:
        t_start = r_sample + int(CFG.t_wave_fallback_start_ms * fs / 1000)
        t_end = r_sample + int(CFG.t_wave_fallback_end_ms * fs / 1000)

    t_start = max(0, min(t_start, signal_length - 1))
    t_end = max(t_start + 1, min(t_end, signal_length))
    return t_start, t_end