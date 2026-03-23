# src/feature_extractors/morphology_features.py

import numpy as np
from typing import Dict
from src.config import CFG


def extract_morphology_features_single_beat(
    beat: np.ndarray,
    lead_idx: int,
    fiducials: Dict,
    st_features: Dict,
    lead_name: str,
    fs: int = CFG.fs
) -> Dict[str, float]:
    """
    Extract Brugada-specific morphology features for one beat on one lead.
    Builds on ST features already computed.
    """
    sig = beat[:, lead_idx]
    baseline = fiducials['baseline_mv']
    j = fiducials['j_point_sample']
    t_start = fiducials['t_start_sample']
    t_end = fiducials['t_end_sample']
    st_win_end = fiducials['st_window_end']

    pfx = f"morph_{lead_name}"
    features = {}

    # Pull precomputed ST features (same beat)
    _get = lambda k, default=np.nan: st_features.get(
        f"st_{lead_name}_{k}", default
    )

    j_amp = _get('j_point_amplitude')
    st_slope = _get('st_slope_j0_j40')
    st_convexity = _get('st_convexity')
    st_monotonicity = _get('st_monotonicity')

    # ── T-Wave Features ───────────────────────────────────────────
    if t_end > t_start + 2:
        t_seg = sig[t_start:t_end] - baseline
        t_mean = float(np.mean(t_seg))
        t_peak_val = float(t_seg[np.argmax(np.abs(t_seg))])

        features[f"{pfx}_t_polarity"] = float(np.sign(t_mean))
        features[f"{pfx}_t_amplitude"] = float(np.max(np.abs(t_seg)))
        features[f"{pfx}_t_mean"] = t_mean
        features[f"{pfx}_t_inversion_indicator"] = int(
            t_mean < CFG.t_inversion_threshold_mv
        )
        features[f"{pfx}_t_peak_value"] = t_peak_val
    else:
        for k in ['t_polarity','t_amplitude','t_mean','t_inversion_indicator','t_peak_value']:
            features[f"{pfx}_{k}"] = np.nan

    # ── Second Hump Detection (saddleback secondary peak) ─────────
    if st_win_end > j + 3:
        st_seg = sig[j:st_win_end] - baseline
        second_hump, hump_amp, trough_depth = _detect_second_hump(st_seg)
        features[f"{pfx}_second_hump_present"] = int(second_hump)
        features[f"{pfx}_second_hump_amplitude"] = hump_amp
        features[f"{pfx}_trough_depth_between_humps"] = trough_depth
    else:
        features[f"{pfx}_second_hump_present"] = 0
        features[f"{pfx}_second_hump_amplitude"] = np.nan
        features[f"{pfx}_trough_depth_between_humps"] = np.nan

    # ── Covedness Score (composite) ───────────────────────────────
    # Components: negative slope, positive convexity, T-inversion
    coved = 0.0
    coved_weight = 0.0

    if not np.isnan(st_slope):
        coved += max(0.0, -st_slope * 10)   # Negative slope -> coved
        coved_weight += 1.0

    if not np.isnan(st_convexity):
        coved += max(0.0, st_convexity * 100)  # Positive d2 -> convex -> coved
        coved_weight += 1.0

    t_inv = features.get(f"{pfx}_t_inversion_indicator", np.nan)
    if not np.isnan(t_inv):
        coved += float(t_inv)
        coved_weight += 1.0

    features[f"{pfx}_covedness_score"] = (
        coved / coved_weight if coved_weight > 0 else np.nan
    )

    # ── Saddleback Score (composite) ──────────────────────────────
    saddle = 0.0
    saddle_weight = 0.0

    if not np.isnan(st_slope):
        saddle += max(0.0, st_slope * 10)   # Positive slope -> saddleback
        saddle_weight += 1.0

    if not np.isnan(st_convexity):
        saddle += max(0.0, -st_convexity * 100)  # Negative d2 -> concave -> saddle
        saddle_weight += 1.0

    saddle += float(features[f"{pfx}_second_hump_present"])
    saddle_weight += 1.0

    features[f"{pfx}_saddleback_score"] = (
        saddle / saddle_weight if saddle_weight > 0 else np.nan
    )

    # ── High-Takeoff Binary Indicators ────────────────────────────
    features[f"{pfx}_high_takeoff_gt_2mm"] = int(
        not np.isnan(j_amp) and j_amp >= CFG.high_takeoff_2mm_mv
    )
    features[f"{pfx}_high_takeoff_gt_1mm"] = int(
        not np.isnan(j_amp) and j_amp >= CFG.high_takeoff_1mm_mv
    )

    return features


def _detect_second_hump(
    st_seg: np.ndarray
) -> tuple:
    """
    Detect presence of a secondary local maximum in the ST segment.
    Uses simple local maxima detection with noise tolerance.

    Returns
    -------
    (second_hump_present: bool,
     second_hump_amp: float,
     trough_depth: float)
    """
    if len(st_seg) < 5:
        return False, np.nan, np.nan

    from scipy.signal import find_peaks
    # Minimum peak prominence = 0.05 mV to avoid noise peaks
    peaks, props = find_peaks(st_seg, prominence=0.05, distance=3)

    if len(peaks) >= 2:
        # First peak (high-takeoff), second peak (saddleback hump)
        first_peak_amp = float(st_seg[peaks[0]])
        second_peak_amp = float(st_seg[peaks[1]])
        valley_between = float(np.min(st_seg[peaks[0]:peaks[1]]))
        trough_depth = first_peak_amp - valley_between
        return True, second_peak_amp, trough_depth
    else:
        return False, np.nan, np.nan