# src/feature_extractors/st_features.py

import numpy as np
from typing import Dict, List
from src.config import CFG


def extract_st_features_single_beat(
    beat: np.ndarray,           # (window_samples, n_leads)
    lead_idx: int,
    fiducials: Dict,
    lead_name: str,
    fs: int = CFG.fs
) -> Dict[str, float]:
    """
    Extract ST-segment features for one beat on one lead.
    All amplitudes relative to local baseline.
    """
    sig = beat[:, lead_idx]
    baseline = fiducials['baseline_mv']
    j = fiducials['j_point_sample']
    st_samples = fiducials['st_samples']
    j_confidence = fiducials['confidence']['j_point']

    pfx = f"st_{lead_name}"
    features = {}
    features[f"{pfx}_j_point_confidence"] = (
        0 if 'LOW' in j_confidence or 'FALLBACK' in j_confidence else 1
    )

    # ── R and S Amplitudes ────────────────────────────────────────
    r_sample = fiducials['r_sample']
    s_sample = fiducials['s_nadir_sample']

    features[f"{pfx}_r_amplitude"] = float(sig[r_sample] - baseline)
    features[f"{pfx}_s_amplitude"] = float(sig[s_sample] - baseline)
    features[f"{pfx}_rs_ratio"] = float(
        features[f"{pfx}_r_amplitude"] /
        max(abs(features[f"{pfx}_s_amplitude"]), 1e-4)
    )

    # ── J-Point Amplitude ─────────────────────────────────────────
    j_amp = float(sig[j] - baseline)
    features[f"{pfx}_j_point_amplitude"] = j_amp
    features[f"{pfx}_high_takeoff_gt_2mm"] = int(j_amp >= CFG.high_takeoff_2mm_mv)
    features[f"{pfx}_high_takeoff_gt_1mm"] = int(j_amp >= CFG.high_takeoff_1mm_mv)

    # ── ST Amplitudes at Fixed Offsets ────────────────────────────
    for offset_ms in CFG.st_offsets_ms:
        st_idx = st_samples.get(offset_ms)
        if st_idx is not None and st_idx < len(sig):
            features[f"{pfx}_st_j{int(offset_ms)}"] = float(sig[st_idx] - baseline)
        else:
            features[f"{pfx}_st_j{int(offset_ms)}"] = np.nan

    # ── ST Slopes ─────────────────────────────────────────────────
    st20 = features.get(f"{pfx}_st_j20", np.nan)
    st40 = features.get(f"{pfx}_st_j40", np.nan)
    st80 = features.get(f"{pfx}_st_j80", np.nan)

    if not np.isnan(st40) and not np.isnan(j_amp):
        features[f"{pfx}_st_slope_j0_j40"] = (st40 - j_amp) / 40.0
    else:
        features[f"{pfx}_st_slope_j0_j40"] = np.nan

    if not np.isnan(st80) and not np.isnan(st20):
        features[f"{pfx}_st_slope_j20_j80"] = (st80 - st20) / 60.0
    else:
        features[f"{pfx}_st_slope_j20_j80"] = np.nan

    # ── ST Amplitude Ratios ───────────────────────────────────────
    j_denom = max(abs(j_amp), 0.01)
    for offset_ms in [40, 80]:
        key = f"{pfx}_st_j{offset_ms}"
        if key in features and not np.isnan(features[key]):
            features[f"{pfx}_st_j{offset_ms}_j0_ratio"] = features[key] / j_denom
        else:
            features[f"{pfx}_st_j{offset_ms}_j0_ratio"] = np.nan

    # ── ST Area Above Baseline ────────────────────────────────────
    st_win_start = j
    st_win_end = fiducials['st_window_end']
    if st_win_end > st_win_start:
        st_segment = sig[st_win_start:st_win_end] - baseline
        area = float(np.trapz(np.maximum(st_segment, 0)))
        features[f"{pfx}_st_area_above_baseline"] = area
    else:
        features[f"{pfx}_st_area_above_baseline"] = np.nan

    # ── ST Monotonicity (coved descent proxy) ─────────────────────
    if st_win_end > st_win_start + 2:
        st_seg = sig[st_win_start:st_win_end]
        diffs = np.diff(st_seg)
        features[f"{pfx}_st_monotonicity"] = float(
            np.sum(diffs < 0) / max(len(diffs), 1)
        )
    else:
        features[f"{pfx}_st_monotonicity"] = np.nan

    # ── ST Convexity (second derivative) ─────────────────────────
    # Positive = convex upward = coved; negative = concave = saddleback
    if st_win_end > st_win_start + 3:
        st_seg = sig[st_win_start:st_win_end]
        d2 = np.diff(st_seg, n=2)
        features[f"{pfx}_st_convexity"] = float(np.mean(d2))
        features[f"{pfx}_st_curvature_sign"] = float(np.sign(np.mean(d2)))
    else:
        features[f"{pfx}_st_convexity"] = np.nan
        features[f"{pfx}_st_curvature_sign"] = np.nan

    return features