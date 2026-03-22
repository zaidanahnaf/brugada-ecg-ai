from __future__ import annotations

import numpy as np
from typing import Dict


def extract_st_basic_features(signal_beat: np.ndarray, fs: int = 100) -> Dict[str, float]:
    """
    Placeholder.
    signal_beat shape: (n_samples, n_leads) or per-lead depending on pipeline.
    """
    features = {
        "st_elev_v1_j80": np.nan,
        "st_elev_v2_j80": np.nan,
        "st_elev_v3_j80": np.nan,
        "st_slope_v1": np.nan,
        "st_slope_v2": np.nan,
        "st_slope_v3": np.nan,
    }
    return features


def extract_morphology_features(signal_beat: np.ndarray, fs: int = 100) -> Dict[str, float]:
    """
    Placeholder for:
    - covedness
    - saddlebackness
    - t-wave polarity
    - qrs duration
    """
    features = {
        "coved_score_v1": np.nan,
        "coved_score_v2": np.nan,
        "saddleback_score_v1": np.nan,
        "saddleback_score_v2": np.nan,
        "t_polarity_v1": np.nan,
        "t_polarity_v2": np.nan,
        "qrs_duration_ms": np.nan,
    }
    return features