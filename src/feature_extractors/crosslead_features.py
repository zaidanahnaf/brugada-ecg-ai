# src/feature_extractors/crosslead_features.py

import numpy as np
from typing import Dict
from src.config import CFG


def extract_crosslead_features(
    subject_features: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute cross-lead relational features from aggregated per-lead features.
    Call AFTER subject-level aggregation (operating on median values).
    """
    features = {}
    priority = CFG.priority_leads   # ['V1', 'V2', 'V3']

    # ── ST Elevation Aggregates ───────────────────────────────────
    st40_vals = [
        subject_features.get(f"st_{lead}_st_j40_median", np.nan)
        for lead in priority
    ]
    valid_st40 = [v for v in st40_vals if not np.isnan(v)]

    if valid_st40:
        features['cl_max_st_j40_v1v3'] = float(np.max(valid_st40))
        features['cl_mean_st_j40_v1v3'] = float(np.mean(valid_st40))
        features['cl_var_st_j40_v1v3'] = float(np.var(valid_st40))
        features['cl_argmax_st_v1v3'] = int(np.argmax(st40_vals))
    else:
        for k in ['cl_max_st_j40_v1v3','cl_mean_st_j40_v1v3',
                  'cl_var_st_j40_v1v3','cl_argmax_st_v1v3']:
            features[k] = np.nan

    # ── Lead Differentials ────────────────────────────────────────
    v1_st = subject_features.get("st_V1_st_j40_median", np.nan)
    v2_st = subject_features.get("st_V2_st_j40_median", np.nan)
    v3_st = subject_features.get("st_V3_st_j40_median", np.nan)

    features['cl_v1v2_st_diff'] = (
        float(v1_st - v2_st) if not (np.isnan(v1_st) or np.isnan(v2_st)) else np.nan
    )
    features['cl_v2v3_st_diff'] = (
        float(v2_st - v3_st) if not (np.isnan(v2_st) or np.isnan(v3_st)) else np.nan
    )

    # ── Slope Consistency ─────────────────────────────────────────
    slopes = [
        subject_features.get(f"st_{lead}_st_slope_j0_j40_median", np.nan)
        for lead in ['V1', 'V2']
    ]
    if not any(np.isnan(s) for s in slopes):
        features['cl_v1v2_slope_consistency'] = int(
            np.sign(slopes[0]) == np.sign(slopes[1])
        )
    else:
        features['cl_v1v2_slope_consistency'] = np.nan

    # ── Coved / Saddleback Lead Counts ────────────────────────────
    coved_scores = [
        subject_features.get(f"morph_{lead}_covedness_score_median", np.nan)
        for lead in priority
    ]
    saddle_scores = [
        subject_features.get(f"morph_{lead}_saddleback_score_median", np.nan)
        for lead in priority
    ]
    t_inv_counts = [
        subject_features.get(f"morph_{lead}_t_inversion_indicator_median", np.nan)
        for lead in priority
    ]

    features['cl_n_leads_coved_v1v3'] = float(
        sum(s >= CFG.covedness_threshold
            for s in coved_scores if not np.isnan(s))
    )
    features['cl_n_leads_saddleback_v1v3'] = float(
        sum(s >= CFG.saddleback_threshold
            for s in saddle_scores if not np.isnan(s))
    )
    features['cl_n_leads_t_inverted_v1v3'] = float(
        sum(s >= 0.5 for s in t_inv_counts if not np.isnan(s))
    )

    coved_valid = [s for s in coved_scores if not np.isnan(s)]
    features['cl_max_covedness_v1v3'] = (
        float(np.max(coved_valid)) if coved_valid else np.nan
    )

    return features