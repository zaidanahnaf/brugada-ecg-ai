# src/aggregation.py

import numpy as np
import pandas as pd
from typing import Dict, List
from src.config.__init__ import CFG


def aggregate_beat_features(
    per_beat_features: List[Dict[str, float]],
    stats: List[str] = CFG.feature.aggregation_stats
) -> Dict[str, float]:
    """
    Aggregate per-beat feature dictionaries to subject-level.

    For each feature, computes: mean, median, std, min, max
    across all valid beats.

    NaN beats are excluded per-feature (not per-beat).
    A feature is marked as missing if all beats have NaN for it.
    """
    if not per_beat_features:
        return {}

    all_keys = set()
    for d in per_beat_features:
        all_keys.update(d.keys())

    aggregated = {}
    for key in all_keys:
        vals = np.array([
            d.get(key, np.nan) for d in per_beat_features
        ], dtype=float)
        valid = vals[np.isfinite(vals)]

        if len(valid) == 0:
            for stat in stats:
                aggregated[f"{key}_{stat}"] = np.nan
            aggregated[f"{key}_n_valid"] = 0
        else:
            for stat in stats:
                if stat == 'mean':
                    aggregated[f"{key}_{stat}"] = float(np.mean(valid))
                elif stat == 'median':
                    aggregated[f"{key}_{stat}"] = float(np.median(valid))
                elif stat == 'std':
                    aggregated[f"{key}_{stat}"] = float(np.std(valid))
                elif stat == 'min':
                    aggregated[f"{key}_{stat}"] = float(np.min(valid))
                elif stat == 'max':
                    aggregated[f"{key}_{stat}"] = float(np.max(valid))
                else:
                    raise ValueError(f"Unknown aggregation stat: {stat}")
            aggregated[f"{key}_n_valid"] = len(valid)

    return aggregated