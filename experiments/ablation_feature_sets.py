# experiments/ablation_feature_sets.py

"""
Canonical feature group definitions for ablation study.
All feature set names must resolve to columns in feature_matrix.csv.
Groups are defined by column name PREFIX patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def get_feature_groups(
    feature_cols: List[str]
) -> Dict[str, List[str]]:
    """
    Partition feature columns into named ablation groups
    by prefix pattern.

    Groups are mutually exclusive and collectively exhaustive
    over all feature columns.
    """

    def match(cols, patterns):
        matched = []
        for c in cols:
            if any(c.startswith(p) or p in c for p in patterns):
                matched.append(c)
        return matched

    groups = {}

    # ── A1: Basic Signal / Quality ────────────────────────────────
    groups['A1_signal_quality'] = match(feature_cols, [
        'sq_',              # signal quality prefix
        'hr_estimate',
        'rr_mean', 'rr_std', 'rr_cv',
        'median_rr',
        'n_valid_beats'
    ])

    # ── A2: ST Features (V1–V3) ───────────────────────────────────
    groups['A2_st_v1v2v3'] = match(feature_cols, [
        'st_V1_', 'st_V2_', 'st_V3_'
    ])

    # ── A2b: ST Features (all other leads) ───────────────────────
    other_leads = ['V4','V5','V6','I','II','III','aVR','aVL','aVF']
    groups['A2_st_other_leads'] = match(feature_cols, [
        f'st_{l}_' for l in other_leads
    ])

    # ── A3: Morphology Features (V1–V3) ──────────────────────────
    groups['A3_morph_v1v2v3'] = match(feature_cols, [
        'morph_V1_', 'morph_V2_', 'morph_V3_'
    ])

    # ── A3b: Morphology Features (other leads) ───────────────────
    groups['A3_morph_other_leads'] = match(feature_cols, [
        f'morph_{l}_' for l in other_leads
    ])

    # ── A4: Cross-Lead Relational ─────────────────────────────────
    groups['A4_crosslead'] = match(feature_cols, ['cl_'])

    # ── A5: Generic / Secondary Features ─────────────────────────
    groups['A5_generic'] = match(feature_cols, [
        'wavelet_', 'pca_', 'deriv_', 'spectral_'
    ])

    # Sanity check: coverage
    all_assigned = set(f for g in groups.values() for f in g)
    unassigned = [c for c in feature_cols if c not in all_assigned]
    if unassigned:
        groups['_unassigned'] = unassigned

    return groups


def build_feature_set(
    groups: Dict[str, List[str]],
    include: List[str] = None,
    exclude: List[str] = None
) -> List[str]:
    """
    Build a feature list by including or excluding named groups.

    Usage:
        build_feature_set(groups, include=['A2_st_v1v2v3'])
        build_feature_set(groups, exclude=['A3_morph_v1v2v3'])
    """
    if include is not None:
        cols = []
        for g in include:
            cols.extend(groups.get(g, []))
        return list(dict.fromkeys(cols))   # Preserve order, deduplicate

    if exclude is not None:
        all_cols = [f for g in groups.values() for f in g]
        excluded = set()
        for g in exclude:
            excluded.update(groups.get(g, []))
        return [c for c in all_cols if c not in excluded]

    # Default: all features
    return [f for g in groups.values() for f in g]


def get_ablation_feature_sets(
    feature_cols: List[str],
    groups: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Define ALL 20 ablation feature sets used in the experiment matrix.
    Returns dict: {experiment_id: [feature_col_list]}
    """
    B = lambda inc=None, exc=None: build_feature_set(groups, inc, exc)

    sets = {

        # ── AXIS 1: Additive Build-Up ─────────────────────────────
        'EXP_01_generic_stats_only':
            B(include=['A1_signal_quality']),

        'EXP_02_generic_plus_quality':
            B(include=['A1_signal_quality', 'A5_generic']),

        'EXP_03_st_only_v1v3':
            B(include=['A2_st_v1v2v3']),

        'EXP_04_morph_only_v1v3':
            B(include=['A3_morph_v1v2v3']),

        'EXP_05_st_plus_morph_v1v3':
            B(include=['A2_st_v1v2v3', 'A3_morph_v1v2v3']),

        'EXP_06_st_morph_crosslead':
            B(include=['A2_st_v1v2v3', 'A3_morph_v1v2v3', 'A4_crosslead']),

        'EXP_07_all_handcrafted':
            B(exclude=['A5_generic', '_unassigned']),

        'EXP_08_all_features_including_generic':
            B(exclude=['_unassigned']),

        # ── AXIS 2: Lead Ablation ─────────────────────────────────
        'EXP_09_v1v3_only_all_feature_types':
            B(include=['A2_st_v1v2v3', 'A3_morph_v1v2v3',
                       'A4_crosslead', 'A1_signal_quality']),

        'EXP_10_no_v1v3_all_others':
            B(exclude=['A2_st_v1v2v3', 'A3_morph_v1v2v3', '_unassigned']),

        'EXP_11_v1_only':
            B(include=[]) + _lead_only_features(
                groups, ['V1'], ['A2_st_v1v2v3', 'A3_morph_v1v2v3']
            ),

        'EXP_12_v2_only':
            B(include=[]) + _lead_only_features(
                groups, ['V2'], ['A2_st_v1v2v3', 'A3_morph_v1v2v3']
            ),

        'EXP_13_v1v2_only':
            B(include=[]) + _lead_only_features(
                groups, ['V1','V2'], ['A2_st_v1v2v3', 'A3_morph_v1v2v3']
            ),

        # ── AXIS 3: Subtractive Knockout ─────────────────────────
        'EXP_14_all_except_st':
            B(exclude=['A2_st_v1v2v3', 'A2_st_other_leads', '_unassigned']),

        'EXP_15_all_except_morph':
            B(exclude=['A3_morph_v1v2v3', 'A3_morph_other_leads', '_unassigned']),

        'EXP_16_all_except_crosslead':
            B(exclude=['A4_crosslead', '_unassigned']),

        'EXP_17_all_except_twave':
            _exclude_twave_features(feature_cols),

        # ── AXIS 4: Metadata Experiments (separate / flagged) ─────
        'EXP_18_metadata_only':
            ['basal_pattern', 'sudden_death'],     # WARNING: exploratory only

        'EXP_19_ecg_handcrafted_plus_metadata':
            B(exclude=['A5_generic', '_unassigned']) + [
                'basal_pattern', 'sudden_death'
            ],

        # ── AXIS 5: Hybrid (requires CNN embeddings) ─────────────
        # Placeholder — actual feature list populated at runtime
        # when CNN embedding file is available from Person 2
        'EXP_20_cnn_only':
            ['__CNN_EMBEDDING__'],   # Sentinel — replaced at runtime

        'EXP_21_cnn_plus_handcrafted':
            ['__CNN_EMBEDDING__'] + B(exclude=['A5_generic', '_unassigned']),

        'EXP_22_cnn_plus_st_morph_only':
            ['__CNN_EMBEDDING__'] + B(
                include=['A2_st_v1v2v3', 'A3_morph_v1v2v3', 'A4_crosslead']
            ),
    }

    return sets


def _lead_only_features(
    groups: Dict[str, List[str]],
    target_leads: List[str],
    source_groups: List[str]
) -> List[str]:
    """Filter feature columns to only those matching target leads."""
    cols = []
    for g in source_groups:
        for c in groups.get(g, []):
            if any(f'_{lead}_' in c for lead in target_leads):
                cols.append(c)
    return list(dict.fromkeys(cols))


def _exclude_twave_features(feature_cols: List[str]) -> List[str]:
    """Return all features except T-wave related ones."""
    t_patterns = ['_t_polarity', '_t_amplitude', '_t_mean',
                  '_t_inversion', '_t_peak', '_t_wave']
    return [
        c for c in feature_cols
        if not any(pat in c for pat in t_patterns)
    ]