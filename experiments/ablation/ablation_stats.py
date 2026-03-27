# experiments/ablation_stats.py

"""
Statistical comparison between ablation experiments.
With 5-fold CV and small N, treat all comparisons as indicative.
Use paired tests where appropriate (same folds).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from pathlib import Path


def paired_fold_comparison(
    exp_id_a: str,
    exp_id_b: str,
    results_dir: str = "outputs/results/ablation",
    metric: str = 'default_auroc'
) -> Dict:
    """
    Paired comparison between two ablation experiments.
    Uses the same fold indices — fold-level AUROC is paired.

    Paired t-test: valid because same fold splits used.
    With n=5 pairs, p-value is indicative only.
    Report effect size (Cohen's d) alongside p-value.
    """
    def load_fold_metric(exp_id, metric):
        path = Path(results_dir) / f"{exp_id}_folds.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if metric not in df.columns:
            return None
        return df.sort_values('val_fold')[metric].values

    scores_a = load_fold_metric(exp_id_a, metric)
    scores_b = load_fold_metric(exp_id_b, metric)

    if scores_a is None or scores_b is None:
        return {'error': 'Missing fold data for one or both experiments'}

    if len(scores_a) != len(scores_b):
        return {'error': f'Fold count mismatch: {len(scores_a)} vs {len(scores_b)}'}

    diff = scores_a - scores_b
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    cohen_d = float(np.mean(diff) / (np.std(diff) + 1e-9))

    return {
        'exp_a':        exp_id_a,
        'exp_b':        exp_id_b,
        'metric':       metric,
        'mean_a':       float(np.mean(scores_a)),
        'mean_b':       float(np.mean(scores_b)),
        'mean_diff':    float(np.mean(diff)),
        'std_diff':     float(np.std(diff)),
        't_statistic':  float(t_stat),
        'p_value':      float(p_val),
        'cohen_d':      float(cohen_d),
        'interpretation': _interpret_comparison(
            np.mean(diff), p_val, cohen_d
        )
    }


def _interpret_comparison(mean_diff, p_val, cohen_d) -> str:
    """
    Interpret the comparison result with appropriate caution.
    With n=5 folds, all p-values are very low power.
    """
    sig = "significant" if p_val < 0.05 else "not significant (n=5 folds, low power)"
    direction = "A > B" if mean_diff > 0 else "B > A"
    magnitude = (
        "negligible" if abs(cohen_d) < 0.2 else
        "small" if abs(cohen_d) < 0.5 else
        "medium" if abs(cohen_d) < 0.8 else
        "large"
    )
    return (
        f"{direction} | Δ={mean_diff:+.3f} | {sig} | "
        f"Effect size: {magnitude} (d={cohen_d:.2f})"
    )


def run_key_comparisons(
    results_dir: str = "outputs/results/ablation"
) -> pd.DataFrame:
    """
    Run the pre-specified key comparisons that answer
    the mandatory ablation questions.
    """
    comparisons = [
        # Q1: Do ST features help over generic stats?
        ('EXP_03_st_only_v1v3',
         'EXP_01_generic_stats_only',
         'Do ST features improve over generic stats?'),

        # Q2: Do morphology features help over ST alone?
        ('EXP_05_st_plus_morph_v1v3',
         'EXP_03_st_only_v1v3',
         'Does morphology add over ST alone?'),

        # Q3: Does cross-lead consistency add over ST+Morph?
        ('EXP_06_st_morph_crosslead',
         'EXP_05_st_plus_morph_v1v3',
         'Does cross-lead add over ST+Morph?'),

        # Q4: Full handcrafted vs. ST+Morph+Cross
        ('EXP_07_all_handcrafted',
         'EXP_06_st_morph_crosslead',
         'Does adding quality features help?'),

        # Q5: Do generic features add over full handcrafted?
        ('EXP_08_all_features_including_generic',
         'EXP_07_all_handcrafted',
         'Do generic features add over clinical features?'),

        # Q6: V1–V3 vs. no V1–V3
        ('EXP_09_v1v3_only_all_feature_types',
         'EXP_10_no_v1v3_all_others',
         'Impact of V1-V3 vs. all other leads'),

        # Q7: V2 alone vs. V1 alone
        ('EXP_12_v2_only',
         'EXP_11_v1_only',
         'V2 vs V1: which single lead is more discriminative?'),

        # Q8: V1+V2 sufficient vs. full V1–V3?
        ('EXP_13_v1v2_only',
         'EXP_09_v1v3_only_all_feature_types',
         'V1+V2 sufficient vs. V1+V2+V3?'),

        # Q9: ST removal impact
        ('EXP_07_all_handcrafted',
         'EXP_14_all_except_st',
         'Impact of removing ST features'),

        # Q10: Morphology removal impact
        ('EXP_07_all_handcrafted',
         'EXP_15_all_except_morph',
         'Impact of removing morphology features'),

        # Q11: T-wave contribution
        ('EXP_07_all_handcrafted',
         'EXP_17_all_except_twave',
         'T-wave feature contribution'),

        # Q12: Hybrid vs. CNN-only
        ('EXP_21_cnn_plus_handcrafted',
         'EXP_20_cnn_only',
         'Hybrid vs. CNN alone'),

        # Q13: Hybrid vs. handcrafted-only
        ('EXP_21_cnn_plus_handcrafted',
         'EXP_07_all_handcrafted',
         'Hybrid vs. handcrafted alone'),
    ]

    rows = []
    for exp_a, exp_b, question in comparisons:
        result = paired_fold_comparison(
            exp_a, exp_b,
            results_dir=results_dir,
            metric='default_auroc'
        )
        result['question'] = question
        rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(f"{results_dir}/key_comparisons.csv", index=False)
    return df