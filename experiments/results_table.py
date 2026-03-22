# experiments/results_table.py

import pandas as pd
import numpy as np
from typing import List, Dict


def build_comparison_table(cv_summaries: List[Dict]) -> pd.DataFrame:
    """
    Build the final model comparison table.
    Columns: model, AUROC, AUPRC, Sensitivity, Specificity,
             F1(pos), Balanced_Acc, Brier, ECE, Threshold
    """
    rows = []
    key_metrics = [
        ('AUROC',        'default_auroc_mean',         'default_auroc_std'),
        ('AUPRC',        'default_auprc_mean',         'default_auprc_std'),
        ('Sensitivity',  'tuned_sensitivity_mean',     'tuned_sensitivity_std'),
        ('Specificity',  'tuned_specificity_mean',     'tuned_specificity_std'),
        ('F1_pos',       'tuned_f1_positive_mean',     'tuned_f1_positive_std'),
        ('Bal_Acc',      'tuned_balanced_accuracy_mean','tuned_balanced_accuracy_std'),
        ('Brier',        'cal_brier_score_mean',       'cal_brier_score_std'),
        ('ECE',          'cal_ece_mean',               'cal_ece_std'),
        ('Sens@90Spec',  'default_sens_at_90spec_mean','default_sens_at_90spec_std'),
        ('Threshold',    'recommended_threshold',      'recommended_threshold_std'),
    ]

    for summary in cv_summaries:
        row = {
            'Model': summary.get('model_name', '?'),
            'Imbalance_Strategy': summary.get('imbalance_strategy', '?'),
            'Feat_Selection': summary.get('feature_selection_method', '?'),
        }
        for label, mean_key, std_key in key_metrics:
            mean_val = summary.get(mean_key, np.nan)
            std_val = summary.get(std_key, np.nan)
            row[label] = f"{mean_val:.3f} ± {std_val:.3f}" \
                if not np.isnan(mean_val) else "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('AUROC', ascending=False)
    return df


def print_comparison_table(cv_summaries: List[Dict]):
    df = build_comparison_table(cv_summaries)
    print("\n" + "="*120)
    print("MODEL COMPARISON TABLE")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120 + "\n")