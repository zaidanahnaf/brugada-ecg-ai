"""
src/metrics.py

Evaluation metrics for the Brugada CNN branch.

All metrics computed from OOF predictions (validation folds only).
Threshold selection: Youden index on the validation fold.

PRIMARY METRIC: AUROC
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float = 0.90,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for one fold or aggregated OOF.

    Args:
        y_true:             binary ground truth (0/1), shape (N,)
        y_prob:             predicted probabilities in [0, 1], shape (N,)
        target_specificity: for "sensitivity at X% specificity" metric

    Returns:
        dict with keys:
          auroc, auprc, sensitivity, specificity, f1_positive, brier_score,
          youden_threshold, sensitivity_at_target_spec, n_pos, n_neg
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_prob = np.array(y_prob, dtype=np.float32)

    # --- AUROC ---
    auroc = roc_auc_score(y_true, y_prob)

    # --- AUPRC ---
    auprc = average_precision_score(y_true, y_prob)

    # --- Brier score ---
    brier = brier_score_loss(y_true, y_prob)

    # --- ROC curve for threshold-based metrics ---
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # --- Youden threshold ---
    youden_scores = tpr - fpr
    youden_idx    = int(np.argmax(youden_scores))
    youden_thresh = float(thresholds[youden_idx])

    y_pred = (y_prob >= youden_thresh).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_pos      = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # --- Sensitivity at target specificity ---
    # Find the threshold where specificity ≈ target_specificity
    # specificity = 1 - fpr
    spec_arr = 1.0 - fpr
    # Find where spec_arr >= target_specificity (take the tightest one)
    valid_mask = spec_arr >= target_specificity
    if valid_mask.any():
        # Among all points with spec >= target, pick the one with highest sensitivity
        sens_at_target = float(tpr[valid_mask].max())
    else:
        sens_at_target = 0.0

    return {
        "auroc":                       float(auroc),
        "auprc":                       float(auprc),
        "sensitivity":                 float(sensitivity),
        "specificity":                 float(specificity),
        "f1_positive":                 float(f1_pos),
        "brier_score":                 float(brier),
        "youden_threshold":            float(youden_thresh),
        f"sensitivity_at_{int(target_specificity*100)}pct_spec": float(sens_at_target),
        "n_pos":                       int(y_true.sum()),
        "n_neg":                       int((y_true == 0).sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def aggregate_fold_metrics(fold_metrics: list) -> Dict[str, float]:
    """
    Aggregate per-fold metrics into mean ± std summary.

    Args:
        fold_metrics: list of dicts from compute_metrics(), one per fold

    Returns:
        dict with {metric_mean, metric_std} for each scalar metric
    """
    scalar_keys = [
        "auroc", "auprc", "sensitivity", "specificity",
        "f1_positive", "brier_score",
    ]

    # Find sensitivity_at_XX key dynamically
    for k in fold_metrics[0]:
        if k.startswith("sensitivity_at_"):
            scalar_keys.append(k)
            break

    summary = {}
    for key in scalar_keys:
        vals = [m[key] for m in fold_metrics if key in m]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"]  = float(np.std(vals))

    return summary


def print_fold_metrics(fold_id: int, metrics: Dict) -> None:
    """Pretty-print metrics for one fold."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_id} METRICS")
    print(f"{'='*60}")
    print(f"  AUROC       : {metrics['auroc']:.4f}")
    print(f"  AUPRC       : {metrics['auprc']:.4f}")
    print(f"  Sensitivity : {metrics['sensitivity']:.4f}  (Youden θ={metrics['youden_threshold']:.3f})")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  F1 (pos)    : {metrics['f1_positive']:.4f}")
    print(f"  Brier       : {metrics['brier_score']:.4f}")
    sens_key = [k for k in metrics if k.startswith("sensitivity_at_")]
    if sens_key:
        print(f"  {sens_key[0]}: {metrics[sens_key[0]]:.4f}")
    print(f"  TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"{'='*60}\n")


def print_summary_metrics(summary: Dict) -> None:
    """Pretty-print aggregated 5-fold summary."""
    print(f"\n{'='*60}")
    print(f"  5-FOLD OOF SUMMARY")
    print(f"{'='*60}")
    keys = ["auroc", "auprc", "sensitivity", "specificity", "f1_positive", "brier_score"]
    for k in keys:
        mean = summary.get(f"{k}_mean", float("nan"))
        std  = summary.get(f"{k}_std",  float("nan"))
        print(f"  {k:<20}: {mean:.4f} ± {std:.4f}")
    print(f"{'='*60}\n")