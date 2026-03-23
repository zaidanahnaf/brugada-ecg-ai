# src/metrics.py
"""
Complete metrics module for Brugada classification pipeline.
All threshold-independent and threshold-dependent metrics in one place.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
from typing import Dict, List


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CORE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute the full metrics suite for one fold / one model.

    Parameters
    ----------
    y_true    : binary ground truth labels (0/1)
    y_prob    : predicted probabilities for positive class
    threshold : decision threshold for binary prediction
    prefix    : string prefix added to all metric keys
                e.g. prefix='default_' → 'default_auroc'

    Returns
    -------
    dict of metric_name → float
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv         = tp / max(tp + fp, 1)   # Precision
    npv         = tn / max(tn + fn, 1)

    # Sensitivity at fixed specificity operating points
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    sens_at_90spec = _sensitivity_at_specificity(fpr, tpr, target_spec=0.90)
    sens_at_85spec = _sensitivity_at_specificity(fpr, tpr, target_spec=0.85)

    metrics = {
        f"{prefix}auroc":              float(roc_auc_score(y_true, y_prob)),
        f"{prefix}auprc":              float(average_precision_score(y_true, y_prob)),
        f"{prefix}f1_positive":        float(f1_score(y_true, y_pred, pos_label=1,
                                                       zero_division=0)),
        f"{prefix}f1_macro":           float(f1_score(y_true, y_pred, average='macro',
                                                       zero_division=0)),
        f"{prefix}balanced_accuracy":  float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}sensitivity":        float(sensitivity),
        f"{prefix}specificity":        float(specificity),
        f"{prefix}ppv":                float(ppv),
        f"{prefix}npv":                float(npv),
        f"{prefix}tp":                 int(tp),
        f"{prefix}fp":                 int(fp),
        f"{prefix}tn":                 int(tn),
        f"{prefix}fn":                 int(fn),
        f"{prefix}brier_score":        float(brier_score_loss(y_true, y_prob)),
        f"{prefix}threshold_used":     float(threshold),
        f"{prefix}sens_at_90spec":     float(sens_at_90spec),
        f"{prefix}sens_at_85spec":     float(sens_at_85spec),
    }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — THRESHOLD TUNING
# ══════════════════════════════════════════════════════════════════════════════

def tune_threshold_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Find optimal decision threshold by maximising the Youden index.

    Youden J = sensitivity + specificity - 1
    Maximising J balances sensitivity and specificity equally.

    Operates on validation fold ONLY — never on training data.

    Returns
    -------
    float : optimal threshold in [0, 1]
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    youden      = tpr + specificity - 1
    best_idx    = int(np.argmax(youden))
    return float(thresholds[best_idx])


def tune_threshold_min_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_sensitivity: float = 0.80
) -> float:
    """
    Find threshold that meets a minimum sensitivity requirement
    while maximising specificity.

    Clinical rationale:
        Missing Brugada is more dangerous than a false positive.
        Enforce minimum sensitivity = 0.80 as a conservative
        screening floor, then maximise specificity above that.

    Parameters
    ----------
    min_sensitivity : minimum required sensitivity (default 0.80)

    Returns
    -------
    float : threshold that satisfies the constraint
            falls back to argmax(sensitivity) if constraint is infeasible
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr

    valid = tpr >= min_sensitivity
    if not valid.any():
        # Constraint infeasible — return threshold giving highest sensitivity
        return float(thresholds[np.argmax(tpr)])

    # Among feasible thresholds, maximise specificity
    best_idx = int(np.argmax(specificity * valid))
    return float(thresholds[best_idx])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CV RESULT AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def summarize_cv_results(
    fold_metrics: List[Dict]
) -> Dict[str, float]:
    """
    Aggregate per-fold metric dicts into mean ± std summary.

    Input
    -----
    fold_metrics : list of dicts, one per fold
                   each dict is the output of compute_metrics()

    Returns
    -------
    dict with keys:
        {metric_name}_mean : float
        {metric_name}_std  : float
    for every numeric metric found in fold_metrics
    """
    import pandas as pd

    if not fold_metrics:
        return {}

    df = pd.DataFrame(fold_metrics)
    summary = {}

    for col in df.columns:
        if df[col].dtype in [float, int, np.float64, np.int64]:
            summary[f"{col}_mean"] = float(df[col].mean())
            summary[f"{col}_std"]  = float(df[col].std())

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sensitivity_at_specificity(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_spec: float
) -> float:
    """
    Interpolated sensitivity at a fixed target specificity level.

    Parameters
    ----------
    fpr         : false positive rate array (from roc_curve)
    tpr         : true positive rate array (from roc_curve)
    target_spec : desired specificity level e.g. 0.90

    Returns
    -------
    float : sensitivity at the closest achievable specificity >= target
            returns 0.0 if target specificity is never achieved
    """
    spec  = 1 - fpr
    valid = spec >= target_spec

    if not valid.any():
        return 0.0

    # Among points where specificity >= target, take the last one
    # (highest threshold = most restrictive = lowest tpr but highest spec)
    return float(tpr[valid][-1])