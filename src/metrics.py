# src/metrics.py

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, balanced_accuracy_score,
    brier_score_loss, roc_curve, precision_recall_curve
)
from typing import Dict, Optional


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
    y_true   : binary labels
    y_prob   : predicted probabilities for positive class
    threshold: decision threshold (can be tuned)
    prefix   : string prefix for metric keys
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    sensitivity = tp / max(tp + fn, 1)   # Recall for positive class
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)           # Precisio