# experiments/calibration.py

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Dict


# def calibrate_model(
#     model,
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     method: str = 'isotonic'
# ) -> object:
#     """
#     Calibrate a fitted model's probability outputs.

#     Methods:
#         'isotonic' — non-parametric, better for larger datasets
#         'sigmoid'  — Platt scaling, better for small N (use this for <200 train)

#     IMPORTANT: Calibration is fit on the same training fold.
#     This is standard practice for nested CV.
#     With only ~290 training samples per fold,
#     use 'sigmoid' (Platt) as default — isotonic can overfit.

#     For a cleaner estimate, use a held-out calibration set
#     (5% of train fold), but this is optional given our N.
#     """
#     # Choose method based on training size
#     n_train = len(y_train)
#     if method == 'auto':
#         method = 'sigmoid' if n_train < 500 else 'isotonic'

#     calibrated = CalibratedClassifierCV(
#         estimator=model,
#         method=method,
#         cv='prefit'         # Model already fitted — calibrate only
#     )
#     calibrated.fit(X_train, y_train)
#     return calibrated

def calibrate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = 'sigmoid'
) -> object:
    """
    Calibrate a fitted model's probability outputs.

    sklearn 1.8+ change: cv='prefit' is removed.
    Use cv=None instead — same behaviour, new API.
    """
    if method == 'auto':
        method = 'sigmoid' if len(y_train) < 500 else 'isotonic'

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=None           # ← was 'prefit', now None (sklearn 1.8+)
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Compute calibration quality metrics."""
    brier = float(brier_score_loss(y_true, y_prob))

    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    bin_sizes = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    weights   = bin_sizes / max(bin_sizes.sum(), 1)

    min_len    = min(len(fraction_of_positives), len(mean_predicted), len(weights))
    abs_errors = np.abs(
        fraction_of_positives[:min_len] - mean_predicted[:min_len]
    )
    ece = float(np.sum(weights[:min_len] * abs_errors))
    mce = float(np.max(abs_errors)) if len(abs_errors) > 0 else np.nan

    return {
        'brier_score': brier,
        'ece':         ece,
        'mce':         mce
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration quality metrics.

    Returns:
        brier_score  : overall probability calibration (lower = better)
        ece          : expected calibration error
        mce          : maximum calibration error
    """
    brier = float(brier_score_loss(y_true, y_prob))

    # ECE and MCE
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    bin_sizes = np.histogram(y_prob, bins=n_bins, range=(0,1))[0]
    weights = bin_sizes / max(bin_sizes.sum(), 1)

    min_len = min(len(fraction_of_positives), len(mean_predicted), len(weights))
    abs_errors = np.abs(
        fraction_of_positives[:min_len] - mean_predicted[:min_len]
    )
    ece = float(np.sum(weights[:min_len] * abs_errors))
    mce = float(np.max(abs_errors)) if len(abs_errors) > 0 else np.nan

    return {
        'brier_score': brier,
        'ece': ece,
        'mce': mce
    }