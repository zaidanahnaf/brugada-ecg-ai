# experiments/ensemble.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from typing import List, Dict, Tuple
from src.metrics import compute_metrics, tune_threshold_youden


def build_stacking_ensemble(
    base_model_probs: Dict[str, np.ndarray],   # {model_name: y_prob_val}
    y_val: np.ndarray,
    method: str = 'weighted_average'            # 'weighted_average' | 'logistic_stacking'
) -> Tuple[np.ndarray, Dict]:
    """
    Build a simple calibrated ensemble from top models.

    Methods:
        'weighted_average'   — weight by individual fold AUROC
        'logistic_stacking'  — train logistic meta-learner on val fold probs

    NOTE: Logistic stacking on val fold is optimistic.
    For competition use, prefer weighted average unless you
    have a dedicated stacking holdout set.

    Only call with top 2–3 models by AUROC to avoid dilution.
    """
    names = list(base_model_probs.keys())
    prob_matrix = np.column_stack([base_model_probs[n] for n in names])
    # prob_matrix shape: (n_val_samples, n_models)

    if method == 'weighted_average':
        # Weight each model by its AUROC on this fold
        from sklearn.metrics import roc_auc_score
        aurocs = np.array([
            roc_auc_score(y_val, base_model_probs[n]) for n in names
        ])
        # Softmax weights: higher AUROC -> higher weight
        weights = np.exp(aurocs) / np.exp(aurocs).sum()
        ensemble_prob = prob_matrix @ weights

        meta_info = {
            'method': 'weighted_average',
            'model_names': names,
            'auroc_weights': dict(zip(names, weights.tolist())),
            'individual_aurocs': dict(zip(names, aurocs.tolist()))
        }

    elif method == 'logistic_stacking':
        # Train on half val, evaluate on other half
        n = len(y_val)
        split = n // 2
        meta_lr = LogisticRegression(C=1.0, max_iter=1000)
        meta_lr.fit(prob_matrix[:split], y_val[:split])
        ensemble_prob = meta_lr.predict_proba(prob_matrix[split:])[:, 1]

        # For full-fold coverage, use cross-val predictions
        # (simplified version here — extend if needed)
        meta_info = {
            'method': 'logistic_stacking',
            'model_names': names,
            'note': 'Trained on first half of val fold — optimistic estimate'
        }

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return ensemble_prob, meta_info


def select_top_models(
    cv_summaries: List[Dict],
    n_top: int = 3,
    primary_metric: str = 'default_auroc_mean'
) -> List[str]:
    """
    Select top N models by mean CV AUROC for ensemble building.
    Ties broken by AUPRC.
    """
    ranked = sorted(
        cv_summaries,
        key=lambda d: (
            d.get(primary_metric, 0),
            d.get('default_auprc_mean', 0)
        ),
        reverse=True
    )
    return [m['model_name'] for m in ranked[:n_top]]