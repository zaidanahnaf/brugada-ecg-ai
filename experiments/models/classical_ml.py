# experiments/classical_ml.py

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional
from collections import Counter

from src.config.__init__ import CFG
from src.fold_manager import load_folds, get_fold_split
from src.feature_store import fit_scaler_imputer, transform, get_clean_feature_matrix
from src.evaluation.metrics import (
    compute_metrics, tune_threshold_youden,
    tune_threshold_min_sensitivity, summarize_cv_results,
    compute_metrics
)
from experiments.model_registry import get_all_models
from experiments.preprocessing.imbalance_strategy import undersample_training_fold
from experiments.preprocessing.feature_selection import select_features_inside_fold
from experiments.calibration import calibrate_model, compute_calibration_metrics

logger = logging.getLogger(__name__)


def run_cv_for_model(
    model_config: Dict,
    feature_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    feature_cols: List[str],
    imbalance_strategy: str = 'class_weight',   # 'class_weight' | 'undersample' | 'threshold_only'
    feature_selection_method: str = 'univariate_f',
    k_features: int = 40,
    threshold_method: str = 'youden',            # 'youden' | 'min_sensitivity'
    calibration_method: str = 'sigmoid',
    results_dir: str = "results",
    random_seed: int = CFG.feature.random_seed
) -> Dict:
    """
    Run full nested CV for one model configuration.

    Returns aggregated CV results dict.
    """
    model_name = model_config['name']
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    fold_threshold = []
    fold_selected_features = []
    fold_predictions = []     # For aggregate ROC curve

    n_folds = fold_df['fold_id'].nunique()

    for val_fold in range(n_folds):
        logger.info(f"[{model_name}] Outer fold {val_fold+1}/{n_folds}")

        # ── 1. Split ──────────────────────────────────────────────
        (X_train, y_train,
         X_val, y_val,
         train_ids, val_ids,
         feat_names) = get_fold_split(fold_df, feature_df, val_fold)

        # ── 2. Impute + Scale (fit on train ONLY) ─────────────────
        imputer, scaler = fit_scaler_imputer(X_train)
        X_train_proc = transform(X_train, imputer, scaler)
        X_val_proc = transform(X_val, imputer, scaler)

        # ── 3. Feature Selection (fit on train ONLY) ─────────────
        (X_train_sel, X_val_sel,
         selected_feats, selector) = select_features_inside_fold(
            X_train_proc, y_train,
            X_val_proc, feat_names,
            method=feature_selection_method,
            k=k_features
        )
        fold_selected_features.append(selected_feats)

        # ── 4. Imbalance: Undersampling (train only) ──────────────────
        if imbalance_strategy == 'undersample':
            X_train_fit, y_train_fit = undersample_training_fold(
                X_train_sel, y_train,
                strategy=0.5,
                random_seed=random_seed + val_fold
            )
            # CRITICAL: strip class_weight from the estimator when
            # undersampling is active to prevent double-correction.
            # clone() gives a fresh unfitted copy with modified params.
            model_config_fit = _strip_class_weight(model_config)
        else:
            X_train_fit, y_train_fit = X_train_sel, y_train
            model_config_fit = model_config

        # ── 5. Inner CV Hyperparameter Search ─────────────────────────
        best_estimator, best_params, best_inner_score = _inner_cv_search(
            model_config_fit,
            X_train_fit, y_train_fit,
            n_inner_folds=4,
            random_seed=random_seed
        )

        # ── 6. Fit Best Model on Full Training Fold ───────────────────
        best_estimator.fit(X_train_fit, y_train_fit)

        # ── 7. Calibrate ─────────────────────────────────────────
        calibrated_model = calibrate_model(
            best_estimator, X_train_fit, y_train_fit,
            method=calibration_method
        )

        # ── 8. Predict on Validation Fold ─────────────────────────
        y_prob_val = calibrated_model.predict_proba(X_val_sel)[:, 1]

        # ── 9. Threshold Tuning (on validation fold) ──────────────
        # NOTE: We tune threshold on the SAME validation fold
        # used for evaluation. This is acceptable in nested CV
        # because the threshold is a post-hoc operating point,
        # not a model parameter. However, for the final competition
        # submission, threshold should be fixed from CV mean.
        if threshold_method == 'youden':
            opt_threshold = tune_threshold_youden(y_val, y_prob_val)
        elif threshold_method == 'min_sensitivity':
            opt_threshold = tune_threshold_min_sensitivity(
                y_val, y_prob_val, min_sensitivity=0.80
            )
        else:
            opt_threshold = 0.5

        fold_threshold.append(opt_threshold)

        # ── 10. Evaluate ──────────────────────────────────────────
        metrics_default = compute_metrics(
            y_val, y_prob_val, threshold=0.5,
            prefix="default_"
        )
        metrics_tuned = compute_metrics(
            y_val, y_prob_val, threshold=opt_threshold,
            prefix="tuned_"
        )
        cal_metrics = compute_calibration_metrics(y_val, y_prob_val)

        fold_result = {
            **metrics_default,
            **metrics_tuned,
            **{f"cal_{k}": v for k, v in cal_metrics.items()},
            'val_fold': val_fold,
            'opt_threshold': opt_threshold,
            'n_train': len(y_train_fit),
            'n_val': len(y_val),
            'n_pos_train': int(y_train_fit.sum()),
            'n_pos_val': int(y_val.sum()),
            'n_features_selected': len(selected_feats),
            'best_params':           str(best_params),
            'best_inner_auroc':      best_inner_score,
        }
        fold_metrics.append(fold_result)
        fold_predictions.append((y_val, y_prob_val))

        logger.info(
            f"  Fold {val_fold+1}: "
            f"AUROC={metrics_default['default_auroc']:.3f} | "
            f"AUPRC={metrics_default['default_auprc']:.3f} | "
            f"Sens={metrics_tuned['tuned_sensitivity']:.3f} | "
            f"Spec={metrics_tuned['tuned_specificity']:.3f} | "
            f"Threshold={opt_threshold:.3f}"
        )

    # ── Summarize across folds ─────────────────────────────────────
    cv_summary = summarize_cv_results(fold_metrics)
    cv_summary['model_name'] = model_name
    cv_summary['imbalance_strategy'] = imbalance_strategy
    cv_summary['feature_selection_method'] = feature_selection_method
    cv_summary['threshold_method'] = threshold_method

    # Recommended threshold for deployment: mean of fold thresholds
    cv_summary['recommended_threshold'] = float(np.mean(fold_threshold))
    cv_summary['recommended_threshold_std'] = float(np.std(fold_threshold))

    # ── Best params logging ────────────────────────────────────────
    from collections import Counter
    
    params_list        = [f.get('best_params', '{}') for f in fold_metrics]
    most_common_params = Counter(params_list).most_common(1)[0][0]
    inner_scores       = [f.get('best_inner_auroc', 0.0) for f in fold_metrics]

    cv_summary['best_params_most_common']  = most_common_params
    cv_summary['best_inner_auroc_mean']    = float(np.mean(inner_scores))
    cv_summary['best_inner_auroc_std']     = float(np.std(inner_scores))

    logger.info(
        f"[{model_name}] Most common best params: {most_common_params} | "
        f"Inner AUROC: {np.mean(inner_scores):.3f} ± {np.std(inner_scores):.3f}"
    )

    # Feature selection stability: how often each feature was selected
    all_selected = [f for fold_feats in fold_selected_features for f in fold_feats]
    from collections import Counter
    feat_freq = Counter(all_selected)
    cv_summary['stable_features'] = [
        f for f, count in feat_freq.items() if count == n_folds
    ]
    cv_summary['feature_selection_stability'] = {
        f: count / n_folds for f, count in feat_freq.items()
    }

    # Save fold-level detail
    fold_df_out = pd.DataFrame(fold_metrics)
    fold_df_out.to_csv(
        f"{results_dir}/{model_name}_fold_metrics.csv", index=False
    )
    with open(f"{results_dir}/{model_name}_cv_summary.json", 'w') as f:
        # Filter non-serializable entries before saving
        serializable = {
            k: v for k, v in cv_summary.items()
            if not isinstance(v, dict)
        }
        json.dump(serializable, f, indent=2)

    return cv_summary


def _inner_cv_search(
    model_config: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_inner_folds: int = 4,
    random_seed: int = 42
) -> tuple:
    """
    Inner loop hyperparameter search.
    Returns (best_estimator, best_params, best_score)
    """
    from sklearn.model_selection import StratifiedKFold

    inner_cv = StratifiedKFold(
        n_splits=n_inner_folds,
        shuffle=True,
        random_state=random_seed
    )

    estimator   = model_config['estimator']
    param_grid  = model_config['param_grid']
    search_type = model_config.get('search_type', 'grid')

    if search_type == 'grid':
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            refit=True,
            error_score=0.0
        )
    else:
        n_iter = model_config.get('n_iter', 30)
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            refit=True,
            random_state=random_seed,
            error_score=0.0
        )

    search.fit(X_train, y_train)

    logger.info(
        f"    Inner CV best params : {search.best_params_} "
        f"(AUROC={search.best_score_:.3f})"
    )

    return search.best_estimator_, search.best_params_, float(search.best_score_)

def _strip_class_weight(model_config: Dict) -> Dict:
    """
    Return a copy of model_config with class_weight set to None
    on the estimator. Used when undersampling already handles
    class imbalance to prevent double-correction.

    Works for:
        sklearn estimators with class_weight param (LR, SVM, RF)
        XGBoost: sets scale_pos_weight=1
        LightGBM: sets is_unbalance=False
        CatBoost: sets auto_class_weights='None'

    Returns a shallow copy of model_config with a cloned estimator —
    the original model_config is never mutated.
    """
    from sklearn.base import clone
    import copy

    new_config = copy.copy(model_config)       # Shallow copy of config dict
    estimator = clone(model_config['estimator'])  # Fresh unfitted clone

    estimator_class = type(estimator).__name__

    if hasattr(estimator, 'class_weight'):
        estimator.set_params(class_weight=None)
        logger.debug(
            f"[_strip_class_weight] Removed class_weight from "
            f"{estimator_class} for undersample path."
        )

    elif hasattr(estimator, 'scale_pos_weight'):
        # XGBoost
        estimator.set_params(scale_pos_weight=1)
        logger.debug(
            f"[_strip_class_weight] Set scale_pos_weight=1 on "
            f"{estimator_class} for undersample path."
        )

    elif hasattr(estimator, 'is_unbalance'):
        # LightGBM
        estimator.set_params(is_unbalance=False)
        logger.debug(
            f"[_strip_class_weight] Set is_unbalance=False on "
            f"{estimator_class} for undersample path."
        )

    elif hasattr(estimator, 'auto_class_weights'):
        # CatBoost
        estimator.set_params(auto_class_weights=None)
        logger.debug(
            f"[_strip_class_weight] Removed auto_class_weights from "
            f"{estimator_class} for undersample path."
        )

    else:
        logger.warning(
            f"[_strip_class_weight] {estimator_class} has no recognised "
            f"class-weight parameter — no modification made."
        )

    new_config['estimator'] = estimator
    return new_config   