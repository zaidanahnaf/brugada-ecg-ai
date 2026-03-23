# experiments/interpretability/permutation_importance.py

import numpy as np
import pandas as pd
import logging
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Callable
from pathlib import Path

from src.config import CFG
from src.fold_manager import load_folds, get_fold_split
from src.feature_store import fit_scaler_imputer, transform
from experiments.calibration import calibrate_model

logger = logging.getLogger(__name__)


def compute_permutation_importance_cv(
    model_factory: Callable,
    feature_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    feature_cols: List[str],
    n_repeats: int = 30,
    results_dir: str = "results/interpretability"
) -> pd.DataFrame:
    """
    Compute permutation importance using out-of-fold validation sets.

    CRITICAL DESIGN:
    - Permutation is applied to the VALIDATION fold only
    - Model was never trained on validation fold
    - This is the only leakage-safe permutation importance strategy
    - n_repeats=30 gives stable estimates; reduce to 10 if slow
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    fold_ids = sorted(fold_df['fold_id'].unique())
    fold_importances = []

    for i, val_fold in enumerate(fold_ids):
        logger.info(f"Permutation importance — fold {i+1}/{len(fold_ids)}")

        (X_train, y_train,
         X_val, y_val,
         train_ids, val_ids,
         feat_names) = get_fold_split(fold_df, feature_df, val_fold)

        # Filter to requested feature set
        feat_idx = [j for j, f in enumerate(feat_names) if f in feature_cols]

        if not feat_idx:
            raise ValueError("No requested feature_cols matched feat_names in current fold.")

        missing = [f for f in feature_cols if f not in feat_names]
        if missing:
            logger.warning(
                f"{len(missing)} requested features not found in fold features. "
                f"Examples: {missing[:10]}"
            )

        X_train_sub = X_train[:, feat_idx]
        X_val_sub = X_val[:, feat_idx]
        sub_names = [feat_names[j] for j in feat_idx]

        # Fit preprocessing on train only
        imputer, scaler = fit_scaler_imputer(X_train_sub)
        X_train_proc = transform(X_train_sub, imputer, scaler)
        X_val_proc = transform(X_val_sub, imputer, scaler)

        # Fit calibrated model safely inside training fold
        from sklearn.calibration import CalibratedClassifierCV

        base_model = model_factory()
        model_cal = CalibratedClassifierCV(
            estimator=base_model,
            method='sigmoid',
            cv=3
        )
        model_cal.fit(X_train_proc, y_train)

        # Permutation importance on validation fold
        result = permutation_importance(
            estimator=model_cal,
            X=X_val_proc,
            y=y_val,
            scoring='roc_auc',
            n_repeats=n_repeats,
            random_state=CFG.random_seed + val_fold,
            n_jobs=-1
        )

        n_features_actual = X_val_proc.shape[1]
        sub_names_actual  = sub_names[:n_features_actual]

        if len(sub_names_actual) != len(result.importances_mean):
            logger.warning(
                f"Fold {val_fold}: feature name count mismatch "
                f"({len(sub_names_actual)} names vs "
                f"{len(result.importances_mean)} importances) — truncating"
            )
            n = min(len(sub_names_actual), len(result.importances_mean))
            sub_names_actual = sub_names_actual[:n]
            importances_mean = result.importances_mean[:n]
            importances_std  = result.importances_std[:n]
        else:
            importances_mean = result.importances_mean
            importances_std  = result.importances_std

        fold_df_imp = pd.DataFrame({
            'feature':          sub_names_actual,
            'importance_mean':  importances_mean,
            'importance_std':   importances_std,
            'val_fold':         val_fold
        })
        fold_importances.append(fold_df_imp)

    # Aggregate across folds
    all_folds = pd.concat(fold_importances, ignore_index=True)

    summary = (
        all_folds.groupby('feature')
        .agg(
            perm_importance_mean=('importance_mean', 'mean'),
            perm_importance_std_within=('importance_std', 'mean'),
            perm_importance_fold_std=('importance_mean', 'std'),
            perm_importance_min=('importance_mean', 'min'),
            perm_importance_max=('importance_mean', 'max'),
        )
        .reset_index()
        .sort_values('perm_importance_mean', ascending=False)
    )

    # Stability flag: fold-to-fold variability should be reasonably small
    eps = 1e-8
    summary['perm_stable'] = (
        summary['perm_importance_fold_std'] <=
        0.5 * (summary['perm_importance_mean'].abs() + eps)
    )

    summary.to_csv(
        f"{results_dir}/permutation_importance.csv", index=False
    )

    if not summary.empty:
        logger.info(
            f"Permutation importance saved. "
            f"Top feature: {summary.iloc[0]['feature']} "
            f"(mean={summary.iloc[0]['perm_importance_mean']:.4f})"
        )
    else:
        logger.warning("Permutation importance summary is empty.")

    return summary


def compute_model_native_importance(
    model_factory: Callable,
    feature_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str = 'tree',   # 'tree' | 'linear'
    results_dir: str = "results/interpretability"
) -> pd.DataFrame:
    """
    Extract model-native feature importance across CV folds.

    tree   → model.feature_importances_ (RF / XGB / LGB)
    linear → abs(model.coef_[0]) (LR / LinearSVM)

    Averaged across folds for stability.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    n_folds = fold_df['fold_id'].nunique()
    fold_importances = []

    for val_fold in range(n_folds):
        (X_train, y_train,
         X_val, y_val,
         train_ids, val_ids,
         feat_names) = get_fold_split(fold_df, feature_df, val_fold)

        feat_idx    = [i for i, f in enumerate(feat_names) if f in feature_cols]
        X_train_sub = X_train[:, feat_idx]
        X_val_sub   = X_val[:, feat_idx]
        sub_names   = [feat_names[i] for i in feat_idx]

        imputer, scaler = fit_scaler_imputer(X_train_sub)
        X_train_proc = transform(X_train_sub, imputer, scaler)

        model = model_factory()
        model.fit(X_train_proc, y_train)

        if model_type == 'tree':
            if not hasattr(model, 'feature_importances_'):
                logger.warning("Model has no feature_importances_; skipping")
                continue
            importances = model.feature_importances_

        elif model_type == 'linear':
            if hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                logger.warning("Model has no coef_; skipping")
                continue

        # ── FIX: pastikan panjang match ───────────────────────────
        n_actual  = min(len(sub_names), len(importances))
        sub_names_actual = sub_names[:n_actual]
        importances      = importances[:n_actual]

        if len(sub_names) != len(importances):
            logger.warning(
                f"Native importance fold {val_fold}: "
                f"name/importance length mismatch "
                f"({len(sub_names)} vs {len(importances)}) — truncating to {n_actual}"
            )

        fold_df_imp = pd.DataFrame({
            'feature':           sub_names_actual,
            'native_importance': importances,
            'val_fold':          val_fold
        })
        fold_importances.append(fold_df_imp)

    if not fold_importances:
        return pd.DataFrame()

    all_folds = pd.concat(fold_importances, ignore_index=True)
    summary = (
        all_folds.groupby('feature')
        .agg(
            native_importance_mean=('native_importance', 'mean'),
            native_importance_std=('native_importance', 'std'),
        )
        .reset_index()
        .sort_values('native_importance_mean', ascending=False)
    )

    summary.to_csv(
        f"{results_dir}/native_importance_{model_type}.csv", index=False
    )
    return summary


def build_consensus_importance(
    perm_df: pd.DataFrame,
    native_df: pd.DataFrame,
    top_k: int = 20
) -> pd.DataFrame:
    """
    Build a consensus importance ranking by averaging
    normalized ranks from permutation and native importance.

    Rank aggregation is more robust than value averaging
    when importance scales differ across methods.
    """
    def rank_normalize(series):
        """Higher importance = lower rank number = rank 1."""
        return series.rank(ascending=False, method='min')

    merged = perm_df[['feature', 'perm_importance_mean']].merge(
        native_df[['feature', 'native_importance_mean']],
        on='feature', how='outer'
    ).fillna(0)

    merged['perm_rank'] = rank_normalize(merged['perm_importance_mean'])
    merged['native_rank'] = rank_normalize(merged['native_importance_mean'])
    merged['consensus_rank'] = (merged['perm_rank'] + merged['native_rank']) / 2
    merged = merged.sort_values('consensus_rank')
    merged['consensus_position'] = range(1, len(merged) + 1)

    return merged.head(top_k)