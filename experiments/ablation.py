# experiments/ablation.py

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.config import CFG
from src.fold_manager import load_folds, get_fold_split
from src.feature_store import fit_scaler_imputer, transform
from src.metrics import (
    compute_metrics, tune_threshold_youden, summarize_cv_results
)
from experiments.ablation_feature_sets import (
    get_feature_groups, get_ablation_feature_sets
)
from experiments.calibration import calibrate_model

logger = logging.getLogger(__name__)


def run_single_ablation_experiment(
    exp_id: str,
    feature_subset: List[str],
    feature_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    model_factory,             # Callable: () → fitted sklearn estimator
    results_dir: str = "results/ablation",
    cnn_embedding_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Run one ablation experiment across all CV folds.

    model_factory: a zero-argument callable that returns a fresh
    (unfitted) estimator with fixed hyperparameters from Phase 3.
    Hyperparameters are FIXED — only features vary.

    Returns CV summary dict.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # ── Handle CNN sentinel ───────────────────────────────────────
    is_cnn_experiment = '__CNN_EMBEDDING__' in feature_subset
    if is_cnn_experiment and cnn_embedding_df is None:
        logger.warning(
            f"[{exp_id}] CNN embedding not available — skipping."
        )
        return {'exp_id': exp_id, 'status': 'SKIPPED_NO_CNN'}

    # ── Resolve actual feature columns ───────────────────────────
    actual_cols = _resolve_feature_cols(
        feature_subset, feature_df, cnn_embedding_df
    )

    if len(actual_cols) == 0:
        logger.error(f"[{exp_id}] No valid feature columns resolved.")
        return {'exp_id': exp_id, 'status': 'FAILED_NO_FEATURES'}

    logger.info(f"[{exp_id}] Running with {len(actual_cols)} features...")

    # ── Build merged DataFrame (ECG + optional CNN) ───────────────
    if is_cnn_experiment and cnn_embedding_df is not None:
        merged_df = feature_df.merge(
            cnn_embedding_df, on='patient_id', how='inner'
        )
    else:
        merged_df = feature_df.copy()

    n_folds = fold_df['fold_id'].nunique()
    fold_metrics = []

    for val_fold in range(n_folds):

        # ── Split ─────────────────────────────────────────────────
        train_ids = fold_df[fold_df['fold_id'] != val_fold]['patient_id'].values
        val_ids = fold_df[fold_df['fold_id'] == val_fold]['patient_id'].values

        train_df = merged_df[merged_df['patient_id'].isin(train_ids)]
        val_df = merged_df[merged_df['patient_id'].isin(val_ids)]

        # Filter to available columns
        available = [c for c in actual_cols if c in merged_df.columns]
        if len(available) < len(actual_cols):
            missing = set(actual_cols) - set(available)
            logger.warning(
                f"[{exp_id}] Fold {val_fold}: "
                f"{len(missing)} features not found in DataFrame"
            )

        X_train = train_df[available].values.astype(float)
        y_train = train_df[CFG.target_col].values.astype(int)
        X_val = val_df[available].values.astype(float)
        y_val = val_df[CFG.target_col].values.astype(int)

        # ── Impute + Scale (train only) ───────────────────────────
        imputer, scaler = fit_scaler_imputer(X_train)
        X_train_proc = transform(X_train, imputer, scaler)
        X_val_proc = transform(X_val, imputer, scaler)

        # ── Fit model (fixed hyperparameters) ─────────────────────
        model = model_factory()
        model.fit(X_train_proc, y_train)

        # ── Calibrate ─────────────────────────────────────────────
        model_cal = calibrate_model(model, X_train_proc, y_train, method='sigmoid')

        # ── Predict ───────────────────────────────────────────────
        y_prob = model_cal.predict_proba(X_val_proc)[:, 1]

        # ── Threshold (Youden) ────────────────────────────────────
        threshold = tune_threshold_youden(y_val, y_prob)

        # ── Metrics ───────────────────────────────────────────────
        m_default = compute_metrics(y_val, y_prob, 0.5, prefix='default_')
        m_tuned = compute_metrics(y_val, y_prob, threshold, prefix='tuned_')

        fold_result = {
            **m_default, **m_tuned,
            'val_fold': val_fold,
            'n_features': len(available),
            'n_train': len(y_train),
            'n_val': len(y_val),
        }
        fold_metrics.append(fold_result)

    # ── Summarize ─────────────────────────────────────────────────
    summary = summarize_cv_results(fold_metrics)
    summary['exp_id'] = exp_id
    summary['n_features'] = len(actual_cols)
    summary['feature_list'] = actual_cols
    summary['status'] = 'OK'

    # Save
    pd.DataFrame(fold_metrics).to_csv(
        f"{results_dir}/{exp_id}_folds.csv", index=False
    )
    with open(f"{results_dir}/{exp_id}_summary.json", 'w') as f:
        json.dump(
            {k: v for k, v in summary.items()
             if not isinstance(v, (list, dict))},
            f, indent=2
        )

    logger.info(
        f"[{exp_id}] DONE | "
        f"AUROC={summary.get('default_auroc_mean', 0):.3f} ± "
        f"{summary.get('default_auroc_std', 0):.3f} | "
        f"AUPRC={summary.get('default_auprc_mean', 0):.3f} | "
        f"Sens={summary.get('tuned_sensitivity_mean', 0):.3f}"
    )
    return summary


def _resolve_feature_cols(
    feature_subset: List[str],
    feature_df: pd.DataFrame,
    cnn_df: Optional[pd.DataFrame]
) -> List[str]:
    """Resolve sentinel values and validate column presence."""
    all_available = set(feature_df.columns)
    if cnn_df is not None:
        all_available.update(cnn_df.columns)

    resolved = []
    for col in feature_subset:
        if col == '__CNN_EMBEDDING__':
            if cnn_df is not None:
                cnn_cols = [
                    c for c in cnn_df.columns
                    if c.startswith('cnn_embed_') or c.startswith('cnn_')
                ]
                resolved.extend(cnn_cols)
        else:
            if col in all_available:
                resolved.append(col)
    return list(dict.fromkeys(resolved))


def run_full_ablation_study(
    feature_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    model_factory,
    results_dir: str = "results/ablation",
    cnn_embedding_df: Optional[pd.DataFrame] = None,
    skip_metadata_experiments: bool = False
) -> pd.DataFrame:
    """
    Run all ablation experiments and produce the master results table.
    """
    feature_cols = [
        c for c in feature_df.columns
        if c not in ['patient_id', CFG.target_col,
                     'pipeline_status', 'n_valid_beats']
    ]
    groups = get_feature_groups(feature_cols)
    ablation_sets = get_ablation_feature_sets(feature_cols, groups)

    # Log group sizes
    logger.info("Feature group sizes:")
    for g, cols in groups.items():
        logger.info(f"  {g}: {len(cols)} features")

    all_summaries = []

    for exp_id, feat_cols in ablation_sets.items():

        # Skip metadata experiments if flagged
        if skip_metadata_experiments and exp_id in ['EXP_18_metadata_only',
                                                      'EXP_19_ecg_handcrafted_plus_metadata']:
            logger.info(f"Skipping metadata experiment: {exp_id}")
            continue

        summary = run_single_ablation_experiment(
            exp_id=exp_id,
            feature_subset=feat_cols,
            feature_df=feature_df,
            fold_df=fold_df,
            model_factory=model_factory,
            results_dir=results_dir,
            cnn_embedding_df=cnn_embedding_df
        )
        all_summaries.append(summary)

    # ── Master Results Table ──────────────────────────────────────
    results_df = _build_ablation_table(all_summaries)
    results_df.to_csv(f"{results_dir}/ablation_master_table.csv", index=False)
    logger.info(f"Ablation complete. Saved: {results_dir}/ablation_master_table.csv")

    return results_df


def _build_ablation_table(summaries: List[Dict]) -> pd.DataFrame:
    """Build the formatted ablation results DataFrame."""
    rows = []
    for s in summaries:
        if s.get('status') in ['SKIPPED_NO_CNN', 'FAILED_NO_FEATURES']:
            rows.append({
                'exp_id': s['exp_id'],
                'status': s['status'],
                'AUROC': np.nan, 'AUROC_std': np.nan,
                'AUPRC': np.nan, 'AUPRC_std': np.nan,
                'Sensitivity': np.nan, 'Specificity': np.nan,
                'F1_pos': np.nan, 'Bal_Acc': np.nan,
                'n_features': np.nan
            })
            continue

        rows.append({
            'exp_id':       s.get('exp_id', '?'),
            'status':       s.get('status', 'OK'),
            'n_features':   s.get('n_features', 0),
            'AUROC':        s.get('default_auroc_mean', np.nan),
            'AUROC_std':    s.get('default_auroc_std', np.nan),
            'AUPRC':        s.get('default_auprc_mean', np.nan),
            'AUPRC_std':    s.get('default_auprc_std', np.nan),
            'Sensitivity':  s.get('tuned_sensitivity_mean', np.nan),
            'Spec':         s.get('tuned_specificity_mean', np.nan),
            'F1_pos':       s.get('tuned_f1_positive_mean', np.nan),
            'Bal_Acc':      s.get('tuned_balanced_accuracy_mean', np.nan),
            'Brier':        s.get('cal_brier_score_mean', np.nan),
            'Sens@90Spec':  s.get('default_sens_at_90spec_mean', np.nan),
        })

    df = pd.DataFrame(rows)
    df['AUROC_fmt'] = df.apply(
        lambda r: f"{r['AUROC']:.3f} ± {r['AUROC_std']:.3f}"
        if not np.isnan(r['AUROC']) else 'N/A', axis=1
    )
    df['AUPRC_fmt'] = df.apply(
        lambda r: f"{r['AUPRC']:.3f} ± {r['AUPRC_std']:.3f}"
        if not np.isnan(r['AUPRC']) else 'N/A', axis=1
    )
    return df