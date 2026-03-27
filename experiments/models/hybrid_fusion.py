# experiments/hybrid_fusion.py

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.config.__init__ import CFG
from src.fold_manager import load_folds, get_fold_split
from src.feature_store import fit_scaler_imputer, transform
from src.evaluation.metrics import (
    compute_metrics, tune_threshold_youden, summarize_cv_results
)
from experiments.calibration import calibrate_model

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY A — EARLY FUSION
# concat(handcrafted_features, CNN_embedding) → single meta-learner
# ═══════════════════════════════════════════════════════════════════════

def run_early_fusion_cv(
    feature_df: pd.DataFrame,
    cnn_embedding_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    handcrafted_cols: List[str],
    cnn_embedding_cols: List[str],
    meta_model_factory: Callable,
    results_dir: str = "outputs/results/hybrid/early_fusion"
) -> Dict:
    """
    Early fusion: concatenate handcrafted features with CNN embeddings,
    then train a single meta-learner on the combined representation.

    ALIGNMENT CONTRACT:
        Both DataFrames must have 'patient_id' column.
        Inner join on patient_id — subjects missing from either are excluded.
        This exclusion must be reported and accounted for.

    SCALING CONTRACT:
        Handcrafted features: use Person 3 imputer + scaler (fit on train only)
        CNN embeddings: Person 2 provides pre-normalized embeddings
            OR Person 3 applies a separate StandardScaler (fit on train only)
        The two scaled representations are concatenated AFTER separate scaling.
        Do NOT apply a single shared scaler to the concatenated vector —
        this would allow CNN embedding statistics to affect handcrafted scaling.

    OVERFITTING RISK:
        With 363 subjects, concatenating 121 handcrafted + 64 CNN = 185 features
        is HIGH risk. Apply feature selection or dimensionality reduction
        before the meta-learner. Recommended: top-20 handcrafted + CNN embeddings.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Align subjects
    merged_df = feature_df.merge(
        cnn_embedding_df, on='patient_id', how='inner'
    )
    n_excluded = len(feature_df) - len(merged_df)
    if n_excluded > 0:
        logger.warning(
            f"Early fusion: {n_excluded} subjects excluded due to "
            f"missing CNN embeddings. Verify alignment with Person 2."
        )

    n_folds = fold_df['fold_id'].nunique()
    fold_metrics = []

    for val_fold in range(n_folds):
        train_ids = fold_df[fold_df['fold_id'] != val_fold]['patient_id'].values
        val_ids = fold_df[fold_df['fold_id'] == val_fold]['patient_id'].values

        # Restrict to subjects with both modalities
        train_ids = np.intersect1d(train_ids, merged_df['patient_id'].values)
        val_ids = np.intersect1d(val_ids, merged_df['patient_id'].values)

        train_df = merged_df[merged_df['patient_id'].isin(train_ids)]
        val_df = merged_df[merged_df['patient_id'].isin(val_ids)]

        y_train = train_df[CFG.data.target_col].values.astype(int)
        y_val = val_df[CFG.data.target_col].values.astype(int)

        # ── Branch 1: Handcrafted Features ───────────────────────
        hc_avail = [c for c in handcrafted_cols if c in merged_df.columns]
        X_hc_train = train_df[hc_avail].values.astype(float)
        X_hc_val = val_df[hc_avail].values.astype(float)

        imp_hc, scaler_hc = fit_scaler_imputer(X_hc_train)
        X_hc_train_sc = transform(X_hc_train, imp_hc, scaler_hc)
        X_hc_val_sc = transform(X_hc_val, imp_hc, scaler_hc)

        # ── Branch 2: CNN Embeddings ──────────────────────────────
        cnn_avail = [c for c in cnn_embedding_cols if c in merged_df.columns]
        X_cnn_train = train_df[cnn_avail].values.astype(float)
        X_cnn_val = val_df[cnn_avail].values.astype(float)

        # Person 2 may provide pre-normalized embeddings
        # If not, apply separate scaler here
        imp_cnn, scaler_cnn = fit_scaler_imputer(X_cnn_train)
        X_cnn_train_sc = transform(X_cnn_train, imp_cnn, scaler_cnn)
        X_cnn_val_sc = transform(X_cnn_val, imp_cnn, scaler_cnn)

        # ── Concatenate ───────────────────────────────────────────
        X_train_fused = np.hstack([X_hc_train_sc, X_cnn_train_sc])
        X_val_fused = np.hstack([X_hc_val_sc, X_cnn_val_sc])

        # ── Meta-learner ─────────────────────────────────────────
        meta_model = meta_model_factory()
        meta_model.fit(X_train_fused, y_train)
        meta_cal = calibrate_model(meta_model, X_train_fused, y_train, 'sigmoid')

        y_prob = meta_cal.predict_proba(X_val_fused)[:, 1]
        threshold = tune_threshold_youden(y_val, y_prob)

        fold_result = {
            **compute_metrics(y_val, y_prob, 0.5, 'default_'),
            **compute_metrics(y_val, y_prob, threshold, 'tuned_'),
            'val_fold': val_fold,
            'n_hc_features': len(hc_avail),
            'n_cnn_features': len(cnn_avail),
            'n_fused_features': X_train_fused.shape[1],
        }
        fold_metrics.append(fold_result)
        logger.info(
            f"  [Early Fusion] Fold {val_fold}: "
            f"AUROC={fold_result['default_auroc']:.3f} | "
            f"HC={len(hc_avail)} | CNN={len(cnn_avail)}"
        )

    summary = summarize_cv_results(fold_metrics)
    summary['fusion_strategy'] = 'early_fusion'
    summary['n_hc_features'] = len(handcrafted_cols)
    summary['n_cnn_features'] = len(cnn_embedding_cols)
    return summary


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY B — LATE FUSION
# weighted_average(P_handcrafted, P_CNN) using AUROC-based weights
# ═══════════════════════════════════════════════════════════════════════

def run_late_fusion_cv(
    oof_probs_handcrafted: pd.DataFrame,   # patient_id, oof_prob_brugada, fold_id
    oof_probs_cnn: pd.DataFrame,           # Same schema from Person 2
    fold_df: pd.DataFrame,
    weight_method: str = 'auroc_weighted',  # 'equal'|'auroc_weighted'|'optimized'
    results_dir: str = "outputs/results/hybrid/late_fusion"
) -> Dict:
    """
    Late fusion using OOF probability predictions from both branches.

    ALIGNMENT CONTRACT:
        Both DataFrames must have identical patient_id sets and fold_ids.
        Person 2 generates oof_probs_cnn.csv using the SAME fold_assignments.csv.
        Verify alignment before fusion.

    WEIGHT METHODS:
        equal           → simple average (0.5, 0.5)
        auroc_weighted  → weight by fold-level AUROC of each branch
        optimized       → grid search over weight alpha ∈ [0,1] on each fold
                          (use with caution: 5 folds = low power for optimization)

    CRITICAL:
        All OOF predictions must come from the held-out fold only.
        Using training-set predictions for late fusion is target leakage.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Align and verify
    aligned = _align_oof_predictions(
        oof_probs_handcrafted, oof_probs_cnn, fold_df
    )
    if aligned is None:
        return {'error': 'OOF alignment failed'}

    n_folds = fold_df['fold_id'].nunique()
    fold_metrics = []
    fold_weights = []

    for val_fold in range(n_folds):
        fold_data = aligned[aligned['fold_id'] == val_fold]

        y_val = fold_data[CFG.data.target_col].values.astype(int)
        p_hc = fold_data['oof_prob_brugada_hc'].values
        p_cnn = fold_data['oof_prob_brugada_cnn'].values

        # Determine weights
        if weight_method == 'equal':
            alpha = 0.5

        elif weight_method == 'auroc_weighted':
            from sklearn.metrics import roc_auc_score
            auroc_hc = roc_auc_score(y_val, p_hc)
            auroc_cnn = roc_auc_score(y_val, p_cnn)
            # Softmax-style weighting
            alpha = np.exp(auroc_hc) / (np.exp(auroc_hc) + np.exp(auroc_cnn))

        elif weight_method == 'optimized':
            alpha = _grid_search_fusion_weight(y_val, p_hc, p_cnn)

        else:
            raise ValueError(f"Unknown weight_method: {weight_method}")

        y_prob_fused = alpha * p_hc + (1 - alpha) * p_cnn
        threshold = tune_threshold_youden(y_val, y_prob_fused)

        fold_result = {
            **compute_metrics(y_val, y_prob_fused, 0.5, 'default_'),
            **compute_metrics(y_val, y_prob_fused, threshold, 'tuned_'),
            'val_fold': val_fold,
            'alpha_hc': float(alpha),
            'alpha_cnn': float(1 - alpha),
        }
        fold_metrics.append(fold_result)
        fold_weights.append(alpha)

        logger.info(
            f"  [Late Fusion] Fold {val_fold}: "
            f"AUROC={fold_result['default_auroc']:.3f} | "
            f"alpha_hc={alpha:.3f}"
        )

    summary = summarize_cv_results(fold_metrics)
    summary['fusion_strategy'] = f'late_fusion_{weight_method}'
    summary['mean_alpha_hc'] = float(np.mean(fold_weights))
    summary['std_alpha_hc'] = float(np.std(fold_weights))
    return summary


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY C — STACKING
# OOF probs from both branches → logistic meta-learner
# ═══════════════════════════════════════════════════════════════════════

def run_stacking_fusion_cv(
    oof_probs_handcrafted: pd.DataFrame,
    oof_probs_cnn: pd.DataFrame,
    fold_df: pd.DataFrame,
    results_dir: str = "outputs/results/hybrid/stacking"
) -> Dict:
    """
    Stacking fusion: train a logistic meta-learner on OOF probabilities
    from both branches.

    NESTED CV DESIGN:
        For each outer fold k:
            1. Use OOF probs from all OTHER folds as meta-training data
            2. Use OOF probs from fold k as meta-validation data
            3. Fit logistic meta-learner on meta-training
            4. Evaluate on meta-validation

        This is correct nested stacking — no fold k data leaks into
        the meta-learner training.

    WARNING:
        With 5 folds, each meta-training set has ~290 subjects.
        Use strong regularization (C=0.1) for the meta-learner.
        A 2-feature logistic regression (P_hc, P_cnn) is unlikely to overfit.

    INTERPRETATION:
        Meta-learner coefficients indicate the relative contribution
        of each branch after seeing the full CV evidence.
        coef[0] = weight on handcrafted branch
        coef[1] = weight on CNN branch
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    aligned = _align_oof_predictions(
        oof_probs_handcrafted, oof_probs_cnn, fold_df
    )
    if aligned is None:
        return {'error': 'OOF alignment failed'}

    n_folds = fold_df['fold_id'].nunique()
    fold_metrics = []
    meta_coefficients = []

    for val_fold in range(n_folds):
        # Meta-train: all folds EXCEPT val_fold
        meta_train = aligned[aligned['fold_id'] != val_fold]
        meta_val = aligned[aligned['fold_id'] == val_fold]

        X_meta_train = meta_train[
            ['oof_prob_brugada_hc', 'oof_prob_brugada_cnn']
        ].values
        y_meta_train = meta_train[CFG.data.target_col].values.astype(int)

        X_meta_val = meta_val[
            ['oof_prob_brugada_hc', 'oof_prob_brugada_cnn']
        ].values
        y_meta_val = meta_val[CFG.data.target_col].values.astype(int)

        # Simple logistic meta-learner
        meta_lr = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            max_iter=1000,
            random_state=CFG.feature.random_seed
        )
        meta_lr.fit(X_meta_train, y_meta_train)

        y_prob_stacked = meta_lr.predict_proba(X_meta_val)[:, 1]
        threshold = tune_threshold_youden(y_meta_val, y_prob_stacked)

        fold_result = {
            **compute_metrics(y_meta_val, y_prob_stacked, 0.5, 'default_'),
            **compute_metrics(y_meta_val, y_prob_stacked, threshold, 'tuned_'),
            'val_fold': val_fold,
            'meta_coef_hc': float(meta_lr.coef_[0][0]),
            'meta_coef_cnn': float(meta_lr.coef_[0][1]),
        }
        fold_metrics.append(fold_result)
        meta_coefficients.append(meta_lr.coef_[0])

        logger.info(
            f"  [Stacking] Fold {val_fold}: "
            f"AUROC={fold_result['default_auroc']:.3f} | "
            f"coef_hc={meta_lr.coef_[0][0]:.3f} | "
            f"coef_cnn={meta_lr.coef_[0][1]:.3f}"
        )

    mean_coefs = np.mean(meta_coefficients, axis=0)
    summary = summarize_cv_results(fold_metrics)
    summary['fusion_strategy'] = 'stacking'
    summary['mean_meta_coef_hc'] = float(mean_coefs[0])
    summary['mean_meta_coef_cnn'] = float(mean_coefs[1])
    summary['interpretation'] = (
        f"HC branch weight={mean_coefs[0]:.3f}, "
        f"CNN branch weight={mean_coefs[1]:.3f} "
        f"({'HC dominant' if mean_coefs[0] > mean_coefs[1] else 'CNN dominant'})"
    )
    return summary


# ── Helper Functions ──────────────────────────────────────────────────────

def _align_oof_predictions(
    oof_hc: pd.DataFrame,
    oof_cnn: pd.DataFrame,
    fold_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Align OOF predictions from both branches.
    Verifies fold_id consistency and subject overlap.
    """
    # Rename for clarity
    oof_hc = oof_hc.rename(columns={'oof_prob_brugada': 'oof_prob_brugada_hc'})
    oof_cnn = oof_cnn.rename(columns={'oof_prob_brugada': 'oof_prob_brugada_cnn'})

    # Merge on patient_id
    merged = oof_hc[['patient_id', CFG.data.target_col,
                       'oof_prob_brugada_hc', 'fold_id']].merge(
        oof_cnn[['patient_id', 'oof_prob_brugada_cnn', 'fold_id']],
        on=['patient_id', 'fold_id'],
        how='inner'
    )

    n_hc = len(oof_hc)
    n_cnn = len(oof_cnn)
    n_merged = len(merged)

    if n_merged < min(n_hc, n_cnn) * 0.95:
        logger.error(
            f"OOF alignment: only {n_merged}/{min(n_hc, n_cnn)} subjects aligned. "
            f"Check fold_id consistency between Person 2 and Person 3."
        )
        return None

    if n_merged < min(n_hc, n_cnn):
        logger.warning(
            f"OOF alignment: {min(n_hc,n_cnn)-n_merged} subjects dropped "
            f"due to missing CNN or handcrafted predictions."
        )

    # Verify label consistency
    label_mismatch = (merged[CFG.data.target_col] !=
                      oof_cnn.merge(merged[['patient_id']],
                      on='patient_id')[CFG.data.target_col].values
                      if CFG.data.target_col in oof_cnn.columns else False)

    logger.info(f"OOF alignment: {n_merged} subjects aligned successfully.")
    return merged


def _grid_search_fusion_weight(
    y_true: np.ndarray,
    p_hc: np.ndarray,
    p_cnn: np.ndarray,
    n_steps: int = 21
) -> float:
    """
    Grid search over alpha ∈ [0, 1] to find optimal fusion weight
    by maximizing AUROC on the current fold's validation set.

    NOTE: This is in-fold optimization — alpha is tuned on the same
    data it is evaluated on. Use only when n_folds ≥ 5 and treat
    result as slightly optimistic.
    """
    from sklearn.metrics import roc_auc_score
    alphas = np.linspace(0, 1, n_steps)
    best_alpha = 0.5
    best_auroc = 0.0

    for alpha in alphas:
        y_fused = alpha * p_hc + (1 - alpha) * p_cnn
        auroc = roc_auc_score(y_true, y_fused)
        if auroc > best_auroc:
            best_auroc = auroc
            best_alpha = alpha

    return float(best_alpha)

def save_integration_contract(
    path: str = "INTEGRATION_CONTRACT.md"
) -> None:
    """Save the integration contract to a markdown file."""
    contract = """
# HYBRID MODEL INTEGRATION CONTRACT
## Person 3 (Handcrafted Features) ↔ Person 2 (CNN)

---

## SECTION 1: SHARED GROUND TRUTH

FILE: data/splits/fold_assignments.csv
OWNER: Person 3
COLUMNS: patient_id, brugada, fold_id
FOLD RANGE: fold_id in {0, 1, 2, 3, 4}

RULE: This file MUST NOT be regenerated after initial creation.
      Both parties load this file from the shared repository root.

---

## SECTION 2: PERSON 3 DELIVERABLES TO PERSON 2

FILE 1: outputs/features/feature_matrix.csv
    Rows    : One per subject (363 total)
    Columns : patient_id, brugada, pipeline_status, n_valid_beats,
              [all handcrafted feature columns]
    Note    : Exclude pipeline_status == 'FAILED' rows before ML

FILE 2: outputs/features/feature_manifest.json
    Purpose : Complete feature registry with definitions and statistics

FILE 3: outputs/features/scaled/fold_{k}/
    Contents: train_X.npy, train_y.npy, train_ids.csv,
              val_X.npy, val_y.npy, val_ids.csv,
              imputer.pkl, scaler.pkl, feature_names.json
    Critical: DO NOT re-scale — use as-is

FILE 4: outputs/features/oof_predictions_handcrafted.csv
    Columns : patient_id, brugada, oof_prob_brugada, fold_id, model_name
    Critical: OOF only — no training predictions

FILE 5: outputs/features/reduced/top_k_{10,20,35}_feature_matrix.csv
    Note    : Use top_k_20 as default for early fusion

---

## SECTION 3: PERSON 2 DELIVERABLES TO PERSON 3

FILE 1: outputs/features/cnn_embeddings.csv
    Columns : patient_id, cnn_embed_0, ..., cnn_embed_{D-1}
    Critical: All 363 subjects — NaN for failures, do not drop rows

FILE 2: outputs/features/cnn_fold_probs.csv
    Columns : patient_id, brugada, oof_prob_brugada, fold_id
    Critical: Must use same fold_assignments.csv

FILE 3: outputs/results/cnn_cv_summary.json
    Contents: auroc_mean, auroc_std, auprc_mean, sensitivity_mean, etc.

---

## SECTION 4: FUSION EXPERIMENTS

Person 3 runs:
    EXP_20: CNN-only baseline
    EXP_21: CNN + all handcrafted (early fusion)
    EXP_22: CNN + ST+Morph only (early fusion)
    Late fusion: equal, auroc-weighted, optimized
    Stacking: logistic meta-learner

---

## SECTION 5: LEAKAGE PREVENTION CHECKLIST

[ ] fold_assignments.csv SHA256 matches agreed hash
[ ] No subject in both train and val of same fold
[ ] Scaler/imputer fitted ONLY on training data
[ ] CNN embedding extraction does NOT use val fold statistics
[ ] OOF predictions contain ONLY val-fold rows per subject
[ ] No metadata fields in primary features
[ ] Feature importance computed on OOF val data only

---

## SECTION 6: FAILURE HANDLING

IF CNN embeddings missing for some subjects:
    Fill with NaN — do not drop row
    Report count of imputed subjects

IF subject counts differ:
    Inner join on patient_id — report excluded subjects
    If >5% excluded: investigate before running fusion

IF fold assignments conflict:
    STOP — both parties reload from shared fold_assignments.csv
"""

    with open(path, 'w', encoding='utf-8') as f:
        f.write(contract)
    print(f"Integration contract saved: {path}")