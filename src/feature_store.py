# src/feature_store.py

"""
Feature store module.
Responsibilities:
    1. Build and save the feature manifest JSON
    2. Clean feature matrix (drop high-missingness columns)
    3. Fit imputer + scaler on training data only
    4. Transform any split using fitted objects
    5. Export fold-safe scaled matrices (.npy) for Person 2
    6. Export top-K reduced feature sets for hybrid fusion
    7. Save out-of-fold probability predictions for late fusion / stacking
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional

from src.config import CFG

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FEATURE MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

MANIFEST_SCHEMA = {
    "version": "1.0",
    "generated_by": "Person3_FeatureEngineer",
    "dataset": "Brugada-HUCA",
    "fs_hz": 100,
    "n_subjects_total": None,
    "n_subjects_ok": None,
    "n_subjects_partial": None,
    "n_subjects_failed": None,
    "feature_groups": {
        "A1_signal_quality": "Basic signal statistics and quality indicators",
        "A2_st_features":    "QRS, J-point, and ST segment measurements (V1-V3 priority)",
        "A3_morphology":     "Brugada-specific morphology (coved/saddleback/T-wave)",
        "A4_crosslead":      "Cross-lead relational features",
        "A5_generic":        "Generic secondary features (ablation only)",
        "QC":                "Pipeline quality control flags (excluded from ML input)"
    },
    "fold_dependent_features": [],
    "features": []
}


FEATURE_METADATA_REGISTRY = {
    "j_point_amplitude": {
        "description": "J-point elevation above PR-segment baseline",
        "formula": "V(J_sample) - median(V[PR_window])",
        "unit": "mV",
        "clinical_rationale": "Primary Brugada criterion: ≥2mm elevation in V1-V2",
        "direction": "higher->Brugada",
        "clinical_threshold": "≥0.2 mV (2mm)",
        "robustness_risk": "HIGH",
        "robustness_notes": "J-point localization ±10ms at 100Hz propagates error",
        "fold_dependent": False,
        "group": "A2_st_features"
    },
    "st_j40": {
        "description": "ST amplitude at J+40ms above baseline",
        "formula": "V(J+4samples) - baseline_mv",
        "unit": "mV",
        "clinical_rationale": "Standard ST measurement point; J+40ms is established",
        "direction": "higher->Brugada",
        "clinical_threshold": ">0.1 mV significant",
        "robustness_risk": "MEDIUM",
        "robustness_notes": "Less sensitive to J-point error than J+20ms",
        "fold_dependent": False,
        "group": "A2_st_features"
    },
    "st_slope_j0_j40": {
        "description": "ST slope from J-point to J+40ms",
        "formula": "(st_j40 - j_point_amplitude) / 40ms",
        "unit": "mV/ms",
        "clinical_rationale": "Negative = coved (Type 1); positive = saddleback (Type 2)",
        "direction": "more_negative->Brugada_Type1",
        "clinical_threshold": "<-0.0025 mV/ms = descending",
        "robustness_risk": "HIGH",
        "robustness_notes": "Sensitive to J-point anchor uncertainty",
        "fold_dependent": False,
        "group": "A2_st_features"
    },
    "st_convexity": {
        "description": "Mean second derivative of ST segment",
        "formula": "mean(d²V/dt²) over J to J+80ms window",
        "unit": "mV/ms²",
        "clinical_rationale": "Positive=convex=coved; negative=concave=saddleback",
        "direction": "more_positive->Brugada_Type1",
        "clinical_threshold": "sign change is clinically meaningful",
        "robustness_risk": "MEDIUM",
        "robustness_notes": "Second derivative amplifies noise; median agg mitigates",
        "fold_dependent": False,
        "group": "A2_st_features"
    },
    "st_monotonicity": {
        "description": "Fraction of ST samples monotonically decreasing",
        "formula": "count(dV/dt < 0) / total ST samples",
        "unit": "dimensionless",
        "clinical_rationale": "Pure descent = coved; mixed = saddleback",
        "direction": "higher->Brugada_Type1",
        "clinical_threshold": ">0.8 = predominantly monotone descent",
        "robustness_risk": "MEDIUM",
        "robustness_notes": "Noise can artificially reduce monotonicity",
        "fold_dependent": False,
        "group": "A2_st_features"
    },
    "st_area_above_baseline": {
        "description": "Area under ST curve above isoelectric baseline",
        "formula": "trapz(max(V[J:J+100ms] - baseline, 0))",
        "unit": "mV·samples",
        "clinical_rationale": "Integrates ST elevation burden over time",
        "direction": "higher->Brugada",
        "clinical_threshold": "Proportional to severity",
        "robustness_risk": "MEDIUM",
        "robustness_notes": "Robust to single-point J-point error",
        "fold_dependent": False,
        "group": "A2_st_features"
    },
    "covedness_score": {
        "description": "Composite coved morphology score",
        "formula": "weighted_sum(neg_slope, pos_convexity, t_inversion)",
        "unit": "dimensionless",
        "clinical_rationale": "Encodes three independent Type 1 criteria simultaneously",
        "direction": "higher->Brugada",
        "clinical_threshold": ">0.5 = coved-like",
        "robustness_risk": "HIGH",
        "robustness_notes": "Composite — errors in any component propagate",
        "fold_dependent": False,
        "group": "A3_morphology"
    },
    "saddleback_score": {
        "description": "Composite saddleback morphology score",
        "formula": "weighted_sum(pos_slope, neg_convexity, second_hump)",
        "unit": "dimensionless",
        "clinical_rationale": "High score = Type 2/3 or Normal variant, not Type 1",
        "direction": "higher->NOT_Brugada_Type1",
        "clinical_threshold": ">0.5 = saddleback-like",
        "robustness_risk": "HIGH",
        "robustness_notes": "Depends on second hump detection reliability",
        "fold_dependent": False,
        "group": "A3_morphology"
    },
    "t_inversion_indicator": {
        "description": "Binary T-wave inversion flag",
        "formula": "1 if mean(V[T_window]) < -0.05 mV",
        "unit": "binary",
        "clinical_rationale": "Defining Type 1 T-wave criterion after coved ST",
        "direction": "1->Brugada",
        "clinical_threshold": "-0.05 mV mean in T-window",
        "robustness_risk": "HIGH",
        "robustness_notes": "T-wave boundary detection at 100Hz is unreliable",
        "fold_dependent": False,
        "group": "A3_morphology"
    },
    "second_hump_present": {
        "description": "Presence of secondary local maximum in ST segment",
        "formula": "count(local_maxima in ST window, prominence>0.05mV) >= 2",
        "unit": "binary",
        "clinical_rationale": "Saddleback secondary peak — argues against Type 1",
        "direction": "1->NOT_Brugada_Type1",
        "clinical_threshold": "prominence threshold 0.05 mV",
        "robustness_risk": "MEDIUM",
        "robustness_notes": "100Hz may miss subtle humps; prominence threshold is tunable",
        "fold_dependent": False,
        "group": "A3_morphology"
    },
    "high_takeoff_gt_2mm": {
        "description": "Binary: J-point amplitude >= 0.2 mV",
        "formula": "1 if j_point_amplitude >= 0.2",
        "unit": "binary",
        "clinical_rationale": "Necessary criterion for any Brugada pattern type",
        "direction": "1->Brugada (necessary, not sufficient)",
        "clinical_threshold": "0.2 mV (2mm)",
        "robustness_risk": "MEDIUM",
        "robustness_notes": "Binary threshold sensitive to J-point localization error",
        "fold_dependent": False,
        "group": "A3_morphology"
    },
    "cl_max_st_j40_v1v3": {
        "description": "Maximum ST elevation at J+40ms across V1-V3",
        "formula": "max(st_j40_V1, st_j40_V2, st_j40_V3)",
        "unit": "mV",
        "clinical_rationale": "Peak right precordial ST burden",
        "direction": "higher->Brugada",
        "clinical_threshold": ">0.1 mV in any lead",
        "robustness_risk": "LOW",
        "robustness_notes": "Max aggregation robust to single-lead noise",
        "fold_dependent": False,
        "group": "A4_crosslead"
    },
    "cl_n_leads_t_inverted_v1v3": {
        "description": "Count of V1-V3 leads with T-wave inversion",
        "formula": "sum(t_inversion_indicator_V1, V2, V3)",
        "unit": "count [0-3]",
        "clinical_rationale": "Multi-lead T-wave involvement strengthens Type 1 criterion",
        "direction": "higher->Brugada",
        "clinical_threshold": ">=2 leads = stronger criterion",
        "robustness_risk": "MEDIUM",
        "robustness_notes": "Inherits T-wave detection uncertainty",
        "fold_dependent": False,
        "group": "A4_crosslead"
    },
    "qc_j_low_conf_rate": {
        "description": "Fraction of beats with low J-point detection confidence",
        "formula": "count(LOW_CONFIDENCE beats) / total valid beats",
        "unit": "dimensionless [0-1]",
        "clinical_rationale": "Quality gate — high rate means ST features unreliable",
        "direction": "higher->unreliable_features",
        "clinical_threshold": ">0.5 = majority of beats uncertain",
        "robustness_risk": "LOW",
        "robustness_notes": "Pure metadata — no target information",
        "fold_dependent": False,
        "group": "QC"
    },
}


def generate_full_manifest(
    feature_df: pd.DataFrame,
    fold_dependent_features: List[str] = None,
    save_path: str = "features/feature_manifest.json"
) -> Dict:
    """
    Generate and save the complete feature manifest JSON.

    Fills per-feature statistics from the actual feature_matrix.csv.
    Safe to call multiple times — overwrites previous manifest.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    target_meta = [
        'patient_id', CFG.target_col,
        'pipeline_status', 'n_valid_beats'
    ]
    feature_cols = [c for c in feature_df.columns if c not in target_meta]

    manifest = dict(MANIFEST_SCHEMA)
    manifest['n_subjects_total']   = len(feature_df)
    manifest['n_subjects_ok']      = int((feature_df['pipeline_status'] == 'OK').sum())
    manifest['n_subjects_partial'] = int((feature_df['pipeline_status'] == 'PARTIAL').sum())
    manifest['n_subjects_failed']  = int((feature_df['pipeline_status'] == 'FAILED').sum())
    manifest['fold_dependent_features'] = fold_dependent_features or []

    brugada_df = feature_df[feature_df[CFG.target_col] == 1]
    normal_df  = feature_df[feature_df[CFG.target_col] == 0]

    feature_entries = []
    for col in feature_cols:
        vals         = feature_df[col].replace([np.inf, -np.inf], np.nan)
        brugada_vals = brugada_df[col].replace([np.inf, -np.inf], np.nan)
        normal_vals  = normal_df[col].replace([np.inf, -np.inf], np.nan)

        parts = _parse_feature_name(col)
        meta  = _lookup_metadata(parts['base_name'])

        entry = {
            "feature_name":       col,
            "base_name":          parts['base_name'],
            "lead":               parts['lead'],
            "aggregation_stat":   parts['agg_stat'],
            "group":              meta.get('group', _infer_group(col)),
            "description":        meta.get('description', f"Auto-generated: {col}"),
            "formula":            meta.get('formula', 'See source code'),
            "unit":               meta.get('unit', 'unknown'),
            "clinical_rationale": meta.get('clinical_rationale', ''),
            "direction":          meta.get('direction', 'unknown'),
            "clinical_threshold": meta.get('clinical_threshold', ''),
            "robustness_risk":    meta.get('robustness_risk', 'MEDIUM'),
            "robustness_notes":   meta.get('robustness_notes', ''),
            "fold_dependent":     col in (fold_dependent_features or []),
            "n_missing":              int(vals.isna().sum()),
            "pct_missing":            float(100 * vals.isna().mean()),
            "mean":               float(vals.mean())  if vals.notna().any() else None,
            "std":                float(vals.std())   if vals.notna().any() else None,
            "min":                float(vals.min())   if vals.notna().any() else None,
            "max":                float(vals.max())   if vals.notna().any() else None,
            "pct_missing_brugada":float(100 * brugada_vals.isna().mean()),
            "pct_missing_normal": float(100 * normal_vals.isna().mean()),
        }
        feature_entries.append(entry)

    manifest['features']         = feature_entries
    manifest['n_features_total'] = len(feature_entries)

    with open(save_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=_json_safe)

    logger.info(
        f"Feature manifest saved: {save_path} "
        f"({len(feature_entries)} features)"
    )
    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE MATRIX CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def get_clean_feature_matrix(
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    max_missing_pct: float = 50.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop feature columns exceeding the missing-value threshold.

    Parameters
    ----------
    feature_df      : full feature matrix DataFrame
    feature_cols    : candidate ML feature column names
    max_missing_pct : columns with more missing than this are dropped

    Returns
    -------
    (cleaned_df, retained_feature_cols)
    """
    drop_cols = []
    for col in feature_cols:
        pct = 100 * feature_df[col].isna().mean()
        if pct > max_missing_pct:
            drop_cols.append(col)
            logger.debug(f"Dropping {col}: {pct:.1f}% missing")

    if drop_cols:
        logger.info(
            f"Dropped {len(drop_cols)} features with >{max_missing_pct}% missing"
        )

    retained = [c for c in feature_cols if c not in drop_cols]
    return feature_df[
        ['patient_id', CFG.target_col, 'pipeline_status', 'n_valid_beats']
        + retained
    ], retained


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — IMPUTATION AND SCALING
# ══════════════════════════════════════════════════════════════════════════════

def fit_scaler_imputer(
    X_train: np.ndarray
) -> Tuple[SimpleImputer, StandardScaler]:
    """
    Fit median imputer and standard scaler on training data ONLY.

    CRITICAL: Always call this on X_train, never on X_val or the full dataset.
    The returned objects are used to transform both train and val splits
    via the transform() function below.

    Returns
    -------
    (fitted_imputer, fitted_scaler)
    """
    imputer = SimpleImputer(strategy='median')
    X_imp   = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    scaler.fit(X_imp)

    return imputer, scaler


def transform(
    X: np.ndarray,
    imputer: SimpleImputer,
    scaler: StandardScaler
) -> np.ndarray:
    """
    Apply a FITTED imputer and scaler to any data split.

    Never refit here — only transform.
    """
    return scaler.transform(imputer.transform(X))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FOLD-SAFE SCALED MATRIX EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_scaled_feature_matrices(
    feature_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str = "features/scaled"
) -> None:
    """
    Export fold-safe standardized feature matrices for each CV fold.

    For each fold k produces:
        features/scaled/fold_k/train_X.npy
        features/scaled/fold_k/train_y.npy
        features/scaled/fold_k/train_ids.csv   ← CSV not .npy (string IDs)
        features/scaled/fold_k/val_X.npy
        features/scaled/fold_k/val_y.npy
        features/scaled/fold_k/val_ids.csv
        features/scaled/fold_k/imputer.pkl
        features/scaled/fold_k/scaler.pkl
        features/scaled/fold_k/feature_names.json

    CONTRACT FOR PERSON 2:
        - Imputer and scaler are fit on training split ONLY
        - Val split is transformed using training statistics
        - Load arrays with: np.load('train_X.npy')
        - Load IDs with: pd.read_csv('train_ids.csv')['patient_id'].values
        - Do NOT re-scale on Person 2's side
        - feature_names.json defines column order — preserve it exactly
    """
    from src.fold_manager import get_fold_split

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_folds = fold_df['fold_id'].nunique()

    for val_fold in range(n_folds):
        (X_train, y_train,
         X_val,   y_val,
         train_ids, val_ids,
         feat_names) = get_fold_split(fold_df, feature_df, val_fold)

        # Filter to requested feature columns
        feat_idx    = [i for i, f in enumerate(feat_names) if f in feature_cols]
        X_train_sub = X_train[:, feat_idx]
        X_val_sub   = X_val[:,   feat_idx]
        sub_names   = [feat_names[i] for i in feat_idx]

        # Fit on train ONLY
        imputer, scaler = fit_scaler_imputer(X_train_sub)
        X_train_scaled  = transform(X_train_sub, imputer, scaler)
        X_val_scaled    = transform(X_val_sub,   imputer, scaler)

        fold_dir = Path(output_dir) / f"fold_{val_fold}"
        fold_dir.mkdir(exist_ok=True)

        # Arrays
        np.save(fold_dir / "train_X.npy", X_train_scaled)
        np.save(fold_dir / "train_y.npy", y_train)
        np.save(fold_dir / "val_X.npy",   X_val_scaled)
        np.save(fold_dir / "val_y.npy",   y_val)

        # IDs as CSV (string-safe, no pickle needed)
        pd.Series(train_ids, name='patient_id').to_csv(
            fold_dir / "train_ids.csv", index=False
        )
        pd.Series(val_ids, name='patient_id').to_csv(
            fold_dir / "val_ids.csv", index=False
        )

        # Preprocessing objects
        with open(fold_dir / "imputer.pkl", 'wb') as f:
            pickle.dump(imputer, f)
        with open(fold_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        # Feature name manifest for this fold
        with open(fold_dir / "feature_names.json", 'w') as f:
            json.dump({
                'feature_names':    sub_names,
                'n_features':       len(sub_names),
                'val_fold':         val_fold,
                'n_train':          len(y_train),
                'n_val':            len(y_val),
                'n_positive_train': int(y_train.sum()),
                'n_positive_val':   int(y_val.sum()),
            }, f, indent=2)

        logger.info(
            f"Fold {val_fold}: "
            f"train={X_train_scaled.shape} | "
            f"val={X_val_scaled.shape}"
        )

    logger.info(f"Scaled matrices exported to: {output_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TOP-K REDUCED FEATURE SETS
# ══════════════════════════════════════════════════════════════════════════════

def export_top_k_feature_set(
    feature_df: pd.DataFrame,
    consensus_importance: pd.DataFrame,
    k_values: List[int] = None,
    output_dir: str = "features/reduced"
) -> Dict[int, List[str]]:
    """
    Export reduced feature sets based on consensus importance ranking.

    Produces for each k:
        features/reduced/top_k_{k}_feature_names.json
        features/reduced/top_k_{k}_feature_matrix.csv

    Rationale for reduction:
        363 subjects + 64-128 CNN dims + 121 HC features = overfit risk
        in hybrid fusion. Top-20 handcrafted is the recommended default.

    Parameters
    ----------
    feature_df           : full feature matrix
    consensus_importance : DataFrame with 'feature' and 'consensus_position' columns
                           (output of build_consensus_importance())
    k_values             : list of K values to export (default [10, 20, 35])
    """
    if k_values is None:
        k_values = [10, 20, 35]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    selected_sets = {}

    for k in k_values:
        top_features = (
            consensus_importance
            .sort_values('consensus_position')
            .head(k)['feature']
            .tolist()
        )
        selected_sets[k] = top_features

        # Feature name list
        with open(f"{output_dir}/top_k_{k}_feature_names.json", 'w') as f:
            json.dump({
                'k': k,
                'feature_names': top_features,
                'selection_method': 'consensus_rank_permutation_plus_native'
            }, f, indent=2)

        # Reduced feature matrix
        keep_cols = (
            ['patient_id', CFG.target_col]
            + [c for c in top_features if c in feature_df.columns]
        )
        feature_df[keep_cols].to_csv(
            f"{output_dir}/top_k_{k}_feature_matrix.csv", index=False
        )

        logger.info(f"Top-{k} features exported to {output_dir}/")

    return selected_sets


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OOF PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def save_oof_predictions(
    fold_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model_factory,
    feature_cols: List[str],
    model_name: str,
    save_path: str = "features/oof_predictions_handcrafted.csv"
) -> pd.DataFrame:
    """
    Generate and save out-of-fold probability predictions.

    OUTPUT SCHEMA:
        patient_id         : str   — subject identifier
        brugada            : int   — true label (0/1)
        oof_prob_brugada   : float — predicted P(Brugada), range [0, 1]
        fold_id            : int   — which fold this subject was val in
        model_name         : str   — model identifier string

    One row per subject. Every subject appears exactly once,
    in their held-out validation fold only.

    CRITICAL:
        These are the ONLY valid probabilities for stacking / late fusion.
        Training-set probabilities are overfit and must never be used.
        Person 2 uses this file for late fusion — fold_id must match
        fold_assignments.csv exactly.
    """
    from src.fold_manager import get_fold_split
    from experiments.calibration import calibrate_model

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    n_folds  = fold_df['fold_id'].nunique()
    all_preds = []

    for val_fold in range(n_folds):
        (X_train, y_train,
         X_val,   y_val,
         train_ids, val_ids,
         feat_names) = get_fold_split(fold_df, feature_df, val_fold)

        feat_idx    = [i for i, f in enumerate(feat_names) if f in feature_cols]
        X_train_sub = X_train[:, feat_idx]
        X_val_sub   = X_val[:,   feat_idx]

        imputer, scaler = fit_scaler_imputer(X_train_sub)
        X_train_proc    = transform(X_train_sub, imputer, scaler)
        X_val_proc      = transform(X_val_sub,   imputer, scaler)

        model     = model_factory()
        model.fit(X_train_proc, y_train)
        model_cal = calibrate_model(model, X_train_proc, y_train, method='sigmoid')

        fold_probs = model_cal.predict_proba(X_val_proc)[:, 1]

        for pid, prob, label in zip(val_ids, fold_probs, y_val):
            all_preds.append({
                'patient_id':       pid,
                CFG.target_col:     int(label),
                'oof_prob_brugada': float(prob),
                'fold_id':          val_fold,
                'model_name':       model_name
            })

        logger.info(
            f"OOF fold {val_fold}: "
            f"{len(val_ids)} subjects | "
            f"mean_prob={fold_probs.mean():.3f}"
        )

    oof_df = (
        pd.DataFrame(all_preds)
        .sort_values('patient_id')
        .reset_index(drop=True)
    )

    # Sanity check: one row per subject
    assert oof_df['patient_id'].nunique() == len(oof_df), \
        "Duplicate patient_ids in OOF predictions — check fold logic"

    # Sanity check: probabilities in valid range
    assert oof_df['oof_prob_brugada'].between(0, 1).all(), \
        "OOF probabilities outside [0, 1] — check calibration"

    oof_df.to_csv(save_path, index=False)
    logger.info(
        f"OOF predictions saved: {save_path} "
        f"({len(oof_df)} subjects, model={model_name})"
    )
    return oof_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_feature_name(col: str) -> Dict:
    """
    Decompose a feature column name into its constituent parts.

    Example:
        'st_V2_st_j40_median'
            -> base_name='st_V2_st_j40', lead='V2', agg_stat='median'
        'cl_max_st_j40_v1v3'
            -> base_name='cl_max_st_j40_v1v3', lead='multi', agg_stat='scalar'
    """
    leads     = ['V1','V2','V3','V4','V5','V6',
                 'I','II','III','aVR','aVL','aVF']
    agg_stats = ['mean','median','std','min','max','n_valid']

    lead     = 'multi'
    base     = col
    agg_stat = 'scalar'

    for l in leads:
        if f'_{l}_' in col or col.startswith(f'st_{l}_') \
                or col.startswith(f'morph_{l}_') \
                or col.startswith(f'sq_{l}'):
            lead = l
            break

    for stat in agg_stats:
        if col.endswith(f'_{stat}'):
            agg_stat = stat
            base     = col[:-(len(stat) + 1)]
            break

    return {'base_name': base, 'lead': lead, 'agg_stat': agg_stat}


def _lookup_metadata(base_name: str) -> Dict:
    """
    Match a feature base name to the closest entry in
    FEATURE_METADATA_REGISTRY using substring matching.
    Returns empty dict if no match found.
    """
    # Exact match first
    if base_name in FEATURE_METADATA_REGISTRY:
        return FEATURE_METADATA_REGISTRY[base_name]

    # Substring match — most specific key wins (longest match)
    matches = [
        (key, meta)
        for key, meta in FEATURE_METADATA_REGISTRY.items()
        if key in base_name
    ]
    if matches:
        best_key, best_meta = max(matches, key=lambda x: len(x[0]))
        return best_meta

    return {}


def _infer_group(col: str) -> str:
    """Infer feature group from column name prefix."""
    if col.startswith('sq_') or col.startswith('hr_') or col.startswith('rr_'):
        return 'A1_signal_quality'
    if col.startswith('st_'):
        return 'A2_st_features'
    if col.startswith('morph_'):
        return 'A3_morphology'
    if col.startswith('cl_'):
        return 'A4_crosslead'
    if col.startswith('qc_'):
        return 'QC'
    return 'A5_generic'


def _json_safe(obj):
    """JSON serialiser for numpy scalar types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)