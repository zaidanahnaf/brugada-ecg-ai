# experiments/interpretability/shap_analysis.py

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional

from src.config import CFG
from src.fold_manager import load_folds, get_fold_split
from src.feature_store import fit_scaler_imputer, transform
from experiments.calibration import calibrate_model

logger = logging.getLogger(__name__)


def compute_shap_oof(
    model_factory: Callable,
    feature_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str = 'tree',    # 'tree' | 'linear' | 'kernel'
    max_kernel_samples: int = 100,
    results_dir: str = "results/interpretability"
) -> Dict:
    """
    Compute SHAP values using out-of-fold predictions only.

    Strategy:
        tree   → shap.TreeExplainer (exact, fast)
        linear → shap.LinearExplainer (exact for linear models)
        kernel → shap.KernelExplainer (model-agnostic, slow)

    All SHAP values are computed on the VALIDATION fold only.
    The explainer is initialized with TRAINING data as background.

    Returns:
        shap_values_oof : np.ndarray (n_subjects, n_features)
        X_oof           : np.ndarray (n_subjects, n_features)
        y_oof           : np.ndarray (n_subjects,)
        patient_ids_oof : np.ndarray (n_subjects,)
        feature_names   : List[str]
    """
    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap")
        return {}

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    n_folds = fold_df['fold_id'].nunique()

    all_shap = []
    all_X = []
    all_y = []
    all_ids = []
    feature_names_used = None

    for val_fold in range(n_folds):
        logger.info(f"SHAP computation — fold {val_fold+1}/{n_folds}")

        (X_train, y_train,
         X_val,   y_val,
         train_ids, val_ids,
         feat_names) = get_fold_split(fold_df, feature_df, val_fold)

        feat_idx    = [i for i, f in enumerate(feat_names) if f in feature_cols]
        X_train_sub = X_train[:, feat_idx]
        X_val_sub   = X_val[:,   feat_idx]
        sub_names   = [feat_names[i] for i in feat_idx]

        if feature_names_used is None:
            feature_names_used = sub_names

        imputer, scaler = fit_scaler_imputer(X_train_sub)
        X_train_proc    = transform(X_train_sub, imputer, scaler)
        X_val_proc      = transform(X_val_sub,   imputer, scaler)

        # Fit model dulu sebelum apapun
        model = model_factory()
        model.fit(X_train_proc, y_train)

        # Sync feature names dengan actual features yang dipakai model
        if hasattr(model, 'n_features_in_'):
            n_actual   = model.n_features_in_
            sub_names  = sub_names[:n_actual]
            X_val_proc = X_val_proc[:, :n_actual]
            X_train_proc = X_train_proc[:, :n_actual]

        if feature_names_used is None or len(feature_names_used) != len(sub_names):
            feature_names_used = sub_names

        # Compute SHAP
        shap_vals = _compute_shap_for_fold(
            model=model,
            X_train=X_train_proc,
            X_val=X_val_proc,
            model_type=model_type,
            max_kernel_samples=max_kernel_samples,
            shap_module=shap,
            val_fold=val_fold
        )

        if shap_vals is None:
            continue

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)

        # Truncate ke n_actual kalau masih mismatch
        n_names = len(sub_names)
        if shap_vals.shape[1] != n_names:
            n = min(shap_vals.shape[1], n_names)
            shap_vals  = shap_vals[:, :n]
            sub_names  = sub_names[:n]

        all_shap.append(shap_vals)
        all_X.append(X_val_proc[:, :shap_vals.shape[1]])
        all_y.append(y_val)
        all_ids.append(val_ids)

        if len(all_shap) == 1:
            feature_names_used = sub_names

    if not all_shap:
        logger.error("No SHAP values computed — check model type and shap installation.")
        return {}

    # Stack all folds
    shap_oof = np.vstack(all_shap)
    X_oof = np.vstack(all_X)
    y_oof = np.concatenate(all_y)
    ids_oof = np.concatenate(all_ids)

    n_shap_features = shap_oof.shape[1]
    if feature_names_used and len(feature_names_used) != n_shap_features:
        logger.warning(
            f"SHAP feature name mismatch: "
            f"{len(feature_names_used)} names vs {n_shap_features} SHAP cols. "
            f"Truncating names to match."
        )
        feature_names_used = feature_names_used[:n_shap_features]

    # Save raw SHAP matrix
    shap_df = pd.DataFrame(shap_oof, columns=feature_names_used)
    shap_df['patient_id'] = ids_oof
    shap_df['label'] = y_oof
    shap_df.to_csv(f"{results_dir}/shap_values_oof.csv", index=False)

    result = {
        'shap_values': shap_oof,
        'X_oof': X_oof,
        'y_oof': y_oof,
        'patient_ids': ids_oof,
        'feature_names': feature_names_used
    }

    logger.info(
        f"SHAP OOF complete. "
        f"Shape: {shap_oof.shape} | "
        f"Subjects: {len(ids_oof)}"
    )
    return result


def _compute_shap_for_fold(
    model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    model_type: str,
    max_kernel_samples: int,
    shap_module,
    val_fold: int
):
    """Dispatch SHAP computation by model type."""
    try:
        if model_type == 'tree':
            explainer = shap_module.TreeExplainer(
                model,
                data=X_train,
                feature_perturbation='interventional'
            )
            return explainer.shap_values(X_val)

        elif model_type == 'linear':
            explainer = shap_module.LinearExplainer(
                model,
                masker=X_train
            )
            return explainer.shap_values(X_val)

        elif model_type == 'kernel':
            # KernelSHAP: sample background for speed
            background_size = min(max_kernel_samples, len(X_train))
            background = shap_module.kmeans(X_train, background_size)
            explainer = shap_module.KernelExplainer(
                model.predict_proba,
                background
            )
            # Sample val fold for kernel (slow)
            n_val_sample = min(50, len(X_val))
            idx = np.random.choice(len(X_val), n_val_sample, replace=False)
            shap_vals = explainer.shap_values(
                X_val[idx], nsamples=200, silent=True
            )
            # Expand back to full val set (missing rows = 0)
            full_shap = np.zeros((len(X_val), X_val.shape[1]))
            if isinstance(shap_vals, list):
                full_shap[idx] = shap_vals[1]
            else:
                full_shap[idx] = shap_vals
            return full_shap

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    except Exception as e:
        logger.error(f"SHAP failed on fold {val_fold}: {e}")
        return None


def compute_shap_summaries(
    shap_result: Dict,
    results_dir: str = "results/interpretability"
) -> Dict[str, pd.DataFrame]:
    """
    Compute aggregated SHAP summaries:
        1. Global mean |SHAP| per feature
        2. Class-stratified mean |SHAP|
        3. Lead-level SHAP (sum over features per lead)
        4. Feature group SHAP
    """
    if not shap_result:
        return {}

    shap_vals = shap_result['shap_values']
    y_oof = shap_result['y_oof']
    feat_names = shap_result['feature_names']

    # ── 1. Global mean |SHAP| ─────────────────────────────────────
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    global_summary = pd.DataFrame({
        'feature': feat_names,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': shap_vals.mean(axis=0),       # Signed: direction of effect
        'std_shap': shap_vals.std(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)
    global_summary['shap_rank'] = range(1, len(global_summary) + 1)

    # ── 2. Class-stratified mean |SHAP| ──────────────────────────
    brugada_mask = y_oof == 1
    normal_mask = y_oof == 0

    class_summary = pd.DataFrame({'feature': feat_names})
    class_summary['mean_abs_shap_brugada'] = (
        np.abs(shap_vals[brugada_mask]).mean(axis=0)
    )
    class_summary['mean_abs_shap_normal'] = (
        np.abs(shap_vals[normal_mask]).mean(axis=0)
    )
    class_summary['mean_shap_brugada'] = shap_vals[brugada_mask].mean(axis=0)
    class_summary['mean_shap_normal'] = shap_vals[normal_mask].mean(axis=0)
    class_summary['class_differential'] = (
        class_summary['mean_abs_shap_brugada'] -
        class_summary['mean_abs_shap_normal']
    )
    class_summary = class_summary.sort_values(
        'mean_abs_shap_brugada', ascending=False
    )

    # ── 3. Lead-level SHAP aggregation ───────────────────────────
    lead_shap = _aggregate_shap_by_lead(
        feat_names, mean_abs_shap, shap_vals, y_oof
    )

    # ── 4. Feature group SHAP ─────────────────────────────────────
    group_shap = _aggregate_shap_by_group(
        feat_names, mean_abs_shap, shap_vals
    )

    # Save all
    global_summary.to_csv(f"{results_dir}/shap_global_summary.csv", index=False)
    class_summary.to_csv(f"{results_dir}/shap_class_summary.csv", index=False)
    lead_shap.to_csv(f"{results_dir}/shap_lead_summary.csv", index=False)
    group_shap.to_csv(f"{results_dir}/shap_group_summary.csv", index=False)

    return {
        'global': global_summary,
        'class_stratified': class_summary,
        'lead': lead_shap,
        'group': group_shap
    }


def _aggregate_shap_by_lead(
    feat_names: List[str],
    mean_abs_shap: np.ndarray,
    shap_vals: np.ndarray,
    y_oof: np.ndarray
) -> pd.DataFrame:
    """Sum SHAP importance by lead."""
    lead_tags = ['V1','V2','V3','V4','V5','V6',
                 '_I_','_II_','_III_','aVR','aVL','aVF',
                 'cl_']   # cross-lead

    rows = []
    for lead in lead_tags:
        lead_idx = [
            i for i, f in enumerate(feat_names)
            if f'_{lead}_' in f or f.startswith(f'{lead}_') or lead in f
        ]
        if not lead_idx:
            continue
        total_shap = float(mean_abs_shap[lead_idx].sum())
        mean_shap_brugada = float(
            np.abs(shap_vals[y_oof==1][:, lead_idx]).mean()
        ) if (y_oof==1).sum() > 0 else np.nan
        rows.append({
            'lead': lead,
            'n_features': len(lead_idx),
            'total_mean_abs_shap': total_shap,
            'mean_abs_shap_per_feature': total_shap / max(len(lead_idx), 1),
            'mean_abs_shap_brugada': mean_shap_brugada,
        })

    return pd.DataFrame(rows).sort_values(
        'mean_abs_shap_per_feature', ascending=False
    )


def _aggregate_shap_by_group(
    feat_names: List[str],
    mean_abs_shap: np.ndarray,
    shap_vals: np.ndarray
) -> pd.DataFrame:
    """Sum SHAP importance by feature group prefix."""
    group_patterns = {
        'ST_features':      ['st_V1', 'st_V2', 'st_V3'],
        'Morphology':       ['morph_V1', 'morph_V2', 'morph_V3'],
        'Cross_lead':       ['cl_'],
        'Signal_quality':   ['sq_'],
        'HR_RR':            ['hr_', 'rr_'],
        'ST_other_leads':   ['st_V4','st_V5','st_V6','st_I','st_II',
                             'st_III','st_aV'],
        'Generic':          ['wavelet_','pca_','deriv_','spectral_'],
    }

    rows = []
    for group_name, patterns in group_patterns.items():
        idx = [
            i for i, f in enumerate(feat_names)
            if any(f.startswith(p) or p in f for p in patterns)
        ]
        if not idx:
            continue
        rows.append({
            'group': group_name,
            'n_features': len(idx),
            'total_mean_abs_shap': float(mean_abs_shap[idx].sum()),
            'mean_per_feature': float(mean_abs_shap[idx].mean()),
        })

    return pd.DataFrame(rows).sort_values('total_mean_abs_shap', ascending=False)