"""
scripts/hybrid_stacking.py

Meta-learner stacking: CNN OOF probs + Handcrafted OOF probs → CatBoost meta-learner.

DESIGN NOTES:
  - Fold assignments HARUS dari fold_assignments.csv (tidak dibuat ulang)
  - Meta-learner hanya menerima OOF probabilities sebagai fitur (bukan 689 fitur raw)
  - Failed subjects di-exclude secara eksplisit dan di-log
  - SHA256 diverifikasi sebelum apapun
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

# =============================================================================
# CONSTANTS
# =============================================================================

FOLD_CSV        = 'data/splits/fold_assignments.csv'
FOLD_CSV_SHA256 = 'c8988aaa58ff7729ca36a7b54e7b6782f1ae2cd4e950dcf543fe120f11761904'
N_FOLDS         = 5
FAILED_SUBJECTS = {'267630', '1230482'}

# =============================================================================
# HELPERS
# =============================================================================

def verify_sha256(filepath: str, expected: str) -> None:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"SHA256 MISMATCH on {filepath}\n"
            f"  Expected: {expected}\n"
            f"  Actual  : {actual}\n"
            "ABORTED — fold file mungkin sudah berubah."
        )
    print(f"[OK] SHA256 verified: {filepath}")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    target_spec: float = 0.90) -> dict:
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_idx   = int(np.argmax(tpr - fpr))
    best_thresh  = float(thresholds[youden_idx])
    y_pred       = (y_prob >= best_thresh).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Sensitivity at target specificity
    spec_arr     = 1.0 - fpr
    valid        = spec_arr >= target_spec
    sens_at_spec = float(tpr[valid].max()) if valid.any() else 0.0

    return {
        'auroc': auroc, 'auprc': auprc, 'brier': brier,
        'sensitivity': sensitivity, 'specificity': specificity,
        'threshold': best_thresh,
        f'sensitivity_at_{int(target_spec*100)}pct_spec': sens_at_spec,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # 1. Verifikasi SHA256 — HARD STOP jika gagal
    # -------------------------------------------------------------------------
    verify_sha256(FOLD_CSV, FOLD_CSV_SHA256)

    # -------------------------------------------------------------------------
    # 2. Load semua data dengan urutan yang benar
    # -------------------------------------------------------------------------
    df_folds = pd.read_csv(FOLD_CSV, dtype={'patient_id': str})

    df_logreg = pd.read_csv(
        'features/oof_predictions_handcrafted.csv',
        dtype={'patient_id': str}
    )[['patient_id', 'oof_prob_brugada']].rename(
        columns={'oof_prob_brugada': 'prob_logreg'}
    )

    df_cnn = pd.read_csv(
        'features/cnn_fold_probs.csv',   # ← nama file yang benar
        dtype={'patient_id': str}
    )[['patient_id', 'oof_prob_brugada']].rename(
        columns={'oof_prob_brugada': 'prob_cnn'}
    )

    # -------------------------------------------------------------------------
    # 3. Merge ke fold assignments sebagai base — bukan ke feature_matrix
    #    Meta-learner hanya butuh OOF probs, bukan 689 fitur raw
    # -------------------------------------------------------------------------
    df_meta = df_folds[['patient_id', 'brugada', 'fold_id']].copy()
    df_meta = df_meta.merge(df_logreg, on='patient_id', how='left')
    df_meta = df_meta.merge(df_cnn,    on='patient_id', how='left')

    # -------------------------------------------------------------------------
    # 4. Handle failed subjects — eksplisit, bukan diam-diam
    # -------------------------------------------------------------------------
    failed_mask  = df_meta['patient_id'].isin(FAILED_SUBJECTS)
    n_failed     = failed_mask.sum()
    failed_ids   = df_meta.loc[failed_mask, 'patient_id'].tolist()
    failed_labels = df_meta.loc[failed_mask, 'brugada'].tolist()

    print(f"\n[INFO] Excluding {n_failed} failed subjects: {failed_ids}")
    print(f"       Labels: {failed_labels}  (both Brugada-positive)")

    df_meta = df_meta[~failed_mask].reset_index(drop=True)

    # Sanity check — tidak ada NaN yang tidak diharapkan
    nan_logreg = df_meta['prob_logreg'].isna().sum()
    nan_cnn    = df_meta['prob_cnn'].isna().sum()
    if nan_logreg > 0 or nan_cnn > 0:
        bad_ids = df_meta.loc[
            df_meta['prob_logreg'].isna() | df_meta['prob_cnn'].isna(),
            'patient_id'
        ].tolist()
        raise ValueError(
            f"NaN ditemukan di luar failed subjects!\n"
            f"  prob_logreg NaN: {nan_logreg}\n"
            f"  prob_cnn NaN   : {nan_cnn}\n"
            f"  Patient IDs    : {bad_ids}\n"
            "Periksa apakah OOF files lengkap."
        )

    print(f"\n[INFO] Dataset bersih: {len(df_meta)} subjects "
          f"({df_meta['brugada'].sum()} Brugada, "
          f"{(df_meta['brugada']==0).sum()} Normal)")

    # -------------------------------------------------------------------------
    # 5. Feature matrix untuk meta-learner
    #    HANYA OOF probabilities — bukan 689 fitur raw
    # -------------------------------------------------------------------------
    META_FEATURES = ['prob_logreg', 'prob_cnn']
    X_meta = df_meta[META_FEATURES].values     # (N, 2)
    y      = df_meta['brugada'].values
    folds  = df_meta['fold_id'].values

    # -------------------------------------------------------------------------
    # 6. 5-Fold stacking — pakai fold_id dari fold_assignments.csv
    #    BUKAN StratifiedKFold baru!
    # -------------------------------------------------------------------------
    oof_final    = np.full(len(df_meta), np.nan)
    fold_metrics = []

    print(f"\n{'='*55}")
    print(f"  HYBRID STACKING — CatBoost Meta-Learner")
    print(f"  Features: {META_FEATURES}")
    print(f"{'='*55}")

    for fold_id in range(N_FOLDS):
        train_mask = folds != fold_id
        val_mask   = folds == fold_id

        X_train, X_val = X_meta[train_mask], X_meta[val_mask]
        y_train, y_val = y[train_mask],      y[val_mask]

        model = CatBoostClassifier(
            learning_rate=0.05,
            l2_leaf_reg=5,
            iterations=200,
            depth=2,              # ← sangat dangkal untuk N=2 fitur
            border_count=32,
            auto_class_weights='Balanced',
            eval_metric='AUC',
            random_seed=42,
            verbose=0,
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        oof_final[val_mask] = model.predict_proba(X_val)[:, 1]

        fold_m = compute_metrics(y_val, oof_final[val_mask])
        fold_m['fold_id'] = fold_id
        fold_metrics.append(fold_m)

        print(f"  Fold {fold_id} | AUROC={fold_m['auroc']:.4f} | "
              f"Sens={fold_m['sensitivity']:.4f} | "
              f"Spec={fold_m['specificity']:.4f}")

    # -------------------------------------------------------------------------
    # 7. OOF summary metrics
    # -------------------------------------------------------------------------
    assert not np.isnan(oof_final).any(), "Ada NaN di oof_final — cek fold loop"

    oof_metrics = compute_metrics(y, oof_final)

    auroc_per_fold = [m['auroc'] for m in fold_metrics]
    sens_per_fold  = [m['sensitivity'] for m in fold_metrics]
    spec_per_fold  = [m['specificity'] for m in fold_metrics]

    print(f"\n{'='*55}")
    print(f"  OOF COMBINED (363 subjects):")
    print(f"    AUROC       : {oof_metrics['auroc']:.4f}")
    print(f"    AUPRC       : {oof_metrics['auprc']:.4f}")
    print(f"    Sensitivity : {oof_metrics['sensitivity']:.4f}")
    print(f"    Specificity : {oof_metrics['specificity']:.4f}")
    print(f"    Brier       : {oof_metrics['brier']:.4f}")
    print(f"  PER-FOLD AUROC: {np.mean(auroc_per_fold):.4f} ± {np.std(auroc_per_fold):.4f}")
    print(f"{'='*55}")

    # -------------------------------------------------------------------------
    # 8. Simpan hasil
    # -------------------------------------------------------------------------
    os.makedirs('results', exist_ok=True)

    df_out = df_meta[['patient_id', 'brugada', 'fold_id',
                       'prob_logreg', 'prob_cnn']].copy()
    df_out['prob_stacking'] = oof_final
    df_out.to_csv('results/hybrid_stacking_oof.csv', index=False)

    summary = {
        'oof_combined': oof_metrics,
        'per_fold_auroc_mean': float(np.mean(auroc_per_fold)),
        'per_fold_auroc_std':  float(np.std(auroc_per_fold)),
        'per_fold_sensitivity_mean': float(np.mean(sens_per_fold)),
        'per_fold_specificity_mean': float(np.mean(spec_per_fold)),
        'meta_features': META_FEATURES,
        'failed_subjects_excluded': failed_ids,
        'fold_details': fold_metrics,
    }
    with open('results/hybrid_stacking_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Hasil disimpan:")
    print(f"     results/hybrid_stacking_oof.csv")
    print(f"     results/hybrid_stacking_summary.json")


if __name__ == '__main__':
    main()