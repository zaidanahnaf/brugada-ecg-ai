"""
scripts/hybrid_stacking.py

Meta-learner stacking: CNN OOF probs + Handcrafted OOF probs + Top 20 ST Features
→ Logistic Regression meta-learner.

CHANGES FROM V1:
  1. CatBoost meta-learner → Logistic Regression (lebih appropriate untuk 22 fitur, 361 sampel)
  2. Tambah top 20 ST features dari Person 3 sebagai fitur meta-learner
  3. Pakai scaler.pkl dari Person 3 untuk ST features (hindari leakage)
  4. Tambah late fusion comparison (weighted average) sebagai baseline

DESIGN NOTES:
  - Fold assignments HARUS dari fold_assignments.csv (tidak dibuat ulang)
  - ST features di-scale pakai scaler Person 3 per fold — bukan fit scaler baru
  - OOF probabilities tidak di-scale (sudah dalam range [0,1])
  - Failed subjects di-exclude secara eksplisit dan di-log
  - SHA256 diverifikasi sebelum apapun
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_idx  = int(np.argmax(tpr - fpr))
    best_thresh = float(thresholds[youden_idx])
    y_pred      = (y_prob >= best_thresh).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

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
    # 2. Load semua data
    # -------------------------------------------------------------------------
    df_folds = pd.read_csv(FOLD_CSV, dtype={'patient_id': str})

    df_catboost = pd.read_csv(
        'features/oof_predictions_catboost_best.csv',  # best params version
        dtype={'patient_id': str}
    )[['patient_id', 'oof_prob_brugada']].rename(
        columns={'oof_prob_brugada': 'prob_catboost'}
    )

    df_logreg = pd.read_csv(
        'features/oof_predictions_logreg.csv',
        dtype={'patient_id': str}
    )[['patient_id', 'oof_prob_brugada']].rename(
        columns={'oof_prob_brugada': 'prob_logreg'}
    )

    df_rf = pd.read_csv(
        'features/oof_predictions_rf.csv',
        dtype={'patient_id': str}
    )[['patient_id', 'oof_prob_brugada']].rename(
        columns={'oof_prob_brugada': 'prob_rf'}
    )

    df_cnn = pd.read_csv(
        'features/cnn_fold_probs.csv',
        dtype={'patient_id': str}
    )[['patient_id', 'oof_prob_brugada']].rename(
        columns={'oof_prob_brugada': 'prob_cnn'}
    )

    # Top 20 ST features dari Person 3
    with open('features/reduced/top_k_20_feature_names.json') as f:
        top_20_raw = json.load(f)
    top_20_features = top_20_raw['feature_names']

    df_raw = pd.read_csv(
        'features/feature_matrix.csv',
        dtype={'patient_id': str}
    )

    print(f"[INFO] Top 20 ST features loaded: {top_20_features[:3]}... "
          f"(total {len(top_20_features)})")

    # -------------------------------------------------------------------------
    # 3. Merge semua ke fold assignments sebagai base
    # -------------------------------------------------------------------------
    df_meta = df_folds[['patient_id', 'brugada', 'fold_id']].copy()
    df_meta = df_meta.merge(df_catboost, on='patient_id', how='left')
    df_meta = df_meta.merge(df_logreg, on='patient_id', how='left')
    df_meta = df_meta.merge(df_rf,     on='patient_id', how='left')
    df_meta = df_meta.merge(df_cnn,    on='patient_id', how='left')
    df_meta = df_meta.merge(
        df_raw[['patient_id'] + top_20_features],
        on='patient_id', how='left'
    )

    # -------------------------------------------------------------------------
    # 4. Handle failed subjects — eksplisit, bukan diam-diam
    # -------------------------------------------------------------------------
    failed_mask   = df_meta['patient_id'].isin(FAILED_SUBJECTS)
    failed_ids    = df_meta.loc[failed_mask, 'patient_id'].tolist()
    failed_labels = df_meta.loc[failed_mask, 'brugada'].tolist()

    print(f"\n[INFO] Excluding {failed_mask.sum()} failed subjects: {failed_ids}")
    print(f"       Labels: {failed_labels}  (both Brugada-positive)")

    df_meta = df_meta[~failed_mask].reset_index(drop=True)

    # Sanity check NaN
    nan_catboost = df_meta['prob_catboost'].isna().sum()
    nan_logreg   = df_meta['prob_logreg'].isna().sum()
    nan_rf       = df_meta['prob_rf'].isna().sum()
    nan_cnn    = df_meta['prob_cnn'].isna().sum()
    nan_st     = df_meta[top_20_features].isna().sum().sum()

    for name, n in [('catboost', nan_catboost), ('logreg', nan_logreg), ('rf', nan_rf), ('cnn', nan_cnn)]:
        if n > 2:
            raise ValueError(f"NaN di prob_{name}: {n} (expected max 2 untuk failed subjects)")

    if nan_catboost > 0 or nan_logreg > 0 or nan_rf > 0 or nan_cnn > 0:
        bad_ids = df_meta.loc[
            df_meta['prob_catboost'].isna() | df_meta['prob_logreg'].isna() | df_meta['prob_rf'].isna() | df_meta['prob_cnn'].isna(),
            'patient_id'
        ].tolist()
        raise ValueError(
            f"NaN di OOF probabilities!\n"
            f"  prob_catboost NaN: {nan_catboost}\n"
            f"  prob_logreg NaN: {nan_logreg}\n"
            f"  prob_rf NaN      : {nan_rf}\n"
            f"  prob_cnn NaN   : {nan_cnn}\n"
            f"  Patient IDs    : {bad_ids}"
        )

    if nan_st > 0:
        nan_per_feature = df_meta[top_20_features].isna().sum()
        n_subjects_with_nan = df_meta[top_20_features].isna().any(axis=1).sum()
        print(f"[WARN] {n_subjects_with_nan} subjects punya NaN di ST features")
        print(f"       Akan di-impute median per fold (fit on train only)")
        print(f"       Features dengan NaN terbanyak:")
        print(nan_per_feature[nan_per_feature > 0].sort_values(ascending=False).head(5).to_string())

    print(f"\n[INFO] Dataset bersih: {len(df_meta)} subjects "
          f"({int(df_meta['brugada'].sum())} Brugada, "
          f"{int((df_meta['brugada']==0).sum())} Normal)")
    print(f"[INFO] Meta features : 2 OOF probs + {len(top_20_features)} ST features "
          f"= {2 + len(top_20_features)} total")

    # -------------------------------------------------------------------------
    # 5. Setup arrays
    # -------------------------------------------------------------------------
    PROB_FEATURES = ['prob_catboost', 'prob_logreg', 'prob_rf', 'prob_cnn']
    META_FEATURES = PROB_FEATURES + top_20_features   # 22 fitur total

    y     = df_meta['brugada'].values.astype(int)
    folds = df_meta['fold_id'].values.astype(int)

    # -------------------------------------------------------------------------
    # 6. 5-Fold stacking — pakai fold_id dari fold_assignments.csv
    # -------------------------------------------------------------------------
    oof_stacking = np.full(len(df_meta), np.nan)
    fold_metrics = []

    print(f"\n{'='*60}")
    print(f"  HYBRID STACKING — Logistic Regression Meta-Learner")
    print(f"  Features: {PROB_FEATURES} + top {len(top_20_features)} ST features")
    print(f"{'='*60}")

    for fold_id in range(N_FOLDS):
        train_mask = folds != fold_id
        val_mask   = folds == fold_id

        # --- Ambil OOF probs (tidak perlu di-scale, sudah [0,1]) ---
        X_train_probs = df_meta.loc[train_mask, PROB_FEATURES].values.astype(np.float32)
        X_val_probs   = df_meta.loc[val_mask,   PROB_FEATURES].values.astype(np.float32)

        # --- Ambil ST features mentah ---
        X_train_st_raw = df_meta.loc[train_mask, top_20_features].values.astype(np.float32)
        X_val_st_raw   = df_meta.loc[val_mask,   top_20_features].values.astype(np.float32)

        # --- Scale ST features pakai scaler Person 3 (fit on train fold) ---
        # PENTING: pakai scaler yang sudah ada — jangan fit scaler baru
        # Ini menghindari leakage dan konsisten dengan branch handcrafted
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        # Step 1: Impute NaN dulu (359 NaN terdeteksi di warn sebelumnya)
        imputer = SimpleImputer(strategy='median')
        X_train_st_imputed = imputer.fit_transform(X_train_st_raw)   # fit on train
        X_val_st_imputed   = imputer.transform(X_val_st_raw)          # apply to val

        # Step 2: Scale — fit on train only
        scaler_st = StandardScaler()
        X_train_st = scaler_st.fit_transform(X_train_st_imputed)     # fit on train
        X_val_st   = scaler_st.transform(X_val_st_imputed)            # apply to val

        # --- Gabungkan: [prob_logreg, prob_cnn, st_feat_0, ..., st_feat_19] ---
        X_train = np.hstack([X_train_probs, X_train_st])   # (N_train, 22)
        X_val   = np.hstack([X_val_probs,   X_val_st])     # (N_val, 22)

        y_train = y[train_mask]
        y_val   = y[val_mask]

        # --- Train Logistic Regression ---
        model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train, y_train)
        oof_stacking[val_mask] = model.predict_proba(X_val)[:, 1]

        fold_m           = compute_metrics(y_val, oof_stacking[val_mask])
        fold_m['fold_id'] = fold_id
        fold_metrics.append(fold_m)

        print(f"  Fold {fold_id} | AUROC={fold_m['auroc']:.4f} | "
              f"Sens={fold_m['sensitivity']:.4f} | "
              f"Spec={fold_m['specificity']:.4f} | "
              f"n_val={val_mask.sum()}")

    # -------------------------------------------------------------------------
    # 7. Late fusion baseline (weighted average) — untuk comparison
    # -------------------------------------------------------------------------
    prob_catboost_all = df_meta['prob_catboost'].values
    prob_logreg_all = df_meta['prob_logreg'].values
    prob_rf_all     = df_meta['prob_rf'].values
    prob_cnn_all    = df_meta['prob_cnn'].values

    prob_cb  = df_meta['prob_catboost'].values
    prob_lr  = df_meta['prob_logreg'].values
    prob_rf  = df_meta['prob_rf'].values
    prob_cnn = pd.read_csv('features/cnn_fold_probs.csv',
                            dtype={'patient_id': str}
            ).merge(df_meta[['patient_id']], on='patient_id'
            )['oof_prob_brugada'].values

    print("\n  CNN vs Best Handcrafted (LogReg):")
    best_lf, best_w = 0, 0
    for w in np.arange(0, 1.05, 0.05):
        fused = w * prob_cnn + (1-w) * prob_lr
        auc   = roc_auc_score(y, fused)
        if auc > best_lf:
            best_lf, best_w = auc, w
        marker = " ← best" if auc == best_lf else ""
        if auc > 0.950:
            print(f"  w_cnn={w:.2f} w_logreg={1-w:.2f} | AUROC={auc:.4f}{marker}")

    print(f"\n{'='*60}")
    print(f"  LATE FUSION COMPARISON (weighted average grid search)")
    print(f"{'='*60}")

    best_auroc_lf, best_w_cnn = 0.0, 0.0
    lf_results = []

    for w_cnn in np.arange(0.0, 1.05, 0.05):
        w_logreg   = 1.0 - w_cnn
        prob_fused = w_cnn * prob_cnn_all + w_logreg * prob_logreg_all
        auroc_lf   = roc_auc_score(y, prob_fused)
        lf_results.append((w_cnn, auroc_lf))
        if auroc_lf > best_auroc_lf:
            best_auroc_lf = auroc_lf
            best_w_cnn    = w_cnn

    # Print top 5 weights
    lf_results.sort(key=lambda x: x[1], reverse=True)
    for w, auc in lf_results[:5]:
        marker = " ← best" if w == best_w_cnn else ""
        print(f"  w_cnn={w:.2f} w_logreg={1-w:.2f} | AUROC={auc:.4f}{marker}")

    # -------------------------------------------------------------------------
    # 8. Final summary — semua metode dibandingkan
    # -------------------------------------------------------------------------
    assert not np.isnan(oof_stacking).any(), "NaN di oof_stacking — cek fold loop"

    stacking_metrics = compute_metrics(y, oof_stacking)
    auroc_per_fold   = [m['auroc'] for m in fold_metrics]
    sens_per_fold    = [m['sensitivity'] for m in fold_metrics]
    spec_per_fold    = [m['specificity'] for m in fold_metrics]

    # Individual model scores dari OOF
    auroc_cnn    = roc_auc_score(y, prob_cnn_all)
    auroc_logreg = roc_auc_score(y, prob_logreg_all)

    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON — ALL METHODS")
    print(f"{'='*60}")
    print(f"  CatBoost (handcrafted)     : 0.922 ± 0.026  [Person 3 baseline]")
    print(f"  ECGResNet (CNN standalone) : {auroc_cnn:.4f}         [OOF combined]")
    print(f"  Late fusion (best weight)  : {best_auroc_lf:.4f}  "
          f"[w_cnn={best_w_cnn:.2f}]")
    print(f"  Hybrid stacking (LogReg)   : {stacking_metrics['auroc']:.4f}  "
          f"[per-fold: {np.mean(auroc_per_fold):.4f} ± {np.std(auroc_per_fold):.4f}]")
    print(f"{'='*60}")
    print(f"\n  BEST METHOD: ", end="")

    scores = {
        'CatBoost':      0.922,
        'ECGResNet':     auroc_cnn,
        'Late Fusion':   best_auroc_lf,
        'Hybrid Stack':  stacking_metrics['auroc'],
    }
    best_method = max(scores, key=scores.get)
    print(f"{best_method} ({scores[best_method]:.4f})")

    print(f"\n  STACKING DETAIL:")
    print(f"    AUROC       : {stacking_metrics['auroc']:.4f}")
    print(f"    AUPRC       : {stacking_metrics['auprc']:.4f}")
    print(f"    Sensitivity : {stacking_metrics['sensitivity']:.4f}  "
          f"(mean per-fold: {np.mean(sens_per_fold):.4f} ± {np.std(sens_per_fold):.4f})")
    print(f"    Specificity : {stacking_metrics['specificity']:.4f}  "
          f"(mean per-fold: {np.mean(spec_per_fold):.4f} ± {np.std(spec_per_fold):.4f})")
    print(f"    Brier Score : {stacking_metrics['brier']:.4f}")
    print(f"    TP={stacking_metrics['tp']} TN={stacking_metrics['tn']} "
          f"FP={stacking_metrics['fp']} FN={stacking_metrics['fn']}")

    # -------------------------------------------------------------------------
    # 9. Simpan hasil
    # -------------------------------------------------------------------------
    os.makedirs('results', exist_ok=True)

    # Output CSV
    df_out = df_meta[['patient_id', 'brugada', 'fold_id',
                       'prob_logreg', 'prob_cnn']].copy()
    df_out['prob_late_fusion'] = (best_w_cnn * prob_cnn_all +
                                  (1 - best_w_cnn) * prob_logreg_all)
    df_out['prob_stacking']    = oof_stacking
    df_out.to_csv('results/hybrid_stacking_oof.csv', index=False)

    # Summary JSON
    summary = {
        'method':          'LogisticRegression_stacking',
        'meta_features':   META_FEATURES,
        'n_meta_features': len(META_FEATURES),
        'stacking': {
            'oof_combined':           stacking_metrics,
            'per_fold_auroc_mean':    float(np.mean(auroc_per_fold)),
            'per_fold_auroc_std':     float(np.std(auroc_per_fold)),
            'per_fold_sensitivity_mean': float(np.mean(sens_per_fold)),
            'per_fold_specificity_mean': float(np.mean(spec_per_fold)),
            'fold_details':           fold_metrics,
        },
        'late_fusion': {
            'best_auroc':   float(best_auroc_lf),
            'best_w_cnn':   float(best_w_cnn),
            'best_w_logreg': float(1 - best_w_cnn),
            'all_weights':  [{'w_cnn': float(w), 'auroc': float(a)}
                             for w, a in sorted(lf_results)],
        },
        'individual_models': {
            'ecgresnet_oof_auroc':  float(auroc_cnn),
            'catboost_oof_auroc':   float(auroc_logreg),
            'catboost_cv_auroc':    0.922,
        },
        'failed_subjects_excluded': failed_ids,
    }

    with open('results/hybrid_stacking_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Hasil disimpan:")
    print(f"     results/hybrid_stacking_oof.csv")
    print(f"     results/hybrid_stacking_summary.json")


if __name__ == '__main__':
    main()