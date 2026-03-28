"""
scripts/train_catboost_clean.py

CatBoost classifier untuk Brugada ECG — implementasi bersih dari nol.

DESIGN:
  - Re-derive dari feature_matrix.csv (bukan pre-computed arrays)
  - Tidak pakai VarianceThreshold (distorsi ST features)
  - Drop hanya kolom 100% NaN
  - Impute + Scale per fold (fit on train only)
  - Export OOF untuk semua model (CatBoost, LogReg, RF)
  - SHA256 verified sebelum apapun
  - patient_id selalu str
"""

import hashlib, json, os, warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, brier_score_loss
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================
FOLD_CSV        = 'data/splits/fold_assignments.csv'
FOLD_CSV_SHA256 = 'c8988aaa58ff7729ca36a7b54e7b6782f1ae2cd4e950dcf543fe120f11761904'
FEATURE_CSV     = 'features/feature_matrix.csv'
N_FOLDS         = 5
FAILED_SUBJECTS = {'267630', '1230482'}
SEED            = 42

# =============================================================================
# HELPERS
# =============================================================================

def verify_sha256(filepath, expected):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA256 MISMATCH: {filepath}\nExpected: {expected}\nActual: {actual}")
    print(f"[OK] SHA256 verified: {filepath}")


def compute_metrics(y_true, y_prob, target_spec=0.90):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx    = int(np.argmax(tpr - fpr))
    thresh = float(thresholds[idx])
    y_pred = (y_prob >= thresh).astype(int)

    tp = int(((y_pred==1)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())

    sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0

    valid = (1.0-fpr) >= target_spec
    sens_at_spec = float(tpr[valid].max()) if valid.any() else 0.0

    return {
        'auroc': auroc, 'auprc': auprc, 'brier': brier,
        'sensitivity': sens, 'specificity': spec,
        'threshold': thresh,
        'sensitivity_at_90pct_spec': sens_at_spec,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def print_fold(fold_id, m):
    print(f"  Fold {fold_id} | AUROC={m['auroc']:.4f} | "
          f"Sens={m['sensitivity']:.4f} | Spec={m['specificity']:.4f} | "
          f"TP={m['tp']} FN={m['fn']} FP={m['fp']}")


def print_summary(name, fold_metrics):
    aurocs = [m['auroc'] for m in fold_metrics]
    senss  = [m['sensitivity'] for m in fold_metrics]
    specs  = [m['specificity'] for m in fold_metrics]
    print(f"\n  {name}")
    print(f"    AUROC       : {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"    Sensitivity : {np.mean(senss):.4f} ± {np.std(senss):.4f}")
    print(f"    Specificity : {np.mean(specs):.4f} ± {np.std(specs):.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # 1. SHA256 check
    # -------------------------------------------------------------------------
    verify_sha256(FOLD_CSV, FOLD_CSV_SHA256)

    # -------------------------------------------------------------------------
    # 2. Load data
    # -------------------------------------------------------------------------
    df_folds = pd.read_csv(FOLD_CSV, dtype={'patient_id': str})
    df_feat  = pd.read_csv(FEATURE_CSV, dtype={'patient_id': str})

    # Merge fold info ke features
    df = df_folds[['patient_id','brugada','fold_id']].merge(
        df_feat.drop(columns=['brugada'], errors='ignore'),
        on='patient_id', how='left'
    )

    # -------------------------------------------------------------------------
    # 3. Drop kolom non-numerik dan 100% NaN
    #    TIDAK pakai VarianceThreshold — ST features punya variance kecil
    #    tapi sangat discriminative (st_slope AUROC=0.910)
    # -------------------------------------------------------------------------
    drop_always = ['patient_id', 'brugada', 'fold_id', 'pipeline_status']
    feature_cols_raw = [c for c in df.columns if c not in drop_always]

    # Drop kolom yang 100% NaN (tidak ada informasi sama sekali)
    cols_all_nan = [c for c in feature_cols_raw
                    if df[c].isna().all()]
    feature_cols = [c for c in feature_cols_raw if c not in cols_all_nan]

    print(f"\n[INFO] Total features       : {len(feature_cols_raw)}")
    print(f"[INFO] Dropped (all-NaN)    : {len(cols_all_nan)} → {cols_all_nan}")
    print(f"[INFO] Features untuk model : {len(feature_cols)}")

    # -------------------------------------------------------------------------
    # 4. Handle failed subjects
    # -------------------------------------------------------------------------
    failed_mask   = df['patient_id'].isin(FAILED_SUBJECTS)
    failed_info   = df.loc[failed_mask, ['patient_id','brugada','fold_id']]
    print(f"\n[INFO] Failed subjects:")
    print(failed_info.to_string(index=False))

    df_clean = df[~failed_mask].reset_index(drop=True)
    print(f"[INFO] Dataset bersih: {len(df_clean)} subjects "
          f"({int(df_clean['brugada'].sum())} Brugada, "
          f"{int((df_clean['brugada']==0).sum())} Normal)")

    y     = df_clean['brugada'].values.astype(int)
    folds = df_clean['fold_id'].values.astype(int)
    X_raw = df_clean[feature_cols].values.astype(np.float32)

    # -------------------------------------------------------------------------
    # 5. Setup OOF arrays untuk 3 model
    # -------------------------------------------------------------------------
    oof_catboost = np.full(len(df_clean), np.nan)
    oof_catboost2 = np.full(len(df_clean), np.nan)
    oof_logreg   = np.full(len(df_clean), np.nan)
    oof_rf       = np.full(len(df_clean), np.nan)

    fold_metrics_cb = []
    fold_metrics_cb2 = []
    fold_metrics_lr = []
    fold_metrics_rf = []

    # -------------------------------------------------------------------------
    # 6. 5-Fold loop
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  5-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")

    for fold_id in range(N_FOLDS):
        train_mask = folds != fold_id
        val_mask   = folds == fold_id

        X_train_raw = X_raw[train_mask]
        X_val_raw   = X_raw[val_mask]
        y_train     = y[train_mask]
        y_val       = y[val_mask]

        n_pos = y_train.sum()
        n_neg = (y_train == 0).sum()
        print(f"\n  Fold {fold_id} | train={train_mask.sum()} "
              f"(Brugada={n_pos}, Normal={n_neg}) | val={val_mask.sum()}")

        # --- Impute (fit on train only) ---
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp   = imputer.transform(X_val_raw)

        # --- Scale (fit on train only) ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_imp)
        X_val   = scaler.transform(X_val_imp)

        # Save scaler + imputer untuk hybrid stacking nanti
        os.makedirs(f'features/scaled/fold_{fold_id}', exist_ok=True)
        import pickle
        with open(f'features/scaled/fold_{fold_id}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(f'features/scaled/fold_{fold_id}/imputer.pkl', 'wb') as f:
            pickle.dump(imputer, f)

        # -----------------------------------------------------------------
        # MODEL A: CatBoost
        # -----------------------------------------------------------------
        cb = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=5,
            border_count=64,
            auto_class_weights='Balanced',
            eval_metric='AUC',
            early_stopping_rounds=50,
            random_seed=SEED,
            verbose=0,
        )
        cb.fit(X_train, y_train, eval_set=(X_val, y_val))
        oof_catboost[val_mask] = cb.predict_proba(X_val)[:, 1]

        m_cb = compute_metrics(y_val, oof_catboost[val_mask])
        fold_metrics_cb.append(m_cb)
        print(f"  CatBoost  ", end=""); print_fold(fold_id, m_cb)

        cb2 = CatBoostClassifier(
            learning_rate=0.1,
            l2_leaf_reg=3,
            iterations=100,
            depth=3,
            border_count=32,
            auto_class_weights='Balanced',
            eval_metric='AUC',
            early_stopping_rounds=50,
            random_seed=SEED,
            verbose=0
        )
        cb2.fit(X_train, y_train, eval_set=(X_val, y_val))
        oof_catboost2[val_mask] = cb2.predict_proba(X_val)[:, 1]

        m_cb2 = compute_metrics(y_val, oof_catboost2[val_mask])
        fold_metrics_cb2.append(m_cb2)
        print(f"  CatBoost Best Params  ", end=""); print_fold(fold_id, m_cb2)

        # -----------------------------------------------------------------
        # MODEL B: Logistic Regression
        # -----------------------------------------------------------------
        lr = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            solver='saga',
            penalty='l1',
            max_iter=2000,
            random_state=SEED,
        )
        lr.fit(X_train, y_train)
        oof_logreg[val_mask] = lr.predict_proba(X_val)[:, 1]

        m_lr = compute_metrics(y_val, oof_logreg[val_mask])
        fold_metrics_lr.append(m_lr)
        print(f"  LogReg    ", end=""); print_fold(fold_id, m_lr)

        # -----------------------------------------------------------------
        # MODEL C: Random Forest
        # -----------------------------------------------------------------
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        oof_rf[val_mask] = rf.predict_proba(X_val)[:, 1]

        m_rf = compute_metrics(y_val, oof_rf[val_mask])
        fold_metrics_rf.append(m_rf)
        print(f"  RF        ", end=""); print_fold(fold_id, m_rf)

    # -------------------------------------------------------------------------
    # 7. Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS (OOF Combined)")
    print(f"{'='*60}")

    for name, oof, fold_m in [
        ('CatBoost', oof_catboost, fold_metrics_cb),
        ('CatBoost Best Params', oof_catboost2, fold_metrics_cb2),
        ('LogReg',   oof_logreg,   fold_metrics_lr),
        ('RF',       oof_rf,       fold_metrics_rf),
    ]:
        oof_m = compute_metrics(y, oof)
        print_summary(f"{name} (per-fold mean ± std)", fold_m)
        print(f"    OOF combined AUROC: {oof_m['auroc']:.4f}")

    # -------------------------------------------------------------------------
    # 8. Export OOF predictions — schema seragam untuk Person 4
    # -------------------------------------------------------------------------
    os.makedirs('features', exist_ok=True)

    # Base DataFrame dengan semua 363 subjects (termasuk failed)
    df_all = df_folds[['patient_id', 'brugada', 'fold_id']].copy()
    df_all['patient_id'] = df_all['patient_id'].astype(str)

    # Index mapping dari df_clean ke df_all
    clean_ids = df_clean['patient_id'].astype(str).tolist()
    id_to_cb  = dict(zip(clean_ids, oof_catboost))
    id_to_cb2 = dict(zip(clean_ids, oof_catboost))  # CatBoost dengan best params
    id_to_lr  = dict(zip(clean_ids, oof_logreg))
    id_to_rf  = dict(zip(clean_ids, oof_rf))

    for model_name, id_to_oof, out_path in [
        ('CatBoost', id_to_cb, 'features/oof_predictions_catboost.csv'),
        ('CatBoost Best Params', id_to_cb2, 'features/oof_predictions_catboost_best.csv'),
        ('LogReg_L1_balanced', id_to_lr, 'features/oof_predictions_logreg.csv'),
        ('RandomForest', id_to_rf, 'features/oof_predictions_rf.csv'),
    ]:
        df_out = df_all.copy()
        df_out['oof_prob_brugada'] = df_out['patient_id'].map(id_to_oof)
        df_out['model_name']       = model_name
        # Failed subjects → NaN (sudah otomatis karena tidak ada di id_to_oof)
        df_out.to_csv(out_path, index=False)
        n_nan = df_out['oof_prob_brugada'].isna().sum()
        auroc_val = roc_auc_score(
            df_out.dropna(subset=['oof_prob_brugada'])['brugada'],
            df_out.dropna(subset=['oof_prob_brugada'])['oof_prob_brugada']
        )
        print(f"\n[OK] {out_path}")
        print(f"     Rows={len(df_out)} | NaN={n_nan} (failed subjects) | "
              f"OOF AUROC={auroc_val:.4f}")

    # -------------------------------------------------------------------------
    # 9. Save feature names yang dipakai (untuk audit)
    # -------------------------------------------------------------------------
    with open('features/catboost_feature_names.json', 'w') as f:
        json.dump({
            'feature_names': feature_cols,
            'n_features': len(feature_cols),
            'dropped_all_nan': cols_all_nan,
            'note': 'No variance threshold applied — ST features have low absolute variance but high discriminative power'
        }, f, indent=2)

    print(f"\n[OK] Feature manifest: features/catboost_feature_names.json")
    print(f"     {len(feature_cols)} features (no variance threshold)")
    print(f"\nDone. Jalankan hybrid_stacking.py setelah ini.")


if __name__ == '__main__':
    main()