# scripts/hybrid_late_fusion_proper.py
"""
Late fusion dan stacking yang proper menggunakan semua OOF files terbaru.

Strategy:
  - Late fusion primary  : CNN + LogReg (paling interpretable)
  - Late fusion extended : CNN + CatBoost, CNN + RF (untuk comparison)
  - Stacking             : LogReg meta pada [p_catboost, p_logreg, p_rf, p_cnn]
  - Method A (global)    : tune weight pada full OOF — reported with note
  - Method B (nested CV) : tune weight per fold — rigorous untuk paper
"""

import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ── Constants ────────────────────────────────────────────────
FOLD_CSV        = 'data/splits/fold_assignments.csv'
FOLD_CSV_SHA256 = 'c8988aaa58ff7729ca36a7b54e7b6782f1ae2cd4e950dcf543fe120f11761904'
FAILED_SUBJECTS = {'267630', '1230482'}
N_FOLDS         = 5

# ── Helpers ──────────────────────────────────────────────────
def verify_sha256(path, expected):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    actual = h.hexdigest()
    assert actual == expected, f"SHA256 MISMATCH: {actual}"
    print(f"[OK] SHA256 verified")

def compute_metrics(y_true, y_prob, target_spec=0.90):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx   = int(np.argmax(tpr - fpr))
    t     = float(thresholds[idx])
    pred  = (y_prob >= t).astype(int)
    tp = int(((pred==1)&(y_true==1)).sum())
    tn = int(((pred==0)&(y_true==0)).sum())
    fp = int(((pred==1)&(y_true==0)).sum())
    fn = int(((pred==0)&(y_true==1)).sum())
    sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    valid = (1-fpr) >= target_spec
    sens_at = float(tpr[valid].max()) if valid.any() else 0.0
    return dict(auroc=auroc, auprc=auprc, brier=brier,
                sensitivity=sens, specificity=spec, threshold=t,
                sens_at_90spec=sens_at,
                tp=tp, tn=tn, fp=fp, fn=fn)

def late_fusion_grid(y, p_a, p_b, label_a='A', label_b='B'):
    """Grid search optimal weight for p_final = w*p_a + (1-w)*p_b."""
    best_w, best_auc = 0.0, 0.0
    results = []
    for w in np.arange(0.0, 1.05, 0.05):
        auc = roc_auc_score(y, w*p_a + (1-w)*p_b)
        results.append((round(w, 2), round(auc, 6)))
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"  Best: w_{label_a}={best_w:.2f} w_{label_b}={1-best_w:.2f} "
          f"AUROC={best_auc:.4f}")
    return best_w, best_auc, results

def late_fusion_nested_cv(df, y, folds, col_a, col_b, label='fusion'):
    """Nested CV: tune weight on train folds, evaluate on val fold."""
    oof = np.zeros(len(df))
    fold_aurocs, fold_weights = [], []

    for val_fold in range(N_FOLDS):
        val_mask   = folds == val_fold
        train_mask = ~val_mask

        # tune on train folds
        best_w_fold, best_auc_fold = 0.0, 0.0
        for w in np.arange(0.0, 1.05, 0.05):
            p_tr = w*df.loc[train_mask, col_a] + (1-w)*df.loc[train_mask, col_b]
            auc  = roc_auc_score(y[train_mask], p_tr)
            if auc > best_auc_fold:
                best_auc_fold, best_w_fold = auc, w

        # apply to val
        oof[val_mask] = (best_w_fold * df.loc[val_mask, col_a] +
                         (1-best_w_fold) * df.loc[val_mask, col_b])
        val_auc = roc_auc_score(y[val_mask], oof[val_mask])
        fold_aurocs.append(val_auc)
        fold_weights.append(best_w_fold)
        print(f"    Fold {val_fold}: w={best_w_fold:.2f} | val AUROC={val_auc:.4f}")

    final_auc = roc_auc_score(y, oof)
    print(f"  Nested CV OOF AUROC: {final_auc:.4f} "
          f"({np.mean(fold_aurocs):.4f} +- {np.std(fold_aurocs):.4f})")
    print(f"  Weights per fold: {[round(w,2) for w in fold_weights]}")
    return oof, final_auc, np.mean(fold_aurocs), np.std(fold_aurocs)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():

    verify_sha256(FOLD_CSV, FOLD_CSV_SHA256)

    # ── Load all OOF files ────────────────────────────────────
    fold_df = pd.read_csv(FOLD_CSV, dtype={'patient_id': str})

    def load_oof(path, col_name):
        df = pd.read_csv(path, dtype={'patient_id': str})
        return df[['patient_id','oof_prob_brugada']].rename(
            columns={'oof_prob_brugada': col_name})

    df = fold_df[['patient_id','brugada','fold_id']].copy()
    df = df.merge(load_oof('features/oof_predictions_catboost_best.csv', 'p_catboost'), on='patient_id')
    df = df.merge(load_oof('features/oof_predictions_logreg.csv',        'p_logreg'),   on='patient_id')
    df = df.merge(load_oof('features/oof_predictions_rf.csv',            'p_rf'),       on='patient_id')
    df = df.merge(load_oof('features/cnn_fold_probs.csv',                'p_cnn'),      on='patient_id')

    # exclude failed subjects
    df = df[~df['patient_id'].isin(FAILED_SUBJECTS)].reset_index(drop=True)
    df = df.dropna(subset=['p_catboost','p_logreg','p_rf','p_cnn'])

    y     = df['brugada'].values.astype(int)
    folds = df['fold_id'].values.astype(int)

    print(f"\nDataset: {len(df)} subjects | "
          f"{y.sum()} Brugada | {(y==0).sum()} Normal\n")

    # ── Individual baselines ──────────────────────────────────
    print("=" * 60)
    print("INDIVIDUAL MODEL BASELINES (OOF)")
    print("=" * 60)
    for name, col in [('CatBoost', 'p_catboost'),
                      ('LogReg',   'p_logreg'),
                      ('RF',       'p_rf'),
                      ('CNN',      'p_cnn')]:
        auc = roc_auc_score(y, df[col])
        print(f"  {name:<12}: AUROC={auc:.4f}")

    # ── Late fusion — Method A (global, reported with note) ───
    print("\n" + "=" * 60)
    print("LATE FUSION — METHOD A (global weight, slightly optimistic)")
    print("=" * 60)

    print("\n  CNN + LogReg:")
    w_cnn_lr, auc_cnn_lr, _ = late_fusion_grid(
        y, df['p_cnn'], df['p_logreg'], 'cnn', 'logreg')

    print("\n  CNN + CatBoost:")
    w_cnn_cb, auc_cnn_cb, _ = late_fusion_grid(
        y, df['p_cnn'], df['p_catboost'], 'cnn', 'catboost')

    print("\n  CNN + RF:")
    w_cnn_rf, auc_cnn_rf, _ = late_fusion_grid(
        y, df['p_cnn'], df['p_rf'], 'cnn', 'rf')

    # 3-model fusion: CNN + best two HC
    print("\n  CNN + CatBoost + LogReg (3-model):")
    best_3m, best_3m_auc = 0, 0
    for w1 in np.arange(0.0, 1.05, 0.05):       # w_cnn
        for w2 in np.arange(0.0, 1.05-w1, 0.05): # w_catboost
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0: continue
            p = w1*df['p_cnn'] + w2*df['p_catboost'] + w3*df['p_logreg']
            auc = roc_auc_score(y, p)
            if auc > best_3m_auc:
                best_3m_auc = auc
                best_3m = (w1, w2, w3)
    print(f"  Best: w_cnn={best_3m[0]:.2f} w_catboost={best_3m[1]:.2f} "
          f"w_logreg={best_3m[2]:.2f} AUROC={best_3m_auc:.4f}")

    # ── Late fusion — Method B (nested CV, rigorous) ──────────
    print("\n" + "=" * 60)
    print("LATE FUSION — METHOD B (nested CV, rigorous for paper)")
    print("=" * 60)

    # Primary: CNN + LogReg
    print("\n  CNN + LogReg (nested CV):")
    oof_lr, auc_lr_final, auc_lr_mean, auc_lr_std = late_fusion_nested_cv(
        df, y, folds, 'p_cnn', 'p_logreg', 'cnn_logreg')

    # Secondary: CNN + CatBoost
    print("\n  CNN + CatBoost (nested CV):")
    oof_cb, auc_cb_final, auc_cb_mean, auc_cb_std = late_fusion_nested_cv(
        df, y, folds, 'p_cnn', 'p_catboost', 'cnn_catboost')

    # ── Stacking — all 4 models ───────────────────────────────
    print("\n" + "=" * 60)
    print("STACKING — LogReg meta on [p_catboost, p_logreg, p_rf, p_cnn]")
    print("=" * 60)

    STACK_FEATURES = ['p_catboost', 'p_logreg', 'p_rf', 'p_cnn']
    oof_stack  = np.zeros(len(df))
    stack_fold_metrics = []

    for val_fold in range(N_FOLDS):
        val_mask   = folds == val_fold
        train_mask = ~val_mask

        X_tr = df.loc[train_mask, STACK_FEATURES].values
        X_vl = df.loc[val_mask,   STACK_FEATURES].values
        y_tr = y[train_mask]
        y_vl = y[val_mask]

        meta = LogisticRegression(
            C=1.0, class_weight='balanced',
            solver='lbfgs', max_iter=1000, random_state=42)
        meta.fit(X_tr, y_tr)
        oof_stack[val_mask] = meta.predict_proba(X_vl)[:, 1]

        fold_auc = roc_auc_score(y_vl, oof_stack[val_mask])
        stack_fold_metrics.append(fold_auc)
        print(f"  Fold {val_fold}: AUROC={fold_auc:.4f}")

    stack_oof_auc = roc_auc_score(y, oof_stack)
    print(f"  OOF combined: {stack_oof_auc:.4f} "
          f"({np.mean(stack_fold_metrics):.4f} +- "
          f"{np.std(stack_fold_metrics):.4f})")

    # ── Final comparison ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL COMPARISON — ALL METHODS")
    print("=" * 60)
    results = {
        'CatBoost (HC, CV)':          0.922,
        'CNN standalone (OOF)':       float(roc_auc_score(y, df['p_cnn'])),
        'Late Fusion CNN+LR (A)':     auc_cnn_lr,
        'Late Fusion CNN+CB (A)':     auc_cnn_cb,
        'Late Fusion 3-model (A)':    best_3m_auc,
        'Late Fusion CNN+LR (B)':     auc_lr_final,
        'Late Fusion CNN+CB (B)':     auc_cb_final,
        'Stacking 4-model':           stack_oof_auc,
    }
    for method, auc in sorted(results.items(), key=lambda x: -x[1]):
        marker = ' ← BEST' if auc == max(results.values()) else ''
        print(f"  {method:<35}: {auc:.4f}{marker}")

    # ── Save ─────────────────────────────────────────────────
    best_method = max(results, key=results.get)
    summary = {
        'best_method':      best_method,
        'best_auroc':       float(results[best_method]),
        'all_results':      {k: float(v) for k, v in results.items()},
        'late_fusion_primary': {
            'method':       'CNN + LogReg weighted average',
            'method_a_weight_cnn': float(w_cnn_lr),
            'method_a_auroc':     float(auc_cnn_lr),
            'method_b_oof_auroc': float(auc_lr_final),
            'method_b_mean':      float(auc_lr_mean),
            'method_b_std':       float(auc_lr_std),
            'formula': f'p_final = {w_cnn_lr:.2f}*p_CNN + {1-w_cnn_lr:.2f}*p_LogReg',
            'note': 'Method A weight tuned on full OOF (slightly optimistic). '
                    'Method B nested CV is rigorous estimate for paper.'
        },
        'stacking': {
            'oof_auroc':  float(stack_oof_auc),
            'fold_mean':  float(np.mean(stack_fold_metrics)),
            'fold_std':   float(np.std(stack_fold_metrics)),
        }
    }

    import os
    os.makedirs('results/hybrid', exist_ok=True)

    with open('results/hybrid/final_fusion_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save best OOF predictions
    df_out = df[['patient_id','brugada','fold_id']].copy()
    df_out['p_catboost']   = df['p_catboost'].values
    df_out['p_logreg']     = df['p_logreg'].values
    df_out['p_rf']         = df['p_rf'].values
    df_out['p_cnn']        = df['p_cnn'].values
    df_out['p_late_fusion_lr']   = oof_lr
    df_out['p_late_fusion_cb']   = oof_cb
    df_out['p_stacking']         = oof_stack
    df_out.to_csv('results/hybrid/all_fusion_oof.csv', index=False)

    print(f"\n[OK] Saved: results/hybrid/final_fusion_summary.json")
    print(f"[OK] Saved: results/hybrid/all_fusion_oof.csv")
    print(f"\nRECOMMENDATION FOR PAPER:")
    print(f"  Report Method B (nested CV) as primary result")
    print(f"  Report Method A in supplementary with note: 'tuned on full OOF'")


if __name__ == '__main__':
    main()