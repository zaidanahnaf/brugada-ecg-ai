# scripts/diagnose_oof_files.py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

fold_df  = pd.read_csv("data/splits/fold_assignments.csv", dtype={'patient_id': str})
failed   = {'267630', '1230482'}

def load_oof(path, col):
    df = pd.read_csv(path, dtype={'patient_id': str})
    return df[['patient_id','oof_prob_brugada']].rename(
        columns={'oof_prob_brugada': col})

base = fold_df[['patient_id','brugada']].copy()
base = base.merge(load_oof('features/oof_predictions_catboost_best.csv','p_cb'), on='patient_id')
base = base.merge(load_oof('features/oof_predictions_logreg.csv','p_lr'),        on='patient_id')
base = base.merge(load_oof('features/oof_predictions_rf.csv','p_rf'),            on='patient_id')
base = base.merge(load_oof('features/cnn_fold_probs.csv','p_cnn'),               on='patient_id')
base = base[~base['patient_id'].isin(failed)].dropna()

y = base['brugada'].values

print("=== INDIVIDUAL AUROC ===")
for name, col in [('CatBoost','p_cb'),('LogReg','p_lr'),('RF','p_rf'),('CNN','p_cnn')]:
    auc = roc_auc_score(y, base[col])
    print(f"  {name:<12}: {auc:.4f}")

print("\n=== PROBABILITY DISTRIBUTIONS ===")
for name, col in [('CatBoost','p_cb'),('LogReg','p_lr'),('RF','p_rf'),('CNN','p_cnn')]:
    p = base[col]
    print(f"  {name:<12}: min={p.min():.3f} mean={p.mean():.3f} "
          f"max={p.max():.3f} std={p.std():.3f}")

print("\n=== CORRELATION BETWEEN MODELS ===")
for a, b in [('p_cb','p_cnn'),('p_lr','p_cnn'),('p_rf','p_cnn'),('p_cb','p_lr')]:
    corr = base[[a,b]].corr().iloc[0,1]
    print(f"  {a} vs {b}: r={corr:.3f}")

print("\n=== LATE FUSION GRID: CNN + EACH MODEL ===")
for name, col in [('CatBoost','p_cb'),('LogReg','p_lr'),('RF','p_rf')]:
    best_w, best_auc = 0, 0
    for w in np.arange(0, 1.05, 0.05):
        auc = roc_auc_score(y, w*base['p_cnn'] + (1-w)*base[col])
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"  CNN + {name:<12}: best AUROC={best_auc:.4f} at w_cnn={best_w:.2f}")