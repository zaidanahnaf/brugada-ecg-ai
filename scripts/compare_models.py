# scripts/compare_models.py
# Bandingan Simple CNN vs ECGResNet dari cnn_cv_summary.json

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

# ── Load hasil training kamu ───────────────────────────────────────────────
with open('results/cnn_cv_summary.json') as f:
    summary = json.load(f)

df = pd.read_csv('features/cnn_fold_probs.csv', dtype={'patient_id': str})
df = df.dropna(subset=['oof_prob_brugada'])

# ── Hitung per-fold metrics ────────────────────────────────────────────────
fold_aurocs, fold_sens, fold_spec = [], [], []

for fold_id in sorted(df['fold_id'].unique()):
    fold_df = df[df['fold_id'] == fold_id]
    y_true  = fold_df['brugada'].values
    y_prob  = fold_df['oof_prob_brugada'].values

    auroc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx   = np.argmax(tpr - fpr)
    thresh = thresholds[idx]
    y_pred = (y_prob >= thresh).astype(int)

    tp = ((y_pred==1)&(y_true==1)).sum()
    tn = ((y_pred==0)&(y_true==0)).sum()
    fp = ((y_pred==1)&(y_true==0)).sum()
    fn = ((y_pred==0)&(y_true==1)).sum()

    fold_aurocs.append(auroc)
    fold_sens.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)
    fold_spec.append(tn/(tn+fp) if (tn+fp)>0 else 0.0)

cnn_auroc = np.mean(fold_aurocs)
cnn_std   = np.std(fold_aurocs)

# ── Referensi ─────────────────────────────────────────────────────────────
CATBOOST_AUROC = 0.922
CATBOOST_STD   = 0.026
HYBRID_AUROC   = 0.937
HYBRID_STD     = 0.043

# ── Diagnosis otomatis ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("  MODEL COMPARISON")
print("="*55)
print(f"  CatBoost (handcrafted) : {CATBOOST_AUROC:.3f} ± {CATBOOST_STD:.3f}")
print(f"  ECGResNet (CNN branch) : {cnn_auroc:.3f} ± {cnn_std:.3f}  ← hasil kamu")
print(f"  Hybrid stacking        : {HYBRID_AUROC:.3f} ± {HYBRID_STD:.3f}")
print("="*55)

# Diagnosis
print("\n  DIAGNOSIS:")

if cnn_auroc >= 0.92:
    print("  ✅ CNN sangat kuat — sebanding dengan CatBoost")
    print("     Hybrid fusion akan sangat menjanjikan")
elif cnn_auroc >= 0.88:
    print("  ✅ CNN kuat — memenuhi semua minimum targets")
    print("     CNN memberi sinyal komplementer ke CatBoost")
elif cnn_auroc >= 0.85:
    print("  ⚠️  CNN cukup — memenuhi minimum target (>0.85)")
    print("     Tapi headroom untuk improvement masih ada")
    print("     Coba: lebih banyak epochs, lr schedule, atau augmentasi lebih")
elif cnn_auroc >= 0.80:
    print("  ⚠️  CNN di bawah target — revisit diperlukan")
    print("     Kemungkinan penyebab: overfitting atau preprocessing issue")
    print("     Coba: kurangi depth ResNet, tambah dropout, cek ST distortion")
else:
    print("  ❌ CNN jauh di bawah target (<0.80)")
    print("     Kemungkinan ada bug di preprocessing atau data loading")
    print("     Wajib cek: apakah V1/V2/V3 indices benar?")

# Apakah CNN komplementer ke CatBoost?
correlation_proxy = abs(cnn_auroc - CATBOOST_AUROC)
print(f"\n  COMPLEMENTARITY:")
if correlation_proxy > 0.05:
    print(f"  ✅ Perbedaan AUROC {correlation_proxy:.3f} → kemungkinan besar komplementer")
    print(f"     Hybrid fusion berpotensi signifikan")
else:
    print(f"  ℹ️  AUROC mirip ({correlation_proxy:.3f} diff) → mungkin belajar hal serupa")
    print(f"     Hybrid masih bisa membantu tapi gain lebih kecil")

print(f"\n  SENSITIVITY: {np.mean(fold_sens):.3f} ± {np.std(fold_sens):.3f}")
print(f"  SPECIFICITY: {np.mean(fold_spec):.3f} ± {np.std(fold_spec):.3f}")

# Warning variance tinggi
if cnn_std > 0.05:
    print(f"\n  ⚠️  Variance tinggi (std={cnn_std:.3f}) — wajar untuk dataset kecil")
    print(f"     Tapi laporkan ini di technical report")

print("="*55)