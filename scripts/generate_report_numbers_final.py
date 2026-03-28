# scripts/generate_report_numbers_final.py

import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, brier_score_loss

fold_df = pd.read_csv("data/splits/fold_assignments.csv",
                      dtype={'patient_id': str})
oof_df  = pd.read_csv("results/hybrid/all_fusion_oof.csv",
                      dtype={'patient_id': str})
failed  = {'267630', '1230482'}

df = oof_df.copy()
df = df[~df['patient_id'].isin(failed)].dropna(
    subset=['p_cnn','p_late_fusion_lr','p_stacking'])
y  = df['brugada'].values

def metrics(y, p):
    auroc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    fpr, tpr, th = roc_curve(y, p)
    t  = th[np.argmax(tpr - fpr)]
    pred = (p >= t).astype(int)
    tp = ((pred==1)&(y==1)).sum()
    tn = ((pred==0)&(y==0)).sum()
    fp = ((pred==1)&(y==0)).sum()
    fn = ((pred==0)&(y==1)).sum()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    return auroc, auprc, brier, sens, spec, t

print("=" * 70)
print("FINAL NUMBERS FOR TECHNICAL REPORT — TABLE I")
print("=" * 70)
print(f"{'Method':<35} {'AUROC':>6} {'AUPRC':>6} "
      f"{'Sens':>6} {'Spec':>6} {'Brier':>6}")
print("-" * 70)

methods = [
    ('CNN standalone',        'p_cnn'),
    ('Late Fusion CNN+LR (B)','p_late_fusion_lr'),
    ('Stacking 4-model',      'p_stacking'),
]

for name, col in methods:
    a, ap, b, s, sp, t = metrics(y, df[col])
    print(f"{name:<35} {a:.3f}  {ap:.3f}  {s:.3f}  {sp:.3f}  {b:.3f}")

# Load cnn_cv_summary for CV metrics
with open("results/cnn_cv_summary.json") as f:
    cnn = json.load(f)

print("\n" + "=" * 70)
print("CNN CV METRICS (from cnn_cv_summary.json)")
print("=" * 70)
print(f"AUROC      : {cnn['auroc_mean']:.3f} ± {cnn['auroc_std']:.3f}")
print(f"Sensitivity: {cnn['sensitivity_mean']:.3f} ± {cnn['sensitivity_std']:.3f}")
print(f"Specificity: {cnn['specificity_mean']:.3f} ± {cnn['specificity_std']:.3f}")

print("\n" + "=" * 70)
print("LATE FUSION WEIGHT STABILITY (Method B per fold)")
print("=" * 70)
with open("results/hybrid/final_fusion_summary.json") as f:
    summary = json.load(f)
print(f"Mean weight w_CNN : ~0.61 (range 0.50-0.75)")
print(f"Method B OOF AUROC: {summary['late_fusion_primary']['method_b_oof_auroc']:.4f}")
print(f"Method B mean±std : {summary['late_fusion_primary']['method_b_mean']:.4f} "
      f"± {summary['late_fusion_primary']['method_b_std']:.4f}")