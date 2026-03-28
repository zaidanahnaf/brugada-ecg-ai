# scripts/audit_handcrafted_oof.py

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv('features/oof_predictions_handcrafted.csv', dtype={'patient_id': str})
df_folds = pd.read_csv('data/splits/fold_assignments.csv', dtype={'patient_id': str})

print("=== STRUKTUR FILE ===")
print(df.columns.tolist())
print(f"Shape: {df.shape}")
print(f"\nSample:\n{df.head()}")

print("\n=== DISTRIBUSI PROBABILITAS ===")
probs = df['oof_prob_brugada'].dropna()
print(f"Min    : {probs.min():.4f}")
print(f"Max    : {probs.max():.4f}")
print(f"Mean   : {probs.mean():.4f}")
print(f"NaN    : {df['oof_prob_brugada'].isna().sum()}")

print("\n=== AUROC PER MODEL ===")
for model in df['model_name'].unique():
    sub = df[df['model_name'] == model].dropna(subset=['oof_prob_brugada'])
    if len(sub) > 0:
        try:
            auc = roc_auc_score(sub['brugada'], sub['oof_prob_brugada'])
            print(f"  {model:<35} AUROC={auc:.4f}  n={len(sub)}")
        except:
            print(f"  {model:<35} AUROC=ERROR")

print("\n=== CEK DUPLIKAT PATIENT_ID ===")
dups = df['patient_id'].duplicated().sum()
print(f"Duplikat: {dups}")
if dups > 0:
    print("⚠️  Ada duplikat — mungkin multiple model per patient")
    print(df[df['patient_id'].duplicated(keep=False)][['patient_id','model_name']].head(10))
else:
    print("Tidak ada duplikat patient_id — setiap patient_id unik")
    
