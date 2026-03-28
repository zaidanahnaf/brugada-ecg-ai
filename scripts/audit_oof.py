# scripts/audit_oof.py — jalankan ini dulu sebelum apapun

import pandas as pd
import numpy as np

df_folds = pd.read_csv('data/splits/fold_assignments.csv', dtype={'patient_id': str})
df_cnn   = pd.read_csv('features/cnn_fold_probs.csv',     dtype={'patient_id': str})

merged = df_folds.merge(df_cnn[['patient_id','fold_id']], 
                         on='patient_id', suffixes=('_true','_cnn'))

mismatch = merged[merged['fold_id_true'] != merged['fold_id_cnn']]

print(f"Total subjects   : {len(merged)}")
print(f"Fold mismatches  : {len(mismatch)}")

if len(mismatch) == 0:
    print("✅ Fold alignment OK — OOF tidak bocor")
else:
    print("❌ LEAKAGE DETECTED:")
    print(mismatch[['patient_id','fold_id_true','fold_id_cnn']].head(10))

df_cnn = pd.read_csv('features/cnn_fold_probs.csv', dtype={'patient_id': str})
df_cnn = df_cnn.dropna(subset=['oof_prob_brugada'])

print(f"\nTotal OOF rows   : {len(df_cnn)}  (expected: 361)")
print(f"Unique fold_ids  : {sorted(df_cnn['fold_id'].unique())}")
print(f"Brugada subjects : {df_cnn['brugada'].sum()}")

# Cek distribusi probabilitas
probs = df_cnn['oof_prob_brugada'].values
print(f"\nProb distribution:")
print(f"  Min    : {probs.min():.4f}")
print(f"  Max    : {probs.max():.4f}")
print(f"  Mean   : {probs.mean():.4f}")
print(f"  Median : {np.median(probs):.4f}")

# Cek separasi — model yang overfit akan punya prob sangat ekstrem
very_high = (probs > 0.95).sum()
very_low  = (probs < 0.05).sum()
print(f"\n  Prob > 0.95 : {very_high} subjects")
print(f"  Prob < 0.05 : {very_low} subjects")

# Cek apakah subject Brugada punya prob tinggi semua
brugada_probs = df_cnn[df_cnn['brugada']==1]['oof_prob_brugada'].values
normal_probs  = df_cnn[df_cnn['brugada']==0]['oof_prob_brugada'].values
print(f"\n  Brugada prob mean : {brugada_probs.mean():.4f}  (expected: high)")
print(f"  Normal prob mean  : {normal_probs.mean():.4f}  (expected: low)")
print(f"  Overlap (Brugada < 0.5) : {(brugada_probs < 0.5).sum()} subjects")
print(f"  Overlap (Normal > 0.5)  : {(normal_probs > 0.5).sum()} subjects")