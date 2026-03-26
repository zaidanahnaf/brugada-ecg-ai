# scripts/feature_discriminability.py
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("features/feature_matrix.csv")
brugada = df[df['brugada'] == 1]
normal  = df[df['brugada'] == 0]

feature_cols = [c for c in df.columns
                if c not in ['patient_id','brugada',
                             'pipeline_status','n_valid_beats']
                and not c.startswith('qc_')
                and df[c].isna().mean() < 0.5]

rows = []
for feat in feature_cols:
    b_vals = brugada[feat].dropna()
    n_vals = normal[feat].dropna()
    if len(b_vals) < 5 or len(n_vals) < 5:
        continue
    stat, pval = stats.mannwhitneyu(b_vals, n_vals, alternative='two-sided')
    effect = (b_vals.mean() - n_vals.mean()) / (df[feat].std() + 1e-9)
    rows.append({
        'feature': feat,
        'brugada_mean': b_vals.mean(),
        'normal_mean':  n_vals.mean(),
        'effect_size':  effect,
        'abs_effect':   abs(effect),
        'p_value':      pval
    })

result = pd.DataFrame(rows).sort_values('abs_effect', ascending=False)
result.to_csv("results/feature_discriminability.csv", index=False)

print("Top 20 most discriminative features:")
print(result[['feature','brugada_mean','normal_mean',
              'effect_size','p_value']].head(20).to_string())
print(f"\nFeatures with |effect| > 0.5: {(result['abs_effect'] > 0.5).sum()}")
print(f"Features with |effect| > 0.3: {(result['abs_effect'] > 0.3).sum()}")
print(f"Features with |effect| < 0.1 (useless): {(result['abs_effect'] < 0.1).sum()}")