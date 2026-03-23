# scripts/check_covedness.py

import pandas as pd
import numpy as np

df = pd.read_csv("features/feature_matrix.csv")

print("=== COVEDNESS SCORE INVESTIGATION ===\n")

# Cek komponen penyusun covedness
components = [
    'st_V1_st_slope_j0_j40_median',
    'st_V1_st_convexity_median',
    'morph_V1_t_inversion_indicator_median',
    'morph_V1_covedness_score_median',
    'morph_V1_saddleback_score_median',
]

print(f"{'Feature':<45} {'Brugada':>10} {'Normal':>10} {'Direction':>15}")
print("-" * 83)
for feat in components:
    if feat in df.columns:
        b = df[df['brugada']==1][feat].mean()
        n = df[df['brugada']==0][feat].mean()
        expected = "OK" if b > n else "REVERSED"
        # slope is expected to be more negative for Brugada
        if 'slope' in feat:
            expected = "OK" if b < n else "REVERSED"
        print(f"{feat:<45} {b:>10.4f} {n:>10.4f} {expected:>15}")
    else:
        print(f"{feat:<45} {'MISSING':>10}")

# Cek distribusi t_inversion
print("\n=== T-INVERSION DISTRIBUTION ===")
t_inv = 'morph_V1_t_inversion_indicator_median'
if t_inv in df.columns:
    print(f"Brugada  — any T-inversion (>0): {(df[df['brugada']==1][t_inv] > 0).sum()} / {(df['brugada']==1).sum()}")
    print(f"Normal   — any T-inversion (>0): {(df[df['brugada']==0][t_inv] > 0).sum()} / {(df['brugada']==0).sum()}")
    print(f"\nValue counts Brugada:\n{df[df['brugada']==1][t_inv].value_counts().head()}")
    print(f"\nValue counts Normal:\n{df[df['brugada']==0][t_inv].value_counts().head()}")