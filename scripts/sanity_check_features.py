# scripts/sanity_check_features.py

import pandas as pd
import numpy as np

df = pd.read_csv("outputs/features/feature_matrix.csv")

print("=" * 60)
print("FEATURE MATRIX SANITY CHECK")
print("=" * 60)
print(f"Shape              : {df.shape}")
print(f"Brugada positive   : {df['brugada'].sum()}")
print(f"Brugada negative   : {(df['brugada']==0).sum()}")
print(f"Failed subjects    : {(df['pipeline_status']=='FAILED').sum()}")
print(f"Partial subjects   : {(df['pipeline_status']=='PARTIAL').sum()}")
print(f"Overall missing %  : {df.isna().mean().mean()*100:.1f}%")
print(f"QC columns         : {sum(1 for c in df.columns if c.startswith('qc_'))}")

print("\n── Key feature comparison (Brugada vs Normal) ──────────")
key_feats = [
    'st_V1_j_point_amplitude_median',
    'st_V2_st_j40_median',
    'st_V1_st_slope_j0_j40_median',
    'morph_V1_covedness_score_median',
    'morph_V2_t_inversion_indicator_median',
    'cl_max_st_j40_v1v3',
]

print(f"{'Feature':<45} {'Brugada':>10} {'Normal':>10} {'Diff':>10}")
print("-" * 78)
for feat in key_feats:
    if feat in df.columns:
        b = df[df['brugada']==1][feat].mean()
        n = df[df['brugada']==0][feat].mean()
        print(f"{feat:<45} {b:>10.4f} {n:>10.4f} {b-n:>+10.4f}")
    else:
        print(f"{feat:<45} {'MISSING':>10}")

print("\n── Failed subjects ──────────────────────────────────────")
failed = df[df['pipeline_status']=='FAILED'][['patient_id','brugada','pipeline_status']]
print(failed.to_string(index=False))