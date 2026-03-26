# scripts/audit_nan_features.py
import pandas as pd
import numpy as np

df = pd.read_csv("features/feature_matrix.csv")

feature_cols = [c for c in df.columns
                if c not in ['patient_id', 'brugada',
                             'pipeline_status', 'n_valid_beats']
                and not c.startswith('qc_')]

nan_rates = df[feature_cols].isna().mean().sort_values(ascending=False)

print("=== FITUR DENGAN NaN TINGGI ===\n")
print("100% NaN (completely useless):")
print(nan_rates[nan_rates == 1.0].index.tolist())

print(f"\n>80% NaN (practically useless): {(nan_rates > 0.8).sum()} features")
print(f">50% NaN (high missing): {(nan_rates > 0.5).sum()} features")
print(f">20% NaN (moderate missing): {(nan_rates > 0.2).sum()} features")
print(f"<5% NaN (good): {(nan_rates < 0.05).sum()} features")

print("\nTop 20 highest NaN features:")
print(nan_rates.head(20).to_string())