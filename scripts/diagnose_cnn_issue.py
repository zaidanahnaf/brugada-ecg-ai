# scripts/diagnose_cnn_issues.py

import pandas as pd
import numpy as np

hc  = pd.read_csv("features/oof_predictions_handcrafted.csv")
cnn = pd.read_csv("features/cnn_fold_probs.csv")

hc['patient_id']  = hc['patient_id'].astype(str)
cnn['patient_id'] = cnn['patient_id'].astype(str)

print("=== CNN PROB RANGE CHECK ===")
out_of_range = cnn[~cnn['oof_prob_brugada'].between(0, 1)]
print(f"Out of range: {len(out_of_range)} rows")
if len(out_of_range) > 0:
    print(out_of_range[['patient_id','oof_prob_brugada']].head(10))

print("\n=== LABEL CONSISTENCY CHECK ===")
merged = cnn.merge(hc[['patient_id','brugada']],
                   on='patient_id', suffixes=('_cnn','_hc'))
mismatch = merged[merged['brugada_cnn'] != merged['brugada_hc']]
print(f"Label mismatches: {len(mismatch)}")
if len(mismatch) > 0:
    print(mismatch[['patient_id','brugada_cnn','brugada_hc']].head(20))

print("\n=== CNN PROB DISTRIBUTION ===")
print(cnn['oof_prob_brugada'].describe())
print(f"\nNaN count: {cnn['oof_prob_brugada'].isna().sum()}")
print(f"< 0 count: {(cnn['oof_prob_brugada'] < 0).sum()}")
print(f"> 1 count: {(cnn['oof_prob_brugada'] > 1).sum()}")