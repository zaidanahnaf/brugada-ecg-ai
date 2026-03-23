# scripts/fix_label_encoding.py
"""
Re-encode brugada labels:
    0 → 0 (Normal)
    1 → 1 (Brugada Type 1)
    2 → 1 (Brugada Type 2 — treated as positive class)

This matches the original brief: 76 Brugada vs 287 Normal.
Type 2 subjects are included as positive class because:
    - They carry the same genetic substrate
    - Clinical management is similar
    - Brief explicitly states 76 positive cases (69+7=76)
"""

import pandas as pd

# Fix metadata
metadata = pd.read_csv("data/raw/brugada/metadata.csv")
metadata['brugada_original'] = metadata['brugada'].copy()
metadata['brugada'] = (metadata['brugada'] >= 1).astype(int)

print("=== LABEL ENCODING FIX ===")
print(f"Before: {metadata['brugada_original'].value_counts().to_dict()}")
print(f"After : {metadata['brugada'].value_counts().to_dict()}")
print(f"\nType 2 subjects re-encoded to 1:")
type2 = metadata[metadata['brugada_original'] == 2][
    ['patient_id', 'basal_pattern', 'sudden_death', 'brugada_original', 'brugada']
]
print(type2.to_string(index=False))

# Save corrected metadata
metadata.to_csv("data/raw/brugada/metadata.csv", index=False)
print("\nmetadata.csv updated.")

# Fix feature matrix
features = pd.read_csv("features/feature_matrix.csv")
features['brugada'] = (features['brugada'] >= 1).astype(int)
features.to_csv("features/feature_matrix.csv", index=False)
print("feature_matrix.csv updated.")

print(f"\nFinal: {features['brugada'].sum()} positive, {(features['brugada']==0).sum()} negative")