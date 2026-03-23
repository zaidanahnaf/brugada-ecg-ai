# scripts/check_label_discrepancy.py

import pandas as pd

metadata = pd.read_csv("data/raw/brugada/metadata.csv")
features = pd.read_csv("features/feature_matrix.csv")

print("=== LABEL DISCREPANCY CHECK ===\n")
print(f"Metadata  — positive: {metadata['brugada'].sum()}, negative: {(metadata['brugada']==0).sum()}")
print(f"Features  — positive: {features['brugada'].sum()}, negative: {(features['brugada']==0).sum()}")
print(f"\nDifference: {int(features['brugada'].sum()) - int(metadata['brugada'].sum())} extra positives in feature matrix")

# Cek apakah ada nilai selain 0 dan 1
print(f"\nUnique brugada values in metadata : {sorted(metadata['brugada'].unique())}")
print(f"Unique brugada values in features : {sorted(features['brugada'].unique())}")

# Cari subjek yang labelnya beda
merged = metadata[['patient_id','brugada']].merge(
    features[['patient_id','brugada']],
    on='patient_id',
    suffixes=('_metadata', '_features')
)
different = merged[merged['brugada_metadata'] != merged['brugada_features']]
print(f"\nSubjects with different labels: {len(different)}")
if len(different) > 0:
    print(different.to_string(index=False))

# Cek apakah metadata punya nilai 2 (mungkin Type 2 Brugada)
if 2 in metadata['brugada'].values:
    type2 = metadata[metadata['brugada'] == 2]
    print(f"\nSubjects with brugada=2 in metadata: {len(type2)}")
    print(type2[['patient_id','brugada']].head(10).to_string(index=False))