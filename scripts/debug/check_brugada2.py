# scripts/check_brugada2.py

import pandas as pd

metadata = pd.read_csv("data/raw/brugada/metadata.csv")

print("=== BRUGADA VALUE=2 SUBJECTS ===\n")
b2 = metadata[metadata['brugada'] == 2]
print(b2.to_string(index=False))

print(f"\nTotal subjects    : {len(metadata)}")
print(f"brugada=0 (Normal): {(metadata['brugada']==0).sum()}")
print(f"brugada=1 (Type 1): {(metadata['brugada']==1).sum()}")
print(f"brugada=2 (Type 2): {(metadata['brugada']==2).sum()}")

# Cek kolom lain yang ada di metadata
print(f"\nMetadata columns: {list(metadata.columns)}")
print(f"\nSample rows:\n{metadata.head(10).to_string(index=False)}")