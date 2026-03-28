import pandas as pd, json
import numpy as np
from sklearn.feature_selection import VarianceThreshold

df = pd.read_csv('features/feature_matrix.csv', dtype={'patient_id': str})
top_20 = json.load(open('features/reduced/top_k_20_feature_names.json'))['feature_names']

# Cek kolom non-numerik dulu
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Kolom non-numerik: {non_numeric}")

# Drop kolom non-numerik dan identifier sebelum fit
drop_cols = ['patient_id', 'brugada'] + [c for c in non_numeric if c not in ['patient_id','brugada']]
X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])

print(f"Shape setelah drop non-numerik: {X.shape}")

selector = VarianceThreshold(threshold=0.01)
selector.fit(X)
removed = set(X.columns[~selector.get_support()])

st_removed    = [f for f in removed if f.startswith('st_') or f.startswith('morph_')]
top20_removed = [f for f in top_20 if f in removed]

print(f"\nTotal removed by variance threshold : {len(removed)}")
print(f"ST/morph features removed           : {len(st_removed)}")
print(f"Top-20 features removed             : {top20_removed}")

if st_removed:
    print(f"\nST features yang kena threshold:")
    for f in sorted(st_removed):
        print(f"  {f:50s}  var={X[f].var():.6f}")