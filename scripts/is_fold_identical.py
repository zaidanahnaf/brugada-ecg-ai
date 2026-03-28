import pandas as pd
from src.fold_manager import create_folds

# Load metadata
metadata = pd.read_csv("data/raw/brugada/metadata.csv")

# Generate 2x
df1 = create_folds(metadata, output_dir="data/splits/test1")
df2 = create_folds(metadata, output_dir="data/splits/test2")

# Compare
same = (df1['fold_id'].values == df2['fold_id'].values).all()

print("Are fold assignments identical?", same)