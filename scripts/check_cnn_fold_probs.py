import pandas as pd

df = pd.read_csv("features/cnn_fold_probs.csv")
print(df.describe())