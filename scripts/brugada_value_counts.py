import pandas as pd

df = pd.read_csv("data/raw/brugada/metadata.csv")

print(df['brugada'].value_counts())