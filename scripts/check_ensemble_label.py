import pandas as pd

hc = pd.read_csv("features/oof_predictions_handcrafted.csv")
cnn = pd.read_csv("features/cnn_fold_probs.csv")

merged = hc.merge(cnn, on="patient_id", suffixes=('_hc','_cnn'))

diff = merged[merged['brugada_hc'] != merged['brugada_cnn']]

print("Jumlah mismatch:", len(diff))
print(diff.head())

cnn = cnn.drop(columns=["brugada"])  # buang label lama
cnn = cnn.merge(hc[["patient_id", "brugada"]], on="patient_id", how="left")

cnn["brugada"].isna().sum()

cnn = cnn.dropna(subset=["brugada"])

cnn = cnn.dropna(subset=["oof_prob_brugada"])

df = hc.merge(
    cnn[["patient_id", "oof_prob_brugada"]],
    on="patient_id",
    suffixes=("_hc", "_cnn")
)

(df["brugada"] == df["brugada"]).all()