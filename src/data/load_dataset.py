import os
import wfdb
import pandas as pd
from src.config import FULL_DATA_PATH  

DATA_PATH = FULL_DATA_PATH


def load_metadata():
    metadata_path = os.path.join(DATA_PATH, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    return metadata


def load_ecg(patient_id):
    record_path = os.path.join(
        DATA_PATH,
        "files",           
        str(patient_id),
        str(patient_id)
    )

    print(f"[DEBUG] Loading: {record_path}")

    if not os.path.exists(record_path + ".hea"):
        raise FileNotFoundError(f"File not found: {record_path}.hea")

    record = wfdb.rdrecord(record_path)

    signals = record.p_signal
    fs = record.fs
    lead_names = record.sig_name

    return signals, fs, lead_names


def debug_sample(n=5):
    metadata = load_metadata()

    print("Total samples:", len(metadata))
    print("Brugada cases:", ((metadata["brugada"] == 1) | (metadata["brugada"] == 2)).sum())
    print("Normal cases:", (metadata["brugada"] == 0).sum())

    for i in range(n):
        sample_id = metadata.iloc[i]["patient_id"]

        print(f"\n[INFO] Loading patient: {sample_id}")
    signals, fs, leads = load_ecg(sample_id)

    print("Sample ID:", sample_id)
    print("Signal shape:", signals.shape)
    print("Sampling rate:", fs)
    print("Leads:", leads)


if __name__ == "__main__":
    debug_sample(5)