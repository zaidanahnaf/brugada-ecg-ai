from pathlib import Path
import wfdb
import pandas as pd
from src.config import DATA_PATH


def load_metadata():
    return pd.read_csv(DATA_PATH / "metadata.csv")


def load_ecg(patient_id):
    record_path = DATA_PATH / "files" / str(patient_id) / str(patient_id)

    print(f"[DEBUG] Loading: {record_path}")

    if not (record_path.with_suffix(".hea")).exists():
        raise FileNotFoundError(f"File not found: {record_path}.hea")

    record = wfdb.rdrecord(str(record_path))

    return record.p_signal, record.fs, record.sig_name


def debug_sample(n=5):
    metadata = load_metadata()

    print("Total samples:", len(metadata))
    print("Brugada cases:", ((metadata["brugada"] == 1) | (metadata["brugada"] == 2)).sum())
    print("Normal cases:", (metadata["brugada"] == 0).sum())

    for i in range(n):
        sample_id = metadata.iloc[i]["patient_id"]

        print(f"\n[INFO] Loading patient: {sample_id}")

        signals, fs, leads = load_ecg(sample_id)

        print("Signal shape:", signals.shape)
        print("Sampling rate:", fs)
        print("Leads:", leads)

if __name__ == "__main__":
    debug_sample(5)