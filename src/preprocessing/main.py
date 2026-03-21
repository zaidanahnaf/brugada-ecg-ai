import os
import pandas as pd
import numpy as np

from src.preprocessing.pipeline import preprocess_pipeline
from src.data.load_dataset import load_ecg
from src.config import RAW_PATH, PROCESSED_PATH

# main function to run the preprocessing pipeline
if __name__ == "__main__":
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    metadata = pd.read_csv(os.path.join(RAW_PATH, "metadata.csv"))
    metadata['brugada'] = metadata['brugada'].apply(lambda x: 1 if x > 0 else 0)

    # Iterate through each patient in the metadata
    for idx, row in metadata.iterrows():

        pid = row['patient_id']
        label = row['brugada']

        print(f"[INFO] Processing patient: {pid}")

        # LOAD
        signals, fs, leads = load_ecg(pid)
        print(f"Loaded: {signals.shape}")
        print(f"Sampling Frequency: {fs} Hz", f"Leads: {len(leads)}")

        # PREPROCESS
        segments = preprocess_pipeline(signals, fs)
        print(f"Segments: {len(segments)}")
        print(f"Segment shape: {segments[0].shape}") # DEBUG SEGMENT

        # SAVE
        patient_dir = os.path.join(PROCESSED_PATH, str(pid))
        os.makedirs(patient_dir, exist_ok=True)

        for i, seg in enumerate(segments):
            file_name = f"{pid}_{label}_seg{i}.npy"
            print(f"Saving: {file_name}")

            np.save(os.path.join(patient_dir, file_name), seg)

        