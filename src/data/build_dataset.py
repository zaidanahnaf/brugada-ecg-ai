import os
import numpy as np
from src.config import PROCESSED_PATH, DATASET_PATH

X = []
y = []
groups = []  # Input

print("Building dataset...")

for pid in sorted(os.listdir(PROCESSED_PATH)):
    patient_dir = os.path.join(PROCESSED_PATH, pid)

    if not os.path.isdir(patient_dir):
        continue

    for file in sorted(os.listdir(patient_dir)):
        if not file.endswith(".npy"):
            continue

        path = os.path.join(patient_dir, file)

        try:
            data = np.load(path)

            # shape validation
            if data.shape != (300, 12):
                print(f"[SKIP] Invalid shape: {file} → {data.shape}")
                continue

            X.append(data.astype(np.float32))

            # LABEL
            label = int(file.split("_")[1])
            y.append(label)

            # GROUP (patient_id)
            groups.append(int(pid))

        except Exception as e:
            print(f"[ERROR] {file}: {e}")

# convert to numpy arrays
X = np.array(X)
y = np.array(y)
groups = np.array(groups)

# dataset information
print("\n=== FINAL DATASET ===")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)

assert len(X) == len(y) == len(groups), "Mismatch length!"
assert X.shape[1:] == (300, 12), "Shape error!"

# SAVE
os.makedirs("data", exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True) # make folder for dataset if not exists

np.save(os.path.join(DATASET_PATH, "X.npy"), X)
np.save(os.path.join(DATASET_PATH, "y.npy"), y)
np.save(os.path.join(DATASET_PATH, "groups.npy"), groups)

print("\nDataset saved successfully!")