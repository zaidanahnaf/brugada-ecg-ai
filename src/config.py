from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Data Path
DATA_PATH = BASE_DIR / "data" # Data directory
RAW_PATH = DATA_PATH / "raw" # raw data
PROCESSED_PATH = DATA_PATH / "processed" # processed data
DATASET_PATH = DATA_PATH / "dataset" # final dataset (X.npy, y.npy, groups.npy)