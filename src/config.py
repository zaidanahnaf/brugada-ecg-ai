from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Data Path
DATA_PATH = BASE_DIR / "data" / "raw"  # raw data
PROCESSED_PATH = DATA_PATH / "processed" # processed data