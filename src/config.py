"""
config.py — Central configuration for the Brugada CNN pipeline.
All paths, hyperparameters, and constants live here.
Person 2: DO NOT hardcode values elsewhere — always import from here.
"""

import os
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Z:\My Projects\brugada-ecg-ai\

DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_FILES_DIR     = DATA_RAW_DIR / "files"
METADATA_CSV       = DATA_RAW_DIR / "metadata.csv"
SPLITS_DIR         = PROJECT_ROOT / "data" / "splits"
FOLD_ASSIGNMENTS   = SPLITS_DIR / "fold_assignments.csv"

FEATURES_DIR       = PROJECT_ROOT / "features"
RESULTS_DIR        = PROJECT_ROOT / "results"
MODELS_DIR         = PROJECT_ROOT / "models"
LOGS_DIR           = PROJECT_ROOT / "logs"

# Required deliverable files
CNN_EMBEDDINGS_CSV = FEATURES_DIR / "cnn_embeddings.csv"
CNN_FOLD_PROBS_CSV = FEATURES_DIR / "cnn_fold_probs.csv"
CNN_CV_SUMMARY_JSON = RESULTS_DIR / "cnn_cv_summary.json"

# ============================================================
# FOLD INTEGRITY
# ============================================================
FOLD_SHA256 = "c8988aaa58ff7729ca36a7b54e7b6782f1ae2cd4e950dcf543fe120f11761904"
N_FOLDS = 5

# ============================================================
# SIGNAL PARAMETERS
# ============================================================
SAMPLING_RATE = 100          # Hz
N_LEADS = 12
N_SAMPLES = 1200             # 12 seconds × 100 Hz
DURATION_SEC = 12.0

# Lead index mapping (0-indexed, standard WFDB order)
# I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
V1_IDX = 6
V2_IDX = 7
V3_IDX = 8
PRECORDIAL_LEADS = [V1_IDX, V2_IDX, V3_IDX]   # indices for 3-lead model

# ============================================================
# KNOWN PROBLEMATIC SUBJECTS
# Decision: exclude from training but keep rows in output (NaN embeddings)
# Both are Brugada-positive — documented here for traceability.
# ============================================================
PROBLEMATIC_SUBJECTS = ["267630", "1230482"]

# ============================================================
# PREPROCESSING
# ============================================================
BASELINE_KERNEL_MS   = 600          # median filter kernel size in ms
BASELINE_KERNEL_SAMPLES = int(BASELINE_KERNEL_MS / 1000 * SAMPLING_RATE)  # =60
# Make kernel odd
if BASELINE_KERNEL_SAMPLES % 2 == 0:
    BASELINE_KERNEL_SAMPLES += 1    # -> 61

LOWPASS_CUTOFF_HZ    = 40.0         # Butterworth low-pass cutoff
LOWPASS_ORDER        = 4
APPLY_LOWPASS        = True         # set False to skip

# ============================================================
# BEAT SEGMENTATION  (Work Package B update)
# ============================================================
BEAT_PRE_MS          = 200          # ms before R-peak
BEAT_POST_MS         = 500          # ms after R-peak
BEAT_PRE_SAMPLES     = int(BEAT_PRE_MS  / 1000 * SAMPLING_RATE)   # 20
BEAT_POST_SAMPLES    = int(BEAT_POST_MS / 1000 * SAMPLING_RATE)   # 50
BEAT_WINDOW_SAMPLES  = BEAT_PRE_SAMPLES + BEAT_POST_SAMPLES        # 70
MIN_BEATS_PER_RECORD = 3            # discard recording if fewer valid beats

# Beat aggregation strategy: "mean" | "max" | "attention"
BEAT_AGGREGATION     = "mean"       # recommended default

# ============================================================
# MODEL
# ============================================================
# Two model variants — ablation
MODEL_VARIANTS = {
    "3lead": {"n_leads": 3,  "lead_indices": PRECORDIAL_LEADS},
    "12lead": {"n_leads": 12, "lead_indices": list(range(12))},
}
EMBEDDING_DIM = 64
DROPOUT       = 0.5

# ============================================================
# TRAINING
# ============================================================
SEED            = 42
EPOCHS          = 150
PATIENCE        = 20                 # early stopping patience
LEARNING_RATE   = 1e-4               # default; tune in {1e-4, 5e-4, 1e-3}
BATCH_SIZE      = 16                 # default; tune in {8, 16, 32}
WEIGHT_DECAY    = 1e-4

# Class imbalance: pos_weight = n_negative / n_positive
N_POSITIVE      = 76
N_NEGATIVE      = 287
POS_WEIGHT      = N_NEGATIVE / N_POSITIVE   # ≈ 3.776

# ============================================================
# AUGMENTATION (allowed)
# ============================================================
AUG_AMPLITUDE_RANGE = (0.9, 1.1)
AUG_BASELINE_SHIFT_STD = 0.02   # std of Gaussian noise added to baseline

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_all_seeds(seed: int = SEED):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False