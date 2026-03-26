"""
Central configuration for Brugada CNN branch.
All paths, hyperparameters, and constants defined here.
Edit this file only — do not hardcode values in other modules.
"""

import os

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42

# =============================================================================
# PATHS  (relative to project root, i.e. where you run the scripts from)
# =============================================================================
DATA_DIR          = "data/raw/brugada/files"          # WFDB .dat + .hea files
FOLD_CSV          = "data/splits/fold_assignments.csv"
FEATURE_MATRIX    = "features/feature_matrix.csv"

OUT_EMBEDDINGS    = "features/cnn_embeddings.csv"
OUT_FOLD_PROBS    = "features/cnn_fold_probs.csv"
OUT_CV_SUMMARY    = "results/cnn_cv_summary.json"
MODEL_DIR         = "models"
PREPROC_STATS_DIR = "preprocessing/fold_stats"

# =============================================================================
# FOLD FILE INTEGRITY
# =============================================================================
FOLD_CSV_SHA256 = "c8988aaa58ff7729ca36a7b54e7b6782f1ae2cd4e950dcf543fe120f11761904"
N_FOLDS         = 5

# =============================================================================
# KNOWN FAILED SUBJECTS
# =============================================================================
FAILED_SUBJECTS = {
    "267630":  "TOO_FEW_RPEAKS:0   — excluded from training; NaN in outputs",
    "1230482": "TOO_FEW_VALID_BEATS:0 — excluded from training; NaN in outputs",
}

# =============================================================================
# SIGNAL PROPERTIES
# =============================================================================
FS              = 100        # Hz
N_SAMPLES       = 1200       # 12 s × 100 Hz
N_LEADS_FULL    = 12
LEAD_NAMES      = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# Clinically critical leads for Brugada detection
V_LEADS         = ["V1","V2","V3"]
V_LEAD_INDICES  = [6, 7, 8]   # indices in the 12-lead array — verified against WFDB standard

# =============================================================================
# PREPROCESSING
# =============================================================================
MEDIAN_FILTER_SAMPLES = 61       # ~610 ms at 100 Hz (target: 600 ms)
LP_CUTOFF_HZ          = 40.0
LP_ORDER              = 4
APPLY_LP_FILTER       = True     # ablatable flag

# =============================================================================
# MODEL
# =============================================================================
# Primary model: 3-lead (V1/V2/V3)
USE_3_LEAD       = True          # True = Model A (primary); False = Model B (12-lead)
IN_CHANNELS_A    = 3
IN_CHANNELS_B    = 12
EMBED_DIM        = 128

# =============================================================================
# CLASS IMBALANCE
# =============================================================================
N_BRUGADA        = 76
N_NORMAL         = 287
POS_WEIGHT       = N_NORMAL / N_BRUGADA   # ≈ 3.776

# =============================================================================
# TRAINING HYPERPARAMETERS (default / best starting point)
# =============================================================================
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 5e-4
BATCH_SIZE       = 8
MAX_EPOCHS       = 100
PATIENCE         = 15            # early stopping on val AUROC
DROPOUT          = 0.3

# Augmentation (training only)
AUG_NOISE_STD    = 0.005         # Gaussian noise std (relative to unit-normalized signal)
AUG_AMP_MIN      = 0.95
AUG_AMP_MAX      = 1.05
USE_AUGMENTATION = True

# =============================================================================
# HYPERPARAMETER SEARCH GRID  (used by tune_hyperparams.py)
# =============================================================================
HP_GRID = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "batch_size":    [8, 16, 32],
    "dropout":       [0.3, 0.5],
    "weight_decay":  [1e-4, 1e-3],
}

# =============================================================================
# EVALUATION
# =============================================================================
TARGET_SPECIFICITY_FOR_SENSITIVITY = 0.90   # "sensitivity at 90% specificity"