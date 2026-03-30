#!/bin/bash
# =============================================================================
# run_pipeline.sh
# Full KINTAKA pipeline for Brugada syndrome ECG classification
# Works on Linux / macOS / WSL
#
# Usage:
#   bash scripts/run_pipeline.sh                    # full run
#   bash scripts/run_pipeline.sh --skip-download    # skip dataset download
#   bash scripts/run_pipeline.sh --skip-features    # skip feature extraction
#   bash scripts/run_pipeline.sh --skip-cnn         # skip CNN training
#   bash scripts/run_pipeline.sh --only-fusion      # only run fusion
#
# Requirements:
#   pip install -e .
#   pip install -r requirements.txt
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " KINTAKA Pipeline — Brugada Syndrome Detection"
echo " Project root: $PROJECT_ROOT"
echo "============================================================"

# ── Parse flags ──────────────────────────────────────────────
RUN_DOWNLOAD=true
RUN_FEATURES=true
RUN_CLASSICAL=true
RUN_CNN=true
RUN_FUSION=true
RUN_ABLATION=true
RUN_INTERP=true

for arg in "$@"; do
    case $arg in
        --skip-download)  RUN_DOWNLOAD=false ;;
        --skip-features)  RUN_FEATURES=false ;;
        --skip-classical) RUN_CLASSICAL=false ;;
        --skip-cnn)       RUN_CNN=false ;;
        --skip-fusion)    RUN_FUSION=false ;;
        --skip-ablation)  RUN_ABLATION=false ;;
        --skip-interp)    RUN_INTERP=false ;;
        --only-fusion)
            RUN_DOWNLOAD=false
            RUN_FEATURES=false
            RUN_CLASSICAL=false
            RUN_CNN=false
            RUN_ABLATION=false
            RUN_INTERP=false
            ;;
    esac
done

echo "Steps: download=$RUN_DOWNLOAD features=$RUN_FEATURES classical=$RUN_CLASSICAL cnn=$RUN_CNN fusion=$RUN_FUSION ablation=$RUN_ABLATION interp=$RUN_INTERP"
echo ""

# ── Directories ───────────────────────────────────────────────
mkdir -p data/raw data/splits features results models logs

# ── STEP 1: Download dataset ──────────────────────────────────
if [ "$RUN_DOWNLOAD" = true ]; then
    echo "------------------------------------------------------------"
    echo "[STEP 1/7] Downloading Brugada-HUCA dataset..."
    echo "------------------------------------------------------------"
    bash scripts/download_dataset.sh --skip-if-exists
else
    echo "[STEP 1/7] Skipped (--skip-download)"
fi

# ── STEP 2: Fix labels + Feature extraction ───────────────────
if [ "$RUN_FEATURES" = true ]; then
    echo "------------------------------------------------------------"
    echo "[STEP 2/7] Label correction + Feature extraction..."
    echo "------------------------------------------------------------"
    python scripts/fix_brugada_labels.py
    python -m tests.run_pipeline
else
    echo "[STEP 2/7] Skipped (--skip-features)"
fi

# ── STEP 3: Classical ML (CatBoost, LogReg, RF) ──────────────
if [ "$RUN_CLASSICAL" = true ]; then
    echo "------------------------------------------------------------"
    echo "[STEP 3/7] Training classical ML models..."
    echo "------------------------------------------------------------"
    python -m tests.run_classical_ml
    python -m scripts.regenerate_catboost_oof
else
    echo "[STEP 3/7] Skipped (--skip-classical)"
fi

# ── STEP 4: ECGResNet CNN ─────────────────────────────────────
if [ "$RUN_CNN" = true ]; then
    echo "------------------------------------------------------------"
    echo "[STEP 4/7] Training ECGResNet (1D-CNN)..."
    echo "------------------------------------------------------------"
    python cnn/train.py
else
    echo "[STEP 4/7] Skipped (--skip-cnn)"
fi

# ── STEP 5: Hybrid fusion ─────────────────────────────────────
if [ "$RUN_FUSION" = true ]; then
    echo "------------------------------------------------------------"
    echo "[STEP 5/7] Running hybrid fusion (late fusion + stacking)..."
    echo "------------------------------------------------------------"
    python scripts/hybrid_late_fusion.py
else
    echo "[STEP 5/7] Skipped (--skip-fusion)"
fi

# ── STEP 6: Ablation study ────────────────────────────────────
if [ "$RUN_ABLATION" = true ]; then
    echo "------------------------------------------------------------"
    echo "[STEP 6/7] Running ablation study (22 experiments)..."
    echo "------------------------------------------------------------"
    python -m tests.run_ablation
else
    echo "[STEP 6/7] Skipped (--skip-ablation)"
fi

# ── STEP 7: Interpretability ──────────────────────────────────
if [ "$RUN_INTERP" = true ]; then
    echo "------------------------------------------------------------"
    echo "[STEP 7/7] SHAP interpretability + error analysis..."
    echo "------------------------------------------------------------"
    python -m tests.run_interpretability
else
    echo "[STEP 7/7] Skipped (--skip-interp)"
fi

echo ""
echo "============================================================"
echo " Pipeline complete."
echo " Results: results/"
echo " Figures: results/interpretability/"
echo "============================================================"