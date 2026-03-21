#!/bin/bash

# Pipeline workflow:
# 1. Download dataset (optional, can be skipped if already downloaded)
# 2. Load dataset (metadata, files, etc.)
# 3. Preprocess (optional, can be skipped if already preprocessed)
# 4. Build dataset (final dataset ready for modeling)

# Reminder: this pipeline workflow still on progress. So
#           the pipeline can have more steps in the future, such as training, evaluation, etc. The pipeline can be extended as needed.

set -ex

echo " RUNNING PIPELINE "

# project root directory (one level up from script directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "[INFO] Project root: $PROJECT_ROOT"

# default to running all steps
RUN_DOWNLOAD=true
RUN_PREPROCESS=true

# ===== PARSE ARGUMENT =====
for arg in "$@"
do
    case $arg in
        --skip-download)
        RUN_DOWNLOAD=false
        shift
        ;;
        --skip-preprocess)
        RUN_PREPROCESS=false
        shift
        ;;
    esac
done

# Note: The --skip-download and --skip-preprocess flags can be used to skip the download and preprocess steps if they have already been completed. This allows for faster iteration during development.

echo "RUN_DOWNLOAD=$RUN_DOWNLOAD"
echo "RUN_PREPROCESS=$RUN_PREPROCESS"

# data directories
mkdir -p data
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/dataset

# STEP 1: donwload dataset
if [ "$RUN_DOWNLOAD" = true ]; then
    echo "[STEP 1] Download dataset..."
    bash scripts/download_dataset.sh
fi

# STEP 2: load dataset
echo "[STEP 2] Load dataset..."
python -m src.data.load_dataset

# STEP 3: preprocess
if [ "$RUN_PREPROCESS" = true ]; then
    echo "[STEP 3] Preprocessing..."
    python -m src.preprocessing.main
fi

# STEP 4: build dataset
echo "[STEP 4] Building dataset..."
python -m src.data.build_dataset

echo " PIPELINE DONE "