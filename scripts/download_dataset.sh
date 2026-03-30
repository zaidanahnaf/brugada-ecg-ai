#!/bin/bash
# =============================================================================
# download_dataset.sh
# Download Brugada-HUCA dataset from PhysioNet
# Usage: bash scripts/download_dataset.sh [--skip-if-exists]
# =============================================================================

set -e

DATASET_VERSION="1.0.0"
BASE_URL="https://physionet.org/files/brugada-huca/${DATASET_VERSION}"
EXPECTED_SUBJECTS=363

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_RAW="${PROJECT_ROOT}/data/raw"

echo "============================================================"
echo " Brugada-HUCA Dataset Downloader v${DATASET_VERSION}"
echo " Target: ${DATA_RAW}"
echo "============================================================"

# Skip if already downloaded
if [ "$1" = "--skip-if-exists" ]; then
    if [ -f "${DATA_RAW}/metadata.csv" ] && [ -d "${DATA_RAW}/brugada/files" ]; then
        echo "[OK] Dataset already exists. Skipping download."
        exit 0
    fi
fi

mkdir -p "${DATA_RAW}"
cd "${DATA_RAW}"

# Download metadata
echo "[1/3] Downloading metadata..."
wget -q --show-progress -N -c "${BASE_URL}/metadata.csv" || \
    curl -# -C - -O "${BASE_URL}/metadata.csv"
wget -q --show-progress -N -c "${BASE_URL}/metadata_dictionary.csv" || \
    curl -# -C - -O "${BASE_URL}/metadata_dictionary.csv"

# Download files.zip
echo "[2/3] Downloading ECG files (~2GB, may take a while)..."
wget -q --show-progress -N -c "${BASE_URL}/files.zip" || \
    curl -# -C - -O "${BASE_URL}/files.zip"

# Extract
echo "[3/3] Extracting..."
unzip -q -o files.zip
rm -f files.zip

# Clean macOS artifacts
rm -rf __MACOSX
find . -name "._*" -type f -delete
find . -name ".DS_Store" -type f -delete

# Validate
if [ ! -f "${DATA_RAW}/metadata.csv" ]; then
    echo "[ERROR] metadata.csv not found. Download may have failed."
    exit 1
fi

N_SUBJECTS=$(find "${DATA_RAW}" -name "*.hea" | wc -l)
echo "[INFO] Found ${N_SUBJECTS} .hea files (expected ~${EXPECTED_SUBJECTS})"

if [ "${N_SUBJECTS}" -lt 300 ]; then
    echo "[WARNING] Fewer files than expected. Check download integrity."
fi

echo "============================================================"
echo " Download complete. Files stored in data/raw/"
echo "============================================================"