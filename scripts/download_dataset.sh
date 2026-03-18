#!/bin/bash

set -e

DATASET_VERSION="1.0.0"
BASE_URL="https://physionet.org/files/brugada-huca/${DATASET_VERSION}"

# script file location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# project root directory (one level up from script directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "PROJECT ROOT: $PROJECT_ROOT"

echo "==========================================================="
echo " Downloading Brugada-HUCA v${DATASET_VERSION} Dataset "
echo "==========================================================="

# absolute path
mkdir -p "${PROJECT_ROOT}/src/data/raw"
cd "${PROJECT_ROOT}/src/data/raw"

echo "[INFO] Downloading dataset..."
wget -r -N -c -np -nH --cut-dirs=3 ${BASE_URL}/

if [ ! -f "metadata.csv" ]; then
  echo "[ERROR] metadata.csv not found. Download may have failed."
  exit 1
fi

echo "==========================================================="
echo " Download Complete! Files stored in data/raw/ "
echo "==========================================================="