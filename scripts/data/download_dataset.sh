#!/bin/bash

set -e  # stop on error

DATASET_VERSION="1.0.0"
BASE_URL="https://physionet.org/files/brugada-huca/${DATASET_VERSION}"

# script file location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# project root directory
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RAW_DIR="${PROJECT_ROOT}/data/raw/brugada"
FILES_DIR="${RAW_DIR}/files"

echo "PROJECT ROOT: $PROJECT_ROOT"

echo "==========================================================="
echo " Downloading Brugada-HUCA v${DATASET_VERSION} Dataset "
echo "==========================================================="

# create directory
mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

echo "[INFO] Downloading metadata..."
wget -N -c "${BASE_URL}/metadata.csv"
wget -N -c "${BASE_URL}/metadata_dictionary.csv"

echo "[INFO] Downloading ECG files..."
wget -N -c "${BASE_URL}/files.zip"

# validate zip exists
if [ ! -f "files.zip" ]; then
  echo "[ERROR] files.zip not found. Download failed."
  exit 1
fi

# extract
echo "[INFO] Extracting files.zip..."
unzip -o files.zip

# validate extraction
if [ ! -d "${FILES_DIR}" ]; then
  echo "[ERROR] Extraction failed. 'files/' directory not found."
  exit 1
fi

# cleanup
rm -f files.zip
rm -rf __MACOSX
find . -name "._*" -type f -delete

# final validation
if [ ! -f "metadata.csv" ]; then
  echo "[ERROR] metadata.csv missing."
  exit 1
fi

# check minimal data integrity
FILE_COUNT=$(find "${FILES_DIR}" -type f | wc -l)
if [ "$FILE_COUNT" -lt 10 ]; then
  echo "[ERROR] Too few ECG files detected. Dataset likely corrupted."
  exit 1
fi

echo "==========================================================="
echo " Download Complete!"
echo " Metadata: ${RAW_DIR}/metadata.csv"
echo " ECG Files: ${FILES_DIR}/"
echo " Total files: $FILE_COUNT"
echo "==========================================================="