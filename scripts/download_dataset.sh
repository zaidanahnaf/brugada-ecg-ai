
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
mkdir -p "${PROJECT_ROOT}/data/raw"
cd "${PROJECT_ROOT}/data/raw"

echo "[INFO] Downloading dataset..."
# metadata
wget -N -c ${BASE_URL}/metadata.csv
wget -N -c ${BASE_URL}/metadata_dictionary.csv

# files.zip
wget -N -c ${BASE_URL}/files.zip

# extract
echo "[INFO] Extracting files.zip..."
unzip -o files.zip
rm files.zip # Delete zip file 

# delete macOS folder
rm -rf __MACOSX
find . -name "._*" -type f -delete # delete macOS hidden files

if [ ! -f "metadata.csv" ]; then
  echo "[ERROR] metadata.csv not found. Download may have failed."
  exit 1
fi

echo "==========================================================="
echo " Download Complete! Files stored in data/raw/ "
echo "==========================================================="