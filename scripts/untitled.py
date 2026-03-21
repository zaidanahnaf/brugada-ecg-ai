import os
import requests
import zipfile
from pathlib import Path

DATASET_VERSION = "1.0.0"
BASE_URL = f"https://physionet.org/files/brugada-huca/{DATASET_VERSION}"

# =========================
# GET PROJECT ROOT (ANTI ERROR)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # naik dari scripts/ ke root
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] PROJECT ROOT: {PROJECT_ROOT}")
print(f"[INFO] DATA DIR: {DATA_DIR}")

# =========================
# FUNCTIONS
# =========================
def download_file(url, output_path):
    if output_path.exists():
        print(f"[SKIP] Already exists: {output_path.name}")
        return

    print(f"[DOWNLOAD] {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"[OK] Saved: {output_path.name}")


def extract_zip(zip_path, extract_to):
    print("[INFO] Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("[OK] Extraction complete")


def clean_macos_files(folder):
    print("[INFO] Cleaning macOS files...")

    macosx = folder / "__MACOSX"
    if macosx.exists():
        for root, dirs, files in os.walk(macosx, topdown=False):
            for name in files:
                os.remove(Path(root) / name)
            for name in dirs:
                os.rmdir(Path(root) / name)
        os.rmdir(macosx)

    for root, dirs, files in os.walk(folder):
        for name in files:
            if name.startswith("._"):
                os.remove(Path(root) / name)

    print("[OK] Cleaned")


# =========================
# MAIN
# =========================
def main():
    print("===========================================================")
    print(f" Downloading Brugada-HUCA v{DATASET_VERSION} Dataset ")
    print("===========================================================")

    os.chdir(DATA_DIR)

    # Download files
    download_file(f"{BASE_URL}/metadata.csv", DATA_DIR / "metadata.csv")
    download_file(f"{BASE_URL}/metadata_dictionary.csv", DATA_DIR / "metadata_dictionary.csv")
    download_file(f"{BASE_URL}/files.zip", DATA_DIR / "files.zip")

    # Extract
    zip_path = DATA_DIR / "files.zip"
    if zip_path.exists():
        extract_zip(zip_path, DATA_DIR)
        os.remove(zip_path)

    # Clean
    clean_macos_files(DATA_DIR)

    # Validation
    if not (DATA_DIR / "metadata.csv").exists():
        raise FileNotFoundError("[ERROR] metadata.csv not found. Download failed.")

    print("===========================================================")
    print(" Download Complete! Files stored in data/raw/ ")
    print("===========================================================")


if __name__ == "__main__":
    main()