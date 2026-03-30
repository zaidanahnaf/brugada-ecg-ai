"""
scripts/download_dataset.py
===========================
Download Brugada-HUCA dataset from PhysioNet — cross-platform.
Equivalent to download_dataset.sh but works on Windows PowerShell.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --skip-if-exists
    python scripts/download_dataset.py --version 1.0.0
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"

DATASET_VERSION   = "1.0.0"
BASE_URL          = f"https://physionet.org/files/brugada-huca/{DATASET_VERSION}"
EXPECTED_SUBJECTS = 363

FILES = [
    "metadata.csv",
    "metadata_dictionary.csv",
    "files.zip",
]


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct  = min(downloaded / total_size * 100, 100)
        mb   = downloaded / 1_048_576
        tot  = total_size / 1_048_576
        bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{tot:.1f} MB", end="", flush=True)
    else:
        mb = downloaded / 1_048_576
        print(f"\r  Downloaded {mb:.1f} MB", end="", flush=True)


def download_file(url: str, dest: Path) -> None:
    print(f"  Downloading {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print()
    except Exception as e:
        print()
        print(f"[ERROR] Failed: {e}")
        print(f"  Try manually:")
        print(f"  wget -N -c {url}")
        sys.exit(1)


def validate_download() -> int:
    hea_files = list(DATA_RAW.rglob("*.hea"))
    return len(hea_files)


def main():
    parser = argparse.ArgumentParser(
        description="Download Brugada-HUCA dataset from PhysioNet"
    )
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Skip if dataset already downloaded")
    parser.add_argument("--version", default=DATASET_VERSION,
                        help="Dataset version (default: 1.0.0)")
    args = parser.parse_args()

    base_url = f"https://physionet.org/files/brugada-huca/{args.version}"

    print("=" * 60)
    print(f" Brugada-HUCA Dataset Downloader v{args.version}")
    print(f" Target: {DATA_RAW}")
    print("=" * 60)

    # Check if already downloaded
    if args.skip_if_exists:
        meta_exists  = (DATA_RAW / "metadata.csv").exists()
        files_exists = (DATA_RAW / "brugada" / "files").exists()
        if meta_exists and files_exists:
            n = validate_download()
            print(f"[OK] Dataset already exists ({n} .hea files). Skipping.")
            return

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Download each file
    for filename in FILES:
        dest = DATA_RAW / filename
        if dest.exists():
            print(f"  Already exists: {filename} — skipping")
            continue
        download_file(f"{base_url}/{filename}", dest)

    # Extract files.zip
    zip_path = DATA_RAW / "files.zip"
    if zip_path.exists():
        print(f"\n[INFO] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            total  = len(zf.namelist())
            for i, member in enumerate(zf.namelist()):
                zf.extract(member, DATA_RAW)
                pct = (i + 1) / total * 100
                print(f"\r  Extracting... {pct:.0f}%", end="", flush=True)
        print()
        zip_path.unlink()
        print("[INFO] Extraction complete. Removed files.zip.")

    # Clean macOS artifacts
    for f in DATA_RAW.rglob("._*"):
        f.unlink()
    for f in DATA_RAW.rglob(".DS_Store"):
        f.unlink()
    macos = DATA_RAW / "__MACOSX"
    if macos.exists():
        import shutil
        shutil.rmtree(macos)

    # Validate
    if not (DATA_RAW / "metadata.csv").exists():
        print("[ERROR] metadata.csv not found. Download may have failed.")
        sys.exit(1)

    n_subjects = validate_download()
    print(f"\n[INFO] Found {n_subjects} .hea files (expected ~{EXPECTED_SUBJECTS})")

    if n_subjects < 300:
        print("[WARNING] Fewer files than expected. Check download integrity.")
    else:
        print("[OK] Download validated.")

    print("=" * 60)
    print(" Download complete. Files stored in data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()