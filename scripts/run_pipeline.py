"""
scripts/run_pipeline.py
=======================
Full KINTAKA pipeline runner — cross-platform (Windows/Linux/macOS).
Equivalent to run_pipeline.sh but runnable on PowerShell.

Usage:
    python scripts/run_pipeline.py                     # full run
    python scripts/run_pipeline.py --skip-download     # skip download
    python scripts/run_pipeline.py --skip-features     # skip feature extraction
    python scripts/run_pipeline.py --skip-cnn          # skip CNN training
    python scripts/run_pipeline.py --only-fusion       # only hybrid fusion
    python scripts/run_pipeline.py --help              # show all options
"""

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# ── Project root ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# ── Colors for terminal output ────────────────────────────────
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW= "\033[93m"
RED   = "\033[91m"

def banner(msg):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN} {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

def info(msg):
    print(f"{GREEN}[INFO]{RESET} {msg}")

def warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")

def error(msg):
    print(f"{RED}[ERROR]{RESET} {msg}")

def run(cmd, label=""):
    """Run a command and stream output. Raise on failure."""
    if label:
        info(f"Running: {label}")
    info(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        error(f"Command failed with code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)

def step_skipped(n, total, reason):
    print(f"{YELLOW}[STEP {n}/{total}] Skipped ({reason}){RESET}")

# ── Download helper (Python fallback for wget) ────────────────
def download_dataset():
    """
    Download Brugada-HUCA from PhysioNet using requests.
    Falls back to wget/curl if available.
    """
    import urllib.request
    import zipfile

    DATASET_VERSION = "1.0.0"
    BASE_URL = f"https://physionet.org/files/brugada-huca/{DATASET_VERSION}"
    DATA_RAW = ROOT / "data" / "raw"
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    if (DATA_RAW / "metadata.csv").exists() and \
       (DATA_RAW / "brugada" / "files").exists():
        info("Dataset already exists. Skipping download.")
        return

    files_to_download = [
        "metadata.csv",
        "metadata_dictionary.csv",
        "files.zip",
    ]

    for filename in files_to_download:
        dest = DATA_RAW / filename
        if dest.exists():
            info(f"Already exists: {filename}")
            continue

        url = f"{BASE_URL}/{filename}"
        info(f"Downloading {filename} from {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            info(f"Saved: {dest}")
        except Exception as e:
            error(f"Failed to download {filename}: {e}")
            error("Try manually: wget -r -N -c -np " + BASE_URL)
            sys.exit(1)

    # Extract files.zip
    zip_path = DATA_RAW / "files.zip"
    if zip_path.exists():
        info("Extracting files.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATA_RAW)
        zip_path.unlink()
        info("Extraction complete.")

    # Validate
    hea_files = list(DATA_RAW.rglob("*.hea"))
    info(f"Found {len(hea_files)} .hea files (expected ~363)")
    if len(hea_files) < 300:
        warn("Fewer files than expected. Check download integrity.")

# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="KINTAKA pipeline — Brugada syndrome ECG classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python scripts/run_pipeline.py                   Full pipeline
          python scripts/run_pipeline.py --skip-download   Skip download
          python scripts/run_pipeline.py --skip-cnn        Skip CNN training
          python scripts/run_pipeline.py --only-fusion     Only hybrid fusion
        """)
    )

    parser.add_argument("--skip-download",  action="store_true")
    parser.add_argument("--skip-features",  action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-cnn",       action="store_true")
    parser.add_argument("--skip-fusion",    action="store_true")
    parser.add_argument("--skip-ablation",  action="store_true")
    parser.add_argument("--skip-interp",    action="store_true")
    parser.add_argument("--only-fusion", action="store_true",
                        help="Only run hybrid fusion (assumes all OOF files exist)")

    args = parser.parse_args()

    if args.only_fusion:
        args.skip_download  = True
        args.skip_features  = True
        args.skip_classical = True
        args.skip_cnn       = True
        args.skip_ablation  = True
        args.skip_interp    = True

    TOTAL = 7
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD} KINTAKA — Brugada Syndrome Detection Pipeline{RESET}")
    print(f"{BOLD} Project root: {ROOT}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Create required directories
    for d in ["data/raw", "data/splits", "features",
              "results", "models", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Download ──────────────────────────────────────
    banner(f"STEP 1/{TOTAL}: Dataset download")
    if not args.skip_download:
        download_dataset()
    else:
        step_skipped(1, TOTAL, "--skip-download")

    # ── STEP 2: Label fix + Feature extraction ─────────────────
    banner(f"STEP 2/{TOTAL}: Label correction + Feature extraction")
    if not args.skip_features:
        run([sys.executable, "scripts/fix_brugada_labels.py"],
            "Fix Brugada type 1/2 labels")
        run([sys.executable, "-m", "tests.run_pipeline"],
            "Feature extraction (363 subjects → 681 features)")
    else:
        step_skipped(2, TOTAL, "--skip-features")

    # ── STEP 3: Classical ML ───────────────────────────────────
    banner(f"STEP 3/{TOTAL}: Classical ML training")
    if not args.skip_classical:
        run([sys.executable, "-m", "tests.run_classical_ml"],
            "CatBoost + LogReg + RF (5-fold CV)")
        run([sys.executable, "-m", "scripts.regenerate_catboost_oof"],
            "Regenerate CatBoost OOF with best params")
    else:
        step_skipped(3, TOTAL, "--skip-classical")

    # ── STEP 4: ECGResNet CNN ─────────────────────────────────
    banner(f"STEP 4/{TOTAL}: ECGResNet training")
    if not args.skip_cnn:
        run([sys.executable, "cnn/train.py"],
            "ECGResNet 1D-CNN (V1+V2+V3, 471K params, 5-fold CV)")
    else:
        step_skipped(4, TOTAL, "--skip-cnn")

    # ── STEP 5: Hybrid fusion ─────────────────────────────────
    banner(f"STEP 5/{TOTAL}: Hybrid fusion")
    if not args.skip_fusion:
        run([sys.executable, "scripts/hybrid_late_fusion.py"],
            "Late fusion CNN+LogReg (w=0.65) + 4-model stacking")
    else:
        step_skipped(5, TOTAL, "--skip-fusion")

    # ── STEP 6: Ablation ──────────────────────────────────────
    banner(f"STEP 6/{TOTAL}: Ablation study")
    if not args.skip_ablation:
        run([sys.executable, "-m", "tests.run_ablation"],
            "22 experiments: feature group contributions")
    else:
        step_skipped(6, TOTAL, "--skip-ablation")

    # ── STEP 7: Interpretability ──────────────────────────────
    banner(f"STEP 7/{TOTAL}: SHAP interpretability")
    if not args.skip_interp:
        run([sys.executable, "-m", "tests.run_interpretability"],
            "SHAP values + permutation importance + error analysis")
    else:
        step_skipped(7, TOTAL, "--skip-interp")

    # ── Done ──────────────────────────────────────────────────
    print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN} Pipeline complete.{RESET}")
    print(f"{BOLD}{GREEN} Results : results/{RESET}")
    print(f"{BOLD}{GREEN} Figures : results/interpretability/{RESET}")
    print(f"{BOLD}{GREEN} Models  : models/{RESET}")
    print(f"{BOLD}{GREEN}{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()