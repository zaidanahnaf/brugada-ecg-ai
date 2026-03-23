"""
run_pipeline.py — Master script to run the full Brugada CNN pipeline.

Runs BOTH variants (3-lead and 12-lead) and produces all deliverables.
After training, runs integration checks automatically.

Usage:
    # Full ablation (both variants):
    python run_pipeline.py

    # Single variant:
    python run_pipeline.py --variant 3lead
    python run_pipeline.py --variant 12lead

    # Quick smoke test (5 epochs, skip if already trained):
    python run_pipeline.py --epochs 5 --smoke_test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows: force UTF-8 so logger never crashes on special characters
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    FEATURES_DIR,
    FOLD_ASSIGNMENTS,
    LEARNING_RATE,
    LOGS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    set_all_seeds,
)
from src.integration_check import run_all_checks
from src.load_data import verify_fold_sha256
from src.train import run_cross_validation

LOGS_DIR.mkdir(parents=True, exist_ok=True)  # ensure exists before FileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create all required output directories."""
    for d in [FEATURES_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Output directories ready.")


def print_environment():
    """Log environment info for reproducibility."""
    import platform
    import numpy as np
    import pandas as pd
    import sklearn

    logger.info("=" * 60)
    logger.info("ENVIRONMENT")
    logger.info(f"  Python:     {platform.python_version()}")
    logger.info(f"  PyTorch:    {torch.__version__}")
    logger.info(f"  CUDA:       {torch.version.cuda}")
    logger.info(f"  NumPy:      {np.__version__}")
    logger.info(f"  Pandas:     {pd.__version__}")
    logger.info(f"  Sklearn:    {sklearn.__version__}")
    logger.info(f"  GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    logger.info("=" * 60)

    # Save to file for reproducibility record
    env_info = {
        "python": platform.python_version(),
        "torch":  torch.__version__,
        "cuda":   torch.version.cuda,
        "numpy":  np.__version__,
        "pandas": pd.__version__,
        "gpu":    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(LOGS_DIR / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Brugada CNN Training Pipeline")
    parser.add_argument(
        "--variant", default="both",
        choices=["3lead", "12lead", "both"],
        help="Which model variant to train (default: both for ablation)"
    )
    parser.add_argument(
        "--model_type", default="cnn",
        choices=["cnn", "resnet"],
        help="CNN architecture (default: cnn)"
    )
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Quick test: 5 epochs, skip integration checks"
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 5
        logger.info("SMOKE TEST MODE: 5 epochs only.")

    # ---- Setup ----
    setup_directories()
    print_environment()
    set_all_seeds()

    # ---- Pre-flight: verify fold file ----
    logger.info("\nPRE-FLIGHT: Verifying fold file SHA256...")
    verify_fold_sha256(FOLD_ASSIGNMENTS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    # ---- Determine which variants to run ----
    variants = ["3lead", "12lead"] if args.variant == "both" else [args.variant]
    all_summaries = {}
    all_ok = True

    for variant in variants:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# TRAINING VARIANT: {variant}")
        logger.info(f"{'#'*60}")
        t0 = time.time()

        try:
            summary = run_cross_validation(
                variant_name=variant,
                model_type=args.model_type,
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                device=device,
            )
            all_summaries[variant] = summary

            elapsed = time.time() - t0
            logger.info(
                f"\n[{variant}] Training complete in {elapsed/60:.1f} min. "
                f"AUROC = {summary.get('auroc_mean', 0):.4f} ± {summary.get('auroc_std', 0):.4f}"
            )

        except Exception as e:
            logger.error(f"[{variant}] Training FAILED: {e}", exc_info=True)
            all_ok = False
            continue

        # ---- Integration checks ----
        if not args.smoke_test:
            logger.info(f"\n[{variant}] Running integration checks...")
            ok = run_all_checks(variant)
            if not ok:
                logger.error(f"[{variant}] Integration checks FAILED — do not deliver!")
                all_ok = False

    # ---- Final comparison (ablation) ----
    if len(all_summaries) == 2:
        print_ablation_comparison(all_summaries)

    # ---- Save standard deliverable names (as per contract) ----
    # Person 4 expects: cnn_fold_probs.csv and cnn_embeddings.csv (no variant suffix)
    # We copy the best variant's files to the contract names
    if all_summaries:
        best_variant = max(all_summaries, key=lambda v: all_summaries[v].get("auroc_mean", 0))
        save_final_deliverables(best_variant)
        logger.info(f"\nBest variant: {best_variant} — used as final deliverable.")

    sys.exit(0 if all_ok else 1)


def print_ablation_comparison(summaries: dict):
    """Print side-by-side comparison of 3-lead vs 12-lead."""
    print("\n" + "=" * 70)
    print("ABLATION COMPARISON: 3-lead (V1-V3) vs 12-lead")
    print("=" * 70)
    metrics = ["auroc", "auprc", "sensitivity", "specificity", "f1_positive"]
    print(f"  {'Metric':<20} {'3-lead':>12} {'12-lead':>12}")
    print(f"  {'-'*44}")
    for m in metrics:
        v3  = summaries.get("3lead",  {}).get(f"{m}_mean", float("nan"))
        v12 = summaries.get("12lead", {}).get(f"{m}_mean", float("nan"))
        winner = "← 3lead" if v3 > v12 else ("← 12lead" if v12 > v3 else "tie")
        print(f"  {m:<20} {v3:>12.4f} {v12:>12.4f}  {winner}")
    print("=" * 70)

    # Clinical note
    auroc_3  = summaries.get("3lead",  {}).get("auroc_mean", 0)
    auroc_12 = summaries.get("12lead", {}).get("auroc_mean", 0)
    if abs(auroc_3 - auroc_12) < 0.02:
        print("\n  -> AUROC difference < 0.02: 3-lead model is sufficient (clinically preferred)")
        print("    Justification: V1-V3 contain all diagnostically relevant ST morphology.")
    elif auroc_12 > auroc_3:
        print(f"\n  -> 12-lead model is better by {auroc_12 - auroc_3:.4f} AUROC.")
        print("    Additional leads may provide complementary information.")
    else:
        print(f"\n  -> 3-lead model is better by {auroc_3 - auroc_12:.4f} AUROC.")
        print("    V1-V3 focus prevents noise from non-diagnostic leads.")


def save_final_deliverables(best_variant: str):
    """
    Copy the best variant's files to the contract-specified names:
        cnn_fold_probs.csv, cnn_embeddings.csv, cnn_cv_summary.json
    """
    import shutil
    file_map = {
        FEATURES_DIR / f"cnn_fold_probs_{best_variant}.csv":   FEATURES_DIR / "cnn_fold_probs.csv",
        FEATURES_DIR / f"cnn_embeddings_{best_variant}.csv":   FEATURES_DIR / "cnn_embeddings.csv",
        RESULTS_DIR  / f"cnn_cv_summary_{best_variant}.json":  RESULTS_DIR  / "cnn_cv_summary.json",
    }
    for src, dst in file_map.items():
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"  {src.name} -> {dst.name}")
    logger.info("Final deliverable files written.")


if __name__ == "__main__":
    main()