"""
scripts/tune_hyperparams.py

Lightweight hyperparameter search using FOLD 0 ONLY as the proxy validation.

Strategy:
  - Search over HP_GRID from config.py
  - Train one model per config on fold 0 only
  - Rank by validation AUROC on fold 0
  - Best config is then used for the full 5-fold run in train_cnn.py

This avoids leaking hyperparameter choices into all folds.
Note: Fold 0 is just used for HP selection, not for model selection.
The full 5-fold run is always the source of final metrics.

Usage:
    python scripts/tune_hyperparams.py
    python scripts/tune_hyperparams.py --leads 3
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch

from src.cnn_config import (
    AUG_AMP_MAX, AUG_AMP_MIN, AUG_NOISE_STD, DATA_DIR,
    EMBED_DIM, FAILED_SUBJECTS, FOLD_CSV, FOLD_CSV_SHA256,
    HP_GRID, IN_CHANNELS_A, IN_CHANNELS_B, LP_CUTOFF_HZ, LP_ORDER,
    MAX_EPOCHS, MEDIAN_FILTER_SAMPLES, N_SAMPLES,
    APPLY_LP_FILTER, PATIENCE, POS_WEIGHT, SEED,
    TARGET_SPECIFICITY_FOR_SENSITIVITY, V_LEAD_INDICES,
)
from src.data.cnn_data_loader import load_dataset, load_fold_assignments
from src.cnn_dataset import make_dataloader
from src.cnn_metrics import compute_metrics
from src.cnn_model import build_model
from src.cnn_preprocessing import preprocess_fold
from src.cnn_trainer import evaluate, train_fold
from src.cnn_utils import (
    ensure_dirs, get_logger, save_json, set_seed, verify_fold_sha256,
)

logger = get_logger("tune_hyperparams")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leads",     type=int, default=3, choices=[3, 12])
    parser.add_argument("--tune-fold", type=int, default=0,
                        help="Which fold to use for HP tuning (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device      = torch.device("cpu")
    use_3_lead  = (args.leads == 3)
    in_channels = IN_CHANNELS_A if use_3_lead else IN_CHANNELS_B
    tune_fold   = args.tune_fold

    ensure_dirs("results")
    verify_fold_sha256(FOLD_CSV, FOLD_CSV_SHA256, logger)

    fold_df   = load_fold_assignments(FOLD_CSV)
    label_map = dict(zip(fold_df["patient_id"].astype(str), fold_df["brugada"].astype(int)))

    # Load data for the tuning fold once (reuse across all HP configs)
    train_ids = fold_df.loc[fold_df["fold_id"] != tune_fold, "patient_id"].astype(str).tolist()
    val_ids   = fold_df.loc[fold_df["fold_id"] == tune_fold, "patient_id"].astype(str).tolist()

    train_raw, train_labels, train_loaded = load_dataset(
        DATA_DIR, train_ids, label_map, FAILED_SUBJECTS,
        N_SAMPLES, V_LEAD_INDICES, use_3_lead,
    )
    val_raw, val_labels, val_loaded = load_dataset(
        DATA_DIR, val_ids, label_map, FAILED_SUBJECTS,
        N_SAMPLES, V_LEAD_INDICES, use_3_lead,
    )

    train_proc, val_proc, _ = preprocess_fold(
        train_raw, val_raw,
        kernel_samples=MEDIAN_FILTER_SAMPLES,
        apply_lp=APPLY_LP_FILTER,
        lp_cutoff=LP_CUTOFF_HZ,
        lp_order=LP_ORDER,
        fs=100.0,
    )

    # Build all HP combinations
    keys   = list(HP_GRID.keys())
    combos = list(itertools.product(*[HP_GRID[k] for k in keys]))
    logger.info(f"HP tuning: {len(combos)} configurations | fold={tune_fold}")

    results = []

    for i, combo in enumerate(combos):
        hp = dict(zip(keys, combo))
        logger.info(f"\n[{i+1}/{len(combos)}] Config: {hp}")
        set_seed(SEED + i)

        train_loader = make_dataloader(
            train_proc, train_labels, train_loaded,
            batch_size=hp["batch_size"], shuffle=True, augment=True,
            seed=SEED+i, noise_std=AUG_NOISE_STD,
            amp_min=AUG_AMP_MIN, amp_max=AUG_AMP_MAX,
        )
        val_loader = make_dataloader(
            val_proc, val_labels, val_loaded,
            batch_size=hp["batch_size"], shuffle=False, augment=False,
        )

        model = build_model(
            in_channels=in_channels,
            embed_dim=EMBED_DIM,
            dropout=hp["dropout"],
        )

        model_path = f"models/tune_fold{tune_fold}_config{i}.pt"
        model, history = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            fold_id=tune_fold,
            model_save_path=model_path,
            learning_rate=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
            max_epochs=min(MAX_EPOCHS, 60),   # shorter for HP search
            patience=PATIENCE,
            pos_weight=POS_WEIGHT,
            device=device,
        )

        # Evaluate
        import torch.nn as nn
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([POS_WEIGHT], dtype=torch.float32)
        )
        _, auroc, probs, labels = evaluate(model, val_loader, criterion, device)
        metrics = compute_metrics(labels, probs, TARGET_SPECIFICITY_FOR_SENSITIVITY)

        row = {**hp, "auroc": metrics["auroc"], "auprc": metrics["auprc"],
               "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
               "best_epoch": history["best_epoch"]}
        results.append(row)
        logger.info(f"  AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f} "
                    f"| best_epoch={history['best_epoch']}")

        # Clean up tune model
        if os.path.exists(model_path):
            os.remove(model_path)

    # Sort by AUROC
    results_df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    logger.info("\n" + "="*60)
    logger.info("HP TUNING RESULTS (sorted by AUROC):")
    logger.info(results_df.to_string(index=False))

    results_df.to_csv("results/hp_tuning_results.csv", index=False)
    save_json({"best_config": results[0], "all_results": results},
              "results/hp_tuning_summary.json")

    best = results_df.iloc[0]
    logger.info(f"\nBEST CONFIG: lr={best['learning_rate']} "
                f"bs={best['batch_size']} dropout={best['dropout']} "
                f"wd={best['weight_decay']}")
    logger.info("Run train_cnn.py with these hyperparameters for final 5-fold training.")


if __name__ == "__main__":
    main()