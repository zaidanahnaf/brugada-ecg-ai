"""
scripts/train_cnn.py

Main training script for the Brugada CNN branch.

Usage:
    python scripts/train_cnn.py              # default: 3-lead Model A
    python scripts/train_cnn.py --leads 12   # 12-lead Model B (ablation)
    python scripts/train_cnn.py --no-aug     # disable augmentation

What this script does:
    1. Verify fold_assignments.csv SHA256
    2. Load fold assignments
    3. For each fold 0-4:
       a. Load raw signals for train/val subjects
       b. Preprocess (fit on train, apply to val)
       c. Train model with early stopping
       d. Evaluate on val fold
       e. Extract OOF embeddings and probabilities
    4. Assemble and save all deliverables:
       - features/cnn_embeddings.csv
       - features/cnn_fold_probs.csv
       - results/cnn_cv_summary.json
       - models/cnn_fold_{0-4}.pt

CRITICAL: fold_assignments.csv SHA256 is verified before any training begins.
CRITICAL: Normalization statistics are fit on training fold only.
CRITICAL: patient_id is always treated and saved as STRING.
"""

import argparse
import json
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch

from src.cnn_config import (
    APPLY_LP_FILTER, AUG_AMP_MAX, AUG_AMP_MIN, AUG_NOISE_STD,
    BATCH_SIZE, DATA_DIR, DROPOUT, EMBED_DIM, FAILED_SUBJECTS,
    FOLD_CSV, FOLD_CSV_SHA256, LP_CUTOFF_HZ, LP_ORDER,
    LEARNING_RATE, MAX_EPOCHS, MEDIAN_FILTER_SAMPLES, MODEL_DIR,
    N_FOLDS, N_SAMPLES, OUT_CV_SUMMARY, OUT_EMBEDDINGS, OUT_FOLD_PROBS,
    PATIENCE, POS_WEIGHT, PREPROC_STATS_DIR, SEED,
    TARGET_SPECIFICITY_FOR_SENSITIVITY, USE_AUGMENTATION,
    V_LEAD_INDICES, WEIGHT_DECAY,
    IN_CHANNELS_A, IN_CHANNELS_B,
)
from src.data.cnn_data_loader import load_dataset, load_fold_assignments, verify_lead_indices
from src.cnn_dataset import make_dataloader
from src.cnn_metrics import (
    aggregate_fold_metrics, compute_metrics, print_fold_metrics, print_summary_metrics,
)
# from src.cnn_model import build_model
from src.cnn_resnet import ECGResNet
from src.cnn_preprocessing import preprocess_fold
from src.cnn_trainer import extract_embeddings, train_fold
from src.cnn_utils import (
    ensure_dirs, get_logger, save_json, set_seed, verify_fold_sha256,
)

logger = get_logger("train_cnn")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Brugada CNN Training")
    parser.add_argument("--leads",    type=int,  default=3,
                        choices=[3, 12],
                        help="Number of leads to use: 3 (V1/V2/V3) or 12 (full). Default: 3")
    parser.add_argument("--no-aug",   action="store_true",
                        help="Disable training augmentation")
    parser.add_argument("--lr",       type=float, default=LEARNING_RATE)
    parser.add_argument("--wd",       type=float, default=WEIGHT_DECAY)
    parser.add_argument("--bs",       type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",   type=int,   default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int,   default=PATIENCE)
    parser.add_argument("--dropout",  type=float, default=DROPOUT)
    parser.add_argument("--seed",     type=int,   default=SEED)
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # 0. Setup
    # -------------------------------------------------------------------------
    set_seed(args.seed)
    device     = torch.device("cpu")
    use_3_lead = (args.leads == 3)
    in_channels = IN_CHANNELS_A if use_3_lead else IN_CHANNELS_B
    use_aug     = USE_AUGMENTATION and not args.no_aug

    ensure_dirs(MODEL_DIR, "features", "results", PREPROC_STATS_DIR)

    logger.info("=" * 60)
    logger.info(f"Brugada CNN Training | leads={args.leads} | aug={use_aug}")
    logger.info(f"Seed={args.seed} | Device={device}")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Verify fold file SHA256  (HARD STOP on failure)
    # -------------------------------------------------------------------------
    verify_fold_sha256(FOLD_CSV, FOLD_CSV_SHA256, logger)

    # -------------------------------------------------------------------------
    # 2. Load fold assignments
    # -------------------------------------------------------------------------
    fold_df = load_fold_assignments(FOLD_CSV)
    label_map = dict(zip(fold_df["patient_id"].astype(str), fold_df["brugada"].astype(int)))
    all_patient_ids = fold_df["patient_id"].astype(str).tolist()

    # -------------------------------------------------------------------------
    # 3. Verify V1/V2/V3 lead indices on a sample record
    # -------------------------------------------------------------------------
    sample_id = [pid for pid in all_patient_ids
                 if pid not in FAILED_SUBJECTS][0]
    try:
        verify_lead_indices(DATA_DIR, sample_id)
    except Exception as e:
        logger.warning(f"Lead verification failed for {sample_id}: {e}. "
                       "Proceeding with default indices [6,7,8].")

    # -------------------------------------------------------------------------
    # 4. OOF accumulators
    # -------------------------------------------------------------------------
    oof_probs_rows      = []   # list of dicts → cnn_fold_probs.csv
    oof_embeddings_dict = {}   # patient_id → np.array(embed_dim,)
    fold_metrics_list   = []

    # Pre-fill NaN entries for failed subjects
    for pid, reason in FAILED_SUBJECTS.items():
        logger.warning(f"FAILED SUBJECT {pid}: {reason}")
        oof_probs_rows.append({
            "patient_id":     str(pid),
            "brugada":        label_map.get(str(pid), np.nan),
            "oof_prob_brugada": np.nan,
            "fold_id":        fold_df.loc[fold_df["patient_id"] == str(pid), "fold_id"]
                              .values[0] if str(pid) in fold_df["patient_id"].values else np.nan,
        })
        oof_embeddings_dict[str(pid)] = np.full(EMBED_DIM, np.nan, dtype=np.float32)

    # -------------------------------------------------------------------------
    # 5. 5-FOLD TRAINING LOOP
    # -------------------------------------------------------------------------
    for fold_id in range(N_FOLDS):
        logger.info(f"\n{'#'*60}\n  FOLD {fold_id}\n{'#'*60}")
        set_seed(args.seed + fold_id)   # different seed per fold, still reproducible

        # --- Split patient IDs ---
        train_ids = fold_df.loc[fold_df["fold_id"] != fold_id, "patient_id"].astype(str).tolist()
        val_ids   = fold_df.loc[fold_df["fold_id"] == fold_id, "patient_id"].astype(str).tolist()

        logger.info(f"Train: {len(train_ids)} subjects | Val: {len(val_ids)} subjects")

        # --- Load raw signals ---
        logger.info("Loading training signals...")
        train_signals, train_labels, train_loaded_ids = load_dataset(
            data_dir=DATA_DIR,
            patient_ids=train_ids,
            labels=label_map,
            failed_subjects=FAILED_SUBJECTS,
            n_samples=N_SAMPLES,
            v_lead_indices=V_LEAD_INDICES,
            use_3_lead=use_3_lead,
        )

        logger.info("Loading validation signals...")
        val_signals, val_labels, val_loaded_ids = load_dataset(
            data_dir=DATA_DIR,
            patient_ids=val_ids,
            labels=label_map,
            failed_subjects=FAILED_SUBJECTS,
            n_samples=N_SAMPLES,
            v_lead_indices=V_LEAD_INDICES,
            use_3_lead=use_3_lead,
        )

        # --- Preprocessing (fit on train, apply to both) ---
        stats_path = os.path.join(PREPROC_STATS_DIR, f"fold_{fold_id}_normalizer")
        train_proc, val_proc, normalizer = preprocess_fold(
            train_signals=train_signals,
            val_signals=val_signals,
            kernel_samples=MEDIAN_FILTER_SAMPLES,
            apply_lp=APPLY_LP_FILTER,
            lp_cutoff=LP_CUTOFF_HZ,
            lp_order=LP_ORDER,
            fs=100.0,
            stats_save_path=stats_path,
        )

        # --- DataLoaders ---
        train_loader = make_dataloader(
            signals=train_proc, labels=train_labels, patient_ids=train_loaded_ids,
            batch_size=args.bs, shuffle=True, augment=use_aug,
            seed=args.seed + fold_id,
            noise_std=AUG_NOISE_STD, amp_min=AUG_AMP_MIN, amp_max=AUG_AMP_MAX,
        )
        val_loader = make_dataloader(
            signals=val_proc, labels=val_labels, patient_ids=val_loaded_ids,
            batch_size=args.bs, shuffle=False, augment=False,
        )

        # --- Build fresh model ---
        model = ECGResNet(
            in_channels=in_channels,
            dropout=args.dropout,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

        num_pos = np.sum(train_labels == 1)
        num_neg = np.sum(train_labels == 0)
        fold_pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(device)
        
        logger.info(f"Class Distribution: Brugada={num_pos}, Normal={num_neg}")
        logger.info(f"Dynamic pos_weight for Fold {fold_id}: {fold_pos_weight.item():.4f}")
        
        # --- Train ---
        model_path = os.path.join(MODEL_DIR, f"cnn_fold_{fold_id}.pt")
        model, history = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            fold_id=fold_id,
            model_save_path=model_path,
            learning_rate=args.lr,
            weight_decay=args.wd,
            max_epochs=args.epochs,
            patience=args.patience,
            pos_weight=fold_pos_weight,
            device=device,
        )

        # --- Extract OOF embeddings and probabilities ---
        logger.info("Extracting OOF embeddings...")
        # Use val_loader without augmentation (already set above)
        val_loader_no_aug = make_dataloader(
            signals=val_proc, labels=val_labels, patient_ids=val_loaded_ids,
            batch_size=args.bs, shuffle=False, augment=False,
        )
        embeddings, probs, _ = extract_embeddings(model, val_loader_no_aug, device)

        # --- Compute metrics ---
        fold_m = compute_metrics(
            y_true=val_labels,
            y_prob=probs,
            target_specificity=TARGET_SPECIFICITY_FOR_SENSITIVITY,
        )
        fold_m["fold_id"] = fold_id
        fold_m["best_epoch"] = history["best_epoch"]
        fold_metrics_list.append(fold_m)
        print_fold_metrics(fold_id, fold_m)

        # --- Check minimum performance gate ---
        if fold_m["auroc"] < 0.70:
            logger.warning(
                f"Fold {fold_id} AUROC={fold_m['auroc']:.4f} is very low. "
                "Consider revisiting preprocessing or architecture."
            )

        # --- Accumulate OOF outputs ---
        for i, pid in enumerate(val_loaded_ids):
            oof_probs_rows.append({
                "patient_id":       str(pid),
                "brugada":          int(val_labels[i]),
                "oof_prob_brugada": float(probs[i]),
                "fold_id":          fold_id,
            })
            oof_embeddings_dict[str(pid)] = embeddings[i]

        logger.info(f"Fold {fold_id} complete. Accumulated {len(oof_probs_rows)} OOF rows.")

    # =========================================================================
    # 6. ASSEMBLE AND SAVE DELIVERABLES
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("  ASSEMBLING DELIVERABLES")
    logger.info("="*60)

    # --- cnn_fold_probs.csv ---
    probs_df = pd.DataFrame(oof_probs_rows)
    probs_df["patient_id"] = probs_df["patient_id"].astype(str)
    probs_df = probs_df.sort_values("patient_id").reset_index(drop=True)

    if len(probs_df) != len(fold_df):
        logger.error(f"probs_df has {len(probs_df)} rows but expected {len(fold_df)}")
    probs_df.to_csv(OUT_FOLD_PROBS, index=False)
    logger.info(f"Saved: {OUT_FOLD_PROBS} ({len(probs_df)} rows)")

    # --- cnn_embeddings.csv ---
    embed_rows = []
    for pid in all_patient_ids:
        pid = str(pid)
        emb = oof_embeddings_dict.get(pid, np.full(EMBED_DIM, np.nan, dtype=np.float32))
        row = {"patient_id": pid}
        for j, v in enumerate(emb):
            row[f"cnn_embed_{j}"] = float(v)
        embed_rows.append(row)

    embed_df = pd.DataFrame(embed_rows)
    embed_df["patient_id"] = embed_df["patient_id"].astype(str)
    embed_df.to_csv(OUT_EMBEDDINGS, index=False)
    logger.info(f"Saved: {OUT_EMBEDDINGS} ({len(embed_df)} rows)")

    # --- 5-fold OOF summary metrics ---
    # Use the combined OOF predictions (all folds together) for single summary AUROC
    valid_probs_df = probs_df.dropna(subset=["oof_prob_brugada"])
    oof_summary_combined = compute_metrics(
        y_true=valid_probs_df["brugada"].values,
        y_prob=valid_probs_df["oof_prob_brugada"].values,
        target_specificity=TARGET_SPECIFICITY_FOR_SENSITIVITY,
    )

    per_fold_agg = aggregate_fold_metrics(fold_metrics_list)
    print_summary_metrics(per_fold_agg)

    # --- results/cnn_cv_summary.json ---
    summary = {
        **per_fold_agg,
        "oof_combined_auroc": oof_summary_combined["auroc"],
        "best_params": {
            "learning_rate": args.lr,
            "weight_decay":  args.wd,
            "batch_size":    args.bs,
            "dropout":       args.dropout,
            "max_epochs":    args.epochs,
            "patience":      args.patience,
            "leads":         args.leads,
            "pos_weight":    POS_WEIGHT,
            "augmentation":  use_aug,
            "seed":          args.seed,
        },
        "n_parameters":       n_params,
        "failed_subjects":    list(FAILED_SUBJECTS.keys()),
        "failed_subject_handling": "excluded from training; NaN embeddings/probs in outputs",
        "fold_details":       fold_metrics_list,
    }

    ensure_dirs(os.path.dirname(OUT_CV_SUMMARY))
    save_json(summary, OUT_CV_SUMMARY)
    logger.info(f"Saved: {OUT_CV_SUMMARY}")

    # =========================================================================
    # 7. FINAL PERFORMANCE GATE CHECK
    # =========================================================================
    auroc_mean = per_fold_agg.get("auroc_mean", 0.0)
    auprc_mean = per_fold_agg.get("auprc_mean", 0.0)
    sens_mean  = per_fold_agg.get("sensitivity_mean", 0.0)

    logger.info("\n" + "="*60)
    logger.info("  MINIMUM PERFORMANCE GATE CHECK")
    logger.info("="*60)
    gate_data = {
        "AUROC mean > 0.85":       auroc_mean > 0.85,
        "AUPRC mean > 0.70":       auprc_mean > 0.70,
        "Sensitivity mean > 0.75": sens_mean  > 0.75,
    }

    gate_passed = True
    for desc, val in gate_data.items():
        threshold = float(desc.split(">")[-1])
        passed = val > threshold
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {desc} (actual={val:.4f})")
        if not passed:
            gate_passed = False

    if gate_passed:
        logger.info("\n  ALL GATES PASSED — ready for integration checks.")
    else:
        logger.warning("\n  ONE OR MORE GATES FAILED — revisit model before delivery.")

    logger.info("\nTraining complete.")
    return summary


if __name__ == "__main__":
    main()