"""
scripts/qc_preprocessing.py

Visual quality control for preprocessing pipeline.
Plots raw vs preprocessed V1/V2/V3 signals for 3 Brugada + 3 Normal subjects.

PURPOSE:
  Confirm that ST segment morphology is PRESERVED after:
    1. Baseline wander removal (median filter)
    2. Low-pass filter
    3. Amplitude normalization

Run this BEFORE training to visually verify preprocessing is not distorting
the J-point or ST segment — the clinically critical region for Brugada.

Usage:
    python scripts/qc_preprocessing.py
    python scripts/qc_preprocessing.py --fold 0   # use fold 0 train stats
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import numpy as np

from src.cnn_config import (
    APPLY_LP_FILTER, DATA_DIR, FAILED_SUBJECTS, FOLD_CSV,
    LP_CUTOFF_HZ, LP_ORDER, MEDIAN_FILTER_SAMPLES, N_SAMPLES,
    SEED, V_LEAD_INDICES,
)
from src.data.cnn_data_loader import load_dataset, load_fold_assignments
from src.cnn_preprocessing import apply_lowpass_filter, remove_baseline_wander
from src.cnn_utils import get_logger, verify_fold_sha256
from src.cnn_config import FOLD_CSV_SHA256

logger = get_logger("qc_preprocessing")

FS          = 100
LEAD_NAMES  = ["V1", "V2", "V3"]
OUTPUT_DIR  = "results/qc_plots"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",     type=int, default=0, help="Fold to use for train/val split")
    parser.add_argument("--n-pos",    type=int, default=3, help="Number of Brugada subjects to plot")
    parser.add_argument("--n-neg",    type=int, default=3, help="Number of Normal subjects to plot")
    return parser.parse_args()


def plot_subject(
    pid: str,
    raw_signal: np.ndarray,
    proc_signal: np.ndarray,
    label: int,
    save_path: str,
    fs: int = 100,
):
    """
    Plot raw vs preprocessed V1/V2/V3 for a single subject.
    Highlights the ST segment region (80–320 ms after signal start as proxy).
    """
    t = np.arange(N_SAMPLES) / fs   # time in seconds
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    fig.suptitle(
        f"Subject {pid} | {'BRUGADA' if label == 1 else 'NORMAL'}\n"
        f"Left: Raw | Right: Preprocessed (median BW removal + LP 40Hz + normalization)",
        fontsize=10, fontweight="bold"
    )

    for row, (lead_name, lead_idx) in enumerate(zip(LEAD_NAMES, range(3))):
        raw   = raw_signal[lead_idx]
        proc  = proc_signal[lead_idx]

        # Left: raw
        ax_raw = axes[row, 0]
        ax_raw.plot(t, raw, color="steelblue", linewidth=0.8)
        ax_raw.set_ylabel(lead_name, fontsize=10)
        ax_raw.set_title("Raw" if row == 0 else "")
        ax_raw.axvspan(0.08, 0.32, alpha=0.1, color="red", label="ST region (proxy)")
        ax_raw.grid(True, alpha=0.3)

        # Right: preprocessed
        ax_proc = axes[row, 1]
        ax_proc.plot(t, proc, color="darkorange", linewidth=0.8)
        ax_proc.set_title("Preprocessed" if row == 0 else "")
        ax_proc.axvspan(0.08, 0.32, alpha=0.1, color="red")
        ax_proc.grid(True, alpha=0.3)

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {save_path}")


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Verify fold file
    verify_fold_sha256(FOLD_CSV, FOLD_CSV_SHA256, logger)
    fold_df  = load_fold_assignments(FOLD_CSV)
    label_map = dict(zip(fold_df["patient_id"].astype(str), fold_df["brugada"].astype(int)))

    # Pick sample subjects
    train_ids = fold_df.loc[fold_df["fold_id"] != args.fold, "patient_id"].astype(str).tolist()

    pos_ids = [p for p in train_ids
               if label_map.get(p, 0) == 1 and p not in FAILED_SUBJECTS][:args.n_pos]
    neg_ids = [p for p in train_ids
               if label_map.get(p, 0) == 0 and p not in FAILED_SUBJECTS][:args.n_neg]
    sample_ids = pos_ids + neg_ids

    logger.info(f"QC subjects: Brugada={pos_ids}, Normal={neg_ids}")

    # Load raw 3-lead signals
    raw_signals, raw_labels, loaded_ids = load_dataset(
        data_dir=DATA_DIR,
        patient_ids=sample_ids,
        labels=label_map,
        failed_subjects=FAILED_SUBJECTS,
        n_samples=N_SAMPLES,
        v_lead_indices=V_LEAD_INDICES,
        use_3_lead=True,
    )

    # Step-by-step preprocessing for visualization
    after_bw   = remove_baseline_wander(raw_signals, kernel_samples=MEDIAN_FILTER_SAMPLES)
    if APPLY_LP_FILTER:
        after_lp = apply_lowpass_filter(after_bw, cutoff_hz=LP_CUTOFF_HZ,
                                        order=LP_ORDER, fs=float(FS))
    else:
        after_lp = after_bw.copy()

    # Normalize using training fold stats (compute from all loaded subjects)
    mean_ = after_lp.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
    std_  = after_lp.std(axis=(0, 2), keepdims=True).clip(min=1e-8)
    after_norm = (after_lp - mean_) / std_

    # Plot each subject
    for i, pid in enumerate(loaded_ids):
        label = int(raw_labels[i])
        label_str = "brugada" if label == 1 else "normal"
        save_path = os.path.join(OUTPUT_DIR, f"qc_{label_str}_{pid}.png")
        plot_subject(
            pid=pid,
            raw_signal=raw_signals[i],
            proc_signal=after_norm[i],
            label=label,
            save_path=save_path,
            fs=FS,
        )

    logger.info(f"\nQC plots saved to: {OUTPUT_DIR}")
    logger.info("Review these plots to confirm ST segment morphology is preserved.")
    logger.info("IMPORTANT: The coved ST pattern in V1/V2 for Brugada subjects should")
    logger.info("remain visible after preprocessing — if it is absent, preprocessing is distorting the signal.")


if __name__ == "__main__":
    main()