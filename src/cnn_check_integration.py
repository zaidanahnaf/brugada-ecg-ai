"""
scripts/check_integration.py

Mandatory integration checks before delivering outputs to Person 4.

Checks performed:
  1. SHA256 of fold_assignments.csv
  2. Fold ID alignment: cnn_fold_probs.csv vs fold_assignments.csv
  3. Patient ID type: all must be strings
  4. Row completeness: 363 rows in both output CSVs
  5. No missing patient_ids
  6. OOF probability sanity: values in [0, 1]
  7. Embedding shape: 64 dimensions per subject

Usage:
    python scripts/check_integration.py

Exits with code 0 on full PASS, code 1 on any FAIL.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.cnn_config import (
    EMBED_DIM, FAILED_SUBJECTS, FOLD_CSV, FOLD_CSV_SHA256,
    OUT_EMBEDDINGS, OUT_FOLD_PROBS,
)
from src.cnn_utils import get_logger, sha256_file

logger = get_logger("check_integration")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def run_checks() -> bool:
    all_passed = True

    print("\n" + "="*60)
    print("  INTEGRATION CHECKS — Brugada CNN Branch")
    print("="*60 + "\n")

    # ------------------------------------------------------------------
    # CHECK 1: SHA256
    # ------------------------------------------------------------------
    print("CHECK 1: SHA256 of fold_assignments.csv")
    try:
        actual = sha256_file(FOLD_CSV)
        if actual == FOLD_CSV_SHA256:
            print(f"  {PASS} SHA256 matches: {actual[:16]}...")
        else:
            print(f"  {FAIL} SHA256 MISMATCH")
            print(f"    Expected: {FOLD_CSV_SHA256}")
            print(f"    Actual  : {actual}")
            all_passed = False
    except FileNotFoundError:
        print(f"  {FAIL} fold_assignments.csv not found at {FOLD_CSV}")
        all_passed = False

    # ------------------------------------------------------------------
    # CHECK 2: Load required files
    # ------------------------------------------------------------------
    print("\nCHECK 2: Required output files exist")
    for fpath in [OUT_FOLD_PROBS, OUT_EMBEDDINGS]:
        if os.path.exists(fpath):
            print(f"  {PASS} {fpath}")
        else:
            print(f"  {FAIL} MISSING: {fpath}")
            all_passed = False

    if not all_passed:
        print("\nCannot continue — required files missing.")
        return False

    # Load files
    fold_df  = pd.read_csv(FOLD_CSV, dtype={"patient_id": str})
    probs_df = pd.read_csv(OUT_FOLD_PROBS, dtype={"patient_id": str})
    embed_df = pd.read_csv(OUT_EMBEDDINGS, dtype={"patient_id": str})

    expected_n = len(fold_df)

    # ------------------------------------------------------------------
    # CHECK 3: Row counts
    # ------------------------------------------------------------------
    print(f"\nCHECK 3: Row completeness (expected {expected_n} rows)")
    for name, df in [("cnn_fold_probs.csv", probs_df), ("cnn_embeddings.csv", embed_df)]:
        n = len(df)
        ok = n == expected_n
        symbol = PASS if ok else FAIL
        print(f"  {symbol} {name}: {n} rows (expected {expected_n})")
        if not ok:
            all_passed = False

    # ------------------------------------------------------------------
    # CHECK 4: Patient ID type (must be str, not int)
    # ------------------------------------------------------------------
    print("\nCHECK 4: Patient ID type (must be str)")
    for name, df in [("cnn_fold_probs.csv", probs_df), ("cnn_embeddings.csv", embed_df)]:
        col_type = df["patient_id"].dtype
        all_str  = df["patient_id"].apply(lambda x: isinstance(x, str)).all()
        if all_str:
            print(f"  {PASS} {name}: patient_id is str")
        else:
            print(f"  {FAIL} {name}: patient_id dtype={col_type} — some values are not str")
            all_passed = False

    # ------------------------------------------------------------------
    # CHECK 5: Fold ID alignment
    # ------------------------------------------------------------------
    print("\nCHECK 5: Fold ID alignment with fold_assignments.csv")
    fold_map    = dict(zip(fold_df["patient_id"].astype(str), fold_df["fold_id"].astype(int)))
    probs_map   = dict(zip(probs_df["patient_id"].astype(str), probs_df["fold_id"]))

    mismatches = []
    for pid, expected_fold in fold_map.items():
        if pid not in probs_map:
            mismatches.append(f"  MISSING patient_id={pid} in cnn_fold_probs.csv")
            continue
        actual_fold = probs_map[pid]
        if pd.isna(actual_fold):
            # NaN fold_id only allowed for failed subjects
            if pid not in FAILED_SUBJECTS:
                mismatches.append(f"  NaN fold_id for non-failed subject {pid}")
        elif int(actual_fold) != int(expected_fold):
            mismatches.append(f"  fold_id mismatch: patient={pid} "
                              f"expected={expected_fold} actual={actual_fold}")

    if not mismatches:
        print(f"  {PASS} All fold IDs match fold_assignments.csv")
    else:
        for msg in mismatches[:10]:
            print(f"  {FAIL} {msg}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more mismatches")
        all_passed = False

    # ------------------------------------------------------------------
    # CHECK 6: OOF probability sanity
    # ------------------------------------------------------------------
    print("\nCHECK 6: OOF probability values in [0, 1]")
    valid_probs = probs_df["oof_prob_brugada"].dropna()
    out_of_range = ((valid_probs < 0) | (valid_probs > 1)).sum()
    n_nan = probs_df["oof_prob_brugada"].isna().sum()

    if out_of_range == 0:
        print(f"  {PASS} All {len(valid_probs)} valid probabilities in [0, 1]")
    else:
        print(f"  {FAIL} {out_of_range} probability values outside [0, 1]")
        all_passed = False

    if n_nan > 0:
        nan_ids = probs_df.loc[probs_df["oof_prob_brugada"].isna(), "patient_id"].tolist()
        expected_nan = set(FAILED_SUBJECTS.keys())
        unexpected = set(nan_ids) - expected_nan
        if unexpected:
            print(f"  {FAIL} Unexpected NaN probabilities for: {unexpected}")
            all_passed = False
        else:
            print(f"  {WARN} {n_nan} NaN probabilities — all are known failed subjects: {nan_ids}")

    # ------------------------------------------------------------------
    # CHECK 7: Embedding shape
    # ------------------------------------------------------------------
    print(f"\nCHECK 7: Embedding dimensions (expected {EMBED_DIM})")
    embed_cols = [c for c in embed_df.columns if c.startswith("cnn_embed_")]
    n_dims = len(embed_cols)
    if n_dims == EMBED_DIM:
        print(f"  {PASS} {n_dims} embedding dimensions found")
    else:
        print(f"  {FAIL} {n_dims} embedding dimensions found (expected {EMBED_DIM})")
        all_passed = False

    # NaN check in embeddings
    n_nan_embed = embed_df[embed_cols].isna().any(axis=1).sum()
    if n_nan_embed > 0:
        nan_embed_ids = embed_df.loc[embed_df[embed_cols].isna().any(axis=1), "patient_id"].tolist()
        unexpected = set(nan_embed_ids) - set(FAILED_SUBJECTS.keys())
        if unexpected:
            print(f"  {FAIL} Unexpected NaN embeddings for: {unexpected}")
            all_passed = False
        else:
            print(f"  {WARN} {n_nan_embed} subjects with NaN embeddings — all known failed subjects")

    # ------------------------------------------------------------------
    # CHECK 8: No duplicate patient_ids
    # ------------------------------------------------------------------
    print("\nCHECK 8: No duplicate patient_ids")
    for name, df in [("cnn_fold_probs.csv", probs_df), ("cnn_embeddings.csv", embed_df)]:
        dups = df["patient_id"].duplicated().sum()
        if dups == 0:
            print(f"  {PASS} {name}: no duplicates")
        else:
            dup_ids = df.loc[df["patient_id"].duplicated(keep=False), "patient_id"].unique()
            print(f"  {FAIL} {name}: {dups} duplicate patient_ids: {dup_ids[:5]}")
            all_passed = False

    # ------------------------------------------------------------------
    # CHECK 9: OOF label consistency
    # ------------------------------------------------------------------
    print("\nCHECK 9: OOF labels match fold_assignments.csv")
    label_map = dict(zip(fold_df["patient_id"].astype(str), fold_df["brugada"].astype(int)))
    probs_label_map = dict(zip(probs_df["patient_id"].astype(str), probs_df["brugada"]))
    label_mismatches = []
    for pid, expected_label in label_map.items():
        if pid not in probs_label_map:
            continue
        actual = probs_label_map[pid]
        if pd.isna(actual):
            continue
        if int(actual) != int(expected_label):
            label_mismatches.append(f"patient={pid} expected={expected_label} actual={actual}")

    if not label_mismatches:
        print(f"  {PASS} All labels match fold_assignments.csv")
    else:
        for msg in label_mismatches[:5]:
            print(f"  {FAIL} {msg}")
        all_passed = False

    # ------------------------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    if all_passed:
        print(f"  {PASS} ALL INTEGRATION CHECKS PASSED — READY FOR DELIVERY")
    else:
        print(f"  {FAIL} INTEGRATION CHECKS FAILED — DO NOT DELIVER")
    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    passed = run_checks()
    sys.exit(0 if passed else 1)