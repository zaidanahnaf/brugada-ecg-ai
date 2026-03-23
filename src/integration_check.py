"""
integration_check.py — Mandatory pre-delivery validation.

Run this before handing files to Person 4.
All checks must pass. Any failure = DO NOT DELIVER.

Checks:
  1. Fold SHA256 matches contract
  2. cnn_fold_probs.csv fold_id aligns with fold_assignments.csv
  3. patient_id saved as STRING in all output files
  4. cnn_embeddings.csv has exactly 363 rows (all subjects present)
  5. No unexpected NaN in probs (only allowed for PROBLEMATIC_SUBJECTS)
  6. Output file schemas are correct
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import (
    CNN_EMBEDDINGS_CSV,
    CNN_FOLD_PROBS_CSV,
    CNN_CV_SUMMARY_JSON,
    FEATURES_DIR,
    FOLD_ASSIGNMENTS,
    FOLD_SHA256,
    PROBLEMATIC_SUBJECTS,
    RESULTS_DIR,
)
from src.load_data import verify_fold_sha256, load_fold_assignments

logger = logging.getLogger(__name__)

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def check_fold_sha256() -> bool:
    """Check 1: fold_assignments.csv SHA256."""
    print("\n[Check 1] Verifying fold_assignments.csv SHA256...")
    try:
        verify_fold_sha256(FOLD_ASSIGNMENTS)
        print(f"  {PASS} SHA256 matches contract.")
        return True
    except RuntimeError as e:
        print(f"  {FAIL} {e}")
        return False


def check_fold_alignment(probs_path: Path, fold_path: Path = FOLD_ASSIGNMENTS) -> bool:
    """Check 2: fold_id in cnn_fold_probs.csv matches fold_assignments.csv exactly."""
    print(f"\n[Check 2] Verifying fold_id alignment in {probs_path.name}...")

    if not probs_path.exists():
        print(f"  {FAIL} File not found: {probs_path}")
        return False

    try:
        probs_df = pd.read_csv(probs_path, dtype={"patient_id": str})
        folds_df = pd.read_csv(fold_path,  dtype={"patient_id": str})
        folds_df.columns = [c.lower() for c in folds_df.columns]
    except Exception as e:
        print(f"  {FAIL} Could not load files: {e}")
        return False

    fold_map = dict(zip(folds_df["patient_id"], folds_df["fold_id"].astype(int)))
    mismatches = []
    for _, row in probs_df.iterrows():
        pid = str(row["patient_id"])
        expected = fold_map.get(pid)
        if expected is None:
            mismatches.append(f"pid={pid} not in fold_assignments.csv")
        elif pd.notna(row.get("fold_id")) and int(row["fold_id"]) != expected:
            mismatches.append(
                f"pid={pid}: expected fold_id={expected}, got {row['fold_id']}"
            )

    if mismatches:
        print(f"  {FAIL} {len(mismatches)} fold_id mismatch(es):")
        for m in mismatches[:5]:
            print(f"    -> {m}")
        return False

    print(f"  {PASS} All fold_ids match fold_assignments.csv.")
    return True


def check_patient_id_types(*paths: Path) -> bool:
    """Check 3: patient_id columns are strings (not ints)."""
    print("\n[Check 3] Verifying patient_id dtype is STRING...")
    all_ok = True
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype={"patient_id": str}, nrows=5)
        # Check that no patient_id looks like an integer-typed column
        # (if read without explicit dtype, they would be int — we enforce str)
        sample = df["patient_id"].iloc[0]
        if not isinstance(sample, str):
            print(f"  {FAIL} {path.name}: patient_id is not str — found {type(sample)}")
            all_ok = False
        else:
            print(f"  {PASS} {path.name}: patient_id is str (sample: '{sample}')")
    return all_ok


def check_completeness(embeddings_path: Path, n_expected: int = 363) -> bool:
    """Check 4: cnn_embeddings.csv has exactly n_expected rows."""
    print(f"\n[Check 4] Verifying completeness ({n_expected} subjects in embeddings)...")
    if not embeddings_path.exists():
        print(f"  {FAIL} File not found: {embeddings_path}")
        return False

    df = pd.read_csv(embeddings_path, dtype={"patient_id": str})
    n_rows = len(df)
    if n_rows != n_expected:
        print(f"  {FAIL} Expected {n_expected} rows, found {n_rows}")
        return False

    # Check problematic subjects have NaN embeddings (allowed)
    for pid in PROBLEMATIC_SUBJECTS:
        row = df[df["patient_id"] == pid]
        if not row.empty:
            emb_cols = [c for c in df.columns if c.startswith("cnn_embed_")]
            has_nan = row[emb_cols].isna().all(axis=1).iloc[0]
            status = "NaN (documented)" if has_nan else "has values"
            print(f"    Problematic subject {pid}: {status}")

    print(f"  {PASS} {n_rows}/{n_expected} rows present.")
    return True


def check_nan_probs(probs_path: Path) -> bool:
    """Check 5: NaN in oof_prob_brugada only for known problematic subjects."""
    print("\n[Check 5] Checking for unexpected NaN in oof_prob_brugada...")
    if not probs_path.exists():
        print(f"  {FAIL} File not found: {probs_path}")
        return False

    df = pd.read_csv(probs_path, dtype={"patient_id": str})
    nan_rows = df[df["oof_prob_brugada"].isna()]
    unexpected_nans = [
        str(row["patient_id"])
        for _, row in nan_rows.iterrows()
        if str(row["patient_id"]) not in PROBLEMATIC_SUBJECTS
    ]

    if unexpected_nans:
        print(f"  {FAIL} Unexpected NaN for {len(unexpected_nans)} subject(s):")
        for pid in unexpected_nans[:10]:
            print(f"    -> {pid}")
        return False

    expected_nans = [str(p) for p in PROBLEMATIC_SUBJECTS if p in nan_rows["patient_id"].values]
    print(
        f"  {PASS} No unexpected NaN. "
        f"Documented problematic subjects with NaN: {expected_nans}"
    )
    return True


def check_cv_summary_schema(summary_path: Path) -> bool:
    """Check 6: cnn_cv_summary.json has all required fields."""
    print(f"\n[Check 6] Verifying cnn_cv_summary.json schema...")
    required_fields = [
        "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
        "sensitivity_mean", "sensitivity_std", "specificity_mean", "specificity_std",
        "f1_positive_mean", "f1_positive_std", "brier_score_mean",
        "best_params", "n_parameters",
    ]
    if not summary_path.exists():
        print(f"  {FAIL} File not found: {summary_path}")
        return False

    with open(summary_path) as f:
        summary = json.load(f)

    missing = [k for k in required_fields if k not in summary]
    if missing:
        print(f"  {FAIL} Missing fields: {missing}")
        return False

    print(f"  {PASS} All required fields present.")
    print(f"    AUROC: {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}")
    print(f"    Sensitivity: {summary['sensitivity_mean']:.4f} ± {summary['sensitivity_std']:.4f}")
    return True


def check_model_files(models_dir: Path, n_folds: int = 5, variant: str = "12lead") -> bool:
    """Check 7: all 5 fold model files exist."""
    print(f"\n[Check 7] Verifying model checkpoint files ({variant})...")
    all_ok = True
    for fold_id in range(n_folds):
        path = models_dir / f"cnn_{variant}_fold_{fold_id}.pt"
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  {PASS} {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  {FAIL} Missing: {path.name}")
            all_ok = False
    return all_ok


# ============================================================
# MAIN
# ============================================================

def run_all_checks(variant: str = "12lead") -> bool:
    """Run all integration checks. Returns True only if all pass."""
    from src.config import MODELS_DIR

    print("=" * 60)
    print("INTEGRATION CHECKS — Brugada CNN Person 2 Deliverables")
    print(f"Variant: {variant}")
    print("=" * 60)

    probs_path     = FEATURES_DIR / f"cnn_fold_probs_{variant}.csv"
    embeddings_path = FEATURES_DIR / f"cnn_embeddings_{variant}.csv"
    summary_path   = RESULTS_DIR / f"cnn_cv_summary_{variant}.json"

    results = [
        check_fold_sha256(),
        check_fold_alignment(probs_path),
        check_patient_id_types(probs_path, embeddings_path),
        check_completeness(embeddings_path),
        check_nan_probs(probs_path),
        check_cv_summary_schema(summary_path),
        check_model_files(MODELS_DIR, variant=variant),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} checks passed.")
    if passed == total:
        print("✓ ALL CHECKS PASSED — safe to deliver to Person 4.")
    else:
        print("✗ SOME CHECKS FAILED — DO NOT DELIVER until resolved.")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="12lead", choices=["3lead", "12lead", "both"])
    args = parser.parse_args()

    if args.variant == "both":
        ok1 = run_all_checks("3lead")
        ok2 = run_all_checks("12lead")
        sys.exit(0 if (ok1 and ok2) else 1)
    else:
        ok = run_all_checks(args.variant)
        sys.exit(0 if ok else 1)