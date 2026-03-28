"""
validate_feature_matrix.py
===========================
Validates feature_matrix.csv against the frozen schema spec.

Usage:
    python scripts/validate_feature_matrix.py
    python scripts/validate_feature_matrix.py --csv path/to/feature_matrix.csv

All checks raise ValueError on mismatch — no silent failures.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("validate_feature_matrix")

# ---------------------------------------------------------------------------
# Expected constants (mirrors prompt_metadata spec exactly)
# ---------------------------------------------------------------------------
EXPECTED_ROWS = 363
EXPECTED_TOTAL_COLS = 693
EXPECTED_FEATURE_COLS = 681
EXPECTED_BRUGADA_POS = 76
EXPECTED_BRUGADA_NEG = 287
EXPECTED_QC_COLS = 7

META_COLS = ["patient_id", "brugada", "pipeline_status", "n_valid_beats", "median_rr_ms"]
QC_PREFIX = "qc_"
QC_COLS_EXPECTED = [
    "qc_V1_j_low_conf_rate",
    "qc_V1_j_fallback_rate",
    "qc_V2_j_low_conf_rate",
    "qc_V2_j_fallback_rate",
    "qc_V3_j_low_conf_rate",
    "qc_V3_j_fallback_rate",
    "qc_any_lead_j_unreliable",
]

PIPELINE_STATUS_EXPECTED = {"OK": 356, "PARTIAL": 5, "FAILED": 2}

FAILED_SUBJECTS = ["267630", "1230482"]

FULLY_NAN_COLS_EXPECTED = [
    "morph_V3_trough_depth_between_humps_mean",
    "morph_V3_trough_depth_between_humps_median",
    "morph_V3_trough_depth_between_humps_std",
    "morph_V3_trough_depth_between_humps_min",
    "morph_V3_trough_depth_between_humps_max",
    "morph_V3_second_hump_amplitude_mean",
    "morph_V3_second_hump_amplitude_median",
    "morph_V3_second_hump_amplitude_std",
    "morph_V3_second_hump_amplitude_min",
    "morph_V3_second_hump_amplitude_max",
]

# Sentinel feature used as a spot-check for numerical stability
SENTINEL_COL = "st_V1_st_slope_j0_j40_mean"
SENTINEL_MEAN_EXPECTED = -0.00106
SENTINEL_MEAN_ATOL = 1e-4  # allow ±0.0001 across platform differences

# Fold spec
FOLD_SPEC = {
    0: {"n": 73, "brugada": 15},
    1: {"n": 73, "brugada": 15},
    2: {"n": 73, "brugada": 16},
    3: {"n": 72, "brugada": 15},
    4: {"n": 72, "brugada": 15},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(msg: str) -> None:
    """Raise a ValueError with a clearly formatted message."""
    raise ValueError(f"\n{'='*70}\nVALIDATION FAILED: {msg}\n{'='*70}")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    if rows != EXPECTED_ROWS:
        _fail(f"Row count mismatch: expected {EXPECTED_ROWS}, got {rows}")
    if cols != EXPECTED_TOTAL_COLS:
        _fail(
            f"Column count mismatch: expected {EXPECTED_TOTAL_COLS}, got {cols}. "
            f"Diff: {cols - EXPECTED_TOTAL_COLS:+d}"
        )
    log.info("✓ Shape: %d rows × %d columns", rows, cols)


def check_meta_cols(df: pd.DataFrame) -> None:
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        _fail(f"Missing meta columns: {missing}")
    log.info("✓ All 5 meta columns present")


def check_qc_cols(df: pd.DataFrame) -> None:
    actual_qc = [c for c in df.columns if c.startswith(QC_PREFIX)]
    if len(actual_qc) != EXPECTED_QC_COLS:
        _fail(
            f"QC column count mismatch: expected {EXPECTED_QC_COLS}, "
            f"got {len(actual_qc)}: {actual_qc}"
        )
    missing_qc = [c for c in QC_COLS_EXPECTED if c not in df.columns]
    if missing_qc:
        _fail(f"Expected QC columns not found: {missing_qc}")
    log.info("✓ QC columns: %d columns present, all named correctly", len(actual_qc))


def check_label_distribution(df: pd.DataFrame) -> None:
    counts = df["brugada"].value_counts()
    n_pos = int(counts.get(1, 0))
    n_neg = int(counts.get(0, 0))
    if n_pos != EXPECTED_BRUGADA_POS:
        _fail(
            f"Brugada-positive count mismatch: expected {EXPECTED_BRUGADA_POS}, got {n_pos}"
        )
    if n_neg != EXPECTED_BRUGADA_NEG:
        _fail(
            f"Brugada-negative count mismatch: expected {EXPECTED_BRUGADA_NEG}, got {n_neg}"
        )
    log.info("✓ Label distribution: %d positive, %d negative", n_pos, n_neg)


def check_pipeline_status(df: pd.DataFrame) -> None:
    counts = df["pipeline_status"].value_counts().to_dict()
    for status, expected_n in PIPELINE_STATUS_EXPECTED.items():
        actual_n = counts.get(status, 0)
        if actual_n != expected_n:
            _fail(
                f"pipeline_status='{status}' count mismatch: "
                f"expected {expected_n}, got {actual_n}"
            )
    log.info(
        "✓ Pipeline status: OK=%d, PARTIAL=%d, FAILED=%d",
        counts.get("OK", 0),
        counts.get("PARTIAL", 0),
        counts.get("FAILED", 0),
    )


def check_failed_subjects(df: pd.DataFrame) -> None:
    """Failed subjects must be present and have all-NaN feature columns."""
    feature_cols = [
        c for c in df.columns
        if c not in META_COLS and not c.startswith(QC_PREFIX)
    ]
    df_str = df.copy()
    df_str["patient_id"] = df_str["patient_id"].astype(str)

    for subj in FAILED_SUBJECTS:
        rows = df_str[df_str["patient_id"] == subj]
        if rows.empty:
            _fail(f"Failed subject '{subj}' not found in patient_id column")
        n_nan = rows[feature_cols].isna().sum().sum()
        n_total = len(feature_cols)
        if n_nan != n_total:
            _fail(
                f"Subject '{subj}' expected all-NaN features ({n_total} cols), "
                f"but only {n_nan} are NaN"
            )
    log.info("✓ Failed subjects %s: present with all-NaN features", FAILED_SUBJECTS)


def check_fully_nan_columns(df: pd.DataFrame) -> None:
    """The 10 documented fully-NaN columns must exist and be 100% NaN."""
    for col in FULLY_NAN_COLS_EXPECTED:
        if col not in df.columns:
            _fail(f"Expected fully-NaN column '{col}' not found in DataFrame")
        pct_nan = df[col].isna().mean()
        if pct_nan < 1.0:
            _fail(
                f"Column '{col}' expected to be 100% NaN, "
                f"but {pct_nan*100:.1f}% NaN (some values present — investigate)"
            )

    # Check for unexpected fully-NaN columns beyond the documented 10
    all_cols = [c for c in df.columns if c not in META_COLS and not c.startswith(QC_PREFIX)]
    unexpected_full_nan = [
        c for c in all_cols
        if df[c].isna().mean() == 1.0 and c not in FULLY_NAN_COLS_EXPECTED
    ]
    if unexpected_full_nan:
        _fail(
            f"Unexpected fully-NaN columns found (not in documented list): "
            f"{unexpected_full_nan}"
        )
    log.info(
        "✓ Fully-NaN columns: all 10 documented columns confirmed 100% NaN, "
        "no unexpected additional ones"
    )


def check_sentinel_feature(df: pd.DataFrame) -> None:
    """Spot-check numerical stability of a key discriminative feature."""
    if SENTINEL_COL not in df.columns:
        _fail(f"Sentinel column '{SENTINEL_COL}' not found — possible column rename")
    actual_mean = df[SENTINEL_COL].mean()
    if not np.isclose(actual_mean, SENTINEL_MEAN_EXPECTED, atol=SENTINEL_MEAN_ATOL):
        _fail(
            f"Sentinel feature '{SENTINEL_COL}' mean mismatch: "
            f"expected ≈{SENTINEL_MEAN_EXPECTED}, got {actual_mean:.6f} "
            f"(tolerance ±{SENTINEL_MEAN_ATOL})"
        )
    log.info(
        "✓ Sentinel feature '%s': mean=%.6f (expected ≈%.5f)",
        SENTINEL_COL,
        actual_mean,
        SENTINEL_MEAN_EXPECTED,
    )


def check_fold_assignments(df: pd.DataFrame, fold_path: Path) -> None:
    """If fold_assignments.csv is present, verify fold sizes and stratification."""
    if not fold_path.exists():
        log.warning(
            "⚠  fold_assignments.csv not found at %s — skipping fold check", fold_path
        )
        return

    folds = pd.read_csv(fold_path)
    folds["patient_id"] = folds["patient_id"].astype(str)
    df2 = df.copy()
    df2["patient_id"] = df2["patient_id"].astype(str)

    # Drop brugada from folds if present to avoid _x/_y suffix collision
    folds = folds.drop(columns=["brugada"], errors="ignore")
    merged = folds.merge(df2[["patient_id", "brugada"]], on="patient_id", how="left")

    for fold_id, spec in FOLD_SPEC.items():
        fold_rows = merged[merged["fold_id"] == fold_id]
        n_actual = len(fold_rows)
        n_brugada = int(fold_rows["brugada"].sum())
        if n_actual != spec["n"]:
            _fail(
                f"Fold {fold_id}: expected {spec['n']} subjects, got {n_actual}"
            )
        if n_brugada != spec["brugada"]:
            _fail(
                f"Fold {fold_id}: expected {spec['brugada']} Brugada-positive, "
                f"got {n_brugada}"
            )
    log.info("✓ Fold assignments: all 5 folds match expected sizes and stratification")


def check_manifest_coverage(df: pd.DataFrame, manifest_path: Path) -> None:
    """Warn (not fail) if manifest features are absent from CSV — known discrepancy."""
    if not manifest_path.exists():
        log.warning(
            "⚠  feature_manifest.json not found at %s — skipping manifest check",
            manifest_path,
        )
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    raw_features = manifest.get("features", [])
    # features may be a list of strings or a list of dicts with a 'feature_name' key
    if raw_features and isinstance(raw_features[0], dict):
        manifest_features = set(f["feature_name"] for f in raw_features if "feature_name" in f)
    else:
        manifest_features = set(raw_features)
    csv_features = set(
        c for c in df.columns
        if c not in META_COLS and not c.startswith(QC_PREFIX)
    )

    in_manifest_not_csv = manifest_features - csv_features
    in_csv_not_manifest = csv_features - manifest_features

    if in_manifest_not_csv:
        log.warning(
            "⚠  %d features in manifest but not in CSV (known discrepancy of ~8): %s",
            len(in_manifest_not_csv),
            sorted(in_manifest_not_csv)[:10],
        )
    if in_csv_not_manifest:
        log.warning(
            "⚠  %d features in CSV but not in manifest: %s",
            len(in_csv_not_manifest),
            sorted(in_csv_not_manifest)[:10],
        )

    log.info(
        "✓ Manifest cross-check: manifest=%d, CSV=%d, "
        "missing_from_csv=%d, extra_in_csv=%d",
        len(manifest_features),
        len(csv_features),
        len(in_manifest_not_csv),
        len(in_csv_not_manifest),
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def validate(csv_path: Path, fold_path: Path, manifest_path: Path) -> None:
    log.info("Loading %s ...", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    log.info("--- Running validation checks ---")
    check_shape(df)
    check_meta_cols(df)
    check_qc_cols(df)
    check_label_distribution(df)
    check_pipeline_status(df)
    check_failed_subjects(df)
    check_fully_nan_columns(df)
    check_sentinel_feature(df)
    check_fold_assignments(df, fold_path)
    check_manifest_coverage(df, manifest_path)

    # SHA-256 audit trail — always log, never fail on hash mismatch
    # (hash changes if the file is legitimately regenerated)
    sha = sha256_of_file(csv_path)
    log.info("SHA256(%s) = %s", csv_path.name, sha)
    log.info(
        "NOTE: Commit this SHA alongside the file for reproducibility audit. "
        "If this differs from the previously committed SHA, investigate before modeling."
    )

    log.info("=== ALL CHECKS PASSED ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate feature_matrix.csv against frozen schema spec."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("features/feature_matrix.csv"),
        help="Path to feature_matrix.csv (default: features/feature_matrix.csv)",
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=Path("data/splits/fold_assignments.csv"),
        help="Path to fold_assignments.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("features/feature_manifest.json"),
        help="Path to feature_manifest.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        validate(args.csv, args.folds, args.manifest)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)