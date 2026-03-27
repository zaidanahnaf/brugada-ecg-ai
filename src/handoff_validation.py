# src/handoff_validation.py

"""
Pre-flight validation suite.
Run before any hybrid fusion experiment.
Catches misalignment, leakage, and schema violations early.
"""

import numpy as np
import pandas as pd
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from src.config.__init__ import CFG

logger = logging.getLogger(__name__)


def validate_all_handoff_files(
    feature_matrix_path:    str = "outputs/features/feature_matrix.csv",
    feature_manifest_path:  str = "outputs/features/feature_manifest.json",
    fold_assignments_path:  str = "data/splits/fold_assignments.csv",
    oof_probs_hc_path:      str = "outputs/features/oof_predictions_handcrafted.csv",
    cnn_embeddings_path:    str = "outputs/features/cnn_embeddings.csv",
    cnn_oof_probs_path:     str = "outputs/features/cnn_fold_probs.csv",
    expected_n_subjects:    int = 363,
    expected_n_positive:    int = 76,
    expected_n_folds:       int = 5
) -> Dict[str, bool]:
    """
    Run all validation checks. Returns dict of {check_name: passed}.
    """
    results = {}

    # ── Check 1: Fold assignments ─────────────────────────────────
    results['fold_file_exists'] = _check_file_exists(fold_assignments_path)
    if results['fold_file_exists']:
        results.update(
            _validate_fold_assignments(
                fold_assignments_path,
                expected_n_subjects,
                expected_n_positive,
                expected_n_folds
            )
        )

    # ── Check 2: Feature matrix ───────────────────────────────────
    results['feature_matrix_exists'] = _check_file_exists(feature_matrix_path)
    if results['feature_matrix_exists']:
        results.update(
            _validate_feature_matrix(feature_matrix_path, fold_assignments_path)
        )

    # ── Check 3: OOF predictions (HC) ────────────────────────────
    results['oof_hc_exists'] = _check_file_exists(oof_probs_hc_path)
    if results['oof_hc_exists']:
        results.update(
            _validate_oof_predictions(
                oof_probs_hc_path, fold_assignments_path, prefix='hc'
            )
        )

    # ── Check 4: CNN embeddings ───────────────────────────────────
    results['cnn_embeddings_exists'] = _check_file_exists(cnn_embeddings_path)
    if results['cnn_embeddings_exists']:
        results.update(
            _validate_cnn_embeddings(cnn_embeddings_path, feature_matrix_path)
        )

    # ── Check 5: CNN OOF predictions ─────────────────────────────
    results['cnn_oof_exists'] = _check_file_exists(cnn_oof_probs_path)
    if results['cnn_oof_exists']:
        results.update(
            _validate_oof_predictions(
                cnn_oof_probs_path, fold_assignments_path, prefix='cnn'
            )
        )

    # ── Check 6: Cross-party label consistency ────────────────────
    if all([
        results.get('oof_hc_exists', False),
        results.get('cnn_oof_exists', False)
    ]):
        results['label_consistency'] = _check_label_consistency(
            oof_probs_hc_path, cnn_oof_probs_path
        )

    # ── Summary ───────────────────────────────────────────────────
    _print_validation_report(results)
    return results


def _check_file_exists(path: str) -> bool:
    exists = Path(path).exists()
    if not exists:
        logger.warning(f"FILE NOT FOUND: {path}")
    return exists


def _validate_fold_assignments(
    path: str,
    expected_n: int,
    expected_pos: int,
    expected_folds: int
) -> Dict[str, bool]:
    df = pd.read_csv(path)
    checks = {}

    checks['fold_n_subjects'] = len(df) == expected_n
    if not checks['fold_n_subjects']:
        logger.error(f"Fold: expected {expected_n} rows, got {len(df)}")

    checks['fold_n_positive'] = int(df[CFG.data.target_col].sum()) == expected_pos
    if not checks['fold_n_positive']:
        logger.error(
            f"Fold: expected {expected_pos} positive, "
            f"got {int(df[CFG.data.target_col].sum())}"
        )

    actual_folds = df['fold_id'].nunique()
    checks['fold_n_folds'] = actual_folds == expected_folds
    if not checks['fold_n_folds']:
        logger.error(f"Fold: expected {expected_folds} folds, got {actual_folds}")

    checks['fold_no_overlap'] = _check_no_fold_overlap(df)
    checks['fold_stratified'] = _check_fold_stratification(df, expected_folds)

    # Compute and log SHA256
    sha = _sha256_file(path)
    logger.info(f"Fold assignments SHA256: {sha}")
    checks['fold_sha256_logged'] = True

    return checks


def _check_no_fold_overlap(fold_df: pd.DataFrame) -> bool:
    """Each patient_id should appear exactly once."""
    n_unique = fold_df['patient_id'].nunique()
    no_dup = n_unique == len(fold_df)
    if not no_dup:
        logger.error(
            f"Fold overlap detected: {len(fold_df) - n_unique} duplicates"
        )
    return no_dup


def _check_fold_stratification(
    fold_df: pd.DataFrame,
    n_folds: int
) -> bool:
    """Check that each fold has reasonable positive class representation."""
    ok = True
    for f in range(n_folds):
        fold_data = fold_df[fold_df['fold_id'] == f]
        pos_rate = fold_data[CFG.data.target_col].mean()
        global_rate = fold_df[CFG.data.target_col].mean()
        # Allow ±30% deviation from global rate
        if abs(pos_rate - global_rate) / global_rate > 0.30:
            logger.warning(
                f"Fold {f}: positive rate {pos_rate:.3f} deviates "
                f">30% from global {global_rate:.3f}"
            )
            ok = False
    return ok


def _validate_feature_matrix(
    feat_path: str,
    fold_path: str
) -> Dict[str, bool]:
    feat_df = pd.read_csv(feat_path)
    fold_df = pd.read_csv(fold_path)
    checks = {}

    # patient_id alignment
    feat_ids = set(feat_df['patient_id'].astype(str))
    fold_ids = set(fold_df['patient_id'].astype(str))
    checks['feat_id_alignment'] = feat_ids == fold_ids
    if not checks['feat_id_alignment']:
        diff = feat_ids.symmetric_difference(fold_ids)
        logger.error(f"Feature matrix ID mismatch: {len(diff)} subjects differ")

    # No target leakage columns
    forbidden = ['basal_pattern', 'sudden_death']
    leak_found = [c for c in forbidden if c in feat_df.columns]
    checks['feat_no_metadata_leakage'] = len(leak_found) == 0
    if leak_found:
        logger.error(
            f"LEAKAGE: feature_matrix.csv contains forbidden columns: {leak_found}"
        )

    # Required columns present
    required = ['patient_id', CFG.data.target_col, 'pipeline_status']
    missing = [c for c in required if c not in feat_df.columns]
    checks['feat_required_cols'] = len(missing) == 0
    if missing:
        logger.error(f"Feature matrix missing required columns: {missing}")

    return checks


def _validate_oof_predictions(
    oof_path: str,
    fold_path: str,
    prefix: str
) -> Dict[str, bool]:
    oof_df = pd.read_csv(oof_path)
    fold_df = pd.read_csv(fold_path)
    checks = {}

    # One row per subject
    n_unique = oof_df['patient_id'].nunique()
    checks[f'oof_{prefix}_one_row_per_subject'] = n_unique == len(oof_df)

    # Probability range [0, 1]
    probs = oof_df['oof_prob_brugada']
    checks[f'oof_{prefix}_prob_range'] = (
        probs.between(0, 1).all()
    )
    if not checks[f'oof_{prefix}_prob_range']:
        logger.error(
            f"OOF {prefix}: probabilities outside [0,1]: "
            f"{(~probs.between(0, 1)).sum()} violations"
        )

    # Fold_id consistency
    oof_fold_map = dict(zip(oof_df['patient_id'].astype(str),
                             oof_df['fold_id']))
    ref_fold_map = dict(zip(fold_df['patient_id'].astype(str),
                             fold_df['fold_id']))
    mismatches = sum(
        1 for pid, fid in oof_fold_map.items()
        if pid in ref_fold_map and ref_fold_map[pid] != fid
    )
    checks[f'oof_{prefix}_fold_consistency'] = mismatches == 0
    if mismatches > 0:
        logger.error(
            f"OOF {prefix}: {mismatches} fold_id mismatches vs fold_assignments.csv"
        )

    return checks


def _validate_cnn_embeddings(
    cnn_path: str,
    feat_path: str
) -> Dict[str, bool]:
    cnn_df = pd.read_csv(cnn_path)
    feat_df = pd.read_csv(feat_path)
    checks = {}

    # patient_id overlap
    cnn_ids = set(cnn_df['patient_id'].astype(str))
    feat_ids = set(feat_df['patient_id'].astype(str))
    overlap = len(cnn_ids & feat_ids)
    total = len(feat_ids)
    checks['cnn_subject_overlap'] = overlap >= total * 0.95
    if overlap < total:
        logger.warning(
            f"CNN embeddings: {total - overlap} subjects missing "
            f"(expected {total}, got {overlap})"
        )

    # Embedding columns exist
    embed_cols = [c for c in cnn_df.columns if c.startswith('cnn_embed_')]
    checks['cnn_embed_cols_exist'] = len(embed_cols) > 0
    if not embed_cols:
        logger.error(
            "CNN embeddings: no 'cnn_embed_*' columns found. "
            "Expected cnn_embed_0, cnn_embed_1, ..."
        )

    # No NaN > 10% per embedding dimension
    if embed_cols:
        max_missing = cnn_df[embed_cols].isna().mean().max()
        checks['cnn_embed_low_missing'] = max_missing < 0.10
        if max_missing >= 0.10:
            logger.warning(
                f"CNN embedding: max missing rate = {max_missing:.1%} "
                f"(threshold 10%)"
            )

    return checks


def _check_label_consistency(
    oof_hc_path: str,
    oof_cnn_path: str
) -> bool:
    hc = pd.read_csv(oof_hc_path)[['patient_id', CFG.data.target_col]]
    cnn = pd.read_csv(oof_cnn_path)[['patient_id', CFG.data.target_col]]

    merged = hc.merge(cnn, on='patient_id', suffixes=('_hc', '_cnn'))
    mismatches = (
        merged[f'{CFG.data.target_col}_hc'] != merged[f'{CFG.data.target_col}_cnn']
    ).sum()

    if mismatches > 0:
        logger.error(
            f"Label inconsistency: {mismatches} subjects have different "
            f"labels in HC vs CNN OOF files."
        )
    return mismatches == 0


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def _print_validation_report(results: Dict[str, bool]) -> None:
    print("\n" + "="*60)
    print("HANDOFF VALIDATION REPORT")
    print("="*60)
    passed = sum(results.values())
    total = len(results)
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {check}")
    print("-"*60)
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  ✓ ALL CHECKS PASSED — Fusion experiments may proceed.")
    else:
        print("  ✗ FAILURES DETECTED — Resolve before running hybrid experiments.")
    print("="*60 + "\n")