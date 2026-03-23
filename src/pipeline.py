# src/pipeline.py

import numpy as np
from packaging import metadata
from packaging import metadata
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from src.config import CFG
from src.data.data_loader import load_record, load_metadata, get_lead_index
from src.preprocessing.preprocessing import preprocess_signal
from src.preprocessing.beat_segmentation import detect_rpeaks, filter_beats, extract_beat_windows
from src.fiducial_detection import detect_fiducials
from src.feature_extractors.signal_quality import extract_signal_quality_features
from src.feature_extractors.st_features import extract_st_features_single_beat
from src.feature_extractors.morphology_features import extract_morphology_features_single_beat
from src.feature_extractors.crosslead_features import extract_crosslead_features
from src.fold_manager import create_folds, load_folds
from src.preprocessing.aggregation import aggregate_beat_features

logger = logging.getLogger(__name__)

fold_path = "data/splits/fold_assignments.csv"

if Path(fold_path).exists():
    print(f"Fold assignments already exist — loading from {fold_path}")
    fold_df = load_folds(fold_path)
else:
    print("Creating fold assignments for the first time...")
    fold_df = create_folds(metadata)
    
def run_subject(
    patient_id: str,
    data_dir: str = CFG.data_dir
) -> Dict:
    """
    Full pipeline for one subject.

    Returns
    -------
    dict:
        patient_id       : str
        features         : dict of aggregated features (or {})
        pipeline_status  : 'OK' | 'PARTIAL' | 'FAILED'
        failure_reason   : str or None
        warnings         : List[str]
        n_valid_beats    : int
    """
    result = {
        'patient_id': patient_id,
        'features': {},
        'pipeline_status': 'FAILED',
        'failure_reason': None,
        'warnings': [],
        'n_valid_beats': 0
    }

    # ── Step 1: Load ──────────────────────────────────────────────
    record = load_record(patient_id, data_dir)
    if not record['valid']:
        result['failure_reason'] = record['failure_reason']
        return result

    signal_raw = record['signal']
    fs = record['fs']
    lead_names = record['lead_names']

    if record.get('missing_priority_leads'):
        result['warnings'].append(
            f"MISSING_PRIORITY_LEADS:{record['missing_priority_leads']}"
        )

    # ── Step 2: Preprocess ────────────────────────────────────────
    signal = preprocess_signal(signal_raw, fs=fs)

    # ── Step 3: R-Peak Detection ──────────────────────────────────
    rpeak_result = detect_rpeaks(signal, fs=fs, lead_names=lead_names)
    if rpeak_result['failed']:
        result['failure_reason'] = rpeak_result['failure_reason']
        return result

    if rpeak_result['method_used'] != CFG.rpeak_method:
        result['warnings'].append(f"RPEAK_FALLBACK:{rpeak_result['method_used']}")

    # ── Step 4: Beat Filtering ────────────────────────────────────
    filter_result = filter_beats(
        rpeak_result['rpeaks'],
        rpeak_result['rr_ms'],
        signal.shape[0],
        fs=fs
    )
    if filter_result['failed']:
        result['failure_reason'] = filter_result['failure_reason']
        return result

    valid_rpeaks = filter_result['valid_rpeaks']
    median_rr_ms = filter_result['median_rr_ms']
    result['n_valid_beats'] = filter_result['n_valid']

    # ── Step 5: Signal Quality Features (whole-record) ───────────
    sq_features = extract_signal_quality_features(
        signal, lead_names, rpeak_result['rr_ms'], fs
    )

    # ── Step 6: Beat Windows ──────────────────────────────────────
    beats = extract_beat_windows(signal, valid_rpeaks, fs)
    # beats: (n_beats, window_samples, n_leads)

    # ── Step 7: Per-Beat Feature Extraction ───────────────────────
    per_beat_all = []

    # Confidence accumulators: {lead: {'low': int, 'fallback': int, 'total': int}}
    confidence_counts = {
        lead: {'low': 0, 'fallback': 0, 'total': 0}
        for lead in CFG.priority_leads
    }

    for beat_idx, beat in enumerate(beats):
        beat_features = {}

        for lead in CFG.priority_leads:
            lead_idx = get_lead_index(lead_names, lead)
            if lead_idx is None:
                result['warnings'].append(
                    f"BEAT{beat_idx}_LEAD_{lead}_NOT_FOUND"
                )
                continue

            fid = detect_fiducials(beat, lead_idx, fs, median_rr_ms)
            j_conf = fid['confidence'].get('j_point', 'OK')

            # ── Accumulate confidence counts ──────────────────────
            confidence_counts[lead]['total'] += 1
            if 'LOW' in j_conf:
                confidence_counts[lead]['low'] += 1
                result['warnings'].append(
                    f"BEAT{beat_idx}_{lead}_J_POINT_LOW_CONFIDENCE"
                )
            if 'FALLBACK' in j_conf:
                confidence_counts[lead]['fallback'] += 1
                # Only warn on first fallback per lead to reduce log noise
                if confidence_counts[lead]['fallback'] == 1:
                    result['warnings'].append(
                        f"BEAT0_{lead}_J_POINT_FALLBACK (subsequent suppressed)"
                    )

            st_feats = extract_st_features_single_beat(
                beat, lead_idx, fid, lead, fs
            )
            beat_features.update(st_feats)

            morph_feats = extract_morphology_features_single_beat(
                beat, lead_idx, fid, st_feats, lead, fs
            )
            beat_features.update(morph_feats)

        per_beat_all.append(beat_features)

    # ── Step 8: Aggregate Beat → Subject ──────────────────────────
    aggregated = aggregate_beat_features(per_beat_all)

    # ── Step 9: Cross-Lead Features ───────────────────────────────
    cl_features = extract_crosslead_features(aggregated)

    # ── Step 9b: Compute J-Point Confidence Features ───────────────────
    confidence_features = _compute_confidence_features(confidence_counts)

    # ── Step 10: Combine All Features ─────────────────────────────────
    all_features = {
        **sq_features,
        **aggregated,
        **cl_features,
        **confidence_features,          # ← NEW: confidence quality features
    }
    all_features['n_valid_beats'] = filter_result['n_valid']
    all_features['median_rr_ms'] = median_rr_ms

    result['features'] = all_features
    result['pipeline_status'] = (
        'PARTIAL' if result['warnings'] else 'OK'
    )
    return result


def run_full_pipeline(
    patient_ids: List[str],
    metadata: pd.DataFrame,
    data_dir: str = CFG.data_dir,
    output_dir: str = CFG.features_dir
) -> pd.DataFrame:
    """
    Run pipeline for all subjects.
    Returns a subject × feature DataFrame.
    Saves feature_matrix.csv and failed_subjects.csv.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    failed = []

    for pid in patient_ids:
        logger.info(f"Processing {pid} ...")
        out = run_subject(pid, data_dir)

        row = {'patient_id': pid}
        row.update(out['features'])
        row['pipeline_status'] = out['pipeline_status']
        row['n_valid_beats'] = out['n_valid_beats']
        rows.append(row)

        if out['pipeline_status'] == 'FAILED':
            failed.append({
                'patient_id': pid,
                'failure_reason': out['failure_reason'],
                'warnings': '|'.join(out['warnings'])
            })
            logger.error(f"[{pid}] FAILED: {out['failure_reason']}")
        elif out['warnings']:
            logger.warning(f"[{pid}] PARTIAL: {out['warnings']}")

    feature_df = pd.DataFrame(rows)

    # Merge labels
    feature_df = feature_df.merge(
        metadata[['patient_id', CFG.target_col]],
        on='patient_id', how='left'
    )

    # Save
    feature_df.to_csv(f"{output_dir}/feature_matrix.csv", index=False)

    if failed:
        pd.DataFrame(failed).to_csv(
            f"{output_dir}/failed_subjects.csv", index=False
        )

    logger.info(
        f"Pipeline complete. "
        f"OK: {(feature_df['pipeline_status']=='OK').sum()} | "
        f"PARTIAL: {(feature_df['pipeline_status']=='PARTIAL').sum()} | "
        f"FAILED: {len(failed)}"
    )
    return feature_df

def _compute_confidence_features(
    confidence_counts: dict
) -> dict:
    """
    Compute subject-level J-point confidence quality features
    from per-beat accumulator counts.

    Output columns:
        qc_{lead}_j_low_conf_rate   : float [0, 1]
            Fraction of valid beats with LOW_CONFIDENCE J-point detection.
            0.0 = all beats had reliable J-point.
            1.0 = every beat had uncertain J-point → feature values unreliable.

        qc_{lead}_j_fallback_rate   : float [0, 1]
            Fraction of valid beats using a FALLBACK J-point estimate
            (S_nadir + fixed offset rather than derivative minimum).

        qc_any_lead_j_unreliable    : int {0, 1}
            Binary flag: 1 if ANY priority lead has j_low_conf_rate > 0.5.
            This is the primary quality gate feature for downstream models.

    Clinical rationale:
        A subject with high j_low_conf_rate has unreliable ST features.
        This is a confounder: a truly Brugada subject with noisy V1 may
        appear Normal because J-point features are attenuated. Including
        this flag allows models to discount ST features when confidence
        is low, and alerts clinicians to request a repeat ECG.

    Leakage safety:
        These features are computed from signal processing metadata only.
        They carry no target information — a Normal subject with noisy V1
        also gets a high qc_ value. Safe to include in all experiments.
    """
    features = {}
    any_unreliable = False

    for lead, counts in confidence_counts.items():
        total = max(counts['total'], 1)   # Avoid div-by-zero

        low_rate = counts['low'] / total
        fallback_rate = counts['fallback'] / total

        features[f'qc_{lead}_j_low_conf_rate'] = float(low_rate)
        features[f'qc_{lead}_j_fallback_rate'] = float(fallback_rate)

        if low_rate > 0.5:
            any_unreliable = True

    features['qc_any_lead_j_unreliable'] = int(any_unreliable)

    return features