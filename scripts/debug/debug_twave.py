# scripts/debug_twave.py

import numpy as np
import wfdb
from pathlib import Path
from src.config.__init__ import CFG
from src.data.data_loader import load_record, get_lead_index
from src.preprocessing.preprocessing import preprocess_signal
from src.preprocessing.beat_segmentation import detect_rpeaks, filter_beats, extract_beat_windows
from src.fiducial.fiducial_detection import detect_fiducials

def debug_one_subject(patient_id: str):
    print(f"\n{'='*60}")
    print(f"DEBUG: {patient_id}")
    print('='*60)

    # Load
    record = load_record(patient_id)
    if not record['valid']:
        print(f"Load failed: {record['failure_reason']}")
        return

    signal = preprocess_signal(record['signal'], fs=record['fs'])
    lead_names = record['lead_names']

    # R-peaks
    rpeak_result = detect_rpeaks(signal, fs=record['fs'], lead_names=lead_names)
    if rpeak_result['failed']:
        print(f"R-peak failed: {rpeak_result['failure_reason']}")
        return

    filter_result = filter_beats(
        rpeak_result['rpeaks'], rpeak_result['rr_ms'],
        signal.shape[0], fs=record['fs']
    )
    if filter_result['failed']:
        print(f"Beat filter failed: {filter_result['failure_reason']}")
        return

    beats = extract_beat_windows(signal, filter_result['valid_rpeaks'], record['fs'])
    median_rr = filter_result['median_rr_ms']

    print(f"Signal shape   : {signal.shape}")
    print(f"Beat window    : {beats.shape}  ({beats.shape[1]} samples = {beats.shape[1]/record['fs']*1000:.0f}ms)")
    print(f"Valid beats    : {filter_result['n_valid']}")
    print(f"Median RR      : {median_rr:.1f} ms")

    # Inspect first beat, V1 lead
    v1_idx = get_lead_index(lead_names, 'V1')
    if v1_idx is None:
        print("V1 not found")
        return

    beat = beats[0]
    fid = detect_fiducials(beat, v1_idx, record['fs'], median_rr)

    pre_samples = int(CFG.preprocessing.beat_window_pre_ms * record['fs'] / 1000)
    post_samples = int(CFG.preprocessing.beat_window_post_ms * record['fs'] / 1000)

    print(f"\n── Fiducials (beat 0, V1) ──────────────────────────────")
    print(f"Beat array length : {len(beat)} samples")
    print(f"Pre/Post window   : {pre_samples}/{post_samples} samples")
    print(f"R sample (anchor) : {fid['r_sample']}")
    print(f"S nadir sample    : {fid['s_nadir_sample']}")
    print(f"J point sample    : {fid['j_point_sample']}")
    print(f"J confidence      : {fid['confidence'].get('j_point','?')}")
    print(f"T start sample    : {fid['t_start_sample']}")
    print(f"T end sample      : {fid['t_end_sample']}")
    print(f"T window length   : {fid['t_end_sample'] - fid['t_start_sample']} samples")
    print(f"Baseline mV       : {fid['baseline_mv']:.4f}")

    # Check T-wave content
    t_start = fid['t_start_sample']
    t_end   = fid['t_end_sample']

    if t_end > t_start:
        t_seg = beat[:, v1_idx][t_start:t_end] - fid['baseline_mv']
        print(f"\n── T-wave segment (V1) ─────────────────────────────────")
        print(f"T segment values  : min={t_seg.min():.4f}, max={t_seg.max():.4f}, mean={t_seg.mean():.4f}")
        print(f"T inversion?      : mean < -0.05 = {t_seg.mean() < -0.05}")
        print(f"Raw T values      : {np.round(t_seg[:10], 4)}")
    else:
        print(f"\nT-WAVE WINDOW EMPTY: t_start={t_start} >= t_end={t_end}")
        print(f"Beat array length={len(beat)} — window falls outside beat!")

        # Diagnose why
        rr_samples = int(median_rr * record['fs'] / 1000)
        t_start_expected = fid['r_sample'] + int(CFG.preprocessing.t_wave_start_fraction * rr_samples)
        t_end_expected   = fid['r_sample'] + int(CFG.preprocessing.t_wave_end_fraction * rr_samples)
        print(f"\nExpected T start  : {t_start_expected} samples")
        print(f"Expected T end    : {t_end_expected} samples")
        print(f"Beat array ends at: {len(beat)-1} samples")
        print(f"post_window (ms)  : {CFG.preprocessing.beat_window_post_ms}")
        print(f"\nFIX NEEDED: beat_window_post_ms too short for T-wave at HR={60000/median_rr:.0f} BPM")


if __name__ == "__main__":
    # Test on one Brugada subject and one Normal
    debug_one_subject("188981")   # First subject in dataset
    debug_one_subject("251972")   # Second subject