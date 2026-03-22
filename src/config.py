# src/config.py
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class PipelineConfig:

    # ── Dataset ──────────────────────────────────────────────────
    data_dir: str = "data/raw"
    metadata_path: str = "data/raw/brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0/metadata.csv"
    fs: int = 100                          # Sampling frequency (Hz)
    signal_duration_s: float = 12.0
    n_leads: int = 12
    lead_names: List[str] = field(default_factory=lambda: [
        'I','II','III','aVR','aVL','aVF',
        'V1','V2','V3','V4','V5','V6'
    ])
    priority_leads: List[str] = field(default_factory=lambda: ['V1','V2','V3'])
    target_col: str = 'brugada'

    # ── Preprocessing ────────────────────────────────────────────
    apply_baseline_removal: bool = True
    baseline_removal_method: str = 'median_filter'  # or 'wavelet'
    median_filter_kernel_ms: float = 600.0  # ~200-600ms typical
    apply_bandpass: bool = False  # OFF by default: preserve ST morphology
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    apply_notch: bool = False              # 50/60 Hz notch optional

    # ── Beat Segmentation ────────────────────────────────────────
    rpeak_method: str = 'neurokit'         # 'neurokit' | 'pantompkins' | 'wfdb'
    min_rr_ms: float = 400.0              # Max HR ~150 BPM
    max_rr_ms: float = 1500.0             # Min HR ~40 BPM
    rr_stability_threshold: float = 0.15  # ±15% of median RR
    min_valid_beats: int = 3              # Minimum clean beats required
    beat_window_pre_ms: float = 200.0     # Before R-peak
    beat_window_post_ms: float = 500.0    # After R-peak

    # ── Fiducial Detection ───────────────────────────────────────
    # J-point proxy: QRS end estimation
    qrs_end_search_start_ms: float = 20.0  # After S-nadir
    qrs_end_search_end_ms: float = 80.0    # After S-nadir
    qrs_end_method: str = 'min_deriv'      # 'min_deriv' | 'threshold_crossing'

    # ST window (relative to J-point)
    st_offsets_ms: List[float] = field(default_factory=lambda: [20, 40, 60, 80])
    st_window_end_ms: float = 120.0

    # T-wave window (relative to R-peak, HR-adjusted fallback)
    t_wave_start_fraction: float = 0.55   # of RR interval
    t_wave_end_fraction: float = 0.85     # of RR interval
    t_wave_fallback_start_ms: float = 240.0
    t_wave_fallback_end_ms: float = 400.0

    # Baseline reference window (relative to R-peak, pre-QRS)
    baseline_ref_start_ms: float = -180.0
    baseline_ref_end_ms: float = -80.0    # PR segment proxy

    # ── Feature Thresholds ───────────────────────────────────────
    high_takeoff_2mm_mv: float = 0.2
    high_takeoff_1mm_mv: float = 0.1
    t_inversion_threshold_mv: float = -0.05
    covedness_threshold: float = 0.5      # For binary coved flag
    saddleback_threshold: float = 0.5

    # ── Signal Quality ───────────────────────────────────────────
    noise_band_low_hz: float = 35.0
    noise_band_high_hz: float = 49.0
    baseline_drift_band_hz: float = 0.5
    min_snr_proxy: float = 2.0            # Below this → flag as noisy

    # ── Aggregation ──────────────────────────────────────────────
    aggregation_stats: List[str] = field(default_factory=lambda: [
        'mean', 'median', 'std', 'min', 'max'
    ])
    primary_aggregation: str = 'median'   # Primary subject-level stat

    # ── Output ───────────────────────────────────────────────────
    features_dir: str = "features"
    logs_dir: str = "logs"
    random_seed: int = 42
    n_folds: int = 5

    # ── Quality Control Columns ───────────────────────────────────────
    # These are written to feature_matrix.csv but must be handled carefully:
    #   - EXCLUDE from ML feature columns by default (not predictive signals)
    #   - USE as quality gates in error analysis and interpretability
    #   - USE as features in a dedicated 'QC-aware' ablation experiment
    #   - DO NOT use as primary model features (risk of proxy for label)
    quality_control_cols: List[str] = field(default_factory=lambda: [
        'pipeline_status',
        'n_valid_beats',
        'median_rr_ms',
    ])
    # Pattern prefix for QC feature columns (auto-excluded in get_fold_split)
    qc_feature_prefix: str = 'qc_'

CFG = PipelineConfig()