from dataclasses import dataclass, field
from typing import List

@dataclass
class FeatureConfig:
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
    features_dir: str = "outputs/features"
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