from dataclasses import dataclass, field
from typing import List

@dataclass
class PreprocessingConfig:

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
    qrs_end_search_start_ms: float = 20.0   # After S-nadir
    qrs_end_search_end_ms: float = 80.0     # After S-nadir
    qrs_end_method: str = 'min_deriv'       # 'min_deriv' | 'threshold_crossing'


    # ST window (relative to J-point)
    st_offsets_ms: List[float] = field(default_factory=lambda: [20, 40, 60, 80])
    st_window_end_ms: float = 120.0

    # T-wave window (relative to R-peak, HR-adjusted fallback)
    t_wave_start_fraction: float = 0.55         # of RR interval
    t_wave_end_fraction: float = 0.85           # of RR interval
    t_wave_fallback_start_ms: float = 180.0
    t_wave_fallback_end_ms: float = 420.0

    # Baseline reference window (relative to R-peak, pre-QRS)
    baseline_ref_start_ms: float = -180.0
    baseline_ref_end_ms: float = -80.0      # PR segment proxy