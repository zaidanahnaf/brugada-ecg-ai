# src/feature_extractors/signal_quality.py

import numpy as np
from scipy.signal import welch
from typing import Dict
from src.config.__init__ import CFG


def extract_signal_quality_features(
    signal: np.ndarray,        # full record (n_samples, n_leads)
    lead_names: list,
    rr_ms: np.ndarray,
    fs: int = CFG.data.fs
) -> Dict[str, float]:
    """
    Extract basic signal quality and global statistics.
    Safe for all leads — no fiducial detection required.
    """
    features = {}

    for i, lead in enumerate(lead_names):
        sig = signal[:, i]
        prefix = f"sq_{lead}"

        features[f"{prefix}_mean"] = float(np.mean(sig))
        features[f"{prefix}_std"] = float(np.std(sig))
        features[f"{prefix}_rms"] = float(np.sqrt(np.mean(sig**2)))
        features[f"{prefix}_min"] = float(np.min(sig))
        features[f"{prefix}_max"] = float(np.max(sig))
        features[f"{prefix}_range"] = float(np.max(sig) - np.min(sig))

        # Noise estimate: energy in high-frequency band
        freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
        noise_mask = (freqs >= CFG.feature.noise_band_low_hz) & (freqs <= CFG.feature.noise_band_high_hz)
        signal_mask = (freqs >= 1.0) & (freqs <= 30.0)

        noise_energy = float(np.trapezoid(psd[noise_mask], freqs[noise_mask])) \
            if noise_mask.sum() > 0 else 0.0
        signal_energy = float(np.trapezoid(psd[signal_mask], freqs[signal_mask])) \
            if signal_mask.sum() > 0 else 1e-9

        features[f"{prefix}_noise_energy"] = noise_energy
        features[f"{prefix}_snr_proxy"] = signal_energy / max(noise_energy, 1e-9)

        # Baseline drift: energy below 0.5 Hz
        drift_mask = freqs <= CFG.feature.baseline_drift_band_hz
        features[f"{prefix}_baseline_drift"] = float(
            np.trapezoid(psd[drift_mask], freqs[drift_mask])
        ) if drift_mask.sum() > 1 else 0.0

    # ── Heart Rate and RR Statistics ──────────────────────────────
    if len(rr_ms) >= 2:
        features['hr_estimate_bpm'] = float(60000 / np.mean(rr_ms))
        features['rr_mean_ms'] = float(np.mean(rr_ms))
        features['rr_std_ms'] = float(np.std(rr_ms))
        features['rr_cv'] = float(np.std(rr_ms) / np.mean(rr_ms))
    else:
        features['hr_estimate_bpm'] = np.nan
        features['rr_mean_ms'] = np.nan
        features['rr_std_ms'] = np.nan
        features['rr_cv'] = np.nan

    return features