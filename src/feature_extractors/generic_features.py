# src/feature_extractors/generic_features.py — needs implementation
def extract_wavelet_features(beat, lead_idx, fs):
    import pywt
    sig = beat[:, lead_idx]
    coeffs = pywt.wavedec(sig, 'db4', level=5)
    return {f'wavelet_energy_L{i}': float(np.sum(c**2))
            for i, c in enumerate(coeffs)}

def extract_derivative_features(beat, lead_idx):
    sig = beat[:, lead_idx]
    d1 = np.diff(sig)
    return {
        'deriv_mean': float(np.mean(d1)),
        'deriv_std': float(np.std(d1)),
        'deriv_max': float(np.max(np.abs(d1)))
    }