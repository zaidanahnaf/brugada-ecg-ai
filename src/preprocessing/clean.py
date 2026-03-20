import numpy as np
from src.preprocessing.dwt_denoise import dwt_denoise

def clean_ecg(raw_signals):
    cleaned = np.zeros_like(raw_signals)

    for i in range(raw_signals.shape[1]):
        cleaned[:, i] = dwt_denoise(raw_signals[:, i]) # Can combine other methods here in the future

    return cleaned