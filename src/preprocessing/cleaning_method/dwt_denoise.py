import pywt
import numpy as np

def dwt_denoise(signal, wavelet='bior6.8', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    coeffs_thresh = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]
    ]

    return pywt.waverec(coeffs_thresh, wavelet)