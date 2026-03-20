from src.preprocessing.clean import clean_ecg
from src.preprocessing.segmentation_3 import segment_ecg

def preprocess_pipeline(raw_signals, fs):
    cleaned = clean_ecg(raw_signals)
    segments = segment_ecg(cleaned, fs)
    return segments