def segment_ecg(signal, fs, duration=3):
    window_size = int(duration * fs)

    segments = []
    for i in range(0, signal.shape[0], window_size):
        seg = signal[i:i + window_size]
        if seg.shape[0] == window_size:
            segments.append(seg.astype("float32"))

    return segments