WORK PACKAGE B — INPUT STRATEGY (UPDATED)

Karena panitia mengizinkan segmentasi beat:
Person 2 punya dua opsi:

Option A — Recording-level (simpler):
Input: full 12-second recording (1200 samples × 12 leads)
1 subject = 1 training sample
Total: ~360 training samples per fold
Risk: sangat sedikit untuk deep learning

Option B — Beat-level (recommended untuk CNN):
Segment each recording into individual beats (~9-14 beats per recording)
1 subject = ~10 training samples
Total: ~3,600 training samples per fold -> lebih feasible untuk CNN
CRITICAL: aggregasi prediksi kembali ke subject-level sebelum evaluasi
Aggregasi: mean(beat_probabilities) per subject, atau max, atau median

Beat segmentation bisa menggunakan hasil Person 3:

- R-peak locations sudah dideteksi di beat_segmentation.py
- Valid beats sudah difilter
- Atau Person 2 bisa deteksi ulang secara independen

Prediksi final HARUS per-subject (1 row per patient di submission)

Di Work Package B, tambahkan:
INPUT STRATEGY — BEAT vs RECORDING LEVEL

Panitia mengizinkan segmentasi beat (konfirmasi 19 Maret).
Prediksi FINAL harus individual-based (1 prediction per subject).

Recommended: Beat-level training

- Setiap recording ~9-14 valid beats -> ~3,600 samples/fold
- Jauh lebih feasible untuk CNN daripada 360 recordings
- Person 3 sudah mendeteksi R-peaks — bisa digunakan langsung
  atau Person 2 deteksi ulang secara independen
- Beat window: 200ms pre-R + 500ms post-R (sama dengan Person 3)
- Aggregasi ke subject-level: mean(beat_probs) ATAU attention-weighted mean

Aggregation strategy (penting untuk hybrid fusion):
subject_prob = mean(beat_probs) ← simple, recommended
atau
subject_prob = max(beat_probs) ← lebih sensitive, lebih FP
atau
subject_prob = attention_weighted_mean(beat_probs) ← learnable
Di bagian akhir sebelum "How to Respond", tambahkan:
COMPETITION JUDGING NOTES

Judges menilai methodology dan reasoning, bukan hanya angka.
Untuk setiap design choice, siapkan justifikasi:

Architecture choice -> kenapa 1D-CNN bukan 2D-CNN atau transformer?
Lead selection -> kenapa V1-V3? (dukung dengan ablation Person 3)
Beat segmentation -> kenapa beat-level? (justifikasi data augmentation)
Loss function -> kenapa weighted BCE? (justifikasi class imbalance)
Threshold -> kenapa Youden index? (justifikasi clinical tradeoff)

Technical report harus menjawab:

1. Apa masalah klinisnya dan kenapa ECG-based approach tepat?
2. Kenapa arsitektur CNN ini dipilih untuk dataset kecil?
3. Bagaimana hybrid fusion meningkatkan performance?
4. Apa limitasi metode dan bagaimana bisa diperbaiki?
