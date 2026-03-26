🧠 Catatan Eksperimen & Evaluasi Model CNN Brugada ECG
📌 1. Ringkasan Hasil Model
✅ Performa Akhir (5-Fold Cross Validation)
AUROC: 0.9357 ± 0.0263
AUPRC: 0.8443 ± 0.0661
Sensitivity: 0.8539 ± 0.0611
Specificity: 0.9197 ± 0.0528
F1-score (positive): 0.7950 ± 0.0672
Brier Score: 0.1113 ± 0.0396
📊 Interpretasi
Model memiliki kemampuan diskriminasi yang kuat (AUROC > 0.93)
Sensitivity tinggi → cocok untuk screening medis
Specificity tinggi → false positive terkendali
Performa cukup stabil, namun terdapat variasi antar fold
⚠️ 2. Temuan Masalah
❗ A. Error Logging (Runtime Crash)

Error:

ValueError: Unknown format code 'f' for object of type 'str'

Penyebab:

Penggunaan:

locals().get(..., '?')
Return '?' (string) diformat sebagai float (:.4f)

Dampak:

Training selesai dengan baik
Crash hanya terjadi pada tahap reporting/logging
❗ B. Missing Data (File Tidak Ditemukan)

Contoh:

No such file or directory: 1009874.hea

Penyebab:

File ECG tidak tersedia / path tidak valid

Dampak:

Beberapa subject tidak digunakan dalam training
Potensi bias jika tidak ditangani konsisten
❗ C. Data Filtering Issue

Contoh:

TOO_FEW_RPEAKS
TOO_FEW_VALID_BEATS

Dampak:

Data dibuang → ukuran dataset efektif berkurang
Distribusi kelas bisa berubah
❗ D. Variasi Performa Antar Fold

Contoh:

Fold terbaik: AUROC ~0.96
Fold terburuk: AUROC ~0.889

Analisis:

Data heterogen
Kemungkinan noise / subject sulit
Indikasi generalisasi belum optimal
❗ E. Overfitting Lokal

Ciri:

train_loss turun drastis
val_loss naik

Dampak:

Model terlalu “yakin” pada training data
Generalisasi menurun pada fold tertentu
🔍 3. Analisis Arsitektur Model
⚠️ Kelemahan CNN Baseline
Receptive field terbatas → tidak menangkap pola global ECG
Tidak ada mekanisme attention
Tidak memanfaatkan multi-scale feature
Arsitektur terlalu shallow untuk kompleksitas sinyal ECG
🚀 4. Rekomendasi Perbaikan
✅ A. Arsitektur Model

1. Gunakan Residual Network
   Stabilitas training meningkat
   Bisa lebih dalam tanpa degradasi performa
2. Tambahkan Dilated Convolution
   Memperluas receptive field
   Menangkap hubungan antar beat
3. Multi-scale Kernel
   Kernel kecil → detail cepat (QRS)
   Kernel besar → morphology (ST segment)
4. SE Attention Block
   Model fokus pada lead yang relevan
   Meningkatkan interpretasi fitur
5. Global Average Pooling
   Menghindari overfitting dari flatten layer
   ✅ B. Hyperparameter Tuning

Rekomendasi:

learning_rate: 5e-5
batch_size: 8
dropout: 0.3
weight_decay: 5e-4
✅ C. Training Strategy
Early stopping (sudah baik)
Label smoothing
Mixup augmentation
Test Time Augmentation (TTA)
✅ D. Data Handling
Validasi keberadaan file sebelum load
Logging subject yang di-drop
Monitoring distribusi kelas setelah filtering
✅ E. Evaluasi & Reporting
Gunakan threshold global (bukan per fold)
Tambahkan ROC curve & PR curve
Simpan hasil dalam format JSON/CSV konsisten
🧹 5. Manajemen Eksperimen
❗ Yang Harus Dihapus saat Ganti Model
models/
features/
results/
✅ Yang Harus Dipertahankan
data/splits/
⚠️ Opsional
preprocessing/fold_stats/
💡 Best Practice

Gunakan versioning:

models/
cnn_v1_baseline/
cnn_v2_resnet/

features/
cnn_v1/
cnn_v2/
📈 6. Target Peningkatan
Metric Sebelum Target
AUROC 0.9357 0.95 – 0.97
AUPRC 0.84 > 0.87
Stability Sedang Tinggi
🧠 7. Insight Penting
Performa model sudah cukup untuk use-case screening
Bottleneck utama:
arsitektur
stabilitas antar fold
Bukan lagi sekadar tuning hyperparameter
🚀 8. Rekomendasi Lanjutan
🔥 Tahap Berikutnya
CNN + Transformer Hybrid
Stacking model (gunakan embeddings)
Model calibration (Platt / Isotonic)
Paper / publikasi ilmiah
🧾 9. Kesimpulan
Model baseline sudah kuat namun belum optimal
Error yang terjadi bersifat non-kritis (logging)
Upgrade arsitektur memiliki potensi peningkatan signifikan
Pipeline sudah cukup matang untuk dikembangkan ke level lanjut
