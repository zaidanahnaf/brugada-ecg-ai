# Catatan Pengembangan Model Hybrid: CNN + Handcrafted Features

1. Temuan Utama & Analisis Masalah

- Kesenjangan Antar Model: Terdeteksi adanya perbedaan keputusan antara model CNN (berbasis visual/sinyal) dan LogReg (berbasis fitur statistik medis).

- Kasus False Positive (FP): Ditemukan kasus (seperti ID 3096254) di mana CNN memberikan probabilitas tinggi karena kemiripan bentuk gelombang, namun fitur statistik menunjukkan angka yang normal.

- Pentingnya Meta-Probabilitas: Probabilitas OOF (Out-of-Fold) dari model tahap pertama (LogReg/CNN) adalah fitur yang sangat kuat karena membawa "ringkasan kecerdasan" dari masing-masing arsitektur.

- Risiko Data Leakage: Penggunaan probabilitas tanpa sinkronisasi fold yang ketat dapat menyebabkan model CatBoost mengalami overfitting (skor tinggi di lokal, anjlok di data baru).

2. Arsitektur Hybrid (Stacking) yang Direkomendasikan
   Model dikembangkan menggunakan strategi Two-Level Stacking:

Level 0 (Base Learners):

CNN: Menangkap pola morfologi kompleks pada Lead V1-V2.

LogReg (Balanced L1): Mengevaluasi 693 fitur handcrafted (seperti interval RR, durasi QRS, elevasi ST) secara linear.

Level 1 (Meta-Learner):

CatBoost: Bertindak sebagai "hakim" yang menggabungkan probabilitas dari Level 0 dengan fitur mentah dari feature_matrix.csv.

3. Praktik Terbaik (Best Practices) & Rekomendasi
   A. Sinkronisasi Fold (Wajib)
   Selalu gunakan file fold_assignment.csv yang konsisten di semua tahap.

Alasan: Memastikan data yang digunakan untuk validasi di Level 1 benar-benar data yang belum pernah dilihat oleh model di Level 0.

Dampak: Menghindari bias dan memastikan estimasi performa (AUC) yang jujur.

B. Konfigurasi Hyperparameter CatBoost
Untuk Meta-Learner, gunakan parameter yang konservatif (Light & Fast) untuk menghindari overfitting pada noise:

depth=3: Agar model tidak membuat aturan yang terlalu kompleks.

iterations=100: Cukup untuk menggabungkan probabilitas yang sudah kuat.

auto_class_weights='Balanced': Wajib digunakan karena kasus Brugada biasanya langka (imbalanced).

C. Penanganan Fitur
Feature Importance: Selalu cek apakah prob_cnn dan prob_logreg masuk dalam deretan fitur paling berpengaruh. Jika tidak, evaluasi kembali kualitas probabilitas tersebut.

Data Cleaning: Lakukan merge berdasarkan patient_id dan pastikan tidak ada data yang hilang (NaN) setelah penggabungan antar model.

4. Rencana Aksi (Action Plan) Selanjutnya
   Generate OOF CNN: Pastikan file OOF CNN memiliki kolom patient_id dan oof_prob_brugada.

Integrasi Meta-Feature: Gabungkan feature_matrix, OOF LogReg, dan OOF CNN ke dalam satu tabel utama.

Training Per Fold: Jalankan training CatBoost secara manual menggunakan iterasi fold_id dari fold_assignment.csv.

Analisis Error: Periksa kembali pasien-pasien yang tetap menjadi False Positive setelah model di-hybrid. Apakah ada fitur medis tambahan yang perlu dibuat?

Kesimpulan: Dengan menggabungkan "Mata" (CNN) dan "Logika" (Handcrafted), model hybrid ini memiliki potensi jauh lebih tinggi untuk menekan angka False Positive dibandingkan hanya mengandalkan satu jenis model saja.
