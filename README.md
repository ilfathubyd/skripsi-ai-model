# Skripsi AI Model: TCNN & MediaPipe Pose

## Deskripsi
Repositori ini berisi *pipeline* pembelajaran mesin (Machine Learning) untuk penelitian skripsi "Analisis dan Evaluasi Gerakan Olahraga Berbasis Pose Estimation Menggunakan Temporal Convolutional Neural Network (TCNN)". Fokus repositori ini adalah pada pemrosesan data (dataset sekunder/primer), ekstraksi *keypoints*, dan pelatihan model TCNN.

## Fitur Utama
1. **Ekstraksi Keypoints:** Skrip Python untuk memproses video olahraga dan mengekstraksi 33 *keypoints* tubuh menggunakan MediaPipe Pose.
2. **Pemrosesan Data Time-Series:** Mengubah data koordinat spasial (X, Y, Z, Visibility) menjadi format *time-series* untuk *training* dan *testing*.
3. **Model TCNN:** Arsitektur *Temporal Convolutional Neural Network* untuk mengklasifikasi gerakan (contoh: Benar, Salah Postur, Salah Tempo).
4. **Konversi Model:** Skrip untuk mengonversi model `.h5`/`.pb` yang telah dilatih menjadi format TensorFlow Lite (`.tflite`) dengan *quantization* untuk perangkat *mobile*.

## Persyaratan Sistem (Requirements)
- Python 3.8+
- TensorFlow / Keras
- OpenCV (`opencv-python`)
- MediaPipe
- Pandas & NumPy

## Struktur Repositori
- `/dataset/` : Folder untuk menyimpan data mentah (video) dan data terstruktur (CSV/JSON).
- `/scripts/` : Skrip pra-pemrosesan (ekstraksi MediaPipe).
- `/notebooks/` : Jupyter/Colab notebook untuk eksperimen *training* dan evaluasi TCNN.
- `/models/` : Folder keluaran model yang sudah dilatih (termasuk versi `.tflite`).

## Cara Penggunaan
1. Siapkan dataset pada folder `dataset/raw/`.
2. Jalankan `extract_pose.py` (di dalam folder scripts) untuk mengonversi video menjadi `dataset/processed/time_series.csv`.
3. Buka `train_tcnn.ipynb` untuk melatih model berdasarkan data *time-series* tersebut.
4. Jalankan `convert_tflite.py` untuk mendapatkan model yang siap diimplementasikan ke aplikasi *mobile*.
