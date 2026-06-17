# Pipeline Processing

## 1. Ekstraksi Keypoints (`1_preprocessing_pipeline.py`)

### A. Ekstraksi Pose dengan MediaPipe

```python
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    ...
)
```

Setiap frame video diproses oleh MediaPipe Pose yang mendeteksi **33 landmark tubuh**.

Namun tidak semua landmark digunakan. Hanya **12 joint** yang relevan untuk gerakan olahraga badan atas dan badan bawah.

```python
TARGET_INDICES = [
    11,12,13,14,15,16,
    23,24,25,26,27,28
]
```

**Joint yang digunakan:**

* Bahu kiri & kanan
* Siku kiri & kanan
* Pergelangan tangan kiri & kanan
* Pinggul kiri & kanan
* Lutut kiri & kanan
* Pergelangan kaki kiri & kanan

### Mengapa hanya 12 dari 33 landmark?

Karena gerakan seperti:

* Squat
* Lunge
* Arm Raise

dapat dibedakan secara efektif menggunakan joint tersebut.

Keuntungan:

* Mengurangi noise
* Mengurangi dimensi fitur
* Mempercepat proses training

---

### B. Normalisasi Hip-Centered

```python
mid_hip_x = (left_hip.x + right_hip.x) / 2.0
norm_x = lm.x - mid_hip_x
```

Koordinat MediaPipe bersifat absolut terhadap posisi kamera.

Masalah:

* Orang yang lebih dekat ke kamera menghasilkan koordinat berbeda.
* Orang yang lebih jauh menghasilkan koordinat berbeda.

Solusi:

Menggunakan titik tengah pinggul sebagai anchor.

Sehingga koordinat menjadi relatif terhadap tubuh pengguna sendiri.

Keuntungan:

* Invariant terhadap posisi kamera
* Invariant terhadap ukuran tubuh
* Invariant terhadap jarak pengguna dari kamera

---

### C. Zero Imputation untuk Sendi Tidak Terlihat

```python
if lm.visibility < VISIBILITY_THRESHOLD:
    norm_x, norm_y, norm_z = 0.0, 0.0, 0.0
```

Jika sebuah joint tidak terlihat atau tidak terdeteksi, nilainya diisi:

```text
(0, 0, 0)
```

Alasan:

1. Dalam ruang hip-centered, `(0,0,0)` merupakan posisi netral.
2. Nilai visibility tetap disimpan sehingga model mengetahui tingkat kepercayaan deteksi joint tersebut.

---

### D. Last-Good-Frame Fallback

```python
else:
    frames_data.append(last_good_frame)
```

Jika suatu frame gagal mendeteksi pose sama sekali karena:

* Motion blur
* Oklusi
* Kesalahan deteksi

maka digunakan frame valid terakhir.

Tujuan:

* Menjaga panjang sequence tetap konsisten.
* Menghindari kehilangan data video.

---

### E. Uniform Temporal Resampling (60 Frame)

```python
x_old = np.linspace(0, 1, original_frames)
x_new = np.linspace(0, 1, TARGET_FRAMES)

interpolator = interp1d(
    x_old,
    data_array,
    axis=0,
    kind='linear'
)
```

Durasi video berbeda-beda:

* 2 detik
* 3 detik
* 5 detik
* dan seterusnya

Sedangkan TCNN membutuhkan panjang input yang tetap.

Semua video di-resample menjadi:

```text
TARGET_FRAMES = 60
```

#### Dampak Resampling

* Video < 60 frame → di-stretch (slow motion)
* Video > 60 frame → dikompresi menjadi 60 frame

---

### Output Akhir Preprocessing

File `.npy`

```text
Shape:
(60, 48)
```

Perhitungan fitur:

```text
60 frame
12 joint
4 nilai per joint:
- x
- y
- z
- visibility
```

```text
60 × (12 × 4)
= 60 × 48
```

---

# 2. Split Data (`2_train_model.py`)

## A. Parsing Subject dari Nama File

```python
SUBJECT_PATTERN = re.compile(
    r'_S(\d+)_',
    re.IGNORECASE
)
```

Contoh:

```text
squat_S101_R3.npy
```

Menghasilkan:

```text
subject_id = 101
```

---

## B. Subject-Aware Split

```python
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)
```

### Tujuan

Mencegah data dari orang yang sama muncul pada:

* Training set
* Testing set

secara bersamaan.

Contoh yang dihindari:

```text
Training:
Orang A - Repetisi 1

Testing:
Orang A - Repetisi 2
```

Karena dapat menyebabkan data leakage.

---

## C. One-Hot Encoding

```python
y_train_oh = np.eye(num_classes)[y_train]
```

Contoh:

```text
0 = Squat
1 = Lunge
```

Menjadi:

```text
Squat -> [1, 0]
Lunge -> [0, 1]
```

Karena loss function yang digunakan:

```python
categorical_crossentropy
```

---

### Output Akhir Dataset

```text
dataset_siap_train/

├── X_train.npy
├── y_train.npy
├── X_test.npy
└── y_test.npy
```

Shape:

```text
X_train : (N_train, 60, 48)
y_train : (N_train, 2)

X_test  : (N_test, 60, 48)
y_test  : (N_test, 2)
```

---

# 3. Arsitektur TCNN

## Mengapa Menggunakan Conv1D?

### Keunggulan Conv1D dibanding LSTM/RNN

1. Memindai pola temporal lokal.

```text
kernel_size = 3
```

Artinya model melihat:

```text
Frame t-1
Frame t
Frame t+1
```

Cocok untuk mendeteksi fase gerakan:

* Lutut mulai turun
* Lutut naik
* Lengan mulai terangkat

2. Lebih efisien secara komputasi.

3. Dapat diproses secara paralel.

4. Tidak mengalami vanishing gradient seperti RNN.

5. Pada dataset gerakan berukuran kecil-menengah sering menghasilkan performa lebih baik daripada LSTM.

---

## Mengapa Menggunakan GlobalAveragePooling1D?

Tidak menggunakan:

```python
Flatten()
```

Melainkan:

```python
GlobalAveragePooling1D()
```

Keuntungan:

* Lebih ringan
* Mengurangi overfitting
* Tidak bergantung pada posisi frame tertentu
* Dapat bekerja pada panjang sequence yang lebih fleksibel saat inferensi

---

## Mengapa Dropout = 0.4?

```python
Dropout(0.4)
```

Karena dataset olahraga umumnya tidak terlalu besar.

Dropout 40% membantu:

* Mengurangi overfitting
* Meningkatkan generalisasi model

---

## Struktur TCNN

```text
Input (60,48)
        │
        ▼
Conv1D(64, kernel=3) + ReLU
        │
BatchNorm
        │
Conv1D(128, kernel=3) + ReLU
        │
BatchNorm
        │
Conv1D(64, kernel=3) + ReLU
        │
BatchNorm
        │
GlobalAveragePooling1D
        │
Dropout(0.4)
        │
Dense(2) + Softmax
        │
        ▼
Output:
P(Squat), P(Lunge)
```

---

# 4. Callbacks Training

| Callback                       | Fungsi                                                |
| ------------------------------ | ----------------------------------------------------- |
| EarlyStopping(patience=20)     | Stop jika validation loss tidak turun selama 20 epoch |
| ReduceLROnPlateau(patience=10) | Menurunkan learning rate ketika training stagnan      |
| ModelCheckpoint                | Menyimpan model terbaik berdasarkan validation loss   |
| CSVLogger                      | Menyimpan seluruh metrik training ke file CSV         |

---

## Class Weights

```python
class_weights_arr = compute_class_weight(
    "balanced",
    classes=...,
    y=y_int
)
```

Jika jumlah data:

```text
Squat = 1000
Lunge = 300
```

Model akan cenderung bias ke kelas mayoritas.

`compute_class_weight("balanced")` memberikan bobot lebih tinggi pada kelas minoritas sehingga kontribusi loss menjadi lebih seimbang selama proses training.
