# TCNN Sport Movement Classification — Pipeline Step 6 hingga 8

> **Konteks untuk AI Agent:** Proyek ini membangun model klasifikasi gerakan olahraga menggunakan Temporal Convolutional Neural Network (TCNN). Step 1–5 (ekstraksi keypoint, zero-imputation, normalisasi, resampling, one-hot encoding & stratified split) **sudah selesai**. File ini mendefinisikan pekerjaan yang harus kamu lakukan mulai Step 6 hingga Step 8.

---

## Asumsi Wajib Sebelum Memulai

Sebelum menulis kode apa pun, verifikasi dulu kondisi berikut:

| Variabel | Dimensi yang Diharapkan | Keterangan |
|---|---|---|
| `X_train` | `(N_train, 60, 48)` | 60 frame, 48 fitur (16 sendi × 3 koordinat XYZ) |
| `X_test` | `(N_test, 60, 48)` | Format sama dengan train |
| `y_train` | `(N_train, num_classes)` | One-hot encoded |
| `y_test` | `(N_test, num_classes)` | One-hot encoded |
| `class_names` | `list[str]` | Contoh: `["Squat", "Lunge", "Arm Raise"]` |

Jika dimensi tidak sesuai, **hentikan eksekusi dan laporkan** error dimensi secara eksplisit sebelum melanjutkan.

> **Best Practice Note:** 48 fitur diasumsikan dari **16 landmark** MediaPipe × 3 (X, Y, Z). Jika kamu menggunakan **33 landmark** penuh MediaPipe Pose, dimensi fiturnya adalah 33 × 3 = 99, sehingga shape menjadi `(N, 60, 99)`. Sesuaikan konstanta `INPUT_FEATURES` di konfigurasi di bawah.

---

## Konfigurasi Global

Definisikan semua hyperparameter di satu tempat (atas file/notebook) agar mudah diubah tanpa menelusuri seluruh kode:

```python
# ============================================================
# KONFIGURASI GLOBAL — ubah di sini saja
# ============================================================
SEQUENCE_LENGTH  = 60       # frame per sampel (hasil resampling)
INPUT_FEATURES   = 48       # jumlah fitur per frame (sesuaikan jika perlu)
NUM_CLASSES      = len(class_names)

# Arsitektur
FILTERS_1        = 64
FILTERS_2        = 128
FILTERS_3        = 64
KERNEL_SIZE      = 3
DROPOUT_RATE     = 0.4

# Training
BATCH_SIZE       = 32
MAX_EPOCHS       = 200      # EarlyStopping akan menghentikan lebih awal
LEARNING_RATE    = 1e-3
PATIENCE_ES      = 20       # EarlyStopping patience
PATIENCE_LR      = 10       # ReduceLROnPlateau patience
LR_FACTOR        = 0.5      # faktor pengurangan LR
MIN_LR           = 1e-6

# Path output
MODEL_DIR        = "saved_models"
MODEL_PATH       = f"{MODEL_DIR}/best_tcnn_model.keras"
RESULTS_DIR      = "results"
```

---

## Step 6 — Pembangunan Arsitektur TCNN Anti-Overfitting

### Tujuan
Membangun model yang mampu menangkap **pola temporal** dalam deret keypoint tanpa overfitting ke data latih yang terbatas.

### Kode Implementasi

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_tcnn_model(sequence_length, input_features, num_classes,
                     filters_1=FILTERS_1, filters_2=FILTERS_2, filters_3=FILTERS_3,
                     kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE,
                     learning_rate=LEARNING_RATE):
    """
    Arsitektur TCNN untuk klasifikasi gerakan olahraga.

    Alasan desain:
    - BatchNormalization setelah setiap Conv1D: menstabilkan distribusi
      aktivasi per batch, mempercepat konvergensi, dan berperan sebagai
      regularizer ringan — mengurangi ketergantungan pada Dropout tunggal.
    - Conv1D dengan kernel_size=3: menangkap hubungan lokal antar frame
      berurutan (konteks 3 frame). Stacking 3 lapis Conv1D memberikan
      receptive field efektif 7 frame tanpa parameter yang membengkak.
    - GlobalAveragePooling1D (bukan Flatten): meringkas seluruh timeline
      menjadi satu vektor fitur dengan cara rata-rata — jauh lebih robust
      terhadap pergeseran temporal dibanding Flatten, dan drastis mengurangi
      jumlah parameter.
    - Dropout(0.4) sebelum Dense output: memaksa model tidak bergantung
      pada neuron tertentu, menekan overfitting secara eksplisit.
    - Aktivasi output Softmax: menghasilkan distribusi probabilitas antar
      kelas yang berjumlah 1.0 — sesuai untuk klasifikasi multi-kelas.
    """
    inputs = keras.Input(shape=(sequence_length, input_features), name="keypoint_sequence")

    # Blok 1
    x = layers.Conv1D(filters_1, kernel_size, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)

    # Blok 2
    x = layers.Conv1D(filters_2, kernel_size, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    # Blok 3
    x = layers.Conv1D(filters_3, kernel_size, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)

    # Pooling temporal → vektor fitur global
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # Regularisasi akhir
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="TCNN_SportMovement")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Inisialisasi model
model = build_tcnn_model(SEQUENCE_LENGTH, INPUT_FEATURES, NUM_CLASSES)
model.summary()

# Verifikasi dimensi output sebelum training
assert model.output_shape == (None, NUM_CLASSES), \
    f"Output shape salah: {model.output_shape}, harusnya (None, {NUM_CLASSES})"
print("✅ Arsitektur model valid.")
```

### Best Practice Tambahan — Pertimbangkan L2 Regularization

Jika dataset sangat kecil (< 200 sampel total), tambahkan L2 regularization pada layer Conv1D:

```python
# Ganti baris Conv1D dengan:
from tensorflow.keras import regularizers

x = layers.Conv1D(filters_1, kernel_size, padding="same", activation="relu",
                  kernel_regularizer=regularizers.l2(1e-4), name="conv1")(inputs)
```

**Alasan:** Dropout bekerja dengan mematikan neuron secara acak, sementara L2 membatasi besaran bobot. Keduanya bersifat komplementer — L2 khususnya efektif ketika data latih sangat sedikit karena menghukum kompleksitas model secara langsung di fungsi loss.

---

## Step 7 — Pelatihan Terkawal dengan Callbacks

### Tujuan
Melatih model secara efisien dengan mekanisme pengawas otomatis sehingga tidak perlu memonitor training secara manual.

### Kode Implementasi

```python
# ============================================================
# DEFINISI CALLBACKS
# ============================================================

callbacks = [
    # 1. EarlyStopping
    # Monitor: val_loss (bukan val_accuracy) — lebih stabil dan tidak
    # terpengaruh class imbalance kecil. Restore bobot terbaik otomatis.
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE_ES,
        restore_best_weights=True,
        verbose=1
    ),

    # 2. ReduceLROnPlateau
    # Jika val_loss tidak membaik selama PATIENCE_LR epoch, kurangi LR
    # sebesar faktor LR_FACTOR. Ini membantu model "menyempurnakan" bobot
    # di lembah loss yang sudah dekat optimal.
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_FACTOR,
        patience=PATIENCE_LR,
        min_lr=MIN_LR,
        verbose=1
    ),

    # 3. ModelCheckpoint
    # Simpan hanya bobot terbaik (bukan setiap epoch) berdasarkan val_loss.
    # Format .keras lebih direkomendasikan dibanding .h5 di TF 2.x ke atas.
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    # 4. CSVLogger — Best Practice Tambahan
    # Simpan riwayat training ke CSV untuk analisis pasca-training tanpa
    # perlu membuka notebook lagi.
    keras.callbacks.CSVLogger(
        filename=f"{RESULTS_DIR}/training_log.csv",
        append=False
    ),
]


# ============================================================
# PROSES TRAINING
# ============================================================

print(f"Training dimulai — max {MAX_EPOCHS} epoch, batch size {BATCH_SIZE}")
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print(f"\n✅ Training selesai pada epoch {len(history.history['loss'])}")
print(f"   Best val_loss  : {min(history.history['val_loss']):.4f}")
print(f"   Best val_accuracy: {max(history.history['val_accuracy']):.4f}")
```

### Visualisasi Kurva Training

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history["loss"],     label="Train Loss", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
axes[0].set_title("Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Categorical Crossentropy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2, linestyle="--")
axes[1].set_title("Accuracy per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150)
plt.show()
print("✅ Kurva training disimpan.")
```

**Apa yang harus kamu perhatikan di kurva ini:**
- Jika `val_loss` naik sementara `train_loss` terus turun → **overfitting**, pertimbangkan menaikkan `DROPOUT_RATE` atau menambah L2.
- Jika keduanya turun lambat dan sejajar → **underfitting**, coba tambah `FILTERS_2` atau tambah satu blok Conv1D.

---

## Step 8 — Evaluasi Metrik Klasifikasi

### Tujuan
Mengukur performa model secara komprehensif menggunakan metrik yang valid untuk laporan akademis/riset.

### Kode Implementasi

```python
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# LOAD MODEL TERBAIK (pastikan pakai bobot yang disimpan checkpoint)
# ============================================================
best_model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model terbaik dimuat dari: {MODEL_PATH}")


# ============================================================
# PREDIKSI
# ============================================================
y_pred_prob = best_model.predict(X_test, verbose=0)   # probabilitas per kelas
y_pred      = np.argmax(y_pred_prob, axis=1)           # kelas prediksi
y_true      = np.argmax(y_test, axis=1)                # kelas ground truth


# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Confusion Matrix — TCNN Sport Movement Classifier", fontsize=13)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
plt.show()
print("✅ Confusion matrix disimpan.")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================
report_str  = classification_report(y_true, y_pred, target_names=class_names, digits=4)
report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                    digits=4, output_dict=True)

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(report_str)

# Simpan ke file teks dan JSON untuk dokumentasi
with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
    f.write(report_str)

with open(f"{RESULTS_DIR}/classification_report.json", "w") as f:
    json.dump(report_dict, f, indent=2)

print(f"✅ Classification report disimpan ke {RESULTS_DIR}/")


# ============================================================
# RINGKASAN AKHIR — PRINT METRIK UTAMA
# ============================================================
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print("\n" + "="*60)
print("RINGKASAN PERFORMA MODEL AKHIR")
print("="*60)
print(f"  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Macro F1      : {report_dict['macro avg']['f1-score']:.4f}")
print(f"  Weighted F1   : {report_dict['weighted avg']['f1-score']:.4f}")
print("="*60)
```

### Panduan Interpretasi Metrik untuk Laporan

| Metrik | Penjelasan Singkat | Kapan Digunakan |
|---|---|---|
| **Precision** | Dari semua prediksi kelas X, berapa % yang benar? | Penting jika false positive mahal |
| **Recall** | Dari semua sampel kelas X asli, berapa % yang terdeteksi? | Penting jika false negative mahal |
| **F1-Score** | Harmonic mean Precision & Recall — metrik paling seimbang | Standar untuk laporan akademis |
| **Macro avg** | Rata-rata F1 per kelas tanpa memperhatikan frekuensi | Gunakan jika kelas seimbang |
| **Weighted avg** | Rata-rata F1 per kelas dengan bobot frekuensi | Gunakan jika kelas tidak seimbang |

---

## Best Practice Tambahan yang Direkomendasikan

### 1. Class Weight untuk Dataset Tidak Seimbang

Jika jumlah sampel per kelas tidak seimbang setelah split, tambahkan `class_weight` ke training:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weight_dict = dict(enumerate(class_weights_arr))
print("Class weights:", class_weight_dict)

# Tambahkan ke model.fit():
history = model.fit(
    ...,
    class_weight=class_weight_dict,  # ← tambahkan ini
)
```

**Alasan:** Stratified split menjaga proporsi, tetapi jika proporsi kelas aslinya memang timpang (misal 70% Squat, 15% Lunge, 15% Arm Raise), model cenderung bias ke kelas mayoritas. Class weight memaksa model memberi perhatian lebih ke kelas minoritas.

### 2. Seed Global untuk Reproducibility

Tambahkan ini di baris paling atas sebelum import apapun:

```python
import os, random, numpy as np, tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

**Alasan:** Tanpa seed tetap, hasil training akan berbeda setiap run karena inisialisasi bobot acak dan operasi paralel GPU/CPU. Reproducibility adalah syarat minimum untuk riset yang valid.

### 3. Export Model ke Format TFLite (Deployment)

Jika model akan digunakan di mobile/edge device:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization
tflite_model = converter.convert()

with open(f"{MODEL_DIR}/tcnn_sport.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Model TFLite disimpan.")
```

---

## Checklist Validasi Akhir untuk AI Agent

Setelah menyelesaikan step 6–8, verifikasi semua item berikut:

- [ ] `model.summary()` menampilkan output shape `(None, NUM_CLASSES)` di layer terakhir
- [ ] Tidak ada error dimensi saat `model.fit()` dipanggil
- [ ] File `best_tcnn_model.keras` tersimpan di `saved_models/`
- [ ] File `training_log.csv` tersimpan di `results/`
- [ ] Kurva training menunjukkan `val_loss` yang konvergen (tidak hanya `train_loss`)
- [ ] Confusion matrix menampilkan semua `NUM_CLASSES` kelas
- [ ] Classification report berisi kolom `precision`, `recall`, `f1-score` untuk setiap kelas
- [ ] Tidak ada kelas dengan nilai `support = 0` di classification report
- [ ] Weighted F1-Score dilaporkan secara eksplisit di output akhir

---

## Pertanyaan / Hal yang Perlu Dikonfirmasi

Sebelum AI Agent memulai eksekusi, informasikan jika ada ketidakjelasan berikut:

1. **Jumlah landmark yang dipakai:** Apakah 16 landmark (fitur = 48) atau 33 landmark penuh MediaPipe (fitur = 99)?
2. **Jumlah kelas:** Apakah hanya 3 kelas ("Squat", "Lunge", "Arm Raise") atau ada tambahan?
3. **Environment:** Apakah training di GPU lokal, Google Colab, atau server? (Mempengaruhi `BATCH_SIZE` optimal)
4. **Ukuran dataset:** Berapa total sampel video? (Mempengaruhi keputusan apakah perlu augmentasi data atau tidak)
5. **Format data tersimpan:** Apakah `X_train`, `X_test`, `y_train`, `y_test` sudah tersimpan sebagai file `.npy` atau masih di memori notebook?

---

*Dokumen ini adalah instruksi lengkap untuk AI Agent menyelesaikan pipeline TCNN klasifikasi gerakan olahraga dari Step 6 hingga Step 8. Jalankan secara berurutan dan laporkan hasil setiap step sebelum melanjutkan ke step berikutnya.*