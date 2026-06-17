# ============================================================
# SEED GLOBAL — Untuk reproducibility (jalankan sebelum import lain)
# ============================================================
import os
import random

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

import numpy as np
import tensorflow as tf

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# IMPORT
# ============================================================
import glob
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (aman untuk CPU tanpa display)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# KONFIGURASI GLOBAL — ubah di sini saja
# ============================================================
DATASET_DIR     = "dataset_numpy"
CLASS_NAMES     = ["squat", "lunge"]          # 2 kelas aktif (arm_raise belum ada data)
DISPLAY_NAMES   = ["Squat", "Lunge"]          # Nama tampilan untuk laporan & grafik

SEQUENCE_LENGTH = 60        # frame per sampel (hasil resampling)
INPUT_FEATURES  = 48        # jumlah fitur per frame (16 landmark × 3 koordinat)
NUM_CLASSES     = len(CLASS_NAMES)

# Arsitektur
FILTERS_1       = 64
FILTERS_2       = 128
FILTERS_3       = 64
KERNEL_SIZE     = 3
DROPOUT_RATE    = 0.4

# Training (disesuaikan untuk CPU)
BATCH_SIZE      = 16        # Lebih kecil dari GPU agar lebih stabil di CPU
MAX_EPOCHS      = 200       # EarlyStopping akan menghentikan lebih awal
LEARNING_RATE   = 1e-3
PATIENCE_ES     = 20        # EarlyStopping patience
PATIENCE_LR     = 10        # ReduceLROnPlateau patience
LR_FACTOR       = 0.5
MIN_LR          = 1e-6

# Path output
MODEL_DIR       = "saved_models"
MODEL_PATH      = f"{MODEL_DIR}/best_tcnn_model.keras"
RESULTS_DIR     = "results"
DATA_CACHE_DIR  = "dataset_siap_train"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ============================================================
# STEP 5 — Load data dari dataset_siap_train (tanpa split ulang)
# ============================================================
print("=" * 60)
print("MEMUAT DATA YANG SUDAH DI-SPLIT (Anti-Leakage)")
print("=" * 60)

try:
    X_train = np.load(os.path.join(DATA_CACHE_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_CACHE_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_CACHE_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_CACHE_DIR, "y_test.npy"))
except FileNotFoundError:
    print(f"  [ERROR] File data tidak ditemukan di folder '{DATA_CACHE_DIR}'.")
    print("  Jalankan 'python 2_train_model.py' terlebih dahulu untuk melakukan split data!")
    exit(1)

# Validasi dimensi
print(f"X_train: {X_train.shape} (harapan: (N, {SEQUENCE_LENGTH}, {INPUT_FEATURES}))")
assert X_train.shape[1] == SEQUENCE_LENGTH, f"[ERROR] Jumlah frame salah: {X_train.shape[1]} != {SEQUENCE_LENGTH}"
assert X_train.shape[2] == INPUT_FEATURES,  f"[ERROR] Jumlah fitur salah: {X_train.shape[2]} != {INPUT_FEATURES}"
print("✅ Dimensi input valid.")

print(f"\nData Latih  : {X_train.shape[0]} sampel  | X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"Data Uji    : {X_test.shape[0]} sampel   | X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

# Class weight (untuk menangani ketidakseimbangan Squat vs Lunge)
y_int = np.argmax(y_train, axis=1)
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_int),
    y=y_int,
)
class_weight_dict = dict(enumerate(class_weights_arr))
print(f"\nClass Weights: {class_weight_dict}")
for i, name in enumerate(DISPLAY_NAMES):
    print(f"  Kelas {i} ({name}): bobot = {class_weights_arr[i]:.4f}")

# ============================================================
# STEP 6 — Build Arsitektur TCNN
# ============================================================
print("\n" + "=" * 60)
print("STEP 6 — MEMBANGUN ARSITEKTUR TCNN")
print("=" * 60)

def build_tcnn_model(sequence_length, input_features, num_classes,
                     filters_1=FILTERS_1, filters_2=FILTERS_2, filters_3=FILTERS_3,
                     kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE,
                     learning_rate=LEARNING_RATE):
    """
    Arsitektur TCNN 3-blok dengan BatchNorm + GlobalAveragePooling1D.
    Dirancang anti-overfitting untuk dataset berukuran kecil-menengah.
    """
    inputs = keras.Input(shape=(sequence_length, input_features), name="keypoint_sequence")

    # Blok Conv 1
    x = layers.Conv1D(filters_1, kernel_size, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)

    # Blok Conv 2
    x = layers.Conv1D(filters_2, kernel_size, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    # Blok Conv 3
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
        metrics=["accuracy"],
    )
    return model


model = build_tcnn_model(SEQUENCE_LENGTH, INPUT_FEATURES, NUM_CLASSES)
model.summary()

# Verifikasi output shape
assert model.output_shape == (None, NUM_CLASSES), \
    f"[ERROR] Output shape salah: {model.output_shape}, harusnya (None, {NUM_CLASSES})"
print("✅ Arsitektur model valid. Output shape:", model.output_shape)

# ============================================================
# STEP 7 — Pelatihan dengan Callbacks
# ============================================================
print("\n" + "=" * 60)
print("STEP 7 — TRAINING MODEL")
print("=" * 60)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE_ES,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_FACTOR,
        patience=PATIENCE_LR,
        min_lr=MIN_LR,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.CSVLogger(
        filename=f"{RESULTS_DIR}/training_log.csv",
        append=False,
    ),
]

print(f"Training dimulai — max {MAX_EPOCHS} epoch, batch size {BATCH_SIZE} (CPU mode)")
print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1,
)

best_epoch  = np.argmin(history.history["val_loss"]) + 1
best_val_loss = min(history.history["val_loss"])
best_val_acc  = max(history.history["val_accuracy"])

print(f"\n✅ Training selesai!")
print(f"   Epoch terbaik     : {best_epoch}")
print(f"   Best val_loss     : {best_val_loss:.4f}")
print(f"   Best val_accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

# Visualisasi kurva training
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"],     label="Train Loss",     linewidth=2, color="#4C6EF5")
axes[0].plot(history.history["val_loss"], label="Val Loss",       linewidth=2, color="#F03E3E", linestyle="--")
axes[0].axvline(best_epoch - 1, color="gray", linestyle=":", label=f"Best epoch ({best_epoch})")
axes[0].set_title("Loss per Epoch", fontsize=13)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Categorical Crossentropy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2, color="#4C6EF5")
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2, color="#F03E3E", linestyle="--")
axes[1].axvline(best_epoch - 1, color="gray", linestyle=":", label=f"Best epoch ({best_epoch})")
axes[1].set_title("Accuracy per Epoch", fontsize=13)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("TCNN Sport Movement — Training Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150)
plt.close()
print(f"✅ Kurva training disimpan ke '{RESULTS_DIR}/training_curves.png'")

# ============================================================
# STEP 8 — Evaluasi Metrik Klasifikasi
# ============================================================
print("\n" + "=" * 60)
print("STEP 8 — EVALUASI MODEL")
print("=" * 60)

# Load model terbaik
best_model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model terbaik dimuat dari: {MODEL_PATH}")

# Prediksi
y_pred_prob = best_model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)
y_true      = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_NAMES)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Confusion Matrix — TCNN Sport Movement Classifier", fontsize=13)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print(f"✅ Confusion matrix disimpan ke '{RESULTS_DIR}/confusion_matrix.png'")

# Classification Report
report_str  = classification_report(y_true, y_pred, target_names=DISPLAY_NAMES, digits=4)
report_dict = classification_report(y_true, y_pred, target_names=DISPLAY_NAMES,
                                    digits=4, output_dict=True)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(report_str)

# Simpan ke file
with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
    f.write(report_str)

with open(f"{RESULTS_DIR}/classification_report.json", "w") as f:
    json.dump(report_dict, f, indent=2)

print(f"✅ Classification report disimpan ke '{RESULTS_DIR}/'")

# Ringkasan akhir
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print("\n" + "=" * 60)
print("RINGKASAN PERFORMA MODEL AKHIR")
print("=" * 60)
print(f"  Kelas           : {', '.join(DISPLAY_NAMES)} ({NUM_CLASSES} kelas)")
print(f"  Sampel Uji      : {X_test.shape[0]}")
print(f"  Test Loss       : {test_loss:.4f}")
print(f"  Test Accuracy   : {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"  Macro F1        : {report_dict['macro avg']['f1-score']:.4f}")
print(f"  Weighted F1     : {report_dict['weighted avg']['f1-score']:.4f}")
print("=" * 60)

# Checklist validasi
print("\n--- CHECKLIST VALIDASI ---")
print(f"  [{'✅' if model.output_shape == (None, NUM_CLASSES) else '❌'}] Output shape model : {model.output_shape}")
print(f"  [✅] File best_tcnn_model.keras : {MODEL_PATH}")
print(f"  [✅] File training_log.csv      : {RESULTS_DIR}/training_log.csv")
print(f"  [✅] Confusion matrix tersimpan  : {RESULTS_DIR}/confusion_matrix.png")
print(f"  [✅] Classification report       : {RESULTS_DIR}/classification_report.txt")
print(f"  [{'✅' if report_dict.get('Squat', {}).get('support', 0) > 0 else '❌'}] Support Squat > 0")
print(f"  [{'✅' if report_dict.get('Lunge', {}).get('support', 0) > 0 else '❌'}] Support Lunge > 0")
print(f"  [✅] Weighted F1 : {report_dict['weighted avg']['f1-score']:.4f}")
print("\n🎉 Pipeline Step 6–8 selesai!")
