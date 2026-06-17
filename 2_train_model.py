import os
import re
import glob
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# KONFIGURASI
# ==========================================
DATASET_DIR = 'dataset_numpy'
OUTPUT_DIR  = 'dataset_siap_train'
CLASSES     = ['squat', 'lunge']           # Hanya kelas yang punya data

# Pola nama file: GERAKAN_S<no>_R<no>.npy
# Grup split berdasarkan NO SUBJEK agar satu subjek
# tidak bocor ke train dan test sekaligus.
SUBJECT_PATTERN = re.compile(r'_S(\d+)_', re.IGNORECASE)

TEST_SIZE    = 0.20   # 20% subjek masuk test
RANDOM_SEED  = 42


# ==========================================
# FUNGSI LOAD DATASET + EKSTRAK SUBJEK ID
# ==========================================
def load_dataset():
    """
    Membaca semua .npy per kelas, me-return:
      X         : array data keypoints   shape (N, frames, fitur)
      y         : array label integer    shape (N,)
      groups    : array string subjek ID shape (N,)  <- KUNCI anti-leakage
      filenames : list nama file (untuk laporan)
    """
    X, y, groups, filenames = [], [], [], []

    print("Memuat data...")
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"  [SKIP] Folder tidak ditemukan: {class_dir}")
            continue

        npy_files = sorted(glob.glob(os.path.join(class_dir, '*.npy')))
        loaded, skipped = 0, 0
        for fp in npy_files:
            fname = os.path.basename(fp)
            m = SUBJECT_PATTERN.search(fname)
            if not m:
                print(f"  [PERINGATAN] Tidak dapat parse subjek dari '{fname}', file dilewati.")
                skipped += 1
                continue

            subject_id = m.group(1)          # mis. "105"
            data = np.load(fp)               # shape (60, 48)

            X.append(data)
            y.append(label_idx)
            groups.append(subject_id)        # <-- identitas subjek
            filenames.append(fname)
            loaded += 1

        print(f"  Kelas '{class_name}' (label {label_idx}): {loaded} file dimuat"
              + (f", {skipped} dilewati" if skipped else ""))

    return (np.array(X), np.array(y),
            np.array(groups), filenames)


# ==========================================
# FUNGSI SPLIT BERBASIS SUBJEK (anti-leakage)
# ==========================================
def subject_aware_split(X, y, groups):
    """
    Membagi data sehingga SEMUA repetisi milik satu subjek
    berada di sisi yang SAMA (train ATAU test), tidak terpecah.

    Menggunakan GroupShuffleSplit dari scikit-learn.
    """
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
    # groups di sini adalah subjek ID — scikit-learn memastikan
    # tidak ada group yang sama muncul di train dan test.
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return train_idx, test_idx


# ==========================================
# FUNGSI LAPORAN DISTRIBUSI
# ==========================================
def print_split_report(y, groups, filenames, train_idx, test_idx):
    train_subjects = set(groups[train_idx])
    test_subjects  = set(groups[test_idx])
    bocor = train_subjects & test_subjects  # harus kosong!

    print("\n" + "=" * 60)
    print("  LAPORAN SUBJECT-AWARE SPLIT")
    print("=" * 60)
    print(f"  Total sampel   : {len(y)}")
    print(f"  Train sampel   : {len(train_idx)}  ({len(train_idx)/len(y)*100:.1f}%)")
    print(f"  Test  sampel   : {len(test_idx)}   ({len(test_idx)/len(y)*100:.1f}%)")
    print(f"  Subjek di train: {len(train_subjects)}")
    print(f"  Subjek di test : {len(test_subjects)}")

    if bocor:
        print(f"\n  [BAHAYA] Kebocoran subjek terdeteksi: {bocor}")
    else:
        print(f"\n  [OK] ZERO subject leakage — tidak ada subjek yang bocor ke test!")

    print("\n  Distribusi kelas di train:")
    for i, cls in enumerate(CLASSES):
        n = np.sum(y[train_idx] == i)
        print(f"    {cls:<12}: {n} sampel")

    print("\n  Distribusi kelas di test:")
    for i, cls in enumerate(CLASSES):
        n = np.sum(y[test_idx] == i)
        print(f"    {cls:<12}: {n} sampel")

    # Subjek test (untuk transparansi)
    print(f"\n  Subjek yang masuk TEST set: {sorted(test_subjects, key=lambda x: int(x))}")
    print("=" * 60)


# ==========================================
# EKSEKUSI UTAMA
# ==========================================
if __name__ == "__main__":

    # 1. Load semua data
    X_data, y_labels, subject_groups, filenames = load_dataset()

    if len(X_data) == 0:
        print("[ERROR] Tidak ada data yang berhasil dimuat. Periksa folder dataset_numpy.")
        exit(1)

    print(f"\nShape X: {X_data.shape}  -> (sampel, frame, fitur)")
    print(f"Shape y: {y_labels.shape}")
    print(f"Jumlah subjek unik (gabungan semua kelas): "
          f"{len(set(subject_groups))}")

    # 2. Subject-Aware Split
    print("\nMelakukan subject-aware split...")
    train_idx, test_idx = subject_aware_split(X_data, y_labels, subject_groups)

    X_train = X_data[train_idx]
    X_test  = X_data[test_idx]
    y_train = y_labels[train_idx]
    y_test  = y_labels[test_idx]

    # Laporan verifikasi
    print_split_report(y_labels, subject_groups, filenames,
                       train_idx, test_idx)

    # 3. One-Hot Encoding
    num_classes = len(CLASSES)
    y_train_oh  = np.eye(num_classes)[y_train]
    y_test_oh   = np.eye(num_classes)[y_test]

    print(f"\n  y_train one-hot shape : {y_train_oh.shape}")
    print(f"  y_test  one-hot shape : {y_test_oh.shape}")

    # 4. Simpan ke disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train_oh)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'),  X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),  y_test_oh)

    print(f"\n[SELESAI] 4 file disimpan di folder '{OUTPUT_DIR}':")
    print(f"  X_train.npy  {X_train.shape}")
    print(f"  y_train.npy  {y_train_oh.shape}")
    print(f"  X_test.npy   {X_test.shape}")
    print(f"  y_test.npy   {y_test_oh.shape}")
    print("\nData siap untuk training TCNN tanpa kebocoran subjek!")
