import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from scipy.interpolate import interp1d

# ==========================================
# KONFIGURASI
# ==========================================
INPUT_DIR = 'dataset_raw_video'
OUTPUT_DIR = 'dataset_numpy'
TARGET_FRAMES = 60
VISIBILITY_THRESHOLD = 0.5

# Indeks keypoints Mediapipe yang digunakan
# 11-16 (Bahu-Siku-Tangan), 23-28 (Pinggul-Lutut-Kaki)
TARGET_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

JOINT_NAMES = {
    11: "L.Bahu", 12: "R.Bahu",
    13: "L.Siku", 14: "R.Siku",
    15: "L.Tangan", 16: "R.Tangan",
    23: "L.Pinggul", 24: "R.Pinggul",
    25: "L.Lutut", 26: "R.Lutut",
    27: "L.Kaki", 28: "R.Kaki",
}

# Koneksi antar joint untuk digambar
TARGET_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),  # Bahu kiri - siku - tangan
    (12, 14), (14, 16),              # Bahu kanan - siku - tangan
    (11, 23), (12, 24),              # Bahu ke pinggul
    (23, 24),                        # Pinggul
    (23, 25), (25, 27),              # Kaki kiri
    (24, 26), (26, 28),              # Kaki kanan
]

# Warna (BGR)
COLOR_DETECTED    = (0, 255, 120)    # Hijau neon — sendi terdeteksi
COLOR_ZERO_IMP    = (0, 100, 255)    # Oranye — sendi zero-imputed (visibility rendah)
COLOR_CONNECTION  = (200, 200, 200)  # Abu terang — koneksi
COLOR_NO_DETECT   = (60, 60, 200)    # Biru redup — frame tanpa deteksi
COLOR_HIP_CENTER  = (255, 80, 200)   # Pink — titik anchor pinggul
COLOR_PANEL_BG    = (20, 20, 30)     # Latar panel info
COLOR_TEXT_TITLE  = (255, 220, 80)   # Kuning — judul
COLOR_TEXT_OK     = (100, 255, 150)  # Hijau — status OK
COLOR_TEXT_WARN   = (60, 140, 255)   # Oranye — peringatan
COLOR_TEXT_INFO   = (200, 200, 220)  # Abu — info biasa

# Inisialisasi Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# FUNGSI GAMBAR PANEL INFO (SAMPING KANAN)
# ==========================================
def draw_info_panel(panel, frame_idx, total_frames, class_name, filename,
                    detected, joint_status_list):
    h, w = panel.shape[:2]
    panel[:] = COLOR_PANEL_BG  # Bersihkan panel

    y = 18
    dy = 22  # Jarak antar baris

    def put(text, color, bold=False, size=0.5):
        nonlocal y
        thickness = 2 if bold else 1
        cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)
        y += dy

    put("== VISUAL PREPROCESSING ==", COLOR_TEXT_TITLE, bold=True, size=0.45)
    put(f"Kelas  : {class_name}", COLOR_TEXT_INFO, size=0.42)
    # Potong nama file jika terlalu panjang
    fname_disp = filename if len(filename) <= 22 else filename[:19] + "..."
    put(f"File   : {fname_disp}", COLOR_TEXT_INFO, size=0.42)
    y += 4

    # Progress bar
    progress = frame_idx / max(total_frames - 1, 1)
    bar_w = w - 16
    bar_h = 12
    cv2.rectangle(panel, (8, y), (8 + bar_w, y + bar_h), (60, 60, 80), -1)
    cv2.rectangle(panel, (8, y), (8 + int(bar_w * progress), y + bar_h), (80, 200, 120), -1)
    cv2.rectangle(panel, (8, y), (8 + bar_w, y + bar_h), (120, 120, 140), 1)
    y += bar_h + 6
    put(f"Frame  : {frame_idx + 1} / {total_frames}", COLOR_TEXT_INFO, size=0.42)
    put(f"Target : {TARGET_FRAMES} frames (setelah resample)", COLOR_TEXT_INFO, size=0.42)
    y += 4

    # Status deteksi
    if detected:
        put("[POSE TERDETEKSI]", COLOR_TEXT_OK, bold=True, size=0.44)
    else:
        put("[GAGAL DETEKSI - last frame]", COLOR_TEXT_WARN, bold=True, size=0.40)
    y += 4

    put("Status Sendi Aktif:", COLOR_TEXT_TITLE, size=0.43)

    # Tampilkan status tiap sendi target
    for idx, vis, imputed in joint_status_list:
        name = JOINT_NAMES.get(idx, str(idx))
        if not detected:
            status_txt = f"  {name:<12} [last frame]"
            color = COLOR_TEXT_WARN
        elif imputed:
            status_txt = f"  {name:<12} ZERO ({vis:.2f})"
            color = COLOR_TEXT_WARN
        else:
            status_txt = f"  {name:<12} OK   ({vis:.2f})"
            color = COLOR_TEXT_OK
        put(status_txt, color, size=0.38)

    # Legend
    y = h - 80
    put("LEGENDA:", COLOR_TEXT_TITLE, size=0.40)
    cv2.circle(panel, (16, y + 4), 5, COLOR_DETECTED, -1)
    put(f"  Sendi terdeteksi (vis>={VISIBILITY_THRESHOLD})", COLOR_TEXT_OK, size=0.38)
    cv2.circle(panel, (16, y + 4), 5, COLOR_ZERO_IMP, -1)
    put(f"  Sendi zero-imputed (vis<{VISIBILITY_THRESHOLD})", COLOR_TEXT_WARN, size=0.38)
    cv2.circle(panel, (16, y + 4), 5, COLOR_HIP_CENTER, -1)
    put(f"  Titik anchor pinggul", COLOR_TEXT_INFO, size=0.38)


# ==========================================
# FUNGSI 1: EKSTRAKSI, NORMALISASI + VISUAL
# ==========================================
def extract_and_normalize_visual(video_path, class_name):
    cap = cv2.VideoCapture(video_path)
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    filename = os.path.basename(video_path)
    frames_data = []
    last_good_frame = np.zeros((len(TARGET_INDICES) * 4,))

    DISPLAY_H = 540
    PANEL_W = 280

    window_name = f"TCNN Preprocessing - {filename}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # --- Resize frame untuk display ---
        h_orig, w_orig = frame.shape[:2]
        scale = DISPLAY_H / h_orig
        display_w = int(w_orig * scale)
        vis_frame = cv2.resize(frame, (display_w, DISPLAY_H))

        detected = False
        joint_status_list = []
        current_frame_data = []
        mid_hip_px = None

        if results.pose_landmarks:
            detected = True
            landmarks = results.pose_landmarks.landmark

            # Hitung Titik Tengah Pinggul (Anchor)
            left_hip  = landmarks[23]
            right_hip = landmarks[24]
            mid_hip_x = (left_hip.x + right_hip.x) / 2.0
            mid_hip_y = (left_hip.y + right_hip.y) / 2.0
            mid_hip_z = (left_hip.z + right_hip.z) / 2.0

            # Gambar titik anchor pinggul
            mid_hip_px = (int(mid_hip_x * display_w), int(mid_hip_y * DISPLAY_H))
            cv2.circle(vis_frame, mid_hip_px, 8, COLOR_HIP_CENTER, -1)
            cv2.circle(vis_frame, mid_hip_px, 8, (255, 255, 255), 1)

            # Gambar koneksi antar sendi target
            idx_to_px = {}
            for idx in TARGET_INDICES:
                lm = landmarks[idx]
                px = (int(lm.x * display_w), int(lm.y * DISPLAY_H))
                idx_to_px[idx] = px

            for (a, b) in TARGET_CONNECTIONS:
                if a in idx_to_px and b in idx_to_px:
                    cv2.line(vis_frame, idx_to_px[a], idx_to_px[b], COLOR_CONNECTION, 2, cv2.LINE_AA)

            # Gambar titik sendi dan kumpulkan data
            for idx in TARGET_INDICES:
                lm = landmarks[idx]
                imputed = lm.visibility < VISIBILITY_THRESHOLD
                joint_status_list.append((idx, lm.visibility, imputed))

                px = idx_to_px[idx]
                color = COLOR_ZERO_IMP if imputed else COLOR_DETECTED
                cv2.circle(vis_frame, px, 7, color, -1)
                cv2.circle(vis_frame, px, 7, (255, 255, 255), 1)

                # Label nama sendi
                name = JOINT_NAMES.get(idx, str(idx))
                cv2.putText(vis_frame, name, (px[0] + 8, px[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

                if imputed:
                    norm_x, norm_y, norm_z = 0.0, 0.0, 0.0
                else:
                    norm_x = lm.x - mid_hip_x
                    norm_y = lm.y - mid_hip_y
                    norm_z = lm.z - mid_hip_z

                current_frame_data.extend([norm_x, norm_y, norm_z, lm.visibility])

            frames_data.append(current_frame_data)
            last_good_frame = current_frame_data

        else:
            # Tidak terdeteksi — pakai last good frame
            frames_data.append(last_good_frame)
            joint_status_list = [(idx, 0.0, True) for idx in TARGET_INDICES]
            # Overlay teks peringatan di frame
            cv2.putText(vis_frame, "NO POSE DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NO_DETECT, 2, cv2.LINE_AA)

        # Overlay nama file & frame di sudut kiri atas
        cv2.rectangle(vis_frame, (0, 0), (display_w, 22), (0, 0, 0), -1)
        cv2.putText(vis_frame, f"{filename}  |  frame {frame_idx+1}/{total_frames_raw}",
                    (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    COLOR_TEXT_TITLE, 1, cv2.LINE_AA)

        # Buat panel info
        info_panel = np.zeros((DISPLAY_H, PANEL_W, 3), dtype=np.uint8)
        draw_info_panel(info_panel, frame_idx, total_frames_raw, class_name,
                        filename, detected, joint_status_list)

        # Gabung frame + panel
        combined = np.hstack([vis_frame, info_panel])
        cv2.imshow(window_name, combined)
        cv2.resizeWindow(window_name, display_w + PANEL_W, DISPLAY_H)

        # Tekan 'q' untuk skip video ini, 's' untuk slow-mo, ESC untuk berhenti total
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"  [SKIP] {filename} - di-skip user.")
            cap.release()
            cv2.destroyWindow(window_name)
            return None
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            print("\n[BERHENTI] User menekan ESC. Program dihentikan.")
            exit(0)

        frame_idx += 1

    cap.release()
    cv2.destroyWindow(window_name)
    return np.array(frames_data) if frames_data else None


# ==========================================
# FUNGSI 2: UNIFORM TEMPORAL RESAMPLING
# ==========================================
def resample_temporal_data(data_array, target_frames=TARGET_FRAMES):
    original_frames = data_array.shape[0]
    if original_frames == 0:
        return None
    if original_frames == target_frames:
        return data_array

    x_old = np.linspace(0, 1, original_frames)
    x_new = np.linspace(0, 1, target_frames)
    interpolator = interp1d(x_old, data_array, axis=0, kind='linear')
    return interpolator(x_new)


# ==========================================
# EKSEKUSI UTAMA
# ==========================================
if __name__ == "__main__":
    classes = ['squat', 'lunge', 'arm_raise']

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("=" * 55)
    print("  TCNN PREPROCESSING — MODE VISUAL MEDIAPIPE")
    print("=" * 55)
    print("  Kontrol keyboard saat window aktif:")
    print("   [Q]   - Skip video ini, lanjut ke video berikutnya")
    print("   [ESC] - Hentikan seluruh proses")
    print("=" * 55)

    for class_name in classes:
        input_class_dir = os.path.join(INPUT_DIR, class_name)
        output_class_dir = os.path.join(OUTPUT_DIR, class_name)

        if not os.path.exists(input_class_dir):
            print(f"\n[WARNING] Folder tidak ditemukan: {input_class_dir}, dilewati.")
            continue

        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        video_files = sorted(glob.glob(os.path.join(input_class_dir, '*.mp4')))
        print(f"\nKelas '{class_name}': {len(video_files)} video ditemukan")

        for i, video_path in enumerate(video_files):
            filename = os.path.basename(video_path)
            name_only = os.path.splitext(filename)[0]
            output_filepath = os.path.join(output_class_dir, f"{name_only}.npy")

            if os.path.exists(output_filepath):
                print(f"  [LEWAT] {filename} (sudah ada .npy)")
                continue

            print(f"  [{i+1}/{len(video_files)}] Memproses: {filename} ...")

            raw_data = extract_and_normalize_visual(video_path, class_name)

            if raw_data is None or len(raw_data) == 0:
                print(f"  [GAGAL] {filename} - tidak ada data pose.")
                continue

            resampled_data = resample_temporal_data(raw_data, TARGET_FRAMES)

            if resampled_data is not None:
                np.save(output_filepath, resampled_data)
                print(f"  [OK]    {filename} -> shape {resampled_data.shape}")
            else:
                print(f"  [GAGAL] {filename}")

    cv2.destroyAllWindows()
    print("\n" + "=" * 55)
    print("  [SELESAI] Data NumPy siap untuk training TCNN!")
    print("=" * 55)
