# """
# 4_realtime_inference.py
# =======================
# Inferensi REAL-TIME menggunakan webcam + model TCNN yang sudah dilatih.

# Cara kerja:
#   1. Kamera menangkap frame secara terus-menerus.
#   2. Keypoints diekstraksi per frame oleh MediaPipe (identik dengan pipeline training).
#   3. Frame ditampung dalam "sliding window" (WINDOW_FRAMES frame terakhir).
#   4. Setiap N_SKIP_FRAMES, window di-resample jadi 60 frame → disuapkan ke model.
#   5. Prediksi kelas + confidence ditampilkan langsung di atas video feed.

# Kontrol keyboard (saat window aktif):
#   [R]   - Reset buffer / mulai ulang pengumpulan frame
#   [S]   - Screenshot frame saat ini
#   [ESC] - Keluar
# """

# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# from collections import deque
# from scipy.interpolate import interp1d
# from tensorflow import keras
# import time

# # ============================================================
# # KONFIGURASI — sesuaikan jika perlu
# # ============================================================
# MODEL_PATH        = "saved_models/best_tcnn_model.keras"
# CLASS_NAMES       = ["Squat", "Lunge"]          # Urutan HARUS sama dengan saat training
# TARGET_FRAMES     = 60                           # Frames yang diharapkan model
# WINDOW_FRAMES     = 90                           # Ukuran sliding window (lebih besar = lebih stabil)
# N_SKIP_FRAMES     = 5                            # Jalankan prediksi setiap N frame (hemat CPU)
# VISIBILITY_THRESHOLD = 0.5
# TARGET_INDICES    = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# CAMERA_INDEX      = 0                            # Ganti ke 1, 2, dst. jika kamera salah
# SCREENSHOT_DIR    = "screenshots"

# # Koneksi skeleton (sama persis dengan pipeline training)
# TARGET_CONNECTIONS = [
#     (11, 12), (11, 13), (13, 15),
#     (12, 14), (14, 16),
#     (11, 23), (12, 24),
#     (23, 24),
#     (23, 25), (25, 27),
#     (24, 26), (26, 28),
# ]

# JOINT_NAMES = {
#     11: "L.Bahu", 12: "R.Bahu",
#     13: "L.Siku", 14: "R.Siku",
#     15: "L.Tgn", 16: "R.Tgn",
#     23: "L.Ping", 24: "R.Ping",
#     25: "L.Lutut", 26: "R.Lutut",
#     27: "L.Kaki", 28: "R.Kaki",
# }

# # Warna (BGR)
# C_GREEN   = (50, 220, 100)
# C_ORANGE  = (30, 140, 255)
# C_WHITE   = (255, 255, 255)
# C_YELLOW  = (60, 220, 255)
# C_PINK    = (200, 80, 255)
# C_BLUE    = (255, 180, 50)
# C_GRAY    = (160, 160, 160)
# C_BLACK   = (0, 0, 0)
# C_RED     = (60, 60, 220)

# # Warna per kelas prediksi
# CLASS_COLORS = {
#     "Squat": (50, 220, 100),   # Hijau
#     "Lunge": (255, 160, 50),   # Biru muda
# }

# os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# # ============================================================
# # LOAD MODEL
# # ============================================================
# if not os.path.exists(MODEL_PATH):
#     print(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")
#     print("  Jalankan '3_build_and_train.py' terlebih dahulu!")
#     exit(1)

# print(f"Memuat model dari {MODEL_PATH} ...")
# model = keras.models.load_model(MODEL_PATH)
# print(f"  Model input shape : {model.input_shape}")
# print(f"  Model output shape: {model.output_shape}")
# print(f"  Kelas            : {CLASS_NAMES}")


# # ============================================================
# # INISIALISASI MEDIAPIPE
# # ============================================================
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     smooth_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )


# # ============================================================
# # FUNGSI EKSTRAKSI KEYPOINT (identik dengan pipeline training)
# # ============================================================
# def extract_keypoints(landmarks, frame_w, frame_h):
#     """
#     Mengambil 12 keypoints target, normalisasi relatif terhadap titik tengah pinggul.
#     Mengembalikan:
#       - feature_vec : np.array shape (48,)  — untuk buffer
#       - idx_to_px   : dict idx -> (px_x, px_y) untuk gambar
#       - mid_hip_px  : tuple (px_x, px_y) titik anchor
#       - joint_info  : list (idx, visibility, imputed) untuk warna
#     """
#     left_hip  = landmarks[23]
#     right_hip = landmarks[24]
#     mid_hip_x = (left_hip.x + right_hip.x) / 2.0
#     mid_hip_y = (left_hip.y + right_hip.y) / 2.0
#     mid_hip_z = (left_hip.z + right_hip.z) / 2.0

#     mid_hip_px = (int(mid_hip_x * frame_w), int(mid_hip_y * frame_h))

#     feature_vec  = []
#     idx_to_px    = {}
#     joint_info   = []

#     for idx in TARGET_INDICES:
#         lm = landmarks[idx]
#         imputed = lm.visibility < VISIBILITY_THRESHOLD

#         idx_to_px[idx] = (int(lm.x * frame_w), int(lm.y * frame_h))
#         joint_info.append((idx, lm.visibility, imputed))

#         if imputed:
#             norm_x, norm_y, norm_z = 0.0, 0.0, 0.0
#         else:
#             norm_x = lm.x - mid_hip_x
#             norm_y = lm.y - mid_hip_y
#             norm_z = lm.z - mid_hip_z

#         feature_vec.extend([norm_x, norm_y, norm_z, lm.visibility])

#     return np.array(feature_vec), idx_to_px, mid_hip_px, joint_info


# def resample(buffer, target=TARGET_FRAMES):
#     """Resample buffer (list of 48-vec) ke TARGET_FRAMES menggunakan interpolasi linear."""
#     arr = np.array(buffer)
#     n   = len(arr)
#     if n == 0:
#         return None
#     if n == target:
#         return arr
#     x_old = np.linspace(0, 1, n)
#     x_new = np.linspace(0, 1, target)
#     return interp1d(x_old, arr, axis=0, kind='linear')(x_new)


# # ============================================================
# # FUNGSI GAMBAR OVERLAY
# # ============================================================
# def draw_skeleton(frame, idx_to_px, joint_info, mid_hip_px):
#     # Koneksi
#     for (a, b) in TARGET_CONNECTIONS:
#         if a in idx_to_px and b in idx_to_px:
#             cv2.line(frame, idx_to_px[a], idx_to_px[b], C_GRAY, 2, cv2.LINE_AA)

#     # Titik anchor pinggul
#     cv2.circle(frame, mid_hip_px, 9, C_PINK, -1)
#     cv2.circle(frame, mid_hip_px, 9, C_WHITE, 1)

#     # Sendi
#     for idx, vis, imputed in joint_info:
#         px    = idx_to_px[idx]
#         color = C_ORANGE if imputed else C_GREEN
#         cv2.circle(frame, px, 7, color, -1)
#         cv2.circle(frame, px, 7, C_WHITE, 1)
#         name = JOINT_NAMES.get(idx, str(idx))
#         cv2.putText(frame, name, (px[0] + 8, px[1] - 4),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)


# def draw_buffer_bar(frame, filled, total, x, y, w, h):
#     """Progress bar pengisian buffer."""
#     pct = min(filled / total, 1.0)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 60), -1)
#     bar_fill = int(w * pct)
#     color = (50, 200, 80) if pct >= 1.0 else (60, 140, 200)
#     cv2.rectangle(frame, (x, y), (x + bar_fill, y + h), color, -1)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 140), 1)
#     pct_txt = f"{int(pct * 100)}%"
#     cv2.putText(frame, pct_txt, (x + w + 6, y + h - 1),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1, cv2.LINE_AA)


# def draw_prediction_box(frame, pred_class, confidence, fps, frame_w, frame_h):
#     """Kotak prediksi di bagian atas frame."""
#     box_h = 80
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (frame_w, box_h), (15, 15, 25), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     if pred_class is not None:
#         cls_color = CLASS_COLORS.get(pred_class, C_WHITE)

#         # Bar confidence
#         bar_w = int((frame_w - 160) * confidence)
#         cv2.rectangle(frame, (10, 52), (frame_w - 150, 68), (40, 40, 60), -1)
#         cv2.rectangle(frame, (10, 52), (10 + bar_w, 68), cls_color, -1)
#         cv2.rectangle(frame, (10, 52), (frame_w - 150, 68), (100, 100, 120), 1)

#         # Nama kelas
#         cv2.putText(frame, pred_class.upper(), (10, 44),
#                     cv2.FONT_HERSHEY_DUPLEX, 1.2, cls_color, 2, cv2.LINE_AA)

#         # Angka confidence
#         conf_txt = f"{confidence * 100:.1f}%"
#         cv2.putText(frame, conf_txt, (frame_w - 145, 68),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, cls_color, 1, cv2.LINE_AA)

#     else:
#         cv2.putText(frame, "Mengumpulkan frame...", (10, 44),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_YELLOW, 2, cv2.LINE_AA)

#     # FPS di kanan atas
#     cv2.putText(frame, f"FPS: {fps:.1f}", (frame_w - 100, 22),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GRAY, 1, cv2.LINE_AA)

#     # Label "TCNN LIVE"
#     cv2.putText(frame, "TCNN LIVE", (frame_w // 2 - 50, 22),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 180), 1, cv2.LINE_AA)


# def draw_history_bar(frame, history, frame_w, frame_h):
#     """Bar riwayat prediksi di bagian bawah."""
#     if not history:
#         return
#     bar_h = 28
#     y0 = frame_h - bar_h
#     seg_w = frame_w // max(len(history), 1)

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, y0), (frame_w, frame_h), (15, 15, 25), -1)
#     cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

#     for i, (cls, conf) in enumerate(history):
#         x0 = i * seg_w
#         x1 = x0 + seg_w
#         color = CLASS_COLORS.get(cls, C_GRAY)
#         alpha = 0.3 + 0.7 * conf
#         cv2.rectangle(frame, (x0 + 1, y0 + 2), (x1 - 1, frame_h - 2),
#                       tuple(int(c * alpha) for c in color), -1)
#         if seg_w > 40:
#             cv2.putText(frame, cls[:1], (x0 + seg_w // 2 - 4, y0 + 19),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)


# # ============================================================
# # MAIN LOOP
# # ============================================================
# def main():
#     cap = cv2.VideoCapture(CAMERA_INDEX)
#     if not cap.isOpened():
#         print(f"[ERROR] Kamera index {CAMERA_INDEX} tidak bisa dibuka.")
#         print("  Coba ganti CAMERA_INDEX di bagian KONFIGURASI.")
#         exit(1)

#     # Set resolusi kamera (opsional — hapus jika kamera tidak support)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     print("\n" + "=" * 55)
#     print("  TCNN REAL-TIME INFERENCE — WEBCAM")
#     print("=" * 55)
#     print(f"  Kelas  : {CLASS_NAMES}")
#     print(f"  Window : {WINDOW_FRAMES} frame  |  Target resample: {TARGET_FRAMES} frame")
#     print(f"  Prediksi dijalankan setiap {N_SKIP_FRAMES} frame")
#     print("  Kontrol: [R] Reset buffer  [S] Screenshot  [ESC] Keluar")
#     print("=" * 55 + "\n")

#     # Buffer: menyimpan WINDOW_FRAMES keypoint terakhir
#     frame_buffer = deque(maxlen=WINDOW_FRAMES)

#     # State
#     pred_class    = None
#     confidence    = 0.0
#     pred_history  = deque(maxlen=30)   # Riwayat prediksi untuk bar bawah
#     frame_count   = 0
#     screenshot_n  = 0
#     last_good_kp  = np.zeros(len(TARGET_INDICES) * 4)

#     # FPS
#     fps_times = deque(maxlen=30)
#     t_prev    = time.time()

#     window_name = "TCNN Real-Time Inference"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[WARNING] Gagal membaca frame dari kamera.")
#             break

#         frame = cv2.flip(frame, 1)   # Mirror agar terasa natural
#         frame_h, frame_w = frame.shape[:2]

#         # FPS hitung
#         t_now = time.time()
#         fps_times.append(1.0 / max(t_now - t_prev, 1e-9))
#         t_prev = t_now
#         fps = np.mean(fps_times)

#         # --- Ekstraksi Pose ---
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results   = pose.process(frame_rgb)

#         detected = False
#         if results.pose_landmarks:
#             detected  = True
#             landmarks = results.pose_landmarks.landmark
#             kp_vec, idx_to_px, mid_hip_px, joint_info = extract_keypoints(
#                 landmarks, frame_w, frame_h)
#             draw_skeleton(frame, idx_to_px, joint_info, mid_hip_px)
#             last_good_kp = kp_vec
#         else:
#             kp_vec = last_good_kp   # fallback ke frame sebelumnya
#             cv2.putText(frame, "NO POSE DETECTED", (10, frame_h // 2),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_RED, 2, cv2.LINE_AA)

#         frame_buffer.append(kp_vec)
#         frame_count += 1

#         # --- Inferensi setiap N_SKIP_FRAMES ---
#         if frame_count % N_SKIP_FRAMES == 0 and len(frame_buffer) >= TARGET_FRAMES:
#             sequence = resample(list(frame_buffer), TARGET_FRAMES)
#             if sequence is not None:
#                 inp        = sequence[np.newaxis, ...]          # (1, 60, 48)
#                 probs      = model.predict(inp, verbose=0)[0]   # (num_classes,)
#                 pred_idx   = int(np.argmax(probs))
#                 confidence = float(probs[pred_idx])
#                 pred_class = CLASS_NAMES[pred_idx]
#                 pred_history.append((pred_class, confidence))

#         # --- Gambar UI ---
#         draw_prediction_box(frame, pred_class, confidence, fps, frame_w, frame_h)

#         # Buffer progress bar
#         draw_buffer_bar(frame,
#                         filled=len(frame_buffer),
#                         total=WINDOW_FRAMES,
#                         x=10, y=frame_h - 60,
#                         w=200, h=12)
#         cv2.putText(frame, f"Buffer ({len(frame_buffer)}/{WINDOW_FRAMES})",
#                     (10, frame_h - 65),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)

#         # Status pose
#         status_txt = "Pose: TERDETEKSI" if detected else "Pose: TIDAK TERDETEKSI"
#         status_clr = C_GREEN if detected else C_RED
#         cv2.putText(frame, status_txt, (10, frame_h - 35),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_clr, 1, cv2.LINE_AA)

#         # Riwayat prediksi (bar bawah)
#         draw_history_bar(frame, list(pred_history), frame_w, frame_h)

#         # Hint keyboard
#         cv2.putText(frame, "[R]eset  [S]creenshot  [ESC]Keluar",
#                     (frame_w - 280, frame_h - 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)

#         cv2.imshow(window_name, frame)

#         # --- Input keyboard ---
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:           # ESC
#             print("\n[KELUAR] Program dihentikan oleh user.")
#             break
#         elif key == ord('r') or key == ord('R'):
#             frame_buffer.clear()
#             pred_class  = None
#             confidence  = 0.0
#             pred_history.clear()
#             print("[RESET] Buffer dikosongkan.")
#         elif key == ord('s') or key == ord('S'):
#             screenshot_n += 1
#             path = os.path.join(SCREENSHOT_DIR, f"screenshot_{screenshot_n:04d}.png")
#             cv2.imwrite(path, frame)
#             print(f"[SCREENSHOT] Disimpan: {path}")

#     cap.release()
#     cv2.destroyAllWindows()
#     print("[SELESAI]")


# if __name__ == "__main__":
#     main()
