# """
# 5_video_inference.py
# ====================
# Analisis gerakan dari file video menggunakan model TCNN yang sudah dilatih.

# Cara penggunaan:
#   python 5_video_inference.py                          <- dialog pilih file
#   python 5_video_inference.py video_saya.mp4           <- langsung dari argumen

# Cara kerja:
#   1. Video diputar frame per frame.
#   2. MediaPipe mengekstrak keypoints per frame (identik pipeline training).
#   3. Sliding window 90 frame → diresample ke 60 → disuapkan ke model TCNN.
#   4. Prediksi & confidence ditampilkan overlay di atas video secara sinkron.
#   5. Di akhir video, ditampilkan ringkasan statistik gerakan yang terdeteksi.

# Kontrol keyboard saat playback:
#   [SPACE] - Pause / Resume
#   [R]     - Restart dari awal
#   [S]     - Screenshot frame saat ini
#   [ESC]   - Keluar
# """

# import os
# import sys
# import cv2
# import numpy as np
# from collections import deque
# from scipy.interpolate import interp1d
# from tensorflow import keras
# import time
# import tkinter as tk
# from tkinter import filedialog

# # ============================================================
# # KONFIGURASI
# # ============================================================
# MODEL_PATH        = "saved_models/best_tcnn_model.keras"
# CLASS_NAMES       = ["Squat", "Lunge"]
# TARGET_FRAMES     = 60
# WINDOW_FRAMES     = 90
# N_SKIP_FRAMES     = 3            # Prediksi setiap N frame (lebih kecil = lebih responsif)
# VISIBILITY_THRESHOLD = 0.5
# TARGET_INDICES    = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# SCREENSHOT_DIR    = "screenshots"
# PLAYBACK_SPEED    = 1.0          # 1.0 = normal, 0.5 = lambat, 2.0 = cepat

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
#     15: "L.Tgn",  16: "R.Tgn",
#     23: "L.Ping", 24: "R.Ping",
#     25: "L.Lutut",26: "R.Lutut",
#     27: "L.Kaki", 28: "R.Kaki",
# }

# CLASS_COLORS = {
#     "Squat": (50, 220, 100),
#     "Lunge": (255, 160, 50),
# }

# # Warna (BGR)
# C_GREEN  = (50, 220, 100)
# C_ORANGE = (30, 140, 255)
# C_WHITE  = (255, 255, 255)
# C_YELLOW = (60, 220, 255)
# C_PINK   = (200, 80, 255)
# C_GRAY   = (160, 160, 160)
# C_RED    = (60, 60, 220)
# C_DARK   = (15, 15, 25)

# os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# # ============================================================
# # PILIH FILE VIDEO
# # ============================================================
# def pick_video_file():
#     """Dialog GUI untuk memilih file video. Return path atau None."""
#     root = tk.Tk()
#     root.withdraw()
#     root.attributes("-topmost", True)
#     path = filedialog.askopenfilename(
#         title="Pilih File Video",
#         filetypes=[
#             ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv"),
#             ("All files", "*.*"),
#         ]
#     )
#     root.destroy()
#     return path if path else None


# # ============================================================
# # LOAD MODEL
# # ============================================================
# if not os.path.exists(MODEL_PATH):
#     print(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")
#     print("  Jalankan '3_build_and_train.py' terlebih dahulu!")
#     exit(1)

# print(f"Memuat model dari {MODEL_PATH} ...")
# model = keras.models.load_model(MODEL_PATH)
# print(f"  Input shape : {model.input_shape}  |  Output: {model.output_shape}")
# print(f"  Kelas       : {CLASS_NAMES}\n")


# # ============================================================
# # INISIALISASI MEDIAPIPE
# # ============================================================
# import mediapipe as mp
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     smooth_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )


# # ============================================================
# # FUNGSI EKSTRAKSI KEYPOINT (identik pipeline training)
# # ============================================================
# def extract_keypoints(landmarks, frame_w, frame_h):
#     left_hip  = landmarks[23]
#     right_hip = landmarks[24]
#     mid_hip_x = (left_hip.x + right_hip.x) / 2.0
#     mid_hip_y = (left_hip.y + right_hip.y) / 2.0
#     mid_hip_z = (left_hip.z + right_hip.z) / 2.0
#     mid_hip_px = (int(mid_hip_x * frame_w), int(mid_hip_y * frame_h))

#     feature_vec, idx_to_px, joint_info = [], {}, []
#     for idx in TARGET_INDICES:
#         lm = landmarks[idx]
#         imputed = lm.visibility < VISIBILITY_THRESHOLD
#         idx_to_px[idx] = (int(lm.x * frame_w), int(lm.y * frame_h))
#         joint_info.append((idx, lm.visibility, imputed))
#         if imputed:
#             nx, ny, nz = 0.0, 0.0, 0.0
#         else:
#             nx = lm.x - mid_hip_x
#             ny = lm.y - mid_hip_y
#             nz = lm.z - mid_hip_z
#         feature_vec.extend([nx, ny, nz, lm.visibility])

#     return np.array(feature_vec), idx_to_px, mid_hip_px, joint_info


# def resample(buffer, target=TARGET_FRAMES):
#     arr = np.array(buffer)
#     n   = len(arr)
#     if n < 2:
#         return None
#     if n == target:
#         return arr
#     x_old = np.linspace(0, 1, n)
#     x_new = np.linspace(0, 1, target)
#     return interp1d(x_old, arr, axis=0, kind='linear')(x_new)


# # ============================================================
# # FUNGSI GAMBAR
# # ============================================================
# def draw_skeleton(frame, idx_to_px, joint_info, mid_hip_px):
#     for (a, b) in TARGET_CONNECTIONS:
#         if a in idx_to_px and b in idx_to_px:
#             cv2.line(frame, idx_to_px[a], idx_to_px[b], C_GRAY, 2, cv2.LINE_AA)

#     cv2.circle(frame, mid_hip_px, 9, C_PINK, -1)
#     cv2.circle(frame, mid_hip_px, 9, C_WHITE, 1)

#     for idx, vis, imputed in joint_info:
#         px    = idx_to_px[idx]
#         color = C_ORANGE if imputed else C_GREEN
#         cv2.circle(frame, px, 7, color, -1)
#         cv2.circle(frame, px, 7, C_WHITE, 1)
#         name = JOINT_NAMES.get(idx, str(idx))
#         cv2.putText(frame, name, (px[0] + 8, px[1] - 4),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)


# def draw_top_bar(frame, pred_class, confidence, frame_w,
#                  current_frame, total_frames, filename):
#     box_h = 85
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (frame_w, box_h), C_DARK, -1)
#     cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

#     # Nama file (kiri)
#     fname = os.path.basename(filename)
#     fname = fname if len(fname) <= 35 else fname[:32] + "..."
#     cv2.putText(frame, fname, (10, 18),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1, cv2.LINE_AA)

#     # Progress timeline (frame progress)
#     tl_x, tl_y, tl_w, tl_h = 10, 22, frame_w - 20, 8
#     pct = current_frame / max(total_frames - 1, 1)
#     cv2.rectangle(frame, (tl_x, tl_y), (tl_x + tl_w, tl_y + tl_h), (50, 50, 70), -1)
#     cv2.rectangle(frame, (tl_x, tl_y), (tl_x + int(tl_w * pct), tl_y + tl_h), (90, 160, 220), -1)
#     cv2.rectangle(frame, (tl_x, tl_y), (tl_x + tl_w, tl_y + tl_h), (100, 100, 120), 1)
#     cv2.putText(frame, f"{current_frame}/{total_frames}", (frame_w - 100, tl_y + 7),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1, cv2.LINE_AA)

#     # Prediksi
#     if pred_class is not None:
#         cls_color = CLASS_COLORS.get(pred_class, C_WHITE)
#         cv2.putText(frame, pred_class.upper(), (10, 68),
#                     cv2.FONT_HERSHEY_DUPLEX, 1.3, cls_color, 2, cv2.LINE_AA)

#         # Bar confidence
#         bar_x, bar_y = 150, 52
#         bar_w = int((frame_w - 310) * confidence)
#         cv2.rectangle(frame, (bar_x, bar_y), (frame_w - 160, bar_y + 16), (40, 40, 60), -1)
#         cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16), cls_color, -1)
#         cv2.rectangle(frame, (bar_x, bar_y), (frame_w - 160, bar_y + 16), (100, 100, 120), 1)
#         cv2.putText(frame, f"{confidence * 100:.1f}%", (frame_w - 155, bar_y + 13),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, cls_color, 1, cv2.LINE_AA)

#         # Label semua kelas (confidence masing-masing)
#         cv2.putText(frame, "Confidence:", (150, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GRAY, 1, cv2.LINE_AA)
#     else:
#         cv2.putText(frame, "Mengumpulkan frame...", (10, 65),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_YELLOW, 2, cv2.LINE_AA)

#     # Label TCNN VIDEO
#     cv2.putText(frame, "TCNN VIDEO", (frame_w // 2 - 45, 18),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 180), 1, cv2.LINE_AA)


# def draw_all_class_probs(frame, probs, frame_w, frame_h):
#     """Panel kecil di kanan bawah: confidence semua kelas."""
#     panel_w = 180
#     panel_h = len(CLASS_NAMES) * 28 + 20
#     px = frame_w - panel_w - 10
#     py = frame_h - panel_h - 40

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (px - 5, py - 5), (frame_w - 5, py + panel_h), C_DARK, -1)
#     cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

#     cv2.putText(frame, "Probabilitas:", (px, py + 12),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)

#     for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
#         y = py + 28 + i * 28
#         color = CLASS_COLORS.get(cls, C_GRAY)
#         bar_w = int(160 * prob)
#         cv2.rectangle(frame, (px, y), (px + 160, y + 16), (40, 40, 60), -1)
#         cv2.rectangle(frame, (px, y), (px + bar_w, y + 16), color, -1)
#         cv2.rectangle(frame, (px, y), (px + 160, y + 16), (80, 80, 100), 1)
#         cv2.putText(frame, f"{cls}: {prob*100:.1f}%", (px + 2, y + 12),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WHITE, 1, cv2.LINE_AA)


# def draw_bottom_bar(frame, pred_history, paused, frame_w, frame_h):
#     """Bar bawah: riwayat prediksi + hint keyboard."""
#     bar_h = 32
#     y0 = frame_h - bar_h

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, y0), (frame_w, frame_h), C_DARK, -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     # Riwayat warna
#     if pred_history:
#         seg_w = max((frame_w - 250) // len(pred_history), 1)
#         for i, (cls, conf) in enumerate(pred_history):
#             x0  = i * seg_w
#             clr = CLASS_COLORS.get(cls, C_GRAY)
#             alpha = 0.3 + 0.7 * conf
#             cv2.rectangle(frame, (x0, y0 + 2), (x0 + seg_w - 1, frame_h - 2),
#                           tuple(int(c * alpha) for c in clr), -1)

#     # Hint
#     pause_hint = "[SPACE] Resume" if paused else "[SPACE] Pause"
#     hints = f"{pause_hint}  [R] Restart  [S] Screenshot  [ESC] Keluar"
#     cv2.putText(frame, hints, (8, frame_h - 8),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.37, C_GRAY, 1, cv2.LINE_AA)

#     if paused:
#         cv2.putText(frame, "|| PAUSED", (frame_w - 100, frame_h - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_YELLOW, 1, cv2.LINE_AA)


# def draw_summary_screen(frame_w, frame_h, pred_history, video_name, total_frames):
#     """Frame ringkasan yang ditampilkan di akhir video."""
#     canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
#     canvas[:] = (18, 18, 28)

#     # Hitung statistik
#     from collections import Counter
#     counts = Counter(cls for cls, _ in pred_history)
#     avg_conf = {}
#     for cls in CLASS_NAMES:
#         confs = [c for cl, c in pred_history if cl == cls]
#         avg_conf[cls] = np.mean(confs) if confs else 0.0

#     dominant = counts.most_common(1)[0][0] if counts else "N/A"
#     dom_color = CLASS_COLORS.get(dominant, C_WHITE)

#     # Judul
#     cv2.putText(canvas, "HASIL ANALISIS VIDEO", (frame_w // 2 - 180, 60),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.0, C_YELLOW, 2, cv2.LINE_AA)
#     cv2.line(canvas, (40, 75), (frame_w - 40, 75), (60, 60, 80), 1)

#     # Nama file
#     fname = os.path.basename(video_name)
#     cv2.putText(canvas, f"File: {fname}", (40, 105),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1, cv2.LINE_AA)
#     cv2.putText(canvas, f"Total frame dianalisa: {total_frames}", (40, 130),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_GRAY, 1, cv2.LINE_AA)

#     # Gerakan dominan (besar di tengah)
#     cv2.putText(canvas, "Gerakan Terdeteksi:", (40, 170),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1, cv2.LINE_AA)
#     cv2.putText(canvas, dominant.upper(), (40, 230),
#                 cv2.FONT_HERSHEY_DUPLEX, 2.5, dom_color, 3, cv2.LINE_AA)

#     # Distribusi per kelas
#     y = 280
#     cv2.putText(canvas, "Distribusi Prediksi:", (40, y),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1, cv2.LINE_AA)
#     y += 30
#     total_preds = max(sum(counts.values()), 1)
#     for cls in CLASS_NAMES:
#         count = counts.get(cls, 0)
#         pct   = count / total_preds
#         conf  = avg_conf[cls]
#         color = CLASS_COLORS.get(cls, C_GRAY)

#         bar_w = int(400 * pct)
#         cv2.rectangle(canvas, (40, y), (440, y + 24), (40, 40, 60), -1)
#         cv2.rectangle(canvas, (40, y), (40 + bar_w, y + 24), color, -1)
#         cv2.rectangle(canvas, (40, y), (440, y + 24), (80, 80, 100), 1)
#         cv2.putText(canvas, f"{cls}: {count} frame ({pct*100:.1f}%)  avg conf: {conf*100:.1f}%",
#                     (450, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
#         y += 38

#     # Timeline berwarna
#     y += 10
#     cv2.putText(canvas, "Timeline Prediksi:", (40, y),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1, cv2.LINE_AA)
#     y += 16
#     tl_w = frame_w - 80
#     if pred_history:
#         seg_w = max(tl_w // len(pred_history), 1)
#         for i, (cls, conf) in enumerate(pred_history):
#             x0 = 40 + i * seg_w
#             clr = CLASS_COLORS.get(cls, C_GRAY)
#             alpha = 0.4 + 0.6 * conf
#             cv2.rectangle(canvas, (x0, y), (x0 + seg_w - 1, y + 22),
#                           tuple(int(c * alpha) for c in clr), -1)

#     # Legend
#     y += 38
#     for cls in CLASS_NAMES:
#         color = CLASS_COLORS.get(cls, C_GRAY)
#         cv2.circle(canvas, (50, y), 7, color, -1)
#         cv2.putText(canvas, cls, (65, y + 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
#         y += 26

#     # Hint
#     cv2.line(canvas, (40, frame_h - 50), (frame_w - 40, frame_h - 50), (60, 60, 80), 1)
#     cv2.putText(canvas, "[R] Putar Ulang   [ESC] Keluar", (frame_w // 2 - 130, frame_h - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GRAY, 1, cv2.LINE_AA)

#     return canvas


# # ============================================================
# # FUNGSI PUTAR VIDEO
# # ============================================================
# def play_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"[ERROR] Tidak bisa membuka video: {video_path}")
#         return

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30
#     wait_ms      = max(1, int(1000 / fps_video / PLAYBACK_SPEED))

#     print(f"\n{'='*55}")
#     print(f"  File      : {os.path.basename(video_path)}")
#     print(f"  Frame     : {total_frames}")
#     print(f"  FPS video : {fps_video:.1f}")
#     print(f"  Kontrol   : [SPACE] Pause  [R] Restart  [S] Screenshot  [ESC] Keluar")
#     print(f"{'='*55}\n")

#     frame_buffer  = deque(maxlen=WINDOW_FRAMES)
#     pred_class    = None
#     confidence    = 0.0
#     all_probs     = np.zeros(len(CLASS_NAMES))
#     pred_history  = deque(maxlen=300)
#     last_good_kp  = np.zeros(len(TARGET_INDICES) * 4)
#     frame_count   = 0
#     screenshot_n  = 0
#     paused        = False

#     window_name = f"TCNN Analisis — {os.path.basename(video_path)}"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#     while True:
#         if not paused:
#             ret, frame = cap.read()
#             if not ret:
#                 # Video habis → tampilkan ringkasan
#                 summary = draw_summary_screen(
#                     frame.shape[1] if frame is not None else 1280,
#                     frame.shape[0] if frame is not None else 720,
#                     list(pred_history), video_path, frame_count
#                 )
#                 cv2.imshow(window_name, summary)
#                 print("\n[SELESAI] Video habis. Tekan [R] untuk ulang atau [ESC] untuk keluar.")
#                 while True:
#                     k = cv2.waitKey(50) & 0xFF
#                     if k == 27:
#                         cap.release()
#                         cv2.destroyAllWindows()
#                         return
#                     elif k == ord('r') or k == ord('R'):
#                         cap.release()
#                         play_video(video_path)   # Rekursif restart
#                         return
#                     # Update window
#                     cv2.imshow(window_name, summary)
#                 break

#             frame_h, frame_w = frame.shape[:2]

#             # Ekstraksi pose
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results   = pose.process(frame_rgb)

#             if results.pose_landmarks:
#                 kp_vec, idx_to_px, mid_hip_px, joint_info = extract_keypoints(
#                     results.pose_landmarks.landmark, frame_w, frame_h)
#                 draw_skeleton(frame, idx_to_px, joint_info, mid_hip_px)
#                 last_good_kp = kp_vec
#             else:
#                 kp_vec = last_good_kp
#                 cv2.putText(frame, "NO POSE DETECTED", (10, frame_h // 2),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_RED, 2, cv2.LINE_AA)

#             frame_buffer.append(kp_vec)
#             frame_count += 1

#             # Inferensi
#             # Boleh inferensi jika buffer sudah punya minimal 10 frame (nanti di-resample ke 60)
#             if frame_count % N_SKIP_FRAMES == 0 and len(frame_buffer) >= 10:
#                 seq = resample(list(frame_buffer), TARGET_FRAMES)
#                 if seq is not None:
#                     probs      = model.predict(seq[np.newaxis, ...], verbose=0)[0]
#                     pred_idx   = int(np.argmax(probs))
#                     confidence = float(probs[pred_idx])
#                     pred_class = CLASS_NAMES[pred_idx]
#                     all_probs  = probs
#                     pred_history.append((pred_class, confidence))

#             # Overlay UI
#             draw_top_bar(frame, pred_class, confidence, frame_w,
#                          frame_count, total_frames, video_path)
#             if pred_class is not None:
#                 draw_all_class_probs(frame, all_probs, frame_w, frame_h)
#             draw_bottom_bar(frame, list(pred_history), paused, frame_w, frame_h)

#             cv2.imshow(window_name, frame)

#         # Keyboard input
#         key = cv2.waitKey(wait_ms if not paused else 50) & 0xFF

#         if key == 27:       # ESC
#             print("\n[KELUAR] Dihentikan user.")
#             break
#         elif key == ord(' '):
#             paused = not paused
#             print("[PAUSE]" if paused else "[RESUME]")
#         elif key == ord('r') or key == ord('R'):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             frame_buffer.clear()
#             pred_history.clear()
#             pred_class = None
#             confidence = 0.0
#             all_probs  = np.zeros(len(CLASS_NAMES))
#             frame_count = 0
#             paused = False
#             print("[RESTART] Video dimulai ulang.")
#         elif key == ord('s') or key == ord('S'):
#             screenshot_n += 1
#             path = os.path.join(SCREENSHOT_DIR, f"video_ss_{screenshot_n:04d}.png")
#             if not paused and 'frame' in dir():
#                 cv2.imwrite(path, frame)
#                 print(f"[SCREENSHOT] {path}")

#     cap.release()
#     cv2.destroyAllWindows()


# # ============================================================
# # MAIN
# # ============================================================
# if __name__ == "__main__":
#     # Argumen CLI atau dialog pilih file
#     if len(sys.argv) > 1:
#         video_path = sys.argv[1]
#         if not os.path.exists(video_path):
#             print(f"[ERROR] File tidak ditemukan: {video_path}")
#             exit(1)
#     else:
#         print("Membuka dialog pemilihan file video...")
#         video_path = pick_video_file()
#         if not video_path:
#             print("[BATAL] Tidak ada file yang dipilih.")
#             exit(0)

#     play_video(video_path)
#     print("[SELESAI]")
