# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# import glob
# from scipy.interpolate import interp1d

# # ==========================================
# # KONFIGURASI SUPER PARAMETER
# # ==========================================
# INPUT_DIR = 'dataset_raw_video'
# OUTPUT_DIR = 'dataset_numpy'
# TARGET_FRAMES = 60
# VISIBILITY_THRESHOLD = 0.5  # Batas toleransi sendi terlihat (0.0 s/d 1.0)

# # Indeks keypoints Mediapipe
# # 11-16 (Bahu-Siku-Tangan), 23-28 (Pinggul-Lutut-Kaki)
# TARGET_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# # Inisialisasi Mediapipe (Tangguh Mode)
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,       # Gunakan '1' untuk balance speed/accuracy. Pakai '2' jika PC kuat & video sangat buram.
#     smooth_landmarks=True,    # OTOMATIS MENGURANGI JITTER PADA VIDEO BURIK
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # ==========================================
# # FUNGSI 1: EKSTRAKSI & NORMALISASI
# # ==========================================
# def extract_and_normalize(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames_data = []
#     last_good_frame = np.zeros((len(TARGET_INDICES) * 4,)) 

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)
        
#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
            
#             # Hitung Titik Tengah Pinggul (Anchor Point)
#             left_hip = landmarks[23]
#             right_hip = landmarks[24]
#             mid_hip_x = (left_hip.x + right_hip.x) / 2.0
#             mid_hip_y = (left_hip.y + right_hip.y) / 2.0
#             mid_hip_z = (left_hip.z + right_hip.z) / 2.0
            
#             current_frame_data = []
            
#             # Ambil target sendi dengan logika Zero-Imputation
#             for idx in TARGET_INDICES:
#                 lm = landmarks[idx]
                
#                 # CEK VISIBILITY (Solusi Video Terpotong)
#                 if lm.visibility >= VISIBILITY_THRESHOLD:
#                     # Sendi terlihat -> Lakukan Translasi Normal
#                     norm_x = lm.x - mid_hip_x
#                     norm_y = lm.y - mid_hip_y
#                     norm_z = lm.z - mid_hip_z
#                 else:
#                     # Sendi hilang/terpotong -> Paksa ke 0,0,0 (Titik Pinggul)
#                     norm_x, norm_y, norm_z = 0.0, 0.0, 0.0
                
#                 current_frame_data.extend([norm_x, norm_y, norm_z, lm.visibility])
            
#             frames_data.append(current_frame_data)
#             last_good_frame = current_frame_data 
            
#         else:
#             # Gagal deteksi orang -> pakai frame terakhir
#             frames_data.append(last_good_frame)

#     cap.release()
#     return np.array(frames_data)

# # ==========================================
# # FUNGSI 2: UNIFORM TEMPORAL RESAMPLING
# # ==========================================
# def resample_temporal_data(data_array, target_frames=TARGET_FRAMES):
#     original_frames = data_array.shape[0]
#     if original_frames == 0: return None
#     if original_frames == target_frames: return data_array
        
#     x_old = np.linspace(0, 1, original_frames)
#     x_new = np.linspace(0, 1, target_frames)
    
#     interpolator = interp1d(x_old, data_array, axis=0, kind='linear')
#     return interpolator(x_new)

# # ==========================================
# # EKSEKUSI UTAMA
# # ==========================================
# if __name__ == "__main__":
#     classes = ['squat', 'lunge', 'arm_raise']
#     if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
#     for class_name in classes:
#         input_class_dir = os.path.join(INPUT_DIR, class_name)
#         output_class_dir = os.path.join(OUTPUT_DIR, class_name)
        
#         if not os.path.exists(input_class_dir): continue
#         if not os.path.exists(output_class_dir): os.makedirs(output_class_dir)
            
#         video_files = glob.glob(os.path.join(input_class_dir, '*.mp4'))
#         print(f"Memproses {len(video_files)} video kelas '{class_name}'...")
        
#         for video_path in video_files:
#             filename = os.path.basename(video_path)
#             name_only = os.path.splitext(filename)[0]
#             output_filepath = os.path.join(output_class_dir, f"{name_only}.npy")
            
#             if os.path.exists(output_filepath): continue
                
#             raw_data = extract_and_normalize(video_path)
#             resampled_data = resample_temporal_data(raw_data, TARGET_FRAMES)
            
#             if resampled_data is not None:
#                 np.save(output_filepath, resampled_data)
#                 print(f"  [OK] {filename}")
#             else:
#                 print(f"  [GAGAL] {filename}")
                
#     print("\n[SELESAI] Data NumPy siap untuk training TCNN!")