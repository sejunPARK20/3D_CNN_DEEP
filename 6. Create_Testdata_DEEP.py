import os
import cv2
import numpy as np
import pandas as pd
import torch  # GPU 가속 적용
from scipy.interpolate import interp1d
from mediapipe import solutions

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 범위 설정
my_data_range = [(0, 0)]
data_range = []
for start, end in my_data_range:
    data_range.extend(range(start, end + 1))

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, 'data', 'test_data')
final_path = os.path.join(script_dir, 'data', 'dataset_Test_Final')
os.makedirs(final_path, exist_ok=True)

# Mediapipe 초기화
mp_face_detection = solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# 시퀀스 길이 설정 (T)
T = 300  
fps_video = 30

X_combined, Y_combined = [], []

def extract_face_roi(frame, detector, size=(64, 64)):
    """ 얼굴 ROI를 추출하고 64x64로 변환 """
    results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w, _ = frame.shape
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        x_min, y_min = int(bbox.xmin * w), int(bbox.ymin * h)
        width, height = int(bbox.width * w), int(bbox.height * h)
        face = frame[y_min:y_min + height, x_min:x_min + width]
        face_resized = cv2.resize(face, size)
        return face_resized
    return None

for data_num in data_range:
    print(f"★ Processing data number ★ : {data_num}")

    time_file = os.path.join(base_path, f"{data_num}_time.txt")
    ppg_file = os.path.join(base_path, f"{data_num}_ppg.txt")
    nir_video_file = os.path.join(base_path, f"{data_num}_ir.mp4")
    rgb_video_file = os.path.join(base_path, f"{data_num}_rgb.mp4")

    if not all(os.path.exists(f) for f in [time_file, ppg_file, nir_video_file, rgb_video_file]):
        print(f"Missing file: {data_num}")
        continue

    # 데이터 로드
    time_data = np.loadtxt(time_file)
    ppg_data = pd.read_csv(ppg_file, sep=r'\s+', header=None)
    ppg_time, ppg_signals = ppg_data.iloc[1:, 0].values, ppg_data.iloc[1:, 1].values
    ppg_interp = interp1d(ppg_time, ppg_signals, kind='cubic', fill_value="extrapolate")

    cap_nir, cap_rgb = cv2.VideoCapture(nir_video_file), cv2.VideoCapture(rgb_video_file)
    frames_nir, frames_rgb = [], []
    while True:
        ret_nir, frame_nir = cap_nir.read()
        ret_rgb, frame_rgb = cap_rgb.read()
        if not ret_nir or not ret_rgb:
            break
        frames_nir.append(frame_nir)
        frames_rgb.append(frame_rgb)
    cap_nir.release()
    cap_rgb.release()

    for start_idx in range(0, len(time_data) - T + 1, T):
        clip_frames_rgb = frames_rgb[start_idx:start_idx + T]
        clip_frames_nir = frames_nir[start_idx:start_idx + T]
        if len(clip_frames_rgb) < T or len(clip_frames_nir) < T:
            continue

        roi_frames_combined = []
        ppg_values = []

        for idx in range(T):
            frame_rgb = clip_frames_rgb[idx]
            frame_nir = clip_frames_nir[idx]
            ppg_value = ppg_interp(time_data[start_idx + idx])

            face_rgb = extract_face_roi(frame_rgb, face_detection, size=(64, 64))
            face_nir = extract_face_roi(frame_nir, face_detection, size=(64, 64))
            if face_rgb is None or face_nir is None:
                continue

            face_nir_gray = cv2.cvtColor(face_nir, cv2.COLOR_BGR2GRAY)  # NIR을 단일 채널로 변환
            face_nir_gray = face_nir_gray[..., np.newaxis]  # (64, 64, 1) 형태로 변경
            combined_frame = np.concatenate([face_rgb, face_nir_gray], axis=-1)  # (64, 64, 4)

            roi_frames_combined.append(combined_frame)
            ppg_values.append(ppg_value)

        if len(roi_frames_combined) == T:
            X_combined.append(np.array(roi_frames_combined))
            Y_combined.append(np.array(ppg_values))


X_combined = np.array(X_combined).astype(np.float32) / 255.0
Y_combined = np.array(Y_combined)

np.save(os.path.join(final_path, "X_dataset_test.npy"), X_combined)
np.save(os.path.join(final_path, "Y_dataset_test.npy"), Y_combined)


print("★★★ 데이터 저장 완료! ★★★")
print("X_data_test shape:", X_combined.shape)
print("Y_data_test shape:", Y_combined.shape)

