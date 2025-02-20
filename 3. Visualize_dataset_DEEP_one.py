import os
import numpy as np
import matplotlib.pyplot as plt

# Final_dataset 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
final_path = os.path.join(script_dir, 'data', 'dataset_Final')  # data/dataset_Final 경로

# 데이터 로드
y_file_path = os.path.join(final_path, "Y_dataset.npy")

# Y 데이터 로드 (PPG 신호)
Y_data = np.load(y_file_path)  # shape: (samples, T)

# 특정 샘플 인덱스 지정
specific_indices = [5,6,7,8]  # 원하는 샘플 번호 입력

# 시각화
fig, axes = plt.subplots(len(specific_indices), 1, figsize=(10, len(specific_indices) * 2))
fig.suptitle("Sample Visualization of Y_dataset Dataset")

for i, idx in enumerate(specific_indices):
    ax = axes[i] if len(specific_indices) > 1 else axes
    ax.plot(Y_data[idx], label=f"Sample {idx}")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("PPG [BPM]")
    ax.legend()

plt.tight_layout()
plt.show()
