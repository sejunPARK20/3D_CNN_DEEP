import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. Device 및 random seed 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


# 2. 3D-CNN 모델 클래스 정의 (학습 시와 동일)
class PhysNet(nn.Module):
    def __init__(self):
        super(PhysNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2)),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.conv1(x)  
        x = self.conv2(x)   
        x = self.conv3(x)   
        x = self.conv4(x)   
        # 시간 축(T)을 앞쪽으로 옮김: (B, T, 64, 8, 8)
        x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.shape
        x = x.reshape(B, T, C * H * W)  # (B, T, 4096)
        # 각 time step별로 fc 적용 → (B, T, 1)
        x = self.fc(x)
        x = x.squeeze(-1)  # (B, T)
        return x

# 3. 테스트 데이터셋 불러오기 및 전처리

# ★★★★★★★★★★★★★★★★★★★★★★★★★
test_num = 5  # 원하는 Tesetset 번호 설정 (1~5)
# ★★★★★★★★★★★★★★★★★★★★★★★★★

# 파일 경로
test_data_dir = os.path.join(os.getcwd(), "data", "dataset_Test_Final")
X_test_path = os.path.join(test_data_dir, f"X_dataset_test{test_num}.npy")
Y_test_path = os.path.join(test_data_dir, f"Y_dataset_test{test_num}.npy")

X_test = np.load(X_test_path)   
Y_test = np.load(Y_test_path)  

X_test = np.transpose(X_test, (0, 4, 1, 2, 3))
X_test = X_test.astype(np.float32)
Y_test = Y_test.astype(np.float32)

X_test_tensor = torch.from_numpy(X_test)   # (N, 4, T, 64, 64)
Y_test_tensor = torch.from_numpy(Y_test)   # (N, T)

# TensorDataset 및 DataLoader 생성
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# 4. 저장된 모델 및 정규화 파라미터 불러오기
# 모델은 현재 폴더 내의 data/model_3DCNN 폴더에 위치, 파일명은 model_3DCNN.pth
model_dir = os.path.join(os.getcwd(), "data", "model_3DCNN")
model_path = os.path.join(model_dir, "model_3DCNN.pth")

# 모델 인스턴스 생성 후 state_dict 불러오기
model = PhysNet().to(device)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 저장된 정규화 파라미터 불러오기
y_mean = checkpoint['y_mean']
y_std = checkpoint['y_std']
print(f"Loading -> y_mean: {y_mean:.4f}, y_std: {y_std:.4f}")


# 5. 테스트 데이터셋으로 예측 수행 및 MAE 계산
all_preds = []
all_truths = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device) 
        outputs = model(batch_x)        
        all_preds.append(outputs.cpu())
        all_truths.append(batch_y)

all_preds = torch.cat(all_preds, dim=0)   
all_truths = torch.cat(all_truths, dim=0) 

preds_denorm = all_preds * y_std + y_mean

# 전체 MAE 계산 (전체 시퀀스에 대해)
mae = torch.mean(torch.abs(preds_denorm - all_truths))
print(f"MAE at Testset: {mae.item():.4f}")


# 6. 실제 PPG와 예측 PPG 그래프로 비교 
num_clips_to_plot = 4  # 0,1,2,3번 클립 사용
concatenated_true_ppg = []
concatenated_pred_ppg = []
concatenated_time_axis = []

for i in range(num_clips_to_plot):
    true_ppg = all_truths[i].numpy()   
    pred_ppg = preds_denorm[i].numpy()  
    
    # 시간축을 연속적으로 연결하기 위해 오프셋 추가
    time_offset = i * len(true_ppg)
    time_axis = np.arange(len(true_ppg)) + time_offset

    concatenated_true_ppg.extend(true_ppg)
    concatenated_pred_ppg.extend(pred_ppg)
    concatenated_time_axis.extend(time_axis)

plt.figure(figsize=(12, 6))
plt.plot(concatenated_time_axis, concatenated_true_ppg, label="True PPG", linewidth=2, color="blue")
plt.plot(concatenated_time_axis, concatenated_pred_ppg, 'r.', markersize=4, label="Predicted PPG")  

# 각 클립의 경계를 표시
for i in range(1, num_clips_to_plot):
    plt.axvline(x=i * len(true_ppg), color='gray', linestyle='--', linewidth=0.5)

plt.xlabel("Time Frame")
plt.ylabel("PPG [BPM]")
plt.title(f"TotalFace_700_Samples - <test{test_num}>")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


