import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# 1. Device 및 랜덤시드 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


# 2. 데이터 불러오기 및 전처리
# 현재 폴더 기준 data/dataset_Final 폴더에서 데이터 로드
data_dir = os.path.join(os.getcwd(), "data", "dataset_Final")
X_path = os.path.join(data_dir, "X_dataset.npy")
Y_path = os.path.join(data_dir, "Y_dataset.npy")

# X: ( # , 300, 64, 64, 4), Y: ( # , 300)
X = np.load(X_path)  # RGB + NIR 영상 데이터 (ROI 추출 후 64x64, 4채널)
Y = np.load(Y_path)  # PPG 데이터 (아직 정규화 X)

# PyTorch에서 사용하는 텐서 shape는 (batch, channel, T, H, W)이므로 차원 순서 변경
# ( # , 300, 64, 64, 4) -> ( # , 4, 300, 64, 64)
X = np.transpose(X, (0, 4, 1, 2, 3))

# float32로 형태 변환
X = X.astype(np.float32)
Y = Y.astype(np.float32)

# Tensor로 변환
X_tensor = torch.from_numpy(X)  # shape: ( # , 4, 300, 64, 64)
Y_tensor = torch.from_numpy(Y)  # shape: ( # , 300)


# 3. Train/Val 데이터셋 구성 및 y 정규화
num_samples = X_tensor.shape[0]
val_size = int(0.1 * num_samples)
train_size = num_samples - val_size

# TensorDataset 생성 (X, Y)
dataset = TensorDataset(X_tensor, Y_tensor)

# random_split으로 train과 validation Dataset 분리
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# 학습 전 train 데이터 y의 전체 평균과 표준편차를 구함 (정규화에 사용)
# train_dataset은 Subset 객체이므로 각 샘플을 순회
train_y_all = []
for _, y in train_dataset:
    train_y_all.append(y)
train_y_all = torch.cat(train_y_all, dim=0)  # (train_size * 300,)
y_mean = train_y_all.mean().item()
y_std = train_y_all.std().item()
print(f"Computed y_mean: {y_mean:.4f}, y_std: {y_std:.4f}")

# y 정규화 함수
def normalize_y(y):
    return (y - y_mean) / y_std

# train, val 각각의 y를 정규화하여 새로운 TensorDataset 생성
def create_normalized_dataset(subset):
    xs, ys = [], []
    for x, y in subset:
        xs.append(x)
        ys.append(normalize_y(y))
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return TensorDataset(xs, ys)

train_dataset = create_normalized_dataset(train_dataset)
val_dataset = create_normalized_dataset(val_dataset)

# DataLoader 생성
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# 4. PhysNet (3D CNN 기반) 모델 정의
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
        
        # T 차원을 유지하기 위해 (B, T, 64, 8, 8)로 차원 변경
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, 64, 8, 8)
        B, T, C, H, W = x.shape
        x = x.reshape(B, T, C * H * W)  # (B, T, 4096)
        x = self.fc(x)  # (B, T, 1)
        x = x.squeeze(-1)  # (B, T)
        return x

# 모델 생성 및 device로 이동
model = PhysNet().to(device)


# 5. Loss, Optimizer, 학습 파라미터 설정
criterion = nn.MSELoss()  # 정규화된 값에 대해 MSE Loss 사용
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 1  # 필요에 따라 조정

train_losses = []
val_losses = []
val_mae_list = []


# 6. 학습 및 검증 루프
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)  # (B, 4, 300, 64, 64)
        batch_y = batch_y.to(device)  # (B, 300)
        
        optimizer.zero_grad()
        outputs = model(batch_x)  # 예측: (B, 300)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_x.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation
    model.eval()
    val_running_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            outputs = model(val_x)  # 정규화된 출력
            loss = criterion(outputs, val_y)
            val_running_loss += loss.item() * val_x.size(0)
            
            # 검증 시, 예측값과 GT를 원래 PPG BPM 형태로 변환
            outputs_denorm = outputs * y_std + y_mean
            val_y_denorm = val_y * y_std + y_mean
            mae = torch.mean(torch.abs(outputs_denorm - val_y_denorm))
            total_mae += mae.item() * val_x.size(0)
            
    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    epoch_val_mae = total_mae / len(val_loader.dataset)
    val_mae_list.append(epoch_val_mae)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, "
          f"Val Loss: {epoch_val_loss:.6f}, Val MAE: {epoch_val_mae:.6f}")


# 7. Loss 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()


# 8. 모델 저장 (정규화 파라미터와 함께 저장)
save_dir = os.path.join(os.getcwd(), "data", "model_3DCNN")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "physnet_model.pth")

torch.save({
    'model_state_dict': model.state_dict(),
    'y_mean': y_mean,
    'y_std': y_std
}, save_path)
print("최종 모델이 저장되었습니다:", save_path)
