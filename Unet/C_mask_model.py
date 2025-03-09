import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import Dataset
from torchvision import transforms

from Unet.UnetVessel import SEModule


class CMaskDataset(Dataset):
    def __init__(self, df, transform=None):
        self.valid_samples = []
        self.transform = transform

        # 扫描有效样本（必须同时存在左右眼）
        for _, row in df.iterrows():
            left_path = os.path.join("../dataset/Archive/mask", f"{row['ID']}_left_mask.jpg")
            right_path = os.path.join("../dataset/Archive/mask", f"{row['ID']}_right_mask.jpg")

            if os.path.exists(left_path) and os.path.exists(right_path):
                self.valid_samples.append({
                    'id': row['ID'],
                    'left': left_path,
                    'right': right_path,
                    'label': row['C']
                })
        print(f"Loaded {len(self.valid_samples)} valid pairs")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        try:
            left_img = Image.open(sample['left']).convert('L')
            right_img = Image.open(sample['right']).convert('L')

            if self.transform:
                left_img = self.transform(left_img)  # [1, 256, 512]
                right_img = self.transform(right_img)

            # 水平拼接成512x512
            combined = torch.cat([left_img, right_img], dim=2)  # dim=2为宽度维度
            return combined, torch.tensor([sample['label']], dtype=torch.float32)
        except Exception as e:
            print(f"加载失败 {sample['id']}: {str(e)}")
            return torch.Tensor(), torch.Tensor()


# 数据转换配置
train_transform = transforms.Compose([
    transforms.Resize((256, 512)),  # 高度256，宽度512
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道
])

val_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

import torch.nn as nn
from torchvision.models import resnet34


class CClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改base网络为仅保留特征提取部分
        self.base = nn.Sequential(
            *list(resnet34(pretrained=True).children())[:-2]
        )
        self.base[0] = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.se = SEModule(512)
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 调整前向传播顺序
        features = self.base(x)  # 输出形状 [batch, 512, H, W]
        features = self.se(features)
        return self.head(features.mean([2, 3]))  # 全局平均池化



from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit


def main():
    # 加载数据（修改自UnetVessel.py的load_data）
    df = pd.read_csv("../C_resnet/dataset/full_df.csv")
    df = df[['ID', 'C']].drop_duplicates()

    # 数据集划分
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['ID']))
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # 计算类别权重
    pos_weight = torch.tensor([(len(train_df) - train_df['C'].sum()) / train_df['C'].sum()])

    # 初始化模型和训练配置
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = CClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # 早停配置
    best_val_loss = float('inf')
    early_stop_counter = 0
    EARLY_STOP_PATIENCE = 5


    # 训练循环
    for epoch in range(50):
        # 训练阶段
        model.train()
        train_loss = 0
        for imgs, labels in DataLoader(CMaskDataset(train_df, train_transform),
                                       batch_size=16, shuffle=True):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in DataLoader(CMaskDataset(val_df, val_transform),
                                           batch_size=16):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        train_loss = train_loss / len(train_df)
        val_loss = val_loss / len(val_df)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        # 学习率调整
        scheduler.step(val_loss)

        # 早停与模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_c_classifier.pth')
        else:
            early_stop_counter += 1

        # 打印日志
        print(f"Epoch {epoch + 1}/50")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    main()
