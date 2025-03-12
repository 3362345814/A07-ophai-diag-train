import pandas as pd
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from config import ROOT_DIR

# 配置参数
CLASS_NAMES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']  # 按实际列名顺序
ID_COLUMN = 'ID'  # 患者ID列名
IMAGE_DIR = ROOT_DIR / 'dataset/Archive/preprocessed_images'


# 标签处理函数
def get_label(row):
    """从多列二值标记中获取单一标签"""
    for idx, cls in enumerate(CLASS_NAMES):
        if row[cls] == 1:
            return idx
    return None  # 无效数据


# 数据预处理类
class EyeDataset(Dataset):
    def __init__(self, df, img_size=(256, 512), transform=None):
        self.df = df
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_id = item[ID_COLUMN]
        label = item['Label']

        # 加载图像
        def load_hsv(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # (宽, 高)
            return img

        left_img = load_hsv(f"{IMAGE_DIR}/{img_id}_left.jpg")
        right_img = load_hsv(f"{IMAGE_DIR}/{img_id}_right.jpg")

        # 拼接图像
        combined = np.concatenate([left_img, right_img], axis=1)

        # 转换为Tensor
        combined = torch.from_numpy(combined).permute(2, 0, 1).float()
        combined[0] /= 179.0  # H通道归一化
        combined[1:] /= 255.0  # S/V通道归一化

        if self.transform:
            combined = self.transform(combined)

        return combined, label


# 数据增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载并预处理数据
df = pd.read_csv('../../dataset/Archive/full_df.csv')

# 过滤有效样本
valid_data = []
for _, row in df.iterrows():
    try:
        # 获取标签
        label = get_label(row)
        if label is None:
            continue

        # 验证图像存在
        img_id = row[ID_COLUMN]
        left_path = f"{IMAGE_DIR}/{img_id}_left.jpg"
        right_path = f"{IMAGE_DIR}/{img_id}_right.jpg"
        if not (os.path.exists(left_path) and os.path.exists(right_path)):
            continue

        # 记录有效数据
        valid_data.append({
            ID_COLUMN: img_id,
            'Label': label
        })
    except Exception as e:
        print(f"处理样本{row[ID_COLUMN]}时出错: {str(e)}")

valid_df = pd.DataFrame(valid_data)


# 划分数据集
train_df, val_df = train_test_split(
    valid_df,
    test_size=0.2,
    stratify=valid_df['Label'],
    random_state=42
)


# 创建数据加载器
def create_loader(df, transform, batch_size=8):
    dataset = EyeDataset(df, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=transform is not None,
        num_workers=2,
        pin_memory=True
    )


train_loader = create_loader(train_df, train_transform)
val_loader = create_loader(val_df, val_transform)


# 模型定义
def create_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 8)
    )
    return model


# 训练配置
device = torch.device('mps')
model = create_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)


# 训练循环
def main():
    best_acc = 0
    for epoch in range(50):
        # 训练阶段
        model.train()
        train_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_acc += (preds == labels).sum().item()

        # 计算指标
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        # 更新学习率
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_model_epoch{epoch + 1}_acc{val_acc:.4f}.pth')

        # 打印日志
        print(f"Epoch {epoch + 1:02}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")


if __name__ == '__main__':
    main()
