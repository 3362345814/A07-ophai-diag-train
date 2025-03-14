import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

from config import ROOT_DIR

# 配置参数
RANDOM_SEED = 42  # 随机种子
TEST_SIZE = 0.2  # 验证集比例
PATIENCE = 5  # 早停计数器


def remove_black_borders(pil_img):
    def smart_retina_preprocessing(cv_img):
        """处理视网膜图像预处理（使用OpenCV）"""
        # 转换为numpy数组后尺寸访问方式
        h, w = cv_img.shape[:2]

        # 计算填充尺寸
        if h > w:
            top = bottom = 0
            left = right = (h - w) // 2
        else:
            top = bottom = (w - h) // 2
            left = right = 0

        # 添加黑色边框
        padded = cv2.copyMakeBorder(cv_img,
                                    top, bottom,
                                    left, right,
                                    cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])
        return padded

    # 将PIL Image转换为OpenCV格式（BGR）
    cv_img = np.array(pil_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # 执行预处理
    cv_img = smart_retina_preprocessing(cv_img)

    # 转换为灰度图并进行阈值处理
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓并找到最大轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img  # 如果没有找到轮廓返回原图
    cnt = max(contours, key=cv2.contourArea)

    # 获取边界矩形
    x, y, w, h = cv2.boundingRect(cnt)

    # 计算最大内接正方形
    square_size = max(w, h)
    if square_size < 10:
        return pil_img

    center_x = x + w // 2
    center_y = y + h // 2

    # 计算裁剪坐标
    crop_x1 = max(0, center_x - square_size // 2)
    crop_y1 = max(0, center_y - square_size // 2)
    crop_x2 = min(cv_img.shape[1], crop_x1 + square_size)
    crop_y2 = min(cv_img.shape[0], crop_y1 + square_size)

    # 执行裁剪
    cropped = cv_img[crop_y1:crop_y2, crop_x1:crop_x2]

    # 转换回PIL Image（需要转换颜色空间）
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)


# 数据集加载逻辑
def load_dataset():
    """加载并构建数据集"""
    # 路径设置
    glaucoma_dir = ROOT_DIR / 'dataset/C_D_G/glaucoma_optic_disk'
    normal_dir = ROOT_DIR / 'dataset/C_D_G/normal_optic_disk'

    # 收集样本路径
    glaucoma_samples = [f for f in glaucoma_dir.glob('*.jpg') if f.is_file()]
    normal_samples = [f for f in normal_dir.glob('*.jpg') if f.is_file()]

    # 构建DataFrame
    data = []
    for path in glaucoma_samples:
        data.append({'left_path': path, 'label': 1})
    for path in normal_samples:
        data.append({'left_path': path, 'label': 0})

    return pd.DataFrame(data)

# 修改后的数据集类
class DiabeticDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.resize = transforms.Resize((512, 256))

        # 按类别划分样本路径
        self.positive_samples = df[df['label'] == 1]['left_path'].tolist()
        self.negative_samples = df[df['label'] == 0]['left_path'].tolist()
        self.all_samples = df['left_path'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        left_path = item['left_path']
        label = item['label']

        # 加载左眼图像
        left_img = Image.open(left_path)
        left_img = remove_black_borders(left_img)
        left_img = self.resize(left_img)

        # 选择右眼图像
        if label == 1:  # 青光眼样本
            # 随机选择任意样本作为右眼
            right_path = random.choice(self.all_samples)
        else:  # 正常样本
            # 只从正常样本中随机选择
            right_path = random.choice(self.negative_samples)

        # 加载右眼图像
        right_img = Image.open(right_path)
        right_img = remove_black_borders(right_img)
        right_img = self.resize(right_img)

        # 创建拼接图像
        combined = Image.new('RGB', (512, 512))
        combined.paste(left_img, (0, 0))
        combined.paste(right_img, (256, 0))


        if self.transform:
            combined = self.transform(combined)

        return combined, label

def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载并分割数据集
    full_df = load_dataset()
    train_df, val_df = train_test_split(
        full_df,
        test_size=TEST_SIZE,
        stratify=full_df['label'],
        random_state=RANDOM_SEED
    )

    print(f"总样本数: {len(full_df)}")
    print(f"训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(val_df)}")
    print("训练集类别分布:\n", train_df['label'].value_counts())
    print("验证集类别分布:\n", val_df['label'].value_counts())

    # 计算类别权重
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"\n类别权重: {class_weights.numpy()}")

    # 创建数据集
    train_dataset = DiabeticDataset(train_df, transform)
    val_dataset = DiabeticDataset(val_df, transform)

    # 数据加载器
    BATCH_SIZE = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )

    # 训练配置
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 训练循环
    best_f1 = 0
    early_stop_counter = 0

    for epoch in range(100):
        # 训练阶段
        model.train()
        train_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        report = classification_report(all_labels, all_preds,
                                       target_names=['健康', '患病'],
                                       output_dict=True)

        # 早停逻辑
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'G_best.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"\n早停触发！最佳验证F1-score: {best_f1:.4f}")
                break

        # 打印指标
        print(f"\nEpoch {epoch + 1:02}")
        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        print(f"准确率: {val_acc:.4f} | F1-score: {val_f1:.4f}")
        print(f"患病召回率: {report['患病']['recall']:.4f} | 患病精确率: {report['患病']['precision']:.4f}")
        print(f"健康召回率: {report['健康']['recall']:.4f} | 健康精确率: {report['健康']['precision']:.4f}")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()