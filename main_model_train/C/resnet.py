import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import ROOT_DIR

# 读取CSV文件并过滤有效样本
df = pd.read_csv(ROOT_DIR / 'dataset/Archive/full_df.csv')
valid_data = []

for _, row in df.iterrows():
    id = row['ID']
    left_path = ROOT_DIR / f'dataset/Archive/vessel_mask/{id}_left_mask.jpg'
    right_path = ROOT_DIR / f'dataset/Archive/vessel_mask/{id}_right_mask.jpg'

    if os.path.exists(left_path) and os.path.exists(right_path):
        valid_data.append({
            'id': id,
            'label': row['C']
        })

valid_df = pd.DataFrame(valid_data)


# 自定义数据集类
class EyeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # 加载左右眼图像
        left_img = Image.open(ROOT_DIR / f'dataset/Archive/vessel_mask/{item["id"]}_left_mask.jpg')
        right_img = Image.open(ROOT_DIR / f'dataset/Archive/vessel_mask/{item["id"]}_right_mask.jpg')

        # 调整尺寸
        resize = transforms.Resize((512, 256))
        left_img = resize(left_img)
        right_img = resize(right_img)

        # 创建拼接图像
        combined = Image.new('L', (512, 512))
        combined.paste(left_img, (0, 0))
        combined.paste(right_img, (256, 0))

        # 转换为RGB三通道
        combined = combined.convert('RGB')

        if self.transform:
            combined = self.transform(combined)

        return combined, item['label']

def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 划分数据集后添加类别权重计算
    train_df, val_df = train_test_split(valid_df, test_size=0.2, random_state=42, stratify=valid_df['label'])

    # 计算类别权重
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"\n类别分布: {dict(train_df['label'].value_counts())}")
    print(f"类别权重: {class_weights.numpy()}")

    # 创建数据集和数据加载器
    train_dataset = EyeDataset(train_df, transform=transform)
    val_dataset = EyeDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # 初始化模型
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 二分类输出

    # 训练配置
    device = torch.device('mps')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # 添加权重
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    best_val_acc = 0
    early_stop_counter = 0
    PATIENCE = 5
    for epoch in range(20):
        # 训练阶段
        model.train()
        train_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            inputs = inputs.to(device)
            labels = labels.to(device).long()  # 确保标签是长整型

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

        # 计算多种指标
        val_loss /= len(val_dataset)
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        report = classification_report(all_labels, all_preds,
                                       target_names=['健康', '患病'],
                                       output_dict=True)

        # 早停逻辑（基于F1-score）
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'C_best.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"\n早停触发！最佳验证F1-score: {best_f1:.4f}")
                break

        # 详细指标输出
        print(f"\nEpoch {epoch + 1:02}")
        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        print(f"准确率: {val_acc:.4f} | F1-score: {val_f1:.4f}")
        print(f"患病召回率: {report['患病']['recall']:.4f} | 患病精确率: {report['患病']['precision']:.4f}")
        print(classification_report(all_labels, all_preds,
                                    target_names=['健康', '患病'],
                                    zero_division=0))

    print(f"\n最佳验证F1-score: {best_f1:.4f}")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()