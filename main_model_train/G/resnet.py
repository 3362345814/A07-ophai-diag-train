import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import ROOT_DIR

# 配置参数
LABEL_FILE = ROOT_DIR / 'dataset/Archive/full_df.csv'
IMAGE_DIR = ROOT_DIR / 'dataset/Archive/optic_disk'
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


def process_labels(df):
    """处理标签生成二分类（0=健康，1=患病）"""

    return df[['ID', 'G']]


def load_dataset():
    """加载并验证数据集"""
    raw_df = pd.read_csv(LABEL_FILE)
    processed_df = process_labels(raw_df)

    valid_data = []
    for _, row in processed_df.iterrows():
        img_id = row['ID']
        left_path = os.path.join(IMAGE_DIR, f"{img_id}_left.jpg")
        right_path = os.path.join(IMAGE_DIR, f"{img_id}_right.jpg")

        if os.path.exists(left_path) and os.path.exists(right_path):
            valid_data.append({
                'id': img_id,
                'label': row['G']
            })
    return pd.DataFrame(valid_data)


# 自定义数据集类
class DiabeticDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.resize = transforms.Resize((256, 512))  # 统一调整尺寸

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # 加载左右眼图像
        left_img = Image.open(os.path.join(IMAGE_DIR, f"{item['id']}_left.jpg"))
        right_img = Image.open(os.path.join(IMAGE_DIR, f"{item['id']}_right.jpg"))

        left_img = remove_black_borders(left_img)
        right_img = remove_black_borders(right_img)

        # 调整尺寸并拼接
        left_img = self.resize(left_img.convert('RGB'))
        right_img = self.resize(right_img.convert('RGB'))

        combined = Image.new('RGB', (512, 512))
        combined.paste(left_img, (0, 0))
        combined.paste(right_img, (256, 0))

        if self.transform:
            combined = self.transform(combined)

        return combined, item['label']


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
        nn.Linear(num_features, 2)  # 二分类输出
    )

    # 训练配置
    device = torch.device('mps')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 训练循环
    best_acc = 0
    early_stop_counter = 0
    for epoch in range(20):
        # 训练阶段
        model.train()
        train_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        # 计算指标
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'G_best.pth')
        else:
            early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered")
            break

        print(f"Epoch {epoch + 1:02}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
