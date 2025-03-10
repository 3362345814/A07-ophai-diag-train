import pandas as pd
import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 配置参数
TRAIN_LABEL_FILE = '../dataset/AMD/label.csv'
IMAGE_DIR = '../dataset/AMD/OriginalImages'
RANDOM_SEED = 42
TEST_SIZE = 0.2


class DiabeticDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.resize = transforms.Resize((256, 512))  # 调整单眼图片尺寸

        # 预处理索引
        self.healthy_indices = []
        self.all_indices = []
        for idx in range(len(df)):
            if df.iloc[idx]['AMD'] == 0:
                self.healthy_indices.append(idx)
            self.all_indices.append(idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取当前样本
        current_row = self.df.iloc[idx]
        current_img = Image.open(os.path.join(self.image_dir, current_row['fnames'])).convert('RGB')
        current_img = self.resize(current_img)
        current_label = current_row['AMD']

        # 选择配对样本
        if current_label == 1:
            candidates = [i for i in self.all_indices if i != idx]
        else:
            candidates = [i for i in self.healthy_indices if i != idx]

        pair_idx = random.choice(candidates) if candidates else idx

        # 加载配对样本
        pair_row = self.df.iloc[pair_idx]
        pair_img = Image.open(os.path.join(self.image_dir, pair_row['fnames'])).convert('RGB')
        pair_img = self.resize(pair_img)

        # 拼接双眼图像
        combined = Image.new('RGB', (512, 512))
        combined.paste(current_img, (0, 0))  # 左眼
        combined.paste(pair_img, (256, 0))  # 右眼

        if self.transform:
            combined = self.transform(combined)

        return combined, current_label


def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据
    train_df = pd.read_csv(TRAIN_LABEL_FILE)

    # 分割训练验证集
    train_df, val_df = train_test_split(
        train_df,
        test_size=TEST_SIZE,
        stratify=train_df['AMD'],
        random_state=RANDOM_SEED
    )

    print(f"训练样本: {len(train_df)}, 验证样本: {len(val_df)}")

    # 创建数据集
    train_dataset = DiabeticDataset(train_df, IMAGE_DIR, transform)
    val_dataset = DiabeticDataset(val_df, IMAGE_DIR, transform)

    # 数据加载器
    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 初始化模型
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2)
    )

    # 训练配置
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 训练循环
    best_acc = 0
    for epoch in range(20):
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

        # 验证
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        # 计算指标
        train_loss = train_loss / len(train_dataset)
        val_loss = val_loss / len(val_dataset)
        val_acc = correct / len(val_dataset)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'amd_best.pth')

        print(f"Epoch {epoch + 1:02}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
