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

# 读取CSV文件并过滤有效样本
df = pd.read_csv('../dataset/Archive/full_df.csv')
valid_data = []

for _, row in df.iterrows():
    id = row['ID']
    left_path = f'../dataset/Archive/mask/{id}_left_mask.jpg'
    right_path = f'../dataset/Archive/mask/{id}_right_mask.jpg'

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
        left_img = Image.open(f'../dataset/Archive/mask/{item["id"]}_left_mask.jpg')
        right_img = Image.open(f'../dataset/Archive/mask/{item["id"]}_right_mask.jpg')

        # 调整尺寸
        resize = transforms.Resize((256, 512))
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

    # 划分训练集和验证集
    train_df, val_df = train_test_split(valid_df, test_size=0.2, random_state=42)

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    best_val_acc = 0
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
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # 计算指标
        train_loss = train_loss / len(train_dataset)
        val_loss = val_loss / len(val_dataset)
        val_acc = correct / total

        print(f'Epoch {epoch + 1:02}')
        print(f'Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}')
        print(f'Val Acc: {val_acc:.4f}\n')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Best Validation Accuracy: {best_val_acc:.4f}')


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()