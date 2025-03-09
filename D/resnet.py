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

# 配置参数
TRAIN_LABELS = '../dataset/Diabetes/trainLabels.csv'
TEST_LABELS = '../dataset/Diabetes/testLabels.csv'
TRAIN_IMG_DIR = '../dataset/Diabetes/train'
TEST_IMG_DIR = '../dataset/Diabetes/test'


def process_labels(df):
    """处理标签生成二分类（0=健康，1=患病）"""
    # 分离ID和左右眼信息
    df[['id', 'eye']] = df['image'].str.split('_', n=1, expand=True)

    # 创建透视表
    pivoted = df.pivot(index='id', columns='eye', values='level').reset_index()
    pivoted.columns = ['id', 'left', 'right']

    # 处理缺失值（假设缺失的眼睛为健康）
    pivoted.fillna(0, inplace=True)

    # 生成二分类标签（任一眼睛level>0则为1）
    pivoted['label'] = ((pivoted['left'] > 0) | (pivoted['right'] > 0)).astype(int)
    return pivoted[['id', 'label']]


def load_dataset(label_path, img_dir):
    """加载并验证数据集"""
    raw_df = pd.read_csv(label_path)
    processed_df = process_labels(raw_df)

    valid_data = []
    for _, row in processed_df.iterrows():
        img_id = row['id']
        left_path = os.path.join(img_dir, f"{img_id}_left.jpeg")
        right_path = os.path.join(img_dir, f"{img_id}_right.jpeg")

        if os.path.exists(left_path) and os.path.exists(right_path):
            valid_data.append({
                'id': img_id,
                'label': row['label']
            })

    return pd.DataFrame(valid_data)


# 自定义数据集类
class DiabeticDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.resize = transforms.Resize((256, 512))  # 统一调整尺寸

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # 加载左右眼图像
        left_img = Image.open(os.path.join(self.img_dir, f"{item['id']}_left.jpeg"))
        right_img = Image.open(os.path.join(self.img_dir, f"{item['id']}_right.jpeg"))

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

    # 加载数据集
    train_df = load_dataset(TRAIN_LABELS, TRAIN_IMG_DIR)
    test_df = load_dataset(TEST_LABELS, TEST_IMG_DIR)

    print(f"训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")
    print("训练集类别分布:\n", train_df['label'].value_counts())
    print("测试集类别分布:\n", test_df['label'].value_counts())

    # 创建数据集
    train_dataset = DiabeticDataset(train_df, TRAIN_IMG_DIR, transform)
    test_dataset = DiabeticDataset(test_df, TEST_IMG_DIR, transform)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

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
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        # 计算指标
        train_loss /= len(train_loader.dataset)
        val_loss /= len(test_loader.dataset)
        val_acc = correct / len(test_loader.dataset)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'diabetic_best.pth')

        print(f"Epoch {epoch + 1:02}")
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
