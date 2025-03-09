import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# -------------------- 配置参数 --------------------
DATA_ROOT = "Archive/preprocessed_images"
CSV_PATH = "../C_resnet/dataset/full_df.csv"
IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 50
CLASS_NAMES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']  # 假设这是所有可能的类别
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOP_PATIENCE = 5

# -------------------- 数据准备 --------------------
def load_data():
    """加载数据并生成包含双眼路径和类别标签的DataFrame"""
    df = pd.read_csv(CSV_PATH)
    df['ID'] = df['ID'].astype(str)

    # 创建类别映射字典
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}

    data = []
    for _, row in df.iterrows():
        eye_id = row['ID']
        left_path = os.path.join(DATA_ROOT, f"{eye_id}_left.jpg")
        right_path = os.path.join(DATA_ROOT, f"{eye_id}_right.jpg")

        if os.path.exists(left_path) and os.path.exists(right_path):
            # 解析标签（假设labels列格式为['D']这样的列表字符串）
            label_str = row['labels'].strip("[]'")
            label = class_to_idx[label_str]

            data.append({
                "ID": eye_id,
                "left_path": left_path,
                "right_path": right_path,
                "label": label
            })
    return pd.DataFrame(data)

# -------------------- 模型定义 --------------------
class SiameseUNetPlusPlus(nn.Module):
    """双流网络联合分析双眼（多分类版）"""
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = smp.UnetPlusPlus(encoder_name="resnet34", classes=1).encoder
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * 2, num_classes)  # 输出维度为类别数

    def forward(self, left_imgs, right_imgs):
        left_feat = self.encoder(left_imgs)[-1]
        left_feat = self.gap(left_feat).flatten(1)

        right_feat = self.encoder(right_imgs)[-1]
        right_feat = self.gap(right_feat).flatten(1)

        combined = torch.cat([left_feat, right_feat], dim=1)
        return self.fc(combined)

# -------------------- 数据集类 --------------------
class EyeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        left_img = Image.open(row['left_path']).convert('RGB')
        right_img = Image.open(row['right_path']).convert('RGB')

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        label = torch.tensor(row['label'], dtype=torch.long)  # 改为long类型
        return left_img, right_img, label

# -------------------- 指标计算 --------------------
def calculate_metrics(preds, labels):
    """计算多分类指标"""
    return {
        'accuracy': accuracy_score(labels, preds),
        'macro_F1': f1_score(labels, preds, average='macro'),
        'mAUC': roc_auc_score(pd.get_dummies(labels), preds, multi_class='ovo')  # 多分类AUC
    }

# -------------------- 训练函数 --------------------
def main():
    torch.multiprocessing.freeze_support()
    full_df = load_data()
    print(f"总样本数: {len(full_df)}")

    # 数据集划分
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valtest_idx = next(gss.split(full_df, groups=full_df['ID']))
    train_df = full_df.iloc[train_idx]
    valtest_df = full_df.iloc[valtest_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val.split(valtest_df, groups=valtest_df['ID']))
    val_df = valtest_df.iloc[val_idx]
    test_df = valtest_df.iloc[test_idx]

    # 数据转换
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = EyeDataset(train_df, train_transform)
    val_dataset = EyeDataset(val_df, val_transform)
    test_dataset = EyeDataset(test_df, val_transform)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # 初始化模型
    model = SiameseUNetPlusPlus(num_classes=NUM_CLASSES).to(DEVICE)

    # 类别权重计算（处理不平衡数据）
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1. / (class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    # 训练配置
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 改用交叉熵损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    # 早停机制
    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        # 训练阶段
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for left_imgs, right_imgs, labels in pbar:
            left_imgs, right_imgs = left_imgs.to(DEVICE), right_imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(left_imgs, right_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * left_imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 验证阶段
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for left_imgs, right_imgs, labels in val_loader:
                left_imgs, right_imgs = left_imgs.to(DEVICE), right_imgs.to(DEVICE)
                outputs = model(left_imgs, right_imgs)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_metrics = calculate_metrics(np.array(val_preds), np.array(val_labels))

        # 更新学习率
        scheduler.step(val_metrics['accuracy'])

        # 保存最佳模型
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("✨ 发现新最佳模型")
        else:
            early_stop_counter += 1

        # 打印验证指标
        print(f"\nEpoch {epoch+1} 验证指标:")
        print(f"Accuracy: {val_metrics['accuracy']:.4f} | Macro F1: {val_metrics['macro_F1']:.4f}")
        print(f"mAUC: {val_metrics['mAUC']:.4f}")
        print(f"当前最佳Accuracy: {best_acc:.4f} (Epoch {best_epoch+1})")

        # 早停判断
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\n早停触发! 连续{EARLY_STOP_PATIENCE}个epoch未提升")
            break

    # 最终测试
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for left_imgs, right_imgs, labels in test_loader:
            left_imgs, right_imgs = left_imgs.to(DEVICE), right_imgs.to(DEVICE)
            outputs = model(left_imgs, right_imgs)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_metrics = calculate_metrics(np.array(test_preds), np.array(test_labels))

    print("\n最终测试结果:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_F1']:.4f}")
    print(f"mAUC: {test_metrics['mAUC']:.4f}")

if __name__ == '__main__':
    main()