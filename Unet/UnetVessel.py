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
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, hamming_loss
from tqdm import tqdm

# -------------------- 配置参数 --------------------
DATA_ROOT = "../Archive/preprocessed_images"
CSV_PATH = "../Archive/full_df.csv"
IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_CLASSES = 8
CLASS_NAMES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'][:NUM_CLASSES]
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
EARLY_STOP_PATIENCE = 5  # 新增早停耐心参数

# -------------------- 数据准备 --------------------
def load_data():
    """加载数据并生成包含双眼路径的DataFrame"""
    df = pd.read_csv(CSV_PATH)
    df['ID'] = df['ID'].astype(str)

    # 生成双眼路径
    data = []
    for _, row in df.iterrows():
        eye_id = row['ID']
        left_path = os.path.join(DATA_ROOT, f"{eye_id}_left.jpg")
        right_path = os.path.join(DATA_ROOT, f"{eye_id}_right.jpg")

        # 仅保留同时存在双眼的数据
        if os.path.exists(left_path) and os.path.exists(right_path):
            data.append({
                "ID": eye_id,
                "left_path": left_path,
                "right_path": right_path,
                **{col: row[col] for col in CLASS_NAMES}
            })
    return pd.DataFrame(data)

# -------------------- 模型定义 --------------------
class SEModule(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SiameseUNetPlusPlus(nn.Module):
    """双流网络联合分析双眼"""
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = smp.UnetPlusPlus(
            encoder_name="resnet34",
            in_channels=4,
            classes=1
        ).encoder
        self.se = SEModule(512)  # 新增SE注意力模块
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, left_imgs, right_imgs):
        left_feat = self.encoder(left_imgs)[-1]
        left_feat = self.se(left_feat)  # 应用注意力
        left_feat = self.gap(left_feat).flatten(1)

        right_feat = self.encoder(right_imgs)[-1]
        right_feat = self.se(right_feat)  # 应用注意力
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

        # 新增：生成掩膜路径
        def get_mask_path(img_path):
            base_dir = os.path.dirname(os.path.dirname(img_path))  # 获取父目录
            return os.path.join(base_dir, "mask", os.path.basename(img_path).replace(".jpg", "_mask.jpg"))

        # 修改：加载原图和掩膜
        left_img = Image.open(row['left_path']).convert('RGB')
        left_mask = Image.open(get_mask_path(row['left_path'])).convert('L')  # 单通道掩膜
        right_img = Image.open(row['right_path']).convert('RGB')
        right_mask = Image.open(get_mask_path(row['right_path'])).convert('L')

        if self.transform:
            # 分别处理RGB和掩膜
            rgb_transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            mask_transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
            ])

            # 应用同步的空间变换
            seed = torch.randint(0, 2**32, ())

            # 处理左眼
            torch.manual_seed(seed)
            left_img = rgb_transform(left_img)
            torch.manual_seed(seed)
            left_mask = mask_transform(left_mask)

            # 处理右眼
            torch.manual_seed(seed)
            right_img = rgb_transform(right_img)
            torch.manual_seed(seed)
            right_mask = mask_transform(right_mask)

            # 合并通道
            left_img = torch.cat([left_img, left_mask], dim=0)
            right_img = torch.cat([right_img, right_mask], dim=0)

        label = torch.tensor([row[col] for col in CLASS_NAMES], dtype=torch.float32)
        return left_img, right_img, label

# -------------------- 指标计算 --------------------
def calculate_metrics(preds, labels, threshold=0.5):
    """计算多标签分类指标"""
    binary_preds = (preds > threshold).astype(int)
    return {
        'mAUC': roc_auc_score(labels, preds, average='macro'),
        'macro_F1': f1_score(labels, binary_preds, average='macro'),
        'accuracy': accuracy_score(labels, binary_preds),
        'hamming_loss': hamming_loss(labels, binary_preds)
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

    # 类别权重计算
    pos_counts = train_df[CLASS_NAMES].sum(axis=0).values
    total_samples = len(train_df)
    pos_weights = (total_samples - pos_counts) / (pos_counts + 1e-6)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)

    # 训练配置
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    # 早停机制
    best_auc = 0.0
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
                val_preds.append(outputs.sigmoid().cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_metrics = calculate_metrics(val_preds, val_labels)

        # 更新学习率
        scheduler.step(val_metrics['mAUC'])

        # 保存最佳模型
        if val_metrics['mAUC'] > best_auc:
            best_auc = val_metrics['mAUC']
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("✨ 发现新最佳模型")
        else:
            early_stop_counter += 1

        # 打印验证指标
        print(f"\nEpoch {epoch+1} 验证指标:")
        print(f"mAUC: {val_metrics['mAUC']:.4f} | F1: {val_metrics['macro_F1']:.4f}")
        print(f"Accuracy: {val_metrics['accuracy']:.4f} | Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        print(f"当前最佳mAUC: {best_auc:.4f} (Epoch {best_epoch+1})")

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
            test_preds.append(outputs.sigmoid().cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)
    test_metrics = calculate_metrics(test_preds, test_labels)

    print("\n最终测试结果:")
    print(f"mAUC: {test_metrics['mAUC']:.4f} | F1: {test_metrics['macro_F1']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f} | Hamming Loss: {test_metrics['hamming_loss']:.4f}")


if __name__ == '__main__':
    # 在代码中添加设备验证
    print(f"当前使用设备: {DEVICE}")
    print(f"Metal可用性: {torch.backends.mps.is_available()}")
    print(f"CUDA可用性: {torch.cuda.is_available()}")
    main()