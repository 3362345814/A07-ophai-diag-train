import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import seaborn as sns

from config import ROOT_DIR
from optic_disk.optic_disk_detection import OpticDiscSegmentor
from vessel.vessel_detection import VesselSegmentor

# 配置参数
RANDOM_SEED = 42  # 随机种子
TEST_SIZE = 0.2  # 验证集比例
PATIENCE = 5  # 早停计数器
VESSEL_MODEL_PATH = ROOT_DIR / 'vessel/best_vessel_model.pth'
OPTIC_DISC_MODEL_PATH = ROOT_DIR / 'optic_disk/best_disk_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')


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


# 修改数据加载函数
def load_dataset():
    """加载糖尿病视网膜病变数据集"""
    # 加载标签文件
    labels_path = ROOT_DIR / 'dataset/Diabetes/trainLabels.csv'
    labels_df = pd.read_csv(labels_path)

    # 构建图像路径
    base_dir = ROOT_DIR / 'dataset/Diabetes/preprocess'
    data = []


    for _, row in labels_df.iterrows():
        # 根据实际文件后缀调整（如.jpg/.jpeg/.png等）
        img_path = base_dir / f"{row['image']}.jpeg"  # 假设图像文件后缀为jpeg

        # 验证文件是否存在
        if not img_path.exists():
            continue

        # 转换标签：0为健康，1-4为患病
        label = 0 if row['level'] == 0 else 1


        data.append({
            'path': img_path,
            'label': label,
            'original_level': row['level']  # 保留原始分级信息
        })
    return pd.DataFrame(data)

class DiabeticDataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.df = df
        self.is_train = is_train

        # 初始化血管和视盘模型
        self.vessel_model = self._load_vessel_model()
        self.disk_model = self._load_disk_model()

        # 修改transform处理3通道输入
        self.base_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            self.base_transform
        ])

        self.transform = transform if transform else (
            self.train_transform if is_train else self.base_transform
        )

    def _load_vessel_model(self):
        model = VesselSegmentor().to(DEVICE)
        model.load_state_dict(torch.load(VESSEL_MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model

    def _load_disk_model(self):
        model = OpticDiscSegmentor().to(DEVICE)
        model.load_state_dict(torch.load(OPTIC_DISC_MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model

    def _get_channels(self, img_path):
        # 通道1：灰度图
        gray = Image.open(img_path).convert('L').resize((224, 224))

        # 通道2：血管分割
        vessel_mask = predict_vessels(self.vessel_model, img_path, DEVICE)

        # 通道3：视盘分割
        disk_mask = predict_disk(self.disk_model, img_path, DEVICE)

        gray = np.array(gray)




        # 合并通道
        combined = np.stack([gray, vessel_mask, disk_mask], axis=-1)
        return Image.fromarray(combined)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_path = str(item['path'])

        # 生成三通道图像
        img = self._get_channels(img_path)
        img = self.transform(img)

        # 单独标准化每个通道
        img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        return img, item['label']

    def __len__(self):
        return len(self.df)

# 添加预测函数
def predict_vessels(model, img_path, device):
    # 预处理
    img = Image.open(img_path).convert('RGB')
    img = remove_black_borders(img)
    img = img.resize((512, 512))

    # 转换为模型输入
    input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 后处理
    mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    img = img.convert('L')
    img = np.array(img)
    img[mask == 0] = [0]  # 黑色背景
    img = np.array(Image.fromarray(img).resize((224, 224)))

    return img

def predict_disk(model, img_path, device):
    # 预处理
    img = Image.open(img_path).convert('RGB')
    img = remove_black_borders(img)
    img = img.resize((512, 512))

    # 转换为模型输入
    input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 后处理
    mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # 应用到原图
    img = np.array(img)
    img[mask != 0] = [0, 0, 0]  # 黑色背景
    img = remove_black_borders(Image.fromarray(img)).resize((224, 224))
    img = np.array(img.convert('L'))

    return img

# 修改模型输入通道
class CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 直接使用原始resnet50
        self.base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # 仅修改最后的全连接层
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base.fc.in_features, 2)
        )

    def forward(self, x):
        return self.base(x)

def main():
    # 加载完整数据集
    full_df = load_dataset()

    # 显示数据分布
    print("原始标签分布:")
    print(full_df['original_level'].value_counts().sort_index())
    print("\n二分类分布:")
    print(full_df['label'].value_counts())

    # 分层分割数据集
    train_df, val_df = train_test_split(
        full_df,
        test_size=TEST_SIZE,
        stratify=full_df['label'],
        random_state=RANDOM_SEED
    )

    # 创建带数据增强的数据集
    train_dataset = DiabeticDataset(train_df, is_train=True)
    val_dataset = DiabeticDataset(val_df, is_train=False)

    # 计算类别权重
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"\n类别权重: {class_weights.numpy()}")

    # 数据加载器
    BATCH_SIZE = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 初始化模型
    model = CustomResNet()

    # 训练配置
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


    # 初始化指标存储列表
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []

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

        # 每个epoch结束后记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)


        # 早停逻辑
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'D_best.pth')
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

    # 训练结束后绘制图表
    # ----------------------------------
    # 加载最佳模型
    model.load_state_dict(torch.load('D_best.pth'))
    model.eval()

    # 收集验证集的预测结果
    all_probs = []
    all_preds = []
    all_true = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    # 1. 绘制训练曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validate Loss')
    plt.title('Training and Validating Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Accuracy')
    plt.plot(val_f1_scores, label='F1-score')
    plt.title('Accuracy and F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 2. 绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(all_true, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    # 3. 绘制混淆矩阵
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Health', 'Ill'],
                yticklabels=['Health', 'Ill'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()