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
from config import ROOT_DIR

# 配置参数
TRAIN_LABEL_FILE = ROOT_DIR / 'dataset/AMD/label.csv'
ORIGINAL_IMAGE_DIR = ROOT_DIR / 'dataset/AMD/OriginalImages'
OPTIC_DISK_DIR = ROOT_DIR / 'dataset/AMD/optic_disk'
VESSEL_DIR = ROOT_DIR / 'dataset/AMD/vessel'
RANDOM_SEED = 42
TEST_SIZE = 0.2
PATIENCE = 5


class DiabeticDataset(Dataset):
    def __init__(self, df, original_dir, optic_dir, vessel_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.original_dir = original_dir
        self.optic_dir = optic_dir
        self.vessel_dir = vessel_dir
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
        filename = current_row['fnames']

        # 加载三通道图像
        original_img = Image.open(self.original_dir / filename).convert('L')
        optic_img = Image.open(self.optic_dir / filename).convert('L')
        vessel_img = Image.open(self.vessel_dir / filename).convert('L')

        # 调整尺寸
        original_img = self.resize(original_img)
        optic_img = self.resize(optic_img)
        vessel_img = self.resize(vessel_img)

        # 合并三通道
        combined = Image.merge("RGB", (original_img, optic_img, vessel_img))

        # 选择配对样本
        current_label = current_row['AMD']
        if current_label == 1:
            candidates = [i for i in self.all_indices if i != idx]
        else:
            candidates = [i for i in self.healthy_indices if i != idx]

        pair_idx = random.choice(candidates) if candidates else idx

        # 加载配对样本的三通道图像
        pair_row = self.df.iloc[pair_idx]
        pair_filename = pair_row['fnames']

        pair_original = Image.open(self.original_dir / pair_filename).convert('L')
        pair_optic = Image.open(self.optic_dir / pair_filename).convert('L')
        pair_vessel = Image.open(self.vessel_dir / pair_filename).convert('L')

        pair_original = self.resize(pair_original)
        pair_optic = self.resize(pair_optic)
        pair_vessel = self.resize(pair_vessel)

        # 拼接双眼图像
        final_image = Image.new('RGB', (512, 512))

        # 左眼（当前样本）
        left_eye = Image.merge("RGB", (original_img, optic_img, vessel_img))
        final_image.paste(left_eye, (0, 0))

        # 右眼（配对样本）
        right_eye = Image.merge("RGB", (pair_original, pair_optic, pair_vessel))
        final_image.paste(right_eye, (256, 0))

        if self.transform:
            final_image = self.transform(final_image)

        return final_image, current_label


def main():
    # 数据预处理（保持原有归一化参数）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据
    train_df = pd.read_csv(TRAIN_LABEL_FILE)
    train_df['AMD'] = train_df['AMD'].apply(lambda x: 1 if x != 0 else 0)

    # 分割训练验证集
    train_df, val_df = train_test_split(
        train_df,
        test_size=TEST_SIZE,
        stratify=train_df['AMD'],
        random_state=RANDOM_SEED
    )

    # 创建数据集
    train_dataset = DiabeticDataset(
        train_df,
        ORIGINAL_IMAGE_DIR,
        OPTIC_DISK_DIR,
        VESSEL_DIR,
        transform
    )

    val_dataset = DiabeticDataset(
        val_df,
        ORIGINAL_IMAGE_DIR,
        OPTIC_DISK_DIR,
        VESSEL_DIR,
        transform
    )

    # 剩余代码保持不变...
    # ...（数据加载器、模型初始化、训练循环等部分与原始代码相同）


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
