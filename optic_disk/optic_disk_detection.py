import os
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm

from config import ROOT_DIR


class OpticDiscDataset(Dataset):
    def __init__(self, root_dir, img_size=512, transform=None):
        self.root = root_dir
        self.img_size = img_size
        self.transform = transform

        # 获取所有眼底图像文件
        image_dir = os.path.join(root_dir, "fundus_image")
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        # 验证掩码文件是否存在
        self.valid_files = []
        for f in self.image_files:
            base_name = os.path.splitext(f)[0]
            mask_path = os.path.join(root_dir, "Disc_Masks", f"{base_name}.bmp")
            if os.path.exists(mask_path):
                self.valid_files.append(f)
            else:
                print(f"Warning: 缺少掩码文件 {base_name}.bmp")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_name = self.valid_files[idx]
        base_name = os.path.splitext(file_name)[0]

        # 构建文件路径
        img_path = os.path.join(self.root, "fundus_image", file_name)
        mask_path = os.path.join(self.root, "Disc_Masks", f"{base_name}.bmp")

        # 读取数据
        image = cv2.imread(img_path)
        disc_mask = cv2.imread(mask_path, 0)  # 单通道读取

        # 校验数据
        if image is None:
            raise FileNotFoundError(f"无法读取眼底图像: {img_path}")
        if disc_mask is None:
            raise FileNotFoundError(f"无法读取视盘掩码: {mask_path}")

        # 数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=disc_mask)
            image = transformed["image"]
            disc_mask = transformed["mask"]

        # 转换为张量
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        disc_mask = torch.from_numpy(disc_mask).float().unsqueeze(0) / 255.0

        return image, disc_mask


# 数据增强（保持与血管分割相同）
train_transform = A.Compose([
    A.Resize(512, 512),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomBrightnessContrast(p=0.5),
    A.CLAHE(p=0.3),
    A.GaussNoise(p=0.2),
])


class OpticDiscSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )

    def forward(self, x):
        return self.model(x)


def main():
    device = torch.device("mps")
    model = OpticDiscSegmentor().to(device)
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 创建完整数据集
    full_dataset = OpticDiscDataset(
        root_dir=ROOT_DIR / "dataset/optic_disk/Train",
        transform=None
    )

    # 分割训练集和验证集
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    # 创建子集并应用变换
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_subset.dataset.transform = A.Compose([*train_transform.transforms])
    val_subset.dataset.transform = A.Resize(512, 512)

    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

    best_val_loss = float('inf')

    for epoch in range(50):
        model.train()
        # 训练阶段
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_masks).item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_disc_model.pth")
            print(f"Epoch {epoch}: 保存最佳模型，验证损失: {avg_val_loss:.4f}")


if __name__ == "__main__":
    main()
