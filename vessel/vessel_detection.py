import os
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm


class VesselDataset(Dataset):
    def __init__(self, root_dir, img_size=512, transform=None):
        self.root = root_dir
        self.img_size = img_size
        self.transform = transform

        # 新增：只保留1-40的图片文件
        all_files = os.listdir(os.path.join(root_dir, "Fundus_Images"))
        self.image_files = sorted([
            f for f in all_files
            if f.split('.')[0].isdigit() and 1 <= int(f.split('.')[0]) <= 40
        ], key=lambda x: int(x.split('.')[0]))  # 按数字顺序排序

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 使用实际存在的文件名解析ID
        file_name = self.image_files[idx]
        base_name = os.path.splitext(file_name)[0]

        # 构建文件路径（移除mask_path）
        img_path = os.path.join(self.root, "Fundus_Images_Preprocess", file_name)
        vessel_path = os.path.join(self.root, "Manual_Segmentations_Preprocess", f"{base_name}_manual_orig.png")

        # 读取数据（仅保留图像和血管标注）
        image = cv2.imread(img_path)
        vessel_mask = cv2.imread(vessel_path, 0)  # 血管标注

        # 校验逻辑（移除ROI掩码检查）
        if image is None:
            raise FileNotFoundError(f"无法读取眼底图像: {img_path}")
        if vessel_mask is None:
            raise FileNotFoundError(f"无法读取血管标注: {vessel_path}")


        # 数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=vessel_mask)
            image = transformed["image"]
            vessel_mask = transformed["mask"]

        # 转换为张量
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        vessel_mask = torch.from_numpy(vessel_mask).float().unsqueeze(0) / 255.0

        return image, vessel_mask


train_transform = A.Compose([
    A.Resize(512, 512),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomBrightnessContrast(p=0.5),
    A.CLAHE(p=0.3),
    A.GaussNoise(p=0.2),
])



class VesselSegmentor(nn.Module):
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
    # 初始化
    device = torch.device("mps")
    model = VesselSegmentor().to(device)
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 创建完整数据集
    full_dataset = VesselDataset(
        root_dir="../DRHAGIS",
        transform=None  # 原始数据集不应用增强
    )

    # 分割训练集和验证集 (9:1)
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    # 创建子集
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    # 克隆transform对象避免共享参数
    train_transform_clone = A.Compose([*train_transform.transforms])

    # 应用不同transform
    train_subset.dataset.transform = train_transform_clone
    val_subset.dataset.transform = A.Resize(512, 512)  # 验证集只需resize

    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

    # 初始化最佳损失
    best_val_loss = float('inf')

    # 训练循环
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

        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_vessel_model.pth")
            print(f"Epoch {epoch}: 保存最佳模型，验证损失: {avg_val_loss:.4f}")



if __name__ == "__main__":
    main()