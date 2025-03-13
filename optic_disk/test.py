import torch
import cv2
import numpy as np
from albumentations import Compose, Resize
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import ROOT_DIR


class OpticDiscSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,  # We'll load pretrained weights manually
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )

    def forward(self, x):
        return self.model(x)


def predict_and_save(image_path, model_path, output_path, device='auto'):
    # 设备设置
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    else:
        device = torch.device(device)

    # 加载模型
    model = OpticDiscSegmentor()
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"加载模型失败: {str(e)}")
        return
    model = model.to(device)
    model.eval()

    # 读取图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 记录原始尺寸
    original_height, original_width = original_image.shape[:2]

    # 预处理流程（与验证集保持一致）
    transform = Compose([
        Resize(512, 512),  # 必须与训练尺寸一致
    ])

    # 应用预处理
    transformed = transform(image=original_image)
    processed_image = transformed["image"]

    # 转换为张量并归一化
    input_tensor = torch.from_numpy(processed_image).float()
    input_tensor = input_tensor.permute(2, 0, 1) / 255.0  # HWC -> CHW 并归一化
    input_tensor = input_tensor.unsqueeze(0).to(device)  # 添加batch维度

    # 推理
    with torch.no_grad():
        prediction = model(input_tensor)

    # 后处理
    mask = prediction.squeeze().cpu().numpy()  # 转换为numpy数组
    mask = (mask > 0.5).astype(np.uint8) * 255  # 二值化并转换为0-255

    # 恢复到原始图像尺寸
    resized_mask = cv2.resize(mask, (original_width, original_height),
                              interpolation=cv2.INTER_NEAREST)

    # 保存结果
    cv2.imwrite(output_path, resized_mask)


if __name__ == "__main__":
    # 使用示例
    predict_and_save(
        image_path=ROOT_DIR / "dataset/optic_disk/PALM-Testing400-Images/T0012.jpg",  # 输入图像路径
        model_path="best_disc_model.pth",  # 训练好的模型路径
        output_path="predicted_mask.jpg"  # 输出掩码路径
    )
