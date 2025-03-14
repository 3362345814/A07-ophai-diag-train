import cv2
import numpy as np
import torch
from albumentations import Compose, Resize
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image
import warnings


from config import ROOT_DIR
from main_model_train.G.resnet import remove_black_borders
from optic_disk.optic_disk_detection import OpticDiscSegmentor

# 禁用PIL的DecompressionBombWarning警告
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

OPTIC_MODEL_PATH = ROOT_DIR / 'optic_disk/best_disk_model.pth'


def mask_predict(image_path, model_path, device='auto'):
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
    return resized_mask


def process_image(original_path, mask):
    # 1. rgb读取图像
    original = cv2.imread(original_path, cv2.IMREAD_COLOR_RGB)

    # 2. 二值化处理（确保掩码为0/255）
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. 反转掩码（使目标区域变为白色）
    inverted_mask = cv2.bitwise_not(binary_mask)

    # 4. 形态学膨胀（扩大目标区域）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))  # 椭圆核，尺寸根据需求调整
    dilated_mask = cv2.dilate(inverted_mask, kernel, iterations=1)

    # 6. 应用掩码（显示掩码黑色区域，隐藏白色区域）
    result = cv2.bitwise_and(original, original, mask=dilated_mask)

    return Image.fromarray(result)



def load_trained_model(model_path, device):
    """加载训练好的模型（修正版）"""
    # 保持与训练完全一致的结构
    model = resnet50(weights=None)
    num_features = model.fc.in_features

    # 必须包含与训练一致的Sequential结构
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )

    # 加载权重时严格匹配
    model.load_state_dict(
        torch.load(model_path, map_location=device),
        strict=True
    )
    return model.to(device).eval()


def preprocess_images(left_path, right_path, device):
    """预处理图像：调整大小、拼接、归一化"""
    # 定义与训练一致的预处理流程
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载并调整图像大小
    resize = transforms.Resize((512, 256))
    left_mask = mask_predict(left_path, OPTIC_MODEL_PATH, device)
    right_mask = mask_predict(right_path, OPTIC_MODEL_PATH, device)

    left_img = resize(remove_black_borders(process_image(left_path, left_mask)))
    right_img = resize(remove_black_borders(process_image(right_path, right_mask)))

    # 拼接图像（与训练时完全一致）
    combined = Image.new('RGB', (512, 512))
    combined.paste(left_img, (0, 0))
    combined.paste(right_img, (256, 0))
    combined.save('test.jpg')

    # 应用预处理并添加batch维度
    return transform(combined).unsqueeze(0).to(device)


def predict_probability(model, processed_image):
    """执行预测并返回患病概率"""
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.softmax(outputs, dim=1)
        # 假设输出索引1对应患病概率
        return probabilities[0][1].item()


def run_inference(left_path, right_path, model_path='G_best.pth'):
    """执行完整推理流程"""
    # 自动选择设备（优先使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


    # 加载模型和图像
    model = load_trained_model(model_path, device)
    processed_img = preprocess_images(left_path, right_path, device)

    # 执行预测
    prob = predict_probability(model, processed_img)
    return {
        'healthy_probability': 1 - prob,
        'disease_probability': prob,
        'status': 'success'
    }



# 使用示例
if __name__ == '__main__':
    # 替换为实际的图像路径
    left_eye_path = ROOT_DIR / 'dataset/Archive/preprocessed_images/167_left.jpg'
    right_eye_path = ROOT_DIR / 'dataset/Archive/preprocessed_images/167_right.jpg'

    result = run_inference(left_eye_path, right_eye_path)

    if result['status'] == 'success':
        print(f"健康概率: {result['healthy_probability'] * 100:.2f}%")
        print(f"患病概率: {result['disease_probability'] * 100:.2f}%")
    else:
        print(f"错误发生: {result['error']}")
