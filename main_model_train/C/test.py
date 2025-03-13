import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image
import warnings

from config import ROOT_DIR
from vessel.preprocess import remove_black_borders

from vessel.vessel_detection import VesselSegmentor

# 禁用PIL的DecompressionBombWarning警告
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
VESSEL_MODEL_PATH = ROOT_DIR / 'vessel/best_vessel_model.pth'


def preprocess_single(img_path, target_size=512):
    """单图像预处理流程"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 应用相同的黑边去除逻辑
    x, y, size = remove_black_borders(img)
    img_cropped = img[y:y + size, x:x + size]
    return cv2.resize(img_cropped, (target_size, target_size))


def predict_vessels(model, img_path, device):
    # 预处理
    processed_img = preprocess_single(img_path)

    # 转换为模型输入格式
    input_tensor = torch.from_numpy(processed_img).float().permute(2, 0, 1) / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 后处理
    mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # 新增形态学操作（先膨胀后腐蚀）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.dilate(mask, kernel, iterations=1)  # 先膨胀连接断裂血管
    mask = cv2.erode(mask, kernel, iterations=1)   # 再腐蚀消除细小噪声

    return mask


def load_trained_model(model_path, device):
    """加载训练好的模型（修正后）"""
    model = resnet50(weights=None)  # 确保不使用预训练权重
    num_features = model.fc.in_features

    # 关键修改：保持与训练代码一致的线性层
    model.fc = nn.Linear(num_features, 2)

    # 加载权重时添加严格匹配检查
    model.load_state_dict(
        torch.load(model_path, map_location=device),
        strict=True  # 添加严格模式校验
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

    model = VesselSegmentor().to(device)
    model.load_state_dict(torch.load(VESSEL_MODEL_PATH))
    model.eval()

    left_img = predict_vessels(model, left_path, device)
    right_img = predict_vessels(model, right_path, device)

    left_img = resize(Image.fromarray(left_img))
    right_img = resize(Image.fromarray(right_img))


    # 拼接图像（与训练时完全一致）
    combined = Image.new('RGB', (512, 512))
    combined.paste(left_img, (0, 0))
    combined.paste(right_img, (256, 0))

    # 应用预处理并添加batch维度
    return transform(combined).unsqueeze(0).to(device)


def predict_probability(model, processed_image):
    """执行预测并返回患病概率"""
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.softmax(outputs, dim=1)
        # 假设输出索引1对应患病概率
        return probabilities[0][1].item()


def run_inference(left_path, right_path, model_path='C_best.pth'):
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
    left_eye_path = ROOT_DIR / 'dataset/C_D_G/cataract/cataract_075.png'
    right_eye_path = ROOT_DIR / 'dataset/C_D_G/cataract/cataract_079.png'

    result = run_inference(left_eye_path, right_eye_path)

    if result['status'] == 'success':
        print(f"健康概率: {result['healthy_probability'] * 100:.2f}%")
        print(f"患病概率: {result['disease_probability'] * 100:.2f}%")
    else:
        print(f"错误发生: {result['error']}")
