import torch
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image
import warnings

from config import ROOT_DIR

# 禁用PIL的DecompressionBombWarning警告
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


def load_trained_model(model_path, device):
    """加载训练好的模型"""
    model = resnet50()  # 不需要预训练权重
    # 修改全连接层与训练时一致
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    # 加载训练权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


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
    left_img = resize(Image.open(left_path).convert('RGB'))
    right_img = resize(Image.open(right_path).convert('RGB'))

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


def run_inference(left_path, right_path, model_path='D_best.pth'):
    """执行完整推理流程"""
    # 自动选择设备（优先使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    try:
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

    except FileNotFoundError as e:
        return {'error': str(e), 'status': 'failed'}
    except Exception as e:
        return {'error': f'推理错误: {str(e)}', 'status': 'failed'}


# 使用示例
if __name__ == '__main__':
    # 替换为实际的图像路径
    left_eye_path = ROOT_DIR / 'dataset/Archive/preprocessed_images/14_left.jpg'
    right_eye_path = ROOT_DIR / 'dataset/Archive/preprocessed_images/13_right.jpg'

    result = run_inference(left_eye_path, right_eye_path)

    if result['status'] == 'success':
        print(f"健康概率: {result['healthy_probability'] * 100:.2f}%")
        print(f"患病概率: {result['disease_probability'] * 100:.2f}%")
    else:
        print(f"错误发生: {result['error']}")
