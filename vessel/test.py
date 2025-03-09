import cv2
import torch
import numpy as np
from tqdm import tqdm

from vessel.preprocess import remove_black_borders
from vessel_detection import VesselSegmentor  # 从训练代码中导入模型


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


# ... 前面的导入和函数保持不变 ...

if __name__ == "__main__":
    # 配置参数
    checkpoint_path = "best_vessel_model.pth"
    input_dir = "../dataset/Archive/preprocessed_images"
    output_dir = "../dataset/Archive/mask"

    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 初始化（保持不变）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VesselSegmentor().to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # 获取所有待预测图片
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    # 批量预测
    for filename in tqdm(image_files):
        try:
            img_path = os.path.join(input_dir, filename)
            # 生成输出路径
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_mask.jpg")

            # 执行预测
            mask = predict_vessels(model, img_path, device)
            cv2.imwrite(output_path, mask)

        except Exception as e:
            print(f"处理 {filename} 失败: {str(e)}")
