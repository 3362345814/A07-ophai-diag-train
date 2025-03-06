import cv2
import torch
import numpy as np

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
    return mask


if __name__ == "__main__":
    # 配置参数
    checkpoint_path = "best_vessel_model.pth"  # 训练保存的最佳模型
    test_image_path = "../Archive/preprocessed_images/3_left.jpg"  # 待预测图片路径
    output_path = "vessel_mask.png"  # 输出路径

    # 初始化
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VesselSegmentor().to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # 执行预测
    try:
        mask = predict_vessels(model, test_image_path, device)
        cv2.imwrite(output_path, mask)
        print(f"预测完成，结果已保存至 {output_path}")
    except Exception as e:
        print(f"预测失败: {str(e)}")
