import cv2
import os

from tqdm import tqdm

from config import ROOT_DIR

# 配置路径
original_dir = ROOT_DIR / "dataset/AMD/OriginalImages"
mask_dir = ROOT_DIR / "dataset/AMD/vessel_mask"
output_dir = ROOT_DIR / "dataset/AMD/vessel"
os.makedirs(output_dir, exist_ok=True)


def process_image(original_path, mask_path, output_path):
    # 1. 读取图像
    original = cv2.imread(original_path)
    original = cv2.resize(original, (512, 512))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 灰度模式读取

    # 2. 二值化处理（确保掩码为0/255）
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 6. 应用掩码（显示掩码黑色区域，隐藏白色区域）
    result = cv2.bitwise_and(original, original, mask=binary_mask)

    # 7. 保存结果
    cv2.imwrite(output_path, result)


# 遍历处理所有图像
for filename in tqdm(os.listdir(original_dir)):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        original_path = os.path.join(original_dir, filename)
        mask_path = os.path.join(mask_dir, f"{os.path.splitext(filename)[0]}_mask{os.path.splitext(filename)[1]}")

        if os.path.exists(mask_path):
            output_path = os.path.join(output_dir, filename)
            process_image(original_path, mask_path, output_path)

print("处理完成！")
