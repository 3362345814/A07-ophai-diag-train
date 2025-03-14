import cv2
import os

from tqdm import tqdm

from config import ROOT_DIR

# 配置路径
original_dir = ROOT_DIR / "dataset/C_D_G/normal"
mask_dir = ROOT_DIR / "dataset/C_D_G/normal_optic_mask"
output_dir = ROOT_DIR / "dataset/C_D_G/normal_optic_disk"
os.makedirs(output_dir, exist_ok=True)


def process_image(original_path, mask_path, output_path):
    # 1. 读取图像
    original = cv2.imread(original_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 灰度模式读取

    # 2. 二值化处理（确保掩码为0/255）
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. 反转掩码（使目标区域变为白色）
    inverted_mask = cv2.bitwise_not(binary_mask)

    # 4. 形态学膨胀（扩大目标区域）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))  # 椭圆核，尺寸根据需求调整
    dilated_mask = cv2.dilate(inverted_mask, kernel, iterations=1)

    # 6. 应用掩码（显示掩码黑色区域，隐藏白色区域）
    result = cv2.bitwise_and(original, original, mask=dilated_mask)

    # 7. 保存结果
    cv2.imwrite(output_path, result)


# 遍历处理所有图像
for filename in tqdm(os.listdir(original_dir)):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        original_path = os.path.join(original_dir, filename)
        mask_path = os.path.join(mask_dir, f"{os.path.splitext(filename)[0]}_mask.jpg")

        if os.path.exists(mask_path):
            output_path = os.path.join(output_dir, filename)
            process_image(original_path, mask_path, output_path)

print("处理完成！")
