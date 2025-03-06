import os

import cv2
from tqdm import tqdm


def remove_black_borders(img):
    """去除眼底图像黑边，返回裁剪区域坐标"""
    # 转换为灰度图并进行阈值处理
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 查找最大轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # 计算最大内接正方形
    square_size = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2
    crop_x1 = max(0, center_x - square_size // 2)
    crop_y1 = max(0, center_y - square_size // 2)

    return (crop_x1, crop_y1, square_size)


def preprocess_pair(img_path, mask_path, target_size=512):
    """同步处理原图和血管标注图"""
    # 读取原图
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 读取血管标注
    mask = cv2.imread(mask_path, 0)

    # 获取统一裁剪区域
    crop_params = remove_black_borders(img)
    x, y, size = crop_params

    # 执行相同裁剪
    img_cropped = img[y:y + size, x:x + size]
    mask_cropped = mask[y:y + size, x:x + size]

    # 统一缩放到目标尺寸
    img_resized = cv2.resize(img_cropped, (target_size, target_size))
    mask_resized = cv2.resize(mask_cropped, (target_size, target_size))

    return img_resized, mask_resized


def main():
    # 原始路径和新路径配置
    original_img_dir = "../DRHAGIS/Fundus_Images"
    original_mask_dir = "../DRHAGIS/Manual_Segmentations"

    preprocessed_img_dir = "../DRHAGIS/Fundus_Images_Preprocess"
    preprocessed_mask_dir = "../DRHAGIS/Manual_Segmentations_Preprocess"

    # 创建输出目录
    os.makedirs(preprocessed_img_dir, exist_ok=True)
    os.makedirs(preprocessed_mask_dir, exist_ok=True)

    # 获取所有原始图片
    img_files = [f for f in os.listdir(original_img_dir) if f.endswith('.jpg')]

    # 处理进度条
    for filename in tqdm(img_files, desc="预处理进度"):
        try:
            # 构建文件路径
            img_path = os.path.join(original_img_dir, filename)
            base_name = os.path.splitext(filename)[0]
            mask_filename = f"{base_name}_manual_orig.png"
            mask_path = os.path.join(original_mask_dir, mask_filename)

            # 执行预处理
            img_pre, mask_pre = preprocess_pair(img_path, mask_path)

            # 保存处理结果（保持原文件名）
            cv2.imwrite(os.path.join(preprocessed_img_dir, filename),
                        cv2.cvtColor(img_pre, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(preprocessed_mask_dir, mask_filename), mask_pre)

        except Exception as e:
            print(f"\n处理 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    main()
