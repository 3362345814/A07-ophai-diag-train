import os

import cv2
import numpy as np
from skimage.filters import frangi


def remove_black_borders(img):
    def smart_retina_preprocessing(img):

        # 获取原始尺寸
        h, w = img.shape[:2]

        if h > w:
            top = bottom = 0
            left = right = (h - w) // 2
        else:
            top = bottom = (w - h) // 2
            left = right = 0


        # 添加黑色边框
        padded = cv2.copyMakeBorder(img,
                                    top, bottom,
                                    left, right,
                                    cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])

        return padded

    img = smart_retina_preprocessing(img)

    # 转换为灰度图并进行阈值处理
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓并找到最大轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    # 获取有效区域的边界矩形
    x, y, w, h = cv2.boundingRect(cnt)

    # 计算最大内接正方形（保持眼球完整）
    square_size = max(w, h)
    center_x = x + w//2
    center_y = y + h//2

    # 计算裁剪坐标（确保不越界）
    crop_x1 = max(0, center_x - square_size//2)
    crop_y1 = max(0, center_y - square_size//2)
    crop_x2 = min(img.shape[1], crop_x1 + square_size)
    crop_y2 = min(img.shape[0], crop_y1 + square_size)

    # 执行裁剪
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped

def adaptive_contrast_enhancement(img, clip_limit=3.0, grid_size=(8,8)):
    """
    使用CLAHE算法增强对比度
    :param img: 图片
    :param clip_limit: 对比度限制阈值（推荐2-4）
    :param grid_size: 网格划分大小（推荐8x8到16x16）
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE应用在L通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)

    merged = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def vessel_enhancement(img, sigma_range=(1, 3), steps=5):
    """
    多尺度Frangi滤波增强血管结构
    :param sigma_range: 高斯核尺度范围
    :param steps: 尺度采样数
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 多尺度融合
    enhanced = np.zeros_like(gray, dtype=np.float32)
    for sigma in np.linspace(sigma_range[0], sigma_range[1], steps):
        enhanced += frangi(gray, sigmas=[sigma], black_ridges=False)

    # 归一化并融合到原图
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.addWeighted(img, 0.7,
                           cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_GRAY2RGB),
                           0.3, 0)


def gray_world_normalization(img):
    """
    灰度世界颜色校正算法
    """
    avg_r = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_b = np.mean(img[:,:,2])
    avg_gray = (avg_r + avg_g + avg_b) / 3.0

    img_normalized = np.zeros_like(img, dtype=np.float32)
    img_normalized[:,:,0] = img[:,:,0] * (avg_gray / avg_r)
    img_normalized[:,:,1] = img[:,:,1] * (avg_gray / avg_g)
    img_normalized[:,:,2] = img[:,:,2] * (avg_gray / avg_b)

    return cv2.normalize(img_normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def full_processing_pipeline(img_path, output_size=512):
    # 1. 基础预处理
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 图像增强
    img = remove_black_borders(img)
    img = adaptive_contrast_enhancement(img)
    img = vessel_enhancement(img)

    # 3. 色彩标准化
    img = gray_world_normalization(img)

    # 4. ROI处理
    # disc_roi, disc_coords = optic_disc_segmentation(img)

    # # 5. 标准化输出
    # disc_roi = cv2.resize(disc_roi, (256, 256))
    #
    # # 6. 可视化标注
    # annotated = img.copy()
    # cv2.rectangle(annotated, (disc_coords[0], disc_coords[1]),
    #               (disc_coords[2], disc_coords[3]), (255,0,0), 2)

    return {
        'full_image': cv2.resize(img, (output_size, output_size)),
        # 'disc_roi': disc_roi,
        # 'annotated': annotated
    }

# 在现有预处理函数后新增批量处理功能
def batch_process_images(input_dir, output_dir):
    """
    批量处理眼底图像
    :param input_dir: 输入目录路径 (包含*_left.jpg和*_right.jpg)
    :param output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有待处理文件
    image_files = [f for f in os.listdir(input_dir)
                   if f.endswith(('_left.jpg', '_right.jpg'))]

    print(f"发现 {len(image_files)} 张待处理图像...")

    for filename in tqdm(image_files):
        try:
            # 执行完整处理流程
            result = full_processing_pipeline(os.path.join(input_dir, filename))

            # 保存处理后的图像
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path,
                        cv2.cvtColor(result['full_image'], cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")



# 在文件末尾添加执行代码
if __name__ == '__main__':
    from tqdm import tqdm

    # 使用示例（请根据实际情况修改路径）
    batch_process_images(
        input_dir="../dataset/Archive/preprocessed_images",
        output_dir="../dataset/Archive/preprocessed_images1"
    )