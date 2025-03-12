import os
from tqdm import tqdm

from config import ROOT_DIR
from optic_disk.test import predict_and_save

# 配置路径
input_dir = ROOT_DIR / "dataset/Archive/preprocessed_images"
output_dir = ROOT_DIR / "dataset/Archive/optic_mask"
os.makedirs(output_dir, exist_ok=True)  # 自动创建输出目录


# 遍历所有图像文件
for filename in tqdm(os.listdir(input_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        # 构造路径
        input_path = os.path.join(input_dir, filename)

        # 生成输出文件名
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_mask{ext}"
        output_path = os.path.join(output_dir, output_filename)

        predict_and_save(
            image_path=input_path,
            model_path="best_disc_model.pth",
            output_path=output_path
        )

print("批量处理完成！")