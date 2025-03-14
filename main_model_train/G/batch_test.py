import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from PIL import Image
from albumentations import Compose, Resize
from torchvision.transforms import transforms
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from config import ROOT_DIR
from main_model_train.G.resnet import remove_black_borders
from main_model_train.G.test import OPTIC_MODEL_PATH, load_trained_model, process_image
from optic_disk.test import OpticDiscSegmentor

# 新增配置项
CONFIG = {
    'data_root': ROOT_DIR / 'dataset/Archive',
    'image_dir': 'preprocessed_images',
    'label_csv': 'full_df.csv',
    'batch_size': 8,
    'threshold': 0.5  # 分类阈值
}

def batch_inference(model, optic_model, df, device):
    """批量推理主函数"""
    y_true = []
    y_pred = []
    probas = []
    skipped = []

    # 按batch处理
    for i in tqdm(range(0, len(df), CONFIG['batch_size']), desc="Processing"):
        batch = df.iloc[i:i+CONFIG['batch_size']]

        batch_inputs = []
        batch_labels = []

        for _, row in batch.iterrows():
            try:
                # 构建图像路径
                left_path = CONFIG['data_root'] / CONFIG['image_dir'] / f"{row['ID']}_left.jpg"
                right_path = CONFIG['data_root'] / CONFIG['image_dir'] / f"{row['ID']}_right.jpg"

                # 检查文件是否存在
                if not left_path.exists() or not right_path.exists():
                    skipped.append(row['ID'])
                    continue

                # 预处理图像
                processed_img = preprocess_images(
                    left_path,
                    right_path,
                    optic_model,
                    device
                )

                batch_inputs.append(processed_img)
                batch_labels.append(row['G'])

            except Exception as e:
                print(f"处理样本 {row['ID']} 失败: {str(e)}")
                skipped.append(row['ID'])

        # 执行批量推理
        if batch_inputs:
            with torch.no_grad():
                inputs = torch.cat(batch_inputs).to(device)
                outputs = model(inputs)
                batch_probas = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            probas.extend(batch_probas.tolist())
            y_true.extend(batch_labels)

    # 转换为numpy数组
    y_true = np.array(y_true)
    probas = np.array(probas)
    y_pred = (probas >= CONFIG['threshold']).astype(int)

    return y_true, y_pred, probas, skipped

def evaluate_performance(y_true, y_pred, probas):
    """计算并打印性能指标"""
    metrics = {
        '准确率': accuracy_score(y_true, y_pred),
        '精确率': precision_score(y_true, y_pred),
        '召回率': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, probas)
    }

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=['真实健康', '真实患病'],
                         columns=['预测健康', '预测患病']
                         )

    # 分类报告
    report = classification_report(
        y_true, y_pred,
        target_names=['健康', '患病'],
        output_dict=True
    )

    return metrics, cm_df, report

# 修改后的预处理函数（移除测试保存）
def preprocess_images(left_path, right_path, optic_model, device):
    """预处理图像（优化版）"""
    # 生成掩码
    left_mask = mask_predict(optic_model, left_path, device)
    right_mask = mask_predict(optic_model, right_path, device)

    # 处理图像
    left_img = remove_black_borders(process_image(left_path, left_mask))
    right_img = remove_black_borders(process_image(right_path, right_mask))

    # 调整尺寸
    resize = transforms.Resize((512, 256))
    left_img = resize(left_img)
    right_img = resize(right_img)

    # 拼接图像
    combined = Image.new('RGB', (512, 512))
    combined.paste(left_img, (0, 0))
    combined.paste(right_img, (256, 0))

    # 转换和归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(combined).unsqueeze(0)

# 优化后的mask_predict（接受预加载模型）
def mask_predict(model, image_path, device):
    """生成掩码（优化版）"""
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 预处理
    transform = Compose([Resize(512, 512)])
    processed_image = transform(image=original_image)["image"]

    # 转换为张量
    input_tensor = torch.from_numpy(processed_image).float()
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # 推理
    with torch.no_grad():
        prediction = model(input_tensor)

    # 后处理
    mask = prediction.squeeze().cpu().numpy()
    return (mask > 0.5).astype(np.uint8) * 255

def main():
    # 设备配置
    device = torch.device('mps')

    # 加载数据
    df = pd.read_csv(CONFIG['data_root'] / CONFIG['label_csv'])

    print(f"原始数据量: {len(df)}")

    # 过滤无效样本
    valid_ids = []
    for _, row in df.iterrows():
        left_path = CONFIG['data_root'] / CONFIG['image_dir'] / f"{row['ID']}_left.jpg"
        right_path = CONFIG['data_root'] / CONFIG['image_dir'] / f"{row['ID']}_right.jpg"
        if left_path.exists() and right_path.exists():
            valid_ids.append(row['ID'])

    df = df[df['ID'].isin(valid_ids)].reset_index(drop=True)
    print(f"有效数据量: {len(df)}")

    # 预加载模型
    print("加载模型中...")
    optic_model = OpticDiscSegmentor().to(device).eval()
    optic_model.load_state_dict(torch.load(OPTIC_MODEL_PATH, map_location=device))

    classifier = load_trained_model('G_best.pth', device)

    # 执行批量推理
    y_true, y_pred, probas, skipped = batch_inference(classifier, optic_model, df, device)

    # 评估性能
    metrics, cm, report = evaluate_performance(y_true, y_pred, probas)

    # 打印结果
    print("\n性能指标:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n混淆矩阵:")
    print(cm)

    print("\n分类报告:")
    print(pd.DataFrame(report).T)

    if skipped:
        print(f"\n跳过的样本ID: {skipped}")

if __name__ == '__main__':
    main()