import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image
import warnings
from pathlib import Path

from tqdm import tqdm

from config import ROOT_DIR

# 禁用警告
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


def load_trained_model(model_path, device):
    """加载训练好的模型"""
    model = resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def preprocess_images(left_path, right_path, device):
    """预处理图像并返回张量"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    left_path = ROOT_DIR / 'dataset/Archive/preprocessed_images' / left_path
    right_path = ROOT_DIR / 'dataset/Archive/preprocessed_images' / right_path

    resize = transforms.Resize((512, 256))
    try:
        left_img = resize(Image.open(left_path).convert('RGB'))
        right_img = resize(Image.open(right_path).convert('RGB'))
    except Exception as e:
        raise ValueError(f"图像加载失败: {str(e)}")

    combined = Image.new('RGB', (512, 512))
    combined.paste(left_img, (0, 0))
    combined.paste(right_img, (256, 0))

    return transform(combined).unsqueeze(0).to(device)


def predict_probability(model, processed_image):
    """执行预测并返回患病概率"""
    with torch.no_grad():
        outputs = model(processed_image)
        return torch.softmax(outputs, dim=1)[0][1].item()


def batch_test(test_csv_path, model_path='D_best.pth', threshold=0.5):
    """
    批量测试并生成性能报告
    参数:
        test_csv_path: 包含以下列的CSV文件路径:
                       left_path, right_path, label
        model_path: 训练好的模型路径
        threshold: 分类阈值 (默认0.5)
    返回:
        包含测试结果的字典
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    results = {
        'true_labels': [],
        'pred_labels': [],
        'probabilities': [],
        'confusion_matrix': None,
        'classification_report': None,
        'errors': []
    }

    try:
        # 加载模型
        model = load_trained_model(model_path, device)

        # 读取测试数据
        df = pd.read_csv(test_csv_path)

        # 遍历测试样本
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # 预处理图像
                tensor = preprocess_images(row['Left-Fundus'], row['Right-Fundus'], device)

                # 预测概率
                prob = predict_probability(model, tensor)
                pred_label = int(prob >= threshold)

                # 记录结果
                results['true_labels'].append(row['D'])
                results['pred_labels'].append(pred_label)
                results['probabilities'].append(prob)
                print(pred_label)

            except Exception as e:
                results['errors'].append({
                    'index': idx,
                    'error': str(e),
                    'data': row.to_dict()
                })

        # 计算性能指标
        if len(results['true_labels']) > 0:
            # 混淆矩阵
            cm = confusion_matrix(results['true_labels'], results['pred_labels'])
            results['confusion_matrix'] = cm

            # 分类报告
            report = classification_report(
                results['true_labels'], results['pred_labels'],
                target_names=['Healthy', 'Disease'],
                output_dict=True
            )
            results['classification_report'] = report

            # 绘制混淆矩阵
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Healthy', 'Disease'],
                        yticklabels=['Healthy', 'Disease'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('confusion_matrix.png')
            plt.close()

    except Exception as e:
        results['errors'].append({'error': f'全局错误: {str(e)}'})

    return results


def print_results(results):
    """打印测试结果"""
    if results['confusion_matrix'] is not None:
        print("\n=== 混淆矩阵 ===")
        print(results['confusion_matrix'])

    if results['classification_report'] is not None:
        print("\n=== 分类报告 ===")
        print(pd.DataFrame(results['classification_report']).transpose())

    if results['errors']:
        print("\n=== 错误汇总 ===")
        for error in results['errors']:
            print(f"样本 {error.get('index', '?')}: {error['error']}")


if __name__ == '__main__':
    # 示例使用
    test_csv = Path(ROOT_DIR / 'dataset/Archive/full_df.csv')

    test_results = batch_test(test_csv)
    print_results(test_results)
