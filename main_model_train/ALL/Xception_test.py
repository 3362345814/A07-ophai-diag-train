import os
from itertools import cycle

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.api.utils import Sequence
from sklearn.metrics import roc_curve, auc, classification_report, multilabel_confusion_matrix

from config import ROOT_DIR
from main_model_train.ALL.Xception import IMAGE_SIZE, CLASS_NAMES, BATCH_SIZE, MacroF1, MacroRecall

TEST_CSV_PATH = ROOT_DIR / "dataset/combined_dataset.csv"
MODEL_PATH = "models/fina_30epoch_0.76acc_model.h5"


# 在原有测试代码基础上添加以下可视化函数

def plot_confusion_matrices(confusion_matrices, class_names):
    """绘制多标签混淆矩阵"""
    plt.figure(figsize=(15, 12))
    n_classes = len(class_names)
    n_cols = 3
    n_rows = int(np.ceil(n_classes / n_cols))

    for i, (matrix, name) in enumerate(zip(confusion_matrices, class_names)):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1'])
        plt.title(f'Class {name}\nTP={matrix[1, 1]} FN={matrix[1, 0]}\nFP={matrix[0, 1]} TN={matrix[0, 0]}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()


def plot_roc_curves(y_true, y_pred_prob, class_names):
    """绘制多标签ROC曲线"""
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink',
                    'navy', 'red', 'green', 'purple'])

    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)

        fpr_dict[class_name] = fpr
        tpr_dict[class_name] = tpr
        roc_auc_dict[class_name] = roc_auc

        plt.plot(fpr, tpr, color=next(colors),
                 label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()

    return roc_auc_dict


# 测试数据生成器
class TestGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32):
        self.df = dataframe
        self.batch_size = batch_size
        self.indices = np.arange(len(self.df))

        # 创建标签到索引的映射字典
        self.label_to_indices = {}
        for idx, row in self.df.iterrows():
            # 将多标签转换为元组作为字典键
            label_key = tuple(row[CLASS_NAMES].astype(int))
            if label_key not in self.label_to_indices:
                self.label_to_indices[label_key] = []
            self.label_to_indices[label_key].append(idx)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        X = np.empty((len(batch_indices), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        y = np.empty((len(batch_indices), len(CLASS_NAMES)), dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            main_row = self.df.iloc[idx]
            # 获取主样本标签
            main_label = tuple(main_row[CLASS_NAMES].astype(int))

            # 随机选择相同标签的样本作为"右眼"
            candidate_indices = [x for x in self.label_to_indices.get(main_label, []) if x != idx]
            if not candidate_indices:  # 如果找不到其他样本，使用自身
                candidate_indices = [idx]

            random_idx = np.random.choice(candidate_indices)
            random_row = self.df.iloc[random_idx]

            # 加载并拼接图像
            img = self._load_images(
                main_row['image_name'],  # 左眼路径
                random_row['image_name']  # 伪右眼路径（实际是另一个左眼）
            )

            X[i] = img
            y[i] = main_row[CLASS_NAMES].values.astype(np.float32)

        return X, y

    def _load_images(self, left_path, pseudo_right_path):
        """加载并拼接两个图像（实际都是左眼）"""
        try:
            # 加载主图像（左眼）
            img_left = cv2.imread(left_path)
            img_left = cv2.resize(img_left, (IMAGE_SIZE, IMAGE_SIZE))

            # 加载伪右眼图像（实际是另一个左眼）
            img_right = cv2.imread(pseudo_right_path)
            img_right = cv2.resize(img_right, (IMAGE_SIZE, IMAGE_SIZE))

            # 水平拼接
            combined = np.concatenate([img_left, img_right], axis=1)

            # 调整最终尺寸
            final_img = cv2.resize(combined, (IMAGE_SIZE, IMAGE_SIZE))

            return final_img.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Error loading {left_path} or {pseudo_right_path}: {str(e)}")
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# 加载测试数据时只需要验证单个图像存在
def load_test_data():
    test_df = pd.read_csv(TEST_CSV_PATH)

    valid_rows = []
    for idx, row in test_df.iterrows():
        row['image_name'] = ROOT_DIR / row['image_name']
        if os.path.exists(row['image_name']):
            valid_rows.append(row)

    print(f"Filtered {len(test_df) - len(valid_rows)} invalid samples")
    return pd.DataFrame(valid_rows)


# 修改后的evaluate_model函数
def evaluate_model(model, test_gen):
    # 进行预测
    y_pred_prob = model.predict(test_gen)
    y_true = np.concatenate([test_gen[i][1] for i in range(len(test_gen))])

    # 将概率转换为二进制预测
    y_pred_bin = (y_pred_prob > 0.5).astype(int)

    # 计算分类报告
    print("Classification Report:")
    print(classification_report(y_true, y_pred_bin, target_names=CLASS_NAMES))

    # 计算多标签混淆矩阵
    print("\nConfusion Matrices:")
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred_bin)
    plot_confusion_matrices(confusion_matrices, CLASS_NAMES)

    # 计算并绘制ROC曲线
    roc_auc_dict = plot_roc_curves(y_true, y_pred_prob, CLASS_NAMES)
    print("\nROC AUC Scores:")
    for cls, auc_val in roc_auc_dict.items():
        print(f"{cls}: {auc_val:.4f}")


# 修改后的main函数
def main():
    # 加载数据
    test_df = load_test_data()
    print(f"Loaded {len(test_df)} valid test samples")

    # 创建生成器
    test_gen = TestGenerator(test_df, BATCH_SIZE)

    # 加载模型
    model = keras.models.load_model(MODEL_PATH, custom_objects={
        'MacroRecall': lambda: MacroRecall(len(CLASS_NAMES)),
        'MacroF1': lambda: MacroF1(len(CLASS_NAMES)),
        'weighted_bce': lambda y_true, y_pred: keras.losses.binary_crossentropy(y_true, y_pred)
    })

    # 评估模型
    print("Evaluating model...")
    evaluate_model(model, test_gen)

    # 计算自定义指标
    print("\nCalculating additional metrics...")
    model.compile(metrics=[
        'accuracy',
        MacroRecall(num_classes=len(CLASS_NAMES)),
        MacroF1(num_classes=len(CLASS_NAMES))
    ])
    results = model.evaluate(test_gen)
    print(f"\nTest Accuracy: {results[1]:.4f}")
    print(f"Test Macro Recall: {results[2]:.4f}")
    print(f"Test Macro F1: {results[3]:.4f}")


if __name__ == "__main__":
    # 设置绘图风格
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.dpi'] = 150
    main()
