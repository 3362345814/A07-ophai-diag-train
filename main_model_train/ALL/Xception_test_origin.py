import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score
)
from itertools import cycle
import seaborn as sns
import pandas as pd
from keras.api.models import load_model
import numpy as np

from main_model_train.ALL.Xception import CSV_PATH, CLASS_NAMES, EnhancedEyeGenerator, BATCH_SIZE, MacroF1, MacroRecall, \
    weighted_bce, SEBlock


def evaluate_model(model_path, test_df):
    # 加载模型
    model = load_model(model_path, custom_objects={
        'weighted_bce': weighted_bce,
        'MacroRecall': lambda: MacroRecall(len(CLASS_NAMES)),
        'MacroF1': lambda: MacroF1(len(CLASS_NAMES)),
        'SEBlock': SEBlock
    })

    # 创建测试数据生成器（不进行过采样和增强）
    test_gen = EnhancedEyeGenerator(test_df, class_weights=np.ones(len(CLASS_NAMES)),
                                    batch_size=BATCH_SIZE, augment=False)

    # 获取真实标签和预测结果
    y_true = []
    y_pred = []
    for i in range(len(test_gen)):
        X, y = test_gen[i]
        y_true.extend(y)
        y_pred.extend(model.predict(X, verbose=0))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 将概率转换为二进制预测
    y_pred_bin = (y_pred > 0.5).astype(int)

    # 初始化结果存储
    metrics = {
        'Class': [],
        'Accuracy': [],
        'F1-Score': [],
        'Recall': [],
        'AUC': []
    }

    # 计算每个类别的指标
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink',
                    'navy', 'red', 'green', 'purple'])

    for i, class_name in enumerate(CLASS_NAMES):
        # 计算各项指标
        metrics['Class'].append(class_name)
        metrics['Accuracy'].append(accuracy_score(y_true[:, i], y_pred_bin[:, i]))
        metrics['F1-Score'].append(f1_score(y_true[:, i], y_pred_bin[:, i]))
        metrics['Recall'].append(recall_score(y_true[:, i], y_pred_bin[:, i]))
        metrics['AUC'].append(roc_auc_score(y_true[:, i], y_pred[:, i]))

        # 计算ROC曲线
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # 绘制单类ROC曲线
        plt.plot(fpr[i], tpr[i], color=next(colors),
                 label=f'{class_name} (AUC = {roc_auc[i]:.4f})')

    # 计算宏观平均指标
    metrics['Class'].append('Macro Avg')
    metrics['Accuracy'].append(accuracy_score(y_true, y_pred_bin))
    metrics['F1-Score'].append(f1_score(y_true, y_pred_bin, average='macro'))
    metrics['Recall'].append(recall_score(y_true, y_pred_bin, average='macro'))
    metrics['AUC'].append(np.mean(metrics['AUC']))

    # 绘制总ROC曲线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('result/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 生成分类报告
    # 生成分类报告并保存为柱状图
    report = classification_report(y_true, y_pred_bin, target_names=CLASS_NAMES, output_dict=True)

    # 创建指标数据框
    df_report = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Class'})
    df_report = df_report[df_report['Class'].isin(CLASS_NAMES + ['macro avg'])]  # 过滤需要展示的类别

    # 创建柱状图
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    sns.barplot(x='Class', y='f1-score', data=df_report, color='skyblue', label='F1-Score')
    sns.barplot(x='Class', y='recall', data=df_report, color='orange', label='Recall', alpha=0.7)
    plt.title('Classification Metrics')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    sns.barplot(x='Class', y='precision', data=df_report, color='lightgreen', label='Precision')
    plt.title('Precision Metrics')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('result/classification_report.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 生成混淆矩阵面板
    plt.figure(figsize=(15, 12))
    for i, class_name in enumerate(CLASS_NAMES):
        plt.subplot(3, 3, i + 1)
        cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'{class_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('result/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存指标到Excel
    df_metrics = pd.DataFrame(metrics)
    df_metrics = df_metrics.round(3)

    return df_metrics


if __name__ == "__main__":
    # 加载完整数据
    df = pd.read_csv(CSV_PATH)

    # 使用全部数据进行测试
    results = evaluate_model('models/final_model_20250319_154850.h5', df)

    # 打印结果
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("Evaluation Results:")
    print(results)
