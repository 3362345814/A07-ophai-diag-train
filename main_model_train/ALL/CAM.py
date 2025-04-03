import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model
from keras.src.saving import load_model

from config import ROOT_DIR

# 配置参数
MODEL_PATH = "models/final_model_20250319_154850.h5"  # 替换为你的模型路径
IMAGE_SIZE = 299
CLASS_NAMES = ['D', 'G', 'C', 'A', 'H', 'M', 'O']


# 以下是从原代码中复制的必要组件，确保可以独立运行
class SEBlock(keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.se = keras.Sequential([
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(self.channels // self.ratio, activation='relu'),
            keras.layers.Dense(self.channels, activation='sigmoid'),
            keras.layers.Reshape((1, 1, self.channels))
        ])
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.se(inputs)


class MacroRecall(keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_recall', **kwargs):
        super(MacroRecall, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.recall_per_class = [keras.metrics.Recall() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        for i in range(self.num_classes):
            self.recall_per_class[i].update_state(y_true[:, i], y_pred[:, i], sample_weight)

    def result(self):
        return tf.reduce_mean([recall.result() for recall in self.recall_per_class])

    def reset_state(self):
        for recall in self.recall_per_class:
            recall.reset_state()


class MacroF1(keras.metrics.Metric):
    def __init__(self, num_classes, name='macro_f1', **kwargs):
        super(MacroF1, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision_per_class = [keras.metrics.Precision() for _ in range(num_classes)]
        self.recall_per_class = [keras.metrics.Recall() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        for i in range(self.num_classes):
            self.precision_per_class[i].update_state(y_true[:, i], y_pred[:, i], sample_weight)
            self.recall_per_class[i].update_state(y_true[:, i], y_pred[:, i], sample_weight)

    def result(self):
        f1_scores = []
        for i in range(self.num_classes):
            p = self.precision_per_class[i].result()
            r = self.recall_per_class[i].result()
            f1 = 2 * p * r / (p + r + keras.backend.epsilon())
            f1_scores.append(f1)
        return tf.reduce_mean(f1_scores)

    def reset_state(self):
        for p in self.precision_per_class:
            p.reset_state()
        for r in self.recall_per_class:
            r.reset_state()


def weighted_bce(y_true, y_pred):
    """加权损失函数"""
    class_counts = np.array([1000, 800, 1200, 900, 1100, 950, 850])  # 示例值，替换为实际值
    median = np.median(class_counts)
    class_weights = tf.cast(median / (class_counts + 1e-6), tf.float32)
    loss = keras.losses.binary_crossentropy(y_true, y_pred)
    weights = tf.reduce_sum(class_weights * y_true, axis=-1) + tf.reduce_sum(1.0 * (1 - y_true), axis=-1)
    return tf.reduce_mean(loss * weights)


class LayerCAM:
    def __init__(self, model, target_layer_names):
        """
        初始化LayerCAM可视化器

        参数:
            model: 加载的Keras模型
            target_layer_names: 要可视化的目标层名称列表
        """
        self.model = model
        self.target_layers = [model.get_layer(name) for name in target_layer_names]

        # 创建模型输出和指定层的梯度计算图
        self.grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.output] + [layer.output for layer in self.target_layers]
        )

    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        计算LayerCAM热力图

        参数:
            image: 输入图像(预处理后的)
            class_idx: 目标类别索引，None表示预测类别
            eps: 防止除零的小常数

        返回:
            各层的热力图字典{layer_name: heatmap}
        """
        # 转换为batch形式
        img_tensor = tf.convert_to_tensor(image[np.newaxis, ...])

        with tf.GradientTape() as tape:
            outputs = self.grad_model(img_tensor)
            preds = outputs[0]
            layer_outputs = outputs[1:]

            if class_idx is None:
                class_idx = tf.argmax(preds[0])

            # 获取目标类别的分数
            class_score = preds[:, class_idx]

        # 计算梯度
        grads = tape.gradient(class_score, layer_outputs)

        heatmaps = {}
        for i, (layer_name, layer_output, grad) in enumerate(
                zip([layer.name for layer in self.target_layers], layer_outputs, grads)):
            # LayerCAM核心计算
            weights = tf.nn.relu(grad)  # 使用ReLU过滤负梯度
            weighted_output = weights * layer_output

            # 沿通道维度求和
            heatmap = tf.reduce_sum(weighted_output, axis=-1)

            # 归一化处理
            heatmap = tf.squeeze(heatmap).numpy()
            heatmap = np.maximum(heatmap, 0)  # ReLU
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + eps)

            # 调整大小到输入图像尺寸
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)

            heatmaps[layer_name] = heatmap

        return heatmaps, class_idx.numpy()


def visualize_heatmap(model_path, image_id, output_dir):
    """
    可视化热力图的主函数

    参数:
        model_path: 模型文件路径
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    # 加载模型
    model = load_model(model_path, custom_objects={
        'SEBlock': SEBlock,  # 从您的训练代码中导入SEBlock
        'weighted_bce': weighted_bce,
        'MacroRecall': lambda: MacroRecall(len(CLASS_NAMES)),
        'MacroF1': lambda: MacroF1(len(CLASS_NAMES)),
    })

    # 定义要可视化的目标层(根据您的模型结构调整)
    target_layers = [
        'block1_conv1',  # 浅层特征
        'block1_conv2',
        'block4_sepconv2_act',  # 中层特征
        'block8_sepconv2_act',  # 中深层特征
        'block14_sepconv2_act',  # 深层特征
        'se_block'  # 您添加的SE注意力层
    ]

    # 初始化LayerCAM
    layercam = LayerCAM(model, target_layers)

    # 加载并预处理图像(与训练时相同)
    img_left = cv2.imread(ROOT_DIR / f"dataset/Archive/preprocessed_images/{image_id}_left.jpg")
    img_right = cv2.imread(ROOT_DIR / f"dataset/Archive/preprocessed_images/{image_id}_right.jpg")
    img = np.concatenate([img_left, img_right], axis=1)
    img = cv2.resize(img, (299, 299))
    img = img / 255.0

    # 计算热力图
    heatmaps, pred_class = layercam.compute_heatmap(img)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 可视化结果
    plt.figure(figsize=(20, 15))
    plt.subplot(1, len(heatmaps) + 1, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    for i, (layer_name, heatmap) in enumerate(heatmaps.items()):
        plt.subplot(1, len(heatmaps) + 1, i + 2)
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title(layer_name)
        plt.axis('off')

    plt.suptitle(f'LayerCAM Visualization (Predicted Class: {pred_class})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/layercam_vis.png')
    plt.close()


if __name__ == "__main__":
    model_path = MODEL_PATH
    image_id = "43"
    output_dir = "heatmaps"
    visualize_heatmap(model_path, image_id, output_dir)
