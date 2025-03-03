import tensorflow as tf
import keras
from keras.src.layers import *
from keras.src.models import Model
from keras.src.applications.resnet import ResNet50
import pandas as pd
import os
import numpy as np
from keras.src.callbacks import ModelCheckpoint


# --------------------------
# 数据加载模块（已修改）
# --------------------------
class EyePairDataLoader:
    def __init__(self, data_dir, label_csv, img_size=(512, 512)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.label_df = pd.read_csv(label_csv)
        self.label_cols = ['G', 'C', 'A', 'H', 'M', 'O']
        self.pairs = self._validate_pairs()

    def _validate_pairs(self):
        valid_pairs = []
        for _, row in self.label_df.iterrows():
            img_id = str(row['ID'])
            left_path = os.path.join(self.data_dir, f"{img_id}_left.jpg")
            right_path = os.path.join(self.data_dir, f"{img_id}_right.jpg")
            if os.path.exists(left_path) and os.path.exists(right_path):
                valid_pairs.append({
                    'left': left_path,
                    'right': right_path,
                    'labels': row[self.label_cols].values.astype(np.float32)
                })
        return valid_pairs

    def _process_image(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        return img

    def _augment(self, left_img, right_img, labels, weights):

        # 随机裁剪（保留90%区域）
        if tf.random.uniform(()) > 0.3:
            crop_size = int(self.img_size[0] * 0.9)
            left_img = tf.image.random_crop(left_img, [crop_size, crop_size, 3])
            right_img = tf.image.random_crop(right_img, [crop_size, crop_size, 3])
            left_img = tf.image.resize(left_img, self.img_size)
            right_img = tf.image.resize(right_img, self.img_size)

        # 增加对比度调整
        left_img = tf.image.random_contrast(left_img, 0.8, 1.2)
        right_img = tf.image.random_contrast(right_img, 0.8, 1.2)
        left_img = tf.image.random_brightness(left_img, max_delta=0.1)
        right_img = tf.image.random_brightness(right_img, max_delta=0.1)
        left_img = left_img / 255.0
        right_img = right_img / 255.0
        return (left_img, right_img), labels, weights

    def get_dataset(self, batch_size=16, is_train=True):
        # 生成基础数据
        left_paths = [p['left'] for p in self.pairs]
        right_paths = [p['right'] for p in self.pairs]
        labels = [p['labels'] for p in self.pairs]  # 每个样本包含6个标签

        # 计算任务权重（关键修改）
        task_weights = self._calculate_task_weights()

        # 生成样本权重字典（按任务维度）
        sample_weights = {
            task: [
                task_weights[task][int(labels[i][task_idx])]  # 获取对应任务的标签
                for i in range(len(labels))
            ]
            for task_idx, task in enumerate(self.label_cols)
        }

        # 验证数据维度
        num_samples = len(left_paths)
        assert all(len(v) == num_samples for v in sample_weights.values()), \
            "权重维度必须与样本数一致"

        # 构建数据集
        dataset = tf.data.Dataset.from_tensor_slices((
            {'left_eye': left_paths, 'right_eye': right_paths},  # 输入（N,）
            {task: [labels[i][task_idx] for i in range(num_samples)]  # 标签（N,）
             for task_idx, task in enumerate(self.label_cols)},
            {task: sample_weights[task] for task in self.label_cols}  # 权重（N,）
        ))

        # 数据预处理
        def _map_func(inputs, labels, weights):
            left_img = self._process_image(inputs['left_eye'])
            right_img = self._process_image(inputs['right_eye'])
            return (left_img, right_img), labels, weights

        dataset = dataset.map(_map_func, num_parallel_calls=tf.data.AUTOTUNE)

        # 数据增强
        if is_train:
            dataset = dataset.map(
                lambda imgs, lbls, wgts: self._augment(imgs[0], imgs[1], lbls, wgts),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def _calculate_task_weights(self):
        df = self.label_df
        weights = {}
        for task in self.label_cols:
            pos = df[task].sum()
            neg = len(df) - pos
            if pos == 0 or neg == 0:
                weights[task] = {0: 1.0, 1: 1.0}
            else:
                weight_0 = np.float32((1 / neg) * (pos + neg) / 2.0)
                weight_1 = np.float32((1 / pos) * (pos + neg) / 2.0)
                weights[task] = {0: weight_0, 1: weight_1}
        return weights


# --------------------------
# 模型定义（保持不变）
# --------------------------
def build_multi_label_model(input_shape=(512, 512, 3)):
    # 双输入流
    left_input = Input(shape=input_shape, name='left_eye')
    right_input = Input(shape=input_shape, name='right_eye')

    base_cnn = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # 解冻最后两个残差块
    for layer in base_cnn.layers:
        layer.trainable = False
    for layer in base_cnn.layers[-50:]:  # 解冻最后约1/3的层
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    # 双分支特征提取
    left_features = base_cnn(left_input)
    right_features = base_cnn(right_input)

    # 改进的注意力融合模块
    def dual_attention_fusion(feat1, feat2):
        """改进版注意力融合"""
        # 获取原始特征通道数
        channels = feat1.shape[-1]

        # 并行注意力处理
        def cbam_block(input_feature):
            # 通道注意力
            channel = GlobalAveragePooling2D()(input_feature)
            channel = Dense(int(channels / 8), activation='relu')(channel)
            channel = Dense(channels, activation='sigmoid')(channel)
            channel = Reshape((1, 1, channels))(channel)

            # 空间注意力
            spatial = Conv2D(1, 7, padding='same', activation='sigmoid')(input_feature)

            return Multiply()([input_feature, channel]), Multiply()([input_feature, spatial])

        # 应用CBAM到两个分支
        feat1_att, _ = cbam_block(feat1)
        feat2_att, _ = cbam_block(feat2)

        # 跨模态融合（修复通道数）
        cross_att = Concatenate(axis=-1)([feat1_att, feat2_att])
        cross_att = Conv2D(channels, 3, padding='same', activation='swish')(cross_att)  # 输出通道调整为2048

        # 残差连接（确保通道一致）
        return Add()([feat1, cross_att]), Add()([feat2, cross_att])

    # 注意力融合
    left_att, right_att = dual_attention_fusion(left_features, right_features)

    def global_stream(features):
        x = GlobalAveragePooling2D()(features)
        x = Dense(512, activation='swish')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        return x

    left_global = global_stream(left_att)
    right_global = global_stream(right_att)

    # 特征融合部分修改
    merged = Concatenate()([left_global, right_global])
    merged = Dense(1024, activation='swish')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(512, activation='swish')(merged)

    # 多任务输出层
    outputs = [
        Dense(1, activation='sigmoid', name='G')(merged),
        Dense(1, activation='sigmoid', name='C')(merged),
        Dense(1, activation='sigmoid', name='A')(merged),
        Dense(1, activation='sigmoid', name='H')(merged),
        Dense(1, activation='sigmoid', name='M')(merged),
        Dense(1, activation='sigmoid', name='O')(merged)
    ]

    return Model(inputs=[left_input, right_input], outputs=outputs)


# --------------------------
# 注意力子模块
# --------------------------
class ChannelAttention(Layer):
    """通道注意力机制"""

    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channel = input_shape[-1]
        self.dense1 = Dense(self.channel // self.ratio, activation='relu')
        self.dense2 = Dense(self.channel, activation='sigmoid')

    def call(self, inputs):
        x = GlobalAveragePooling2D()(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return Reshape((1, 1, self.channel))(x)

    def get_config(self):  # 关键修正
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


class SpatialAttention(Layer):
    """空间注意力机制"""

    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = Conv2D(1, 7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=3, keepdims=True)
        x = Concatenate()([avg_out, max_out])
        return self.conv(x)


# --------------------------
# 自定义加权损失
# --------------------------
class WeightedBinaryCrossentropy(keras.losses.Loss):
    def __init__(self, pos_weight=1.0, neg_weight=1.0, name='weighted_bce'):
        super().__init__(name=name)
        self.pos_weight = tf.convert_to_tensor(pos_weight)
        self.neg_weight = tf.convert_to_tensor(neg_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        loss = tf.reduce_mean(
            -(self.pos_weight * y_true * tf.math.log(y_pred + 1e-7) +
              self.neg_weight * (1 - y_true) * tf.math.log(1 - y_pred + 1e-7))
        )
        return loss


# --------------------------
# 训练流程
# --------------------------
if __name__ == "__main__":
    # 初始化组件
    loader = EyePairDataLoader("data_test", "Archive/full_df.csv")
    train_dataset = loader.get_dataset()
    val_dataset = loader.get_dataset(is_train=False)
    model = build_multi_label_model()

    # 获取任务权重
    task_weights = loader._calculate_task_weights()

    # 配置多任务损失
    loss_dict = {
        task: WeightedBinaryCrossentropy(
            pos_weight=task_weights[task][1],
            neg_weight=task_weights[task][0]
        ) for task in loader.label_cols
    }

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=loss_dict,
        metrics={
            task: [keras.metrics.AUC(name='auc'),
                   keras.metrics.BinaryAccuracy(name='acc')]
            for task in loader.label_cols
        }
    )

    # 回调函数
    checkpoint = ModelCheckpoint(
        'saved_models/best_model.keras',
        monitor='val_loss',
        save_best_only=True
    )

    # 增加学习率调度器
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )

    # 增加早停机制
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # 训练模型
    history = model.fit(
        train_dataset.map(lambda x, y, w: ((x[0], x[1]), y)),  # 重组输入格式
        validation_data=val_dataset.map(lambda x, y, w: ((x[0], x[1]), y)),
        epochs=50,
        callbacks=[checkpoint, lr_scheduler, early_stopping],
    )

    # 保存最终模型
    model.save('saved_models/final_model.keras')
    print("训练完成，模型已保存")
