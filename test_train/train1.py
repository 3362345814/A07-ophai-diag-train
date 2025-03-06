import os
import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.src.layers import *
from keras.src.optimizers import Adam
from keras.src.applications.xception import Xception
from keras.src.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from albumentations import (
    Compose, HorizontalFlip, Rotate, RandomBrightnessContrast,
    ElasticTransform, GridDistortion, OpticalDistortion
)


# 配置参数
class Config:
    IMG_SIZE = 512
    BATCH_SIZE = 16
    EPOCHS = 50
    LR_START = 1e-4
    LR_END = 1e-6
    NUM_CLASSES = 8
    DATA_PATH = "../Archive"
    SAVE_DIR = "../saved_models"
    TEST_SIZE = 0.2
    SEED = 42


config = Config()


# 数据预处理增强管道
def get_augmenter():
    return Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=15, p=0.5),
        RandomBrightnessContrast(p=0.5),
        ElasticTransform(p=0.3),
        GridDistortion(p=0.2),
    ], additional_targets={'image_right': 'image'})


# 双目数据生成器
class EyePairGenerator(keras.utils.Sequence):
    def __init__(self, df, augmenter=None, is_train=True):
        super().__init__()
        self.df = df
        self.augmenter = augmenter
        self.is_train = is_train
        self.indices = np.arange(len(df))

    def __len__(self):
        return len(self.df) // config.BATCH_SIZE

    def __getitem__(self, index):
        batch_indices = self.indices[index * config.BATCH_SIZE:(index + 1) * config.BATCH_SIZE]
        batch_data = self.df.iloc[batch_indices]

        left_images = []
        right_images = []
        labels = []

        for _, row in batch_data.iterrows():
            # 读取并预处理左右眼图像
            left = self._load_and_preprocess(row['left_path'])
            right = self._load_and_preprocess(row['right_path'])

            # 应用同步数据增强
            if self.augmenter and self.is_train:
                augmented = self.augmenter(
                    image=left,
                    image_right=right
                )
                left = augmented['image']
                right = augmented['image_right']

            left_images.append(left)
            right_images.append(right)
            labels.append(row[['G', 'C', 'A', 'H', 'M', 'O']].values.astype(np.float32))

        return [np.array(left_images), np.array(right_images)], np.array(labels)

    def _load_and_preprocess(self, path):
        # 医学图像预处理
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

        # CLAHE增强
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 归一化
        return img.astype(np.float32) / 255.0


# 构建双目模型
def build_binocular_model():
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(512, 512, 3)
    )

    # 冻结预训练层
    base_model.trainable = False

    # 双目输入
    left_input = Input(shape=(512, 512, 3))
    right_input = Input(shape=(512, 512, 3))

    # 共享特征提取
    left_features = base_model(left_input)  # (16,16,2048)
    right_features = base_model(right_input)

    # 注意力融合
    attended = Attention()([left_features, right_features])

    # 全局池化
    pooled = GlobalAveragePooling2D()(attended)

    # 分类头
    outputs = Dense(8, activation='sigmoid')(pooled)

    return Model(inputs=[left_input, right_input], outputs=outputs)


# 自定义注意力层
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[0][-1], input_shape[1][-1]),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        left, right = inputs
        e = tf.matmul(left, tf.matmul(self.W, right), transpose_b=True)
        a = tf.nn.softmax(e, axis=-1)
        return tf.matmul(a, right)


# 学习率调度器
def lr_scheduler(epoch):
    lr = config.LR_START * (config.LR_END / config.LR_START) ** (epoch / config.EPOCHS)
    return lr


# 主训练流程
def main():
    # 准备数据
    df = pd.read_csv(os.path.join(config.DATA_PATH, "full_df.csv"))
    train_df, val_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.SEED
    )

    # 数据增强
    augmenter = get_augmenter()

    # 创建数据生成器
    train_gen = EyePairGenerator(train_df, augmenter=augmenter)
    val_gen = EyePairGenerator(val_df, is_train=False)

    # 构建模型
    model = build_binocular_model()

    # 编译模型
    model.compile(
        optimizer=Adam(config.LR_START),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # 回调函数
    callbacks = [
        ModelCheckpoint(
            os.path.join(config.SAVE_DIR, "best_model.h5"),
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        ),
        LearningRateScheduler(lr_scheduler)
    ]

    # 训练模型
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )

    # 保存最终模型
    model.save(os.path.join(config.SAVE_DIR, "final_model.h5"))

    # 保存训练日志
    pd.DataFrame(history.history).to_csv(
        os.path.join(config.SAVE_DIR, "training_log.csv"), index=False
    )


if __name__ == "__main__":
    # 创建保存目录
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # 设置GPU配置
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    main()
