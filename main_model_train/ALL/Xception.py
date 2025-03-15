import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.utils import Sequence
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from keras.src.applications.xception import Xception
from keras import layers, Model
import keras

from config import ROOT_DIR

# 配置参数
DATA_DIR = ROOT_DIR / "dataset/Archive/preprocessed_images"
CSV_PATH = ROOT_DIR / "dataset/Archive/full_df.csv"
IMAGE_SIZE = 299
BATCH_SIZE = 32
CLASS_NAMES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# 数据增强序列
aug_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-15, 15), shear=(-8, 8), translate_px=(-20, 20)),
    iaa.GaussianBlur(sigma=(0, 1.5)),
    iaa.LinearContrast((0.8, 1.2)),
    iaa.AddToHueAndSaturation((-20, 20))
])


# 改进的数据生成器（包含过采样）
class EnhancedEyeGenerator(Sequence):
    def __init__(self, df, class_weights, batch_size=32, augment=True):
        super().__init__()
        self.df = self._oversample_minority(df)
        self.batch_size = batch_size
        self.augment = augment
        self.class_weights = class_weights
        self._prepare_data()

    def _oversample_minority(self, df):
        # 获取每个类别样本数
        class_counts = df[CLASS_NAMES].sum(axis=0)
        median_count = np.median(class_counts)

        # 创建过采样后的数据集
        dfs = [df]
        for idx, cls in enumerate(CLASS_NAMES):
            if class_counts[idx] < median_count * 0.3:
                minority_df = df[df[cls] == 1]
                repeat_times = int(median_count / class_counts[idx])
                dfs.append(minority_df.sample(n=len(minority_df) * repeat_times, replace=True))

        return pd.concat(dfs).sample(frac=1).reset_index(drop=True)

    def _prepare_data(self):
        self.sample_weights = []
        for _, row in self.df.iterrows():
            weight = sum([self.class_weights[i] for i, val in enumerate(row[CLASS_NAMES]) if val == 1])
            self.sample_weights.append(weight)
        self.sample_weights = np.array(self.sample_weights) / sum(self.sample_weights)

        self.indices = np.arange(len(self.df))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = np.random.choice(self.indices, size=self.batch_size, p=self.sample_weights)
        return self._generate_batch(batch_indices)

    def _generate_batch(self, batch_indices):
        X = np.zeros((len(batch_indices), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        y = np.zeros((len(batch_indices), len(CLASS_NAMES)), dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            row = self.df.iloc[idx]
            img = self._load_image(row)
            if img is None:
                continue
            X[i] = img
            y[i] = row[CLASS_NAMES].values.astype(np.float32)

        return X, y

    def _load_image(self, row):
        if not os.path.exists(os.path.join(DATA_DIR, f"{row['ID']}_left.jpg")) or \
                not os.path.exists(os.path.join(DATA_DIR, f"{row['ID']}_right.jpg")):
            return None
        img_left = cv2.imread(os.path.join(DATA_DIR, f"{row['ID']}_left.jpg"))
        img_right = cv2.imread(os.path.join(DATA_DIR, f"{row['ID']}_right.jpg"))
        img_left = cv2.resize(img_left, (IMAGE_SIZE, IMAGE_SIZE))
        img_right = cv2.resize(img_right, (IMAGE_SIZE, IMAGE_SIZE))

        if self.augment:
            img_left = aug_seq.augment_image(img_left)
            img_right = aug_seq.augment_image(img_right)

        # 图片左右拼接并改为正方形
        img = np.concatenate([img_left, img_right], axis=1)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imwrite('test.jpg', img)

        return img / 255.0


# 构建Xception模型
def build_xception():
    base = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # 自定义顶层
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation='sigmoid')(x)

    return Model(inputs=base.input, outputs=outputs)

# 自定义指标类
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

# 分步训练函数
def train_in_two_stages():
    # 数据准备
    df = pd.read_csv(CSV_PATH)
    train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df[CLASS_NAMES].sum(axis=1))

    # 计算类别权重
    class_counts = train_df[CLASS_NAMES].sum(axis=0)
    class_weights = (1 / (class_counts + 1e-6)) ** 0.5

    # 创建生成器
    train_gen = EnhancedEyeGenerator(train_df, class_weights, BATCH_SIZE)
    valid_gen = EnhancedEyeGenerator(valid_df, class_weights, BATCH_SIZE, augment=False)

    # 第一阶段：训练顶层
    model = build_xception()
    model.layers[0].trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=weighted_bce,
        metrics=[
            'accuracy',
            MacroRecall(num_classes=len(CLASS_NAMES)),
            MacroF1(num_classes=len(CLASS_NAMES)),
        ]
    )

    phase1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=20,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                'phase1_best.h5',
                save_best_only=True,
                monitor='val_macro_f1',
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=2,
                factor=0.5,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_macro_f1',
                mode='max',
                patience=5,
                verbose=1,
                restore_best_weights=True
            )
        ]
    )

    # 第二阶段：微调整个模型
    model = keras.models.load_model('phase1_best.h5', custom_objects={
        'weighted_bce': weighted_bce,
        'MacroRecall': lambda: MacroRecall(len(CLASS_NAMES)),
        'MacroF1': lambda: MacroF1(len(CLASS_NAMES))
    })
    model.layers[0].trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=weighted_bce,
        metrics=[
            'accuracy',
            MacroRecall(num_classes=len(CLASS_NAMES)),
            MacroF1(num_classes=len(CLASS_NAMES)),
        ]
    )

    phase2 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=30,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                'final_model.h5',
                save_best_only=True,
                monitor='val_macro_f1',
                mode='max'
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_macro_f1',
                mode='max',
                patience=5,
                verbose=1,
                restore_best_weights=True
            )
        ]
    )


# 改进的加权损失函数
def weighted_bce(y_true, y_pred):
    df = pd.read_csv(CSV_PATH)
    class_counts = df[CLASS_NAMES].sum(axis=0)
    median = np.median(class_counts)
    class_weights = tf.cast(median / (class_counts + 1e-6), tf.float32)
    loss = keras.losses.binary_crossentropy(y_true, y_pred)
    weights = tf.reduce_sum(class_weights * y_true, axis=-1) + tf.reduce_sum(1.0 * (1 - y_true), axis=-1)
    return tf.reduce_mean(loss * weights)


if __name__ == "__main__":
    train_in_two_stages()
