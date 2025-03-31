import os
import warnings

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from imgaug import augmenters as iaa
from keras import layers, Model
from keras.api.utils import Sequence
from keras.src.applications.xception import Xception
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings('ignore', category=FutureWarning)

from config import ROOT_DIR

# 配置参数
DATA_DIR = ROOT_DIR / "dataset/Archive/preprocessed_images"
CSV_PATH = ROOT_DIR / "dataset/Archive/full_df.csv"
IMAGE_SIZE = 299
BATCH_SIZE = 16
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 数据增强序列
aug_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-15, 15), shear=(-8, 8), translate_px=(-20, 20)),
    iaa.GaussianBlur(sigma=(0, 1.5)),
    iaa.LinearContrast((0.8, 1.2)),
    iaa.AddToHueAndSaturation((-20, 20))
])


class EnhancedEyeGenerator(Sequence):
    def __init__(self, df, class_weights, batch_size=32, augment=True):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.augment = augment
        self.class_weights = class_weights
        self._prepare_data()

    def _prepare_data(self):
        # 计算样本权重
        self.sample_weights = self.df[self.class_names].apply(
            lambda row: sum([self.class_weights[cls] for cls in self.class_names if row[cls] == 1]), axis=1
        )
        self.sample_weights = (self.sample_weights / self.sample_weights.sum()).values
        self.indices = np.arange(len(self.df))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = np.random.choice(self.indices, size=self.batch_size, p=self.sample_weights)
        return self._generate_batch(batch_indices)

    def _generate_batch(self, batch_indices):
        X = np.zeros((len(batch_indices), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        y = np.zeros((len(batch_indices), len(self.class_names)), dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            row = self.df.iloc[idx]
            img = self._load_image(row)
            if img is None:
                continue
            X[i] = img
            y[i] = row[self.class_names].values.astype(np.float32)

        return X, y

    def _load_image(self, row):
        img_path = row['image_path']
        if not os.path.exists(img_path):
            return None
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        if self.augment:
            img = aug_seq.augment_image(img)

        return img / 255.0

    @property
    def class_names(self):
        return [col for col in self.df.columns if col not in ['ID', 'keywords', 'image_path']]


class SEBlock(layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.se = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.channels // self.ratio, activation='relu'),
            layers.Dense(self.channels, activation='sigmoid'),
            layers.Reshape((1, 1, self.channels))
        ])
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.se(inputs)


def build_xception(num_classes):
    base = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    x = base.output
    x = SEBlock()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    return Model(inputs=base.input, outputs=outputs)


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


class CompositeEarlyStopping(keras.callbacks.Callback):
    def __init__(self, monitor_metrics=('accuracy', 'macro_recall', 'macro_f1'),
                 patience=4, verbose=1, restore_best_weights=True):
        super().__init__()
        self.metrics = monitor_metrics
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best_score = -np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_score = np.mean([logs.get(f'val_{m}') for m in self.metrics])
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print(f'\nEpoch {epoch}: Early stopping')
                    print(f'Best composite score: {self.best_score:.4f}')
                if self.restore_best_weights:
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True


def get_weighted_bce(class_weights):
    def weighted_bce(y_true, y_pred):
        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1) + 1.0
        return tf.reduce_mean(loss * weights)

    return weighted_bce


def prepare_data():
    # 原始数据加载
    df = pd.read_csv(CSV_PATH)

    # 创建样本数据集
    samples = []
    for _, row in df.iterrows():
        for eye in ['Left', 'Right']:
            img_path = DATA_DIR / f"{row['ID']}_{eye.lower()}.jpg"
            if img_path.exists():
                samples.append({
                    'ID': f"{row['ID']}_{eye}",
                    'image_path': str(img_path),
                    'keywords': row[f'{eye}-Diagnostic Keywords']
                })

    samples_df = pd.DataFrame(samples)
    samples_df['keywords'] = samples_df['keywords'].str.split(', ')

    # 多标签编码
    mlb = MultiLabelBinarizer()
    keyword_labels = mlb.fit_transform(samples_df['keywords'])
    class_names = mlb.classes_

    for idx, cls in enumerate(class_names):
        samples_df[cls] = keyword_labels[:, idx]

    return samples_df, class_names


def train_in_two_stages():
    # 数据准备
    samples_df, CLASS_NAMES = prepare_data()
    train_df, valid_df = train_test_split(samples_df, test_size=0.2, random_state=SEED)

    # 计算类别权重
    class_counts = train_df[CLASS_NAMES].sum(axis=0)
    class_weights = (1 / (class_counts + 1e-6)) ** 0.5
    class_weights_tensor = tf.constant(class_weights.values, dtype=tf.float32)

    # 创建生成器
    train_gen = EnhancedEyeGenerator(train_df, class_weights, BATCH_SIZE)
    valid_gen = EnhancedEyeGenerator(valid_df, class_weights, BATCH_SIZE, augment=False)

    # 第一阶段：训练顶层
    model = build_xception(len(CLASS_NAMES))
    model.layers[0].trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=get_weighted_bce(class_weights_tensor),
        metrics=[
            'accuracy',
            MacroRecall(num_classes=len(CLASS_NAMES)),
            MacroF1(num_classes=len(CLASS_NAMES)),
        ]
    )

    phase1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=100,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                'phase1_best.h5',
                save_best_only=True,
                monitor='val_macro_f1',
                mode='max'
            ),
            CompositeEarlyStopping(
                monitor_metrics=('accuracy', 'macro_recall', 'macro_f1'),
                patience=5,
                verbose=1,
                restore_best_weights=True
            )
        ]
    )

    # 第二阶段：微调整个模型
    model = keras.models.load_model('phase1_best.h5', custom_objects={
        'weighted_bce': get_weighted_bce(class_weights_tensor),
        'MacroRecall': lambda: MacroRecall(len(CLASS_NAMES)),
        'MacroF1': lambda: MacroF1(len(CLASS_NAMES)),
        'SEBlock': SEBlock
    })
    model.layers[0].trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=get_weighted_bce(class_weights_tensor),
        metrics=[
            'accuracy',
            MacroRecall(num_classes=len(CLASS_NAMES)),
            MacroF1(num_classes=len(CLASS_NAMES)),
        ]
    )

    phase2 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=100,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                'final_model.h5',
                save_best_only=True,
                monitor='val_macro_f1',
                mode='max'
            ),
            CompositeEarlyStopping(
                monitor_metrics=('accuracy', 'macro_recall', 'macro_f1'),
                patience=5,
                verbose=1,
                restore_best_weights=True
            )
        ]
    )


if __name__ == "__main__":
    train_in_two_stages()
