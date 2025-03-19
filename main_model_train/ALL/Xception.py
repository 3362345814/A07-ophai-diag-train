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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from config import ROOT_DIR

# 配置参数
DATA_DIR = ROOT_DIR / "dataset/Archive/preprocessed_images"
CSV_PATH = ROOT_DIR / "dataset/Archive/full_df.csv"
IMAGE_SIZE = 299
BATCH_SIZE = 16
CLASS_NAMES = ['D', 'G', 'C', 'A', 'H', 'M', 'O']
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
        print(median_count)
        print(class_counts)

        # 创建过采样后的数据集
        dfs = [df]
        for idx, cls in enumerate(CLASS_NAMES):
            if class_counts[idx] < median_count:
                print(cls)
                minority_df = df[df[cls] == 1]
                repeat_times = int(median_count / class_counts[idx] * 2)
                print(repeat_times)
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

        return img / 255.0


# 构建Xception模型
# 新增SE注意力模块
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

# 修改后的模型构建函数
def build_xception():
    base = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # 添加SE注意力模块
    x = base.output
    x = SEBlock()(x)  # 新增SE注意力层
    x = layers.GlobalAveragePooling2D()(x)
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
# 在MacroF1类后添加自定义早停回调
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

# 修改两处EarlyStopping回调（第一阶段和第二阶段）
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
        epochs=100,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                'phase1_best.h5',
                save_best_only=True,
                monitor='val_macro_f1',
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_macro_f1',
                patience=2,
                factor=0.5,
                verbose=1
            ),
            CompositeEarlyStopping(
                monitor_metrics=('accuracy', 'macro_recall', 'macro_f1'),
                patience=5,
                verbose=1,
                restore_best_weights=True
            ),
            keras.callbacks.TensorBoard(
                log_dir='./logs/stage1',
                histogram_freq=1,
                update_freq='epoch',
                write_graph=True,
                write_images=False
            )
        ]
    )

    # 第二阶段：微调整个模型
    model = keras.models.load_model('phase1_best.h5', custom_objects={
        'weighted_bce': weighted_bce,
        'MacroRecall': lambda: MacroRecall(len(CLASS_NAMES)),
        'MacroF1': lambda: MacroF1(len(CLASS_NAMES)),
        'SEBlock': SEBlock  # 添加SEBlock到custom_objects
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
        epochs=100,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                # 添加保存时间戳
                'final_model' + '_' + str(pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") + '.h5'),
                save_best_only=True,
                monitor='val_macro_f1',
                mode='max'
            ),
            CompositeEarlyStopping(
                monitor_metrics=('accuracy', 'macro_recall', 'macro_f1'),
                patience=5,
                verbose=1,
                restore_best_weights=True
            ),
            keras.callbacks.TensorBoard(
                log_dir='./logs/stage2',
                histogram_freq=1,
                update_freq='epoch',
                write_graph=True,
                write_images=False
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