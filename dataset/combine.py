import os
import pandas as pd
from glob import glob

# 定义目标列顺序
target_columns = ['image_name', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# 初始化结果列表
combined_data = []

# ----------------- 处理第一个数据集 (AMD) -----------------
amd_labels = pd.read_csv('AMD/label.csv')
for _, row in amd_labels.iterrows():
    image_path = os.path.join('AMD/OriginalImages', row['fnames'])

    # 初始化所有标签为0
    labels = {col: 0 for col in target_columns[1:]}

    # 映射原始标签
    labels['N'] = row['normal']
    labels['D'] = row['DR'] = 0 if row['DR'] == 0 else 1
    labels['G'] = row['glaucoma']
    labels['A'] = row['AMD'] = 0 if row['AMD'] == 0 else 1
    labels['H'] = row['hyper']
    labels['M'] = row['myopia']

    # 处理其他类别（RVO/LS/others任一为1则O=1）
    labels['O'] = 1 if any([row['RVO'], row['LS'], row['others']]) else 0

    combined_data.append([image_path] + [labels[col] for col in target_columns[1:]])

# ----------------- 处理第二个数据集 (C_D_G) -----------------
class_mapping = {
    'normal':      {'N': 1},
    'cataract':    {'C': 1},
    'glaucoma':    {'G': 1},
    'diabetic_retinopathy': {'D': 1}
}

for class_name in class_mapping:
    class_path = os.path.join('C_D_G', class_name)
    for img_path in glob(os.path.join(class_path, '*')):
        # 初始化所有标签为0
        labels = {col: 0 for col in target_columns[1:]}
        # 设置对应类别标签
        labels.update(class_mapping[class_name])
        combined_data.append([img_path] + [labels[col] for col in target_columns[1:]])

# ----------------- 处理第三个数据集 (eyepac) -----------------
eyepac_meta = pd.read_csv('eyepac-light-v2-512-jpg/metadata.csv')

# 遍历所有可能的子目录
for split in ['train', 'test', 'validation']:
    split_path = os.path.join('eyepac-light-v2-512-jpg', split)
    if not os.path.exists(split_path):
        continue
    for subdir in ['RG', 'NRG']:
        subdir_path = os.path.join(split_path, subdir)
        if not os.path.exists(subdir_path):
            continue

    # 获取该split下的所有图片
    for img_file in os.listdir(subdir_path):
        full_path = os.path.join(subdir_path, img_file)
        # 在metadata中查找对应的标签
        match = eyepac_meta[eyepac_meta['file_name'] == img_file]
        if not match.empty:
            label = match.iloc[0]['label_binary']
            labels = {col: 0 for col in target_columns[1:]}
            labels['G'] = label
            labels['N'] = 1 - label  # 假设阴性样本都是正常
            combined_data.append([full_path] + [labels[col] for col in target_columns[1:]])

# ----------------- 创建最终DataFrame并保存 -----------------
df_combined = pd.DataFrame(combined_data, columns=target_columns)

# 去重（如果有重复路径）
df_combined = df_combined.drop_duplicates(subset=['image_name'])
df_combined['image_name'] = 'dataset/' + df_combined['image_name']
# 保存结果
df_combined.to_csv('combined_dataset.csv', index=False)

print(f"数据集合并完成，共 {len(df_combined)} 条记录")
print("结果已保存至 combined_dataset.csv")