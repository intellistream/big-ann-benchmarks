
import struct
import numpy as np
import random
from collections import Counter


def read_u8bin_file(filename):
    """读取 u8bin 文件并返回数据集"""
    with open(filename, 'rb') as f:
        num_vectors = struct.unpack('I', f.read(4))[0]  # 向量数量
        dim = struct.unpack('I', f.read(4))[0]  # 维度
        data = np.fromfile(f, dtype=np.uint8).reshape((num_vectors, dim))
    return num_vectors, dim, data


def read_labels_file(filename):
    """读取标签文件并返回标签列表"""
    with open(filename, 'r', encoding='utf-8') as f:
        labels = [line.strip().split(',') for line in f.readlines()]
    return labels


def write_u8bin_file(filename, data):
    num_vectors, dim = data.shape
    data = data.astype(np.float32)  # 确保数据是 float32

    with open(filename, 'wb') as f:
        np.array([num_vectors, dim], dtype=np.int32).tofile(f)  # 写入文件头 (n, d)
        data.tofile(f)  # 写入数据


def write_labels_file(filename, labels):
    """将标签写入 txt 文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join([','.join(map(str, l)) for l in labels]) + '\n')


# 读取数据集和标签集
vector_file = '../../data/YouTube-audio/filter/vectors_audio.u8bin'
label_file = '../../data/YouTube-audio/filter/labels_audio.txt'
print("正在读取向量数据...")
num_vectors, dim, vectors = read_u8bin_file(vector_file)
print(f"总向量数: {num_vectors}, 维度: {dim}")
assert num_vectors == 1120231, "数据集数量不匹配！"
assert dim == 128, "向量维度不匹配！"

print("正在读取标签数据...")
labels = read_labels_file(label_file)
assert len(labels) == num_vectors, "向量数据与标签数据数量不匹配！"

# 统计标签出现频率
print("统计标签出现频率...")
label_counter = Counter(label for sublist in labels for label in sublist)

# 随机抽取 10000 个候选查询数据
print("正在抽取 10000 个候选查询数据...")
candidate_indices = set(random.sample(range(num_vectors), 100000))
candidate_vectors = [vectors[i] for i in candidate_indices]
candidate_labels = [labels[i] for i in candidate_indices]

# 从剩余数据中随机抽取 1000000 条
print("正在抽取 1000000 个基础数据...")
remaining_indices = list(set(range(num_vectors)) - candidate_indices)
base_indices = set(random.sample(remaining_indices, 1000000))
base_vectors = np.array([vectors[i] for i in base_indices], dtype=np.uint8)
base_labels = [labels[i] for i in base_indices]

# 筛选符合条件的查询数据
print("筛选符合条件的查询数据...")
final_query_vectors = []
final_query_labels = []

# 比较查询集标签与选定基础数据的标签
for i, label_set in enumerate(candidate_labels):
    # 只保留最多 2 个标签，优先选取出现频率最高的
    sorted_labels = sorted(label_set, key=lambda x: label_counter[x], reverse=True)
    filtered_labels = sorted_labels[:2] if len(sorted_labels) > 2 else sorted_labels

    # 统计与数据库中标签集的交集
    matching_count = sum(1 for lbls in base_labels if set(filtered_labels).issubset(lbls))

    if matching_count >= 10:  # 至少有 10 个匹配对象
        final_query_vectors.append(candidate_vectors[i])
        final_query_labels.append(filtered_labels)
        if len(final_query_vectors) == 1000:
            break  # 选够 1000 个查询数据
        elif len(final_query_vectors)%100 == 0:
            print(len(final_query_vectors))

assert len(final_query_vectors) == 1000, "未能筛选出足够的查询向量！"

# 写入数据文件
print("正在写入数据文件...")
write_u8bin_file('../../data/YouTube-audio/filter/queries_1000_128.u8bin', np.array(final_query_vectors, dtype=np.uint8))
write_labels_file('../../data/YouTube-audio/filter/query_metadata.txt', final_query_labels)
write_u8bin_file('../../data/YouTube-audio/filter/data_1000000_128.u8bin', base_vectors)
write_labels_file('../../data/YouTube-audio/filter/base_metadata.txt', base_labels)

print("数据处理完成，文件已成功生成！")


