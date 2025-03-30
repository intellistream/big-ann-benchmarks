import numpy as np
from scipy.sparse import csr_matrix


def get_voc_size(label_file):
    """
    计算 voc_size（标签 ID 的最大值 + 1）
    :param label_file: 标签文件路径
    :return: 词汇表大小 voc_size
    """
    max_label = 0
    with open(label_file, 'r') as f:
        for line in f:
            labels = list(map(int, line.strip().split(',')))
            if labels:
                max_label = max(max_label, max(labels))
    return max_label + 1  # voc_size = 最大标签 + 1


def convert_labels_to_csr(label_file, voc_size):
    """
    读取文本格式的标签文件，并转换为 CSR 稀疏矩阵
    :param label_file: 标签文件路径，每行是以逗号分隔的标签 ID
    :param voc_size: 词汇表大小（所有可能的标签 ID 的最大值 + 1）
    :return: CSR 格式的标签矩阵
    """
    indices = []
    indptr = [0]
    data = []

    with open(label_file, 'r') as f:
        for line in f:
            labels = list(map(int, line.strip().split(',')))  # 读取并转换为整数列表
            indices.extend(labels)  # 添加索引
            data.extend([1] * len(labels))  # 添加数据值 1.0
            indptr.append(len(indices))  # 记录行索引结束位置

    # 构造 CSR 矩阵
    num_rows = len(indptr) - 1  # 行数（向量数）
    csr_mat = csr_matrix((data, indices, indptr), shape=(num_rows, voc_size), dtype=np.float32)

    return csr_mat


def write_sparse_matrix(mat, fname):
    """
    将 CSR 矩阵保存为 `.spmat` 格式
    :param mat: CSR 稀疏矩阵
    :param fname: 输出文件路径
    """
    with open(fname, "wb") as f:
        sizes = np.array([mat.shape[0], mat.shape[1], mat.nnz], dtype='int64')
        sizes.tofile(f)
        mat.indptr.astype('int64').tofile(f)
        mat.indices.astype('int32').tofile(f)
        mat.data.astype('float32').tofile(f)


# **运行示例**
label_file = "../../data/YouTube-audio/filter/base_metadata.txt"  # 输入的文本标签文件

# 自动计算 voc_size
voc_size = get_voc_size(label_file)
print(f"Computed voc_size: {voc_size}")

# 转换文本标签文件为 CSR 格式
csr_labels = convert_labels_to_csr(label_file, voc_size)

# 打印部分数据检查
print(f"Matrix shape: {csr_labels.shape}")
print(f"Number of non-zero elements: {csr_labels.nnz}")
print(
    f"Row 0: indices={csr_labels.indices[csr_labels.indptr[0]:csr_labels.indptr[1]]}, data={csr_labels.data[csr_labels.indptr[0]:csr_labels.indptr[1]]}")
print(
    f"Row 1: indices={csr_labels.indices[csr_labels.indptr[1]:csr_labels.indptr[2]]}, data={csr_labels.data[csr_labels.indptr[1]:csr_labels.indptr[2]]}")
print(
    f"Row 2: indices={csr_labels.indices[csr_labels.indptr[2]:csr_labels.indptr[3]]}, data={csr_labels.data[csr_labels.indptr[2]:csr_labels.indptr[3]]}")

# **保存 CSR 矩阵**
output_file = "../../data/YouTube-audio/filter/base_metadata.spmat"
write_sparse_matrix(csr_labels, output_file)
print(f"Saved CSR matrix to {output_file}")
