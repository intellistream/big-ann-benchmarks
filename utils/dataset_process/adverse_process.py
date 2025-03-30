import numpy as np
from tqdm import tqdm
import os

def save_fbin(filename, data):
    """ 保存数据集为 .fbin 格式 """
    nvecs, dim = data.shape
    with open(filename, "wb") as f:
        np.array([nvecs, dim], dtype=np.int32).tofile(f)  # 写入头信息
        for vec in tqdm(data, desc=f"Writing {filename}"):
            vec.astype(np.float32).tofile(f)  # 写入向量数据

def save_txt(filename, labels):
    """ 保存标签集为 .txt 格式 """
    with open(filename, "w") as f:
        for label in tqdm(labels, desc=f"Writing {filename}"):
            f.write(f"{label}\n")  # 每行写入一个标签值

# 读取 .npy 文件
data_file = "../../data/adverse/filter/data_1000000_100.npy"
query_file = "../../data/adverse/filter/queries_9900_100.npy"
data_labels_file = "../../data/adverse/filter/base_matadata.npy"
query_labels_file = "../../data/adverse/filter/query_metadata.npy"

data = np.load(data_file)
queries = np.load(query_file)
data_labels = np.load(data_labels_file)
query_labels = np.load(query_labels_file)

# 保存数据集（fbin 格式）
save_fbin("../../data/adverse/filter/data_1000000_100", data)
save_fbin("../../data/adverse/filter/queries_9900_100", queries)

# 保存标签集（txt 格式）
save_txt("../../data/adverse/filter/base_matadata.txt", data_labels)
save_txt("../../data/adverse/filter/query_metadata.txt", query_labels)

print("数据转换完成！")
