import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import os

def generate_gaussian_mixture(
    num_clusters, points_per_cluster, d, intra_cluster_var=0.01, inter_cluster_var=1
):
    data = []
    filter_values = []
    means = []
    for i in range(num_clusters):
        mean = np.random.multivariate_normal(np.zeros(d), np.eye(d) * inter_cluster_var)
        points = np.random.multivariate_normal(
            mean, np.eye(d) * intra_cluster_var, points_per_cluster
        )
        data.append(points)

        # ✅ 先加扰动，再取整，最后防止负数
        perturbed_labels = np.round(i - 0.5 + np.random.uniform(size=points_per_cluster))
        perturbed_labels = np.clip(perturbed_labels, 0, num_clusters - 1)  # 确保标签不小于0
        filter_values.append(perturbed_labels)

        means.append(mean)

    return np.concatenate(data), np.concatenate(filter_values).astype(int), np.vstack(means)


N = 1000000  # 数据集大小
NUM_CLUSTERS = 100  # 类别数
DIM = 100  # 维度
INTRA_CLUSTER_VAR = 0.01  # 类内方差
INTER_CLUSTER_VAR = 1  # 类间方差

#parser = argparse.ArgumentParser()
#parser.add_argument('output_dir', type=str, help='Output directory path')
#args = parser.parse_args()
OUTPUT_DIR = '../../data/adverse/filter'

# 生成数据集
data, filters, means = generate_gaussian_mixture(
    num_clusters=NUM_CLUSTERS,
    points_per_cluster=N // NUM_CLUSTERS,
    d=DIM,
    intra_cluster_var=INTRA_CLUSTER_VAR,
    inter_cluster_var=INTER_CLUSTER_VAR,
)

queries = []
query_filters = []

for query_cluster in tqdm(range(NUM_CLUSTERS)):
    for gt_cluster in range(NUM_CLUSTERS):
        if query_cluster == gt_cluster:
            continue

        query = np.random.multivariate_normal(
            means[query_cluster], np.eye(DIM) * INTRA_CLUSTER_VAR
        )
        queries.append(query)

        # ✅ 查询标签和数据集的标签保持一致
        query_filters.append(gt_cluster)

queries = np.array(queries)
query_filters = np.array(query_filters, dtype=np.int32)  # 强制转换为整数类型

data /= np.linalg.norm(data, axis=1, keepdims=True)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)


np.save(os.path.join(OUTPUT_DIR, f"adversarial-{DIM}-angular.npy"), data)
np.save(os.path.join(OUTPUT_DIR, f"adversarial-{DIM}-angular_queries.npy"), queries)
np.save(os.path.join(OUTPUT_DIR, f"adversarial-{DIM}-angular_filter-values.npy"), filters)
np.save(os.path.join(OUTPUT_DIR, f"adversarial-{DIM}-angular_queries_labels.npy"), query_filters)
