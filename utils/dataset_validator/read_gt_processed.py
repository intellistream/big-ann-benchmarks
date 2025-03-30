import numpy as np


def read_gt_ibin(fname, num_queries_to_print=5):
    with open(fname, "rb") as f:
        # 读取查询数量和最近邻个数
        nq, k = np.fromfile(f, dtype=np.int32, count=2)

        print(f"Number of queries: {nq}")
        print(f"Top-k neighbors per query: {k}\n")

        # 读取最近邻索引
        neighbors = np.fromfile(f, dtype=np.int32, count=nq * k)

        # 重新塑形 (nq, k)
        neighbors = neighbors.reshape(nq, k)

        # 打印部分数据
        for i in range(min(num_queries_to_print, nq)):
            print(f"Query {i}: Nearest neighbors: {neighbors[i]}")

        print("\n(Only showing first {} queries)".format(min(num_queries_to_print, nq)))

    return neighbors

path = '../../data/YouTube-audio/filter/gt_1000000_1000_128'
read_gt_ibin(path, num_queries_to_print=1000)