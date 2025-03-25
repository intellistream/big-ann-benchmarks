import numpy as np
from scipy.sparse import csr_matrix


def read_sparse_matrix(fname, num_rows_to_print=5):
    with open(fname, "rb") as f:
        # 读取矩阵形状和非零元素数量
        n, d, nnz = np.fromfile(f, dtype=np.int64, count=3)

        print(f"Matrix shape: ({n}, {d})")
        print(f"Number of non-zero elements: {nnz}\n")

        # 读取 indptr 数组
        indptr = np.fromfile(f, dtype=np.int64, count=n + 1)

        # 读取 indices 数组
        indices = np.fromfile(f, dtype=np.int32, count=nnz)

        # 读取 data 数组
        data = np.fromfile(f, dtype=np.float32, count=nnz)

        # 重建稀疏矩阵
        sparse_matrix = csr_matrix((data, indices, indptr), shape=(n, d))

        # 检查索引范围
        print("Max index value in indices:", sparse_matrix.indices.max())
        print("Min index value in indices:", sparse_matrix.indices.min())
        print("")

        # 打印部分数据
        for i in range(min(num_rows_to_print, n)):
            row = sparse_matrix.getrow(i)
            print(f"Row {i}: indices={row.indices}, data={row.data}")

        print("\n(Only showing first {} rows)".format(min(num_rows_to_print, n)))

    return sparse_matrix


# read_sparse_matrix('../../data/random-filter100000/data_metadata_100000_50', num_rows_to_print=100000)

read_sparse_matrix('../../data/MSONG/base_metadata.spmat', num_rows_to_print=100)