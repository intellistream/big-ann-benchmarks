import struct
import numpy as np


def read_u8bin_file(filename, num_vectors_to_print=5):
    with open(filename, 'rb') as f:
        # 读取前两个 uint32 数字
        num_queries = struct.unpack('I', f.read(4))[0]  # 查询向量数量
        dim = struct.unpack('I', f.read(4))[0]  # 每个向量的维度

        print(f"Total vectors: {num_queries}")
        print(f"Vector dimension: {dim}\n")

        # 读取查询向量数据
        total_vectors = min(num_vectors_to_print, num_queries)  # 控制输出数量
        vectors = np.fromfile(f, dtype=np.uint8, count=total_vectors * dim)
        vectors = vectors.reshape((total_vectors, dim))

        # 打印部分向量
        for i, vec in enumerate(vectors):
            print(f"Vector {i}: {vec}")

        print("\n(Only showing first {} vectors)".format(total_vectors))


read_u8bin_file('../../data/adverse/filter/queries_9900_100', num_vectors_to_print=10)