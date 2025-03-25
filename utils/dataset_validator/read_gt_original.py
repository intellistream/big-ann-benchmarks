import struct
import os

def read_gt_file(filename, num_queries, K, num_rows=5):
    """
    读取 ground truth (GT) 文件，并显示前 num_rows 行的数据。

    :param filename: GT 文件路径
    :param num_queries: 查询向量的数量
    :param K: 每个查询的最近邻数量
    :param num_rows: 要显示的查询数量（默认 5）
    """
    file_size = os.path.getsize(filename)
    expected_size = num_queries * K * 8  # 每个 (索引, 距离) 对占 8 字节

    print(f"Ground truth file size: {file_size} bytes, Expected size: {expected_size} bytes")

    if file_size != expected_size:
        raise ValueError(f"Error: File size ({file_size} bytes) does not match expected size ({expected_size} bytes)!")

    with open(filename, "rb") as f:
        data = f.read()

    fmt = f"{num_queries * K * 2}I"  # 解析所有的 uint32_t 和 float
    unpacked_data = struct.unpack(fmt, data)

    # 重新组织数据，转换为 (索引, 距离) 结构
    gt_array = []
    for i in range(num_queries):
        neighbors = []
        for j in range(K):
            idx = unpacked_data[i * K * 2 + j * 2]       # 取出索引
            dist = struct.unpack("f", struct.pack("I", unpacked_data[i * K * 2 + j * 2 + 1]))[0]  # 转换为 float
            neighbors.append((idx, dist))
        gt_array.append(neighbors)

    # 打印前 num_rows 个查询的结果
    print(f"\nGround Truth (前 {num_rows} 行数据):\n")
    for i in range(min(num_rows, num_queries)):
        print(f"Query {i}:")
        for j, (idx, dist) in enumerate(gt_array[i]):
            print(f"  Top-{j+1}: Index = {int(idx)}, Distance = {dist:.6f}")
        print("-" * 40)

    return gt_array

# 运行
gt_data = read_gt_file("../../data/SIFT/gt_original", num_queries=10000, K=10, num_rows=10)
