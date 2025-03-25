import struct
import numpy as np
import os

def convert_gt_bin_to_ibin(gt_bin_path, gt_ibin_path, num_queries, K):
    """ 将 `gt.bin` 转换为 `gt.ibin` 格式，包含索引和距离 """
    with open(gt_bin_path, "rb") as f:
        data = f.read()

    # `gt.bin` 存储 (IdxType, float) 对，每个 8 字节 (4 + 4)
    expected_size = num_queries * K * 8
    assert len(data) == expected_size, f"文件大小不匹配: 期望 {expected_size} 字节, 但实际 {len(data)} 字节"

    # 解包所有的 (IdxType, float) 对
    unpacked_data = struct.unpack(f"{num_queries * K * 2}f", data)

    # 提取索引和距离
    neighbors = np.array(unpacked_data[0::2], dtype=np.int32).reshape(num_queries, K)
    distances = np.array(unpacked_data[1::2], dtype=np.float32).reshape(num_queries, K)

    # 写入 `.ibin` 文件
    with open(gt_ibin_path, "wb") as f:
        np.array([num_queries, K], dtype=np.int32).tofile(f)  # 写入头部
        neighbors.tofile(f)  # 写入索引数据
        distances.tofile(f)  # 写入距离数据

    print(f"✅ 转换完成！已生成 `{gt_ibin_path}`")
    print(f"- 查询数: {num_queries}, 每个查询的 Top-{K} 近邻")

# ⚠ 需要手动提供 `num_queries` 和 `K`
convert_gt_bin_to_ibin("../../data/SIFT/gt_original", "../../data/SIFT/gt_processed", num_queries=10000, K=10)
