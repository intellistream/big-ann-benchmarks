import os
import tensorflow as tf

# 你的数据集所在的目录
data_dir = "/mnt/c/Users/Administrator/Desktop/test/yt8m/data/video/"
output_rgb_file = "rgb_vectors.txt"
output_id_file = "video_ids.txt"

# 统计向量个数
vector_count = 0


def parse_tfrecord(example_proto):
    """解析 TFRecord 文件"""
    feature_description = {
        "id": tf.io.FixedLenFeature([], tf.string),
        "mean_rgb": tf.io.FixedLenFeature([1024], tf.float32),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def process_tfrecord_file(filepath, rgb_out, id_out):
    """解析单个 TFRecord 文件"""
    global vector_count  # 使用全局变量统计向量个数
    raw_dataset = tf.data.TFRecordDataset(filepath)

    for raw_record in raw_dataset:
        example = parse_tfrecord(raw_record)
        video_id = example["id"].numpy().decode("utf-8")  # 视频 ID
        mean_rgb = example["mean_rgb"].numpy()  # 1024 维向量

        # 检查向量维度
        if len(mean_rgb) != 1024:
            print(f"警告：视频 {video_id} 的 RGB 向量维度错误 ({len(mean_rgb)} 维)，跳过")
            continue

        # 保存 RGB 特征
        rgb_out.write(" ".join(map(str, mean_rgb)) + "\n\n")  # 空行分隔

        # 保存 ID
        id_out.write(video_id + "\n")

        vector_count += 1  # 统计向量数量


def process_all_tfrecords():
    """遍历所有 TFRecord 文件并处理"""
    global vector_count
    vector_count = 0  # 初始化计数
    with open(output_rgb_file, "w") as rgb_out, open(output_id_file, "w") as id_out:
        for filename in os.listdir(data_dir):
            if filename.endswith(".tfrecord"):
                filepath = os.path.join(data_dir, filename)
                print(f"Processing {filepath} ...")
                process_tfrecord_file(filepath, rgb_out, id_out)

    # 处理完成后打印总数
    print(f"\n处理完成，共提取 {vector_count} 个向量。")


# 运行数据处理
process_all_tfrecords()
