import os
import tensorflow as tf
import requests
import time
import json

# YouTube Data API Key（请替换为你的 API Key）
API_KEY = "AIzaSyBqLCvc11Nz2yfEq2vqy_HIEKihm6_8Q1Q"

# 数据目录 & 输出文件
data_dir = "/mnt/c/Users/Administrator/Desktop/test/yt8m/data/video/"
output_rgb_file = "rgb_vectors.txt"
output_id_file = "video_ids.txt"
output_time_file = "video_times.txt"

# 统计向量个数
vector_count = 0


def parse_tfrecord(example_proto):
    """解析 TFRecord 文件"""
    feature_description = {
        "id": tf.io.FixedLenFeature([], tf.string),
        "mean_rgb": tf.io.FixedLenFeature([1024], tf.float32),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def get_video_upload_time(video_id):
    """查询 YouTube API 获取视频发布时间"""
    url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=snippet&key={API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()

        if "items" in data and len(data["items"]) > 0:
            return data["items"][0]["snippet"]["publishedAt"]  # 获取发布时间
        else:
            print(f"警告：视频 {video_id} 未找到发布时间")
            return "Unknown"

    except Exception as e:
        print(f"错误：查询 {video_id} 失败，{e}")
        return "Error"


def process_tfrecord_file(filepath, rgb_out, id_out, time_out):
    """解析单个 TFRecord 文件"""
    global vector_count
    raw_dataset = tf.data.TFRecordDataset(filepath)

    for raw_record in raw_dataset:
        example = parse_tfrecord(raw_record)
        video_id = example["id"].numpy().decode("utf-8")
        mean_rgb = example["mean_rgb"].numpy()

        # 检查向量维度
        if len(mean_rgb) != 1024:
            print(f"警告：视频 {video_id} 的 RGB 向量维度错误 ({len(mean_rgb)} 维)，跳过")
            continue

        # 获取发布时间
        upload_time = get_video_upload_time(video_id)
        time.sleep(0.1)  # 避免 API 速率限制

        # 保存数据
        rgb_out.write(" ".join(map(str, mean_rgb)) + "\n\n")
        id_out.write(video_id + "\n")
        time_out.write(f"{video_id},{upload_time}\n")

        vector_count += 1


def process_all_tfrecords():
    """遍历所有 TFRecord 文件并处理"""
    global vector_count
    vector_count = 0  # 初始化计数
    with open(output_rgb_file, "w") as rgb_out, open(output_id_file, "w") as id_out, open(output_time_file,
                                                                                          "w") as time_out:
        for filename in os.listdir(data_dir):
            if filename.endswith(".tfrecord"):
                filepath = os.path.join(data_dir, filename)
                print(f"Processing {filepath} ...")
                process_tfrecord_file(filepath, rgb_out, id_out, time_out)

    # 处理完成后打印总数
    print(f"\n处理完成，共提取 {vector_count} 个向量。")


# 运行数据处理
process_all_tfrecords()
