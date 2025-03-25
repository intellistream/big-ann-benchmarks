import tensorflow as tf

# 读取 TFRecord 文件
tfrecord_file = "/mnt/c/Users/Administrator/Desktop/test/yt8m/data/video/train0580.tfrecord"

# 统计视频数量
count = 0
for record in tf.data.TFRecordDataset([tfrecord_file]):
    count += 1

print(f"{tfrecord_file} 中包含 {count} 个视频")