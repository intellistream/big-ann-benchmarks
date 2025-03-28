import tensorflow as tf

def read_tfrecord(file_path, num_records=2):
    """ 读取 TFRecord 文件，并打印前 num_records 个样本的信息 """
    raw_dataset = tf.data.TFRecordDataset(file_path)

    # 定义特征解析的结构
    feature_description = {
        "id": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.VarLenFeature(tf.int64),
        "mean_rgb": tf.io.FixedLenFeature([1024], tf.float32),
        "mean_audio": tf.io.FixedLenFeature([128], tf.float32),
    }

    def parse_example(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    # 遍历前 num_records 个样本
    for i, raw_record in enumerate(raw_dataset.take(num_records)):
        example = parse_example(raw_record)

        video_id = example["id"].numpy().decode("utf-8")
        labels = example["labels"].values.numpy()
        mean_rgb = example["mean_rgb"].numpy()
        mean_audio = example["mean_audio"].numpy()

        print(f"\n======  视频 {i+1}  ======")
        print("视频 ID:", video_id)
        print("标签列表:", labels)
        print("mean_rgb 前5维:", mean_rgb[:5], "...")
        print("mean_audio 前5维:", mean_audio[:5], "...")

file_path = "/mnt/c/Users/Administrator/Desktop/test/yt8m/data/video/train0208.tfrecord"
read_tfrecord(file_path)
