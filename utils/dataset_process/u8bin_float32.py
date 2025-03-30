import struct
import numpy as np

def convert_u8bin_to_float32(input_filename, output_filename):
    """转换 u8bin 文件为 float32bin 文件"""
    with open(input_filename, 'rb') as f:
        num_vectors = struct.unpack('I', f.read(4))[0]
        dim = struct.unpack('I', f.read(4))[0]

        vectors = np.fromfile(f, dtype=np.uint8).astype(np.float32)

    with open(output_filename, 'wb') as f:
        f.write(struct.pack('I', num_vectors))
        f.write(struct.pack('I', dim))
        vectors.tofile(f)

    print(f"已转换 {input_filename} -> {output_filename}，数据类型变为 float32")


convert_u8bin_to_float32('../../data/YouTube-rgb/filter/queries_1000_1024', '../../data/YouTube-rgb/filter/queries_1000_1024_1')
