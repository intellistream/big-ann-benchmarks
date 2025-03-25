import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def analyze_label_distribution(file_path):
    # 读取数据
    queries = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            parts = line.split(',')
            queries.append(parts)
            labels.extend([int(label) for label in parts])

    if not queries:
        print("错误：未找到任何有效查询！")
        return

    # 基础统计
    label_counter = Counter(labels)
    total_labels = len(labels)
    unique_labels = len(label_counter)

    print(f"总标签数: {total_labels}")
    print(f"唯一标签数: {unique_labels}")
    print(f"平均每个查询标签数: {total_labels / len(queries):.2f}")

    # 标签数量分布检查
    query_lengths = [len(q) for q in queries]
    length_counter = Counter(query_lengths)
    print("\n查询标签数量分布:")
    for k in sorted(length_counter.keys()):
        print(f"  {k}个标签: {length_counter[k]}次 ({length_counter[k] / len(queries):.1%})")

    # 计算实际分布
    sorted_labels = sorted(label_counter.items(), key=lambda x: x[1], reverse=True)
    ranks = np.arange(1, len(sorted_labels) + 1)
    frequencies = np.array([x[1] for x in sorted_labels]) / total_labels

    # 计算理论Zipf分布（s=1）
    zipf = 1 / (ranks * np.sum(1 / ranks))  # 归一化Zipf定律

    # 可视化比较
    plt.figure(figsize=(10, 6))
    plt.bar(ranks - 0.2, frequencies, width=0.4, label='实际分布')
    plt.bar(ranks + 0.2, zipf, width=0.4, label='理论Zipf')
    plt.xlabel('标签排名')
    plt.ylabel('出现频率')
    plt.title('实际分布 vs 理论Zipf分布')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 统计检验
    print("\nTop 10标签分布验证:")
    for i, (label, count) in enumerate(sorted_labels[:10], 1):
        actual = count / total_labels
        expected = 1 / (i * np.sum(1 / ranks))
        print(f"第{i}名标签{label}: 实际{actual:.3%} vs 理论{expected:.3%}")


if __name__ == "__main__":
    file_path = "../../data/SIFT/base_label.txt"  # 修改为实际路径
    analyze_label_distribution(file_path)