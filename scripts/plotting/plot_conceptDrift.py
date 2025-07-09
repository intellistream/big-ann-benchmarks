import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import matplotlib

# ===== 全局配置 =====
OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 22
LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
MARKER_SIZE = 12.5

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.fonttype'] = 42

# ===== 算法和颜色 =====
algorithms = [
    "candy_lshapg", "candy_mnru", "candy_sptag", "cufe", "diskann", "faiss_fast_scan",
    "faiss_HNSW", "faiss_IVFPQ", "faiss_lsh", "faiss_NSW", "faiss_onlinepq",
    "faiss_pq", "puck", "pyanns"
]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'pink', 'brown', 'purple', 'gray',
          'lime', 'teal', 'indigo', 'violet', 'gold', 'coral']

def plot_concept_drift_recall():
    # 文件路径写死
    input_file = "F8_conceptDrift.csv"
    output_path = "scripts/plotting/plots/concept_drift/recall_vs_batch.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 读取数据
    df = pd.read_csv(input_file)

    # x轴：0 ~ 94 映射为 0% ~ 94%
    x = np.linspace(0, 1.0, 95)
    x_labels = [0.1 * i for i in range(1, 10)]  # 10% 到 90%

    # 更宽的图，防止横坐标拥挤
    plt.figure(figsize=(12, 5.5))

    # 绘制每条线
    for idx, algorithm in enumerate(df['algorithm']):
        recall_values = df.loc[idx, [f"batchRecall_{i}" for i in range(95)]].to_numpy(dtype=float)
        plt.plot(x, recall_values,
                 color=colors[algorithms.index(algorithm)],
                 linewidth=2)

    # 设置坐标轴
    plt.xlabel("Percentage Ingested", fontsize=TICK_FONT_SIZE)
    plt.ylabel("Recall@10", fontsize=TICK_FONT_SIZE)
    plt.xticks(x_labels, [f"{int(p*100)}%" for p in x_labels])
    plt.xlim(0.0, 1.0)  # 确保线条贴边
    plt.grid(True, which='both', axis='y')

    yticks = np.linspace(0, 1, 6)  # 0.0, 0.2, ..., 1.0
    ytick_labels = ["" if y == 0 else f"{y:.1f}" for y in yticks]
    plt.yticks(yticks, ytick_labels)

    # 构造图例（正方形块，顶部两排）
    legend_handles = []
    for alg in df['algorithm']:
        if '_' in alg:
            label = alg.split('_')[1].upper()
        else:
            label = alg.upper()
        if label == "FAST":
            label = "SCANN"
        legend_handles.append(Line2D([0], [0], marker='s', color='w',
                                     markerfacecolor=colors[algorithms.index(alg)],
                                     markeredgecolor='black',
                                     markersize=MARKER_SIZE, label=label))

    plt.legend(handles=legend_handles,
               prop={'size': 13},
               loc='upper center',
               bbox_to_anchor=(0.5, 1.19),
               ncol=7,
               frameon=False,
               edgecolor='black')

    # 布局调整：预留顶部空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # 保存图像
    plt.savefig(output_path, format='pdf')
    plt.close()

# === 执行绘图 ===
plot_concept_drift_recall()
