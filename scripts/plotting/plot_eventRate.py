import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import LogLocator
from matplotlib.font_manager import FontProperties
import matplotlib

from matplotlib.lines import Line2D


OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 22
LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE)
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)

# you may want to change the color map for different figures
LABEL_WEIGHT = 'bold'
LINE_WIDTH = 3.0
MARKER_SIZE = 12.5
MARKER_FREQUENCY = 1000

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.fonttype'] = 42

markers = ['o', 's', 'D', 'v', '^', 'P', '*', 'X', 'h', '>', '<', 'p', 'H', 'd', '1', '2', '3', '4']
colors = ['crimson', 'gold','b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'pink', 'brown', 'purple', 'gray', 'lime', 'teal', 'indigo',
          'violet', 'gold', 'coral']
algorithms = [
    "gti", "ipdiskann",
    "candy_lshapg", "candy_mnru", "candy_sptag", "cufe", "diskann", "faiss_fast_scan",
    "faiss_HNSW", "faiss_IVFPQ", "faiss_lsh", "faiss_NSW", "faiss_onlinepq",
    "faiss_pq", "puck", "pyanns"
]

# Define the categories
categories = {
    'Tree': ['candy_sptag'],
    'Hash': ['candy_lshapg', 'faiss_lsh'],
    'Clustering': ['faiss_pq', 'faiss_onlinepq', 'faiss_IVFPQ', 'puck', 'faiss_fast_scan'],
    'Graph-based': ['faiss_NSW', 'faiss_HNSW', 'candy_mnru', 'diskann', 'cufe', 'pyanns',"gti", "ipdiskann",]
}

def plot_recall_vs_event_rate(file_path, save_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Filter rows with dataset containing "event" between 1000-50000
    data = data[data['dataset'].str.contains(r'event(?:2500|10000|100000|200000|500000)\.yaml')]

    # Filter rows for specific algorithms
    data = data[data['algorithm'].isin(algorithms)]
    data["eventRate"] = data["dataset"].str.extract(r"event(\d+)\.yaml").astype(int)

    plt.figure(figsize=(7, 6))
    for (algorithm, group), (marker, color) in zip(data.groupby("algorithm"), zip(markers, colors)):
        group = group.sort_values(by="eventRate")
        x = group["eventRate"].to_numpy()
        y = group["continuousRecall_0"].to_numpy()
        plt.plot(x, y, marker='o', markersize=8, color=colors[algorithms.index(algorithm)], label=algorithm)

    plt.xlabel("Event Rate",fontsize=TICK_FONT_SIZE)
    plt.ylabel("Recall@10",fontsize=TICK_FONT_SIZE)
    plt.xscale('log')
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=5))

    # Remove gridlines for x-axis (log scale brings them, but we want to remove them)
    plt.gca().get_xaxis().grid(False)
    plt.grid(True, which='both', axis='y')
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='lower', steps=[1, 2, 3, 4, 5]))

    # Add the custom unified legend for the first plot


    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, format="pdf")
    plt.close()

def plot_throughput_vs_event_rate(file_path, save_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Filter rows with dataset containing "event" between 1000-500000
    data = data[data['dataset'].str.contains(r'event(?:2500|10000|100000|200000|500000)\.yaml')]
    data["eventRate"] = data["dataset"].str.extract(r"event(\d+)\.yaml").astype(int)

    # Filter for specific algorithms
    data = data[data['algorithm'].isin(algorithms)]

    # Create figure
    plt.figure(figsize=(9, 6))

    # Plot each algorithm's throughput
    for (algorithm, group), (marker, color) in zip(data.groupby("algorithm"), zip(markers, colors)):
        group = group.sort_values(by="eventRate")
        x = group["eventRate"].to_numpy()
        y = group["continuousThroughput_0"].to_numpy()
        plt.plot(x, y, marker='o', markersize=8,
                 color=colors[algorithms.index(algorithm)],
                 label=algorithm)

    plt.xlabel("Event Rate", fontsize=TICK_FONT_SIZE)
    plt.ylabel("QPS", fontsize=TICK_FONT_SIZE)
    plt.xscale('log')
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=5))
    plt.gca().get_xaxis().grid(False)
    plt.grid(True, which='both', axis='y')
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='lower', steps=[1, 2, 3, 4, 5]))

    # ==== Add categorized algorithm legend ====
    legend_handles = []

    # Tree
    legend_handles.append(Line2D([0], [0], color='none', label="Tree"))
    for alg in categories['Tree']:
        label = alg.split('_')[1].upper() if '_' in alg else alg.upper()
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=colors[algorithms.index(alg)],
                                     markersize=MARKER_SIZE, label=label))

    # Graph
    legend_handles.append(Line2D([0], [0], color='none', label="Graph"))
    for alg in categories['Graph-based']:
        label = alg.split('_')[1].upper() if '_' in alg else alg.upper()
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=colors[algorithms.index(alg)],
                                     markersize=MARKER_SIZE, label=label))

    # LSH
    legend_handles.append(Line2D([0], [0], color='none', label="LSH"))
    for alg in categories['Hash']:
        label = alg.split('_')[1].upper() if '_' in alg else alg.upper()
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=colors[algorithms.index(alg)],
                                     markersize=MARKER_SIZE, label=label))

    # Clustering
    legend_handles.append(Line2D([0], [0], color='none', label="Clustering"))
    for alg in categories['Clustering']:
        label = alg.split('_')[1].upper() if '_' in alg else alg.upper()
        if label == "FAST":
            label = "SCANN"
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=colors[algorithms.index(alg)],
                                     markersize=MARKER_SIZE, label=label))

    # Plot legend on right side
    custom_legend = plt.legend(handles=legend_handles, prop={'size': 13},
                               loc='center left', bbox_to_anchor=(1.05, 0.5),
                               frameon=True, edgecolor='black', borderpad=1, ncol=1)

    # Bold+Italic for category titles
    for text, entry in zip(custom_legend.get_texts(), legend_handles):
        if isinstance(entry, Line2D) and entry.get_color() == 'none':
            text.set_fontweight('bold')
            text.set_style('italic')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Reserve space for legend

    # Save the plot
    plt.savefig(save_path, format="pdf")
    plt.close()


def plot_event_rate_congestion(file_path, save_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Filter rows with dataset containing "event" between 1000-50000
    data = data[data['dataset'].str.contains(r'event(?:2500|10000|100000|200000|500000)\.yaml')]
    data["eventRate"] = data["dataset"].str.extract(r"event(\d+)\.yaml").astype(int)

    # Filter rows for specific algorithms
    data = data[data['algorithm'].isin(algorithms)]
    plt.figure(figsize=(9, 6))

    # Initialize the plot

    # Sort unique event rates for consistent marker assignment
    unique_values = sorted(data["eventRate"].unique())
    markers_map = {value: markers[i % len(markers)] for i, value in enumerate(unique_values)}

    # Plotting

    plot_lines = []
    for (algorithm, group), (marker, color) in zip(data.groupby("algorithm"), zip(markers, colors)):
        group = group.sort_values(by="eventRate")
        plot_lines.append([])
        for value, sub_group in group.groupby("eventRate"):
            l1 = plt.scatter(
                sub_group["continuousRecall_0"],
                sub_group["continuousThroughput_0"],
                marker=markers_map[value],
                color=colors[algorithms.index(algorithm)],
                s=150,
                label=algorithm if value == group["eventRate"].iloc[0] else "",
                edgecolors="black", linewidth=1
            )
            plot_lines[-1].append(l1)


    # Set labels
    plt.xlabel("Recall@10", fontsize=TICK_FONT_SIZE)
    plt.ylabel("QPS", fontsize=TICK_FONT_SIZE)
    plt.grid(True)

    # Custom algorithm legend
    legend_handles = []
    # Tree category
    legend_handles.append(Line2D([0], [0], color='none', label="Tree"))
    for alg in categories['Tree']:
        label = alg.split('_')[1].upper() if '_' in alg else alg.upper()
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[algorithms.index(alg)],
                                     markersize=MARKER_SIZE, label=label))

    # Graph-based category
    legend_handles.append(Line2D([0], [0], color='none', label="Graph"))
    for alg in categories['Graph-based']:
        label = alg.split('_')[1].upper() if '_' in alg else alg.upper()
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[algorithms.index(alg)],
                                     markersize=MARKER_SIZE, label=label))

    # Hash category
    legend_handles.append(Line2D([0], [0], color='none', label="LSH"))
    for alg in categories['Hash']:
        label = alg.split('_')[1].upper() if '_' in alg else alg.upper()
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[algorithms.index(alg)],
                                     markersize=MARKER_SIZE, label=label))

    # Clustering category
    legend_handles.append(Line2D([0], [0], color='none', label="Clustering"))
    for alg in categories['Clustering']:
        label = alg.split('_')[1].upper() if '_' in alg else alg.upper()
        if label == "FAST":
            label = "SCANN"
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[algorithms.index(alg)],
                                     markersize=MARKER_SIZE, label=label))



    value_legend_handles = []
    for value, marker in markers_map.items():
        value_legend_handles.append(Line2D([0], [0], marker=marker, color='w', markerfacecolor='black',
                                                      markersize=12.5, label=f'{value}'))

    value_legend = plt.legend(handles=value_legend_handles, prop={'size': 13},loc='upper right',bbox_to_anchor=(1.03,1.015),fontsize=15)
    plt.gca().add_artist(value_legend)

    custom_legend = plt.legend(handles=legend_handles, prop={'size': 13}, loc='center left', bbox_to_anchor=(1.05, 0.5),
                               frameon=True, edgecolor='black', borderpad=1, ncol=1)

    for text, entry in zip(custom_legend.get_texts(), legend_handles):
        if isinstance(entry, Line2D) and entry.get_color() == 'none':
            text.set_fontweight('bold')
            text.set_style('italic')




    # Set xticks and xlim as requested
    plt.xticks([0.5, 0.65, 0.8, 0.95])
    plt.xlim(0.5, 1.0)

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close()

# File paths
input_file = "F4_Event_Rate .csv"
output_dir = "scripts/plotting/plots/event/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save each plot separately
plot_throughput_vs_event_rate(input_file, os.path.join(output_dir, "QPS.pdf"))
plot_recall_vs_event_rate(input_file, os.path.join(output_dir, "recall.pdf"))
plot_event_rate_congestion(input_file, os.path.join(output_dir,"vs.pdf"))




