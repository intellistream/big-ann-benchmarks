#!/bin/bash

# 定义算法和数据集
ALGORITHMS=("cufe")
DATASETS=("msong-filter" "cirr-filter" "glove-filter" "sift-filter" "adverse" "YouTube-rgb" "YouTube-audio" "random-filter-s" "sift-uniform" "sift-poisson" )

# 结果存储目录
RESULT_DIR="data_result"

# 确保结果存储目录存在
mkdir -p "$RESULT_DIR"

# 遍历每个算法
for algo in "${ALGORITHMS[@]}"; do
    echo "================ Running Algorithm: $algo ================"

    # 1. 下载 Docker 镜像
    if ! docker images --format '{{.Repository}}' | grep -q "$algo"; then
        echo "Docker image for $algo not found. Downloading..."
        python install.py --neurips23track filter --algorithm "$algo"
    else
        echo "Docker image for $algo already exists. Skipping download."
    fi

    # 2. 遍历每个数据集
    for dataset in "${DATASETS[@]}"; do
        echo "---- Running Dataset: $dataset ----"

        # 2.1 创建数据集
        python create_dataset.py --dataset "$dataset"

        # 2.2 运行实验
        python run.py --neurips23track filter --algorithm "$algo" --dataset "$dataset"

        # 3. 保存 `annb.log`
        LOG_DIR="$RESULT_DIR/$algo/$dataset"
        mkdir -p "$LOG_DIR"
        cp annb.log "$LOG_DIR/annb.log"
    done

    # 4. 导出实验结果
    python data_export.py --out "$RESULT_DIR/$algo/res.csv" --track filter

    # 5. 删除 Docker 镜像（假设基础镜像不用删）
    if [ "$algo" != "cufe" ]; then
        docker rmi "$(docker images --format '{{.Repository}}' | grep "$algo")"
    fi

    echo "====== Finished Algorithm: $algo ======"
done

echo "All experiments completed!"
