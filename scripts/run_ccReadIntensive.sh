#!/bin/bash

# Define the algorithms and datasets
ALGORITHMS=("linear" "candy_flann" "candy_lshapg" "candy_mnru" "candy_sptag" "cufe" "diskann" "faiss_fast_scan" "faiss_HNSW" "faiss_IVFPQ" "faiss_lsh" "faiss_NSW" "faiss_onlinepq" "faiss_pq" "puck" "pyanns")
DATASETS=("sift" "random-xs" "msturing-10M-clustered" "msturing-30M-clustered")
RUNBOOKS=("batch100_w05r95.yaml" "batch100_w10r90.yaml" "batch100_w20r80.yaml"
          "batch200_w05r95.yaml" "batch200_w10r90.yaml" "batch200_w20r80.yaml"
          "batch500_w05r95.yaml" "batch500_w10r90.yaml" "batch500_w20r80.yaml"
          "batch1000_w05r95.yaml" "batch1000_w10r90.yaml" "batch1000_w20r80.yaml"
          )

# Iterate through each combination of algorithm and dataset
for RUN in "${RUNBOOKS[@]}"; do
  for ALGO in "${ALGORITHMS[@]}"; do
    for DS in "${DATASETS[@]}"; do
      echo "Running with algorithm: $ALGO and dataset: $DS Using $DS"
      python3 run.py --neurips23track concurrent --algorithm "$ALGO" --nodocker --rebuild --runbook_path neurips23/runbooks/concurrent/readIntensive/"$RUN" --dataset "$DS"
    done
  done
done

echo "All experiments completed."