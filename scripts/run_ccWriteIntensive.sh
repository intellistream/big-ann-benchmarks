#!/bin/bash

# Define the algorithms and datasets
ALGORITHMS=("nswlib_HNSW" "candy_flann" "candy_lshapg" "candy_mnru" "candy_sptag" "cufe" "diskann" "faiss_fast_scan" "faiss_HNSW" "faiss_IVFPQ" "faiss_lsh" "faiss_NSW" "faiss_onlinepq" "faiss_pq" "puck" "pyanns")
DATASETS=("sift" "random-xs" "msturing-10M-clustered" "msturing-30M-clustered")
RUNBOOKS=("batch100_w50r50.yaml" "batch100_w80r20.yaml" "batch100_w90r10.yaml"
          "batch200_w50r50.yaml" "batch200_w80r20.yaml" "batch200_w90r10.yaml"
          "batch500_w50r50.yaml" "batch500_w80r20.yaml" "batch500_w90r10.yaml"
          "batch1000_w50r50.yaml" "batch1000_w80r20.yaml" "batch1000_w90r10.yaml"
          )

# Iterate through each combination of algorithm and dataset
for RUN in "${RUNBOOKS[@]}"; do
  for ALGO in "${ALGORITHMS[@]}"; do
    for DS in "${DATASETS[@]}"; do
      echo "Running with algorithm: $ALGO and dataset: $DS Using $DS"
      python3 run.py --neurips23track concurrent --algorithm "$ALGO" --nodocker --rebuild --runbook_path neurips23/runbooks/concurrent/writeIntensive/"$RUN" --dataset "$DS"
    done
  done
done

echo "All experiments completed."