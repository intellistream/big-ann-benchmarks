#!/bin/bash

# Define the algorithms and datasets
ALGORITHMS=("candy_lshapg" "candy_mnru" "candy_sptag" "cufe" "diskann" "faiss_fast_scan" "faiss_HNSW" "faiss_IVFPQ" "faiss_lsh" "faiss_NSW" "faiss_onlinepq" "faiss_pq" "puck" "pyanns")
DATASETS=("multiModalProp-0" "multiModalProp-0.2" "multiModalProp-0.4" "multiModalProp-0.6" "multiModalProp-0.8")
# Iterate through each combination of algorithm and dataset
for ALGO in "${ALGORITHMS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    echo "Running with algorithm: $ALGO and dataset: $DS"
    python3 run.py --neurips23track congestion --algorithm "$ALGO" --nodocker --rebuild --runbook_path neurips23/runbooks/congestion/multiModalProp/multiModalProp_experiment.yaml --dataset "$DS"
  done
done

echo "All experiments completed."