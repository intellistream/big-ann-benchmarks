#!/bin/bash

STATIC_ALGOS=("faiss_HNSW")
# STREAMING_ALGOS=("FreshDiskANN" "cufe" "pyanns")
DATASETS=("sift")
RUNBOOKS=("fairness_static_10.yaml")
# RUNBOOKS=("fairness_static_10.yaml" "fairness_static_20.yaml" "fairness_static_50.yaml")

for RUN in "${RUNBOOKS[@]}"; do
  RUNBOOK_PATH="neurips23/runbooks/congestion/fairness/${RUN}"
  for ALGO in "${STATIC_ALGOS[@]}"; do
    for DS in "${DATASETS[@]}"; do
      echo "Running fairness (static) algorithm: $ALGO, dataset: $DS, runbook: $RUN"
      python3 run.py --neurips23track congestion --algorithm "$ALGO" --nodocker --rebuild --runbook_path "$RUNBOOK_PATH" --dataset "$DS"
    done
  done
  # for ALGO in "${STREAMING_ALGOS[@]}"; do
  #   for DS in "${DATASETS[@]}"; do
  #     echo "Running fairness (streaming) algorithm: $ALGO, dataset: $DS, runbook: $RUN"
  #     python3 run.py --neurips23track congestion --algorithm "$ALGO" --nodocker --rebuild --runbook_path "$RUNBOOK_PATH" --dataset "$DS"
  #   done
  # done
done

