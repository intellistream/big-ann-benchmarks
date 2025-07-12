#!/bin/bash

# Define the algorithms and datasets
ALGORITHMS=("ipdiskann" "gti")
DATASETS=("coco")
# Iterate through each combination of algorithm and dataset
for ALGO in "${ALGORITHMS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    echo "Running with algorithm: $ALGO and dataset: $DS"
    python3 run.py --neurips23track congestion --algorithm "$ALGO" --nodocker --rebuild --runbook_path neurips23/runbooks/congestion/multiModal/multiModal_experiment.yaml --dataset "$DS"
  done
done

echo "All experiments completed."