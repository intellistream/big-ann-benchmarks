#!/bin/bash

# Define the algorithms and datasets
ALGORITHMS=("ipdiskann" "gti")
DATASETS=("sift")
RUNBOOKS=("0.1.yaml" "0.2.yaml" "0.3.yaml" "0.4.yaml" "0.5.yaml")

for RUN in "${RUNBOOKS[@]}"; do
  for ALGO in "${ALGORITHMS[@]}"; do
    for DS in "${DATASETS[@]}"; do
      echo "Running stressTest with algorithm: $ALGO and dataset: $DS"
      python3 run.py --neurips23track congestion --algorithm "$ALGO" --nodocker --rebuild --runbook_path "neurips23/runbooks/congestion/stressTest/stressTest${RUN}" --dataset "$DS"
    done
  done
done

echo "StressTest experiments completed."
