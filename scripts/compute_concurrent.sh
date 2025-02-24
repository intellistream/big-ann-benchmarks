#!/bin/bash
DATASETS=("reddit" "sift" "glove" "msong" "msturing-10M")
RUNBOOKS=("batch100_w50r50.yaml" "batch200_w50r50.yaml" "batch500_w50r50.yaml")

# Iterate through each combination of algorithm and dataset
for RUN in "${RUNBOOKS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    python3 benchmark/concurrent/compute_gt.py --runbook neurips23/runbooks/concurrent/writeIntensive/"$RUN" --dataset "$DS" 
  done
done