#!/bin/bash
DATASETS=("sift" "random-xs" "msturing-10M-clustered" "msturing-30M-clustered")
RUNBOOKS=("batch100_w50r50.yaml" "batch100_w80r20.yaml" "batch100_w90r10.yaml"
          "batch200_w50r50.yaml" "batch200_w80r20.yaml" "batch200_w90r10.yaml"
          "batch500_w50r50.yaml" "batch500_w80r20.yaml" "batch500_w90r10.yaml"
          "batch1000_w50r50.yaml" "batch1000_w80r20.yaml" "batch1000_w90r10.yaml"
          )

# Iterate through each combination of algorithm and dataset
for RUN in "${RUNBOOKS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    python3 benchmark/concurrent/compute_gt.py --runbook neurips23/runbooks/concurrent/readIntensive/"$RUN" --dataset "$DS"
  done
done