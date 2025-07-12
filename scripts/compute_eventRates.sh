#!/bin/bash
DATASETS=("sift")
RUNBOOKS=("2500.yaml" "10000.yaml" "100000.yaml" "200000.yaml" "500000.yaml")

# Iterate through each combination of algorithm and dataset
for RUN in "${RUNBOOKS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    python3 benchmark/congestion/compute_gt.py --runbook neurips23/runbooks/congestion/eventRates/event"$RUN" --dataset "$DS" --gt_cmdline_tool ~/DiskANN/build/apps/utils/compute_groundtruth
  done
done