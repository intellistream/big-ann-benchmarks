#!/bin/bash
DATASETS=("multiModalProp-0" "multiModalProp-0.2" "multiModalProp-0.4" "multiModalProp-0.6" "multiModalProp-0.8" "multiModalProp-1.0")
# Iterate through each combination of algorithm and dataset
for DS in "${DATASETS[@]}"; do
  python3 benchmark/congestion/compute_gt.py --runbook neurips23/runbooks/congestion/multiModalProp/multiModalProp_experiment.yaml --dataset "$DS" --gt_cmdline_tool ~/DiskANN/build/apps/utils/compute_groundtruth
done