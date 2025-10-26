#!/bin/bash

DATASETS=("sift")
# RUNBOOKS=("fairness_static_10.yaml" "fairness_static_20.yaml" "fairness_static_50.yaml")
RUNBOOKS=("fairness_static_10.yaml")
RUNBOOK_ROOT="neurips23/runbooks/congestion/fairness"

for RUNBOOK in "${RUNBOOKS[@]}"; do
  RUNBOOK_PATH="${RUNBOOK_ROOT}/${RUNBOOK}"
  for DATASET in "${DATASETS[@]}"; do
    python3 benchmark/congestion/compute_gt.py \
      --dataset "${DATASET}" \
      --runbook_file "${RUNBOOK_PATH}" \
      --gt_cmdline_tool ~/DiskANN/build/apps/utils/compute_groundtruth \
      "$@"
  done
done
