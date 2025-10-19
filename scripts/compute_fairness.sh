#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <dataset> <runbook_path> [--private_query]"
  exit 1
fi

DATASET="$1"
RUNBOOK="$2"
PRIVATE_QUERY="${3:-}"

if [ -z "${GT_CMDLINE_TOOL:-}" ]; then
  echo "Please export GT_CMDLINE_TOOL to the ground-truth executable path (e.g., benchmark/gt/dist/distance_gt)."
  exit 1
fi

python3 -m benchmark.congestion.compute_gt \
  --dataset "${DATASET}" \
  --runbook_file "${RUNBOOK}" \
  --gt_cmdline_tool "${GT_CMDLINE_TOOL}" \
  ${PRIVATE_QUERY:+--private_query}

echo "Ground-truth generation finished for dataset=${DATASET}, runbook=${RUNBOOK}."
