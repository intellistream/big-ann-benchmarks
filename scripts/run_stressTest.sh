#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/home/junyao/code/SAGE-DB-Bench:${PYTHONPATH:-}

# Resolve script and project roots
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Define the algorithms and datasets
# ALGORITHMS=("faiss_HNSW" "faiss_IVFPQ" "faiss_lsh" "faiss_NSW" "faiss_onlinepq" "faiss_pq")
ALGORITHMS=("faiss_HNSW")
DATASETS=("sift")
RUNBOOKS=("0.1.yaml")

# Output CSV for stress summaries
OUT_CSV="${PROJECT_ROOT}/stress_summary.csv"
if [ ! -f "$OUT_CSV" ]; then
  echo "algorithm,dataset,runbook,RStar,BStar,DeltaHatUs" > "$OUT_CSV"
fi

for RUN in "${RUNBOOKS[@]}"; do
  for ALGO in "${ALGORITHMS[@]}"; do
    for DS in "${DATASETS[@]}"; do
      echo "Running stressTest with algorithm: $ALGO and dataset: $DS"
      LOG_FILE="${PROJECT_ROOT}/logs_${ALGO}_${DS}_$(basename "$RUN" .yaml).log"
      (cd "${PROJECT_ROOT}" && python3.10 -u ./run.py --neurips23track congestion --algorithm "$ALGO" --nodocker --rebuild --runbook_path "neurips23/runbooks/congestion/stressTest/stressTest${RUN}" --dataset "$DS") 2>&1 | tee "$LOG_FILE"

      # Parse STRESS_SUMMARY line (robust sed extraction)
      LINE=$(grep -F "STRESS_SUMMARY" "$LOG_FILE" | tail -n1 || true)
      if [ -n "$LINE" ]; then
        RSTAR=$(echo "$LINE" | sed -n 's/.*RStar=\([^,]*\).*/\1/p')
        BSTAR=$(echo "$LINE" | sed -n 's/.*BStar=\([^,]*\).*/\1/p')
        DELTA=$(echo "$LINE" | sed -n 's/.*DeltaHatUs=\([^,]*\).*/\1/p')
        echo "${ALGO},${DS},${RUN},${RSTAR:-},${BSTAR:-},${DELTA:-}" >> "$OUT_CSV"
      else
        echo "${ALGO},${DS},${RUN},,," >> "$OUT_CSV"
      fi
    done
  done
done

echo "StressTest experiments completed."
