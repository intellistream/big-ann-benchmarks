#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG_PATH=""
DATASETS_ARG=""
ALGO_ARG=""

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [--config CONFIG_JSON] [--datasets ds1,ds2] [--algorithm algo] [--help]

Without arguments the script runs run_param_tuning.sh using its built-in defaults.
You can optionally point to a specific JSON config (sets PARAM_GRID_FILE), limit
datasets, or force a single algorithm.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "error: --config requires a value" >&2
        exit 1
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --datasets)
      if [[ $# -lt 2 ]]; then
        echo "error: --datasets requires a value" >&2
        exit 1
      fi
      DATASETS_ARG="$2"
      shift 2
      ;;
    --algorithm)
      if [[ $# -lt 2 ]]; then
        echo "error: --algorithm requires a value" >&2
        exit 1
      fi
      ALGO_ARG="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${CONFIG_PATH}" ]]; then
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "error: config file not found: ${CONFIG_PATH}" >&2
    exit 1
  fi
  export PARAM_GRID_FILE
  PARAM_GRID_FILE=$(python3 - <<'PY'
import os, sys
path = os.path.abspath(sys.argv[1])
print(path)
PY
"${CONFIG_PATH}")
fi

if [[ -n "${DATASETS_ARG}" ]]; then
  export DATASETS="${DATASETS_ARG}"
fi

if [[ -n "${ALGO_ARG}" ]]; then
  export ALGORITHM="${ALGO_ARG}"
fi

cd "${ROOT_DIR}"
bash scripts/run_param_tuning.sh
