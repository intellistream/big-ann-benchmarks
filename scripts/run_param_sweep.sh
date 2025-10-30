#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG_PATH=""
DATASETS_ARG=""

print_usage() {
  cat <<EOF
Usage: $(basename "$0") --config CONFIG_JSON [--datasets ds1,ds2]

Wrapper that sets PARAM_GRID_FILE/DATASETS and invokes run_param_tuning.sh.
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

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "error: --config is required" >&2
  print_usage >&2
  exit 1
fi

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

if [[ -n "${DATASETS_ARG}" ]]; then
  export DATASETS="${DATASETS_ARG}"
fi

cd "${ROOT_DIR}"
bash scripts/run_param_tuning.sh
