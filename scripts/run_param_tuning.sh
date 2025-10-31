#!/bin/bash
set -euo pipefail

ALGORITHM_LIST_DEFAULT=(
  "faiss_HNSW"
  "faiss_NSW"
  "candy_mnru"
  "FreshDiskANN"
  "cufe"
  "pyanns"
  "gti"
  "ipdiskann"
  "candy_lsh"
  "candy_lshapg"
  "candy_sptag"
  "faiss_pq"
  "faiss_onlinepq"
  "faiss_IVFPQ"
  "puck"
  "scann"
)
DATASET_LIST_DEFAULT=("sift")

declare -A PARAM_GRID_MAP
PARAM_GRID_MAP["faiss_HNSW"]="config/param_tuning/hnsw.json"
PARAM_GRID_MAP["faiss_NSW"]="config/param_tuning/nsw.json"
PARAM_GRID_MAP["candy_mnru"]="config/param_tuning/mnru.json"
PARAM_GRID_MAP["FreshDiskANN"]="config/param_tuning/freshdiskann.json"
PARAM_GRID_MAP["cufe"]="config/param_tuning/cufe.json"
PARAM_GRID_MAP["pyanns"]="config/param_tuning/pyanns.json"
PARAM_GRID_MAP["gti"]="config/param_tuning/gti.json"
PARAM_GRID_MAP["ipdiskann"]="config/param_tuning/ipdiskann.json"
PARAM_GRID_MAP["candy_lsh"]="config/param_tuning/lsh.json"
PARAM_GRID_MAP["candy_lshapg"]="config/param_tuning/lshapg.json"
PARAM_GRID_MAP["candy_sptag"]="config/param_tuning/sptag.json"
PARAM_GRID_MAP["faiss_pq"]="config/param_tuning/faiss_pq.json"
PARAM_GRID_MAP["faiss_onlinepq"]="config/param_tuning/faiss_onlinepq.json"
PARAM_GRID_MAP["faiss_IVFPQ"]="config/param_tuning/faiss_ivfpq.json"
PARAM_GRID_MAP["puck"]="config/param_tuning/puck.json"
PARAM_GRID_MAP["scann"]="config/param_tuning/scann.json"

DEFAULT_LOG_DIR="logs_param_tuning"
DEFAULT_SUMMARY_FILE="param_tuning_raw.csv"
DEFAULT_SUMMARY_TABLE="param_tuning_summary.txt"
DEFAULT_RESULTS_ROOT="results/neurips23/congestion"
DEFAULT_RUNBOOK_TEMPLATE="neurips23/runbooks/congestion/param_tuning/{algorithm}_{dataset}.yaml"
DEFAULT_RECALL_FLOOR_FACTOR="0.9"
DEFAULT_RECALL_FLOOR_MIN="0.75"

declare -a ALGORITHM_LIST
ALGORITHM_LIST=("${ALGORITHM_LIST_DEFAULT[@]}")
if [[ -n "${ALGORITHMS:-${ALGORITHM_LIST_OVERRIDE:-}}" ]]; then
  IFS=',' read -r -a _alg_override <<< "${ALGORITHMS:-${ALGORITHM_LIST_OVERRIDE:-}}"
  ALGORITHM_LIST=()
  for entry in "${_alg_override[@]}"; do
    algo=$(echo "${entry}" | xargs)
    [[ -n "${algo}" ]] && ALGORITHM_LIST+=("${algo}")
  done
  if [[ ${#ALGORITHM_LIST[@]} -eq 0 ]]; then
    ALGORITHM_LIST=("${ALGORITHM_LIST_DEFAULT[@]}")
  fi
fi

GROUNDTRUTH_ENABLED=${GROUNDTRUTH_ENABLED:-true}
GROUNDTRUTH_TOOL=${GROUNDTRUTH_TOOL:-~/DiskANN/build/apps/utils/compute_groundtruth}
GROUNDTRUTH_FORCE=${GROUNDTRUTH_FORCE:-false}
GROUNDTRUTH_EXTRA_ARGS_JSON=${GROUNDTRUTH_EXTRA_ARGS_JSON:-"[]"}

declare -a DATASET_LIST
DATASET_LIST=("${DATASET_LIST_DEFAULT[@]}")

if [[ -n "${TARGET_DATASETS:-}" ]]; then
  IFS=',' read -r -a _requested_override <<< "${TARGET_DATASETS}"
  DATASET_LIST=()
  for entry in "${_requested_override[@]}"; do
    ds=$(echo "${entry}" | xargs)
    [[ -z "${ds}" ]] && continue
    DATASET_LIST+=("${ds}")
  done
  if [[ ${#DATASET_LIST[@]} -eq 0 ]]; then
    DATASET_LIST=("${DATASET_LIST_DEFAULT[@]}")
  fi
fi

declare -A DATASET_LABEL_MAP
declare -A DATASET_RUNBOOK_MAP
declare -A DATASET_RESULTS_ROOT_MAP
DATASET_LABEL_MAP["sift"]="sift"
DATASET_LABEL_MAP["glove"]="glove"
DATASET_LABEL_MAP["mson"]="mson"
DATASET_RUNBOOK_MAP["sift"]=""
DATASET_RUNBOOK_MAP["glove"]=""
DATASET_RUNBOOK_MAP["mson"]=""
DATASET_RESULTS_ROOT_MAP["sift"]=""
DATASET_RESULTS_ROOT_MAP["glove"]=""
DATASET_RESULTS_ROOT_MAP["mson"]=""

if [[ -n "${DATASETS:-}" ]]; then
  IFS=',' read -r -a _requested <<< "${DATASETS}"
  SELECTED_DATASETS=()
  for entry in "${_requested[@]}"; do
    ds=$(echo "${entry}" | xargs)
    [[ -z "${ds}" ]] && continue
    found=false
    for candidate in "${DATASET_LIST[@]}"; do
      if [[ "${candidate}" == "${ds}" ]]; then
        SELECTED_DATASETS+=("${ds}")
        found=true
        break
      fi
    done
    if [[ "${found}" == false ]]; then
      echo "Unknown dataset requested via DATASETS: ${ds}" >&2
      exit 1
    fi
  done
  if [[ ${#SELECTED_DATASETS[@]} -eq 0 ]]; then
    echo "DATASETS filter removed all datasets." >&2
    exit 1
  fi
else
  SELECTED_DATASETS=("${DATASET_LIST[@]}")
fi

if [[ ${#SELECTED_DATASETS[@]} -eq 0 ]]; then
  echo "No datasets configured. Populate DATASET_LIST in scripts/run_param_tuning.sh." >&2
  exit 1
fi

ALGORITHM=${ALGORITHM:-${ALGORITHM_LIST[0]}}
if [[ -z "${ALGORITHM}" ]]; then
  echo "Algorithm list is empty; please populate ALGORITHM_LIST_DEFAULT." >&2
  exit 1
fi

if [[ -z "${PARAM_GRID_FILE:-}" ]]; then
  PARAM_GRID_FILE=${PARAM_GRID_MAP[${ALGORITHM}]:-}
fi
if [[ -z "${PARAM_GRID_FILE}" ]]; then
  echo "No parameter grid configured for algorithm '${ALGORITHM}'. Update PARAM_GRID_MAP in scripts/run_param_tuning.sh." >&2
  exit 1
fi

if [[ ! -f "${PARAM_GRID_FILE}" ]]; then
  echo "Parameter grid JSON not found: ${PARAM_GRID_FILE}" >&2
  exit 1
fi

append_suffix() {
  local path="$1" suffix="$2"
  local dir base name ext
  if [[ "$path" == */* ]]; then
    dir=${path%/*}
    base=${path##*/}
  else
    dir="."
    base="$path"
  fi
  if [[ "$base" == *.* ]]; then
    name=${base%.*}
    ext=${base##*.}
    printf '%s/%s_%s.%s' "$dir" "$name" "$suffix" "$ext"
  else
    printf '%s/%s_%s' "$dir" "$base" "$suffix"
  fi
}

COMBO_TEMP=$(mktemp)
META_TEMP=$(mktemp)

python3 - "${PARAM_GRID_FILE}" "${COMBO_TEMP}" "${META_TEMP}" <<'PY'
import json
import sys
from itertools import product
from pathlib import Path

sep = "\u001f"
config_path = Path(sys.argv[1])
combo_path = Path(sys.argv[2])
meta_path = Path(sys.argv[3])

with config_path.open("r", encoding="utf-8") as fh:
    data = json.load(fh)

algorithm = data.get("algorithm")
if not algorithm:
    raise SystemExit("Config must provide 'algorithm'.")


def axis_value(axis, raw):
    if isinstance(raw, dict):
        return raw
    if "template" in axis:
        return axis["template"].format(raw)
    prefix = axis.get("prefix", "")
    suffix = axis.get("suffix", "")
    if prefix or suffix:
        return f"{prefix}{raw}{suffix}"
    return raw


def expand_axes(axes):
    if not axes:
        return []
    processed = []
    for axis in axes:
        key = axis.get("key")
        if not key:
            raise SystemExit("Each axis must define a 'key'.")
        values = axis.get("values")
        if not values:
            raise SystemExit(f"Axis '{key}' must provide non-empty 'values'.")
        processed.append((axis, key, values))
    combos = []
    for raw_combo in product(*[vals for (_, _, vals) in processed]):
        entry = {}
        for (axis, key, _), raw in zip(processed, raw_combo):
            entry[key] = axis_value(axis, raw)
        combos.append(entry)
    return combos


param_name = data.get("param_name", "param")
label_key = data.get("label_key", param_name)
baseline_param = data.get("baseline_param", "")


def short_key(key: str) -> str:
    mapping = {
        "efConstruction": "efc",
        "ef": "ef",
        "indexkey": "",
        "M": "M",
        "degree": "deg",
    }
    return mapping.get(key, key)


def derive_label(args):
    primary_key = None
    if label_key and label_key in args:
        primary_key = label_key
    else:
        for key in ("indexkey", "degree", "M", "param", "efConstruction"):
            if key in args:
                primary_key = key
                break
    if primary_key is None:
        primary_value = json.dumps(args, sort_keys=True)
    else:
        primary_value = args[primary_key]
    primary_label = str(primary_value)
    digits = "".join(ch for ch in primary_label if ch.isdigit() or ch == ".")

    extras = []
    for key in sorted(args):
        if key == primary_key:
            continue
        extras.append(f"{short_key(key)}{args[key]}")

    if extras:
        label_str = "_".join([primary_label] + extras)
    else:
        label_str = primary_label
    return label_str, digits


args_grid = data.get("args_grid")
if not args_grid:
    args_axes = data.get("args_axes")
    args_grid = expand_axes(args_axes)
if not args_grid:
    raise SystemExit("Config must define args_grid or args_axes with values.")

query_grid = data.get("query_args_grid")
if not query_grid:
    query_axes = data.get("query_axes")
    query_grid = expand_axes(query_axes)
if not query_grid:
    query_grid = [{}]

with combo_path.open("w", encoding="utf-8") as out:
    for args_entry in args_grid:
        label, label_numeric = derive_label(args_entry)
        args_json = json.dumps(args_entry, sort_keys=True)
        for query_entry in query_grid:
            query_json = json.dumps(query_entry, sort_keys=True)
            use_query = "1" if query_entry else "0"
            record = sep.join([
                args_json,
                query_json,
                label,
                label_numeric,
                use_query,
            ])
            out.write(record + "\n")

meta = {
    "algorithm": algorithm,
    "param_name": param_name,
    "label_key": label_key,
    "baseline_param": baseline_param,
    "log_dir": data.get("log_dir"),
    "summary_file": data.get("summary_file"),
    "summary_table": data.get("summary_table"),
    "results_root": data.get("results_root"),
    "recall_floor_factor": data.get("recall_floor_factor"),
    "recall_floor_min": data.get("recall_floor_min"),
    "runbook_template": data.get("runbook_template"),
}
with meta_path.open("w", encoding="utf-8") as meta_fh:
    json.dump(meta, meta_fh)
PY

if [[ ! -s "${COMBO_TEMP}" ]]; then
  echo "No parameter combinations generated from ${PARAM_GRID_FILE}" >&2
  rm -f "${COMBO_TEMP}" "${META_TEMP}"
  exit 1
fi

SETTINGS=$(python3 - "${META_TEMP}" <<'PY'
import json
import sys
import shlex

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)


def emit(key, value):
    shell_key = "CFG_" + key.upper()
    if value is None:
        print(f"{shell_key}=")
    else:
        if isinstance(value, str):
            print(f"{shell_key}={shlex.quote(value)}")
        else:
            import json as _json
            print(f"{shell_key}={shlex.quote(_json.dumps(value))}")


for key in ("algorithm", "param_name", "label_key", "baseline_param",
            "log_dir", "summary_file", "summary_table", "results_root",
            "recall_floor_factor", "recall_floor_min", "runbook_template"):
    emit(key, data.get(key))
PY
)
if [[ -n "${SETTINGS}" ]]; then
  eval "${SETTINGS}"
fi

if [[ -n "${CFG_ALGORITHM:-}" ]]; then
  if [[ "${CFG_ALGORITHM}" != "${ALGORITHM}" ]]; then
    echo "Algorithm mismatch: parameter grid reports '${CFG_ALGORITHM}' but '${ALGORITHM}' was selected." >&2
    exit 1
  fi
fi

PARAM_NAME=${PARAM_NAME:-${CFG_PARAM_NAME:-param}}
BASELINE_PARAM=${BASELINE_PARAM:-${CFG_BASELINE_PARAM:-}}
RUNBOOK_TEMPLATE=${RUNBOOK_TEMPLATE:-${CFG_RUNBOOK_TEMPLATE:-${DEFAULT_RUNBOOK_TEMPLATE}}}
GLOBAL_RESULTS_ROOT=${RESULTS_ROOT:-${CFG_RESULTS_ROOT:-${DEFAULT_RESULTS_ROOT}}}
LOG_DIR=${LOG_DIR:-${CFG_LOG_DIR:-${DEFAULT_LOG_DIR}}}
SUMMARY_FILE_BASE=${SUMMARY_FILE:-${CFG_SUMMARY_FILE:-${DEFAULT_SUMMARY_FILE}}}
SUMMARY_TABLE_BASE=${SUMMARY_TABLE:-${CFG_SUMMARY_TABLE:-${DEFAULT_SUMMARY_TABLE}}}
RECALL_FLOOR_FACTOR=${RECALL_FLOOR_FACTOR:-${CFG_RECALL_FLOOR_FACTOR:-${DEFAULT_RECALL_FLOOR_FACTOR}}}
RECALL_FLOOR_MIN=${RECALL_FLOOR_MIN:-${CFG_RECALL_FLOOR_MIN:-${DEFAULT_RECALL_FLOOR_MIN}}}

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${SUMMARY_FILE_BASE}")" 2>/dev/null || true
mkdir -p "$(dirname "${SUMMARY_TABLE_BASE}")" 2>/dev/null || true

CONFIG_PATH="neurips23/congestion/${ALGORITHM}/config.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found for ${ALGORITHM}: ${CONFIG_PATH}" >&2
  exit 1
fi

ORIGINAL_CONFIG=$(mktemp)

cleanup() {
  if [[ -f "${ORIGINAL_CONFIG}" ]]; then
    cp "${ORIGINAL_CONFIG}" "${CONFIG_PATH}" 2>/dev/null || true
    rm -f "${ORIGINAL_CONFIG}"
  fi
  rm -f "${COMBO_TEMP}" "${META_TEMP}"
}
trap cleanup EXIT

cp "${CONFIG_PATH}" "${ORIGINAL_CONFIG}"

ensure_groundtruth() {
  local dataset="$1"
  local runbook="$2"
  if [[ "${GROUNDTRUTH_ENABLED}" != "true" && "${GROUNDTRUTH_ENABLED}" != "1" ]]; then
    return
  fi
  python3 - "$dataset" "$runbook" "$GROUNDTRUTH_TOOL" "$GROUNDTRUTH_FORCE" "$GROUNDTRUTH_EXTRA_ARGS_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

from benchmark.datasets import DATASETS

dataset = sys.argv[1]
runbook = sys.argv[2]
tool = os.path.expanduser(sys.argv[3])
force = sys.argv[4].lower() in {"1", "true", "yes"}
extra_args = json.loads(sys.argv[5])

ds = DATASETS[dataset]()
gt_dir = Path(ds.basedir) / str(ds.nb) / Path(runbook).name

if gt_dir.exists() and any(gt_dir.iterdir()) and not force:
    print(f"Ground-truth already present in {gt_dir}; skipping.")
    sys.exit(0)

cmd = [
    "python3",
    "benchmark/congestion/compute_gt.py",
    "--dataset",
    dataset,
    "--runbook_file",
    runbook,
    "--gt_cmdline_tool",
    tool,
]
cmd.extend(extra_args)
print(f"Generating ground-truth into {gt_dir}")
import subprocess
subprocess.run(cmd, check=True)
PY
}

update_algorithm_config() {
  local dataset="$1"
  local args_json="$2"
  local query_json="$3"
  local use_query="$4"

  python3 - "${CONFIG_PATH}" "${dataset}" "${ALGORITHM}" "${args_json}" "${query_json}" "${use_query}" <<'PYCFG'
import json
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
dataset = sys.argv[2]
algo = sys.argv[3]
args_entry = json.loads(sys.argv[4])
query_entry = json.loads(sys.argv[5])
use_query = sys.argv[6] == "1"

with config_path.open("r", encoding="utf-8") as fh:
    data = yaml.safe_load(fh)

if not isinstance(data, dict):
    raise SystemExit("Unexpected YAML structure in algorithm config.")

updated = False

def update_block(block):
    inner = block.get(algo)
    if not isinstance(inner, dict):
        return False
    run_groups = inner.setdefault("run-groups", {})
    base_group = run_groups.setdefault("base", {})
    base_group["args"] = json.dumps([args_entry], sort_keys=True)
    if use_query:
        base_group["query-args"] = json.dumps([query_entry], sort_keys=True)
    else:
        base_group.pop("query-args", None)
    return True


for key, block in data.items():
    if not isinstance(block, dict):
        continue
    if key in ("random-xs", dataset):
        if update_block(block):
            updated = True

if not updated:
    raise SystemExit(f"Failed to update args for algorithm '{algo}' dataset '{dataset}'.")

with config_path.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(data, fh, sort_keys=False)
PYCFG
}

collect_metrics() {
  local results_root="$1"
  local runbook="$2"
  local dataset="$3"
  local algo="$4"
  local start_ts="$5"
  local label_base="$6"
  local label_full="$7"
  local query_json="$8"

  python3 - "$results_root" "$runbook" "$dataset" "$algo" "$start_ts" "$label_base" "$label_full" "$query_json" <<'PYMET' | tail -n 1
import csv
import json
import math
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
import h5py

from benchmark.datasets import DATASETS
from benchmark.results import load_all_results
from benchmark.plotting.utils import compute_metrics_all_runs

csv.field_size_limit(sys.maxsize)

results_root = Path(sys.argv[1])
runbook_path = Path(sys.argv[2])
dataset = sys.argv[3]
algorithm = sys.argv[4]
start_ts = float(sys.argv[5]) - 1.0
base_label = sys.argv[6]
label_full = sys.argv[7]
query_json_raw = sys.argv[8]

try:
    query_entry = json.loads(query_json_raw)
    if not isinstance(query_entry, dict):
        query_entry = {}
except Exception:
    query_entry = {}

runbook_dir = results_root / runbook_path.name
if not runbook_dir.exists():
    raise SystemExit("ERR Missing results directory for runbook.")

skip_suffixes = (
    "_batchLatency.csv",
    "_batchqueryThroughput.csv",
    "_batchinsertThroughtput.csv",
)

candidates = []
for path in runbook_dir.rglob("*.csv"):
    if not path.is_file():
        continue
    if any(path.name.endswith(sfx) for sfx in skip_suffixes):
        continue
    if algorithm not in path.parts:
        continue
    if dataset not in path.parts:
        continue
    if path.stat().st_mtime <= start_ts:
        continue
    candidates.append(path)

if not candidates:
    for path in runbook_dir.rglob("*.csv"):
        if not path.is_file():
            continue
        if any(path.name.endswith(sfx) for sfx in skip_suffixes):
            continue
        if algorithm not in path.parts:
            continue
        if dataset not in path.parts:
            continue
        candidates.append(path)

if not candidates:
    raise SystemExit("ERR No result CSV located for this trial.")

candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
target = candidates[0]

with target.open("r", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    try:
        row = next(reader)
    except StopIteration as exc:
        raise SystemExit("ERR Empty result CSV.") from exc


def to_float(value):
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple)):
        if not value:
            return float("nan")
        value = value[0]
    try:
        import numpy as np
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return float("nan")
            value = value.flat[0]
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return float("nan")


recall_keys = [
    "continuousRecall_0",
    "continuousRecall",
    "knn_0",
    "knn",
    "k-nn",
    "Recall",
]
qps_keys = [
    "continuousThroughput_0",
    "continuousThroughput",
    "queryThroughput0",
    "queryThroughput_0",
    "search_throughput",
    "steady_throughput",
    "throughput",
]

recall = float("nan")
for key in recall_keys:
    value = row.get(key)
    fvalue = to_float(value)
    if not math.isnan(fvalue):
        recall = fvalue
        break

qps = float("nan")
for key in qps_keys:
    value = row.get(key)
    fvalue = to_float(value)
    if not math.isnan(fvalue):
        qps = fvalue
        break

count_value = row.get("count")
try:
    count_value = int(count_value)
except (TypeError, ValueError):
    count_value = 10

if math.isnan(recall) or math.isnan(qps):
    h5_path = target.with_suffix(".hdf5")
    if h5_path.exists():
        try:
            ds = DATASETS[dataset]()
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
                with h5py.File(h5_path, "r+") as run_file:
                    properties = dict(run_file.attrs)
                    properties["filename"] = h5_path.name
                    properties["_read_only"] = False
                    res_iter = [(properties, run_file)]
                    metrics_list = list(
                        compute_metrics_all_runs(
                            ds,
                            dataset,
                            res_iter,
                            neurips23track="congestion",
                            runbook_path=str(runbook_path),
                        )
                    )
            if metrics_list:
                entry = metrics_list[0]
                if math.isnan(recall):
                    recall = to_float(entry.get("continuousRecall_0"))
                if math.isnan(qps):
                    for key in (
                        "continuousThroughput_0",
                        "queryThroughput0",
                        "continuousThroughput",
                        "throughput",
                    ):
                        qps = to_float(entry.get(key))
                        if not math.isnan(qps):
                            break
        except Exception:
            pass

if math.isnan(recall) or math.isnan(qps):
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        ds = DATASETS[dataset]()
        res_iter = load_all_results(
            dataset,
            count_value,
            neurips23track="congestion",
            runbook_path=str(runbook_path),
        )
        metrics_iter = compute_metrics_all_runs(
            ds,
            dataset,
            res_iter,
            neurips23track="congestion",
            runbook_path=str(runbook_path),
        )

    candidate_labels = {label_full, base_label}
    if query_entry:
        suffix = "_".join(f"{k}{query_entry[k]}" for k in sorted(query_entry))
        candidate_labels.add(f"{base_label}_{suffix}")

    for entry in metrics_iter:
        params = entry.get("parameters")
        if params in candidate_labels:
            if math.isnan(recall):
                recall = to_float(entry.get("continuousRecall_0"))
            if math.isnan(qps):
                for key in (
                    "continuousThroughput_0",
                    "queryThroughput0",
                    "continuousThroughput",
                    "throughput",
                ):
                    qps = to_float(entry.get(key))
                    if not math.isnan(qps):
                        break
            break

if math.isnan(qps):
    throughput_file = target.with_name(target.stem + "_batchqueryThroughput.csv")
    if throughput_file.exists():
        values = []
        with throughput_file.open("r", encoding="utf-8") as fh_bt:
            reader_bt = csv.DictReader(fh_bt)
            for row_bt in reader_bt:
                val = to_float(row_bt.get("batchThroughput"))
                if not math.isnan(val):
                    values.append(val)
        if values:
            qps = sum(values) / len(values)

if not math.isfinite(recall):
    recall = float("nan")
if not math.isfinite(qps):
    qps = float("nan")


def fmt(value):
    return "NaN" if math.isnan(value) else f"{value:.6f}"


print(f"{fmt(recall)} {fmt(qps)} {target}")
PYMET
}

summarize_dataset() {
  local summary_file="$1"
  local baseline="$2"
  local recall_floor_factor="$3"
  local recall_floor_min="$4"
  local summary_out="$5"
  local param_name="$6"
  local algorithm="$7"
  local dataset_id="$8"
  local dataset_label="$9"

  python3 - "$summary_file" "$baseline" "$recall_floor_factor" "$recall_floor_min" "$summary_out" "$param_name" "$algorithm" "$dataset_id" "$dataset_label" <<'PYSUM'
import csv
import math
import sys
from io import StringIO
from pathlib import Path

csv.field_size_limit(sys.maxsize)

summary_path = Path(sys.argv[1])
baseline_raw = sys.argv[2]
floor_factor = float(sys.argv[3])
floor_min = float(sys.argv[4])
summary_out = Path(sys.argv[5])
param_name = sys.argv[6]
algorithm = sys.argv[7]
dataset_id = sys.argv[8]
dataset_label = sys.argv[9]
dataset_display = dataset_label if dataset_label else dataset_id

rows = []
with summary_path.open("r", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for entry in reader:
        rows.append({
            "param": entry.get("param", ""),
            "param_numeric": entry.get("param_numeric", ""),
            "qps": float(entry.get("qps", 0.0)),
            "recall": float(entry.get("recall", 0.0)),
        })

if not rows:
    raise SystemExit("No trials were recorded; summary is empty.")

baseline = None
baseline_str = str(baseline_raw) if baseline_raw else ""
baseline_digits = "".join(ch for ch in baseline_str if ch.isdigit() or ch == ".")
for row in rows:
    if baseline_str and row["param"] == baseline_str:
        baseline = row
        break
    if baseline_digits and row.get("param_numeric") == baseline_digits:
        baseline = row
        break
if baseline is None:
    baseline = rows[0]

recall_floor = max(baseline["recall"] * floor_factor, floor_min)

eligible = [r for r in rows if r["recall"] >= recall_floor]
eligible.sort(key=lambda r: (r["qps"], r["recall"], r["param"]), reverse=True)
best = eligible[0] if eligible else None

jumps = []
for left, right in zip(rows, rows[1:]):
    if left["qps"] <= 0:
        delta_qps_pct = math.inf
    else:
        delta_qps_pct = 100.0 * (right["qps"] - left["qps"]) / left["qps"]
    delta_recall_pp = 100.0 * (right["recall"] - left["recall"])
    cross_floor = (
        (left["recall"] < recall_floor <= right["recall"]) or
        (right["recall"] < recall_floor <= left["recall"])
    )
    near_best = bool(best) and (left["param"] == best["param"] or right["param"] == best["param"])
    jumps.append({
        "from": left["param"],
        "to": right["param"],
        "delta_qps_pct": delta_qps_pct,
        "delta_recall_pp": delta_recall_pp,
        "near_best": near_best,
        "cross_floor": cross_floor,
    })

jumps.sort(key=lambda r: abs(r["delta_qps_pct"]) if math.isfinite(r["delta_qps_pct"]) else float("inf"), reverse=True)

buf = StringIO()
buf.write(f"Summary for {algorithm} on {dataset_display}\n")
if dataset_label and dataset_label != dataset_id:
    buf.write(f"(dataset id: {dataset_id})\n")
buf.write(f"Recall floor = max(baseline×{floor_factor:.2f}, {floor_min:.2f}) = {recall_floor:.6f}\n")
if best:
    buf.write(f"Best {param_name} = {best['param']} | QPS = {best['qps']:.3f} | Recall = {best['recall']:.6f} | Trials = {len(rows)}\n")
else:
    buf.write("No feasible point (all recall below floor).\n")

buf.write("\nPer-trial metrics (sorted by parameter):\n")
buf.write(f"{param_name:>16} {'Recall':>10} {'QPS':>12}\n")
for row in rows:
    buf.write(f"{row['param']:>16} {row['recall']:>10.6f} {row['qps']:>12.3f}\n")

buf.write("\nTop-3 ΔQPS% intervals:\n")
if not jumps:
    buf.write("Not enough trials to compute jumps.\n")
else:
    buf.write(f"{'from→to':<20} {'ΔQPS%':>10} {'ΔRecall(pp)':>14} {'near p*':>10} {'cross floor':>12}\n")
    for jump in jumps[:3]:
        dq = "∞" if not math.isfinite(jump['delta_qps_pct']) else f"{jump['delta_qps_pct']:.2f}"
        dr = f"{jump['delta_recall_pp']:.2f}"
        buf.write(f"{jump['from']}→{jump['to']:<18} {dq:>10} {dr:>14} {str(jump['near_best']):>10} {str(jump['cross_floor']):>12}\n")

report = buf.getvalue()
print()
print(report)
summary_out.write_text(report, encoding="utf-8")
PYSUM
}

for DATASET in "${SELECTED_DATASETS[@]}"; do
  DATASET_LABEL="${DATASET_LABEL_MAP[$DATASET]:-$DATASET}"
  RUNBOOK_OVERRIDE="${DATASET_RUNBOOK_MAP[$DATASET]:-}"
  DATASET_RESULTS_ROOT="${DATASET_RESULTS_ROOT_MAP[$DATASET]:-}"

  RUNBOOK_PATH="${RUNBOOK_OVERRIDE}"
  if [[ -z "${RUNBOOK_PATH}" ]]; then
    RUNBOOK_PATH="${RUNBOOK_TEMPLATE//\{algorithm\}/$ALGORITHM}"
    RUNBOOK_PATH="${RUNBOOK_PATH//\{dataset\}/$DATASET}"
  fi
  if [[ -z "${RUNBOOK_PATH}" ]]; then
    echo "No runbook defined for dataset ${DATASET}." >&2
    exit 1
  fi

  RESULTS_ROOT_CUR="${DATASET_RESULTS_ROOT:-$GLOBAL_RESULTS_ROOT}"
  if [[ -z "${RESULTS_ROOT_CUR}" ]]; then
    echo "No results_root defined for dataset ${DATASET}." >&2
    exit 1
  fi

  dataset_safe=$(printf '%s' "${DATASET}" | tr -cs '[:alnum:]' '_')
  summary_suffix="${ALGORITHM}_${dataset_safe}"
  SUMMARY_FILE_CUR=$(append_suffix "${SUMMARY_FILE_BASE}" "${summary_suffix}")
  SUMMARY_TABLE_CUR=$(append_suffix "${SUMMARY_TABLE_BASE}" "${summary_suffix}")
  mkdir -p "$(dirname "${SUMMARY_FILE_CUR}")"
  mkdir -p "$(dirname "${SUMMARY_TABLE_CUR}")"
  echo "param,param_numeric,qps,recall" > "${SUMMARY_FILE_CUR}"

  echo
  echo "=== Running ${ALGORITHM} on dataset ${DATASET} (${DATASET_LABEL}) ==="
  ensure_groundtruth "${DATASET}" "${RUNBOOK_PATH}"

  while IFS=$'\x1f' read -r ARGS_JSON QUERY_JSON LABEL LABEL_NUMERIC USE_QUERY_FLAG; do
    [[ -z "${ARGS_JSON}" ]] && continue
    update_algorithm_config "${DATASET}" "${ARGS_JSON}" "${QUERY_JSON}" "${USE_QUERY_FLAG}"

    start_ts=$(python3 - <<'PYTS'
import time
print(time.time())
PYTS
    )

    label_safe=$(printf '%s' "${LABEL}" | tr -cs '[:alnum:]' '_')
    query_suffix=$(python3 - "${QUERY_JSON}" <<'PYQ'
import json
import sys
entry = json.loads(sys.argv[1])
if not entry:
    print("")
else:
    parts = []
    for key in sorted(entry):
        parts.append(f"{key}{entry[key]}")
    print("_".join(parts))
PYQ
    )
    if [[ -n "${query_suffix}" ]]; then
      label_safe="${label_safe}_${query_suffix}"
    fi
    LABEL_FULL="${LABEL}"
    if [[ -n "${query_suffix}" ]]; then
      LABEL_FULL="${LABEL}_${query_suffix}"
    fi
    LOG_FILE="${LOG_DIR}/${ALGORITHM}_${DATASET}_${label_safe}.log"

    python3 run.py \
      --neurips23track congestion \
      --algorithm "${ALGORITHM}" \
      --dataset "${DATASET}" \
      --nodocker \
      --rebuild \
      --force \
      --runbook_path "${RUNBOOK_PATH}" 2>&1 | tee "${LOG_FILE}"

    METRICS=$(collect_metrics "${RESULTS_ROOT_CUR}" "${RUNBOOK_PATH}" "${DATASET}" "${ALGORITHM}" "${start_ts}" "${LABEL}" "${LABEL_FULL}" "${QUERY_JSON}")

    read -r RECALL QPS RESULT_FILE <<< "${METRICS}"
    printf "  -> recall=%s, qps=%s (source: %s)\n" "${RECALL}" "${QPS}" "${RESULT_FILE}"
    lc_recall=$(printf "%s" "${RECALL}" | tr '[:upper:]' '[:lower:]')
    lc_qps=$(printf "%s" "${QPS}" | tr '[:upper:]' '[:lower:]')
    if [[ ${lc_recall} == nan || ${lc_qps} == nan ]]; then
      echo "    warning: recall/qps unavailable for ${LABEL_FULL}; skipping entry"
      continue
    fi
    printf "%s,%s,%s,%s\n" "${LABEL_FULL}" "${LABEL_NUMERIC}" "${QPS}" "${RECALL}" >> "${SUMMARY_FILE_CUR}"
  done < "${COMBO_TEMP}"

  DATA_ROW_COUNT=$(($(wc -l < "${SUMMARY_FILE_CUR}") - 1))
  if (( DATA_ROW_COUNT <= 0 )); then
    echo "warning: no successful trials recorded for ${ALGORITHM} on ${DATASET}" >&2
    continue
  fi

  summarize_dataset "${SUMMARY_FILE_CUR}" "${BASELINE_PARAM}" "${RECALL_FLOOR_FACTOR}" "${RECALL_FLOOR_MIN}" "${SUMMARY_TABLE_CUR}" "${PARAM_NAME}" "${ALGORITHM}" "${DATASET}" "${DATASET_LABEL}"

  if [[ "${DATASET_LABEL}" != "${DATASET}" ]]; then
    echo "Results for ${ALGORITHM} on ${DATASET_LABEL} (${DATASET}): raw=${SUMMARY_FILE_CUR}, summary=${SUMMARY_TABLE_CUR}"
  else
    echo "Results for ${ALGORITHM} on ${DATASET}: raw=${SUMMARY_FILE_CUR}, summary=${SUMMARY_TABLE_CUR}"
  fi
done
