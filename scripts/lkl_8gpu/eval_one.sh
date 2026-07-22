#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"

dataset="${1:?Usage: eval_one.sh DATASET GPU_ID}"
gpu_id="${2:?Usage: eval_one.sh DATASET GPU_ID}"
data_file="$EVAL_DATA_ROOT/$dataset.tsv"
work_dir="$EVAL_OUTPUT_ROOT/codefaithful/$dataset"
log_file="$EVAL_LOG_ROOT/colt_eval_${dataset}_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "$data_file" ]]; then
  echo "Missing dataset: $data_file" >&2
  echo "Run scripts/lkl_8gpu/09_download_eval_data.sh first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$gpu_id"
export MMEVAL_ROOT="$work_dir"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE
mkdir -p "$work_dir"

memory_used="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( memory_used >= 500 )); then
  echo "Physical GPU $gpu_id is not free: ${memory_used} MiB is in use." >&2
  exit 1
fi

exec > >(tee -a "$log_file") 2>&1
echo "Dataset: $dataset"
echo "GPU: $gpu_id"
echo "Model: $COLT_EVAL_MODEL_PATH"
echo "Data: $data_file"
echo "Results: $work_dir"
echo "Log: $log_file"

cd "$VLMEVAL_ROOT"
args=(
  run.py
  --data "$dataset"
  --model "$EVAL_MODEL_NAME"
  --work-dir "$work_dir"
  --mode all
  --judge exact_matching
  --reuse
)
if [[ "${EVAL_VERBOSE:-0}" == "1" ]]; then
  args+=(--verbose)
fi
python "${args[@]}"

echo "Evaluation completed: $dataset"
find "$work_dir" -type f \( -name '*_acc.csv' -o -name '*_score.csv' -o -name '*.xlsx' \) -print | sort
