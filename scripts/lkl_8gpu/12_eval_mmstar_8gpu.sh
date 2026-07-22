#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"

dataset="MMStar"
data_file="$EVAL_DATA_ROOT/$dataset.tsv"
run_id="$(date +%Y%m%d_%H%M%S)"
log_file="$EVAL_LOG_ROOT/colt_eval_${dataset}_8gpu_${run_id}.log"
gpu_csv="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
exec > >(tee -a "$log_file") 2>&1

bash "$REPO_ROOT/scripts/lkl_8gpu/07_verify_final_model.sh"
bash "$REPO_ROOT/scripts/lkl_8gpu/09_download_eval_data.sh" "$dataset"

IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
if (( ${#gpu_ids[@]} != 8 )); then
  echo "COLT_EVAL_GPUS must contain exactly eight physical GPU ids: $gpu_csv" >&2
  exit 1
fi

declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id: $gpu" >&2
    exit 1
  fi
  if [[ -n "${seen_gpu[$gpu]:-}" ]]; then
    echo "Duplicate GPU id: $gpu" >&2
    exit 1
  fi
  seen_gpu[$gpu]=1

  if ! memory_used="$(
      nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
    )"; then
    echo "Unable to query physical GPU $gpu." >&2
    exit 1
  fi
  if (( memory_used >= 500 )); then
    echo "Physical GPU $gpu is not free: ${memory_used} MiB is in use." >&2
    exit 1
  fi
done

eval_fingerprint="$(
  {
    sha256sum \
      "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py" \
      "$VLMEVAL_ROOT/vlmeval/vlm/colt_qwen3_vl.py" \
      "$VLMEVAL_ROOT/vlmeval/config.py" \
      "$FINAL_MODEL_DIR/model.safetensors.index.json" \
      "$FINAL_MODEL_DIR/config.json" \
      "$FINAL_MODEL_DIR/generation_config.json" \
      "$FINAL_MODEL_DIR/preprocessor_config.json" \
      "$FINAL_MODEL_DIR/tokenizer_config.json"
    find "$FINAL_MODEL_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' \
      -printf '%f %s %T@\n' | sort
  } \
  | sha256sum \
  | cut -c1-12
)"
eval_profile="dp8_seed${COLT_EVAL_SEED}_${eval_fingerprint}"
work_dir="$EVAL_OUTPUT_ROOT/codefaithful/$dataset/$eval_profile"

export CUDA_VISIBLE_DEVICES="$gpu_csv"
export MMEVAL_ROOT="$work_dir"
export VLMEVAL_EVAL_ID="COLT_${eval_profile}"
export PRED_FORMAT=xlsx
export EVAL_FORMAT=csv
export DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE
mkdir -p "$work_dir"

echo "Dataset: $dataset"
echo "Backend: PyTorch + Transformers + FlashAttention 2 (not vLLM)"
echo "Parallelism: 8 independent model replicas with VLMEvalKit rank-wise data sharding"
echo "Physical GPUs: $CUDA_VISIBLE_DEVICES"
echo "Evaluation fingerprint: $eval_fingerprint"
echo "Model: $COLT_EVAL_MODEL_PATH"
echo "Data: $data_file"
echo "Results: $work_dir"
echo "Log: $log_file"

cd "$VLMEVAL_ROOT"
args=(
  --standalone
  --nnodes=1
  --nproc_per_node=8
  --max_restarts=0
  run.py
  --data "$dataset"
  --model "$EVAL_MODEL_NAME"
  --work-dir "$work_dir"
  --mode all
  --judge exact_matching
)
if [[ "${EVAL_VERBOSE:-0}" == "1" ]]; then
  args+=(--verbose)
fi
torchrun "${args[@]}"

python "$REPO_ROOT/scripts/lkl_8gpu/validate_mmstar_result.py" \
  "$work_dir" \
  "$EVAL_MODEL_NAME" \
  "$data_file"

echo "MMStar 8-GPU evaluation completed."
echo "Log: $log_file"
