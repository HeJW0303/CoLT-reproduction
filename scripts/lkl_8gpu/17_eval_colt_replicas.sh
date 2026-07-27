#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"

group="${1:-remaining7}"
workers_per_gpu="${VLMEVAL_WORKERS_PER_GPU:-3}"
gpu_csv="${COLT_EVAL_GPUS:-4,5,6,7}"
prefetch="${VLMEVAL_PREFETCH:-1}"
empty_cache_every_n="${VLMEVAL_EMPTY_CACHE_EVERY_N:-0}"
dist_backend="${VLMEVAL_DIST_BACKEND:-gloo}"
reseed_per_sample="${COLT_RESEED_PER_SAMPLE:-1}"

case "$group" in
  mmstar)
    datasets=(MMStar)
    ;;
  remaining7)
    datasets=(ChartQA_TEST AI2D_TEST MMBench_DEV_EN ScienceQA_TEST TextVQA_VAL MMT-Bench_VAL SEEDBench_IMG)
    ;;
  all8)
    datasets=(ChartQA_TEST AI2D_TEST MMBench_DEV_EN ScienceQA_TEST TextVQA_VAL MMT-Bench_VAL SEEDBench_IMG MMStar)
    ;;
  *)
    echo "Usage: $0 mmstar|remaining7|all8" >&2
    exit 1
    ;;
esac

if [[ ! "$workers_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "VLMEVAL_WORKERS_PER_GPU must be a positive integer: $workers_per_gpu" >&2
  exit 1
fi

IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
if (( ${#gpu_ids[@]} == 0 )); then
  echo "COLT_EVAL_GPUS must contain at least one physical GPU id." >&2
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
  memory_used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ ! "$memory_used" =~ ^[0-9]+$ ]]; then
    echo "Unexpected memory usage returned for physical GPU $gpu: $memory_used" >&2
    exit 1
  fi
  if (( memory_used >= 500 )); then
    echo "Physical GPU $gpu is not free: ${memory_used} MiB is in use." >&2
    exit 1
  fi
done

nproc_per_node=$(( ${#gpu_ids[@]} * workers_per_gpu ))
run_id="$(date +%Y%m%d_%H%M%S)"
log_file="$EVAL_LOG_ROOT/colt_eval_${group}_replicas_${run_id}.log"
exec > >(tee -a "$log_file") 2>&1

bash "$REPO_ROOT/scripts/lkl_8gpu/07_verify_final_model.sh"
bash "$REPO_ROOT/scripts/lkl_8gpu/09_download_eval_data.sh" "${datasets[@]}"

eval_fingerprint="$(
  {
    sha256sum \
      "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py" \
      "$VLMEVAL_ROOT/run.py" \
      "$VLMEVAL_ROOT/vlmeval/inference.py" \
      "$VLMEVAL_ROOT/vlmeval/vlm/colt_qwen3_vl.py" \
      "$VLMEVAL_ROOT/vlmeval/config.py" \
      "$REPO_ROOT/scripts/lkl_8gpu/17_eval_colt_replicas.sh"
    find "$FINAL_MODEL_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' \
      -printf '%f %s %T@\n' | sort
  } | sha256sum | cut -c1-12
)"
eval_profile="replicas${nproc_per_node}_w${workers_per_gpu}_p${prefetch}_c${empty_cache_every_n}_r${reseed_per_sample}_${group}_seed${COLT_EVAL_SEED}_${eval_fingerprint}"
work_dir="$EVAL_OUTPUT_ROOT/throughput_replicas/$group/$eval_profile"
eval_id="COLT_${eval_profile}"

export CUDA_VISIBLE_DEVICES="$gpu_csv"
export MMEVAL_ROOT="$work_dir"
export VLMEVAL_EVAL_ID="$eval_id"
export PRED_FORMAT=xlsx
export EVAL_FORMAT=csv
export DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"
export VLMEVAL_WORKERS_PER_GPU="$workers_per_gpu"
export VLMEVAL_PREFETCH="$prefetch"
export VLMEVAL_EMPTY_CACHE_EVERY_N="$empty_cache_every_n"
export VLMEVAL_DIST_BACKEND="$dist_backend"
export COLT_RESEED_PER_SAMPLE="$reseed_per_sample"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE
mkdir -p "$work_dir"

echo "Datasets: ${datasets[*]}"
echo "Backend: Transformers + FlashAttention 2 (not vLLM)"
echo "Physical GPUs: $CUDA_VISIBLE_DEVICES"
echo "Workers per GPU: $VLMEVAL_WORKERS_PER_GPU"
echo "Total model workers: $nproc_per_node"
echo "CPU preprocessing prefetch: $VLMEVAL_PREFETCH"
echo "empty_cache frequency: $VLMEVAL_EMPTY_CACHE_EVERY_N (0=disabled)"
echo "Distributed backend: $VLMEVAL_DIST_BACKEND"
echo "Per-sample reseeding: $COLT_RESEED_PER_SAMPLE"
echo "Evaluation fingerprint: $eval_fingerprint"
echo "Evaluation id: $eval_id"
echo "Results: $work_dir"
echo "Log: $log_file"

cd "$VLMEVAL_ROOT"
args=(
  --standalone
  --nnodes=1
  --nproc_per_node="$nproc_per_node"
  --max_restarts=0
  run.py
  --data "${datasets[@]}"
  --model "$EVAL_MODEL_NAME"
  --work-dir "$work_dir"
  --mode all
  --judge exact_matching
  --reuse
)
if [[ "${EVAL_VERBOSE:-0}" == "1" ]]; then
  args+=(--verbose)
fi
torchrun "${args[@]}"

if [[ "$group" == "mmstar" ]]; then
  python "$REPO_ROOT/scripts/lkl_8gpu/validate_mmstar_result.py" \
    "$work_dir" "$EVAL_MODEL_NAME" "$EVAL_DATA_ROOT/MMStar.tsv"
else
  python "$REPO_ROOT/scripts/lkl_8gpu/validate_eval_suite.py" \
    "$work_dir" "$EVAL_MODEL_NAME" "$eval_id" "$EVAL_DATA_ROOT" "${datasets[@]}"
fi

echo "CoLT replica evaluation completed: $group"
echo "Log: $log_file"
