#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env
validate_gpu_profile

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export DECORD_EOF_RETRY_MAX=2048001
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export COLT_DECODER_MODEL_PATH="$DECODER_MODEL_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-CoLT-reproduction}"

train_config="${COLT_TRAIN_CONFIG:-$TRAIN_CONFIG}"
output_dir="${COLT_TRAIN_OUTPUT_DIR:-$OUTPUT_ROOT/colt_codefaithful}"
train_entry_script="${COLT_TRAIN_ENTRY_SCRIPT:-$REPO_ROOT/scripts/lkl_8gpu/06_train.sh}"
run_record_prefix="${COLT_TRAIN_RECORD_PREFIX:-colt_run}"
log_prefix="${COLT_TRAIN_LOG_PREFIX:-colt_train}"
if [[ ! -f "$train_config" ]]; then
  echo "Missing training config: $train_config" >&2
  exit 1
fi
config_output_dir="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$train_config")"
if [[ "$config_output_dir" != "$output_dir" ]]; then
  echo "Training config output_dir does not match the guarded output directory." >&2
  echo "Config: $config_output_dir" >&2
  echo "Guard:  $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir" "$LOG_ROOT"
if find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  if [[ "${RESUME:-0}" != "1" ]]; then
    echo "Output directory is not empty: $output_dir" >&2
    echo "For a verified interrupted run, rerun with RESUME=1 to let LLaMA-Factory resume its last checkpoint." >&2
    exit 1
  fi
  latest_checkpoint="$(find "$output_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
  if [[ -z "$latest_checkpoint" || ! -f "$latest_checkpoint/trainer_state.json" ]]; then
    echo "RESUME=1 was requested, but no complete Trainer checkpoint was found." >&2
    exit 1
  fi
  echo "LLaMA-Factory will resume from: $latest_checkpoint"
fi

require_free_gib 200

cd "$REPO_ROOT"
if [[ "${COLT_PAPER_FAITHFUL:-0}" == "1" ]]; then
  python "$REPO_ROOT/scripts/lkl_8gpu/verify_paper_faithful.py"
fi
if [[ "${COLT_ORACLE_K_ENABLED:-0}" == "1" ]]; then
  python -m unittest discover -s "$REPO_ROOT/tests/oracle_k" -p 'test_*.py' -v
fi
run_stamp="$(date +%Y%m%d_%H%M%S)"
run_record="$LOG_ROOT/${run_record_prefix}_$run_stamp"
mkdir -p "$run_record"
cp "$train_config" "$run_record/"
cp "$REPO_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_a100.json" "$run_record/"
cp "$train_entry_script" "$run_record/"
if [[ "${COLT_PAPER_FAITHFUL:-0}" == "1" ]]; then
  cp "$REPO_ROOT/scripts/lkl_8gpu/06_train.sh" "$run_record/"
  cp "$REPO_ROOT/scripts/lkl_8gpu/verify_paper_faithful.py" "$run_record/"
  cp "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py" "$run_record/"
  cp "$REPO_ROOT/LLaMA-Factory/src/llamafactory/model/adapter.py" "$run_record/"
  cp "$REPO_ROOT/LLaMA-Factory/src/llamafactory/model/model_utils/visual.py" "$run_record/"
  cp "$REPO_ROOT/LLaMA-Factory/src/llamafactory/train/tuner.py" "$run_record/"
fi
if [[ "${COLT_ORACLE_K_ENABLED:-0}" == "1" ]]; then
  cp "$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_oracle_k_predictor.yaml" "$run_record/"
  cp "$REPO_ROOT/LLaMA-Factory/src/llamafactory/train/tuner.py" "$run_record/"
  cp "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py" "$run_record/"
  cp "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_oracle_k.py" "$run_record/"
  cp "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/oracle_k.py" "$run_record/"
  cp "$REPO_ROOT/tests/oracle_k/test_oracle_k.py" "$run_record/"
  cp "$REPO_ROOT/scripts/lkl_8gpu/19_train_oracle_k_predictor.sh" "$run_record/"
fi
git -c safe.directory="$REPO_ROOT" rev-parse HEAD > "$run_record/git_head.txt"
git -c safe.directory="$REPO_ROOT" status --short > "$run_record/git_status.txt"
git -c safe.directory="$REPO_ROOT" diff --binary > "$run_record/git_diff.patch"
python -m pip freeze > "$run_record/pip_freeze.txt"
{
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'FORCE_TORCHRUN=%s\n' "$FORCE_TORCHRUN"
  printf 'NPROC_PER_NODE=%s\n' "$NPROC_PER_NODE"
  printf 'NNODES=%s\n' "$NNODES"
  printf 'NODE_RANK=%s\n' "$NODE_RANK"
  printf 'MASTER_ADDR=%s\n' "$MASTER_ADDR"
  printf 'MASTER_PORT=%s\n' "$MASTER_PORT"
  printf 'OMP_NUM_THREADS=%s\n' "$OMP_NUM_THREADS"
  printf 'NCCL_DEBUG=%s\n' "$NCCL_DEBUG"
  printf 'WANDB_MODE=%s\n' "$WANDB_MODE"
  printf 'WANDB_PROJECT=%s\n' "$WANDB_PROJECT"
  printf 'HF_HOME=%s\n' "$HF_HOME"
  printf 'TMPDIR=%s\n' "$TMPDIR"
  printf 'COLT_DECODER_MODEL_PATH=%s\n' "$COLT_DECODER_MODEL_PATH"
  printf 'COLT_PAPER_FAITHFUL=%s\n' "${COLT_PAPER_FAITHFUL:-0}"
  printf 'COLT_BATCH_AUX_DECODERS=%s\n' "${COLT_BATCH_AUX_DECODERS:-0}"
  printf 'COLT_COMPONENT_LOG_EVERY=%s\n' "${COLT_COMPONENT_LOG_EVERY:-8}"
  printf 'COLT_BENCHMARK_MODE=%s\n' "${COLT_BENCHMARK_MODE:-0}"
  printf 'COLT_SKIP_FINAL_SAVE=%s\n' "${COLT_SKIP_FINAL_SAVE:-0}"
  printf 'COLT_ORACLE_K_ENABLED=%s\n' "${COLT_ORACLE_K_ENABLED:-0}"
  printf 'COLT_ORACLE_K_PREDICTOR_ENABLED=%s\n' "${COLT_ORACLE_K_PREDICTOR_ENABLED:-0}"
  printf 'COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT=%s\n' "${COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT:-0.2}"
  printf 'COLT_ORACLE_K_DYNAMIC_INFERENCE=%s\n' "${COLT_ORACLE_K_DYNAMIC_INFERENCE:-0}"
  printf 'COLT_TRAIN_CONFIG=%s\n' "$train_config"
  printf 'COLT_TRAIN_OUTPUT_DIR=%s\n' "$output_dir"
} > "$run_record/environment.txt"

log_file="$LOG_ROOT/${log_prefix}_$run_stamp.log"
echo "Training log: $log_file"
llamafactory-cli train "$train_config" 2>&1 | tee "$log_file"
