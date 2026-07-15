#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env

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

output_dir="$OUTPUT_ROOT/colt_codefaithful"
mkdir -p "$output_dir" "$WORKSPACE_ROOT/logs"
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
run_stamp="$(date +%Y%m%d_%H%M%S)"
run_record="$WORKSPACE_ROOT/logs/colt_run_$run_stamp"
mkdir -p "$run_record"
cp "$A100_CONFIG" "$run_record/"
cp "$REPO_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_a100.json" "$run_record/"
cp "$REPO_ROOT/scripts/a100/06_train.sh" "$run_record/"
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
} > "$run_record/environment.txt"

log_file="$WORKSPACE_ROOT/logs/colt_train_$run_stamp.log"
echo "Training log: $log_file"
llamafactory-cli train "$A100_CONFIG" 2>&1 | tee "$log_file"
