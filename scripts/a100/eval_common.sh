#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env

VLMEVAL_ROOT="$REPO_ROOT/Evaluation/VLMEvalKit"
FINAL_MODEL_DIR="${COLT_FINAL_MODEL_DIR:-$OUTPUT_ROOT/colt_codefaithful}"
EVAL_ROOT="${COLT_EVAL_ROOT:-$WORKSPACE_ROOT/eval}"
EVAL_DATA_ROOT="${COLT_EVAL_DATA_ROOT:-$EVAL_ROOT/LMUData}"
EVAL_OUTPUT_ROOT="${COLT_EVAL_OUTPUT_ROOT:-$EVAL_ROOT/results}"
EVAL_LOG_ROOT="${COLT_EVAL_LOG_ROOT:-$WORKSPACE_ROOT/logs/eval}"
EVAL_MODEL_NAME="Qwen3-VL-8B-Instruct-COLT"

export COLT_EVAL_MODEL_PATH="$FINAL_MODEL_DIR"
export COLT_DECODER_MODEL_PATH="$DECODER_MODEL_DIR"
export COLT_EVAL_SEED="${COLT_EVAL_SEED:-1234}"
export LMUData="$EVAL_DATA_ROOT"
export PYTHONPATH="$VLMEVAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLMEVAL_FAIL_ON_ERROR=1
export VLMEVAL_ATOMIC_WRITES=1

mkdir -p "$EVAL_DATA_ROOT" "$EVAL_OUTPUT_ROOT" "$EVAL_LOG_ROOT"

root_free_kb="$(df -Pk / | awk 'NR == 2 {print $4}')"
if (( root_free_kb < 2 * 1024 * 1024 )); then
  if [[ "${COLT_ALLOW_FULL_ROOT:-0}" != "1" ]]; then
    echo "Less than 2 GiB is free on the container root filesystem; refusing to evaluate." >&2
    echo "All known evaluation writes are routed away from root, but continuing at zero free space" >&2
    echo "still requires the explicit override COLT_ALLOW_FULL_ROOT=1." >&2
    df -h / >&2
    exit 1
  fi
  for routed_path in \
      "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" \
      "$CUDA_CACHE_PATH" "$TMPDIR" "$EVAL_ROOT" "$EVAL_LOG_ROOT"; do
    if [[ "$routed_path" != "$WORKSPACE_ROOT"/* ]]; then
      echo "Refusing full-root override: path is not under $WORKSPACE_ROOT: $routed_path" >&2
      exit 1
    fi
  done
  if [[ "$TRITON_CACHE_DIR" != /dev/shm/* && "$TRITON_CACHE_DIR" != "$WORKSPACE_ROOT"/* ]]; then
    echo "Refusing full-root override: unsafe TRITON_CACHE_DIR=$TRITON_CACHE_DIR" >&2
    exit 1
  fi
  echo "WARNING: container root has less than 2 GiB free; explicit override accepted." >&2
  echo "Known caches, temporary files, datasets, logs, and results are routed to /workspace or /dev/shm." >&2
  df -h / "$WORKSPACE_ROOT" >&2
elif (( root_free_kb < 5 * 1024 * 1024 )); then
  echo "Warning: less than 5 GiB is free on the container root filesystem." >&2
  df -h / >&2
fi

if [[ "${COLT_REQUIRE_FINAL_MODEL:-1}" == "1" && ! -f "$FINAL_MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "Missing final model index: $FINAL_MODEL_DIR/model.safetensors.index.json" >&2
  exit 1
fi
