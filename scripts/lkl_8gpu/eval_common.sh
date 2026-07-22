#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env
validate_gpu_profile

VLMEVAL_ROOT="$REPO_ROOT/Evaluation/VLMEvalKit"
FINAL_MODEL_DIR="${COLT_FINAL_MODEL_DIR:-$OUTPUT_ROOT/colt_codefaithful}"
EVAL_ROOT="${COLT_EVAL_ROOT:-$EVAL_ROOT_DEFAULT}"
EVAL_DATA_ROOT="${COLT_EVAL_DATA_ROOT:-$EVAL_ROOT/LMUData}"
EVAL_OUTPUT_ROOT="${COLT_EVAL_OUTPUT_ROOT:-$EVAL_ROOT/results}"
EVAL_LOG_ROOT="${COLT_EVAL_LOG_ROOT:-$LOG_ROOT/eval}"
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

if [[ "${COLT_REQUIRE_FINAL_MODEL:-1}" == "1" && ! -f "$FINAL_MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "Missing final model index: $FINAL_MODEL_DIR/model.safetensors.index.json" >&2
  exit 1
fi
