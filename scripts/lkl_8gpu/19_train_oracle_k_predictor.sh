#!/usr/bin/env bash

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_ROOT/../.." && pwd)"

# This is the LKL 8-GPU entry point. Keep the legacy scripts/a100 entry point
# available for old /workspace runs, but use the repository's A100/A800 layout
# and paper-faithful hot-path fixes here.
export COLT_PAPER_FAITHFUL=1
export COLT_BATCH_AUX_DECODERS="${COLT_BATCH_AUX_DECODERS:-0}"
export COLT_COMPONENT_LOG_EVERY="${COLT_COMPONENT_LOG_EVERY:-8}"
export COLT_ORACLE_K_ENABLED=1
export COLT_ORACLE_K_MAX="${COLT_ORACLE_K_MAX:-8}"
export COLT_ORACLE_K_BUDGET_CONDITIONING="${COLT_ORACLE_K_BUDGET_CONDITIONING:-1}"
export COLT_ORACLE_K_PREDICTOR_ENABLED="${COLT_ORACLE_K_PREDICTOR_ENABLED:-1}"
export COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT="${COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT:-0.2}"
export COLT_ORACLE_K_DYNAMIC_INFERENCE="${COLT_ORACLE_K_DYNAMIC_INFERENCE:-1}"
export COLT_ORACLE_K_DATASET_NAME="${COLT_ORACLE_K_DATASET_NAME:-onethinker_sft_image_oracle_k}"
export COLT_ORACLE_K_DATA_FILE="${COLT_ORACLE_K_DATA_FILE:-${COLT_DATA_ROOT:-/data/nvme0/lkl/datasets/CoLT_Train_Dataset}/colt_sft_image_oracle_k.json}"
export COLT_ORACLE_K_TOKENIZED_PATH="${COLT_ORACLE_K_TOKENIZED_PATH:-${COLT_CACHE_ROOT:-$REPO_ROOT/cache}/colt/onethinker_sft_oracle_k_predictor_tokenized}"
export COLT_ORACLE_K_OUTPUT_DIR="${COLT_ORACLE_K_OUTPUT_DIR:-${COLT_OUTPUT_ROOT:-$REPO_ROOT/checkpoints}/colt_oracle_k_predictor}"
export COLT_ORACLE_K_RUN_NAME="${COLT_ORACLE_K_RUN_NAME:-colt_sft_8b_lkl_8gpu_oracle_k_predictor}"

export COLT_TRAIN_CONFIG="${COLT_TRAIN_CONFIG:-$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_oracle_k_predictor.yaml}"
export COLT_TRAIN_OUTPUT_DIR="${COLT_TRAIN_OUTPUT_DIR:-$COLT_ORACLE_K_OUTPUT_DIR}"
export COLT_TRAIN_ENTRY_SCRIPT="$SCRIPT_ROOT/19_train_oracle_k_predictor.sh"
export COLT_TRAIN_RECORD_PREFIX="${COLT_TRAIN_RECORD_PREFIX:-colt_oracle_k_predictor_run}"
export COLT_TRAIN_LOG_PREFIX="${COLT_TRAIN_LOG_PREFIX:-colt_oracle_k_predictor_train}"

exec bash "$SCRIPT_ROOT/06_train.sh"
