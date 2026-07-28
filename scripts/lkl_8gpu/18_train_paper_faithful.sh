#!/usr/bin/env bash

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_ROOT/../.." && pwd)"

export COLT_PAPER_FAITHFUL=1
export COLT_BATCH_AUX_DECODERS="${COLT_BATCH_AUX_DECODERS:-0}"
export COLT_COMPONENT_LOG_EVERY="${COLT_COMPONENT_LOG_EVERY:-8}"
export COLT_TRAIN_CONFIG="$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_paper_faithful.yaml"
export COLT_TRAIN_OUTPUT_DIR="${COLT_OUTPUT_ROOT:-$REPO_ROOT/checkpoints}/colt_paper_faithful"
export COLT_TRAIN_ENTRY_SCRIPT="$SCRIPT_ROOT/18_train_paper_faithful.sh"
export COLT_TRAIN_RECORD_PREFIX=colt_paper_faithful_run
export COLT_TRAIN_LOG_PREFIX=colt_paper_faithful_train

exec bash "$SCRIPT_ROOT/06_train.sh"
