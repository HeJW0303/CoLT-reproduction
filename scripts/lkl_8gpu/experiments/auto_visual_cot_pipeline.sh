#!/usr/bin/env bash
# Auto-pipeline for the CMPO Visual-CoT SFT:
#
#   1. wait for the running smoke SFT to finish
#   2. verify the smoke log has no fatal error
#   3. launch the full 1500-step SFT with visual grounding enabled
#
# Launch once while the smoke job is already running:
#   setsid nohup bash \
#     scripts/lkl_8gpu/experiments/auto_visual_cot_pipeline.sh \
#     > logs/background/auto_visual_cot_pipeline_<ts>.log 2>&1 < /dev/null &

set -euo pipefail

COLT_ENV_FILE="${COLT_ENV_FILE:-/home/dataset-local/lkl/colt-local.env}"
CONDA_PROFILE="${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
REPO="${COLT_REPO:-/home/dataset-local/lkl/CoLT-reproduction}"

SMOKE_LOG="${COLT_VISUAL_COT_SMOKE_LOG:-}"
FULL_CONFIG="${COLT_VISUAL_COT_FULL_CONFIG:-$REPO/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_visual_cot_98k.yaml}"
FULL_DIR="${COLT_VISUAL_COT_FULL_DIR:-$REPO/checkpoints/colt_paper_faithful_visual_cot}"
FULL_EXPECTED_STEP="${COLT_VISUAL_COT_FULL_EXPECTED_STEP:-1500}"
GPU_CSV="${COLT_VISUAL_COT_GPUS:-0,1,2,3,4,5,6,7}"
MAX_WAIT_SECONDS="${COLT_VISUAL_COT_MAX_WAIT_SECONDS:-172800}"

source "$COLT_ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ -z "$SMOKE_LOG" ]]; then
  SMOKE_LOG="$(ls -t logs/colt_paper_faithful_train_*.log 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$SMOKE_LOG" || ! -f "$SMOKE_LOG" ]]; then
  log "ERROR: cannot locate the smoke training log"
  exit 1
fi
log "Smoke log: $SMOKE_LOG"

wait_for_no_train_process() {
  local started now
  started=$(date +%s)
  while pgrep -f 'llamafactory-cli train' >/dev/null; do
    now=$(date +%s)
    if (( now - started > MAX_WAIT_SECONDS )); then
      log "ERROR: timed out after $MAX_WAIT_SECONDS seconds"
      exit 1
    fi
    sleep 60
  done
}

require_finite_log() {
  local path="$1"
  if grep -Eiq 'Traceback|RuntimeError|CUDA out of memory|torch\.OutOfMemoryError|nan' "$path"; then
    log "ERROR: training log contains a fatal error: $path"
    grep -niE 'Traceback|RuntimeError|CUDA out of memory|torch\.OutOfMemoryError|nan' "$path" | tail -20 || true
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Stage 1: wait for the running smoke SFT
# ---------------------------------------------------------------------------
log "Waiting for the smoke SFT to finish"
wait_for_no_train_process
require_finite_log "$SMOKE_LOG"
if ! grep -q 'Training completed' "$SMOKE_LOG"; then
  log "ERROR: smoke log does not contain 'Training completed'"
  exit 1
fi
log "Smoke SFT finished cleanly"

# ---------------------------------------------------------------------------
# Stage 2: launch the full SFT
# ---------------------------------------------------------------------------
FULL_LOG="$REPO/logs/background/visual_cot_98k_full_$(date +%Y%m%d_%H%M%S).log"
log "Starting full SFT"
log "Full log: $FULL_LOG"
(
  export COLT_VISUAL_GROUNDING=1
  export COLT_VISUAL_GROUNDING_WEIGHT=0.1
  export COLT_STOCHASTIC_LATENT=1
  export COLT_LATENT_NOISE_STD=0.1
  export COLT_LATENT_TEMPERATURE=1.0
  export COLT_IMAGE_MASK_PROB=0.0
  export COLT_ANSWER_VISIBILITY=full
  export COLT_TRAIN_GPUS="$GPU_CSV"
  export WANDB_MODE=offline
  bash scripts/lkl_8gpu/colt.sh train paper-faithful \
    --config "$FULL_CONFIG" \
    --output-dir "$FULL_DIR"
) > "$FULL_LOG" 2>&1 < /dev/null &

log "Waiting for the full SFT to finish"
wait_for_no_train_process
require_finite_log "$FULL_LOG"
if ! grep -q 'Training completed' "$FULL_LOG"; then
  log "ERROR: full SFT log does not contain 'Training completed'"
  exit 1
fi

log "Full SFT completed"
log "Full log: $FULL_LOG"
