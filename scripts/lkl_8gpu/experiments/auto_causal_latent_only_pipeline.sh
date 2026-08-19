#!/usr/bin/env bash
# Auto-pipeline for the causal latent-only SFT experiment:
#
#   1. wait for the 300-step smoke SFT to finish
#   2. run a single-GPU ChartQA intervention gate
#   3. launch full 1500-step SFT
#   4. wait for full SFT
#   5. launch all8 evaluation
#
# Launch it once while the smoke job is already running:
#   setsid nohup bash \
#     scripts/lkl_8gpu/experiments/auto_causal_latent_only_pipeline.sh \
#     > logs/background/auto_causal_latent_only_pipeline_<ts>.log 2>&1 < /dev/null &

set -euo pipefail

COLT_ENV_FILE="${COLT_ENV_FILE:-/home/dataset-local/lkl/colt-local.env}"
CONDA_PROFILE="${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
REPO="${COLT_REPO:-/home/dataset-local/lkl/CoLT-reproduction}"

SMOKE_DIR="${COLT_CAUSAL_SMOKE_DIR:-$REPO/checkpoints/colt_paper_faithful_causal_latent_only_smoke}"
SMOKE_LOG="${COLT_CAUSAL_SMOKE_LOG:-}"
FULL_CONFIG="${COLT_CAUSAL_FULL_CONFIG:-$REPO/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_paper_faithful_causal_latent_only.yaml}"
FULL_DIR="${COLT_CAUSAL_FULL_DIR:-$REPO/checkpoints/colt_paper_faithful_causal_latent_only}"
FULL_EXPECTED_STEP="${COLT_CAUSAL_FULL_EXPECTED_STEP:-1500}"
GPU_CSV="${COLT_CAUSAL_GPUS:-0,1,2,3,4,5,6,7}"
INTERVENTION_N="${COLT_CAUSAL_INTERVENTION_N:-100}"
MIN_DROP="${COLT_CAUSAL_MIN_DROP:-5.0}"
MAX_WAIT_SECONDS="${COLT_CAUSAL_MAX_WAIT_SECONDS:-172800}"
INTERVENTION_OUT="${COLT_CAUSAL_INTERVENTION_OUT:-/home/dataset-local/lkl/tmp/causal_latent_only_smoke_intervention.json}"

source "$COLT_ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ -z "$SMOKE_LOG" ]]; then
  SMOKE_LOG="$(ls -t logs/colt_paper_faithful_train_*.log 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$SMOKE_LOG" || ! -f "$SMOKE_LOG" ]]; then
  log "ERROR: cannot locate smoke training log"
  exit 1
fi

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

latest_checkpoint() {
  find "$1" -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null \
    | sort -V | tail -n 1
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
# Stage 1: wait for the already-running smoke SFT
# ---------------------------------------------------------------------------
log "Waiting for smoke SFT to finish"
log "Smoke log: $SMOKE_LOG"
wait_for_no_train_process
require_finite_log "$SMOKE_LOG"

SMOKE_CKPT="$(latest_checkpoint "$SMOKE_DIR")"
if [[ -z "$SMOKE_CKPT" || ! -f "$SMOKE_CKPT/model.safetensors.index.json" ]]; then
  log "ERROR: smoke SFT finished without a complete checkpoint"
  exit 1
fi
log "Smoke checkpoint: $SMOKE_CKPT"

# ---------------------------------------------------------------------------
# Stage 2: causal intervention gate
# ---------------------------------------------------------------------------
log "Running ChartQA intervention gate on $SMOKE_CKPT"
if ! (
  export COLT_ANSWER_VISIBILITY=latent_only
  export COLT_LATENT_TEMPERATURE=0
  export COLT_RESPECT_GENERATION_ARGS=1
  export COLT_DECODER_MODEL_PATH=/home/dataset-local/lkl/models/Qwen3-0.6B
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH="$REPO/transformers-4.57.0/src${PYTHONPATH:+:$PYTHONPATH}"
  python scripts/lkl_8gpu/tools/mid_checkpoint_intervention.py \
    --checkpoint "$SMOKE_CKPT" \
    --tsv eval/LMUData/ChartQA_TEST.tsv \
    --n "$INTERVENTION_N" \
    --out "$INTERVENTION_OUT"
); then
  log "ERROR: intervention gate execution failed"
  exit 1
fi

if ! python scripts/lkl_8gpu/tools/check_latent_only_smoke_gate.py \
  --input "$INTERVENTION_OUT" --min-drop "$MIN_DROP"; then
  log "ERROR: smoke checkpoint did not pass the causal latent-only gate"
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 3: full SFT
# ---------------------------------------------------------------------------
FULL_LOG="$REPO/logs/background/causal_latent_only_full_$(date +%Y%m%d_%H%M%S).log"
log "Starting full SFT"
log "Full log: $FULL_LOG"
(
  export COLT_ANSWER_VISIBILITY=latent_only
  export COLT_STOCHASTIC_LATENT=1
  export COLT_LATENT_NOISE_STD=0.1
  export COLT_LATENT_TEMPERATURE=1.0
  export COLT_IMAGE_MASK_PROB=0.0
  export COLT_TRAIN_GPUS="$GPU_CSV"
  export WANDB_MODE=offline
  bash scripts/lkl_8gpu/colt.sh train paper-faithful \
    --config "$FULL_CONFIG" \
    --output-dir "$FULL_DIR"
) > "$FULL_LOG" 2>&1 < /dev/null &

log "Waiting for full SFT to finish"
sleep 10
wait_for_no_train_process
require_finite_log "$FULL_LOG"

FULL_CKPT="$(latest_checkpoint "$FULL_DIR")"
if [[ -z "$FULL_CKPT" || ! -f "$FULL_CKPT/model.safetensors.index.json" ]]; then
  log "ERROR: full SFT finished without a complete checkpoint"
  exit 1
fi
log "Full checkpoint: $FULL_CKPT"

# ---------------------------------------------------------------------------
# Stage 4: all8 evaluation
# ---------------------------------------------------------------------------
EVAL_LOG="$REPO/logs/background/causal_latent_only_all8_$(date +%Y%m%d_%H%M%S).log"
log "Starting all8 evaluation on $FULL_DIR"
log "Eval log: $EVAL_LOG"
if ! (
  export COLT_EXPECTED_GLOBAL_STEP="$FULL_EXPECTED_STEP"
  export COLT_ANSWER_VISIBILITY=latent_only
  export COLT_LATENT_TEMPERATURE=0
  COLT_LATENT_INTERVENTION=none \
  bash scripts/lkl_8gpu/colt.sh eval paper-faithful all8 \
    --model-path "$FULL_DIR" \
    --gpus "$GPU_CSV" \
    --generation respect-args \
    --empty-response-policy prevent
) > "$EVAL_LOG" 2>&1; then
  log "ERROR: all8 evaluation failed"
  exit 1
fi

log "all8 evaluation completed"
log "Full log: $FULL_LOG"
log "Eval log: $EVAL_LOG"
