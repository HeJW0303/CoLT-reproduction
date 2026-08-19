#!/usr/bin/env bash
# Wait for the stochastic+image-mask SFT to finish, then run the full
# evaluation suite: all8 standard eval, then intervention evals
# (none/zero/random/skip x chart-text+mmstar), then MathVista via DeepSeek
# when OPENAI_API_KEY is set.
#
# Usage (background, survives disconnect):
#   OPENAI_API_KEY=sk-... setsid nohup bash \
#     scripts/lkl_8gpu/experiments/auto_eval_after_train.sh \
#     > /home/dataset-local/lkl/logs/auto_eval_after_train_<ts>.log 2>&1 < /dev/null &

set -euo pipefail

COLT_ENV_FILE="${COLT_ENV_FILE:-/home/dataset-local/lkl/colt-local.env}"
CONDA_PROFILE="${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
REPO="${COLT_REPO:-/home/dataset-local/lkl/CoLT-reproduction}"
MODEL_DIR="${COLT_EVAL_MODEL_DIR:-$REPO/checkpoints/colt_paper_faithful_stochastic}"
GPU_CSV="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_LOG="$(cat /home/dataset-local/lkl/tmp/imagemask_log_path 2>/dev/null || echo /home/dataset-local/lkl/logs/imagemask_sft_resume_*.log)"
MAX_WAIT_SECONDS="${COLT_EVAL_MAX_WAIT_SECONDS:-86400}"  # 24h cap

source "$COLT_ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Stage 0: wait for training to finish
# ---------------------------------------------------------------------------
log "Waiting for SFT training to finish (model=$MODEL_DIR)"
log "Training log: $TRAIN_LOG"
started=$(date +%s)
while true; do
  now=$(date +%s)
  if (( now - started > MAX_WAIT_SECONDS )); then
    log "ERROR: timed out waiting for training after $MAX_WAIT_SECONDS s"
    exit 1
  fi
  if ! pgrep -f 'llamafactory-cli train' >/dev/null; then
    if ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | grep -q .; then
      log "Training finished; final weights present."
      break
    fi
    log "ERROR: training process exited but no final weights in $MODEL_DIR"
    log "Last training log lines:"
    tail -5 "$TRAIN_LOG" 2>/dev/null || true
    exit 1
  fi
  sleep 60
done

# ---------------------------------------------------------------------------
# Stage 1: all8 standard evaluation (greedy, prevent-empty, no intervention)
# ---------------------------------------------------------------------------
export COLT_EXPECTED_GLOBAL_STEP="${COLT_EXPECTED_GLOBAL_STEP:-1389}"
log "Stage 1: all8 standard evaluation"
if COLT_LATENT_INTERVENTION=none \
  bash scripts/lkl_8gpu/colt.sh eval paper-faithful all8 \
    --model-path "$MODEL_DIR" --gpus "$GPU_CSV" \
    --generation respect-args --empty-response-policy prevent \
    >> /home/dataset-local/lkl/logs/auto_eval_stage1_all8.log 2>&1; then
  log "Stage 1 all8 OK"
else
  log "Stage 1 all8 FAILED (see auto_eval_stage1_all8.log)"
fi

# ---------------------------------------------------------------------------
# Stage 2: intervention evals (none/zero/random/skip x chart-text+mmstar)
# ---------------------------------------------------------------------------
log "Stage 2: intervention evals (4 modes x chart-text+mmstar)"
for mode in none zero random skip; do
  for group in chart-text mmstar; do
    if COLT_LATENT_INTERVENTION="$mode" \
      bash scripts/lkl_8gpu/colt.sh eval paper-faithful "$group" \
        --model-path "$MODEL_DIR" --gpus "$GPU_CSV" \
        --generation respect-args --empty-response-policy prevent \
        >> "/home/dataset-local/lkl/logs/auto_eval_interv_${mode}_${group}.log" 2>&1; then
      log "intervention $mode/$group OK"
    else
      log "intervention $mode/$group FAILED"
    fi
  done
done

# ---------------------------------------------------------------------------
# Stage 3 (optional): MathVista intervention via DeepSeek judge
# ---------------------------------------------------------------------------
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  log "Stage 3: MathVista_MINI intervention via DeepSeek judge"
  export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.deepseek.com/v1/chat/completions}"
  export COLT_MATHVISTA_JUDGE_MAX_TOKENS="${COLT_MATHVISTA_JUDGE_MAX_TOKENS:-1024}"
  # DeepSeek is a domestic service; force direct connections for the judge.
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
  for mode in none zero random skip; do
    if COLT_EVAL_JUDGE=deepseek-v4-flash \
      COLT_LATENT_INTERVENTION="$mode" \
      bash scripts/lkl_8gpu/colt.sh eval paper-faithful MathVista_MINI \
        --model-path "$MODEL_DIR" --gpus "$GPU_CSV" \
        --generation respect-args --empty-response-policy prevent \
        >> "/home/dataset-local/lkl/logs/auto_eval_mathvista_${mode}.log" 2>&1; then
      log "MathVista $mode OK"
    else
      log "MathVista $mode FAILED"
    fi
  done
else
  log "Stage 3 skipped: OPENAI_API_KEY not set (no MathVista DeepSeek judge)"
fi

log "All evaluation stages completed."
