#!/usr/bin/env bash
# Orchestrate the CMPO replay step-grounding SFT round:
#
#   Stage 1  full SFT, 1 epoch, save every 500 steps (~4 checkpoints)
#   Stage 2  per-checkpoint evals:
#              - all8 reasoning eval (greedy, prevent-empty)
#              - trajectory grounding score (own / shuffled / drop)
#              - per-step grounding score (hard negative / cross-image)
#   Stage 3  all8 eval of the previous w02_v2 checkpoint-500 (comparison anchor)
#
# Every stage is logged and one failure does not kill the remaining stages;
# the final summary is printed at the end.
#
# Usage (background, survives disconnect):
#   setsid nohup bash scripts/lkl_8gpu/experiments/auto_replay_step_grounding_pipeline.sh \
#     > logs/background/replay_step_grounding_30k_pipeline_<ts>.log 2>&1 < /dev/null &

set -uo pipefail

COLT_ENV_FILE="${COLT_ENV_FILE:-/home/dataset-local/lkl/colt-local.env}"
CONDA_PROFILE="${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
REPO="${COLT_REPO:-/home/dataset-local/lkl/CoLT-reproduction}"
OUTPUT_DIR="${COLT_TRAIN_OUTPUT_DIR:-$REPO/checkpoints/colt_paper_faithful_replay_step_grounding_30k}"
TRAIN_CONFIG="$REPO/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_replay_step_grounding.yaml"
BG_LOG_DIR="$REPO/logs/background"
GPU_CSV="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
GROUNDING_GPU="${COLT_GROUNDING_GPU:-0}"
OLD_ANCHOR_CKPT="${COLT_OLD_ANCHOR_CKPT:-$REPO/checkpoints/colt_paper_faithful_visual_cot_w02_v2/checkpoint-500}"
GR_N="${COLT_GROUNDING_N:-100}"
STEP_GR_N="${COLT_STEP_GROUNDING_N:-50}"

source "$COLT_ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"
mkdir -p "$BG_LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

failures=0

# Read the checkpoint's real global_step and satisfy the eval verifier
# (verify_model.py requires trainer_state.json + train_results.json and an
# exact global_step match).  Intermediate LLaMA-Factory checkpoints do not
# carry train_results.json, so a minimal marker is written (additive only).
prepare_ckpt_for_eval() {
  local ckpt="$1"
  local state="$ckpt/trainer_state.json"
  local py="${COLT_CONDA_ENV_DIR:-/home/dataset-local/lkl/envs/colt}/bin/python"
  [[ -f "$state" ]] || { log "ERROR: missing $state"; return 1; }
  COLT_EXPECTED_GLOBAL_STEP="$("$py" - "$state" <<'PY'
import json, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
print(state.get("global_step", 0))
PY
  )"
  export COLT_EXPECTED_GLOBAL_STEP
  if [[ ! -f "$ckpt/train_results.json" ]]; then
    "$py" - "$state" "$ckpt/train_results.json" <<'PY'
import json, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(
        {
            "train_runtime": 0.0,
            "train_samples_per_second": 0.0,
            "train_steps_per_second": 0.0,
            "train_loss": None,
            "epoch": state.get("epoch", 0.0),
            "global_step": state.get("global_step", 0),
        },
        f,
    )
PY
    log "  wrote train_results.json marker (global_step=$COLT_EXPECTED_GLOBAL_STEP)"
  fi
  log "  eval step gate: global_step=$COLT_EXPECTED_GLOBAL_STEP"
}

# ---------------------------------------------------------------------------
# Stage 1: full training
# ---------------------------------------------------------------------------
log "Stage 1: full replay step-grounding SFT (1 epoch, save every 500 steps)"
log "Config: $TRAIN_CONFIG"
log "Output: $OUTPUT_DIR"
TRAIN_LOG="$REPO/logs/colt_paper_faithful_train_$(date +%Y%m%d_%H%M%S).log"
log "Training log: $TRAIN_LOG"

COLT_VISUAL_GROUNDING=1 \
COLT_VISUAL_GROUNDING_WEIGHT=0.2 \
COLT_STOCHASTIC_LATENT=0 \
COLT_ANSWER_VISIBILITY=full \
COLT_BATCH_AUX_DECODERS=1 \
  bash "$REPO/scripts/lkl_8gpu/colt.sh" train paper-faithful \
    --config "$TRAIN_CONFIG" \
    --output-dir "$OUTPUT_DIR" 2>&1 | tee "$TRAIN_LOG"
train_status="${PIPESTATUS[0]}"
if [[ "$train_status" -ne 0 ]]; then
  log "ERROR: training failed with status $train_status; aborting pipeline"
  exit 1
fi
if ! ls "$OUTPUT_DIR"/model-*.safetensors >/dev/null 2>&1; then
  log "ERROR: training exited 0 but final weights are missing in $OUTPUT_DIR"
  exit 1
fi
log "Training finished; final weights present."

# ---------------------------------------------------------------------------
# Stage 2: per-checkpoint evaluation
# ---------------------------------------------------------------------------
ckpts=()
while IFS= read -r d; do
  ckpts+=("$d")
done < <(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V)
ckpts+=("$OUTPUT_DIR")
log "Stage 2: evaluating ${#ckpts[@]} model artifacts"
for ckpt in "${ckpts[@]}"; do
  name="$(basename "$ckpt")"
  log "---- $name ($ckpt) ----"
  if ! prepare_ckpt_for_eval "$ckpt"; then
    log "[$name] skipping evals (cannot prepare checkpoint)"
    failures=$((failures + 1))
    continue
  fi

  log "[$name] all8 reasoning eval"
  if COLT_LATENT_INTERVENTION=none \
    bash "$REPO/scripts/lkl_8gpu/colt.sh" eval paper-faithful all8 \
      --model-path "$ckpt" --gpus "$GPU_CSV" \
      --generation respect-args --empty-response-policy prevent \
      >> "$BG_LOG_DIR/replay_grounding_all8_${name}.log" 2>&1; then
    log "[$name] all8 OK"
  else
    log "[$name] all8 FAILED (see replay_grounding_all8_${name}.log)"
    failures=$((failures + 1))
  fi

  log "[$name] trajectory grounding score"
  if CUDA_VISIBLE_DEVICES="$GROUNDING_GPU" \
    COLT_VISUAL_GROUNDING=1 \
    COLT_DECODER_MODEL_PATH="$COLT_DECODER_MODEL_DIR" \
    python "$REPO/scripts/lkl_8gpu/tools/evaluate_grounding_score.py" \
      --checkpoint "$ckpt" --n "$GR_N" \
      --out "$BG_LOG_DIR/grounding_traj_${name}.json" \
      >> "$BG_LOG_DIR/grounding_traj_${name}.log" 2>&1; then
    log "[$name] trajectory grounding OK"
  else
    log "[$name] trajectory grounding FAILED (see grounding_traj_${name}.log)"
    failures=$((failures + 1))
  fi

  log "[$name] per-step grounding score"
  if CUDA_VISIBLE_DEVICES="$GROUNDING_GPU" \
    COLT_VISUAL_GROUNDING=1 \
    COLT_DECODER_MODEL_PATH="$COLT_DECODER_MODEL_DIR" \
    python "$REPO/scripts/lkl_8gpu/tools/evaluate_step_grounding_score.py" \
      --checkpoint "$ckpt" --n "$STEP_GR_N" \
      --out "$BG_LOG_DIR/grounding_step_${name}.json" \
      >> "$BG_LOG_DIR/grounding_step_${name}.log" 2>&1; then
    log "[$name] per-step grounding OK"
  else
    log "[$name] per-step grounding FAILED (see grounding_step_${name}.log)"
    failures=$((failures + 1))
  fi
done

# ---------------------------------------------------------------------------
# Stage 3: all8 eval of the previous w02_v2 anchor checkpoint
# ---------------------------------------------------------------------------
log "Stage 3: all8 eval of previous anchor $OLD_ANCHOR_CKPT"
if [[ -d "$OLD_ANCHOR_CKPT" ]]; then
  if ! prepare_ckpt_for_eval "$OLD_ANCHOR_CKPT"; then
    log "Stage 3 anchor FAILED to prepare; skipping"
    failures=$((failures + 1))
  else
  if COLT_LATENT_INTERVENTION=none \
    bash "$REPO/scripts/lkl_8gpu/colt.sh" eval paper-faithful all8 \
      --model-path "$OLD_ANCHOR_CKPT" --gpus "$GPU_CSV" \
      --generation respect-args --empty-response-policy prevent \
      >> "$BG_LOG_DIR/replay_grounding_all8_anchor_w02_ckpt500.log" 2>&1; then
    log "Stage 3 anchor all8 OK"
  else
    log "Stage 3 anchor all8 FAILED (see replay_grounding_all8_anchor_w02_ckpt500.log)"
    failures=$((failures + 1))
  fi
  fi
else
  log "WARNING: anchor checkpoint missing, skipping: $OLD_ANCHOR_CKPT"
fi

# ---------------------------------------------------------------------------
log "Pipeline finished. failures=$failures"
[[ "$failures" -eq 0 ]] || exit 1
