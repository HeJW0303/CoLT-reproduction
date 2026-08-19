#!/usr/bin/env bash
# Wait for the replay step-grounding pipeline's evaluation stages to finish,
# then run on checkpoint-1986:
#   1) reward-discriminability gate (training-recurrence forward, 8-GPU sharded)
#   2) grounding evals re-run under the TRAINING-CONSISTENT latent transition
#   3) all8 re-run under training-consistent transition (consistency check)
#
# Usage (background, survives disconnect):
#   setsid nohup bash scripts/lkl_8gpu/experiments/auto_run_reward_gate.sh \
#     > logs/background/reward_gate_<ts>.log 2>&1 < /dev/null &

set -uo pipefail

COLT_ENV_FILE="${COLT_ENV_FILE:-/home/dataset-local/lkl/colt-local.env}"
CONDA_PROFILE="${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
REPO="${COLT_REPO:-/home/dataset-local/lkl/CoLT-reproduction}"
CKPT="${COLT_GATE_CKPT:-$REPO/checkpoints/colt_paper_faithful_replay_step_grounding_30k/checkpoint-1986}"
OUT="${COLT_GATE_OUT:-/home/dataset-local/lkl/tmp/gate_ckpt1986.json}"
N_SHARDS="${COLT_GATE_SHARDS:-8}"
GATE_N_ANSWER="${COLT_GATE_N_ANSWER:-60}"
GATE_N_GROUND="${COLT_GATE_N_GROUND:-60}"
GATE_K="${COLT_GATE_K:-6}"

source "$COLT_ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Waiting for replay pipeline (train+eval) to finish before running the gate..."
while pgrep -f 'auto_replay_step_grounding_pipeline' >/dev/null; do
  sleep 60
done
sleep 30
log "Pipeline finished; starting reward gate on $CKPT"

if [[ "${COLT_GATE_SKIP_GATE:-0}" != "1" ]]; then
  pids=()
  for i in $(seq 0 $((N_SHARDS - 1))); do
    shard_out="${OUT%.json}_shard${i}.json"
    COLT_VISUAL_GROUNDING=1 \
    COLT_STOCHASTIC_LATENT=1 \
    COLT_COMPONENT_LOG_EVERY=1 \
    COLT_BATCH_AUX_DECODERS=1 \
    COLT_DECODER_MODEL_PATH="$COLT_DECODER_MODEL_DIR" \
    CUDA_VISIBLE_DEVICES="$i" \
      python "$REPO/scripts/lkl_8gpu/tools/gate_reward_discriminability.py" \
        --checkpoint "$CKPT" --out "$shard_out" \
        --n-answer "$GATE_N_ANSWER" --n-ground "$GATE_N_GROUND" --k "$GATE_K" \
        --shard-id "$i" --n-shards "$N_SHARDS" \
        >> "$REPO/logs/background/reward_gate_shard${i}.log" 2>&1 &
    pids+=("$!")
    log "launched gate shard $i (pid ${pids[-1]})"
  done

  gate_fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || gate_fail=1
  done
  if [[ "$gate_fail" -ne 0 ]]; then
    log "At least one gate shard failed; skipping training-consistent re-evals"
    exit 1
  fi

  shard_inputs=()
  for i in $(seq 0 $((N_SHARDS - 1))); do
    shard_inputs+=("${OUT%.json}_shard${i}.json")
  done
  log "Merging gate shards..."
  python "$REPO/scripts/lkl_8gpu/tools/merge_gate_shards.py" \
    --inputs "${shard_inputs[@]}" --out "$OUT"
  log "Gate merged result: $OUT"
else
  log "Skipping gate (COLT_GATE_SKIP_GATE=1), using existing $OUT"
fi

# ---------------------------------------------------------------------------
# Training-consistent grounding re-eval (the previous numbers used 'official')
# ---------------------------------------------------------------------------
log "Re-running grounding evals with COLT_INFERENCE_LATENT_TRANSITION=training-consistent"
export COLT_INFERENCE_LATENT_TRANSITION=training-consistent
if CUDA_VISIBLE_DEVICES=0 \
  COLT_VISUAL_GROUNDING=1 \
  COLT_DECODER_MODEL_PATH="$COLT_DECODER_MODEL_DIR" \
  python "$REPO/scripts/lkl_8gpu/tools/evaluate_grounding_score.py" \
    --checkpoint "$CKPT" --n 100 \
    --out "$REPO/logs/background/grounding_traj_ckpt1986_training_consistent.json" \
    >> "$REPO/logs/background/grounding_traj_ckpt1986_training_consistent.log" 2>&1; then
  log "training-consistent trajectory grounding OK"
else
  log "training-consistent trajectory grounding FAILED"
fi

if CUDA_VISIBLE_DEVICES=0 \
  COLT_VISUAL_GROUNDING=1 \
  COLT_DECODER_MODEL_PATH="$COLT_DECODER_MODEL_DIR" \
  python "$REPO/scripts/lkl_8gpu/tools/evaluate_step_grounding_score.py" \
    --checkpoint "$CKPT" --n 50 \
    --out "$REPO/logs/background/grounding_step_ckpt1986_training_consistent.json" \
    >> "$REPO/logs/background/grounding_step_ckpt1986_training_consistent.log" 2>&1; then
  log "training-consistent step grounding OK"
else
  log "training-consistent step grounding FAILED"
fi

# ---------------------------------------------------------------------------
# Training-consistent all8 re-eval (optional, ~35 min)
# ---------------------------------------------------------------------------
if [[ "${COLT_GATE_SKIP_ALL8:-0}" != "1" ]]; then
  log "Re-running all8 with training-consistent transition"
  EXPECTED_STEP="$(python - "$CKPT/trainer_state.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("global_step", 0))
PY
  )"
  export COLT_EXPECTED_GLOBAL_STEP="$EXPECTED_STEP"
  log "eval step gate: global_step=$COLT_EXPECTED_GLOBAL_STEP"
  if COLT_LATENT_INTERVENTION=none \
    bash "$REPO/scripts/lkl_8gpu/colt.sh" eval paper-faithful all8 \
      --model-path "$CKPT" --gpus 0,1,2,3,4,5,6,7 \
      --generation respect-args --empty-response-policy prevent \
      --latent-transition training-consistent \
      >> "$REPO/logs/background/replay_grounding_all8_ckpt1986_training_consistent.log" 2>&1; then
    log "training-consistent all8 OK"
  else
    log "training-consistent all8 FAILED"
  fi
else
  log "Skipping training-consistent all8 (COLT_GATE_SKIP_ALL8=1)"
fi

log "Gate + consistency re-evals finished."
