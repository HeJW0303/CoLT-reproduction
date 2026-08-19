#!/usr/bin/env bash
# Train the paper-faithful-v2 data-only comparator, then evaluate its final
# model on all8.  This starts from the same Qwen3-VL base and uses the same
# optimization hyperparameters as fix-v2; only the SFT corpus is replaced by
# the strict 3-step Visual-CoT core plus full-supervision GQA Visual-CoT 30K.

set -euo pipefail

COLT_ENV_FILE="${COLT_ENV_FILE:-/home/dataset-local/lkl/colt-local.env}"
CONDA_PROFILE="${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
REPO="${COLT_REPO:-/home/dataset-local/lkl/CoLT-reproduction}"
DATA_ROOT="${COLT_DATA_ROOT:-/home/dataset-local/lkl/datasets/CoLT_Train_Dataset}"
CONFIG="${COLT_TRAIN_CONFIG:-$REPO/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_paper_faithful_v2_visualcot3step87k_gqa30k.yaml}"
OUTPUT_DIR="${COLT_TRAIN_OUTPUT_DIR:-$REPO/checkpoints/colt_paper_faithful_v2_visualcot3step87k_gqa30k}"
GPU_CSV="${COLT_TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
EVAL_GPU_CSV="${COLT_EVAL_GPUS:-$GPU_CSV}"

source "$COLT_ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"

RUN_TMPDIR="${COLT_RUN_TMPDIR:-/home/dataset-local/lkl/tmp/colt_paper_faithful_v2_visualcot3step_gqa}"
mkdir -p "$RUN_TMPDIR"
export TMPDIR="$RUN_TMPDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

[[ -f "$CONFIG" ]] || { log "ERROR: missing config: $CONFIG"; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { log "ERROR: output already exists: $OUTPUT_DIR"; exit 1; }

log "Preflight: checking exact data-only comparator contract"
python - "$CONFIG" "$DATA_ROOT" <<'PY'
import json
import os
import sys

import yaml

config_path, data_root = sys.argv[1:]
config = yaml.safe_load(open(config_path, encoding="utf-8"))
expected_datasets = [
    "onethinker_sft_visual_cot_3step_core",
    "onethinker_sft_gqa_visual_cot_30k_all",
]
if config["model_name_or_path"] != "/home/dataset-local/lkl/models/Qwen3-VL-8B-Instruct":
    raise RuntimeError("data-only comparator must start from the Qwen3-VL base, not a trained checkpoint")
if config["dataset"].split(",") != expected_datasets:
    raise RuntimeError(f"unexpected dataset order: {config['dataset']}")
if config.get("mix_strategy", "concat") != "concat":
    raise RuntimeError("data-only comparator must concatenate each sample exactly once")
if config["num_train_epochs"] != 1.0 or config["seed"] != 42 or config["data_seed"] != 42:
    raise RuntimeError("fix-v2 epoch/seed contract changed")

expectations = [
    ("colt_sft_visual_cot_3step_core.json", 87283),
    ("colt_sft_gqa_visual_cot_30k_all.json", 30000),
]
for filename, expected_count in expectations:
    rows = json.load(open(os.path.join(data_root, filename), encoding="utf-8"))
    if len(rows) != expected_count:
        raise RuntimeError(f"{filename}: expected {expected_count}, found {len(rows)}")
    missing_images = 0
    malformed_steps = 0
    for row in rows:
        images = row.get("images") or []
        if len(images) != 1 or not os.path.isfile(images[0]):
            missing_images += 1
        steps = row.get("step_bboxes") or []
        if len(steps) != 3 or any(not step for step in steps):
            malformed_steps += 1
    if missing_images or malformed_steps:
        raise RuntimeError(
            f"{filename}: missing_images={missing_images}, malformed_step_bboxes={malformed_steps}"
        )
    print(f"validated {filename}: {len(rows)} rows", flush=True)
print("validated total rows: 117283; expected optimizer steps: ceil(117283 / 64) = 1833", flush=True)
PY

# Keep this a data-only comparator.  Bboxes and visual_cot metadata remain in
# the records for auditability, but no new grounding/masking/stochastic loss is
# enabled relative to paper-faithful v2.
export COLT_VISUAL_GROUNDING=0
export COLT_STOCHASTIC_LATENT=0
export COLT_IMAGE_MASK_PROB=0
export COLT_ANSWER_VISIBILITY=full
export COLT_BATCH_AUX_DECODERS=1
export COLT_AUX_MAX_BATCH_TOKENS=4096
export COLT_COMPONENT_LOG_EVERY=8
export COLT_TRAIN_GPUS="$GPU_CSV"
export COLT_EVAL_GPUS="$EVAL_GPU_CSV"
export COLT_INFERENCE_LATENT_TRANSITION=training-consistent
export COLT_LATENT_INTERVENTION=none
export COLT_EVAL_MAX_NEW_TOKENS=8192
export VLMEVAL_WORKERS_PER_GPU=3
export VLMEVAL_PREFETCH=1
export VLMEVAL_EMPTY_CACHE_EVERY_N=0
export COLT_RESEED_PER_SAMPLE=1
export COLT_EVAL_LOG_LABEL=paper-faithful-v2-visualcot3step-gqa
export COLT_TRAIN_RECORD_PREFIX=colt_paper_faithful_v2_visualcot3step_gqa_run
export COLT_TRAIN_LOG_PREFIX=colt_paper_faithful_v2_visualcot3step_gqa_train

log "Stage 1: paper-faithful-v2 training from Qwen3-VL base"
log "Config: $CONFIG"
log "Output: $OUTPUT_DIR"
bash "$REPO/scripts/lkl_8gpu/colt.sh" train paper-faithful \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --batch-aux

[[ -f "$OUTPUT_DIR/trainer_state.json" ]] || { log "ERROR: final trainer_state.json missing"; exit 1; }
if ! compgen -G "$OUTPUT_DIR/model-*.safetensors" > /dev/null; then
    log "ERROR: final model shards missing"
    exit 1
fi
COLT_EXPECTED_GLOBAL_STEP="$(python - "$OUTPUT_DIR/trainer_state.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["global_step"])
PY
)"
export COLT_EXPECTED_GLOBAL_STEP
log "Training completed at global_step=$COLT_EXPECTED_GLOBAL_STEP"

log "Stage 2: all8 evaluation with training-consistent transition"
bash "$REPO/scripts/lkl_8gpu/colt.sh" eval paper-faithful all8 \
    --model-path "$OUTPUT_DIR" \
    --gpus "$EVAL_GPU_CSV" \
    --workers 3 \
    --generation respect-args \
    --latent-transition training-consistent \
    --prefetch 1 \
    --empty-cache-every 0 \
    --reseed-per-sample 1 \
    --empty-response-policy prevent

log "Pipeline completed: training + all8 evaluation"
