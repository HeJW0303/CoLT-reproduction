#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/scripts/lkl_8gpu/colt.sh"

# -----------------------------------------------------------------------------
# Edit this block for a new machine or experiment.
# Every value can also be overridden with the environment variable shown.
# -----------------------------------------------------------------------------
BASE_MODEL_PATH="${COLT_BASE_MODEL_DIR:-/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct}"
DECODER_MODEL_PATH="${COLT_DECODER_MODEL_DIR:-/data/nvme0/lkl/models/Qwen3-0.6B}"

TRAIN_DATASET="${COLT_TRAIN_DATASET:-onethinker_sft_image}"
TRAIN_DATASET_DIR="${COLT_DATA_ROOT:-/data/nvme0/lkl/datasets/CoLT_Train_Dataset}"
TRAIN_MEDIA_DIR="${COLT_TRAIN_MEDIA_DIR:-$TRAIN_DATASET_DIR}"

# Use all8, a built-in group, or one dataset such as ChartQA_TEST/TextVQA_VAL.
EVAL_DATASET_GROUP="${COLT_EVAL_GROUP:-all8}"
EVAL_DATA_ROOT="${COLT_EVAL_DATA_ROOT:-$REPO_ROOT/eval/LMUData}"

WORKSPACE_ROOT="${COLT_LKL_ROOT:-$(dirname "$REPO_ROOT")}"
RUNTIME_ROOT="${COLT_RUNTIME_ROOT:-$REPO_ROOT}"
TRAIN_OUTPUT_DIR="${COLT_TRAIN_OUTPUT_DIR:-$REPO_ROOT/checkpoints/colt_paper_faithful_v2}"
EVAL_MODEL_PATH="${COLT_EVAL_MODEL_PATH:-$TRAIN_OUTPUT_DIR}"
TRAIN_GPUS="${COLT_TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
EVAL_GPUS="${COLT_EVAL_GPUS:-$TRAIN_GPUS}"
EVAL_WORKERS_PER_GPU="${VLMEVAL_WORKERS_PER_GPU:-3}"

RUN_TRAIN="${COLT_RUN_TRAIN:-1}"
RUN_EVAL="${COLT_RUN_EVAL:-1}"
# -----------------------------------------------------------------------------

LOG_ROOT="${COLT_LOG_ROOT:-$RUNTIME_ROOT/logs}"
EVAL_ROOT="${COLT_EVAL_ROOT:-$RUNTIME_ROOT/eval}"
CACHE_ROOT="${COLT_CACHE_ROOT:-$RUNTIME_ROOT/cache}"
TRAIN_CONFIG_TEMPLATE="${COLT_TRAIN_CONFIG_TEMPLATE:-$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_paper_faithful.yaml}"
GENERATED_CONFIG="${COLT_TRAIN_CONFIG:-$LOG_ROOT/paper_faithful_v2/paper_faithful_v2.yaml}"
EVAL_OUTPUT_ROOT="${COLT_EVAL_OUTPUT_ROOT:-$EVAL_ROOT/results-v2}"

resume=0
dry_run=0

usage() {
  cat <<'EOF'
Train and evaluate paper-faithful CoLT v2.

Usage:
  bash scripts/run_paper_faithful_v2.sh [--resume] [--dry-run]

Edit the configuration block at the top of this file, then run it from an
activated CoLT environment. The script has no dependency on a v1 run.

Prerequisite:
  conda activate <your-prepared-environment>

Options:
  --resume   Resume training from the latest Trainer checkpoint in output_dir.
  --dry-run  Generate the training YAML and print commands without running them.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while (( $# > 0 )); do
  case "$1" in
    --resume) resume=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${CONDA_PREFIX:-}" || -n "${VIRTUAL_ENV:-}" ]] || die \
  "No Python environment is active. Run 'conda activate <your-env>' first."
PYTHON_BIN="${COLT_PYTHON:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || die \
  "The active environment does not provide python."
"$PYTHON_BIN" -c 'import yaml' >/dev/null 2>&1 || die \
  "The active environment is missing PyYAML. Activate the prepared CoLT environment."
[[ -f "$LAUNCHER" ]] || die "Missing unified launcher: $LAUNCHER"

print_command() {
  printf 'DRY RUN:'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  if (( dry_run == 1 )); then
    print_command "$@"
  else
    "$@"
  fi
}

create_train_config() {
  mkdir -p "$(dirname "$GENERATED_CONFIG")"
  "$PYTHON_BIN" - \
    "$TRAIN_CONFIG_TEMPLATE" "$GENERATED_CONFIG" \
    "$BASE_MODEL_PATH" "$TRAIN_DATASET" "$TRAIN_DATASET_DIR" "$TRAIN_MEDIA_DIR" \
    "$TRAIN_OUTPUT_DIR" "$CACHE_ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

(
    template,
    destination,
    base_model,
    dataset,
    dataset_dir,
    media_dir,
    output_dir,
    cache_root,
) = sys.argv[1:]

with open(template, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config.update(
    model_name_or_path=base_model,
    dataset=dataset,
    dataset_dir=dataset_dir,
    media_dir=media_dir,
    output_dir=output_dir,
    tokenized_path=str(Path(cache_root) / "colt/paper_faithful_v2_tokenized"),
    run_name="colt_paper_faithful_v2",
    save_steps=500,
    save_total_limit=1,
    seed=42,
    data_seed=42,
)

with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
}

read_trained_model_step() {
  "$PYTHON_BIN" - "$EVAL_MODEL_PATH/trainer_state.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    state = json.load(handle)

print(state["global_step"])
PY
}

export COLT_EXPECTED_REPO_ROOT="$REPO_ROOT"
export COLT_LKL_ROOT="$WORKSPACE_ROOT"
export COLT_RUNTIME_ROOT="$RUNTIME_ROOT"
export COLT_BASE_MODEL_DIR="$BASE_MODEL_PATH"
export COLT_DECODER_MODEL_DIR="$DECODER_MODEL_PATH"
export COLT_DATA_ROOT="$TRAIN_DATASET_DIR"
export COLT_OUTPUT_ROOT="$(dirname "$TRAIN_OUTPUT_DIR")"
export COLT_LOG_ROOT="$LOG_ROOT"
export COLT_EVAL_ROOT="$EVAL_ROOT"
export COLT_CACHE_ROOT="$CACHE_ROOT"
export COLT_TRAIN_GPUS="$TRAIN_GPUS"
export COLT_EVAL_GPUS="$EVAL_GPUS"
export COLT_EVAL_DATA_ROOT="$EVAL_DATA_ROOT"
export COLT_EVAL_OUTPUT_ROOT="$EVAL_OUTPUT_ROOT"
export COLT_EVAL_LOG_ROOT="$LOG_ROOT/eval"
export COLT_EVAL_LOG_LABEL=paper-faithful-v2
export COLT_TRAIN_RECORD_PREFIX=colt_paper_faithful_v2_run
export COLT_TRAIN_LOG_PREFIX=colt_paper_faithful_v2_train
export COLT_AUX_MAX_BATCH_TOKENS="${COLT_AUX_MAX_BATCH_TOKENS:-4096}"
export VLMEVAL_WORKERS_PER_GPU="$EVAL_WORKERS_PER_GPU"

echo "Repository: $REPO_ROOT"
echo "Base model: $BASE_MODEL_PATH"
echo "Training dataset: $TRAIN_DATASET ($TRAIN_DATASET_DIR)"
echo "Training output: $TRAIN_OUTPUT_DIR"
echo "Evaluation model: $EVAL_MODEL_PATH"
echo "Evaluation dataset/group: $EVAL_DATASET_GROUP ($EVAL_DATA_ROOT)"

if [[ "$RUN_TRAIN" == 1 ]]; then
  create_train_config
  train_args=(
    bash "$LAUNCHER" train paper-faithful
    --config "$GENERATED_CONFIG"
    --output-dir "$TRAIN_OUTPUT_DIR"
    --batch-aux
  )
  (( resume == 0 )) || train_args+=(--resume)
  run_command "${train_args[@]}"
fi

if [[ "$RUN_EVAL" == 1 ]]; then
  if (( dry_run == 0 )) && [[ -z "${COLT_EXPECTED_GLOBAL_STEP:-}" ]]; then
    export COLT_EXPECTED_GLOBAL_STEP="$(read_trained_model_step)"
  fi
  run_command bash "$LAUNCHER" eval paper-faithful "$EVAL_DATASET_GROUP" \
    --model-path "$EVAL_MODEL_PATH" \
    --gpus "$EVAL_GPUS" \
    --workers "$EVAL_WORKERS_PER_GPU" \
    --generation respect-args \
    --prefetch 1 \
    --empty-cache-every 0 \
    --reseed-per-sample 1 \
    --empty-response-policy prevent
fi

if (( dry_run == 1 )); then
  echo "Dry run complete; training and evaluation were not started."
else
  echo "paper-faithful v2 pipeline complete."
fi
