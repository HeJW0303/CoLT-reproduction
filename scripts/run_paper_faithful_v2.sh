#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/scripts/lkl_8gpu/colt.sh"

RUNTIME_ROOT="${COLT_RUNTIME_ROOT:-$REPO_ROOT}"
CHECKPOINT_ROOT="${COLT_OUTPUT_ROOT:-$RUNTIME_ROOT/checkpoints}"
LOG_ROOT="${COLT_LOG_ROOT:-$RUNTIME_ROOT/logs}"
EVAL_ROOT="${COLT_EVAL_ROOT:-$RUNTIME_ROOT/eval}"
CACHE_ROOT="${COLT_CACHE_ROOT:-$RUNTIME_ROOT/cache}"
GPU_CSV="${COLT_TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${VLMEVAL_WORKERS_PER_GPU:-3}"
EXPECTED_STEPS="${COLT_EXPECTED_GLOBAL_STEP:-1910}"

V1_CHECKPOINT="$CHECKPOINT_ROOT/colt_paper_faithful_v1"
LEGACY_CHECKPOINT="$CHECKPOINT_ROOT/colt_paper_faithful"
V2_CHECKPOINT="$CHECKPOINT_ROOT/colt_paper_faithful_v2"
V1_EVAL_LOG_DIR="$LOG_ROOT/eval/paper-faithful-v1"
V2_EVAL_LOG_DIR="$LOG_ROOT/eval/paper-faithful-v2"
V1_EVAL_OUTPUT_ROOT="$EVAL_ROOT/results-v1"
V2_EVAL_OUTPUT_ROOT="$EVAL_ROOT/results-v2"
PIPELINE_ROOT="$LOG_ROOT/paper_faithful_v2_pipeline"
V2_CONFIG="$PIPELINE_ROOT/paper_faithful_v2.yaml"

resume=0
dry_run=0

usage() {
  cat <<'EOF'
Train and evaluate the current B1-only paper-faithful CoLT model.

Usage:
  bash scripts/run_paper_faithful_v2.sh [--resume] [--dry-run]

The first run archives the legacy paper-faithful checkpoint/logs/results as v1,
trains v2 on eight GPUs with auxiliary decoder batching, verifies the final
checkpoint, then evaluates all eight datasets with greedy + 8192,
prevent-empty, and three workers per GPU.

Options:
  --resume   Resume v2 only when a complete Trainer checkpoint-* exists.
  --dry-run  Archive nothing and print the train/eval commands.
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
    *) die "Unknown option: $1" ;;
  esac
done

[[ -n "${CONDA_PREFIX:-}" || -n "${VIRTUAL_ENV:-}" ]] || die \
  "No Python environment is active. Run 'conda activate <your-env>' first."
PYTHON_BIN="${COLT_PYTHON:-$(command -v python || true)}"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || die "The active environment does not provide python."
"$PYTHON_BIN" -c 'import yaml' >/dev/null 2>&1 || die "The active environment is missing PyYAML."
[[ -x "$LAUNCHER" ]] || die "Missing executable launcher: $LAUNCHER"
[[ "$EXPECTED_STEPS" =~ ^[1-9][0-9]*$ ]] || die "COLT_EXPECTED_GLOBAL_STEP must be positive."
[[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "VLMEVAL_WORKERS_PER_GPU must be positive."

IFS=',' read -r -a gpu_ids <<< "$GPU_CSV"
[[ "${#gpu_ids[@]}" -eq 8 ]] || die "Exactly eight training GPUs are required: $GPU_CSV"

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

archive_path() {
  local source="$1" destination="$2"
  if [[ -e "$source" && -e "$destination" ]]; then
    die "Both v1 source and destination exist: $source and $destination"
  fi
  if [[ -e "$source" ]]; then
    echo "Archiving v1: $source -> $destination"
    run_command mkdir -p "$(dirname "$destination")"
    run_command mv -- "$source" "$destination"
  elif [[ -e "$destination" ]]; then
    echo "v1 already archived: $destination"
  else
    echo "No legacy artifact to archive: $source"
  fi
}

archive_training_logs() {
  local source destination nested_log nested_destination suffix
  shopt -s nullglob
  for source in "$LOG_ROOT"/colt_paper_faithful_run_*; do
    suffix="${source##*/colt_paper_faithful_run_}"
    destination="$LOG_ROOT/colt_paper_faithful_v1_run_$suffix"
    archive_path "$source" "$destination"
    if (( dry_run == 0 )) && [[ -d "$destination" ]]; then
      for nested_log in "$destination"/colt_paper_faithful_train_*.log; do
        suffix="${nested_log##*/colt_paper_faithful_train_}"
        nested_destination="$destination/colt_paper_faithful_v1_train_$suffix"
        archive_path "$nested_log" "$nested_destination"
      done
    fi
  done
  for source in "$LOG_ROOT"/colt_paper_faithful_train_*.log; do
    suffix="${source##*/colt_paper_faithful_train_}"
    destination="$LOG_ROOT/colt_paper_faithful_v1_train_$suffix"
    archive_path "$source" "$destination"
  done
  shopt -u nullglob
}

version_v1_eval_log_names() {
  local source destination suffix
  [[ -d "$V1_EVAL_LOG_DIR" ]] || return 0
  shopt -s nullglob
  for source in "$V1_EVAL_LOG_DIR"/paper-faithful_*.log; do
    suffix="${source##*/paper-faithful_}"
    destination="$V1_EVAL_LOG_DIR/paper-faithful-v1_$suffix"
    archive_path "$source" "$destination"
  done
  shopt -u nullglob
}

model_is_complete() {
  local output="${1:-$V2_CHECKPOINT}"
  "$PYTHON_BIN" - "$output" "$EXPECTED_STEPS" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_step = int(sys.argv[2])
state = output / "trainer_state.json"
index = output / "model.safetensors.index.json"
if not state.is_file() or not index.is_file():
    raise SystemExit(1)
actual_step = json.loads(state.read_text(encoding="utf-8")).get("global_step")
raise SystemExit(0 if actual_step == expected_step else 1)
PY
}

create_v2_config() {
  local template="$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_paper_faithful.yaml"
  [[ -f "$template" ]] || die "Missing paper-faithful config template: $template"
  mkdir -p "$PIPELINE_ROOT"
  "$PYTHON_BIN" - "$template" "$V2_CONFIG" "$V2_CHECKPOINT" "$CACHE_ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

template, destination, output_dir, cache_root = sys.argv[1:]
with open(template, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config.update(
    output_dir=output_dir,
    tokenized_path=str(Path(cache_root) / "colt/onethinker_sft_tokenized"),
    run_name="colt_paper_faithful_b1_v2",
    save_steps=500,
    save_total_limit=1,
    seed=42,
    data_seed=42,
)

with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
}

run_stamp="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="$PIPELINE_ROOT/paper_faithful_v2_pipeline_$run_stamp.log"
CURRENT_STAGE=preflight
trap 'status=$?; echo "Pipeline failed at stage=${CURRENT_STAGE:-unknown} exit=$status" >&2; exit "$status"' ERR

mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT" "$EVAL_ROOT"
if (( dry_run == 0 )); then
  exec > >(tee -a "$PIPELINE_LOG") 2>&1
fi

echo "Repository: $REPO_ROOT"
echo "Git HEAD: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Active environment: ${CONDA_PREFIX:-${VIRTUAL_ENV:-unknown}}"
echo "Training GPUs: $GPU_CSV"
echo "v1 checkpoint: $V1_CHECKPOINT"
echo "v2 checkpoint: $V2_CHECKPOINT"
echo "Pipeline log: $PIPELINE_LOG"

CURRENT_STAGE=archive_v1
if [[ -e "$LEGACY_CHECKPOINT" ]] && ! model_is_complete "$LEGACY_CHECKPOINT"; then
  die "Legacy paper-faithful checkpoint is not a complete $EXPECTED_STEPS-step model: $LEGACY_CHECKPOINT"
fi
archive_path "$LEGACY_CHECKPOINT" "$V1_CHECKPOINT"
archive_training_logs
archive_path "$LOG_ROOT/eval/paper-faithful" "$V1_EVAL_LOG_DIR"
(( dry_run == 1 )) || version_v1_eval_log_names
archive_path "$EVAL_ROOT/results/paper-faithful" "$V1_EVAL_OUTPUT_ROOT/paper-faithful"

CURRENT_STAGE=create_v2_config
create_v2_config

export COLT_EXPECTED_REPO_ROOT="$REPO_ROOT"
export COLT_RUNTIME_ROOT="$RUNTIME_ROOT"
export COLT_OUTPUT_ROOT="$CHECKPOINT_ROOT"
export COLT_LOG_ROOT="$LOG_ROOT"
export COLT_EVAL_ROOT="$EVAL_ROOT"
export COLT_CACHE_ROOT="$CACHE_ROOT"
export COLT_TRAIN_GPUS="$GPU_CSV"
export COLT_EVAL_GPUS="$GPU_CSV"
export COLT_EXPECTED_GLOBAL_STEP="$EXPECTED_STEPS"
export COLT_AUX_MAX_BATCH_TOKENS="${COLT_AUX_MAX_BATCH_TOKENS:-4096}"
export COLT_TRAIN_RECORD_PREFIX=colt_paper_faithful_v2_run
export COLT_TRAIN_LOG_PREFIX=colt_paper_faithful_v2_train
export COLT_EVAL_LOG_ROOT="$LOG_ROOT/eval"
export COLT_EVAL_LOG_LABEL=paper-faithful-v2
export COLT_EVAL_OUTPUT_ROOT="$V2_EVAL_OUTPUT_ROOT"
export VLMEVAL_WORKERS_PER_GPU="$WORKERS_PER_GPU"

CURRENT_STAGE=train_v2
if model_is_complete "$V2_CHECKPOINT"; then
  echo "v2 training already complete: $V2_CHECKPOINT"
else
  train_args=(
    bash "$LAUNCHER" train paper-faithful
    --config "$V2_CONFIG"
    --output-dir "$V2_CHECKPOINT"
    --batch-aux
  )
  if (( resume == 1 )); then
    train_args+=(--resume)
  elif [[ -d "$V2_CHECKPOINT" ]] && find "$V2_CHECKPOINT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    die "Incomplete v2 output exists. Inspect it, then rerun with --resume: $V2_CHECKPOINT"
  fi
  run_command "${train_args[@]}"
fi

CURRENT_STAGE=verify_v2
run_command bash "$LAUNCHER" verify model paper-faithful --model-path "$V2_CHECKPOINT"

CURRENT_STAGE=eval_v2
run_command bash "$LAUNCHER" eval paper-faithful all8 \
  --model-path "$V2_CHECKPOINT" \
  --gpus "$GPU_CSV" \
  --workers "$WORKERS_PER_GPU" \
  --generation respect-args \
  --prefetch 1 \
  --empty-cache-every 0 \
  --reseed-per-sample 1 \
  --empty-response-policy prevent

if (( dry_run == 1 )); then
  echo "paper-faithful v2 dry-run completed; no training, evaluation, or archival was performed."
else
  echo "paper-faithful v2 training and evaluation completed."
fi
echo "Checkpoint: $V2_CHECKPOINT"
echo "Evaluation logs: $V2_EVAL_LOG_DIR"
echo "Evaluation results: $V2_EVAL_OUTPUT_ROOT/paper-faithful"
