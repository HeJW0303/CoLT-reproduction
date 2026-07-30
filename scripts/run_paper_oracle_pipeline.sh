#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/scripts/lkl_8gpu/colt.sh"

# -----------------------------------------------------------------------------
# Edit these defaults for a new machine, or override them with the same-named
# environment variables. Both training runs intentionally start from BASE_MODEL.
# -----------------------------------------------------------------------------
CONDA_INIT_SH="${COLT_CONDA_INIT_SH:-/data/nvme0/lkl/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_DIR="${COLT_CONDA_ENV_DIR:-/data/nvme0/lkl/conda/envs/colt}"
BASE_MODEL_DIR="${COLT_BASE_MODEL_DIR:-/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct}"
DECODER_MODEL_DIR="${COLT_DECODER_MODEL_DIR:-/data/nvme0/lkl/models/Qwen3-0.6B}"
TRAIN_DATA_DIR="${COLT_DATA_ROOT:-/data/nvme0/lkl/datasets/CoLT_Train_Dataset}"
ORACLE_DATA_FILE="${COLT_ORACLE_K_DATA_FILE:-$TRAIN_DATA_DIR/colt_sft_image_oracle_k.json}"
EVAL_DATA_ROOT="${COLT_EVAL_DATA_ROOT:-$REPO_ROOT/eval/LMUData}"
PIPELINE_ROOT="${COLT_PIPELINE_ROOT:-$REPO_ROOT/pipeline_runs}"
PIPELINE_CACHE_ROOT="${COLT_PIPELINE_CACHE_ROOT:-$REPO_ROOT/cache}"
RUNTIME_ROOT="${COLT_LKL_ROOT:-$PIPELINE_ROOT/runtime}"
GPU_PROFILE="${COLT_GPU_PROFILE:-a100}"
GPU_CSV="${COLT_PIPELINE_GPUS:-0,1,2,3,4,5,6,7}"

mode=full
dry_run=0
explicit_run_dir=""

usage() {
  cat <<'EOF'
Run the complete CoLT reproduction pipeline on one 8-GPU machine.

Sequence:
  1. Train paper-faithful (B1 fixed; official visual/B3/backward decoder)
  2. Verify and evaluate paper-faithful
  3. Train Oracle-K from the same base model
  4. Verify and evaluate Oracle-K

Usage:
  bash scripts/run_paper_oracle_pipeline.sh [--smoke] [--dry-run] [--run-dir PATH]

Options:
  --smoke         Train one optimizer step on 64 samples and evaluate 8 MMStar rows
  --dry-run       Validate paths, generate configs, and print commands without running jobs
  --run-dir PATH  Reuse a specific run directory; completed stages are skipped
  -h, --help      Show this help

Training auxiliary decoder batching is always disabled. Evaluation throughput
acceleration uses the unified launcher's defaults (3 workers/GPU, CPU prefetch,
and no per-sample empty_cache).
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while (( $# > 0 )); do
  case "$1" in
    --smoke) mode=smoke; shift ;;
    --dry-run) dry_run=1; shift ;;
    --run-dir)
      [[ $# -ge 2 ]] || die "--run-dir requires an absolute path"
      explicit_run_dir="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ -x "$CONDA_ENV_DIR/bin/python" ]] || die "Missing Conda Python: $CONDA_ENV_DIR/bin/python"
[[ -f "$CONDA_INIT_SH" ]] || die "Missing Conda activation script: $CONDA_INIT_SH"
[[ -f "$LAUNCHER" ]] || die "Missing unified launcher: $LAUNCHER"
[[ -d "$BASE_MODEL_DIR" ]] || die "Missing base model directory: $BASE_MODEL_DIR"
[[ -d "$DECODER_MODEL_DIR" ]] || die "Missing decoder model directory: $DECODER_MODEL_DIR"
[[ -d "$TRAIN_DATA_DIR" ]] || die "Missing training dataset directory: $TRAIN_DATA_DIR"
[[ -f "$ORACLE_DATA_FILE" ]] || die "Missing Oracle-K data file: $ORACLE_DATA_FILE"
[[ -f "$TRAIN_DATA_DIR/dataset_info.json" ]] || die "Missing dataset registry: $TRAIN_DATA_DIR/dataset_info.json"
[[ "$GPU_PROFILE" == a100 || "$GPU_PROFILE" == a800 ]] || die "COLT_GPU_PROFILE must be a100 or a800"
[[ "$GPU_CSV" == "0,1,2,3,4,5,6,7" ]] || die "This pipeline requires all 8 GPUs in order: 0,1,2,3,4,5,6,7"

for model_dir in "$BASE_MODEL_DIR" "$DECODER_MODEL_DIR"; do
  [[ -f "$model_dir/config.json" ]] || die "Missing model config: $model_dir/config.json"
  compgen -G "$model_dir/*.safetensors" >/dev/null || die "No safetensors weights under $model_dir"
done

"$CONDA_ENV_DIR/bin/python" - "$TRAIN_DATA_DIR/dataset_info.json" "$ORACLE_DATA_FILE" <<'PY'
import json
import sys
from pathlib import Path

registry_path = Path(sys.argv[1]).resolve()
oracle_path = Path(sys.argv[2]).resolve()
registry = json.loads(registry_path.read_text(encoding="utf-8"))
for name in ("onethinker_sft_image", "onethinker_sft_image_oracle_k"):
    if name not in registry:
        raise RuntimeError(f"dataset_info.json does not register {name}")
    registered = Path(registry[name]["file_name"])
    if not registered.is_absolute():
        registered = registry_path.parent / registered
    registered = registered.resolve()
    if not registered.is_file():
        raise RuntimeError(f"Registered dataset file does not exist for {name}: {registered}")
    if name == "onethinker_sft_image_oracle_k" and registered != oracle_path:
        raise RuntimeError(
            f"Oracle-K registry mismatch: dataset_info.json points to {registered}, "
            f"but COLT_ORACLE_K_DATA_FILE is {oracle_path}"
        )
print("Training dataset registry: OK")
PY

run_stamp="$(date +%Y%m%d_%H%M%S)"
if [[ -n "$explicit_run_dir" ]]; then
  [[ "$explicit_run_dir" == /* ]] || die "--run-dir must be absolute: $explicit_run_dir"
  RUN_DIR="$explicit_run_dir"
elif [[ -n "${COLT_PIPELINE_RUN_DIR:-}" ]]; then
  [[ "$COLT_PIPELINE_RUN_DIR" == /* ]] || die "COLT_PIPELINE_RUN_DIR must be absolute"
  RUN_DIR="$COLT_PIPELINE_RUN_DIR"
else
  RUN_DIR="$PIPELINE_ROOT/paper_oracle_${mode}_$run_stamp"
fi

CONFIG_DIR="$RUN_DIR/configs"
STATE_DIR="$RUN_DIR/state"
CHECKPOINT_ROOT="$RUN_DIR/checkpoints"
PAPER_OUTPUT="$CHECKPOINT_ROOT/colt_paper_faithful"
ORACLE_OUTPUT="$CHECKPOINT_ROOT/colt_oracle_k_predictor"
PAPER_CONFIG="$CONFIG_DIR/paper_faithful.yaml"
ORACLE_CONFIG="$CONFIG_DIR/oracle_k.yaml"
PIPELINE_LOG="$RUN_DIR/pipeline.log"
mkdir -p "$CONFIG_DIR" "$STATE_DIR" "$CHECKPOINT_ROOT" "$RUN_DIR/logs" "$RUN_DIR/eval" "$EVAL_DATA_ROOT"

if (( dry_run == 0 )); then
  exec > >(tee -a "$PIPELINE_LOG") 2>&1
fi

expected_steps=1910
eval_group=all8
paper_tokenized="$PIPELINE_CACHE_ROOT/colt/onethinker_sft_tokenized"
oracle_tokenized="$PIPELINE_CACHE_ROOT/colt/onethinker_sft_oracle_k_predictor_tokenized"
if [[ "$mode" == smoke ]]; then
  expected_steps=1
  eval_group=smoke
  paper_tokenized="$RUN_DIR/cache/paper_faithful"
  oracle_tokenized="$RUN_DIR/cache/oracle_k"
fi
expected_steps="${COLT_PIPELINE_EXPECTED_STEPS:-$expected_steps}"
eval_group="${COLT_PIPELINE_EVAL_GROUP:-$eval_group}"
generation_mode="${COLT_PIPELINE_GENERATION:-official}"
[[ "$expected_steps" =~ ^[1-9][0-9]*$ ]] || die "Expected optimizer steps must be positive"
[[ "$generation_mode" == official || "$generation_mode" == respect-args ]] || die \
  "COLT_PIPELINE_GENERATION must be official or respect-args"

create_config() {
  local template="$1" destination="$2" dataset_name="$3" tokenized="$4" output="$5" run_name="$6"
  "$CONDA_ENV_DIR/bin/python" - \
    "$template" "$destination" "$BASE_MODEL_DIR" "$TRAIN_DATA_DIR" "$dataset_name" \
    "$tokenized" "$output" "$run_name" "$mode" "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

(
    template,
    destination,
    base_model,
    data_dir,
    dataset_name,
    tokenized_path,
    output_dir,
    run_name,
    mode,
    repo_root,
) = sys.argv[1:]
with open(template, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config.update(
    model_name_or_path=base_model,
    dataset=dataset_name,
    dataset_dir=data_dir,
    media_dir=data_dir,
    tokenized_path=tokenized_path,
    output_dir=output_dir,
    run_name=run_name,
    deepspeed=str(Path(repo_root) / "LLaMA-Factory/examples/deepspeed/ds_z3_a100.json"),
)
if mode == "smoke":
    config.update(
        max_steps=1,
        max_samples=64,
        save_strategy="no",
        report_to="none",
        plot_loss=False,
        seed=42,
        data_seed=42,
    )

destination_path = Path(destination)
destination_path.parent.mkdir(parents=True, exist_ok=True)
with destination_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
}

create_config \
  "$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_paper_faithful.yaml" \
  "$PAPER_CONFIG" onethinker_sft_image "$paper_tokenized" "$PAPER_OUTPUT" \
  "paper_faithful_${mode}_$run_stamp"
create_config \
  "$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_oracle_k_predictor.yaml" \
  "$ORACLE_CONFIG" onethinker_sft_image_oracle_k "$oracle_tokenized" "$ORACLE_OUTPUT" \
  "oracle_k_${mode}_$run_stamp"

export COLT_EXPECTED_REPO_ROOT="$REPO_ROOT"
export COLT_LKL_ROOT="$RUNTIME_ROOT"
export COLT_CONDA_INIT_SH="$CONDA_INIT_SH"
export COLT_CONDA_ENV_DIR="$CONDA_ENV_DIR"
export COLT_BASE_MODEL_DIR="$BASE_MODEL_DIR"
export COLT_DECODER_MODEL_DIR="$DECODER_MODEL_DIR"
export COLT_DATA_ROOT="$TRAIN_DATA_DIR"
export COLT_OUTPUT_ROOT="$CHECKPOINT_ROOT"
export COLT_CACHE_ROOT="$PIPELINE_CACHE_ROOT"
export COLT_LOG_ROOT="$RUN_DIR/logs"
export COLT_EVAL_ROOT="$RUN_DIR/eval"
export COLT_EVAL_DATA_ROOT="$EVAL_DATA_ROOT"
export COLT_EVAL_OUTPUT_ROOT="$RUN_DIR/eval/results"
export COLT_EVAL_LOG_ROOT="$RUN_DIR/logs/eval"
export COLT_TMP_ROOT="$RUN_DIR/tmp"
export COLT_GPU_PROFILE="$GPU_PROFILE"
export COLT_TRAIN_GPUS="$GPU_CSV"
export COLT_EVAL_GPUS="$GPU_CSV"
export COLT_BATCH_AUX_DECODERS=0
export COLT_EXPECTED_GLOBAL_STEP="$expected_steps"
export COLT_ORACLE_K_DATASET_NAME=onethinker_sft_image_oracle_k
export COLT_ORACLE_K_DATA_FILE="$ORACLE_DATA_FILE"
export COLT_ORACLE_K_TOKENIZED_PATH="$oracle_tokenized"
export COLT_ORACLE_K_OUTPUT_DIR="$ORACLE_OUTPUT"
export COLT_ORACLE_K_DYNAMIC_INFERENCE=1
unset RESUME COLT_TRAIN_CONFIG COLT_TRAIN_OUTPUT_DIR COLT_EVAL_MODEL_PATH
unset COLT_BENCHMARK_MODE COLT_SKIP_FINAL_SAVE COLT_INFERENCE_K

cat > "$RUN_DIR/pipeline_environment.txt" <<EOF
git_head=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
mode=$mode
repo_root=$REPO_ROOT
runtime_root=$RUNTIME_ROOT
conda_env=$CONDA_ENV_DIR
conda_init=$CONDA_INIT_SH
base_model=$BASE_MODEL_DIR
decoder_model=$DECODER_MODEL_DIR
train_data=$TRAIN_DATA_DIR
oracle_data=$ORACLE_DATA_FILE
eval_data=$EVAL_DATA_ROOT
gpus=$GPU_CSV
expected_steps=$expected_steps
eval_group=$eval_group
generation=$generation_mode
batch_aux=0
run_dir=$RUN_DIR
EOF

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

model_is_complete() {
  local output="$1"
  "$CONDA_ENV_DIR/bin/python" - "$output" "$expected_steps" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected = int(sys.argv[2])
state = output / "trainer_state.json"
index = output / "model.safetensors.index.json"
if not state.is_file() or not index.is_file():
    raise SystemExit(1)
actual = json.loads(state.read_text(encoding="utf-8")).get("global_step")
raise SystemExit(0 if actual == expected else 1)
PY
}

train_target() {
  local target="$1" config="$2" output="$3"
  if model_is_complete "$output"; then
    echo "Training output already complete: $output"
    return
  fi
  local -a args=(bash "$LAUNCHER" train "$target" --config "$config" --output-dir "$output")
  if [[ -d "$output" ]] && find "$output" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    local latest_checkpoint
    latest_checkpoint="$(find "$output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
    [[ -n "$latest_checkpoint" && -f "$latest_checkpoint/trainer_state.json" ]] || die \
      "Incomplete non-resumable output at $output. Use a new --run-dir or inspect it manually."
    args+=(--resume)
  fi
  run_command "${args[@]}"
  (( dry_run == 1 )) || run_command bash "$LAUNCHER" verify model "$target" --model-path "$output"
}

evaluate_target() {
  local target="$1" output="$2"
  if (( dry_run == 0 )); then
    run_command bash "$LAUNCHER" verify model "$target" --model-path "$output"
  fi
  run_command bash "$LAUNCHER" eval "$target" "$eval_group" \
    --model-path "$output" --gpus "$GPU_CSV" --generation "$generation_mode"
}

run_stage() {
  local stage="$1"
  shift
  local marker="$STATE_DIR/$stage.complete"
  if [[ -f "$marker" && $dry_run -eq 0 ]]; then
    echo "Skipping completed stage: $stage"
    return
  fi
  CURRENT_STAGE="$stage"
  echo "===== START $stage ====="
  "$@"
  if (( dry_run == 0 )); then
    printf '%s\n' "$(date -Iseconds)" > "$marker"
  fi
  echo "===== DONE $stage ====="
}

CURRENT_STAGE=preflight
trap 'status=$?; echo "Pipeline failed at stage=${CURRENT_STAGE:-unknown} exit=$status" >&2; exit "$status"' ERR

echo "Run directory: $RUN_DIR"
echo "Mode: $mode"
echo "Training auxiliary decoder batching: disabled"
echo "Evaluation acceleration: launcher defaults"
echo "Evaluation group: $eval_group"
echo "Generation mode: $generation_mode"

run_stage paper_train train_target paper-faithful "$PAPER_CONFIG" "$PAPER_OUTPUT"
run_stage paper_eval evaluate_target paper-faithful "$PAPER_OUTPUT"
run_stage oracle_train train_target oracle-k "$ORACLE_CONFIG" "$ORACLE_OUTPUT"
export COLT_LOG_PREDICTED_K=1
run_stage oracle_eval evaluate_target oracle-k "$ORACLE_OUTPUT"

if (( dry_run == 0 )); then
  printf '%s\n' "$(date -Iseconds)" > "$STATE_DIR/pipeline.complete"
fi
echo "Pipeline completed: $RUN_DIR"
