#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env

oracle_config="$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_oracle_k_a100.yaml"
oracle_data="${COLT_ORACLE_K_DATA_FILE:-$DATA_ROOT/colt_sft_image_oracle_k.json}"
oracle_dataset_name="${COLT_ORACLE_K_DATASET_NAME:-onethinker_sft_image_oracle_k}"
output_dir="${COLT_ORACLE_K_OUTPUT_DIR:-$OUTPUT_ROOT/colt_oracle_k_predictor}"
tokenized_path="${COLT_ORACLE_K_TOKENIZED_PATH:-$CACHE_ROOT/colt/onethinker_sft_oracle_k_tokenized}"
run_name="${COLT_ORACLE_K_RUN_NAME:-colt_sft_8b_a100_oracle_k_predictor}"
oracle_original_start="${COLT_ORACLE_K_ORIGINAL_START:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export DECORD_EOF_RETRY_MAX=2048001
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export COLT_DECODER_MODEL_PATH="$DECODER_MODEL_DIR"
export COLT_ORACLE_K_ENABLED=1
export COLT_ORACLE_K_MAX="${COLT_ORACLE_K_MAX:-8}"
export COLT_ORACLE_K_BUDGET_CONDITIONING="${COLT_ORACLE_K_BUDGET_CONDITIONING:-1}"
export COLT_ORACLE_K_PREDICTOR_ENABLED="${COLT_ORACLE_K_PREDICTOR_ENABLED:-1}"
export COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT="${COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT:-0.2}"
export COLT_ORACLE_K_DYNAMIC_INFERENCE="${COLT_ORACLE_K_DYNAMIC_INFERENCE:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-CoLT-oracle-k-predictor}"

python - "$oracle_config" "$COLT_ORACLE_K_MAX" "$COLT_ORACLE_K_BUDGET_CONDITIONING" \
  "$BASE_MODEL_DIR" "$COLT_ORACLE_K_PREDICTOR_ENABLED" \
  "$COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT" "$COLT_ORACLE_K_DYNAMIC_INFERENCE" <<'PY'
import os
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_vl.oracle_k import parse_oracle_k_cot, resolve_oracle_k_settings

base_model_path = sys.argv[4]
config = AutoConfig.from_pretrained(base_model_path, local_files_only=True)
assert "colt_oracle_k_enabled" not in config.__dict__, "Base model config unexpectedly contains Oracle-K settings"
assert os.environ["COLT_ORACLE_K_ENABLED"] == "1"
assert int(sys.argv[2]) >= 1
assert os.environ["COLT_ORACLE_K_MAX"] == sys.argv[2]
assert os.environ["COLT_ORACLE_K_BUDGET_CONDITIONING"] == sys.argv[3]
assert os.environ["COLT_ORACLE_K_PREDICTOR_ENABLED"] == sys.argv[5]
assert os.environ["COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT"] == sys.argv[6]
assert os.environ["COLT_ORACLE_K_DYNAMIC_INFERENCE"] == sys.argv[7]
assert Path(sys.argv[1]).is_file()
settings = resolve_oracle_k_settings(config)
expected_predictor = sys.argv[5].strip().lower() in {"1", "true", "yes", "on"}
expected_dynamic = sys.argv[7].strip().lower() in {"1", "true", "yes", "on"}
assert settings.enabled
assert settings.predictor_enabled is expected_predictor
assert settings.dynamic_inference is expected_dynamic
assert settings.predictor_loss_weight == float(sys.argv[6])
assert config.colt_oracle_k_predictor_enabled is expected_predictor
assert config.colt_oracle_k_dynamic_inference is expected_dynamic
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(os.environ["COLT_DECODER_MODEL_PATH"], local_files_only=True)
assert base_tokenizer.get_vocab() == tokenizer.get_vocab(), "Base and decoder token ID mappings differ"
max_k = int(sys.argv[2])
sample = (
    "<thought_segments>2</thought_segments>plan.<continue_think>derive."
    if max_k >= 2
    else "<thought_segments>1</thought_segments>plan."
)
round_trip = tokenizer.decode(
    tokenizer.encode(sample, add_special_tokens=False),
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False,
)
annotation = parse_oracle_k_cot(round_trip, max_k=max_k)
expected_cot = "plan.derive." if max_k >= 2 else "plan."
assert annotation.original_cot == expected_cot, (round_trip, annotation)
print("Oracle-K environment/config preflight: OK")
PY

python -m unittest discover -s "$REPO_ROOT/tests/oracle_k" -p 'test_*.py' -v

if [[ ! -f "$oracle_data" ]]; then
  echo "Missing Oracle-K dataset: $oracle_data" >&2
  echo "Run scripts/a100/17_import_local_oracle_k_data.sh or scripts/a100/17_prepare_oracle_k_data.sh first." >&2
  exit 1
fi
python - "$DATA_ROOT/dataset_info.json" "$oracle_dataset_name" "$oracle_data" <<'PY'
import json
import sys
from pathlib import Path

info_path = Path(sys.argv[1])
dataset_name = sys.argv[2]
data_path = Path(sys.argv[3]).resolve()
with info_path.open(encoding="utf-8") as file:
    dataset_info = json.load(file)
if dataset_name not in dataset_info:
    raise KeyError(f"dataset_info.json does not register {dataset_name}")
registered_path = (info_path.parent / dataset_info[dataset_name]["file_name"]).resolve()
if registered_path != data_path:
    raise ValueError(f"Dataset registration points to {registered_path}, expected {data_path}")
print(f"Oracle-K dataset registration: {dataset_name} -> {data_path}")
PY
python "$REPO_ROOT/scripts/oracle_k/validate_oracle_blocks.py" \
  --input "$oracle_data" \
  --original "$DATA_ROOT/colt_sft_image.json" \
  --original-start "$oracle_original_start" \
  --max-k "$COLT_ORACLE_K_MAX"

mkdir -p "$output_dir" "$WORKSPACE_ROOT/logs"
if find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  if [[ "${RESUME:-0}" != "1" ]]; then
    echo "Output directory is not empty: $output_dir" >&2
    echo "For an interrupted Oracle-K run, rerun with RESUME=1." >&2
    exit 1
  fi
  latest_checkpoint="$(find "$output_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
  if [[ -z "$latest_checkpoint" || ! -f "$latest_checkpoint/trainer_state.json" ]]; then
    echo "RESUME=1 was requested, but no complete Trainer checkpoint was found." >&2
    exit 1
  fi
  python - "$latest_checkpoint/config.json" "$COLT_ORACLE_K_MAX" "$COLT_ORACLE_K_BUDGET_CONDITIONING" \
    "$COLT_ORACLE_K_PREDICTOR_ENABLED" "$COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT" \
    "$COLT_ORACLE_K_DYNAMIC_INFERENCE" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
if not config_path.is_file():
    raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
with config_path.open(encoding="utf-8") as file:
    config = json.load(file)
expected_budget = sys.argv[3].strip().lower() in {"1", "true", "yes", "on"}
expected_predictor = sys.argv[4].strip().lower() in {"1", "true", "yes", "on"}
expected_dynamic = sys.argv[6].strip().lower() in {"1", "true", "yes", "on"}
if config.get("colt_oracle_k_enabled") is not True:
    raise ValueError("Checkpoint is not an Oracle-K checkpoint")
if int(config.get("colt_oracle_k_max", -1)) != int(sys.argv[2]):
    raise ValueError("Checkpoint colt_oracle_k_max differs from this run")
if bool(config.get("colt_oracle_k_budget_conditioning")) != expected_budget:
    raise ValueError("Checkpoint budget-conditioning setting differs from this run")
if bool(config.get("colt_oracle_k_predictor_enabled")) != expected_predictor:
    raise ValueError("Checkpoint K-predictor setting differs from this run")
if abs(float(config.get("colt_oracle_k_predictor_loss_weight", -1.0)) - float(sys.argv[5])) > 1e-12:
    raise ValueError("Checkpoint K-predictor loss weight differs from this run")
if bool(config.get("colt_oracle_k_dynamic_inference")) != expected_dynamic:
    raise ValueError("Checkpoint dynamic-inference setting differs from this run")
print("Oracle-K resume config: OK")
PY
  echo "LLaMA-Factory will resume from: $latest_checkpoint"
fi

require_free_gib 200
cd "$REPO_ROOT"
run_stamp="$(date +%Y%m%d_%H%M%S)"
run_record="$WORKSPACE_ROOT/logs/colt_oracle_k_run_$run_stamp"
mkdir -p "$run_record"
cp "$oracle_config" "$run_record/"
cp "$REPO_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_a100.json" "$run_record/"
cp "$REPO_ROOT/scripts/a100/18_train_oracle_k.sh" "$run_record/"
cp "$REPO_ROOT/scripts/oracle_k/validate_oracle_blocks.py" "$run_record/"
git -c safe.directory="$REPO_ROOT" rev-parse HEAD > "$run_record/git_head.txt"
git -c safe.directory="$REPO_ROOT" status --short > "$run_record/git_status.txt"
git -c safe.directory="$REPO_ROOT" diff --binary > "$run_record/git_diff.patch"
python -m pip freeze > "$run_record/pip_freeze.txt"
{
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'MASTER_PORT=%s\n' "$MASTER_PORT"
  printf 'COLT_DECODER_MODEL_PATH=%s\n' "$COLT_DECODER_MODEL_PATH"
  printf 'COLT_ORACLE_K_ENABLED=%s\n' "$COLT_ORACLE_K_ENABLED"
  printf 'COLT_ORACLE_K_MAX=%s\n' "$COLT_ORACLE_K_MAX"
  printf 'COLT_ORACLE_K_BUDGET_CONDITIONING=%s\n' "$COLT_ORACLE_K_BUDGET_CONDITIONING"
  printf 'COLT_ORACLE_K_PREDICTOR_ENABLED=%s\n' "$COLT_ORACLE_K_PREDICTOR_ENABLED"
  printf 'COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT=%s\n' "$COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT"
  printf 'COLT_ORACLE_K_DYNAMIC_INFERENCE=%s\n' "$COLT_ORACLE_K_DYNAMIC_INFERENCE"
  printf 'COLT_ORACLE_K_DATA_FILE=%s\n' "$oracle_data"
  printf 'COLT_ORACLE_K_DATASET_NAME=%s\n' "$oracle_dataset_name"
  printf 'COLT_ORACLE_K_OUTPUT_DIR=%s\n' "$output_dir"
  printf 'COLT_ORACLE_K_TOKENIZED_PATH=%s\n' "$tokenized_path"
  printf 'COLT_ORACLE_K_RUN_NAME=%s\n' "$run_name"
  printf 'COLT_ORACLE_K_ORIGINAL_START=%s\n' "$oracle_original_start"
  printf 'WANDB_MODE=%s\n' "$WANDB_MODE"
  printf 'WANDB_PROJECT=%s\n' "$WANDB_PROJECT"
  printf 'HF_HOME=%s\n' "$HF_HOME"
  printf 'TMPDIR=%s\n' "$TMPDIR"
} > "$run_record/environment.txt"

log_file="$WORKSPACE_ROOT/logs/colt_oracle_k_train_$run_stamp.log"
echo "Training log: $log_file"
llamafactory-cli train "$oracle_config" \
  "dataset=$oracle_dataset_name" \
  "tokenized_path=$tokenized_path" \
  "output_dir=$output_dir" \
  "run_name=$run_name" \
  2>&1 | tee "$log_file"
