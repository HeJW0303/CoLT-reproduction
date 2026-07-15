#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="${COLT_WORKSPACE_ROOT:-$(dirname "$REPO_ROOT")}"
VENV_DIR="${COLT_VENV_DIR:-$WORKSPACE_ROOT/envs/colt}"
MODEL_ROOT="${COLT_MODEL_ROOT:-$WORKSPACE_ROOT/models}"
DATA_ROOT="${COLT_DATA_ROOT:-$WORKSPACE_ROOT/data/CoLT_Train_Dataset}"
OUTPUT_ROOT="${COLT_OUTPUT_ROOT:-$WORKSPACE_ROOT/outputs}"
CACHE_ROOT="${COLT_CACHE_ROOT:-$WORKSPACE_ROOT/cache}"

BASE_MODEL_DIR="$MODEL_ROOT/Qwen3-VL-8B-Instruct"
DECODER_MODEL_DIR="$MODEL_ROOT/Qwen3-0.6B"
A100_CONFIG="$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_a100.yaml"

BASE_MODEL_REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DECODER_MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
DATASET_REVISION="7f65a2088bd486b38c24a58c699013d008533388"
REPO_REVISION="331cc78df2d4ab542b9a83822a5a69766e194042"

export HOME="${COLT_HOME:-$WORKSPACE_ROOT/.root_home}"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
unset TRANSFORMERS_CACHE
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$CACHE_ROOT/numba"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
# DeepSpeed warns and may hang during exit when its Triton autotune cache is on NFS.
# `/dev/shm` is host tmpfs because the A100 container uses `--ipc=host`.
export TRITON_CACHE_DIR="${COLT_TRITON_CACHE_DIR:-/dev/shm/colt-hjw-triton}"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export WANDB_DIR="$WORKSPACE_ROOT/logs/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$WORKSPACE_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHON_TEMP_ROOT="$TMPDIR/python"
export JOBLIB_TEMP_FOLDER="$TMPDIR/joblib"
export RAY_TMPDIR="$WORKSPACE_ROOT/ray_tmp"

mkdir -p \
  "$HOME" "$VENV_DIR" "$MODEL_ROOT" "$DATA_ROOT" "$OUTPUT_ROOT" \
  "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" \
  "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$PYTHONPYCACHEPREFIX" \
  "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$PIP_CACHE_DIR" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TMPDIR" "$PYTHON_TEMP_ROOT" \
  "$JOBLIB_TEMP_FOLDER" "$RAY_TMPDIR"

if ! touch "$PIP_CACHE_DIR/.colt_write_test" 2>/dev/null; then
  unset PIP_CACHE_DIR
  export PIP_NO_CACHE_DIR=1
else
  rm -f "$PIP_CACHE_DIR/.colt_write_test"
fi

activate_colt_env() {
  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Missing virtual environment: $VENV_DIR" >&2
    echo "Run scripts/a100/01_setup_env.sh first." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
}

require_workspace_layout() {
  if [[ "$REPO_ROOT" != "$WORKSPACE_ROOT/CoLT" ]]; then
    echo "This A100 profile expects the repository at $WORKSPACE_ROOT/CoLT." >&2
    echo "Current repository: $REPO_ROOT" >&2
    echo "Move it to /workspace/CoLT or set COLT_WORKSPACE_ROOT consistently before setup." >&2
    exit 1
  fi
}

require_free_gib() {
  local required_gib="$1"
  local free_kb
  local required_kb
  free_kb="$(df -Pk "$WORKSPACE_ROOT" | awk 'NR==2 {print $4}')"
  required_kb=$((required_gib * 1024 * 1024))
  if (( free_kb < required_kb )); then
    echo "Less than ${required_gib} GiB is free under $WORKSPACE_ROOT." >&2
    df -h "$WORKSPACE_ROOT" >&2
    exit 1
  fi
}
