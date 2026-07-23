#!/usr/bin/env bash

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_ROOT/../.." && pwd)"
LKL_ROOT="${COLT_LKL_ROOT:-/data/nvme0/lkl}"
EXPECTED_REPO_ROOT="${COLT_EXPECTED_REPO_ROOT:-$LKL_ROOT/CoLT-reproduction}"
PROJECT_RUNTIME_ROOT="${COLT_RUNTIME_ROOT:-$REPO_ROOT}"
WORKSPACE_ROOT="$PROJECT_RUNTIME_ROOT"
MINICONDA_ROOT="${COLT_MINICONDA_ROOT:-$LKL_ROOT/miniconda3}"
CONDA_ENVS_ROOT="${COLT_CONDA_ENVS_ROOT:-$LKL_ROOT/conda/envs}"
CONDA_ENV_DIR="${COLT_CONDA_ENV_DIR:-$CONDA_ENVS_ROOT/colt}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$LKL_ROOT/conda/pkgs}"
MODEL_ROOT="${COLT_MODEL_ROOT:-$LKL_ROOT/models}"
DATA_ROOT="${COLT_DATA_ROOT:-$LKL_ROOT/datasets/CoLT_Train_Dataset}"
OUTPUT_ROOT="${COLT_OUTPUT_ROOT:-$PROJECT_RUNTIME_ROOT/checkpoints}"
CACHE_ROOT="${COLT_CACHE_ROOT:-$PROJECT_RUNTIME_ROOT/cache}"
LOG_ROOT="${COLT_LOG_ROOT:-$PROJECT_RUNTIME_ROOT/logs}"
EVAL_ROOT_DEFAULT="${COLT_EVAL_ROOT:-$PROJECT_RUNTIME_ROOT/eval}"
TMP_ROOT="${COLT_TMP_ROOT:-$PROJECT_RUNTIME_ROOT/tmp}"

BASE_MODEL_DIR="$MODEL_ROOT/Qwen3-VL-8B-Instruct"
DECODER_MODEL_DIR="$MODEL_ROOT/Qwen3-0.6B"
TRAIN_CONFIG="$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu.yaml"

BASE_MODEL_REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DECODER_MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
DATASET_REVISION="7f65a2088bd486b38c24a58c699013d008533388"
REPO_REVISION="331cc78df2d4ab542b9a83822a5a69766e194042"

PROFILE_FILE="${COLT_PROFILE_FILE:-$PROJECT_RUNTIME_ROOT/.colt_gpu_profile}"
if [[ -z "${COLT_GPU_PROFILE:-}" && -f "$PROFILE_FILE" ]]; then
  COLT_GPU_PROFILE="$(tr -d '[:space:]' < "$PROFILE_FILE")"
fi
if [[ -z "${COLT_GPU_PROFILE:-}" ]]; then
  echo "GPU profile is not configured." >&2
  echo "Run: bash scripts/lkl_8gpu/00_verify_host.sh a100  # or a800" >&2
  exit 1
fi
case "$COLT_GPU_PROFILE" in
  a100|a800) ;;
  *)
    echo "Unsupported COLT_GPU_PROFILE=$COLT_GPU_PROFILE (expected a100 or a800)." >&2
    exit 1
    ;;
esac
# shellcheck disable=SC1090
source "$SCRIPT_ROOT/profiles/$COLT_GPU_PROFILE.sh"
export COLT_GPU_PROFILE CONDA_PKGS_DIRS

# Keep the user's real HOME unchanged. Route only large caches and temporary data to NVMe.
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="${COLT_HF_HOME:-$LKL_ROOT/hf-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
unset TRANSFORMERS_CACHE
export TORCH_HOME="${COLT_TORCH_HOME:-$LKL_ROOT/torch-cache}"
export TORCH_EXTENSIONS_DIR="$TORCH_HOME/extensions"
export TORCHINDUCTOR_CACHE_DIR="$TORCH_HOME/inductor"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$CACHE_ROOT/numba"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export TRITON_CACHE_DIR="${COLT_TRITON_CACHE_DIR:-$TMP_ROOT/triton}"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export WANDB_DIR="$LOG_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export TMPDIR="$TMP_ROOT"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHON_TEMP_ROOT="$TMPDIR/python"
export JOBLIB_TEMP_FOLDER="$TMPDIR/joblib"
export RAY_TMPDIR="$TMPDIR/ray"

mkdir -p \
  "$CONDA_ENVS_ROOT" "$CONDA_PKGS_DIRS" "$MODEL_ROOT" "$DATA_ROOT" \
  "$OUTPUT_ROOT" "$LOG_ROOT" "$EVAL_ROOT_DEFAULT" \
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
  local conda_sh="$MINICONDA_ROOT/etc/profile.d/conda.sh"
  if [[ ! -f "$conda_sh" ]]; then
    echo "Missing Miniconda activation script: $conda_sh" >&2
    exit 1
  fi
  if [[ ! -x "$CONDA_ENV_DIR/bin/python" ]]; then
    echo "Missing Conda environment: $CONDA_ENV_DIR" >&2
    echo "Run scripts/lkl_8gpu/01_setup_env.sh first." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$CONDA_ENV_DIR"
}

require_workspace_layout() {
  if [[ "$REPO_ROOT" != "$EXPECTED_REPO_ROOT" ]]; then
    echo "This profile expects the repository at $EXPECTED_REPO_ROOT." >&2
    echo "Current repository: $REPO_ROOT" >&2
    echo "Clone it there, or set COLT_EXPECTED_REPO_ROOT explicitly." >&2
    exit 1
  fi
}

validate_gpu_profile() {
  local -a gpu_names
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is not available." >&2
    exit 1
  fi
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//')
  if [[ "${#gpu_names[@]}" -ne 8 ]]; then
    echo "Expected 8 GPUs, found ${#gpu_names[@]}." >&2
    printf '  %s\n' "${gpu_names[@]}" >&2
    exit 1
  fi
  local gpu_name
  for gpu_name in "${gpu_names[@]}"; do
    if [[ "$gpu_name" != *"$COLT_EXPECTED_GPU_NAME"* ]]; then
      echo "Profile $COLT_GPU_PROFILE expects $COLT_EXPECTED_GPU_NAME, found: $gpu_name" >&2
      exit 1
    fi
  done
}

require_free_gib() {
  local required_gib="$1"
  local free_kb
  local required_kb
  free_kb="$(df -Pk "$LKL_ROOT" | awk 'NR==2 {print $4}')"
  required_kb=$((required_gib * 1024 * 1024))
  if (( free_kb < required_kb )); then
    echo "Less than ${required_gib} GiB is free under $LKL_ROOT." >&2
    df -h "$LKL_ROOT" >&2
    exit 1
  fi
}
