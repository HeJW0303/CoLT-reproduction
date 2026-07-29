#!/usr/bin/env bash

# Shared paths and runtime initialization. Sourcing this file has no side
# effects; commands opt in to profile validation, directory creation, and
# Conda activation explicitly.

COLT_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$COLT_SCRIPT_ROOT/../.." && pwd)"

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
EVAL_ROOT="${COLT_EVAL_ROOT:-$PROJECT_RUNTIME_ROOT/eval}"
TMP_ROOT="${COLT_TMP_ROOT:-$PROJECT_RUNTIME_ROOT/tmp}"

BASE_MODEL_DIR="$MODEL_ROOT/Qwen3-VL-8B-Instruct"
DECODER_MODEL_DIR="$MODEL_ROOT/Qwen3-0.6B"
OFFICIAL_MODEL_DIR_DEFAULT="$MODEL_ROOT/hub/models--hulianyuyy--CoLT-8B/snapshots/8e649ea84e7ae5b6c78b98272a0cd43537fcc66c"
VLMEVAL_ROOT="$REPO_ROOT/Evaluation/VLMEvalKit"

BASE_MODEL_REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DECODER_MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
DATASET_REVISION="7f65a2088bd486b38c24a58c699013d008533388"
PROFILE_FILE="${COLT_PROFILE_FILE:-$PROJECT_RUNTIME_ROOT/.colt_gpu_profile}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

load_gpu_profile() {
  if [[ -z "${COLT_GPU_PROFILE:-}" && -f "$PROFILE_FILE" ]]; then
    COLT_GPU_PROFILE="$(tr -d '[:space:]' < "$PROFILE_FILE")"
  fi
  case "${COLT_GPU_PROFILE:-}" in
    a100|a800) ;;
    *)
      die "GPU profile is not configured. Run: bash scripts/lkl_8gpu/colt.sh profile a100 (or a800)"
      ;;
  esac
  # shellcheck disable=SC1090
  source "$COLT_SCRIPT_ROOT/profiles/$COLT_GPU_PROFILE.sh"
  export COLT_GPU_PROFILE CONDA_PKGS_DIRS
}

configure_runtime_paths() {
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
    "$OUTPUT_ROOT" "$LOG_ROOT" "$EVAL_ROOT" \
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
}

runtime_init() {
  load_gpu_profile
  configure_runtime_paths
}

require_workspace_layout() {
  [[ "$REPO_ROOT" == "$EXPECTED_REPO_ROOT" ]] || die \
    "Expected repository $EXPECTED_REPO_ROOT, found $REPO_ROOT. Set COLT_EXPECTED_REPO_ROOT only for an intentional alternate checkout."
}

activate_colt_env() {
  local conda_sh="$MINICONDA_ROOT/etc/profile.d/conda.sh"
  [[ -f "$conda_sh" ]] || die "Missing Miniconda activation script: $conda_sh"
  [[ -x "$CONDA_ENV_DIR/bin/python" ]] || die \
    "Missing Conda environment $CONDA_ENV_DIR. Run: bash scripts/lkl_8gpu/colt.sh setup env"
  # shellcheck disable=SC1090
  source "$conda_sh"
  conda activate "$CONDA_ENV_DIR"
}

require_free_gib() {
  local required_gib="$1" free_kb required_kb
  free_kb="$(df -Pk "$LKL_ROOT" | awk 'NR==2 {print $4}')"
  required_kb=$((required_gib * 1024 * 1024))
  (( free_kb >= required_kb )) || die "Less than ${required_gib} GiB is free under $LKL_ROOT."
}
