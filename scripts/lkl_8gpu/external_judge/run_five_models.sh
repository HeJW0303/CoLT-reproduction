#!/usr/bin/env bash

set -eEuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LKL_8GPU_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$LKL_8GPU_ROOT/../.." && pwd)"
COLT_LAUNCHER="$LKL_8GPU_ROOT/colt.sh"
EXTERNAL_JUDGE_LAUNCHER="$SCRIPT_DIR/run.sh"

CHECKPOINT_ROOT="${COLT_OUTPUT_ROOT:-$REPO_ROOT/checkpoints}"
CODEFAITHFUL_CHECKPOINT="${COLT_CODEFAITHFUL_CHECKPOINT:-$CHECKPOINT_ROOT/colt_codefaithful}"
PAPER_V1_CHECKPOINT="${COLT_PAPER_V1_CHECKPOINT:-$CHECKPOINT_ROOT/colt_paper_faithful_v1}"
PAPER_V2_CHECKPOINT="${COLT_PAPER_V2_CHECKPOINT:-$CHECKPOINT_ROOT/colt_paper_faithful_v2}"
ORACLE_K_CHECKPOINT="${COLT_ORACLE_K_CHECKPOINT:-$CHECKPOINT_ROOT/colt_oracle_k_predictor_batch_285190c}"
BASE_MODEL="${COLT_BASE_MODEL_DIR:-/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct}"
CONDA_ENV="${COLT_CONDA_ENV_DIR:-/data/nvme0/lkl/conda/envs/colt}"
CONDA_INIT_SH="${COLT_CONDA_INIT_SH:-/data/nvme0/lkl/miniconda3/etc/profile.d/conda.sh}"

GPU_CSV="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${VLMEVAL_WORKERS_PER_GPU:-3}"
API_NPROC="${COLT_EXTERNAL_JUDGE_NPROC:-8}"
JUDGE_RETRY="${COLT_EXTERNAL_JUDGE_RETRY:-5}"
RESULT_ROOT="${COLT_FIVE_MODEL_RESULT_ROOT:-$REPO_ROOT/eval/external_judge/five_models}"
LOG_ROOT="${COLT_FIVE_MODEL_LOG_ROOT:-$REPO_ROOT/logs/eval/external_judge/five_models}"
DRY_RUN=0
CURRENT_STAGE=preflight

usage() {
  cat <<'EOF'
Evaluate four local CoLT checkpoints and the Qwen3-VL-8B-Instruct baseline on
MathVista_MINI, MathVerse_MINI, and MMVet with the configured external judge.

Usage:
  bash run_five_models.sh [--gpus CSV] [--workers N] [--api-nproc N]
                          [--judge-retry N] [--dry-run]

Defaults:
  GPUs: 0,1,2,3,4,5,6,7
  inference workers per GPU: 3
  judge API concurrency: 8
  judge retry count: 5
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while (( $# > 0 )); do
  case "$1" in
    --gpus) [[ $# -ge 2 ]] || die "--gpus requires a value"; GPU_CSV="$2"; shift 2 ;;
    --workers) [[ $# -ge 2 ]] || die "--workers requires a value"; WORKERS_PER_GPU="$2"; shift 2 ;;
    --api-nproc) [[ $# -ge 2 ]] || die "--api-nproc requires a value"; API_NPROC="$2"; shift 2 ;;
    --judge-retry) [[ $# -ge 2 ]] || die "--judge-retry requires a value"; JUDGE_RETRY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ "$GPU_CSV" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpus must be a comma-separated integer list"
[[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "--workers must be a positive integer"
[[ "$API_NPROC" =~ ^[1-9][0-9]*$ ]] || die "--api-nproc must be a positive integer"
[[ "$JUDGE_RETRY" =~ ^[1-9][0-9]*$ ]] || die "--judge-retry must be a positive integer"
[[ -f "$COLT_LAUNCHER" ]] || die "Missing CoLT launcher: $COLT_LAUNCHER"
[[ -f "$EXTERNAL_JUDGE_LAUNCHER" ]] || die "Missing external-judge launcher: $EXTERNAL_JUDGE_LAUNCHER"
[[ -x "$CONDA_ENV/bin/python" ]] || die "Missing colt Conda environment: $CONDA_ENV"

labels=(codefaithful paper-faithful-v1 paper-faithful-v2 oracle-k qwen3-vl-8b-base)
targets=(codefaithful paper-faithful paper-faithful oracle-k baseline)
models=(
  "$CODEFAITHFUL_CHECKPOINT"
  "$PAPER_V1_CHECKPOINT"
  "$PAPER_V2_CHECKPOINT"
  "$ORACLE_K_CHECKPOINT"
  "$BASE_MODEL"
)

for model in "${models[@]}"; do
  [[ "$model" == /* ]] || die "Model path must be absolute: $model"
  [[ -d "$model" ]] || die "Model directory does not exist: $model"
done

common_environment=(
  "COLT_EXPECTED_REPO_ROOT=$REPO_ROOT"
  "COLT_RUNTIME_ROOT=$REPO_ROOT"
  "COLT_OUTPUT_ROOT=$CHECKPOINT_ROOT"
  "COLT_EVAL_ROOT=$REPO_ROOT/eval"
  "COLT_EVAL_DATA_ROOT=$REPO_ROOT/eval/LMUData"
  "COLT_CONDA_ENV_DIR=$CONDA_ENV"
)

print_command() {
  printf 'DRY RUN:'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  if (( DRY_RUN )); then
    print_command "$@"
  else
    "$@"
  fi
}

run_evaluation() {
  local label="$1" target="$2" model_path="$3" transition="$4" empty_policy="$5" reseed="$6"
  CURRENT_STAGE="evaluate-$label"
  echo "Evaluating $label"
  echo "  target=$target"
  echo "  model=$model_path"
  echo "  results=$RESULT_ROOT/$label"
  run_command env \
    "${common_environment[@]}" \
    "COLT_EVAL_LOG_LABEL=$label" \
    "COLT_EXTERNAL_JUDGE_OUTPUT_ROOT=$RESULT_ROOT/$label" \
    "COLT_EXTERNAL_JUDGE_LOG_ROOT=$LOG_ROOT/$label" \
    bash "$EXTERNAL_JUDGE_LAUNCHER" eval "$target" \
      --model-path "$model_path" \
      --gpus "$GPU_CSV" \
      --workers "$WORKERS_PER_GPU" \
      --generation respect-args \
      --latent-transition "$transition" \
      --prefetch 1 \
      --empty-cache-every 0 \
      --dist-backend gloo \
      --reseed-per-sample "$reseed" \
      --empty-response-policy "$empty_policy" \
      --api-nproc "$API_NPROC" \
      --judge-retry "$JUDGE_RETRY"
}

if (( DRY_RUN == 0 )); then
  [[ -f "$CONDA_INIT_SH" ]] || die "Missing Conda activation script: $CONDA_INIT_SH"
  # shellcheck disable=SC1090
  source "$CONDA_INIT_SH"
  conda activate "$CONDA_ENV"
  [[ "$(command -v python)" == "$CONDA_ENV/bin/python" ]] || die \
    "Failed to activate the requested colt environment: $CONDA_ENV"
  mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
  pipeline_log="$LOG_ROOT/run_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$pipeline_log") 2>&1
  trap 'status=$?; echo "Five-model evaluation failed at stage=$CURRENT_STAGE exit=$status" >&2; exit "$status"' ERR
fi

unset COLT_INFERENCE_K

echo "Repository: $REPO_ROOT"
echo "Datasets: MathVista_MINI MathVerse_MINI MMVet"
echo "Physical GPUs: $GPU_CSV"
echo "Inference workers per GPU: $WORKERS_PER_GPU"
echo "Judge API concurrency: $API_NPROC"
echo "Judge retry count: $JUDGE_RETRY"
echo "Result root: $RESULT_ROOT"
echo "Log root: $LOG_ROOT"

CURRENT_STAGE=download-and-verify-datasets
run_command env "${common_environment[@]}" bash "$EXTERNAL_JUDGE_LAUNCHER" download

CURRENT_STAGE=preflight-all-models
echo "Preflighting all five models before starting the first evaluation."
for index in "${!targets[@]}"; do
  run_command env "${common_environment[@]}" bash "$COLT_LAUNCHER" verify model \
    "${targets[$index]}" --model-path "${models[$index]}"
done

run_evaluation codefaithful codefaithful "$CODEFAITHFUL_CHECKPOINT" training-consistent prevent 1
run_evaluation paper-faithful-v1 paper-faithful "$PAPER_V1_CHECKPOINT" training-consistent prevent 1
run_evaluation paper-faithful-v2 paper-faithful "$PAPER_V2_CHECKPOINT" training-consistent prevent 1
run_evaluation oracle-k oracle-k "$ORACLE_K_CHECKPOINT" training-consistent prevent 1
run_evaluation qwen3-vl-8b-base baseline "$BASE_MODEL" official allow 0

if (( DRY_RUN )); then
  echo "Five-model dry-run completed; no download, model verification, inference, or judge call was started."
else
  echo "All five model evaluations completed and validated."
fi
