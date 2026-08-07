#!/usr/bin/env bash

set -eEuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LKL_8GPU_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$LKL_8GPU_ROOT/../.." && pwd)"
COLT_LAUNCHER="$LKL_8GPU_ROOT/colt.sh"
EXTERNAL_JUDGE_LAUNCHER="$SCRIPT_DIR/run.sh"

CHECKPOINT_ROOT="${COLT_OUTPUT_ROOT:-$REPO_ROOT/checkpoints}"
CHECKPOINT="${COLT_ORACLE_K_CHECKPOINT:-$CHECKPOINT_ROOT/colt_oracle_k_predictor_batch_285190c}"
CONDA_ENV="${COLT_CONDA_ENV_DIR:-/data/nvme0/lkl/conda/envs/colt}"
CONDA_INIT_SH="${COLT_CONDA_INIT_SH:-/data/nvme0/lkl/miniconda3/etc/profile.d/conda.sh}"

GPU_CSV="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${VLMEVAL_WORKERS_PER_GPU:-3}"
API_NPROC="${COLT_EXTERNAL_JUDGE_NPROC:-8}"
JUDGE_RETRY="${COLT_EXTERNAL_JUDGE_RETRY:-5}"
RESULT_ROOT="${COLT_ORACLE_K_FIXED_K_RESULT_ROOT:-$REPO_ROOT/eval/external_judge/oracle-k-fixed-k-sweep}"
LOG_ROOT="${COLT_ORACLE_K_FIXED_K_LOG_ROOT:-$REPO_ROOT/logs/eval/external_judge/oracle-k-fixed-k-sweep}"
K_VALUES=(1 2 3 4 5 6 7 8)
NO_REUSE=0
DRY_RUN=0
CURRENT_STAGE=preflight

usage() {
  cat <<'EOF'
Evaluate the Oracle-K checkpoint at each forced K on MathVista_MINI,
MathVerse_MINI, and MMVet with the configured external judge.

Usage:
  bash run_oracle_k_fixed_k_sweep.sh [--k-values LIST] [--model-path PATH]
                                      [--gpus CSV] [--workers N] [--api-nproc N]
                                      [--judge-retry N] [--output-root PATH]
                                      [--log-root PATH] [--no-reuse] [--dry-run]

Defaults:
  forced K values: 1,2,3,4,5,6,7,8
  GPUs: 0,1,2,3,4,5,6,7
  inference workers per GPU: 3
  judge API concurrency: 8
  judge retry count: 5

The runs are sequential. Each K has an isolated result/log root and uses the
same protocol as the completed five-model suite: greedy 8192-token generation,
training-consistent latent transitions, prevention of empty answers, and
per-sample reseeding. The external-judge validator runs after every K.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while (( $# > 0 )); do
  case "$1" in
    --k-values) [[ $# -ge 2 ]] || die "--k-values requires a value"; read -r -a K_VALUES <<< "${2//,/ }"; shift 2 ;;
    --model-path) [[ $# -ge 2 ]] || die "--model-path requires a value"; CHECKPOINT="$2"; shift 2 ;;
    --gpus) [[ $# -ge 2 ]] || die "--gpus requires a value"; GPU_CSV="$2"; shift 2 ;;
    --workers) [[ $# -ge 2 ]] || die "--workers requires a value"; WORKERS_PER_GPU="$2"; shift 2 ;;
    --api-nproc) [[ $# -ge 2 ]] || die "--api-nproc requires a value"; API_NPROC="$2"; shift 2 ;;
    --judge-retry) [[ $# -ge 2 ]] || die "--judge-retry requires a value"; JUDGE_RETRY="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || die "--output-root requires a value"; RESULT_ROOT="$2"; shift 2 ;;
    --log-root) [[ $# -ge 2 ]] || die "--log-root requires a value"; LOG_ROOT="$2"; shift 2 ;;
    --no-reuse) NO_REUSE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ "$CHECKPOINT" == /* ]] || die "Model path must be absolute: $CHECKPOINT"
[[ -d "$CHECKPOINT" ]] || die "Model directory does not exist: $CHECKPOINT"
[[ "$CONDA_ENV" == /* && -x "$CONDA_ENV/bin/python" ]] || die \
  "Missing colt Conda environment: $CONDA_ENV"
[[ "$RESULT_ROOT" == /* ]] || die "--output-root must be absolute: $RESULT_ROOT"
[[ "$LOG_ROOT" == /* ]] || die "--log-root must be absolute: $LOG_ROOT"
[[ "$GPU_CSV" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpus must be a comma-separated integer list"
[[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "--workers must be a positive integer"
[[ "$API_NPROC" =~ ^[1-9][0-9]*$ ]] || die "--api-nproc must be a positive integer"
[[ "$JUDGE_RETRY" =~ ^[1-9][0-9]*$ ]] || die "--judge-retry must be a positive integer"
(( ${#K_VALUES[@]} > 0 )) || die "--k-values must contain at least one value"

declare -A seen_k_values=()
for k in "${K_VALUES[@]}"; do
  [[ "$k" =~ ^[1-8]$ ]] || die "Each K must be an integer in [1, 8], got: $k"
  [[ -z "${seen_k_values[$k]:-}" ]] || die "Duplicate K value: $k"
  seen_k_values[$k]=1
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
  trap 'status=$?; echo "Oracle-K fixed-K sweep failed at stage=$CURRENT_STAGE exit=$status" >&2; exit "$status"' ERR
fi

echo "Repository: $REPO_ROOT"
echo "Checkpoint: $CHECKPOINT"
echo "Datasets: MathVista_MINI MathVerse_MINI MMVet"
echo "Forced K values: ${K_VALUES[*]}"
echo "Physical GPUs: $GPU_CSV"
echo "Inference workers per GPU: $WORKERS_PER_GPU"
echo "Judge API concurrency: $API_NPROC"
echo "Judge retry count: $JUDGE_RETRY"
echo "Result root: $RESULT_ROOT"
echo "Log root: $LOG_ROOT"

CURRENT_STAGE=download-and-verify-datasets
run_command env "${common_environment[@]}" bash "$EXTERNAL_JUDGE_LAUNCHER" download \
  MathVista_MINI MathVerse_MINI MMVet

CURRENT_STAGE=preflight-oracle-k
run_command env -u COLT_INFERENCE_K -u COLT_INFERENCE_TRANSITION_STEPS \
  "${common_environment[@]}" bash "$COLT_LAUNCHER" verify model oracle-k --model-path "$CHECKPOINT"

for k in "${K_VALUES[@]}"; do
  CURRENT_STAGE="evaluate-fixed-k$k"
  echo "Evaluating Oracle-K with forced K=$k"
  eval_command=(
    env -u COLT_INFERENCE_TRANSITION_STEPS
    "${common_environment[@]}"
    "COLT_INFERENCE_K=$k"
    "COLT_LOG_PREDICTED_K=1"
    "COLT_LOG_ORACLE_K_PLAN=1"
    "COLT_EVAL_LOG_LABEL=oracle-k-fixed-k$k"
    "COLT_EXTERNAL_JUDGE_OUTPUT_ROOT=$RESULT_ROOT/k$k"
    "COLT_EXTERNAL_JUDGE_LOG_ROOT=$LOG_ROOT/k$k"
    bash "$EXTERNAL_JUDGE_LAUNCHER" eval oracle-k
    --datasets MathVista_MINI,MathVerse_MINI,MMVet
    --model-path "$CHECKPOINT"
    --gpus "$GPU_CSV"
    --workers "$WORKERS_PER_GPU"
    --generation respect-args
    --latent-transition training-consistent
    --prefetch 1
    --empty-cache-every 0
    --dist-backend gloo
    --reseed-per-sample 1
    --empty-response-policy prevent
    --api-nproc "$API_NPROC"
    --judge-retry "$JUDGE_RETRY"
  )
  (( NO_REUSE == 0 )) || eval_command+=(--no-reuse)
  run_command "${eval_command[@]}"
done

if (( DRY_RUN )); then
  echo "Oracle-K fixed-K sweep dry-run completed; no download, model verification, inference, or judge call was started."
else
  echo "All Oracle-K fixed-K evaluations completed and validated."
fi
