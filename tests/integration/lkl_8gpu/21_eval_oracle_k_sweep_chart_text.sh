#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAUNCHER="$REPO_ROOT/scripts/lkl_8gpu/colt.sh"

RUNTIME_ROOT="${COLT_RUNTIME_ROOT:-$REPO_ROOT}"
CHECKPOINT_ROOT="${COLT_OUTPUT_ROOT:-$RUNTIME_ROOT/checkpoints}"
EVAL_ROOT="${COLT_EVAL_ROOT:-$RUNTIME_ROOT/eval}"
LOG_ROOT="${COLT_LOG_ROOT:-$RUNTIME_ROOT/logs}"
CONDA_ENV_DIR="${COLT_CONDA_ENV_DIR:-/data/nvme0/lkl/conda/envs/colt}"

CHECKPOINT="${COLT_ORACLE_K_CHECKPOINT:-$CHECKPOINT_ROOT/colt_oracle_k_predictor_batch_285190c}"
OUTPUT_ROOT="${COLT_K_SWEEP_OUTPUT_ROOT:-$EVAL_ROOT/results/oracle-k-k-sweep}"
EVAL_LOG_ROOT="${COLT_K_SWEEP_LOG_ROOT:-$LOG_ROOT/eval/oracle-k-k-sweep}"
GPU_CSV="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${VLMEVAL_WORKERS_PER_GPU:-3}"
K_VALUES=(1 2 4 5 6 7 8)
NO_REUSE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Evaluate one Oracle-K checkpoint on ChartQA and TextVQA with forced K values.

Usage:
  bash tests/integration/lkl_8gpu/21_eval_oracle_k_sweep_chart_text.sh [options]

Options:
  --k-values LIST    Comma- or space-separated K values in [1, 8] (default: 1,2,4,5,6,7,8)
  --model-path PATH  Oracle-K checkpoint path
  --gpus CSV         Physical GPU IDs (default: 0,1,2,3,4,5,6,7)
  --workers N        Model replicas per GPU (default: 3)
  --conda-env PATH   Conda environment (default: /data/nvme0/lkl/conda/envs/colt)
  --output-root PATH Base result directory; each K uses PATH/kK
  --log-root PATH    Base log directory; each K uses PATH/kK
  --no-reuse         Disable VLMEvalKit result reuse
  --dry-run          Validate and print commands without starting evaluation
  -h, --help         Show this help

Fixed evaluation protocol:
  datasets=ChartQA_TEST,TextVQA_VAL
  generation=greedy, max_new_tokens=8192
  empty-response-policy=prevent
  latent-transition=official
  reseed-per-sample=1

Environment overrides:
  COLT_ORACLE_K_CHECKPOINT, COLT_K_SWEEP_OUTPUT_ROOT, COLT_K_SWEEP_LOG_ROOT,
  COLT_EVAL_GPUS, VLMEVAL_WORKERS_PER_GPU, COLT_CONDA_ENV_DIR
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while (( $# > 0 )); do
  case "$1" in
    --k-values)
      [[ $# -ge 2 ]] || die "--k-values requires a value"
      read -r -a K_VALUES <<< "${2//,/ }"
      shift 2
      ;;
    --model-path) [[ $# -ge 2 ]] || die "--model-path requires a value"; CHECKPOINT="$2"; shift 2 ;;
    --gpus) [[ $# -ge 2 ]] || die "--gpus requires a value"; GPU_CSV="$2"; shift 2 ;;
    --workers) [[ $# -ge 2 ]] || die "--workers requires a value"; WORKERS_PER_GPU="$2"; shift 2 ;;
    --conda-env) [[ $# -ge 2 ]] || die "--conda-env requires a value"; CONDA_ENV_DIR="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || die "--output-root requires a value"; OUTPUT_ROOT="$2"; shift 2 ;;
    --log-root) [[ $# -ge 2 ]] || die "--log-root requires a value"; EVAL_LOG_ROOT="$2"; shift 2 ;;
    --no-reuse) NO_REUSE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ -x "$LAUNCHER" ]] || die "Missing executable launcher: $LAUNCHER"
[[ "$CONDA_ENV_DIR" == /* ]] || die "Conda environment path must be absolute: $CONDA_ENV_DIR"
[[ -x "$CONDA_ENV_DIR/bin/python" ]] || die "Conda environment does not provide Python: $CONDA_ENV_DIR"
[[ "$CHECKPOINT" == /* ]] || die "Checkpoint path must be absolute: $CHECKPOINT"
[[ -d "$CHECKPOINT" ]] || die "Checkpoint directory does not exist: $CHECKPOINT"
[[ "$OUTPUT_ROOT" == /* ]] || die "Output root must be absolute: $OUTPUT_ROOT"
[[ "$EVAL_LOG_ROOT" == /* ]] || die "Log root must be absolute: $EVAL_LOG_ROOT"
[[ "$GPU_CSV" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpus must be a comma-separated list of GPU IDs"
[[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "--workers must be a positive integer"
(( ${#K_VALUES[@]} > 0 )) || die "--k-values must contain at least one value"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  active_conda_prefix="$(cd "$CONDA_PREFIX" && pwd -P)"
  expected_conda_prefix="$(cd "$CONDA_ENV_DIR" && pwd -P)"
  [[ "$active_conda_prefix" == "$expected_conda_prefix" ]] || die \
    "Active Conda environment is $active_conda_prefix; expected $expected_conda_prefix"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  die "Active virtual environment would override the required Conda environment: $VIRTUAL_ENV"
fi

declare -A seen_gpu_ids=()
IFS=',' read -r -a gpu_ids <<< "$GPU_CSV"
for gpu_id in "${gpu_ids[@]}"; do
  [[ -z "${seen_gpu_ids[$gpu_id]:-}" ]] || die "Duplicate GPU ID: $gpu_id"
  seen_gpu_ids[$gpu_id]=1
done

declare -A seen_k_values=()
for k in "${K_VALUES[@]}"; do
  [[ "$k" =~ ^[1-8]$ ]] || die "Each K must be an integer in [1, 8], got: $k"
  [[ -z "${seen_k_values[$k]:-}" ]] || die "Duplicate K value: $k"
  seen_k_values[$k]=1
done

print_command() {
  printf 'DRY RUN:'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  if (( DRY_RUN == 1 )); then
    print_command "$@"
  else
    "$@"
  fi
}

echo "Repository: $REPO_ROOT"
echo "Git HEAD: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Checkpoint: $CHECKPOINT"
echo "Conda environment: $CONDA_ENV_DIR"
echo "Datasets: ChartQA_TEST TextVQA_VAL"
echo "Forced K values: ${K_VALUES[*]}"
echo "Physical GPUs: $GPU_CSV"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "Generation: greedy, max_new_tokens=8192"
echo "Empty response policy: prevent"
echo "Inference latent transition: official"
echo "Result base: $OUTPUT_ROOT"
echo "Log base: $EVAL_LOG_ROOT"

echo "Preflighting the Oracle-K checkpoint before starting the first evaluation."
run_command env "COLT_CONDA_ENV_DIR=$CONDA_ENV_DIR" \
  bash "$LAUNCHER" verify model oracle-k --model-path "$CHECKPOINT"

for k in "${K_VALUES[@]}"; do
  k_output_root="$OUTPUT_ROOT/k$k"
  k_log_root="$EVAL_LOG_ROOT/k$k"
  log_label="oracle-k-fixed-k$k"
  eval_command=(
    env
    "COLT_CONDA_ENV_DIR=$CONDA_ENV_DIR"
    "COLT_INFERENCE_K=$k"
    "COLT_LOG_PREDICTED_K=1"
    "COLT_EVAL_OUTPUT_ROOT=$k_output_root"
    "COLT_EVAL_LOG_ROOT=$k_log_root"
    "COLT_EVAL_LOG_LABEL=$log_label"
    bash "$LAUNCHER" eval oracle-k chart-text
    --model-path "$CHECKPOINT"
    --gpus "$GPU_CSV"
    --workers "$WORKERS_PER_GPU"
    --generation respect-args
    --latent-transition official
    --prefetch 1
    --empty-cache-every 0
    --dist-backend gloo
    --reseed-per-sample 1
    --empty-response-policy prevent
  )
  (( NO_REUSE == 0 )) || eval_command+=(--no-reuse)

  echo "Evaluating forced K=$k"
  run_command "${eval_command[@]}"
done

if (( DRY_RUN == 1 )); then
  echo "Oracle-K sweep dry-run completed; no evaluation was started."
else
  echo "Oracle-K sweep completed and validated for K values: ${K_VALUES[*]}"
fi
