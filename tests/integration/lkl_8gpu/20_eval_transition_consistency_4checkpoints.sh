#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAUNCHER="$REPO_ROOT/scripts/lkl_8gpu/colt.sh"

RUNTIME_ROOT="${COLT_RUNTIME_ROOT:-$REPO_ROOT}"
CHECKPOINT_ROOT="${COLT_OUTPUT_ROOT:-$RUNTIME_ROOT/checkpoints}"
EVAL_ROOT="${COLT_EVAL_ROOT:-$RUNTIME_ROOT/eval}"
GPU_CSV="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${VLMEVAL_WORKERS_PER_GPU:-3}"
GROUP=all8
DRY_RUN=0

CODEFAITHFUL_CHECKPOINT="${COLT_CODEFAITHFUL_CHECKPOINT:-$CHECKPOINT_ROOT/colt_codefaithful}"
PAPER_V1_CHECKPOINT="${COLT_PAPER_V1_CHECKPOINT:-$CHECKPOINT_ROOT/colt_paper_faithful_v1}"
PAPER_V2_CHECKPOINT="${COLT_PAPER_V2_CHECKPOINT:-$CHECKPOINT_ROOT/colt_paper_faithful_v2}"
ORACLE_K_CHECKPOINT="${COLT_ORACLE_K_CHECKPOINT:-$CHECKPOINT_ROOT/colt_oracle_k_predictor_batch_285190c}"
RESULT_ROOT="${COLT_TRANSITION_EVAL_OUTPUT_ROOT:-$EVAL_ROOT/results-transition-consistent}"

usage() {
  cat <<'EOF'
Evaluate four existing CoLT checkpoints with the training-consistent latent transition.

Usage:
  bash tests/integration/lkl_8gpu/20_eval_transition_consistency_4checkpoints.sh [--group GROUP] [--dry-run]

Defaults:
  GROUP=all8
  GPUs=0,1,2,3,4,5,6,7
  workers per GPU=3
  generation=greedy + 8192
  empty-response policy=prevent
  latent transition=training-consistent

Checkpoint paths can be overridden with COLT_CODEFAITHFUL_CHECKPOINT,
COLT_PAPER_V1_CHECKPOINT, COLT_PAPER_V2_CHECKPOINT, and
COLT_ORACLE_K_CHECKPOINT.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while (( $# > 0 )); do
  case "$1" in
    --group) [[ $# -ge 2 ]] || die "--group requires a value"; GROUP="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ -x "$LAUNCHER" ]] || die "Missing executable launcher: $LAUNCHER"
[[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "VLMEVAL_WORKERS_PER_GPU must be positive"

targets=(codefaithful paper-faithful paper-faithful oracle-k)
labels=(
  codefaithful-transition-consistent
  paper-faithful-v1-transition-consistent
  paper-faithful-v2-transition-consistent
  oracle-k-transition-consistent
)
checkpoints=(
  "$CODEFAITHFUL_CHECKPOINT"
  "$PAPER_V1_CHECKPOINT"
  "$PAPER_V2_CHECKPOINT"
  "$ORACLE_K_CHECKPOINT"
)

for checkpoint in "${checkpoints[@]}"; do
  [[ "$checkpoint" == /* ]] || die "Checkpoint path must be absolute: $checkpoint"
  [[ -d "$checkpoint" ]] || die "Checkpoint directory does not exist: $checkpoint"
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

unset COLT_INFERENCE_K

echo "Repository: $REPO_ROOT"
echo "Git HEAD: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Dataset group: $GROUP"
echo "Physical GPUs: $GPU_CSV"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "Inference latent transition: training-consistent"
echo "Results root: $RESULT_ROOT"

echo "Preflighting all checkpoints before starting the first evaluation."
for index in "${!targets[@]}"; do
  run_command bash "$LAUNCHER" verify model "${targets[$index]}" \
    --model-path "${checkpoints[$index]}"
done

for index in "${!targets[@]}"; do
  target="${targets[$index]}"
  label="${labels[$index]}"
  checkpoint="${checkpoints[$index]}"
  echo "Evaluating $label from $checkpoint"
  run_command env \
    "COLT_EVAL_LOG_LABEL=$label" \
    "COLT_EVAL_OUTPUT_ROOT=$RESULT_ROOT" \
    bash "$LAUNCHER" eval "$target" "$GROUP" \
      --model-path "$checkpoint" \
      --gpus "$GPU_CSV" \
      --workers "$WORKERS_PER_GPU" \
      --generation respect-args \
      --latent-transition training-consistent \
      --prefetch 1 \
      --empty-cache-every 0 \
      --reseed-per-sample 1 \
      --empty-response-policy prevent
done

if (( DRY_RUN == 1 )); then
  echo "Transition-consistency dry-run completed; no evaluation was started."
else
  echo "All four transition-consistent evaluations completed and validated."
fi
