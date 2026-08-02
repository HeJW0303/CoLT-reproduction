#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/runtime.sh"
source "$SCRIPT_DIR/lib/gpu.sh"
source "$SCRIPT_DIR/lib/datasets.sh"
source "$SCRIPT_DIR/lib/model.sh"
source "$SCRIPT_DIR/commands/setup.sh"
source "$SCRIPT_DIR/commands/train.sh"
source "$SCRIPT_DIR/commands/eval.sh"

usage() {
  cat <<'EOF'
CoLT single-node 8-GPU unified launcher

Usage:
  colt.sh profile [generic|a100|a800|LABEL]
  colt.sh setup {env|assets|data|eval|all}
  colt.sh verify {ready|nccl|model TARGET [--model-path PATH]}
  colt.sh train {codefaithful|paper-faithful|oracle-k} [--resume] [--batch-aux]
  colt.sh download {all8|remaining7|phase1|phase2|phase3|DATASET...}
  colt.sh eval TARGET GROUP [options]

Evaluation targets:
  codefaithful     checkpoints/colt_codefaithful
  paper-faithful  checkpoints/colt_paper_faithful
  oracle-k        checkpoints/colt_oracle_k_predictor
  official        local hulianyuyy/CoLT-8B snapshot
  baseline        Qwen3-VL-8B-Instruct textual-CoT baseline

Training options:
  --resume                    Resume a verified interrupted checkpoint
  --batch-aux                 Opt in to auxiliary decoder batching (default: off)

COLT_BATCH_AUX_DECODERS=1 is equivalent to --batch-aux for paper-faithful
and oracle-k training.
COLT_AUX_MAX_BATCH_TOKENS sets the auxiliary decoder padded-token budget
per call when batching is enabled (default: 4096).

Runtime:
  Activate the prepared environment before running commands:
    conda activate <your-env>
  The launcher reuses the active environment and does not require its path.
  GPU idle-memory and strict disk checks are opt-in via
  COLT_CHECK_GPU_FREE=1 or COLT_STRICT_PREFLIGHT=1.

Dataset groups:
  smoke, chartqa, mmstar, phase1, phase2, phase3, remaining7, all8

Evaluation options:
  --model-path PATH           Explicit model path (highest priority)
  --gpus CSV                  Physical GPUs, e.g. 4,5,6,7
  --workers N                 Model replicas per GPU (default: 3)
  --generation MODE           official or respect-args
  --latent-transition MODE    official or training-consistent
  --prefetch {0|1}            CPU image preprocessing prefetch
  --empty-cache-every N       0 disables per-sample empty_cache
  --dist-backend {gloo|nccl}
  --reseed-per-sample {0|1}
  --empty-response-policy {allow|prevent}
  --verbose
  --no-reuse

Model path precedence is always:
  --model-path > COLT_EVAL_MODEL_PATH > target default
EOF
}

command_name="${1:-help}"
shift || true
case "$command_name" in
  help|-h|--help) usage ;;
  profile) cmd_profile "$@" ;;
  setup) cmd_setup "$@" ;;
  verify) cmd_verify "$@" ;;
  train) cmd_train "$@" ;;
  download) cmd_download "$@" ;;
  eval) cmd_eval "$@" ;;
  *) usage >&2; die "Unknown command: $command_name" ;;
esac
