#!/usr/bin/env bash
# Latent-usage intervention experiment on the fixed-v2 paper-faithful checkpoint.
#
# For each intervention mode (none/zero/random/skip) it evaluates the same
# read-image (ChartQA+TextVQA) and reasoning (MMStar) groups under the same
# greedy, prevent-empty settings used by the previous all8 run, so all modes
# are directly comparable and sampling noise is removed.
#
# Usage:
#   nohup bash scripts/lkl_8gpu/experiments/run_latent_intervention.sh \
#     > /home/dataset-local/lkl/logs/latent_intervention_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

COLT_ENV_FILE="${COLT_ENV_FILE:-/home/dataset-local/lkl/colt-local.env}"
CONDA_PROFILE="${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
REPO="${COLT_REPO:-/home/dataset-local/lkl/CoLT-reproduction}"
MODEL="${COLT_INTERVENTION_MODEL:-$REPO/checkpoints/colt_paper_faithful_v2}"
GPU_CSV="${COLT_INTERVENTION_GPUS:-0,1,2,3,4,5,6,7}"
read -r -a INTERVENTION_MODES <<< "${COLT_INTERVENTION_MODES:-none zero random skip}"
read -r -a EVAL_GROUPS <<< "${COLT_INTERVENTION_GROUPS:-chart-text mmstar}"

source "$COLT_ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"

echo "Experiment started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Repo: $REPO"
echo "Model: $MODEL"
echo "GPUs: $GPU_CSV"
echo "Modes: ${INTERVENTION_MODES[*]}"
echo "Groups: ${EVAL_GROUPS[*]}"

for mode in "${INTERVENTION_MODES[@]}"; do
  echo "===== MODE=$mode started at $(date '+%H:%M:%S') ====="
  for group in "${EVAL_GROUPS[@]}"; do
    echo "----- MODE=$mode GROUP=$group started at $(date '+%H:%M:%S') -----"
    COLT_LATENT_INTERVENTION="$mode" \
      bash scripts/lkl_8gpu/colt.sh eval paper-faithful "$group" \
      --model-path "$MODEL" \
      --gpus "$GPU_CSV" \
      --generation respect-args \
      --empty-response-policy prevent
    echo "----- MODE=$mode GROUP=$group finished at $(date '+%H:%M:%S') -----"
  done
  echo "===== MODE=$mode finished at $(date '+%Y-%m-%d %H:%M:%S') ====="
done

echo "All intervention modes completed at $(date '+%Y-%m-%d %H:%M:%S')"
