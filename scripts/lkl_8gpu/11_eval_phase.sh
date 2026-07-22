#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"

phase="${1:-phase1}"
case "$phase" in
  phase1)
    datasets=(ChartQA_TEST MMStar MMBench_DEV_EN)
    ;;
  phase2)
    datasets=(TextVQA_VAL MMT-Bench_VAL)
    ;;
  phase3)
    datasets=(SEEDBench_IMG ScienceQA_TEST AI2D_TEST)
    ;;
  *)
    echo "Usage: $0 phase1|phase2|phase3" >&2
    exit 1
    ;;
esac

IFS=',' read -r -a gpu_ids <<< "${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
if (( ${#gpu_ids[@]} < ${#datasets[@]} )); then
  echo "Not enough GPU ids in COLT_EVAL_GPUS for ${#datasets[@]} datasets." >&2
  exit 1
fi

bash "$REPO_ROOT/scripts/lkl_8gpu/09_download_eval_data.sh" "$phase"

pids=()
for index in "${!datasets[@]}"; do
  dataset="${datasets[$index]}"
  gpu="${gpu_ids[$index]}"
  echo "Launching $dataset on physical GPU $gpu"
  bash "$REPO_ROOT/scripts/lkl_8gpu/eval_one.sh" "$dataset" "$gpu" &
  pids+=("$!")
  if (( index + 1 < ${#datasets[@]} )); then
    sleep "${COLT_EVAL_STAGGER_SECONDS:-60}"
  fi
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

python "$REPO_ROOT/scripts/lkl_8gpu/summarize_colt_eval.py" "$EVAL_OUTPUT_ROOT/codefaithful"
exit "$status"
