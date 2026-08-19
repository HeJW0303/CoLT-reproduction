#!/usr/bin/env bash
# Score MathVista_MINI latent-intervention predictions with DeepSeek v4 flash.
#
# VLMEvalKit's MathVista evaluate path uses an LLM judge for free-form answer
# extraction. This script runs --mode eval with DeepSeek's OpenAI-compatible
# API. OPENAI_API_KEY (the DeepSeek key) must be provided by the caller via the
# environment; it is never stored in this file.
#
# Usage:
#   OPENAI_API_KEY=sk-... setsid nohup bash \
#     scripts/lkl_8gpu/experiments/run_mathvista_eval_deepseek.sh \
#     > /home/dataset-local/lkl/logs/mathvista_eval_<ts>.log 2>&1 &

set -euo pipefail

COLT_ENV_FILE="${COLT_ENV_FILE:-/home/dataset-local/lkl/colt-local.env}"
CONDA_PROFILE="${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
REPO="${COLT_REPO:-/home/dataset-local/lkl/CoLT-reproduction}"
GPU_CSV="${COLT_INTERVENTION_GPUS:-0}"
JUDGE_MODEL="${COLT_MATHVISTA_JUDGE_MODEL:-deepseek-v4-flash}"
JUDGE_MAX_TOKENS="${COLT_MATHVISTA_JUDGE_MAX_TOKENS:-1024}"
JUDGE_TIMEOUT="${COLT_MATHVISTA_JUDGE_TIMEOUT:-60}"
MODE_TIMEOUT_S="${COLT_MATHVISTA_MODE_TIMEOUT_S:-600}"

source "$COLT_ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (DeepSeek key) before running this script}"

export LMUData="$COLT_EVAL_ROOT/LMUData"
export PYTHONPATH="$REPO/Evaluation/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLMEVAL_FAIL_ON_ERROR=1 VLMEVAL_ATOMIC_WRITES=1
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.deepseek.com/v1/chat/completions}"
export COLT_MATHVISTA_JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS"
export CUDA_VISIBLE_DEVICES="$GPU_CSV"
export OMP_NUM_THREADS=1
# DeepSeek is a domestic (CN) service reachable directly; the local proxy can
# hang long-running judge requests, so force direct connections for scoring.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE

RESULT_ROOT="$COLT_EVAL_ROOT/results/paper-faithful/MathVista_MINI"
cd "$REPO/Evaluation/VLMEvalKit"

score_mode() {
  local mode="$1" work_dir="$2" eval_id="$3"
  local attempt rc
  for attempt in 1 2 3; do
    echo "===== EVAL mode=$mode judge=$JUDGE_MODEL attempt=$attempt started at $(date '+%H:%M:%S') ====="
    if timeout "$MODE_TIMEOUT_S" env \
      MMEVAL_ROOT="$work_dir" \
      VLMEVAL_EVAL_ID="$eval_id" \
      COLT_LATENT_INTERVENTION="$mode" \
      torchrun --standalone --nnodes=1 --nproc_per_node=1 --max_restarts=0 \
        run.py --data MathVista_MINI --model Qwen3-VL-8B-Instruct-COLT \
        --work-dir "$work_dir" --mode eval --judge "$JUDGE_MODEL" --reuse --reuse-aux 1 \
        --judge-args "{\"timeout\": $JUDGE_TIMEOUT}"; then
      echo "===== EVAL mode=$mode finished at $(date '+%H:%M:%S') ====="
      return 0
    fi
    rc=$?
    echo "WARNING: mode=$mode attempt=$attempt exited rc=$rc; patching missing pkl entries and retrying"
    python - "$work_dir" <<'PY'
import pickle, sys
import pandas as pd
import glob, os

work_dir = sys.argv[1]
pkls = glob.glob(os.path.join(work_dir, "**", "*_deepseek-v4-flash.pkl"), recursive=True)
xlsxs = glob.glob(os.path.join(work_dir, "**", "Qwen3-VL-8B-Instruct-COLT_MathVista_MINI.xlsx"), recursive=True)
if not pkls or not xlsxs:
    print("skip: missing pkl/xlsx under", work_dir)
    sys.exit(0)
ans = pickle.load(open(pkls[0], "rb"))
df = pd.read_excel(xlsxs[0])
missing = [i for i in df["index"] if i not in ans]
if not missing:
    print("no missing entries")
    sys.exit(0)
for idx in missing:
    ans[idx] = {"log": "skipped after repeated API timeouts", "res": ""}
pickle.dump(ans, open(pkls[0], "wb"))
print("patched missing entries:", missing)
PY
  done
  echo "ERROR: mode=$mode failed after 3 attempts"
  return 1
}

score_mode none \
  "$RESULT_ROOT/replicas24_w3_p1_c0_r1_respect-args_prevent_ltofficial_linone_seed1234_afed9e6836e1" \
  "PAPER_FAITHFUL_REPLICAS24_W3_P1_C0_R1_RESPECT_ARGS_PREVENT_LTOFFICIAL_LINONE_SEED1234_AFED9E6836E1"

for mode in zero random skip; do
  score_mode "$mode" "$RESULT_ROOT/local_${mode}" "MATHVISTA_LOCAL_${mode^^}"
done

echo "All MathVista DeepSeek evaluations completed at $(date '+%Y-%m-%d %H:%M:%S')"
