#!/usr/bin/env bash
# MathVista_MINI latent-intervention inference (infer-only, local scoring).
#
# VLMEvalKit's MathVista evaluate path requires OPENAI_API_KEY for free-form
# extraction, so we run --mode infer for zero/random/skip (none was already
# inferred) and score all four modes locally with score_mathvista_local.py.

set -euo pipefail

COLT_ENV_FILE="${COLT_ENV_FILE:-/home/dataset-local/lkl/colt-local.env}"
CONDA_PROFILE="${CONDA_PROFILE:-/opt/conda/etc/profile.d/conda.sh}"
REPO="${COLT_REPO:-/home/dataset-local/lkl/CoLT-reproduction}"
MODEL="${COLT_INTERVENTION_MODEL:-$REPO/checkpoints/colt_paper_faithful_v2}"
GPU_CSV="${COLT_INTERVENTION_GPUS:-0,1,2,3,4,5,6,7}"
read -r -a MODES <<< "${COLT_INTERVENTION_MODES:-zero random skip}"

source "$COLT_ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"

export LMUData="$COLT_EVAL_ROOT/LMUData"
export PYTHONPATH="$REPO/Evaluation/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLMEVAL_FAIL_ON_ERROR=1 VLMEVAL_ATOMIC_WRITES=1
export COLT_DECODER_MODEL_PATH="$COLT_DECODER_MODEL_DIR"
export COLT_INFERENCE_LATENT_TRANSITION=official
export COLT_RESPECT_GENERATION_ARGS=1
export COLT_PREVENT_EMPTY_RESPONSE=1
export COLT_EVAL_MODEL_PATH="$MODEL"
export CUDA_VISIBLE_DEVICES="$GPU_CSV"
export VLMEVAL_WORKERS_PER_GPU=3 VLMEVAL_PREFETCH=1 VLMEVAL_EMPTY_CACHE_EVERY_N=0
export VLMEVAL_DIST_BACKEND=gloo
export COLT_RESEED_PER_SAMPLE=1 COLT_EVAL_SEED=1234
export OMP_NUM_THREADS=1
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE

RESULT_ROOT="$COLT_EVAL_ROOT/results/paper-faithful/MathVista_MINI"
cd "$REPO/Evaluation/VLMEvalKit"

for mode in "${MODES[@]}"; do
  work_dir="$RESULT_ROOT/local_${mode}"
  echo "===== INFER mode=$mode started at $(date '+%H:%M:%S') ====="
  COLT_LATENT_INTERVENTION="$mode" \
    MMEVAL_ROOT="$work_dir" \
    VLMEVAL_EVAL_ID="MATHVISTA_LOCAL_${mode^^}" \
    torchrun --standalone --nnodes=1 --nproc_per_node=24 --max_restarts=0 \
      run.py --data MathVista_MINI --model Qwen3-VL-8B-Instruct-COLT \
      --work-dir "$work_dir" --mode infer
  echo "===== INFER mode=$mode finished at $(date '+%H:%M:%S') ====="
done

echo "===== Local scoring ====="
find "$RESULT_ROOT" -name 'Qwen3-VL-8B-Instruct-COLT_MathVista_MINI.xlsx' -print0 \
  | xargs -0 python "$REPO/scripts/lkl_8gpu/tools/score_mathvista_local.py"
echo "All inference modes completed at $(date '+%Y-%m-%d %H:%M:%S')"
