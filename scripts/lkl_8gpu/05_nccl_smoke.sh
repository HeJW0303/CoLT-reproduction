#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env
validate_gpu_profile

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

torchrun --standalone --nproc_per_node=8 "$REPO_ROOT/scripts/lkl_8gpu/nccl_smoke.py"
echo "Eight-rank NCCL all-reduce smoke test passed."
