#!/usr/bin/env bash

set -euo pipefail

profile="${1:-${COLT_GPU_PROFILE:-}}"
if [[ "$profile" != "a100" && "$profile" != "a800" ]]; then
  echo "Usage: bash scripts/lkl_8gpu/00_verify_host.sh {a100|a800}" >&2
  exit 2
fi
export COLT_GPU_PROFILE="$profile"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

required_commands=(git curl unzip tmux nvidia-smi)
missing_commands=()
for command_name in "${required_commands[@]}"; do
  command -v "$command_name" >/dev/null 2>&1 || missing_commands+=("$command_name")
done
if (( ${#missing_commands[@]} > 0 )); then
  echo "Missing host commands: ${missing_commands[*]}" >&2
  exit 1
fi

require_workspace_layout
validate_gpu_profile

if [[ ! -x "$MINICONDA_ROOT/bin/conda" ]]; then
  echo "Miniconda is missing: $MINICONDA_ROOT/bin/conda" >&2
  exit 1
fi
if [[ ! -w "$LKL_ROOT" ]]; then
  echo "LKL root is not writable: $LKL_ROOT" >&2
  exit 1
fi

profile_tmp="$PROFILE_FILE.tmp.$$"
printf '%s\n' "$profile" > "$profile_tmp"
mv "$profile_tmp" "$PROFILE_FILE"

echo "GPU profile: $profile ($COLT_EXPECTED_GPU_NAME)"
echo "Repository: $REPO_ROOT"
echo "Miniconda: $MINICONDA_ROOT"
echo "Conda environment: $CONDA_ENV_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
df -h "$LKL_ROOT"
echo "Host verification passed; saved profile to $PROFILE_FILE"
