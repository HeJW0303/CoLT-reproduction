#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout
validate_gpu_profile

required_commands=(git curl unzip tmux nvidia-smi)
missing_commands=()
for command_name in "${required_commands[@]}"; do
  command -v "$command_name" >/dev/null 2>&1 || missing_commands+=("$command_name")
done
if (( ${#missing_commands[@]} > 0 )); then
  echo "Missing host commands: ${missing_commands[*]}" >&2
  echo "Ask the server administrator to install them, then rerun this script." >&2
  exit 1
fi

conda_sh="$MINICONDA_ROOT/etc/profile.d/conda.sh"
if [[ ! -f "$conda_sh" || ! -x "$MINICONDA_ROOT/bin/python" ]]; then
  echo "Miniconda is incomplete under: $MINICONDA_ROOT" >&2
  exit 1
fi

# Finder metadata is binary AppleDouble data. Transformers scans every *.py file,
# so reject or remove metadata accidentally copied from macOS.
find "$REPO_ROOT" -type f \( -name '._*' -o -name '.DS_Store' \) -delete
find "$REPO_ROOT" -type d -name '__MACOSX' -prune -exec rm -rf {} +

"$MINICONDA_ROOT/bin/python" - "$REPO_ROOT/transformers-4.57.0/src/transformers" <<'PY'
import sys
from pathlib import Path

bad_files = []
for path in Path(sys.argv[1]).rglob("*.py"):
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        bad_files.append(f"{path}: {error}")
if bad_files:
    raise RuntimeError("Non-UTF-8 Python files found:\n" + "\n".join(bad_files))
print("Transformers Python source UTF-8 validation: OK")
PY

# shellcheck disable=SC1090
source "$conda_sh"

if [[ ! -x "$CONDA_ENV_DIR/bin/python" ]]; then
  conda create --yes --prefix "$CONDA_ENV_DIR" python=3.11 pip
fi
conda activate "$CONDA_ENV_DIR"

python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 11), sys.version
print(sys.version)
PY

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

python -m pip install \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'

DS_BUILD_OPS=0 python -m pip install -e "$REPO_ROOT/LLaMA-Factory[torch,metrics,deepspeed]" --no-build-isolation
python -m pip install \
  deepspeed==0.16.9 \
  qwen-vl-utils==0.0.14 \
  huggingface_hub==0.36.2 \
  wandb
python -m pip install -e "$REPO_ROOT/transformers-4.57.0" --no-build-isolation --no-deps
python -m pip check

python - "$REPO_ROOT" <<'PY'
import os
import sys
from importlib.metadata import version
from pathlib import Path

import deepspeed
import flash_attn
import qwen_vl_utils
import torch
import transformers

repo_root = Path(sys.argv[1]).resolve()
expected_transformers_root = (repo_root / "transformers-4.57.0").resolve()
assert torch.__version__ == "2.6.0+cu124", torch.__version__
assert transformers.__version__ == "4.57.0", transformers.__version__
assert Path(transformers.__file__).resolve().is_relative_to(expected_transformers_root), transformers.__file__
assert flash_attn.__version__ == "2.7.4.post1", flash_attn.__version__
assert deepspeed.__version__ == "0.16.9", deepspeed.__version__
assert version("llamafactory") == "0.9.4.dev0", version("llamafactory")
assert version("qwen-vl-utils") == "0.0.14", version("qwen-vl-utils")
assert version("huggingface-hub") == "0.36.2", version("huggingface-hub")
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert torch.version.cuda == "12.4", torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI is False
print("torch:", torch.__version__)
print("transformers:", transformers.__version__, transformers.__file__)
print("flash_attn:", flash_attn.__version__)
print("deepspeed:", deepspeed.__version__)
print("GPU count:", torch.cuda.device_count())
PY

hf --help >/dev/null
hf download --help >/dev/null
ds_report

echo "Environment setup completed: $CONDA_ENV_DIR"
