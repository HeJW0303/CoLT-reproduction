#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout

# Finder metadata such as `._modeling_xxx.py` is binary AppleDouble data. Transformers scans every
# file ending in `.py`, so those files must not be present in an archive copied from macOS.
find "$REPO_ROOT" -type f \( -name '._*' -o -name '.DS_Store' \) -delete
find "$REPO_ROOT" -type d -name '__MACOSX' -prune -exec rm -rf {} +

python - "$REPO_ROOT/transformers-4.57.0/src/transformers" <<'PY'
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

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this setup script as root inside the container." >&2
  exit 1
fi

python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 11), sys.version
print(sys.version)
PY

apt-get update
apt-get install -y --no-install-recommends git curl unzip jq libaio-dev tmux
rm -rf /var/lib/apt/lists/*

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  python -m venv "$VENV_DIR"
elif grep -q '^include-system-site-packages = true$' "$VENV_DIR/pyvenv.cfg"; then
  echo "The existing venv exposes base-image site packages: $VENV_DIR" >&2
  echo "Remove that venv directory and rerun this script." >&2
  exit 1
fi
activate_colt_env

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

python - <<'PY'
import os
from importlib.metadata import version
import torch
import transformers
import flash_attn
import deepspeed
import qwen_vl_utils

expected_transformers_root = "/workspace/CoLT/transformers-4.57.0"
assert torch.__version__ == "2.6.0+cu124", torch.__version__
assert transformers.__version__ == "4.57.0", transformers.__version__
assert os.path.realpath(transformers.__file__).startswith(expected_transformers_root), transformers.__file__
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

python - <<'PY'
from importlib.metadata import version
print("huggingface-hub:", version("huggingface-hub"))
PY
hf --help >/dev/null
hf download --help >/dev/null
ds_report

echo "Environment setup completed: $VENV_DIR"
