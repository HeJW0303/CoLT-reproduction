#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"
require_free_gib 10

constraints="$REPO_ROOT/scripts/a100/eval_constraints.txt"

# VLMEvalKit declares broad, unbounded dependencies. Installing it normally can
# upgrade the already validated training stack to incompatible major versions.
python -m pip uninstall -y \
  hf-gradio \
  opencv-python \
  opencv-python-headless \
  opencv-contrib-python \
  opencv-contrib-python-headless \
  >/dev/null 2>&1 || true
python -m pip install \
  --constraint "$constraints" \
  --requirement "$VLMEVAL_ROOT/requirements.txt"
python -m pip install -e "$VLMEVAL_ROOT" --no-build-isolation --no-deps
python -m pip install -e "$REPO_ROOT/transformers-4.57.0" --no-build-isolation --no-deps
touch "$VLMEVAL_ROOT/.env"
python -m pip check

python - <<'PY'
import os
import transformers
import vlmeval
from importlib.metadata import PackageNotFoundError, version
from vlmeval.dataset import build_dataset
from vlmeval.vlm import Qwen3VLChat

assert transformers.__version__ == "4.57.0", transformers.__version__
assert transformers.__file__.startswith("/workspace/CoLT/transformers-4.57.0"), transformers.__file__
assert version("torch").startswith("2.6.0"), version("torch")
assert version("torchvision").startswith("0.21.0"), version("torchvision")
assert version("flash-attn") == "2.7.4.post1", version("flash-attn")
assert version("deepspeed") == "0.16.9", version("deepspeed")
assert version("qwen-vl-utils") == "0.0.14", version("qwen-vl-utils")
assert version("tokenizers") == "0.22.1", version("tokenizers")
assert version("huggingface-hub") == "0.36.2", version("huggingface-hub")
assert version("numpy") == "1.26.4", version("numpy")
assert version("opencv-python-headless") == "4.11.0.86", version("opencv-python-headless")
try:
    version("opencv-python")
except PackageNotFoundError:
    pass
else:
    raise AssertionError("opencv-python GUI wheel must not coexist with opencv-python-headless")
assert version("gradio") == "5.45.0", version("gradio")
assert version("pydantic") == "2.10.6", version("pydantic")
assert version("safetensors") == "0.5.3", version("safetensors")
assert os.environ["LMUData"].startswith("/workspace/"), os.environ["LMUData"]
assert Qwen3VLChat.__module__.endswith("colt_qwen3_vl")
print("transformers:", transformers.__file__)
print("vlmeval:", vlmeval.__file__)
print("CoLT adapter:", Qwen3VLChat.__module__)
print("Evaluation environment validation: OK")
PY
