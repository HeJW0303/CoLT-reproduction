#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env
validate_gpu_profile

export COLT_DECODER_MODEL_PATH="$DECODER_MODEL_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

current_revision="$(git -c safe.directory="$REPO_ROOT" -C "$REPO_ROOT" rev-parse HEAD)"
if [[ "$current_revision" != "$REPO_REVISION" ]]; then
  echo "Warning: repository HEAD is $current_revision, reference revision is $REPO_REVISION." >&2
fi

python "$REPO_ROOT/scripts/lkl_8gpu/validate_dataset.py" --data-root "$DATA_ROOT"

python - <<PY
import os
import json
from pathlib import Path
from importlib.metadata import version

import deepspeed
import flash_attn
import torch
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

repo_root = Path("$REPO_ROOT")
base_model = Path("$BASE_MODEL_DIR")
decoder_model = Path("$DECODER_MODEL_DIR")
data_root = Path("$DATA_ROOT")

assert torch.__version__ == "2.6.0+cu124", torch.__version__
assert transformers.__version__ == "4.57.0", transformers.__version__
assert str(Path(transformers.__file__).resolve()).startswith(str((repo_root / "transformers-4.57.0").resolve()))
assert flash_attn.__version__ == "2.7.4.post1"
assert deepspeed.__version__ == "0.16.9"
assert version("llamafactory") == "0.9.4.dev0"
assert version("qwen-vl-utils") == "0.0.14"
assert version("huggingface-hub") == "0.36.2"
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert (data_root / "dataset_info.json").is_file()
assert (data_root / ".colt_extraction_complete").read_text().strip() == "$DATASET_REVISION"
assert not list(data_root.rglob("*.zip")), "Dataset ZIP files still exist"
assert (base_model / ".colt_verified_revision").read_text().strip() == "$BASE_MODEL_REVISION"
assert (decoder_model / ".colt_verified_revision").read_text().strip() == "$DECODER_MODEL_REVISION"

def verify_weight_index(model_dir: Path) -> None:
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if index_files:
        with index_files[0].open() as file:
            weight_map = json.load(file)["weight_map"]
        missing = sorted({name for name in weight_map.values() if not (model_dir / name).is_file()})
        assert not missing, f"Missing weight shards: {missing}"
    else:
        assert list(model_dir.glob("*.safetensors")), f"No safetensors weights under {model_dir}"

verify_weight_index(base_model)
verify_weight_index(decoder_model)

base_config = AutoConfig.from_pretrained(base_model, local_files_only=True)
decoder_config = AutoConfig.from_pretrained(decoder_model, local_files_only=True)
AutoTokenizer.from_pretrained(base_model, local_files_only=True)
AutoTokenizer.from_pretrained(decoder_model, local_files_only=True)
AutoProcessor.from_pretrained(base_model, local_files_only=True)
assert os.environ["COLT_DECODER_MODEL_PATH"] == str(decoder_model)
print("base model:", base_config.model_type)
print("decoder model:", decoder_config.model_type)
print("transformers:", transformers.__file__)
print("GPU count:", torch.cuda.device_count())
print("Offline model/tokenizer validation: OK")
PY

require_free_gib 200

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
nvidia-smi topo -m
df -h "$WORKSPACE_ROOT"
df -i "$WORKSPACE_ROOT"
echo "Static readiness checks passed. Run 05_nccl_smoke.sh next."
