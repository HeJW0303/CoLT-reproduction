#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"

python - "$FINAL_MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

model_dir = Path(sys.argv[1])
index_path = model_dir / "model.safetensors.index.json"
index = json.loads(index_path.read_text())
shards = sorted(set(index["weight_map"].values()))
assert len(shards) == 5, shards

tensor_count = 0
for shard_name in shards:
    shard = model_dir / shard_name
    assert shard.is_file() and shard.stat().st_size > 0, shard
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor_count += len(handle.keys())

required = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "train_results.json",
]
missing = [name for name in required if not (model_dir / name).is_file()]
assert not missing, f"Missing final model metadata: {missing}"

state = json.loads((model_dir / "trainer_state.json").read_text())
assert state["global_step"] == 1910, state["global_step"]
print(f"Final inference model: 5 shards, {tensor_count} tensors, global_step=1910")

checkpoint = model_dir / "checkpoint-1910"
if checkpoint.is_dir():
    assert (checkpoint / "latest").read_text().strip() == "global_step1910"
    zero_dir = checkpoint / "global_step1910"
    optim = sorted(zero_dir.glob("*optim_states.pt"))
    assert len(optim) == 8, len(optim)
    print("Resume checkpoint: checkpoint-1910 is complete with 8 optimizer shards")
else:
    print("Resume checkpoint: not present; inference model remains complete")
PY

du -sh "$FINAL_MODEL_DIR"
df -h "$WORKSPACE_ROOT"
