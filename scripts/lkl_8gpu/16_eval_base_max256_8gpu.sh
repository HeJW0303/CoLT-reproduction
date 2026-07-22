#!/usr/bin/env bash

set -euo pipefail
export COLT_REQUIRE_FINAL_MODEL=0
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"
require_workspace_layout

datasets=(AI2D_TEST TextVQA_VAL)
eval_model_name="Qwen3-VL-8B-Instruct-BASE-COT-MAX256"
paper_profile="qwen3vl_cot"
generation_limit=256
vision_profile="legacy14_processor_resize"
gpu_csv="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
run_id="$(date +%Y%m%d_%H%M%S)"
log_file="$EVAL_LOG_ROOT/qwen3vl_base_legacy14_greedy_max256_8gpu_${run_id}.log"
exec > >(tee -a "$log_file") 2>&1

# Keep this diagnostic orthogonal to CoLT training and latent inference.
export COLT_DISABLE_LATENT_REASONING=1
export QWEN3_VL_BASE_MODEL_PATH="$BASE_MODEL_DIR"
export COLT_EXPECT_BASE_MAX_NEW_TOKENS="$generation_limit"
export COLT_VISION_PREPROCESS_PROFILE="$vision_profile"
export COLT_RESEED_PER_SAMPLE=0

touch "$VLMEVAL_ROOT/.env"
python -m pip check
python - "$REPO_ROOT" "$eval_model_name" "$generation_limit" <<'PY'
import sys
from functools import partial
from importlib.metadata import version
from pathlib import Path

import transformers
from vlmeval.config import supported_VLM
from vlmeval.vlm import Qwen3VLBaseChat

repo_root = Path(sys.argv[1]).resolve()
model_name = sys.argv[2]
expected_limit = int(sys.argv[3])
transformers_file = Path(transformers.__file__).resolve()
factory = supported_VLM[model_name]

assert transformers.__version__ == "4.57.0", transformers.__version__
assert transformers_file.is_relative_to((repo_root / "transformers-4.57.0").resolve()), transformers_file
assert version("torch").startswith("2.6.0"), version("torch")
assert version("flash-attn") == "2.7.4.post1", version("flash-attn")
assert version("qwen-vl-utils") == "0.0.14", version("qwen-vl-utils")
assert version("numpy") == "1.26.4", version("numpy")
assert Qwen3VLBaseChat.__module__.endswith("qwen3_vl_baseline"), Qwen3VLBaseChat.__module__
assert isinstance(factory, partial), type(factory)
assert factory.func is Qwen3VLBaseChat, factory.func
assert factory.keywords["max_new_tokens"] == expected_limit, factory.keywords
assert factory.keywords["do_sample"] is False, factory.keywords
assert factory.keywords["use_vllm"] is False, factory.keywords
print("Baseline max-256 diagnostic environment validation: OK")
print("transformers:", transformers_file)
print("adapter:", Qwen3VLBaseChat.__module__)
print("generation: greedy, max_new_tokens=", expected_limit, sep="")
PY

python - "$BASE_MODEL_DIR" "$BASE_MODEL_REVISION" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

model_dir = Path(sys.argv[1])
expected_revision = sys.argv[2]
revision_file = model_dir / ".colt_verified_revision"
if not revision_file.is_file() or revision_file.read_text().strip() != expected_revision:
    raise RuntimeError(f"Base model revision marker is missing or incorrect: {revision_file}")

index_path = model_dir / "model.safetensors.index.json"
if not index_path.is_file():
    raise RuntimeError(f"Missing base model index: {index_path}")
index = json.loads(index_path.read_text())
weight_names = set(index["weight_map"])
shards = sorted(set(index["weight_map"].values()))
forbidden_parts = {
    "decoder", "backward_decoder", "prj", "latent_predictor",
    "pj_in", "pj_back", "pj_out", "alpha", "latent_to_decoder_scale",
}
unexpected = sorted(name for name in weight_names if forbidden_parts.intersection(name.split(".")))
if unexpected:
    raise RuntimeError(f"Base model unexpectedly contains CoLT parameters: {unexpected[:10]}")

expected_keys_by_shard = {shard: set() for shard in shards}
for name, shard_name in index["weight_map"].items():
    expected_keys_by_shard[shard_name].add(name)
total_bytes = 0
for shard_name, expected_keys in expected_keys_by_shard.items():
    shard = model_dir / shard_name
    if not shard.is_file() or shard.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty base model shard: {shard}")
    total_bytes += shard.stat().st_size
    with safe_open(shard, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(f"Base model index/shard key mismatch: {shard}")

config = json.loads((model_dir / "config.json").read_text())
preprocessor = json.loads((model_dir / "preprocessor_config.json").read_text())
if config.get("model_type") != "qwen3_vl":
    raise RuntimeError(f"Unexpected base model type: {config.get('model_type')}")
if preprocessor.get("patch_size") != 16 or preprocessor.get("merge_size") != 2:
    raise RuntimeError(f"Unexpected base processor config: {preprocessor}")
print(
    f"Base model verified: revision={expected_revision} shards={len(shards)} "
    f"tensors={len(weight_names)} size_gib={total_bytes / 2**30:.2f} CoLT_parameters=0"
)
PY

bash "$REPO_ROOT/scripts/lkl_8gpu/09_download_eval_data.sh" "${datasets[@]}"

IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
if (( ${#gpu_ids[@]} != 8 )); then
  echo "COLT_EVAL_GPUS must contain exactly eight physical GPU ids: $gpu_csv" >&2
  exit 1
fi
declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${seen_gpu[$gpu]:-}" ]]; then
    echo "Invalid or duplicate GPU id: $gpu" >&2
    exit 1
  fi
  seen_gpu[$gpu]=1
  if ! memory_used="$(
    nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
  )"; then
    echo "Unable to query physical GPU $gpu." >&2
    exit 1
  fi
  echo "Physical GPU $gpu detected: ${memory_used:-unknown} MiB in use; memory usage is not enforced."
done

export CUDA_VISIBLE_DEVICES="$gpu_csv"
export PRED_FORMAT=xlsx
export EVAL_FORMAT=csv
export DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE

eval_fingerprint="$(
  {
    printf '%s\n' \
      "model_mode=baseline" \
      "model_revision=$BASE_MODEL_REVISION" \
      "max_new_tokens=$generation_limit" \
      "do_sample=false" \
      "vision_profile=$vision_profile" \
      "seed=$COLT_EVAL_SEED" \
      "datasets=${datasets[*]}"
    find "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl" \
      -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
    find "$VLMEVAL_ROOT/vlmeval" -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
    sha256sum \
      "$REPO_ROOT/scripts/lkl_8gpu/16_eval_base_max256_8gpu.sh" \
      "$REPO_ROOT/scripts/lkl_8gpu/validate_eval_suite.py" \
      "$REPO_ROOT/scripts/lkl_8gpu/eval_common.sh" \
      "$BASE_MODEL_DIR/model.safetensors.index.json" \
      "$BASE_MODEL_DIR/config.json" \
      "$BASE_MODEL_DIR/preprocessor_config.json"
    find "$BASE_MODEL_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' \
      -printf '%f %s %T@\n' | sort
  } | sha256sum | cut -c1-12
)"
eval_profile="dp8_legacy14_greedy_max${generation_limit}_seed${COLT_EVAL_SEED}_${eval_fingerprint}"
work_dir="$EVAL_OUTPUT_ROOT/diagnostic_generation/base_max256/$eval_profile"
eval_id="BASE_MAX256_${eval_profile}"
export MMEVAL_ROOT="$work_dir"
export VLMEVAL_EVAL_ID="$eval_id"
mkdir -p "$work_dir"

echo "Model mode: untouched Qwen3-VL baseline; CoLT latent modules disabled"
echo "Datasets: ${datasets[*]}"
echo "Generation: native Hugging Face greedy decoding, max_new_tokens=$generation_limit"
echo "Vision preprocessing: $vision_profile"
echo "Parallelism: 8 model replicas; each dataset is sharded across all 8 ranks"
echo "Physical GPUs: $CUDA_VISIBLE_DEVICES"
echo "Evaluation fingerprint: $eval_fingerprint"
echo "Evaluation id: $eval_id"
echo "Results: $work_dir"
echo "Log: $log_file"
echo "Resume: enabled only inside this exact model/code/generation fingerprint"

cd "$VLMEVAL_ROOT"
args=(
  --standalone
  --nnodes=1
  --nproc_per_node=8
  --max_restarts=0
  run.py
  --data "${datasets[@]}"
  --model "$eval_model_name"
  --work-dir "$work_dir"
  --mode all
  --judge exact_matching
  --reuse
)
if [[ "${EVAL_VERBOSE:-0}" == "1" ]]; then
  args+=(--verbose)
fi
torchrun "${args[@]}"

python "$REPO_ROOT/scripts/lkl_8gpu/validate_eval_suite.py" \
  "$work_dir" \
  "$eval_model_name" \
  "$eval_id" \
  "$EVAL_DATA_ROOT" \
  "${datasets[@]}" \
  --paper-profile "$paper_profile"

echo "Baseline max-256 diagnostic completed and both datasets validated."
echo "Results: $work_dir/$eval_model_name/$eval_id"
echo "Log: $log_file"
