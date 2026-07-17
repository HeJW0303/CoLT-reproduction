#!/usr/bin/env bash

set -euo pipefail
export COLT_REQUIRE_FINAL_MODEL=0
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"

datasets=(AI2D_TEST TextVQA_VAL)
variants=(
  legacy14_processor_resize
  model_patch_processor_resize
  model_patch_no_processor_resize
)
model_name="Qwen3-VL-8B-Instruct-BASE-COT"
paper_profile="qwen3vl_cot"
gpu_csv="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"

run_id="$(date +%Y%m%d_%H%M%S)"
log_file="$EVAL_LOG_ROOT/preprocess_ab_baseline_8gpu_${run_id}.log"
exec > >(tee -a "$log_file") 2>&1

# These must be set before any model construction. The vendored model class
# returns from __init__ before creating any CoLT-only modules when disabled.
export COLT_DISABLE_LATENT_REASONING=1
export QWEN3_VL_BASE_MODEL_PATH="$BASE_MODEL_DIR"
export COLT_RESEED_PER_SAMPLE=0

touch "$VLMEVAL_ROOT/.env"
python -m pip check
python - "$REPO_ROOT" <<'PY'
import os
import sys
from importlib.metadata import version
from pathlib import Path

import transformers
from vlmeval.vlm import Qwen3VLBaseChat

repo_root = Path(sys.argv[1]).resolve()
transformers_file = Path(transformers.__file__).resolve()
assert os.environ["COLT_DISABLE_LATENT_REASONING"] == "1"
assert transformers.__version__ == "4.57.0", transformers.__version__
assert transformers_file.is_relative_to((repo_root / "transformers-4.57.0").resolve()), transformers_file
assert version("torch").startswith("2.6.0"), version("torch")
assert version("flash-attn") == "2.7.4.post1", version("flash-attn")
assert version("qwen-vl-utils") == "0.0.14", version("qwen-vl-utils")
assert version("numpy") == "1.26.4", version("numpy")
assert Qwen3VLBaseChat.__module__.endswith("qwen3_vl_baseline"), Qwen3VLBaseChat.__module__
print("Baseline preprocessing A/B environment validation: OK")
print("transformers:", transformers_file)
print("baseline adapter:", Qwen3VLBaseChat.__module__)
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
if not shards:
    raise RuntimeError("Base model index contains no safetensors shards.")

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

for shard_name, expected_keys in expected_keys_by_shard.items():
    shard = model_dir / shard_name
    if not shard.is_file() or shard.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty base model shard: {shard}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"Base model index/shard key mismatch for {shard_name}: "
            f"missing={sorted(expected_keys - actual_keys)[:5]}, "
            f"extra={sorted(actual_keys - expected_keys)[:5]}"
        )

total_bytes = 0
for shard_name in shards:
    shard = model_dir / shard_name
    total_bytes += shard.stat().st_size

config = json.loads((model_dir / "config.json").read_text())
if config.get("model_type") != "qwen3_vl":
    raise RuntimeError(f"Unexpected base model type: {config.get('model_type')}")
preprocessor = json.loads((model_dir / "preprocessor_config.json").read_text())
if preprocessor.get("patch_size") != 16 or preprocessor.get("merge_size") != 2:
    raise RuntimeError(
        "Unexpected Qwen3-VL base preprocessing config: "
        f"patch_size={preprocessor.get('patch_size')}, merge_size={preprocessor.get('merge_size')}"
    )
print(
    f"Base model verified: revision={expected_revision} shards={len(shards)} "
    f"tensors={len(weight_names)} size_gib={total_bytes / 2**30:.2f} "
    "model_type=qwen3_vl CoLT_parameters=0 patch_size=16 merge_size=2"
)
PY

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

bash "$REPO_ROOT/scripts/a100/09_download_eval_data.sh" "${datasets[@]}"
export CUDA_VISIBLE_DEVICES="$gpu_csv"
export PRED_FORMAT=xlsx
export EVAL_FORMAT=csv
export DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE

fingerprint="$(
  {
    printf '%s\n' \
      "model_mode=baseline" \
      "model_revision=$BASE_MODEL_REVISION" \
      "seed=$COLT_EVAL_SEED" \
      "paper_profile=$paper_profile" \
      "datasets=${datasets[*]}" \
      "reseed_per_sample=$COLT_RESEED_PER_SAMPLE" \
      "profile_order=${variants[*]}"
    find "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl" \
      -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
    find "$VLMEVAL_ROOT/vlmeval" -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
    sha256sum \
      "$REPO_ROOT/scripts/a100/15_eval_preprocess_ab_8gpu.sh" \
      "$REPO_ROOT/scripts/a100/compare_preprocess_ab.py" \
      "$REPO_ROOT/scripts/a100/validate_eval_suite.py" \
      "$REPO_ROOT/scripts/a100/eval_common.sh" \
      "$REPO_ROOT/scripts/a100/09_download_eval_data.sh" \
      "$BASE_MODEL_DIR/model.safetensors.index.json" \
      "$BASE_MODEL_DIR/config.json" \
      "$BASE_MODEL_DIR/preprocessor_config.json" \
      "$BASE_MODEL_DIR/tokenizer_config.json"
    find "$BASE_MODEL_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' \
      -printf '%f %s %T@\n' | sort
  } | sha256sum | cut -c1-12
)"
experiment_root="$EVAL_OUTPUT_ROOT/diagnostic_preprocess/baseline_seed${COLT_EVAL_SEED}_${fingerprint}_${run_id}"
mkdir -p "$experiment_root"
compare_args=(
  --model-name "$model_name"
  --datasets "${datasets[@]}"
  --output "$experiment_root/preprocess_ab_summary.csv"
)

for variant in "${variants[@]}"; do
  export COLT_VISION_PREPROCESS_PROFILE="$variant"
  work_dir="$experiment_root/$variant"
  eval_id="PREPROC_baseline_${variant}_seed${COLT_EVAL_SEED}_${fingerprint}"
  export MMEVAL_ROOT="$work_dir"
  export VLMEVAL_EVAL_ID="$eval_id"
  mkdir -p "$work_dir"

  echo
  echo "Model mode: baseline-only (native Qwen3-VL textual CoT)"
  echo "Datasets: ${datasets[*]}"
  echo "Vision preprocessing profile: $variant"
  echo "Per-sample deterministic reseeding: disabled (greedy decoding)"
  echo "Physical GPUs: $CUDA_VISIBLE_DEVICES"
  echo "Evaluation fingerprint: $fingerprint"
  echo "Results: $work_dir"

  if find "$work_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to reuse a non-empty preprocessing profile directory: $work_dir" >&2
    echo "This diagnostic requires a fresh isolated directory for every profile." >&2
    exit 1
  fi

  cd "$VLMEVAL_ROOT"
  args=(
    --standalone
    --nnodes=1
    --nproc_per_node=8
    --max_restarts=0
    run.py
    --data "${datasets[@]}"
    --model "$model_name"
    --work-dir "$work_dir"
    --mode all
    --judge exact_matching
  )
  if [[ "${EVAL_VERBOSE:-0}" == "1" ]]; then
    args+=(--verbose)
  fi
  torchrun "${args[@]}"

  python "$REPO_ROOT/scripts/a100/validate_eval_suite.py" \
    "$work_dir" \
    "$model_name" \
    "$eval_id" \
    "$EVAL_DATA_ROOT" \
    "${datasets[@]}" \
    --paper-profile "$paper_profile"

  result_dir="$work_dir/$model_name/$eval_id"
  compare_args+=(--variant "$variant=$result_dir")
done

python "$REPO_ROOT/scripts/a100/compare_preprocess_ab.py" "${compare_args[@]}"
echo "Completed baseline-only preprocessing A/B: $experiment_root"
echo "Summary: $experiment_root/preprocess_ab_summary.csv"
echo "Log: $log_file"
