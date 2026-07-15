#!/usr/bin/env bash

set -euo pipefail
export COLT_REQUIRE_FINAL_MODEL=0
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"
require_workspace_layout

datasets=(
  MMStar
  ChartQA_TEST
  ScienceQA_TEST
  AI2D_TEST
  MMT-Bench_VAL
  MMBench_DEV_EN
  TextVQA_VAL
  SEEDBench_IMG
)
eval_model_name="Qwen3-VL-8B-Instruct-BASE-COT"
paper_profile="qwen3vl_cot"
gpu_csv="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
run_id="$(date +%Y%m%d_%H%M%S)"
log_file="$EVAL_LOG_ROOT/qwen3vl_base_cot_all8_8gpu_${run_id}.log"
exec > >(tee -a "$log_file") 2>&1

export COLT_DISABLE_LATENT_REASONING=1
export QWEN3_VL_BASE_MODEL_PATH="$BASE_MODEL_DIR"

touch "$VLMEVAL_ROOT/.env"
python -m pip check
python - "$REPO_ROOT" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

import transformers
from vlmeval.vlm import Qwen3VLBaseChat

repo_root = Path(sys.argv[1]).resolve()
transformers_file = Path(transformers.__file__).resolve()
vendored_root = (repo_root / "transformers-4.57.0").resolve()

assert transformers.__version__ == "4.57.0", transformers.__version__
assert transformers_file.is_relative_to(vendored_root), transformers_file
assert version("torch").startswith("2.6.0"), version("torch")
assert version("flash-attn") == "2.7.4.post1", version("flash-attn")
assert version("qwen-vl-utils") == "0.0.14", version("qwen-vl-utils")
assert version("numpy") == "1.26.4", version("numpy")
assert version("opencv-python-headless") == "4.11.0.86", version("opencv-python-headless")
assert version("openpyxl"), version("openpyxl")
assert version("xlsxwriter"), version("xlsxwriter")
assert Qwen3VLBaseChat.__module__.endswith("qwen3_vl_baseline"), Qwen3VLBaseChat.__module__
print("Baseline evaluation environment validation: OK")
print("transformers:", transformers_file)
print("Qwen3-VL baseline adapter:", Qwen3VLBaseChat.__module__)
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
unexpected = sorted(
    name for name in weight_names if forbidden_parts.intersection(name.split("."))
)
if unexpected:
    raise RuntimeError(f"Base model unexpectedly contains CoLT parameters: {unexpected[:10]}")

tensor_count = 0
total_bytes = 0
for shard_name in shards:
    shard = model_dir / shard_name
    if not shard.is_file() or shard.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty base model shard: {shard}")
    total_bytes += shard.stat().st_size
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor_count += len(handle.keys())

config = json.loads((model_dir / "config.json").read_text())
if config.get("model_type") != "qwen3_vl":
    raise RuntimeError(f"Unexpected base model type: {config.get('model_type')}")
print(
    f"Base model verified: revision={expected_revision} shards={len(shards)} "
    f"tensors={tensor_count} size_gib={total_bytes / 2**30:.2f} CoLT_parameters=0"
)
PY

bash "$REPO_ROOT/scripts/a100/09_download_eval_data.sh" "${datasets[@]}"

IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
if (( ${#gpu_ids[@]} != 8 )); then
  echo "COLT_EVAL_GPUS must contain exactly eight physical GPU ids: $gpu_csv" >&2
  exit 1
fi

declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id: $gpu" >&2
    exit 1
  fi
  if [[ -n "${seen_gpu[$gpu]:-}" ]]; then
    echo "Duplicate GPU id: $gpu" >&2
    exit 1
  fi
  seen_gpu[$gpu]=1

  if ! memory_used="$(
      nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
    )"; then
    echo "Unable to query physical GPU $gpu." >&2
    exit 1
  fi
  if [[ ! "$memory_used" =~ ^[0-9]+$ ]]; then
    echo "Unexpected memory usage returned for physical GPU $gpu: $memory_used" >&2
    exit 1
  fi
  if (( memory_used >= 500 )); then
    echo "Physical GPU $gpu is not free: ${memory_used} MiB is in use." >&2
    exit 1
  fi
done

eval_fingerprint="$(
  python - \
    "$REPO_ROOT" \
    "$VLMEVAL_ROOT" \
    "$BASE_MODEL_DIR" \
    "$BASE_MODEL_REVISION" \
    "$COLT_EVAL_SEED" \
    "$paper_profile" \
    "${datasets[@]}" <<'PY'
import hashlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
vlmeval_root = Path(sys.argv[2]).resolve()
model_root = Path(sys.argv[3]).resolve()
model_revision = sys.argv[4]
seed = sys.argv[5]
paper_profile = sys.argv[6]
datasets = sys.argv[7:]
digest = hashlib.sha256()


def add_text(label: str, value: str) -> None:
    digest.update(label.encode())
    digest.update(b"\0")
    digest.update(value.encode())
    digest.update(b"\0")


def add_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    add_text("path", label)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def add_model_tree(root: Path, label: str) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        relative = f"{label}/{path.name}"
        if path.suffix == ".safetensors":
            stat = path.stat()
            add_text("weight", f"{relative}:{stat.st_size}:{stat.st_mtime_ns}")
        elif (
            path.suffix in {".json", ".model", ".py", ".txt"}
            or path.name.startswith("tokenizer")
            or path.name == ".colt_verified_revision"
        ):
            add_file(path, relative)


add_text("mode", "qwen3vl-native-textual-cot-greedy")
add_text("model_revision", model_revision)
add_text("seed", seed)
add_text("paper_profile", paper_profile)
add_text("datasets", "\n".join(datasets))
for distribution in (
    "torch", "transformers", "flash-attn", "qwen-vl-utils", "numpy",
    "pandas", "Pillow", "opencv-python-headless", "accelerate",
    "safetensors", "openpyxl", "xlsxwriter",
):
    try:
        package_version = version(distribution)
    except PackageNotFoundError:
        package_version = "missing"
    add_text("package", f"{distribution}=={package_version}")

transformers_src = repo_root / "transformers-4.57.0/src/transformers"
for source_root in (
    transformers_src / "models/qwen3_vl",
    transformers_src / "generation",
):
    for path in sorted(source_root.rglob("*.py")):
        add_file(path, str(path.relative_to(repo_root)))
for path in (
    repo_root / "scripts/a100/common.sh",
    repo_root / "scripts/a100/eval_common.sh",
    repo_root / "scripts/a100/09_download_eval_data.sh",
    repo_root / "scripts/a100/14_eval_base_qwen3vl_cot_8gpu.sh",
    repo_root / "scripts/a100/validate_eval_suite.py",
    repo_root / "scripts/a100/eval_constraints.txt",
):
    add_file(path, str(path.relative_to(repo_root)))
add_file(vlmeval_root / "run.py", "VLMEvalKit/run.py")
for path in sorted((vlmeval_root / "vlmeval").rglob("*.py")):
    add_file(path, str(path.relative_to(repo_root)))
add_model_tree(model_root, "base_model")
print(digest.hexdigest()[:12])
PY
)"

eval_profile="dp8_all8_greedy_seed${COLT_EVAL_SEED}_${eval_fingerprint}"
work_dir="$EVAL_OUTPUT_ROOT/baseline_qwen3vl_cot/all8/$eval_profile"
eval_id="BASE_COT_${eval_profile}"

export CUDA_VISIBLE_DEVICES="$gpu_csv"
export MMEVAL_ROOT="$work_dir"
export VLMEVAL_EVAL_ID="$eval_id"
export PRED_FORMAT=xlsx
export EVAL_FORMAT=csv
export DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE
mkdir -p "$work_dir"

echo "Datasets: ${datasets[*]}"
echo "Baseline: Qwen3-VL-8B-Instruct textual reasoning (paper average 75.7)"
echo "Generation: native Hugging Face greedy decoding, max_new_tokens=8192"
echo "CoLT latent reasoning: disabled before model construction"
echo "Parallelism: 8 model replicas; every dataset is sharded across all 8 ranks"
echo "Physical GPUs: $CUDA_VISIBLE_DEVICES"
echo "Evaluation fingerprint: $eval_fingerprint"
echo "Evaluation id: $eval_id"
echo "Model: $QWEN3_VL_BASE_MODEL_PATH"
echo "Results: $work_dir"
echo "Log: $log_file"
echo "Resume: enabled only inside this baseline code/model fingerprint"

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

python "$REPO_ROOT/scripts/a100/validate_eval_suite.py" \
  "$work_dir" \
  "$eval_model_name" \
  "$eval_id" \
  "$EVAL_DATA_ROOT" \
  "${datasets[@]}" \
  --paper-profile "$paper_profile"

echo "Qwen3-VL base textual-CoT evaluation completed and all eight datasets validated."
echo "Log: $log_file"
