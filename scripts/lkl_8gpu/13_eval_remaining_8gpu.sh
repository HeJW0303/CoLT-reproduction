#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/eval_common.sh"

datasets=(
  ChartQA_TEST
  AI2D_TEST
  MMBench_DEV_EN
  ScienceQA_TEST
  TextVQA_VAL
  MMT-Bench_VAL
  SEEDBench_IMG
)
gpu_csv="${COLT_EVAL_GPUS:-0,1,2,3,4,5,6,7}"
run_id="$(date +%Y%m%d_%H%M%S)"
log_file="$EVAL_LOG_ROOT/colt_eval_remaining7_8gpu_${run_id}.log"
exec > >(tee -a "$log_file") 2>&1

touch "$VLMEVAL_ROOT/.env"
python -m pip check
python - "$REPO_ROOT" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

import transformers
from vlmeval.vlm import Qwen3VLChat

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
assert Qwen3VLChat.__module__.endswith("colt_qwen3_vl"), Qwen3VLChat.__module__
print("Evaluation environment validation: OK")
print("transformers:", transformers_file)
print("CoLT adapter:", Qwen3VLChat.__module__)
PY

echo "Verifying the final model and the seven remaining evaluation datasets."
bash "$REPO_ROOT/scripts/lkl_8gpu/07_verify_final_model.sh"
bash "$REPO_ROOT/scripts/lkl_8gpu/09_download_eval_data.sh" "${datasets[@]}"

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
    "$FINAL_MODEL_DIR" \
    "$DECODER_MODEL_DIR" \
    "$COLT_EVAL_SEED" \
    "${datasets[@]}" <<'PY'
import hashlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
vlmeval_root = Path(sys.argv[2]).resolve()
model_root = Path(sys.argv[3]).resolve()
decoder_root = Path(sys.argv[4]).resolve()
seed = sys.argv[5]
datasets = sys.argv[6:]
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
        elif path.suffix in {".json", ".model", ".py", ".txt"} or path.name.startswith("tokenizer"):
            add_file(path, relative)


add_text("seed", seed)
add_text("datasets", "\n".join(datasets))
for distribution in (
    "torch",
    "transformers",
    "flash-attn",
    "qwen-vl-utils",
    "numpy",
    "pandas",
    "Pillow",
    "opencv-python-headless",
    "accelerate",
    "safetensors",
    "openpyxl",
    "xlsxwriter",
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
    repo_root / "scripts/lkl_8gpu/common.sh",
    repo_root / "scripts/lkl_8gpu/eval_common.sh",
    repo_root / "scripts/lkl_8gpu/09_download_eval_data.sh",
    repo_root / "scripts/lkl_8gpu/13_eval_remaining_8gpu.sh",
    repo_root / "scripts/lkl_8gpu/validate_eval_suite.py",
    repo_root / "scripts/lkl_8gpu/eval_constraints.txt",
):
    add_file(path, str(path.relative_to(repo_root)))
add_file(vlmeval_root / "run.py", "VLMEvalKit/run.py")
for path in sorted((vlmeval_root / "vlmeval").rglob("*.py")):
    add_file(path, str(path.relative_to(repo_root)))
add_model_tree(model_root, "final_model")
add_model_tree(decoder_root, "decoder_model")
print(digest.hexdigest()[:12])
PY
)"

eval_profile="dp8_remaining7_seed${COLT_EVAL_SEED}_${eval_fingerprint}"
work_dir="$EVAL_OUTPUT_ROOT/codefaithful/remaining7/$eval_profile"
eval_id="COLT_${eval_profile}"

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
echo "Backend: PyTorch + Transformers + FlashAttention 2 (not vLLM)"
echo "Parallelism: 8 model replicas; every dataset is sharded across all 8 ranks"
echo "Model loading: once per rank, then reused across all remaining datasets"
echo "Physical GPUs: $CUDA_VISIBLE_DEVICES"
echo "Evaluation fingerprint: $eval_fingerprint"
echo "Evaluation id: $eval_id"
echo "Model: $COLT_EVAL_MODEL_PATH"
echo "Data root: $EVAL_DATA_ROOT"
echo "Results: $work_dir"
echo "Log: $log_file"
echo "Resume: enabled only inside this seed/code/model fingerprint"
echo "Crash-safe writes: atomic rank pickle, result table, and decoded image replacement"

cd "$VLMEVAL_ROOT"
args=(
  --standalone
  --nnodes=1
  --nproc_per_node=8
  --max_restarts=0
  run.py
  --data "${datasets[@]}"
  --model "$EVAL_MODEL_NAME"
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
  "$EVAL_MODEL_NAME" \
  "$eval_id" \
  "$EVAL_DATA_ROOT" \
  "${datasets[@]}"

echo "All seven remaining 8-GPU evaluations completed and validated."
echo "Log: $log_file"
