#!/usr/bin/env bash

cmd_profile() {
  local profile="${1:-generic}"
  [[ "$profile" =~ ^[a-zA-Z0-9._-]+$ ]] || die "GPU profile label contains invalid characters: $profile"
  export COLT_GPU_PROFILE="$profile"
  runtime_init
  require_workspace_layout
  validate_gpu_profile
  local -a required=(git curl unzip tmux nvidia-smi)
  local command_name
  for command_name in "${required[@]}"; do require_command "$command_name"; done
  [[ -x "$MINICONDA_ROOT/bin/conda" ]] || die "Miniconda is missing: $MINICONDA_ROOT/bin/conda"
  [[ -w "$LKL_ROOT" ]] || die "LKL root is not writable: $LKL_ROOT"
  local profile_tmp="$PROFILE_FILE.tmp.$$"
  printf '%s\n' "$profile" > "$profile_tmp"
  mv "$profile_tmp" "$PROFILE_FILE"
  echo "Profile saved: $profile (${COLT_EXPECTED_GPU_NAME:-any 8-GPU model}) -> $PROFILE_FILE"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
}

setup_env() {
  runtime_init
  require_workspace_layout
  validate_gpu_profile
  local command_name
  for command_name in git curl unzip tmux nvidia-smi; do require_command "$command_name"; done
  local conda_sh="$MINICONDA_ROOT/etc/profile.d/conda.sh"
  [[ -f "$conda_sh" && -x "$MINICONDA_ROOT/bin/python" ]] || die \
    "Miniconda is incomplete under $MINICONDA_ROOT"

  find "$REPO_ROOT" -type f \( -name '._*' -o -name '.DS_Store' \) -delete
  find "$REPO_ROOT" -type d -name '__MACOSX' -prune -exec rm -rf {} +
  "$MINICONDA_ROOT/bin/python" - "$REPO_ROOT/transformers-4.57.0/src/transformers" <<'PY'
import sys
from pathlib import Path

bad = []
for path in Path(sys.argv[1]).rglob("*.py"):
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        bad.append(f"{path}: {error}")
if bad:
    raise RuntimeError("Non-UTF-8 Python files found:\n" + "\n".join(bad))
print("Transformers source encoding: OK")
PY

  # shellcheck disable=SC1090
  source "$conda_sh"
  if [[ ! -x "$CONDA_ENV_DIR/bin/python" ]]; then
    conda create --yes --prefix "$CONDA_ENV_DIR" python=3.11 pip
  fi
  conda activate "$CONDA_ENV_DIR"
  python -m pip install --upgrade pip setuptools wheel packaging ninja
  python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
  if ! python -c 'import flash_attn; assert flash_attn.__version__ == "2.7.4.post1"'; then
    python -m pip install \
      'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
  fi
  python -m pip install -e "$REPO_ROOT/LLaMA-Factory[torch,metrics]" --no-build-isolation
  python "$COLT_SCRIPT_ROOT/tools/install_deepspeed.py"
  python -m pip install qwen-vl-utils==0.0.14 huggingface_hub==0.36.2 wandb
  python -m pip install -e "$REPO_ROOT/transformers-4.57.0" --no-build-isolation --no-deps
  python -m pip check
  echo "Training environment ready: $CONDA_ENV_DIR"
}

download_assets() {
  runtime_init
  require_workspace_layout
  activate_colt_env
  require_free_gib 250

  download_hf_repo() {
    local repo_id="$1" revision="$2" local_dir="$3"
    shift 3
    mkdir -p "$local_dir"
    hf download "$repo_id" --revision "$revision" --local-dir "$local_dir" --max-workers 4 "$@"
  }
  download_hf_repo Qwen/Qwen3-VL-8B-Instruct "$BASE_MODEL_REVISION" "$BASE_MODEL_DIR"
  download_hf_repo Qwen/Qwen3-0.6B "$DECODER_MODEL_REVISION" "$DECODER_MODEL_DIR"
  download_hf_repo hulianyuyy/CoLT_Train_Dataset "$DATASET_REVISION" "$DATA_ROOT" --repo-type dataset

  python "$COLT_SCRIPT_ROOT/tools/verify_hf_files.py" \
    --repo-id Qwen/Qwen3-VL-8B-Instruct --repo-type model \
    --revision "$BASE_MODEL_REVISION" --local-dir "$BASE_MODEL_DIR" --suffix .safetensors
  python "$COLT_SCRIPT_ROOT/tools/verify_hf_files.py" \
    --repo-id Qwen/Qwen3-0.6B --repo-type model \
    --revision "$DECODER_MODEL_REVISION" --local-dir "$DECODER_MODEL_DIR" --suffix .safetensors
  python "$COLT_SCRIPT_ROOT/tools/verify_hf_files.py" \
    --repo-id hulianyuyy/CoLT_Train_Dataset --repo-type dataset \
    --revision "$DATASET_REVISION" --local-dir "$DATA_ROOT" --suffix .zip \
    --file colt_sft_image.json --expected-count 21
  printf '%s\n' "$BASE_MODEL_REVISION" > "$BASE_MODEL_DIR/.colt_verified_revision"
  printf '%s\n' "$DECODER_MODEL_REVISION" > "$DECODER_MODEL_DIR/.colt_verified_revision"
  printf '%s\n' "$DATASET_REVISION" > "$DATA_ROOT/.colt_download_verified"
  echo "Models and compressed dataset downloaded and verified."
}

prepare_data() {
  runtime_init
  require_workspace_layout
  activate_colt_env
  require_free_gib 150
  if [[ "$(cat "$DATA_ROOT/.colt_download_verified" 2>/dev/null || true)" != "$DATASET_REVISION" \
        && ! -f "$DATA_ROOT/.colt_extraction_complete" ]]; then
    die "Dataset marker is missing. Run: colt.sh setup assets"
  fi
  local zip_count zip_path
  zip_count="$(find "$DATA_ROOT" -type f -name '*.zip' | wc -l | tr -d ' ')"
  (( zip_count <= 20 )) || die "Expected at most 20 ZIP files; found $zip_count."
  if (( zip_count > 0 )); then
    while IFS= read -r -d '' zip_path; do unzip -tq "$zip_path" >/dev/null; done \
      < <(find "$DATA_ROOT" -type f -name '*.zip' -print0 | sort -z)
    while IFS= read -r -d '' zip_path; do
      echo "Extracting $zip_path"
      unzip -oq "$zip_path" -d "$(dirname "$zip_path")"
      rm -f "$zip_path"
    done < <(find "$DATA_ROOT" -type f -name '*.zip' -print0 | sort -z)
  fi
  python - "$DATA_ROOT/dataset_info.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
registry = json.loads(path.read_text()) if path.is_file() else {}
registry["onethinker_sft_image"] = {
    "file_name": "colt_sft_image.json",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
    },
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(registry, indent=2) + "\n")
temporary.replace(path)
PY
  python "$COLT_SCRIPT_ROOT/tools/validate_dataset.py" --data-root "$DATA_ROOT"
  printf '%s\n' "$DATASET_REVISION" > "$DATA_ROOT/.colt_extraction_complete"
  echo "Training data extracted and validated."
}

setup_eval() {
  runtime_init
  require_workspace_layout
  activate_colt_env
  require_free_gib 10
  python -m pip uninstall -y \
    hf-gradio opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless \
    >/dev/null 2>&1 || true
  python -m pip install --constraint "$COLT_SCRIPT_ROOT/requirements/eval.txt" \
    --requirement "$VLMEVAL_ROOT/requirements.txt"
  python -m pip install -e "$VLMEVAL_ROOT" --no-build-isolation --no-deps
  python -m pip install -e "$REPO_ROOT/transformers-4.57.0" --no-build-isolation --no-deps
  touch "$VLMEVAL_ROOT/.env"
  python -m pip check
  echo "Evaluation environment ready."
}

verify_ready() {
  runtime_init
  require_workspace_layout
  activate_colt_env
  validate_gpu_profile
  export COLT_DECODER_MODEL_PATH="$DECODER_MODEL_DIR"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python "$COLT_SCRIPT_ROOT/tools/validate_dataset.py" --data-root "$DATA_ROOT"
  python - \
    "$REPO_ROOT" \
    "$BASE_MODEL_DIR" \
    "$DECODER_MODEL_DIR" \
    "$BASE_MODEL_REVISION" \
    "$DECODER_MODEL_REVISION" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path
import torch, transformers

repo, base, decoder = map(Path, sys.argv[1:4])
base_revision, decoder_revision = sys.argv[4:6]
assert torch.__version__ == "2.6.0+cu124", torch.__version__
assert transformers.__version__ == "4.57.0", transformers.__version__
assert Path(transformers.__file__).resolve().is_relative_to(repo / "transformers-4.57.0")
assert version("flash-attn") == "2.7.4.post1"
assert torch.cuda.device_count() == 8
assert (base / ".colt_verified_revision").read_text().strip() == base_revision
assert (decoder / ".colt_verified_revision").read_text().strip() == decoder_revision
for model in (base, decoder):
    has_index = (model / "model.safetensors.index.json").is_file()
    has_single_file = any(model.glob("*.safetensors"))
    assert has_index or has_single_file, f"No safetensors weights under {model}"
print("Static training readiness: OK")
PY
  require_free_gib 200
}

verify_nccl() {
  runtime_init
  require_workspace_layout
  activate_colt_env
  validate_gpu_profile
  local gpu_csv="${COLT_TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
  parse_gpu_csv "$gpu_csv"
  [[ "${#COLT_GPU_IDS[@]}" -eq 8 ]] || die "NCCL smoke requires exactly 8 GPU ids: $gpu_csv"
  require_selected_gpus_free
  export CUDA_VISIBLE_DEVICES="$gpu_csv"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  torchrun --standalone --nproc_per_node=8 "$COLT_SCRIPT_ROOT/tools/nccl_smoke.py"
  echo "Eight-rank NCCL smoke test passed."
}

cmd_setup() {
  local target="${1:-all}"
  case "$target" in
    env) setup_env ;;
    assets) download_assets ;;
    data) prepare_data ;;
    eval) setup_eval ;;
    all)
      setup_env
      download_assets
      prepare_data
      setup_eval
      ;;
    *) die "Usage: colt.sh setup {env|assets|data|eval|all}" ;;
  esac
}

cmd_verify() {
  local target="${1:-ready}"
  shift || true
  case "$target" in
    ready) verify_ready ;;
    nccl) verify_nccl ;;
    model)
      runtime_init
      require_workspace_layout
      activate_colt_env
      local model_target="${1:-codefaithful}" cli_path=""
      shift || true
      while (( $# > 0 )); do
        case "$1" in
          --model-path)
            [[ $# -ge 2 ]] || die "--model-path requires a value"
            cli_path="$2"
            shift 2
            ;;
          *) die "Unknown verify model option: $1" ;;
        esac
      done
      resolve_eval_model "$model_target" "$cli_path"
      verify_eval_model "$model_target"
      ;;
    *) die "Usage: colt.sh verify {ready|nccl|model TARGET [--model-path PATH]}" ;;
  esac
}
