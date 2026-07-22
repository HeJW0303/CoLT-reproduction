#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env
require_free_gib 250

download_repo() {
  local repo_id="$1"
  local revision="$2"
  local local_dir="$3"
  shift 3
  mkdir -p "$local_dir"
  hf download "$repo_id" --revision "$revision" --local-dir "$local_dir" --max-workers 4 "$@"
}

download_repo \
  Qwen/Qwen3-VL-8B-Instruct \
  "$BASE_MODEL_REVISION" \
  "$BASE_MODEL_DIR"

download_repo \
  Qwen/Qwen3-0.6B \
  "$DECODER_MODEL_REVISION" \
  "$DECODER_MODEL_DIR"

download_repo \
  hulianyuyy/CoLT_Train_Dataset \
  "$DATASET_REVISION" \
  "$DATA_ROOT" \
  --repo-type dataset

python "$REPO_ROOT/scripts/lkl_8gpu/verify_hf_files.py" \
  --repo-id Qwen/Qwen3-VL-8B-Instruct \
  --repo-type model \
  --revision "$BASE_MODEL_REVISION" \
  --local-dir "$BASE_MODEL_DIR" \
  --suffix .safetensors

python "$REPO_ROOT/scripts/lkl_8gpu/verify_hf_files.py" \
  --repo-id Qwen/Qwen3-0.6B \
  --repo-type model \
  --revision "$DECODER_MODEL_REVISION" \
  --local-dir "$DECODER_MODEL_DIR" \
  --suffix .safetensors

zip_count="$(find "$DATA_ROOT" -type f -name '*.zip' | wc -l | tr -d ' ')"
if [[ "$zip_count" != "20" ]]; then
  echo "Expected 20 ZIP files, found $zip_count." >&2
  exit 1
fi

json_size="$(stat -c '%s' "$DATA_ROOT/colt_sft_image.json")"
if [[ "$json_size" != "284614647" ]]; then
  echo "Unexpected colt_sft_image.json size: $json_size" >&2
  exit 1
fi

python "$REPO_ROOT/scripts/lkl_8gpu/verify_hf_files.py" \
  --repo-id hulianyuyy/CoLT_Train_Dataset \
  --repo-type dataset \
  --revision "$DATASET_REVISION" \
  --local-dir "$DATA_ROOT" \
  --suffix .zip \
  --file colt_sft_image.json \
  --expected-count 21

printf '%s\n' "$BASE_MODEL_REVISION" > "$BASE_MODEL_DIR/.colt_verified_revision"
printf '%s\n' "$DECODER_MODEL_REVISION" > "$DECODER_MODEL_DIR/.colt_verified_revision"
printf '%s\n' "$DATASET_REVISION" > "$DATA_ROOT/.colt_download_verified"

echo "Models and compressed dataset are downloaded and verified."
