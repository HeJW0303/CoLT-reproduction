#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env
require_free_gib 150

if [[ "$(cat "$DATA_ROOT/.colt_download_verified" 2>/dev/null || true)" != "$DATASET_REVISION" && ! -f "$DATA_ROOT/.colt_extraction_complete" ]]; then
  echo "Dataset download verification marker is missing. Run 02_download_assets.sh first." >&2
  exit 1
fi

zip_count="$(find "$DATA_ROOT" -type f -name '*.zip' | wc -l | tr -d ' ')"
if (( zip_count > 0 && zip_count <= 20 )); then
  echo "Testing CRC and structure for the remaining $zip_count ZIP files."
  while IFS= read -r -d '' zip_path; do
    unzip -tq "$zip_path" >/dev/null
  done < <(find "$DATA_ROOT" -type f -name '*.zip' -print0 | sort -z)

  while IFS= read -r -d '' zip_path; do
    echo "Extracting $zip_path"
    unzip -oq "$zip_path" -d "$(dirname "$zip_path")"
    rm -f "$zip_path"
  done < <(find "$DATA_ROOT" -type f -name '*.zip' -print0 | sort -z)
elif [[ "$zip_count" != "0" ]]; then
  echo "Expected at most 20 ZIP files; found $zip_count." >&2
  exit 1
fi

cat > "$DATA_ROOT/dataset_info.json" <<'JSON'
{
  "onethinker_sft_image": {
    "file_name": "colt_sft_image.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
JSON

python "$REPO_ROOT/scripts/a100/validate_dataset.py" --data-root "$DATA_ROOT"
printf '%s\n' "$DATASET_REVISION" > "$DATA_ROOT/.colt_extraction_complete"

remaining_zips="$(find "$DATA_ROOT" -type f -name '*.zip' | wc -l | tr -d ' ')"
if [[ "$remaining_zips" != "0" ]]; then
  echo "ZIP cleanup failed; $remaining_zips ZIP files remain." >&2
  exit 1
fi

df -h "$WORKSPACE_ROOT"
echo "Dataset extraction, registration, and full media-path validation completed."
