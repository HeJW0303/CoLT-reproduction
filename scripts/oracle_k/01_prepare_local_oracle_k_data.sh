#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_SOURCE_FILE="$REPO_ROOT/data/oracle_k_source/colt_sft_image.json"
SOURCE_FILE="${ORACLE_K_SOURCE_FILE:-$DEFAULT_SOURCE_FILE}"
OUTPUT_ROOT="${ORACLE_K_LOCAL_OUTPUT_ROOT:-$REPO_ROOT/data/oracle_k_local}"

: "${ORACLE_K_TEACHER_MODEL:?Set ORACLE_K_TEACHER_MODEL to the OpenAI-compatible teacher model name.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL, for example https://provider.example/v1.}"

if [[ "$SOURCE_FILE" == "$DEFAULT_SOURCE_FILE" ]]; then
  python "$REPO_ROOT/scripts/oracle_k/download_text_data.py" --output-dir "$(dirname "$SOURCE_FILE")"
elif [[ ! -f "$SOURCE_FILE" ]]; then
  echo "Missing custom Oracle-K source JSON: $SOURCE_FILE" >&2
  exit 1
fi

start="${ORACLE_K_START:-0}"
min_k="${COLT_ORACLE_K_MIN:-1}"
max_k="${COLT_ORACLE_K_MAX:-8}"
dataset_name="${ORACLE_K_DATASET_NAME:-onethinker_sft_image_oracle_k}"
mkdir -p "$OUTPUT_ROOT"

if [[ -n "${ORACLE_K_LIMIT:-}" ]]; then
  output_file="$OUTPUT_ROOT/colt_sft_image_oracle_k_pilot_${start}_${ORACLE_K_LIMIT}.json"
  dataset_name="${ORACLE_K_DATASET_NAME:-onethinker_sft_image_oracle_k_pilot}"
else
  output_file="$OUTPUT_ROOT/colt_sft_image_oracle_k.json"
fi

args=(
  --input "$SOURCE_FILE"
  --output "$output_file"
  --model "$ORACLE_K_TEACHER_MODEL"
  --base-url "$OPENAI_BASE_URL"
  --api-key-env "${ORACLE_K_API_KEY_ENV:-OPENAI_API_KEY}"
  --min-k "$min_k"
  --max-k "$max_k"
  --start "$start"
  --workers "${ORACLE_K_WORKERS:-8}"
  --timeout "${ORACLE_K_TIMEOUT:-180}"
  --max-retries "${ORACLE_K_MAX_RETRIES:-3}"
  --retry-backoff "${ORACLE_K_RETRY_BACKOFF:-2}"
  --max-output-tokens "${ORACLE_K_MAX_OUTPUT_TOKENS:-1024}"
)
if [[ -n "${ORACLE_K_REASONING_EFFORT:-}" ]]; then
  args+=(--reasoning-effort "$ORACLE_K_REASONING_EFFORT")
fi
if [[ -n "${ORACLE_K_LIMIT:-}" ]]; then
  args+=(--limit "$ORACLE_K_LIMIT")
fi
if [[ "${ORACLE_K_RESUME:-0}" == "1" ]]; then
  args+=(--resume)
fi
if [[ "${ORACLE_K_NO_RESPONSE_FORMAT:-0}" == "1" ]]; then
  args+=(--no-response-format)
fi

python "$REPO_ROOT/scripts/oracle_k/segment_teacher_blocks.py" "${args[@]}"
python "$REPO_ROOT/scripts/oracle_k/validate_oracle_blocks.py" \
  --input "$output_file" \
  --original "$SOURCE_FILE" \
  --original-start "$start" \
  --min-k "$min_k" \
  --max-k "$max_k"
python "$REPO_ROOT/scripts/oracle_k/package_oracle_dataset.py" \
  --original "$SOURCE_FILE" \
  --annotated "$output_file" \
  --state-file "$output_file.state.jsonl" \
  --output-dir "$OUTPUT_ROOT/transfer" \
  --dataset-name "$dataset_name" \
  --original-start "$start" \
  --min-k "$min_k" \
  --max-k "$max_k"

echo "Local Oracle-K dataset is ready: $output_file"
echo "Upload the matching .json.gz and .manifest.json from: $OUTPUT_ROOT/transfer"
