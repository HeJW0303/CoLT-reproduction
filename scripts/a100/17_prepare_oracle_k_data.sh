#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
require_workspace_layout
activate_colt_env

: "${ORACLE_K_TEACHER_MODEL:?Set ORACLE_K_TEACHER_MODEL to the OpenAI-compatible teacher model name.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL, for example http://127.0.0.1:8000/v1.}"

input_file="$DATA_ROOT/colt_sft_image.json"
if [[ -n "${ORACLE_K_LIMIT:-}" && -z "${ORACLE_K_OUTPUT_NAME:-}" ]]; then
  output_name="colt_sft_image_oracle_k_pilot_${ORACLE_K_START:-0}_${ORACLE_K_LIMIT}.json"
else
  output_name="${ORACLE_K_OUTPUT_NAME:-colt_sft_image_oracle_k.json}"
fi
output_file="$DATA_ROOT/$output_name"
if [[ -n "${ORACLE_K_LIMIT:-}" && -z "${ORACLE_K_DATASET_NAME:-}" ]]; then
  dataset_name="onethinker_sft_image_oracle_k_pilot"
else
  dataset_name="${ORACLE_K_DATASET_NAME:-onethinker_sft_image_oracle_k}"
fi
min_k="${COLT_ORACLE_K_MIN:-1}"
max_k="${COLT_ORACLE_K_MAX:-8}"
start="${ORACLE_K_START:-0}"
workers="${ORACLE_K_WORKERS:-8}"

args=(
  --input "$input_file"
  --output "$output_file"
  --model "$ORACLE_K_TEACHER_MODEL"
  --base-url "$OPENAI_BASE_URL"
  --api-key-env "${ORACLE_K_API_KEY_ENV:-OPENAI_API_KEY}"
  --min-k "$min_k"
  --max-k "$max_k"
  --start "$start"
  --workers "$workers"
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
  --original "$input_file" \
  --original-start "$start" \
  --min-k "$min_k" \
  --max-k "$max_k"
python "$REPO_ROOT/scripts/oracle_k/register_oracle_dataset.py" \
  --dataset-info "$DATA_ROOT/dataset_info.json" \
  --file-name "$output_name" \
  --dataset-name "$dataset_name"

echo "Oracle-K dataset is ready: $output_file"
echo "Dataset registration: $dataset_name"
