#!/usr/bin/env bash

set -euo pipefail

EXTERNAL_JUDGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LKL_8GPU_ROOT="$(cd "$EXTERNAL_JUDGE_ROOT/.." && pwd)"
REPO_ROOT="$(cd "$LKL_8GPU_ROOT/../.." && pwd)"
COLT_LAUNCHER="$LKL_8GPU_ROOT/colt.sh"
JUDGE_CONFIG="$EXTERNAL_JUDGE_ROOT/judge_config.json"
CODEX_API_LOADER="$EXTERNAL_JUDGE_ROOT/load_codex_api.py"
DEFAULT_CODEX_CONFIG="/home/zpw/.codex/config.toml"
DEFAULT_CODEX_AUTH="/home/zpw/.codex/auth.json"
EXTERNAL_DATASETS=(MathVista_MINI MathVerse_MINI MMVet)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
External-judge evaluation for MathVista_MINI, MathVerse_MINI, and MMVet.

Usage:
  run.sh download [DATASET...]
  run.sh eval TARGET [--datasets CSV] [--api-nproc N] [--judge-retry N]
                     [--api-provider deepseek|codex] [--codex-config PATH]
                     [--codex-auth PATH] [--dry-run]
                     [regular colt.sh eval options]
  run.sh all TARGET [eval options]
  run.sh validate [validate_results.py options]

Judge: deepseek-v4-flash over Chat Completions with low reasoning effort and
thinking disabled. The default API concurrency is 8. The DeepSeek key is read
only from DEEPSEEK_API_KEY at execution time.
EOF
}

validate_dataset_name() {
  case "$1" in
    MathVista_MINI|MathVerse_MINI|MMVet) ;;
    *) die "Unsupported external-judge dataset: $1" ;;
  esac
}

parse_dataset_csv() {
  local csv="$1" dataset
  IFS=',' read -r -a SELECTED_DATASETS <<< "$csv"
  (( ${#SELECTED_DATASETS[@]} > 0 )) || die "--datasets must not be empty"
  declare -A seen=()
  for dataset in "${SELECTED_DATASETS[@]}"; do
    [[ -n "$dataset" ]] || die "--datasets contains an empty name"
    validate_dataset_name "$dataset"
    [[ -z "${seen[$dataset]:-}" ]] || die "Duplicate dataset: $dataset"
    seen[$dataset]=1
  done
}

run_download() {
  local -a datasets=("$@")
  (( ${#datasets[@]} > 0 )) || datasets=("${EXTERNAL_DATASETS[@]}")
  local dataset
  for dataset in "${datasets[@]}"; do validate_dataset_name "$dataset"; done
  bash "$COLT_LAUNCHER" download "${datasets[@]}"
}

print_dry_run() {
  local value
  echo "Dry run; no API credentials were read and no evaluation was started."
  echo "Environment:"
  for value in "$@"; do printf '  %q\n' "$value"; done
}

run_eval() {
  local target="${1:-}"
  [[ -n "$target" ]] || die "eval requires a target"
  shift

  SELECTED_DATASETS=("${EXTERNAL_DATASETS[@]}")
  local api_nproc=8 judge_retry=5 dry_run=0 api_provider=deepseek
  local codex_config="$DEFAULT_CODEX_CONFIG" codex_auth="$DEFAULT_CODEX_AUTH"
  local -a eval_options=()
  while (( $# > 0 )); do
    case "$1" in
      --datasets) [[ $# -ge 2 ]] || die "--datasets requires a value"; parse_dataset_csv "$2"; shift 2 ;;
      --api-nproc) [[ $# -ge 2 ]] || die "--api-nproc requires a value"; api_nproc="$2"; shift 2 ;;
      --judge-retry) [[ $# -ge 2 ]] || die "--judge-retry requires a value"; judge_retry="$2"; shift 2 ;;
      --api-provider) [[ $# -ge 2 ]] || die "--api-provider requires a value"; api_provider="$2"; shift 2 ;;
      --codex-config) [[ $# -ge 2 ]] || die "--codex-config requires a value"; codex_config="$2"; shift 2 ;;
      --codex-auth) [[ $# -ge 2 ]] || die "--codex-auth requires a value"; codex_auth="$2"; shift 2 ;;
      --dry-run) dry_run=1; shift ;;
      *) eval_options+=("$1"); shift ;;
    esac
  done
  [[ "$api_nproc" =~ ^[1-9][0-9]*$ ]] || die "--api-nproc must be a positive integer"
  [[ "$judge_retry" =~ ^[1-9][0-9]*$ ]] || die "--judge-retry must be a positive integer"
  [[ "$api_provider" == deepseek || "$api_provider" == codex ]] || die \
    "--api-provider must be deepseek or codex"

  local -a resolved_config
  mapfile -t resolved_config < <(
    python3 "$EXTERNAL_JUDGE_ROOT/resolve_judge_config.py" "$JUDGE_CONFIG" "${SELECTED_DATASETS[@]}"
  )
  (( ${#resolved_config[@]} == 3 )) || die "Failed to resolve judge configuration"
  local judge_model="${resolved_config[0]}"
  local judge_args="${resolved_config[1]}"
  local judge_profile="${resolved_config[2]}"
  local dataset_csv
  dataset_csv="$(IFS=,; echo "${SELECTED_DATASETS[*]}")"

  local output_root="${COLT_EXTERNAL_JUDGE_OUTPUT_ROOT:-$REPO_ROOT/eval/external_judge/results}"
  local log_root="${COLT_EXTERNAL_JUDGE_LOG_ROOT:-$REPO_ROOT/logs/eval/external_judge}"
  local -a environment=(
    "COLT_EVAL_DATASETS=$dataset_csv"
    "COLT_EVAL_JUDGE=$judge_model"
    "COLT_EVAL_JUDGE_ARGS=$judge_args"
    "COLT_EVAL_JUDGE_NPROC=$api_nproc"
    "COLT_EVAL_JUDGE_RETRY=$judge_retry"
    "COLT_EVAL_JUDGE_PROFILE=$judge_profile"
    "COLT_EVAL_RESULT_KIND=external-judge"
    "COLT_EVAL_OUTPUT_ROOT=$output_root"
    "COLT_EVAL_LOG_ROOT=$log_root"
  )
  local -a command=(
    python "$CODEX_API_LOADER" --provider "$api_provider"
  )
  if [[ "$api_provider" == codex ]]; then
    command+=(--config "$codex_config" --auth "$codex_auth")
  fi
  command+=(--
    bash "$COLT_LAUNCHER" eval "$target" external-judge "${eval_options[@]}"
  )

  if (( dry_run )); then
    print_dry_run "${environment[@]}"
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi

  if [[ "$api_provider" == deepseek ]]; then
    [[ "${DEEPSEEK_API_KEY:-}" == sk-* ]] || die "DEEPSEEK_API_KEY must be set for DeepSeek evaluation"
  else
    [[ -f "$codex_config" ]] || die "Missing Codex config: $codex_config"
    [[ -f "$codex_auth" ]] || die "Missing Codex auth: $codex_auth"
  fi
  # shellcheck disable=SC1091
  source "$LKL_8GPU_ROOT/lib/runtime.sh"
  runtime_init
  activate_colt_env
  env "${environment[@]}" "${command[@]}"
}

command_name="${1:-help}"
shift || true
case "$command_name" in
  help|-h|--help) usage ;;
  download) run_download "$@" ;;
  eval) run_eval "$@" ;;
  all) run_eval "$@" ;;
  validate) exec python3 "$EXTERNAL_JUDGE_ROOT/validate_results.py" "$@" ;;
  *) usage >&2; die "Unknown command: $command_name" ;;
esac
