#!/usr/bin/env bash

download_dataset() {
  local dataset="$1" target partial expected_size expected_md5 url
  target="$EVAL_DATA_ROOT/$dataset.tsv"
  partial="$target.part"
  url="$(dataset_url "$dataset")" || die "Unknown evaluation dataset: $dataset"
  expected_size="$(dataset_size "$dataset")"
  expected_md5="$(dataset_md5 "$dataset")"
  if [[ -f "$target" \
        && "$(stat -c '%s' "$target")" == "$expected_size" \
        && "$(md5sum "$target" | awk '{print $1}')" == "$expected_md5" ]]; then
    echo "Dataset verified: $dataset"
    return
  fi
  rm -f "$target"
  echo "Downloading $dataset ($expected_size bytes)"
  if ! curl --fail --location --continue-at - --retry 5 --retry-delay 5 \
      --output "$partial" "$url"; then
    echo "TLS verification failed; retrying with -k while retaining size and MD5 checks." >&2
    curl -k --fail --location --continue-at - --retry 5 --retry-delay 5 \
      --output "$partial" "$url"
  fi
  if [[ "$(stat -c '%s' "$partial")" != "$expected_size" ]]; then
    rm -f "$partial"
    die "Size check failed: $partial"
  fi
  if [[ "$(md5sum "$partial" | awk '{print $1}')" != "$expected_md5" ]]; then
    rm -f "$partial"
    die "MD5 check failed: $partial"
  fi
  mv "$partial" "$target"
  echo "Dataset downloaded and verified: $target"
}

resolve_download_datasets() {
  if (( $# == 1 )) && dataset_group "$1" >/dev/null 2>&1; then
    mapfile -t DOWNLOAD_DATASETS < <(dataset_group "$1")
  else
    DOWNLOAD_DATASETS=("$@")
  fi
  local dataset
  for dataset in "${DOWNLOAD_DATASETS[@]}"; do
    [[ "$dataset" != "COLT_SMOKE_MMSTAR" ]] || dataset=MMStar
    dataset_url "$dataset" >/dev/null || die "Unknown evaluation dataset: $dataset"
  done
}

cmd_download() {
  (( $# > 0 )) || die "Usage: colt.sh download {all8|remaining7|phase1|phase2|phase3|DATASET...}"
  runtime_init
  require_workspace_layout
  activate_colt_env
  require_free_gib 10
  EVAL_DATA_ROOT="${COLT_EVAL_DATA_ROOT:-$EVAL_ROOT/LMUData}"
  mkdir -p "$EVAL_DATA_ROOT"
  resolve_download_datasets "$@"
  local dataset
  for dataset in "${DOWNLOAD_DATASETS[@]}"; do
    [[ "$dataset" != "COLT_SMOKE_MMSTAR" ]] || dataset=MMStar
    download_dataset "$dataset"
  done
}

prepare_smoke_dataset() {
  local source="$EVAL_DATA_ROOT/MMStar.tsv" target="$EVAL_DATA_ROOT/COLT_SMOKE_MMSTAR.tsv"
  python - "$source" "$target" <<'PY'
import sys
import pandas as pd

source, target = sys.argv[1:]
data = pd.read_csv(source, sep="\t", nrows=8)
if len(data) != 8:
    raise RuntimeError(f"Expected 8 smoke rows, found {len(data)}")
data.to_csv(target, sep="\t", index=False)
print(f"Smoke dataset ready: {target}")
PY
}

cmd_eval() {
  local target="${1:-}" group="${2:-}"
  [[ -n "$target" && -n "$group" ]] || die \
    "Usage: colt.sh eval TARGET GROUP [--model-path PATH] [--gpus CSV] [--workers N] [--generation MODE]"
  shift 2

  local cli_model_path="" gpu_csv="" workers="${VLMEVAL_WORKERS_PER_GPU:-3}"
  local generation="${COLT_GENERATION_MODE:-}"
  local prefetch="${VLMEVAL_PREFETCH:-1}" empty_cache="${VLMEVAL_EMPTY_CACHE_EVERY_N:-0}"
  local backend="${VLMEVAL_DIST_BACKEND:-gloo}" reseed="${COLT_RESEED_PER_SAMPLE:-1}"
  local verbose="${EVAL_VERBOSE:-0}" reuse=1
  while (( $# > 0 )); do
    case "$1" in
      --model-path) [[ $# -ge 2 ]] || die "--model-path requires a value"; cli_model_path="$2"; shift 2 ;;
      --gpus) [[ $# -ge 2 ]] || die "--gpus requires a value"; gpu_csv="$2"; shift 2 ;;
      --workers) [[ $# -ge 2 ]] || die "--workers requires a value"; workers="$2"; shift 2 ;;
      --generation) [[ $# -ge 2 ]] || die "--generation requires a value"; generation="$2"; shift 2 ;;
      --prefetch) [[ $# -ge 2 ]] || die "--prefetch requires a value"; prefetch="$2"; shift 2 ;;
      --empty-cache-every) [[ $# -ge 2 ]] || die "--empty-cache-every requires a value"; empty_cache="$2"; shift 2 ;;
      --dist-backend) [[ $# -ge 2 ]] || die "--dist-backend requires a value"; backend="$2"; shift 2 ;;
      --reseed-per-sample) [[ $# -ge 2 ]] || die "--reseed-per-sample requires a value"; reseed="$2"; shift 2 ;;
      --verbose) verbose=1; shift ;;
      --no-reuse) reuse=0; shift ;;
      *) die "Unknown eval option: $1" ;;
    esac
  done

  case "$target" in codefaithful|paper-faithful|oracle-k|official|baseline) ;; *) die "Unknown target: $target" ;; esac
  case "$generation" in ""|official|respect-args) ;; respect_args) generation=respect-args ;; *) die "Generation must be official or respect-args" ;; esac
  if [[ "$target" == baseline ]]; then
    [[ -z "$generation" || "$generation" == respect-args ]] || die \
      "baseline only supports greedy + 8192; omit --generation or use --generation respect-args"
    generation=respect-args
  else
    generation="${generation:-official}"
  fi
  [[ "$workers" =~ ^[1-9][0-9]*$ ]] || die "--workers must be a positive integer"
  [[ "$prefetch" == 0 || "$prefetch" == 1 ]] || die "--prefetch must be 0 or 1"
  [[ "$empty_cache" =~ ^[0-9]+$ ]] || die "--empty-cache-every must be a non-negative integer"
  [[ "$backend" == gloo || "$backend" == nccl ]] || die "--dist-backend must be gloo or nccl"
  [[ "$reseed" == 0 || "$reseed" == 1 ]] || die "--reseed-per-sample must be 0 or 1"

  runtime_init
  require_workspace_layout
  activate_colt_env
  validate_gpu_profile
  gpu_csv="${gpu_csv:-${COLT_EVAL_GPUS:-$COLT_DEFAULT_EVAL_GPUS}}"
  parse_gpu_csv "$gpu_csv"
  require_selected_gpus_free

  EVAL_DATA_ROOT="${COLT_EVAL_DATA_ROOT:-$EVAL_ROOT/LMUData}"
  EVAL_OUTPUT_ROOT="${COLT_EVAL_OUTPUT_ROOT:-$EVAL_ROOT/results}"
  EVAL_LOG_ROOT="${COLT_EVAL_LOG_ROOT:-$LOG_ROOT/eval}"
  mkdir -p "$EVAL_DATA_ROOT" "$EVAL_OUTPUT_ROOT" "$EVAL_LOG_ROOT"
  resolve_eval_model "$target" "$cli_model_path"
  verify_eval_model "$target"

  local -a datasets
  mapfile -t datasets < <(dataset_group "$group")
  (( ${#datasets[@]} > 0 )) || die "Unknown dataset group: $group"
  if [[ "$group" == smoke ]]; then
    download_dataset MMStar
    prepare_smoke_dataset
  else
    local dataset
    for dataset in "${datasets[@]}"; do download_dataset "$dataset"; done
  fi

  export LMUData="$EVAL_DATA_ROOT"
  export PYTHONPATH="$VLMEVAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1 VLMEVAL_FAIL_ON_ERROR=1 VLMEVAL_ATOMIC_WRITES=1
  export COLT_DECODER_MODEL_PATH="$DECODER_MODEL_DIR"
  touch "$VLMEVAL_ROOT/.env"
  python -m pip check

  local adapter_mode=colt model_name="Qwen3-VL-8B-Instruct-COLT" paper_profile=colt
  local effective_sample=True effective_tokens=256
  local source_model_path="$EVAL_MODEL_PATH" runtime_model_path="$EVAL_MODEL_PATH"
  local run_id="$(date +%Y%m%d_%H%M%S)"
  if [[ "$target" == baseline ]]; then
    adapter_mode=baseline
    model_name="Qwen3-VL-8B-Instruct-BASE-COT"
    paper_profile=qwen3vl_cot
    effective_sample=False
    effective_tokens=8192
    reseed=0
    export COLT_DISABLE_LATENT_REASONING=1
    export COLT_EXPECT_BASE_MAX_NEW_TOKENS=8192
    export QWEN3_VL_BASE_MODEL_PATH="$runtime_model_path"
    export COLT_RESPECT_GENERATION_ARGS=1
  else
    unset COLT_DISABLE_LATENT_REASONING
    if [[ "$generation" == respect-args ]]; then
      create_generation_overlay "$source_model_path" "$run_id"
      runtime_model_path="$EVAL_RUNTIME_MODEL_PATH"
      effective_sample=False
      effective_tokens=8192
      export COLT_RESPECT_GENERATION_ARGS=1
    else
      export COLT_RESPECT_GENERATION_ARGS=0
    fi
    export COLT_EVAL_MODEL_PATH="$runtime_model_path"
  fi
  python "$COLT_SCRIPT_ROOT/tools/verify_eval_env.py" --repo-root "$REPO_ROOT" --adapter "$adapter_mode"

  local nproc=$(( ${#COLT_GPU_IDS[@]} * workers )) fingerprint profile eval_id work_dir log_file
  fingerprint="$(python "$COLT_SCRIPT_ROOT/tools/eval_fingerprint.py" \
    --repo-root "$REPO_ROOT" --model-dir "$source_model_path" \
    --setting "target=$target" --setting "group=$group" --setting "generation=$generation" \
    --setting "seed=${COLT_EVAL_SEED:-1234}" --setting "workers=$workers" \
    --setting "prefetch=$prefetch" --setting "empty_cache=$empty_cache" \
    --setting "backend=$backend" --setting "reseed=$reseed")"
  profile="replicas${nproc}_w${workers}_p${prefetch}_c${empty_cache}_r${reseed}_${generation}_seed${COLT_EVAL_SEED:-1234}_${fingerprint}"
  work_dir="$EVAL_OUTPUT_ROOT/$target/$group/$profile"
  eval_id="$(printf '%s_%s' "$target" "$profile" | tr '[:lower:]-' '[:upper:]_')"
  log_file="$EVAL_LOG_ROOT/${target}_${group}_${profile}_${run_id}.log"
  mkdir -p "$work_dir"
  exec > >(tee -a "$log_file") 2>&1

  export CUDA_VISIBLE_DEVICES="$gpu_csv" MMEVAL_ROOT="$work_dir" VLMEVAL_EVAL_ID="$eval_id"
  export PRED_FORMAT=xlsx EVAL_FORMAT=csv DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"
  export VLMEVAL_WORKERS_PER_GPU="$workers" VLMEVAL_PREFETCH="$prefetch"
  export VLMEVAL_EMPTY_CACHE_EVERY_N="$empty_cache" VLMEVAL_DIST_BACKEND="$backend"
  export COLT_RESEED_PER_SAMPLE="$reseed" COLT_EVAL_SEED="${COLT_EVAL_SEED:-1234}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE

  echo "Target: $target"
  echo "Datasets: ${datasets[*]}"
  echo "Model source: $EVAL_MODEL_PATH_SOURCE"
  echo "Resolved model: $source_model_path"
  [[ "$runtime_model_path" == "$source_model_path" ]] || echo "Runtime overlay: $runtime_model_path"
  echo "Physical GPUs: $gpu_csv"
  echo "Workers per GPU: $workers; total workers: $nproc"
  echo "Prefetch: $prefetch; empty_cache every N: $empty_cache; backend: $backend"
  echo "requested_do_sample=False"
  echo "effective_do_sample=$effective_sample"
  echo "requested_max_new_tokens=8192"
  echo "effective_max_new_tokens=$effective_tokens"
  echo "Fingerprint: $fingerprint"
  echo "Results: $work_dir"
  echo "Log: $log_file"

  cd "$VLMEVAL_ROOT"
  local -a args=(
    --standalone --nnodes=1 --nproc_per_node="$nproc" --max_restarts=0
    run.py --data "${datasets[@]}" --model "$model_name" --work-dir "$work_dir"
    --mode all --judge exact_matching
  )
  (( reuse == 0 )) || args+=(--reuse)
  (( verbose == 0 )) || args+=(--verbose)
  torchrun "${args[@]}"

  if [[ "$group" == smoke ]]; then
    python "$COLT_SCRIPT_ROOT/tools/validate_smoke.py" "$work_dir" "$model_name" "$eval_id"
  else
    python "$COLT_SCRIPT_ROOT/tools/validate_eval_suite.py" \
      "$work_dir" "$model_name" "$eval_id" "$EVAL_DATA_ROOT" "${datasets[@]}" \
      --paper-profile "$paper_profile"
  fi
  echo "Evaluation completed and validated: target=$target group=$group"
}
