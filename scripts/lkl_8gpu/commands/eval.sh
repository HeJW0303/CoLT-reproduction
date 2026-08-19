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
  local latent_transition="${COLT_INFERENCE_LATENT_TRANSITION:-training-consistent}"
  local prefetch="${VLMEVAL_PREFETCH:-1}" empty_cache="${VLMEVAL_EMPTY_CACHE_EVERY_N:-0}"
  local backend="${VLMEVAL_DIST_BACKEND:-gloo}" reseed="${COLT_RESEED_PER_SAMPLE:-1}"
  local empty_response_policy="${COLT_EMPTY_RESPONSE_POLICY:-allow}"
  local log_label="${COLT_EVAL_LOG_LABEL:-$target}"
  local judge="${COLT_EVAL_JUDGE:-exact_matching}"
  local judge_args="${COLT_EVAL_JUDGE_ARGS:-}"
  local judge_nproc="${COLT_EVAL_JUDGE_NPROC:-4}"
  local judge_retry="${COLT_EVAL_JUDGE_RETRY:-3}"
  local judge_profile="${COLT_EVAL_JUDGE_PROFILE:-$judge}"
  local result_kind="${COLT_EVAL_RESULT_KIND:-standard}"
  local verbose="${EVAL_VERBOSE:-0}" reuse=1
  while (( $# > 0 )); do
    case "$1" in
      --model-path) [[ $# -ge 2 ]] || die "--model-path requires a value"; cli_model_path="$2"; shift 2 ;;
      --gpus) [[ $# -ge 2 ]] || die "--gpus requires a value"; gpu_csv="$2"; shift 2 ;;
      --workers) [[ $# -ge 2 ]] || die "--workers requires a value"; workers="$2"; shift 2 ;;
      --generation) [[ $# -ge 2 ]] || die "--generation requires a value"; generation="$2"; shift 2 ;;
      --latent-transition) [[ $# -ge 2 ]] || die "--latent-transition requires a value"; latent_transition="$2"; shift 2 ;;
      --prefetch) [[ $# -ge 2 ]] || die "--prefetch requires a value"; prefetch="$2"; shift 2 ;;
      --empty-cache-every) [[ $# -ge 2 ]] || die "--empty-cache-every requires a value"; empty_cache="$2"; shift 2 ;;
      --dist-backend) [[ $# -ge 2 ]] || die "--dist-backend requires a value"; backend="$2"; shift 2 ;;
      --reseed-per-sample) [[ $# -ge 2 ]] || die "--reseed-per-sample requires a value"; reseed="$2"; shift 2 ;;
      --empty-response-policy) [[ $# -ge 2 ]] || die "--empty-response-policy requires a value"; empty_response_policy="$2"; shift 2 ;;
      --verbose) verbose=1; shift ;;
      --no-reuse) reuse=0; shift ;;
      *) die "Unknown eval option: $1" ;;
    esac
  done

  case "$target" in codefaithful|paper-faithful|oracle-k|official|baseline) ;; *) die "Unknown target: $target" ;; esac
  [[ "$log_label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die \
    "COLT_EVAL_LOG_LABEL must contain only letters, digits, dot, underscore, or hyphen"
  case "$generation" in ""|official|respect-args) ;; respect_args) generation=respect-args ;; *) die "Generation must be official or respect-args" ;; esac
  case "$latent_transition" in official|training-consistent) ;; *) die \
    "Latent transition must be official or training-consistent" ;; esac
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
  [[ "$empty_response_policy" == allow || "$empty_response_policy" == prevent ]] || die \
    "--empty-response-policy must be allow or prevent"
  [[ "$judge" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die \
    "COLT_EVAL_JUDGE contains unsupported characters"
  [[ "$judge_profile" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die \
    "COLT_EVAL_JUDGE_PROFILE contains unsupported characters"
  [[ "$judge_nproc" =~ ^[1-9][0-9]*$ ]] || die "COLT_EVAL_JUDGE_NPROC must be a positive integer"
  [[ "$judge_retry" =~ ^[1-9][0-9]*$ ]] || die "COLT_EVAL_JUDGE_RETRY must be a positive integer"
  case "$result_kind" in standard|external-judge) ;; *) die \
    "COLT_EVAL_RESULT_KIND must be standard or external-judge" ;; esac
  if [[ "$result_kind" == external-judge && "$judge" == exact_matching ]]; then
    die "External-judge results require a non-exact judge"
  fi
  # The textual-CoT baseline has no CoLT latent reasoning; transition is moot.
  if [[ "$target" == baseline ]]; then
    latent_transition=official
  fi
  if [[ "$target" == baseline && "$empty_response_policy" != allow ]]; then
    die "baseline does not support --empty-response-policy prevent"
  fi
  if [[ "$target" == baseline && "$latent_transition" != official ]]; then
    die "baseline does not use latent reasoning and only supports --latent-transition official"
  fi
  if [[ -z "${COLT_EVAL_LOG_LABEL:-}" && "$latent_transition" != official ]]; then
    log_label="${target}-${latent_transition}"
  fi

  runtime_init
  require_workspace_layout
  activate_colt_env
  validate_gpu_profile
  gpu_csv="${gpu_csv:-${COLT_EVAL_GPUS:-$COLT_DEFAULT_EVAL_GPUS}}"
  parse_gpu_csv "$gpu_csv"
  maybe_check_selected_gpus_free

  EVAL_DATA_ROOT="${COLT_EVAL_DATA_ROOT:-$EVAL_ROOT/LMUData}"
  EVAL_OUTPUT_ROOT="${COLT_EVAL_OUTPUT_ROOT:-$EVAL_ROOT/results}"
  EVAL_LOG_ROOT="${COLT_EVAL_LOG_ROOT:-$LOG_ROOT/eval}"
  mkdir -p "$EVAL_DATA_ROOT" "$EVAL_OUTPUT_ROOT" "$EVAL_LOG_ROOT"
  resolve_eval_model "$target" "$cli_model_path"
  verify_eval_model "$target"

  local -a datasets
  if [[ "$result_kind" == external-judge && -n "${COLT_EVAL_DATASETS:-}" ]]; then
    IFS=',' read -r -a datasets <<< "$COLT_EVAL_DATASETS"
    local configured_dataset
    for configured_dataset in "${datasets[@]}"; do
      dataset_group "$configured_dataset" >/dev/null || die \
        "Unknown dataset in COLT_EVAL_DATASETS: $configured_dataset"
    done
  else
    mapfile -t datasets < <(dataset_group "$group")
  fi
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
  export COLT_INFERENCE_LATENT_TRANSITION="$latent_transition"
  if [[ "$empty_response_policy" == prevent ]]; then
    export COLT_PREVENT_EMPTY_RESPONSE=1
  else
    export COLT_PREVENT_EMPTY_RESPONSE=0
  fi
  touch "$VLMEVAL_ROOT/.env"
  python -m pip check

  local adapter_mode=colt model_name="Qwen3-VL-8B-Instruct-COLT" paper_profile=colt
  local requested_max_new_tokens="${COLT_EVAL_MAX_NEW_TOKENS:-8192}"
  [[ "$requested_max_new_tokens" =~ ^[1-9][0-9]*$ ]] || die \
    "COLT_EVAL_MAX_NEW_TOKENS must be a positive integer"
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
    if [[ "$generation" == respect-args || "$empty_response_policy" == prevent || "$latent_transition" != official ]]; then
      local overlay_variant="${generation//-/_}_${empty_response_policy}_lt_${latent_transition//-/_}"
      create_generation_overlay "$source_model_path" "$run_id" "$overlay_variant"
      runtime_model_path="$EVAL_RUNTIME_MODEL_PATH"
    fi
    if [[ "$generation" == respect-args ]]; then
      effective_sample=False
      effective_tokens="$requested_max_new_tokens"
      export COLT_RESPECT_GENERATION_ARGS=1
    else
      export COLT_RESPECT_GENERATION_ARGS=0
    fi
    export COLT_EVAL_MODEL_PATH="$runtime_model_path"
  fi
  python "$COLT_SCRIPT_ROOT/tools/verify_eval_env.py" --repo-root "$REPO_ROOT" --adapter "$adapter_mode"

  local nproc=$(( ${#COLT_GPU_IDS[@]} * workers )) fingerprint profile eval_id work_dir log_dir log_file
  local dataset_fingerprint oracle_k_forced_k oracle_k_forced_transition_steps
  dataset_fingerprint="$(IFS=,; echo "${datasets[*]}")"
  oracle_k_forced_k=""
  oracle_k_forced_transition_steps=""
  if [[ "$target" == oracle-k ]]; then
    oracle_k_forced_k="${COLT_INFERENCE_K:-}"
    oracle_k_forced_transition_steps="${COLT_INFERENCE_TRANSITION_STEPS:-}"
  fi
  local generation_label
  generation_label="$(generation_log_label "$generation" "$empty_response_policy")"
  fingerprint="$(python "$COLT_SCRIPT_ROOT/tools/eval_fingerprint.py" \
    --repo-root "$REPO_ROOT" --model-dir "$source_model_path" \
    --setting "target=$target" --setting "group=$group" --setting "datasets=$dataset_fingerprint" \
    --setting "generation=$generation" \
    --setting "max_new_tokens=$requested_max_new_tokens" \
    --setting "latent_transition=$latent_transition" \
    --setting "latent_intervention=${COLT_LATENT_INTERVENTION:-none}" \
    --setting "answer_visibility=${COLT_ANSWER_VISIBILITY:-full}" \
    --setting "seed=${COLT_EVAL_SEED:-1234}" --setting "workers=$workers" \
    --setting "prefetch=$prefetch" --setting "empty_cache=$empty_cache" \
    --setting "backend=$backend" --setting "reseed=$reseed" \
    --setting "empty_response_policy=$empty_response_policy" \
    --setting "oracle_k_forced_k=${oracle_k_forced_k:-auto}" \
    --setting "oracle_k_forced_transition_steps=${oracle_k_forced_transition_steps:-auto}" \
    --setting "judge=$judge" --setting "judge_args=$judge_args" \
    --setting "judge_nproc=$judge_nproc" --setting "judge_retry=$judge_retry" \
    --setting "result_kind=$result_kind")"
  profile="replicas${nproc}_w${workers}_p${prefetch}_c${empty_cache}_r${reseed}_${generation}_${empty_response_policy}_lt${latent_transition//-/_}_av${COLT_ANSWER_VISIBILITY:-full}_li${COLT_LATENT_INTERVENTION:-none}_mn${requested_max_new_tokens}_seed${COLT_EVAL_SEED:-1234}_${fingerprint}"
  if [[ "$result_kind" == external-judge ]]; then
    profile="${profile}_judge_${judge_profile}"
  fi
  work_dir="$EVAL_OUTPUT_ROOT/$target/$group/$profile"
  eval_id="$(printf '%s_%s' "$target" "$profile" | tr '[:lower:]-' '[:upper:]_')"
  log_dir="$EVAL_LOG_ROOT/$log_label"
  log_file="$log_dir/${log_label}_${group}_${generation_label}_${run_id}.log"
  mkdir -p "$work_dir" "$log_dir"
  exec > >(tee -a "$log_file") 2>&1

  export CUDA_VISIBLE_DEVICES="$gpu_csv" MMEVAL_ROOT="$work_dir" VLMEVAL_EVAL_ID="$eval_id"
  export PRED_FORMAT=xlsx EVAL_FORMAT=csv DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"
  export VLMEVAL_WORKERS_PER_GPU="$workers" VLMEVAL_PREFETCH="$prefetch"
  export VLMEVAL_EMPTY_CACHE_EVERY_N="$empty_cache" VLMEVAL_DIST_BACKEND="$backend"
  export COLT_RESEED_PER_SAMPLE="$reseed" COLT_EVAL_SEED="${COLT_EVAL_SEED:-1234}"
  export COLT_LATENT_TEMPERATURE="${COLT_LATENT_TEMPERATURE:-0.0}"
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
  echo "Empty response policy: $empty_response_policy"
  if [[ "$latent_transition" == training-consistent ]]; then
    echo "Inference latent transition: training-consistent (initial=identity, recurrent=hidden+alpha*prj(hidden))"
  else
    echo "Inference latent transition: official (initial=prj(hidden), recurrent=prj(hidden))"
  fi
  echo "requested_do_sample=False"
  echo "effective_do_sample=$effective_sample"
  echo "requested_max_new_tokens=$requested_max_new_tokens"
  echo "effective_max_new_tokens=$effective_tokens"
  if [[ "$target" == oracle-k ]]; then
    if [[ -n "$oracle_k_forced_k" ]]; then
      echo "Oracle-K inference control: forced K=$oracle_k_forced_k"
    elif [[ -n "$oracle_k_forced_transition_steps" ]]; then
      echo "Oracle-K inference control: forced transition steps=$oracle_k_forced_transition_steps"
    else
      echo "Oracle-K inference control: checkpoint default/dynamic policy"
    fi
  fi
  echo "Fingerprint: $fingerprint"
  echo "Judge: $judge; API workers: $judge_nproc; retries: $judge_retry"
  echo "Results: $work_dir"
  echo "Log: $log_file"

  cd "$VLMEVAL_ROOT"
  local -a args=(
    --standalone --nnodes=1 --nproc_per_node="$nproc" --max_restarts=0
    run.py --data "${datasets[@]}" --model "$model_name" --work-dir "$work_dir"
    --judge "$judge" --api-nproc "$judge_nproc" --retry "$judge_retry"
  )
  [[ -z "$judge_args" ]] || args+=(--judge-args "$judge_args")
  (( reuse == 0 )) || args+=(--reuse)
  (( verbose == 0 )) || args+=(--verbose)

  if [[ "$result_kind" == external-judge ]]; then
    echo "External judge mode: distributed inference followed by single-process judge evaluation."
    torchrun "${args[@]}" --mode infer

    local judge_result_dir="$work_dir/$model_name/$eval_id"
    python "$COLT_SCRIPT_ROOT/external_judge/prepare_judge_resume.py" \
      --result-dir "$judge_result_dir" --model-name "$model_name" --judge-model "$judge" \
      "${datasets[@]}"

    echo "External judge mode: starting single-process judge evaluation."
    local -a single_process_args=(
      --data "${datasets[@]}" --model "$model_name" --work-dir "$work_dir"
      --mode eval --judge "$judge" --api-nproc "$judge_nproc" --retry "$judge_retry"
    )
    [[ -z "$judge_args" ]] || single_process_args+=(--judge-args "$judge_args")
    (( reuse == 0 )) || single_process_args+=(--reuse)
    (( verbose == 0 )) || single_process_args+=(--verbose)
    env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
      VLMEVAL_WORKERS_PER_GPU=1 \
      python run.py "${single_process_args[@]}"
  else
    torchrun "${args[@]}" --mode all
  fi

  if [[ "$result_kind" == external-judge ]]; then
    local -a validation_args=(
      --work-dir "$work_dir" --model-name "$model_name" --eval-id "$eval_id"
      --data-root "$EVAL_DATA_ROOT" --judge-model "$judge"
    )
    if [[ "$empty_response_policy" == allow ]]; then
      validation_args+=(--allow-empty-predictions)
    fi
    validation_args+=("${datasets[@]}")
    python "$COLT_SCRIPT_ROOT/external_judge/validate_results.py" "${validation_args[@]}"
  elif [[ "$group" == smoke ]]; then
    python "$COLT_SCRIPT_ROOT/tools/validate_smoke.py" "$work_dir" "$model_name" "$eval_id"
  else
    python "$COLT_SCRIPT_ROOT/tools/validate_eval_suite.py" \
      "$work_dir" "$model_name" "$eval_id" "$EVAL_DATA_ROOT" "${datasets[@]}" \
      --paper-profile "$paper_profile"
  fi
  echo "Evaluation completed and validated: target=$target group=$group"
}
