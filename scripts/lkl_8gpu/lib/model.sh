#!/usr/bin/env bash

resolve_eval_model() {
  local target="$1" cli_path="${2:-}" default_path
  case "$target" in
    codefaithful) default_path="$OUTPUT_ROOT/colt_codefaithful" ;;
    paper-faithful) default_path="$OUTPUT_ROOT/colt_paper_faithful" ;;
    oracle-k) default_path="$OUTPUT_ROOT/colt_oracle_k_predictor" ;;
    official) default_path="${COLT_OFFICIAL_MODEL_DIR:-$OFFICIAL_MODEL_DIR_DEFAULT}" ;;
    baseline) default_path="$BASE_MODEL_DIR" ;;
    *) die "Unknown evaluation target: $target" ;;
  esac

  # Deliberate precedence: CLI > environment > target default.
  if [[ -n "$cli_path" ]]; then
    EVAL_MODEL_PATH="$cli_path"
    EVAL_MODEL_PATH_SOURCE="--model-path"
  elif [[ -n "${COLT_EVAL_MODEL_PATH:-}" ]]; then
    EVAL_MODEL_PATH="$COLT_EVAL_MODEL_PATH"
    EVAL_MODEL_PATH_SOURCE="COLT_EVAL_MODEL_PATH"
  else
    EVAL_MODEL_PATH="$default_path"
    EVAL_MODEL_PATH_SOURCE="target:$target"
  fi
  [[ "$EVAL_MODEL_PATH" == /* ]] || die "Evaluation model path must be absolute: $EVAL_MODEL_PATH"
  [[ -d "$EVAL_MODEL_PATH" ]] || die "Evaluation model directory does not exist: $EVAL_MODEL_PATH"
  EVAL_MODEL_PATH="$(cd "$EVAL_MODEL_PATH" && pwd -P)"
}

generation_log_label() {
  case "$1" in
    official) printf '%s\n' "sampling_max256" ;;
    respect-args) printf '%s\n' "greedy_max8192" ;;
    *) die "Unknown generation mode for log naming: $1" ;;
  esac
}

verify_eval_model() {
  local target="$1" mode=trained
  case "$target" in
    official) mode=official ;;
    baseline) mode=base ;;
  esac
  local -a args=(
    --mode "$mode"
    --model-dir "$EVAL_MODEL_PATH"
    --expected-step "${COLT_EXPECTED_GLOBAL_STEP:-1910}"
  )
  if [[ "$mode" == base ]]; then
    args+=(--expected-revision "$BASE_MODEL_REVISION")
  fi
  python "$COLT_SCRIPT_ROOT/tools/verify_model.py" "${args[@]}"
}

create_generation_overlay() {
  local source_model="$1" run_id="$2"
  local overlay="$EVAL_ROOT/runtime_models/${run_id}_$$_respect_args"
  mkdir -p "$overlay"
  local model_file source_file
  for model_file in "$source_model"/* "$source_model"/.[!.]*; do
    [[ -e "$model_file" ]] || continue
    ln -s "$model_file" "$overlay/$(basename "$model_file")"
  done
  for source_file in \
    "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py" \
    "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_oracle_k.py" \
    "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/oracle_k.py"; do
    [[ -f "$source_file" ]] && ln -sfn "$source_file" "$overlay/$(basename "$source_file")"
  done
  EVAL_RUNTIME_MODEL_PATH="$overlay"
}
