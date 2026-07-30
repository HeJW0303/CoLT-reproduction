#!/usr/bin/env bash

cmd_train() {
  local target="${1:-}"
  [[ -n "$target" ]] || die "Usage: colt.sh train {codefaithful|paper-faithful|oracle-k} [options]"
  shift
  local resume="${RESUME:-0}" cli_config="" cli_output="" batch_aux="${COLT_BATCH_AUX_DECODERS:-0}"
  while (( $# > 0 )); do
    case "$1" in
      --resume) resume=1; shift ;;
      --config) [[ $# -ge 2 ]] || die "--config requires a path"; cli_config="$2"; shift 2 ;;
      --output-dir) [[ $# -ge 2 ]] || die "--output-dir requires a path"; cli_output="$2"; shift 2 ;;
      --batch-aux) batch_aux=1; shift ;;
      *) die "Unknown train option: $1" ;;
    esac
  done
  [[ "$resume" == 0 || "$resume" == 1 ]] || die "RESUME must be 0 or 1"
  [[ "$batch_aux" == 0 || "$batch_aux" == 1 ]] || die "COLT_BATCH_AUX_DECODERS must be 0 or 1"
  [[ "$target" != codefaithful || "$batch_aux" == 0 ]] || die \
    "Auxiliary decoder batching requires paper-faithful or oracle-k training."

  runtime_init
  require_workspace_layout
  activate_colt_env
  validate_gpu_profile
  local gpu_csv="${COLT_TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
  parse_gpu_csv "$gpu_csv"
  [[ "${#COLT_GPU_IDS[@]}" -eq 8 ]] || die "Training requires exactly 8 GPU ids: $gpu_csv"
  require_selected_gpus_free

  local default_config default_output record_prefix log_prefix
  case "$target" in
    codefaithful)
      default_config="$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu.yaml"
      default_output="$OUTPUT_ROOT/colt_codefaithful"
      record_prefix=colt_codefaithful_run
      log_prefix=colt_codefaithful_train
      export COLT_PAPER_FAITHFUL=0 COLT_ORACLE_K_ENABLED=0 COLT_ORACLE_K_PREDICTOR_ENABLED=0
      export COLT_ORACLE_K_DYNAMIC_INFERENCE=0
      ;;
    paper-faithful)
      default_config="$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_paper_faithful.yaml"
      default_output="$OUTPUT_ROOT/colt_paper_faithful"
      record_prefix=colt_paper_faithful_run
      log_prefix=colt_paper_faithful_train
      export COLT_PAPER_FAITHFUL=1 COLT_ORACLE_K_ENABLED=0 COLT_ORACLE_K_PREDICTOR_ENABLED=0
      export COLT_ORACLE_K_DYNAMIC_INFERENCE=0
      ;;
    oracle-k)
      default_config="$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_oracle_k_predictor.yaml"
      default_output="$OUTPUT_ROOT/colt_oracle_k_predictor"
      record_prefix=colt_oracle_k_predictor_run
      log_prefix=colt_oracle_k_predictor_train
      export COLT_PAPER_FAITHFUL=1 COLT_ORACLE_K_ENABLED=1
      export COLT_ORACLE_K_MAX="${COLT_ORACLE_K_MAX:-8}"
      export COLT_ORACLE_K_BUDGET_CONDITIONING="${COLT_ORACLE_K_BUDGET_CONDITIONING:-1}"
      export COLT_ORACLE_K_PREDICTOR_ENABLED="${COLT_ORACLE_K_PREDICTOR_ENABLED:-1}"
      export COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT="${COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT:-0.2}"
      export COLT_ORACLE_K_DYNAMIC_INFERENCE="${COLT_ORACLE_K_DYNAMIC_INFERENCE:-1}"
      export COLT_ORACLE_K_DATASET_NAME="${COLT_ORACLE_K_DATASET_NAME:-onethinker_sft_image_oracle_k}"
      export COLT_ORACLE_K_DATA_FILE="${COLT_ORACLE_K_DATA_FILE:-$DATA_ROOT/colt_sft_image_oracle_k.json}"
      export COLT_ORACLE_K_TOKENIZED_PATH="${COLT_ORACLE_K_TOKENIZED_PATH:-$CACHE_ROOT/colt/onethinker_sft_oracle_k_predictor_tokenized}"
      export COLT_ORACLE_K_OUTPUT_DIR="${COLT_ORACLE_K_OUTPUT_DIR:-$default_output}"
      export COLT_ORACLE_K_RUN_NAME="${COLT_ORACLE_K_RUN_NAME:-colt_sft_8b_lkl_8gpu_oracle_k_predictor}"
      ;;
    *) die "Unknown training target: $target" ;;
  esac

  record_prefix="${COLT_TRAIN_RECORD_PREFIX:-$record_prefix}"
  log_prefix="${COLT_TRAIN_LOG_PREFIX:-$log_prefix}"

  local train_config="${cli_config:-${COLT_TRAIN_CONFIG:-$default_config}}"
  local output_dir="${cli_output:-${COLT_TRAIN_OUTPUT_DIR:-$default_output}}"
  [[ "$train_config" == /* && "$output_dir" == /* ]] || die "Training config and output paths must be absolute."
  [[ -f "$train_config" ]] || die "Missing training config: $train_config"
  export COLT_BATCH_AUX_DECODERS="$batch_aux"
  export COLT_COMPONENT_LOG_EVERY="${COLT_COMPONENT_LOG_EVERY:-8}"
  export CUDA_VISIBLE_DEVICES="$gpu_csv" FORCE_TORCHRUN=1 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" MASTER_PORT="${MASTER_PORT:-29500}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" DECORD_EOF_RETRY_MAX=2048001
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" COLT_DECODER_MODEL_PATH="$DECODER_MODEL_DIR"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE="${WANDB_MODE:-offline}"
  export WANDB_PROJECT="${WANDB_PROJECT:-CoLT-reproduction}"

  local config_output
  config_output="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$train_config")"
  [[ "$config_output" == "$output_dir" ]] || die \
    "Config output_dir ($config_output) does not match guarded output directory ($output_dir)."
  mkdir -p "$output_dir" "$LOG_ROOT"
  if find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    [[ "$resume" == 1 ]] || die \
      "Output directory is not empty: $output_dir. Use --resume only for a verified interrupted run."
    local latest_checkpoint
    latest_checkpoint="$(find "$output_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
    [[ -n "$latest_checkpoint" && -f "$latest_checkpoint/trainer_state.json" ]] || die \
      "--resume requested, but no complete Trainer checkpoint was found."
    echo "Resuming from: $latest_checkpoint"
  fi
  require_free_gib 200

  cd "$REPO_ROOT"
  [[ "$target" != paper-faithful && "$target" != oracle-k ]] || \
    python "$COLT_SCRIPT_ROOT/tools/verify_paper_faithful.py"
  [[ "$target" != oracle-k ]] || \
    python -m unittest discover -s "$REPO_ROOT/tests/oracle_k" -p 'test_*.py' -v

  local run_stamp="$(date +%Y%m%d_%H%M%S)"
  local run_record="$LOG_ROOT/${record_prefix}_${run_stamp}"
  local log_file="$LOG_ROOT/${log_prefix}_${run_stamp}.log"
  mkdir -p "$run_record"
  cp "$train_config" "$run_record/"
  cp "$REPO_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_a100.json" "$run_record/"
  cp -R "$COLT_SCRIPT_ROOT" "$run_record/lkl_8gpu"
  cp "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py" "$run_record/"
  [[ ! -f "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_oracle_k.py" ]] || \
    cp "$REPO_ROOT/transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_oracle_k.py" "$run_record/"
  git -c safe.directory="$REPO_ROOT" rev-parse HEAD > "$run_record/git_head.txt"
  git -c safe.directory="$REPO_ROOT" status --short > "$run_record/git_status.txt"
  git -c safe.directory="$REPO_ROOT" diff --binary > "$run_record/git_diff.patch"
  python -m pip freeze > "$run_record/pip_freeze.txt"
  {
    printf 'target=%s\n' "$target"
    printf 'config=%s\n' "$train_config"
    printf 'output_dir=%s\n' "$output_dir"
    printf 'resume=%s\n' "$resume"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'COLT_PAPER_FAITHFUL=%s\n' "$COLT_PAPER_FAITHFUL"
    printf 'COLT_BATCH_AUX_DECODERS=%s\n' "$COLT_BATCH_AUX_DECODERS"
    printf 'COLT_ORACLE_K_ENABLED=%s\n' "$COLT_ORACLE_K_ENABLED"
    printf 'COLT_ORACLE_K_PREDICTOR_ENABLED=%s\n' "$COLT_ORACLE_K_PREDICTOR_ENABLED"
    printf 'COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT=%s\n' "${COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT:-0.2}"
  } > "$run_record/environment.txt"
  echo "Training target: $target"
  echo "Training log: $log_file"
  llamafactory-cli train "$train_config" 2>&1 | tee "$log_file"
}
