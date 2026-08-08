#!/usr/bin/env bash

cmd_rl() {
  local action="${1:-audit}"
  if [[ $# -gt 0 ]]; then
    shift
  fi

  case "$action" in
    prepare-data)
      local source_file="${COLT_RL_SOURCE_FILE:-$LKL_ROOT/datasets/onethinker_rl_train.json}"
      local image_dir="${COLT_RL_IMAGE_DIR:-$LKL_ROOT/datasets/CoLT_Train_Dataset}"
      local output_file="${COLT_RL_TRAIN_FILE:-$LKL_ROOT/datasets/onethinker_rl_train_image.json}"
      local python_bin="${COLT_RL_PYTHON:-}"
      local overwrite=0

      while [[ $# -gt 0 ]]; do
        case "$1" in
          --source-file|--image-dir|--output-file|--python)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            case "$1" in
              --source-file) source_file="$2" ;;
              --image-dir) image_dir="$2" ;;
              --output-file) output_file="$2" ;;
              --python) python_bin="$2" ;;
            esac
            shift 2
            ;;
          --overwrite)
            overwrite=1
            shift
            ;;
          *) die "Unknown rl prepare-data option: $1" ;;
        esac
      done

      if [[ -z "$python_bin" ]]; then
        python_bin="$(command -v python3 || true)"
      fi
      [[ -n "$python_bin" && -x "$python_bin" ]] || die "Manifest generation requires an executable Python 3."
      [[ -f "$source_file" ]] || die "Missing OneThinker source dataset: $source_file"
      [[ -d "$image_dir" ]] || die "RL image directory does not exist: $image_dir"

      local -a command=(
        "$python_bin"
        "$COLT_SCRIPT_ROOT/easyr1/build_onethinker_image_rl_manifest.py"
        --source-file "$source_file"
        --media-root "$image_dir"
        --output-file "$output_file"
      )
      (( overwrite == 0 )) || command+=(--overwrite)
      "${command[@]}"
      ;;
    audit)
      local easyr1_root="${COLT_EASYR1_ROOT:-$REPO_ROOT/EasyR1}"
      local model_path="${COLT_RL_MODEL_PATH:-$OUTPUT_ROOT/colt_paper_faithful_v2}"
      local train_file="${COLT_RL_TRAIN_FILE:-$LKL_ROOT/datasets/onethinker_rl_train_image.json}"
      local python_bin="${COLT_RL_PYTHON:-}"
      local allow_incomplete=0
      local json_output=0
      local check_runtime=0

      while [[ $# -gt 0 ]]; do
        case "$1" in
          --easyr1-root)
            [[ $# -ge 2 ]] || die "--easyr1-root requires a path"
            easyr1_root="$2"
            shift 2
            ;;
          --model-path)
            [[ $# -ge 2 ]] || die "--model-path requires a path"
            model_path="$2"
            shift 2
            ;;
          --train-file)
            [[ $# -ge 2 ]] || die "--train-file requires a path"
            train_file="$2"
            shift 2
            ;;
          --python)
            [[ $# -ge 2 ]] || die "--python requires an executable path"
            python_bin="$2"
            shift 2
            ;;
          --allow-incomplete)
            allow_incomplete=1
            shift
            ;;
          --json)
            json_output=1
            shift
            ;;
          --check-runtime)
            check_runtime=1
            shift
            ;;
          *)
            die "Unknown rl audit option: $1"
            ;;
        esac
      done

      if [[ -z "$python_bin" ]]; then
        python_bin="$(command -v python3 || true)"
      fi
      [[ -n "$python_bin" && -x "$python_bin" ]] || die \
        "Python 3 is required. Activate the CoLT environment or pass --python PATH."

      local -a command=(
        "$python_bin"
        "$COLT_SCRIPT_ROOT/easyr1/preflight.py"
        --repo-root "$REPO_ROOT"
        --easyr1-root "$easyr1_root"
        --model-path "$model_path"
        --train-file "$train_file"
      )
      (( allow_incomplete == 0 )) || command+=(--allow-incomplete)
      (( json_output == 0 )) || command+=(--json)
      (( check_runtime == 0 )) || command+=(--check-runtime)
      "${command[@]}"
      ;;
    train)
      local easyr1_root="${COLT_EASYR1_ROOT:-$REPO_ROOT/EasyR1}"
      local model_path="${COLT_RL_MODEL_PATH:-$OUTPUT_ROOT/colt_paper_faithful_v2}"
      local train_file="${COLT_RL_TRAIN_FILE:-$LKL_ROOT/datasets/onethinker_rl_train_image.json}"
      local image_dir="${COLT_RL_IMAGE_DIR:-$LKL_ROOT/datasets/CoLT_Train_Dataset}"
      local output_dir="${COLT_RL_OUTPUT_DIR:-$OUTPUT_ROOT/colt_easyr1/fixed_v2_outcome_grpo}"
      local config_path="${COLT_RL_CONFIG:-$REPO_ROOT/EasyR1/examples/colt_fixed_v2_outcome_grpo.yaml}"
      local python_bin="${COLT_RL_PYTHON:-}"
      local gpus="${COLT_RL_GPUS:-0,1,2,3,4,5,6,7}"
      local max_steps="${COLT_RL_MAX_STEPS:-}"
      local dry_run=0

      while [[ $# -gt 0 ]]; do
        case "$1" in
          --easyr1-root|--model-path|--train-file|--image-dir|--output-dir|--config|--python|--gpus|--max-steps)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            case "$1" in
              --easyr1-root) easyr1_root="$2" ;;
              --model-path) model_path="$2" ;;
              --train-file) train_file="$2" ;;
              --image-dir) image_dir="$2" ;;
              --output-dir) output_dir="$2" ;;
              --config) config_path="$2" ;;
              --python) python_bin="$2" ;;
              --gpus) gpus="$2" ;;
              --max-steps) max_steps="$2" ;;
            esac
            shift 2
            ;;
          --dry-run)
            dry_run=1
            shift
            ;;
          *) die "Unknown rl train option: $1" ;;
        esac
      done

      if [[ -z "$python_bin" ]]; then
        python_bin="$(command -v python || command -v python3 || true)"
      fi
      [[ -n "$python_bin" && -x "$python_bin" ]] || die "RL training requires an executable Python 3."
      [[ -f "$config_path" ]] || die "Missing CoLT EasyR1 config: $config_path"
      [[ -f "$easyr1_root/verl/trainer/main.py" ]] || die "Invalid vendored EasyR1 root: $easyr1_root"
      [[ -f "$model_path/config.json" ]] || die "Invalid CoLT checkpoint: $model_path"
      [[ -f "$train_file" ]] || die "Missing OneThinker RL dataset: $train_file"
      [[ -z "$image_dir" || -d "$image_dir" ]] || die "RL image directory does not exist: $image_dir"
      [[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpus must be a comma-separated list of GPU indices"
      [[ -z "$max_steps" || "$max_steps" =~ ^[1-9][0-9]*$ ]] || die "--max-steps must be a positive integer"

      local gpu_count
      gpu_count="$(awk -F, '{print NF}' <<< "$gpus")"
      local reward_path="$easyr1_root/verl/reward_function/colt_outcome.py"
      local transformers_src="$REPO_ROOT/transformers-4.57.0/src"
      local python_path="$transformers_src:$easyr1_root:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
      PYTHONPATH="$python_path" "$python_bin" "$COLT_SCRIPT_ROOT/easyr1/preflight.py" \
        --repo-root "$REPO_ROOT" \
        --easyr1-root "$easyr1_root" \
        --model-path "$model_path" \
        --train-file "$train_file" \
        --check-runtime

      local -a command=(
        "$python_bin" -m verl.trainer.main
        "config=$config_path"
        "data.train_files=$train_file"
        "data.val_files=$train_file"
        "worker.actor.model.model_path=$model_path"
        "worker.reward.reward_function=$reward_path:compute_score"
        "trainer.n_gpus_per_node=$gpu_count"
        "trainer.save_checkpoint_path=$output_dir"
      )
      [[ -z "$image_dir" ]] || command+=("data.image_dir=$image_dir")
      [[ -z "$max_steps" ]] || command+=("trainer.max_steps=$max_steps")

      if (( dry_run == 1 )); then
        printf 'CUDA_VISIBLE_DEVICES=%q\n' "$gpus"
        printf 'PYTHONPATH=%q\n' "$python_path"
        printf 'COLT_RL_MODE=1\nCOLT_RL_TOKENIZER_PATH=%q\nCOLT_RESPECT_GENERATION_ARGS=1\n' "$model_path"
        printf 'Command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        return
      fi

      PYTHONPATH="$python_path" \
      CUDA_VISIBLE_DEVICES="$gpus" \
      COLT_RL_MODE=1 \
      COLT_RL_TOKENIZER_PATH="$model_path" \
      COLT_RESPECT_GENERATION_ARGS=1 \
        "${command[@]}"
      ;;
    *)
      die "Unknown rl action: $action (expected prepare-data, audit, or train)"
      ;;
  esac
}
