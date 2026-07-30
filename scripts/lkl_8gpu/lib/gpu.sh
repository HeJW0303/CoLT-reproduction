#!/usr/bin/env bash

validate_gpu_profile() {
  local -a gpu_names=()
  require_command nvidia-smi
  local detected_name
  while IFS= read -r detected_name; do
    gpu_names+=("$detected_name")
  done < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//')
  [[ "${#gpu_names[@]}" -eq 8 ]] || die "Profile expects 8 GPUs; found ${#gpu_names[@]}."
  if [[ -n "${COLT_EXPECTED_GPU_NAME:-}" ]]; then
    local gpu_name
    for gpu_name in "${gpu_names[@]}"; do
      [[ "$gpu_name" == *"$COLT_EXPECTED_GPU_NAME"* ]] || die \
        "Profile $COLT_GPU_PROFILE expects $COLT_EXPECTED_GPU_NAME; found $gpu_name."
    done
  fi
}

parse_gpu_csv() {
  local gpu_csv="$1"
  IFS=',' read -r -a COLT_GPU_IDS <<< "$gpu_csv"
  (( ${#COLT_GPU_IDS[@]} > 0 )) || die "At least one GPU id is required."
  local seen="," gpu
  for gpu in "${COLT_GPU_IDS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU id: $gpu"
    [[ "$seen" != *",$gpu,"* ]] || die "Duplicate GPU id: $gpu"
    seen+="$gpu,"
  done
}

require_selected_gpus_free() {
  local threshold_mib="${COLT_GPU_MAX_USED_MIB:-500}" gpu memory_used
  [[ "$threshold_mib" =~ ^[1-9][0-9]*$ ]] || die "COLT_GPU_MAX_USED_MIB must be a positive integer."
  for gpu in "${COLT_GPU_IDS[@]}"; do
    memory_used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" \
      || die "Unable to query physical GPU $gpu."
    [[ "$memory_used" =~ ^[0-9]+$ ]] || die "Invalid memory usage for GPU $gpu: $memory_used"
    (( memory_used < threshold_mib )) || die \
      "Physical GPU $gpu is not free: ${memory_used} MiB used (limit ${threshold_mib} MiB)."
  done
}
