#!/usr/bin/env bash

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_ROOT/../.." && pwd)"
source "$SCRIPT_ROOT/common.sh"
require_workspace_layout
BASE_CONFIG="$REPO_ROOT/LLaMA-Factory/examples/train_full/colt_qwen3_sft_lkl_8gpu_paper_faithful.yaml"

mode="${1:-}"
case "$mode" in
  sequential)
    batch_aux=0
    ;;
  batched)
    batch_aux=1
    ;;
  *)
    echo "Usage: bash $0 {sequential|batched}" >&2
    exit 2
    ;;
esac

run_stamp="$(date +%Y%m%d_%H%M%S)"
benchmark_root="$REPO_ROOT/checkpoints/colt_aux_batch_benchmark"
output_dir="$benchmark_root/${mode}_$run_stamp"
config_dir="$REPO_ROOT/logs/colt_aux_batch_benchmark_${mode}_$run_stamp"
config_path="$config_dir/config.yaml"
mkdir -p "$output_dir" "$config_dir"

"$CONDA_ENV_DIR/bin/python" - "$BASE_CONFIG" "$config_path" "$output_dir" "$mode" <<'PY'
import sys

import yaml

source, destination, output_dir, mode = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config.update(
    output_dir=output_dir,
    max_steps=3,
    num_train_epochs=1.0,
    seed=42,
    data_seed=42,
    logging_steps=1,
    save_strategy="no",
    plot_loss=False,
    report_to="none",
    run_name="colt_aux_batch_benchmark_{}".format(mode),
    overwrite_output_dir=False,
)
with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

export COLT_PAPER_FAITHFUL=1
export COLT_BATCH_AUX_DECODERS="$batch_aux"
export COLT_COMPONENT_LOG_EVERY=1
export COLT_BENCHMARK_MODE=1
export COLT_SKIP_FINAL_SAVE=1
export COLT_TRAIN_CONFIG="$config_path"
export COLT_TRAIN_OUTPUT_DIR="$output_dir"
export COLT_TRAIN_ENTRY_SCRIPT="$SCRIPT_ROOT/19_benchmark_colt_aux_batching.sh"
export COLT_TRAIN_RECORD_PREFIX="colt_aux_batch_benchmark_${mode}_run"
export COLT_TRAIN_LOG_PREFIX="colt_aux_batch_benchmark_${mode}"

echo "Benchmark mode: $mode"
echo "COLT_BATCH_AUX_DECODERS=$COLT_BATCH_AUX_DECODERS"
echo "Output directory: $output_dir"
exec bash "$SCRIPT_ROOT/06_train.sh"
