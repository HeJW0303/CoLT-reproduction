#!/usr/bin/env bash
# 10-step end-to-end gate for the two-stage question-latent curriculum.
# Stage 1 is the documented clean-question/final-latent bottleneck. Stage 2
# restores normal 88K CoLT and adds the already-audited Teacher-A LaRe losses.

set -Eeuo pipefail

readonly REPO=/data/nvme0/lkl/CoLT-reproduction
readonly ENV_FILE="$REPO/colt-local.env"
readonly CONDA_PROFILE=/data/nvme0/lkl/miniconda3/etc/profile.d/conda.sh
readonly STAGE1_CONFIG="$REPO/LLaMA-Factory/examples/train_full/colt_qwen3_two_stage_question_latent_stage1_smoke10.yaml"
readonly STAGE2_CONFIG="$REPO/LLaMA-Factory/examples/train_full/colt_qwen3_two_stage_question_latent_stage2_lare_smoke10.yaml"
readonly STAGE1_OUTPUT="$REPO/checkpoints/two_stage_question_latent_stage1_smoke10"
readonly STAGE2_OUTPUT="$REPO/checkpoints/two_stage_question_latent_stage2_lare_smoke10"
readonly STAGE1_DATA=/data/nvme0/lkl/datasets/CoLT_Train_Dataset/colt_sft_image_grounding_bbox_normalized_strict.json
readonly STAGE2_DATA=/data/nvme0/lkl/datasets/CoLT_Train_Dataset/colt_sft_image_nogrounding.json
readonly TARGETS=/data/nvme0/lkl/cache/colt/qwen_base_colt88k_lare_teacherA_full_targets
readonly GPU_CSV=0,1,2,3,4,5,6,7
readonly EXPECTED_STAGE1_ROWS=33328
readonly EXPECTED_STAGE2_ROWS=88833
readonly EXPECTED_STEPS=10

MODE=run
die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
usage() { echo "Usage: $0 [--dry-run|--resume-stage2]"; }
while (( $# > 0 )); do
  case "$1" in
    --dry-run) MODE=dry-run ;;
    --resume-stage2) MODE=resume-stage2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

[[ -f "$ENV_FILE" && -f "$CONDA_PROFILE" ]] || die "Missing local runtime files."
[[ -f "$STAGE1_CONFIG" && -f "$STAGE2_CONFIG" ]] || die "Missing smoke YAML."
[[ -f "$STAGE1_DATA" && -f "$STAGE2_DATA" ]] || die "Missing Stage-1 or Stage-2 raw data."
source "$ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" WANDB_MODE=offline

python - "$STAGE1_CONFIG" "$STAGE2_CONFIG" "$STAGE1_DATA" "$STAGE2_DATA" "$EXPECTED_STAGE1_ROWS" "$EXPECTED_STAGE2_ROWS" "$STAGE1_OUTPUT" "$STAGE2_OUTPUT" <<'PY'
import json, math, sys
from pathlib import Path
import yaml

(stage1_cfg, stage2_cfg, stage1_data, stage2_data, stage1_rows, stage2_rows, stage1_output, stage2_output) = sys.argv[1:]
stage1_rows, stage2_rows = int(stage1_rows), int(stage2_rows)
c1, c2 = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) for path in (stage1_cfg, stage2_cfg))
required1 = {
    "dataset": "onethinker_sft_image_grounding_bbox_normalized_strict",
    "model_name_or_path": "/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct",
    "output_dir": stage1_output, "max_steps": 10, "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8, "learning_rate": 1e-5, "num_train_epochs": 1.0,
}
required2 = {
    "dataset": "onethinker_sft_image_nogrounding", "model_name_or_path": stage1_output,
    "output_dir": stage2_output, "max_steps": 10, "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8, "learning_rate": 1e-5, "num_train_epochs": 1.0,
}
for name, cfg, required in (("stage1", c1, required1), ("stage2", c2, required2)):
    for key, value in required.items():
        if cfg.get(key) != value:
            raise SystemExit(f"{name} config mismatch: {key}={cfg.get(key)!r}, expected {value!r}")
for path, expected, stage in ((stage1_data, stage1_rows, "stage1"), (stage2_data, stage2_rows, "stage2")):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(rows) != expected:
        raise SystemExit(f"{stage} raw rows={len(rows)}, expected={expected}")
    if stage == "stage1" and not all(
        row.get("causal_grounded") is True and len(row.get("images", [])) == 1 and len(row.get("bboxes", [])) == 1
        and len(row["bboxes"][0]) == 4 and 0 <= row["bboxes"][0][0] < row["bboxes"][0][2] <= 1
        and 0 <= row["bboxes"][0][1] < row["bboxes"][0][3] <= 1
        for row in rows
    ):
        raise SystemExit("stage1 strict corpus does not satisfy causal-grounded one-image/one-normalized-box contract")
if math.ceil(stage1_rows / 64) != 521 or math.ceil(stage2_rows / 64) != 1389:
    raise SystemExit("unexpected epoch horizons for the two-stage data contracts")
print("Two-stage smoke contract OK: stage1=33328 rows/521 natural steps; stage2=88833 rows/1389 natural steps.")
PY

if ! nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | \
  awk -F',' '{gsub(/ MiB| /,"",$2); if (($2 + 0) >= 500) bad=1} END {exit bad ? 1 : 0}'; then
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >&2
  die "The 8-GPU smoke requires every selected GPU below 500 MiB."
fi

if [[ "$MODE" == dry-run ]]; then
  log "Dry run passed. It deliberately does not build Teacher-A targets or launch training."
  exit 0
fi
if [[ "$MODE" == run ]]; then
  [[ ! -e "$STAGE1_OUTPUT" ]] || die "Refusing to overwrite Stage-1 smoke output: $STAGE1_OUTPUT"
else
  [[ -f "$STAGE1_OUTPUT/trainer_state.json" && -f "$STAGE1_OUTPUT/config.json" ]] || \
    die "--resume-stage2 requires the completed Stage-1 smoke root."
fi
[[ ! -e "$STAGE2_OUTPUT" ]] || die "Refusing to overwrite Stage-2 smoke output: $STAGE2_OUTPUT"

if [[ "$MODE" == run ]]; then
  log "Smoke stage 1/2: documented clean-question bottleneck (10 optimizer steps)."
  export COLT_TRAIN_GPUS="$GPU_CSV" COLT_PAPER_FAITHFUL=1 COLT_CHECK_GPU_FREE=1 COLT_STRICT_PREFLIGHT=1
  export COLT_STOCHASTIC_LATENT=0 COLT_ANSWER_VISIBILITY=full COLT_IMAGE_MASK_PROB=0
  export COLT_VISUAL_GROUNDING=0 COLT_KL_ANCHOR=0 COLT_BOOTSTRAP_ENABLED=0
  export COLT_BATCH_AUX_DECODERS=1 COLT_AUX_MAX_BATCH_TOKENS=4096 COLT_COMPONENT_LOG_EVERY=1
  export COLT_CAUSAL_GROUNDED_LATENT=1 COLT_CAUSAL_GROUNDED_ROI_WEIGHT=0.0
  export COLT_CAUSAL_GROUNDED_GEOMETRY_WEIGHT=0.0 COLT_CAUSAL_GROUNDED_INFERENCE_MEDIATED=0
  export COLT_MASKED_LATENT_TRAINING=0
  export COLT_LARE_REFOCUS=0 COLT_LARE_RECON_WEIGHT=0.0 COLT_COT_ATTN_ALIGN=0
  unset COLT_COT_ATTN_TARGETS_PATH
  export COLT_TRAIN_RUN_LABEL=two_stage_question_latent_stage1_smoke10
  bash scripts/lkl_8gpu/colt.sh train paper-faithful --config "$STAGE1_CONFIG" --output-dir "$STAGE1_OUTPUT" --batch-aux
fi

python - "$STAGE1_OUTPUT" "$EXPECTED_STEPS" <<'PY'
import json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
if state.get("global_step") != expected or state.get("max_steps") != expected:
    raise SystemExit(f"Stage-1 smoke incomplete: {state.get('global_step')}/{state.get('max_steps')}")
if config.get("colt_causal_grounded_latent") is not True:
    raise SystemExit("Stage-1 checkpoint did not persist the causal-grounded curriculum flag")
if config.get("colt_causal_grounded_roi_weight") != 0.0 or config.get("colt_causal_grounded_geometry_weight") != 0.0:
    raise SystemExit("Stage-1 checkpoint has non-zero ROI/geometry settings")
print("Stage-1 smoke checkpoint contract OK.")
PY

# Also set the ordinary paper-faithful launcher contract for --resume-stage2,
# which starts in a fresh shell and therefore has no Stage-1 exports.
export COLT_TRAIN_GPUS="$GPU_CSV" COLT_PAPER_FAITHFUL=1 COLT_CHECK_GPU_FREE=1 COLT_STRICT_PREFLIGHT=1
export COLT_STOCHASTIC_LATENT=0 COLT_ANSWER_VISIBILITY=full COLT_IMAGE_MASK_PROB=0
export COLT_VISUAL_GROUNDING=0 COLT_KL_ANCHOR=0 COLT_BOOTSTRAP_ENABLED=0
export COLT_BATCH_AUX_DECODERS=1 COLT_AUX_MAX_BATCH_TOKENS=4096 COLT_COMPONENT_LOG_EVERY=1
log "Building/validating the cache-aligned Teacher-A 88K sidecar before Stage 2."
bash scripts/lkl_8gpu/two_stage_question_latent/build_teacherA_88k_targets.sh

log "Smoke stage 2/2: normal 88K CoLT + LaRe reconstruction=0.1, attention=0.05 (10 optimizer steps)."
export COLT_CAUSAL_GROUNDED_LATENT=0 COLT_CAUSAL_GROUNDED_ROI_WEIGHT=0.0
export COLT_CAUSAL_GROUNDED_GEOMETRY_WEIGHT=0.0 COLT_CAUSAL_GROUNDED_INFERENCE_MEDIATED=0
export COLT_LARE_REFOCUS=1 COLT_LARE_DIM=1536 COLT_LARE_LAYERS=2 COLT_LARE_HEADS=12
export COLT_LARE_QUERIES=4 COLT_LARE_MAX_STEPS=16 COLT_LARE_DROPOUT=0.0
export COLT_LARE_VISUAL_DROPOUT=0.1 COLT_LARE_ATTN_TOPK=0 COLT_LARE_GATE_BIAS=-2.0
export COLT_LARE_RECON_WEIGHT=0.1 COLT_LARE_RECON_STEPS=1000 COLT_LARE_DETACH_VISUAL=1
export COLT_LARE_RECORD_ATTENTION=0 COLT_COT_ATTN_ALIGN=1 COLT_COT_ATTN_ALIGN_WEIGHT=0.05
export COLT_COT_ATTN_MIN_CONFIDENCE=0.05 COLT_COT_ATTN_TARGETS_PATH="$TARGETS"
export COLT_TRAIN_RUN_LABEL=two_stage_question_latent_stage2_lare_smoke10
bash scripts/lkl_8gpu/colt.sh train paper-faithful --config "$STAGE2_CONFIG" --output-dir "$STAGE2_OUTPUT" --batch-aux

python - "$STAGE2_OUTPUT" "$EXPECTED_STEPS" <<'PY'
import json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
if state.get("global_step") != expected or state.get("max_steps") != expected:
    raise SystemExit(f"Stage-2 smoke incomplete: {state.get('global_step')}/{state.get('max_steps')}")
if config.get("colt_lare_refocus") is not True:
    raise SystemExit("Stage-2 checkpoint did not persist LaRe")
if config.get("colt_lare_reconstruction_weight") != 0.1 or config.get("colt_cot_attention_align_weight") != 0.05:
    raise SystemExit("Stage-2 checkpoint did not persist requested LaRe/attention weights")
print("Two-stage 10-step smoke passed.")
PY
