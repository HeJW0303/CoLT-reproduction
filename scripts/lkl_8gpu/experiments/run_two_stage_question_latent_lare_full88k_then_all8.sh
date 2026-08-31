#!/usr/bin/env bash
# Full two-stage question-latent curriculum.
# Stage 1: 33,328 strict grounding rows, full image -> H1..H3, while final
#          answer CE sees clean question KV + H3 only.
# Stage 2: fresh optimizer on 88,833 normal-CoLT rows, initialized from Stage
#          1, with frozen Teacher-A LaRe reconstruction=0.1 and attention=0.05.
# Then:   all8 evaluation with greedy generation (respect-args) and 8192 tokens.

set -Eeuo pipefail

readonly REPO=/data/nvme0/lkl/CoLT-reproduction
readonly ENV_FILE="$REPO/colt-local.env"
readonly CONDA_PROFILE=/data/nvme0/lkl/miniconda3/etc/profile.d/conda.sh
readonly STAGE1_CONFIG="$REPO/LLaMA-Factory/examples/train_full/colt_qwen3_two_stage_question_latent_stage1_full33328.yaml"
readonly STAGE2_CONFIG="$REPO/LLaMA-Factory/examples/train_full/colt_qwen3_two_stage_question_latent_stage2_lare_full88833.yaml"
readonly STAGE1_OUTPUT="$REPO/checkpoints/two_stage_question_latent_stage1_full33328"
readonly STAGE2_OUTPUT="$REPO/checkpoints/two_stage_question_latent_stage2_lare_full88833"
readonly PIPELINE_CONTRACT="$REPO/checkpoints/two_stage_question_latent_lare_full88k.pipeline.contract"
readonly STAGE1_DATA=/data/nvme0/lkl/datasets/CoLT_Train_Dataset/colt_sft_image_grounding_bbox_normalized_strict.json
readonly STAGE2_DATA=/data/nvme0/lkl/datasets/CoLT_Train_Dataset/colt_sft_image_nogrounding.json
readonly STAGE2_TOKENIZED=/data/nvme0/lkl/cache/colt/qwen_base_colt88k_lare_full_tokenized
readonly TEACHER_TARGETS=/data/nvme0/lkl/cache/colt/qwen_base_colt88k_lare_teacherA_full_targets
readonly BASE_MODEL=/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct
readonly GPU_CSV=0,1,2,3,4,5,6,7
readonly EXPECTED_STAGE1_ROWS=33328
readonly EXPECTED_STAGE2_ROWS=88833
readonly EXPECTED_STAGE1_STEPS=521
readonly EXPECTED_STAGE2_STEPS=1389
readonly TEACHER_HEADS='[[23,4],[21,10],[26,20],[19,27],[23,5],[29,11],[25,10],[24,13]]'

MODE=run
die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
usage() { echo "Usage: $0 [--dry-run]"; }
while (( $# > 0 )); do
  case "$1" in
    --dry-run) MODE=dry-run ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

[[ -f "$ENV_FILE" && -f "$CONDA_PROFILE" ]] || die "Missing local runtime files."
[[ -f "$STAGE1_CONFIG" && -f "$STAGE2_CONFIG" ]] || die "Missing full-run YAML."
[[ -f "$STAGE1_DATA" && -f "$STAGE2_DATA" ]] || die "Missing Stage-1 or Stage-2 raw data."
[[ -d "$STAGE2_TOKENIZED" && -d "$TEACHER_TARGETS" ]] || die "Missing Stage-2 cache or Teacher-A sidecar."
[[ -d "$BASE_MODEL" ]] || die "Missing base model: $BASE_MODEL"
source "$ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" WANDB_MODE=offline

python - "$STAGE1_CONFIG" "$STAGE2_CONFIG" "$STAGE1_DATA" "$STAGE2_DATA" "$STAGE2_TOKENIZED" "$TEACHER_TARGETS" "$STAGE1_OUTPUT" "$STAGE2_OUTPUT" "$EXPECTED_STAGE1_ROWS" "$EXPECTED_STAGE2_ROWS" "$EXPECTED_STAGE1_STEPS" "$EXPECTED_STAGE2_STEPS" "$TEACHER_HEADS" <<'PY'
import json, math, sys
from pathlib import Path

import yaml
from datasets import DatasetDict, load_from_disk

(
    stage1_cfg, stage2_cfg, stage1_data, stage2_data, tokenized_path, targets_path,
    stage1_output, stage2_output, stage1_rows, stage2_rows, stage1_steps, stage2_steps, heads_json,
) = sys.argv[1:]
stage1_rows, stage2_rows, stage1_steps, stage2_steps = map(int, (stage1_rows, stage2_rows, stage1_steps, stage2_steps))
heads = json.loads(heads_json)
c1, c2 = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) for path in (stage1_cfg, stage2_cfg))
required1 = {
    "model_name_or_path": "/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct",
    "dataset": "onethinker_sft_image_grounding_bbox_normalized_strict",
    "output_dir": stage1_output, "max_steps": stage1_steps, "num_train_epochs": 1.0,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "learning_rate": 1e-5,
    "weight_decay": 1e-6, "finetuning_type": "full",
}
required2 = {
    "model_name_or_path": stage1_output, "dataset": "onethinker_sft_image_nogrounding",
    "tokenized_path": tokenized_path, "output_dir": stage2_output,
    "max_steps": stage2_steps, "num_train_epochs": 1.0, "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8, "learning_rate": 1e-5, "weight_decay": 1e-6,
    "finetuning_type": "full",
}
for name, config, required in (("stage1", c1, required1), ("stage2", c2, required2)):
    for key, value in required.items():
        if config.get(key) != value:
            raise SystemExit(f"{name} config mismatch: {key}={config.get(key)!r}, expected {value!r}")
for path, expected, stage in ((stage1_data, stage1_rows, "stage1"), (stage2_data, stage2_rows, "stage2")):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(rows) != expected:
        raise SystemExit(f"{stage} raw rows={len(rows)}, expected={expected}")
    if stage == "stage1" and not all(
        row.get("causal_grounded") is True and len(row.get("images", [])) == 1
        and len(row.get("bboxes", [])) == 1 and len(row["bboxes"][0]) == 4
        and 0 <= row["bboxes"][0][0] < row["bboxes"][0][2] <= 1
        and 0 <= row["bboxes"][0][1] < row["bboxes"][0][3] <= 1
        for row in rows
    ):
        raise SystemExit("Stage-1 strict data no longer satisfies one-image/one-normalized-box contract")
cache = load_from_disk(tokenized_path)
cache = cache["train"] if isinstance(cache, DatasetDict) else cache
targets = load_from_disk(targets_path)
targets = targets["train"] if isinstance(targets, DatasetDict) else targets
metadata = json.loads((Path(targets_path) / "metadata.json").read_text(encoding="utf-8"))
if len(cache) != stage2_rows or len(targets) != stage2_rows:
    raise SystemExit(f"Stage-2 cache/sidecar rows mismatch: cache={len(cache)}, targets={len(targets)}")
if metadata.get("source_train_fingerprint") != cache._fingerprint:
    raise SystemExit("Stage-2 Teacher-A sidecar fingerprint does not match canonical 88K tokenized cache")
if (metadata.get("teacher_attention_mode"), metadata.get("teacher_layer"), metadata.get("teacher_layer_fallback")) != (
    "explicit_sparse_layer_head", None, 18
):
    raise SystemExit("Teacher-A metadata must be explicit sparse layer/head, not all heads at layer 18")
if metadata.get("teacher_head_pairs") != heads or metadata.get("teacher_heads") != heads:
    raise SystemExit("Teacher-A sparse head contract mismatch")
if metadata.get("query_pool") != "visual-mass":
    raise SystemExit("Teacher-A query pool mismatch")
coverage = metadata.get("target_coverage") or {}
if coverage.get("rows_with_any_map", 0) <= 0:
    raise SystemExit("Teacher-A sidecar has no usable attention maps")
if math.ceil(stage1_rows / 64) != stage1_steps or math.ceil(stage2_rows / 64) != stage2_steps:
    raise SystemExit("Configured full-step horizons no longer equal one epoch at global batch 64")
print(
    f"Full pipeline preflight OK: Stage1={stage1_rows} rows/{stage1_steps} steps; "
    f"Stage2={stage2_rows} rows/{stage2_steps} steps; cache_fp={cache._fingerprint}; "
    f"teacher_maps={coverage.get('rows_with_any_map')}/{stage2_rows}."
)
PY

if ! nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | \
  awk -F',' '{gsub(/ MiB| /,"",$2); if (($2 + 0) >= 500) bad=1} END {exit bad ? 1 : 0}'; then
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >&2
  die "Full training requires all 8 selected GPUs below 500 MiB before launch."
fi

if [[ "$MODE" == dry-run ]]; then
  log "Dry run passed. No output directory, training, or evaluation was started."
  exit 0
fi
[[ ! -e "$STAGE1_OUTPUT" ]] || die "Refusing to overwrite Stage-1 full output: $STAGE1_OUTPUT"
[[ ! -e "$STAGE2_OUTPUT" ]] || die "Refusing to overwrite Stage-2 full output: $STAGE2_OUTPUT"
[[ ! -e "$PIPELINE_CONTRACT" ]] || die "Refusing to overwrite existing pipeline contract: $PIPELINE_CONTRACT"

printf 'pipeline=two-stage-question-latent-lare-full88k\nstage1_data_rows=%s\nstage1_optimizer_steps=%s\nstage1_objective=clean_question_KV_plus_final_H3_answer_CE__plus_0.2_forward__0.2_backward__0.2_prediction\nstage1_roi_weight=0.0\nstage1_geometry_weight=0.0\nstage2_data_rows=%s\nstage2_optimizer_steps=%s\nstage2_objective=normal_CoLT__plus_LaRe_reconstruction_0.1__plus_TeacherA_attention_alignment_0.05\nstage2_cache=%s\nstage2_teacher_targets=%s\nstage2_teacher_attention_mode=explicit_sparse_layer_head\nstage2_teacher_heads=%s\nstage2_query_pool=visual-mass\nevaluation=all8_greedy_respect_args_max_new_tokens_8192\n' \
  "$EXPECTED_STAGE1_ROWS" "$EXPECTED_STAGE1_STEPS" "$EXPECTED_STAGE2_ROWS" "$EXPECTED_STAGE2_STEPS" \
  "$STAGE2_TOKENIZED" "$TEACHER_TARGETS" "$TEACHER_HEADS" > "$PIPELINE_CONTRACT"

export COLT_TRAIN_GPUS="$GPU_CSV" COLT_EVAL_GPUS="$GPU_CSV"
export COLT_PAPER_FAITHFUL=1 COLT_CHECK_GPU_FREE=1 COLT_STRICT_PREFLIGHT=1
export COLT_STOCHASTIC_LATENT=0 COLT_ANSWER_VISIBILITY=full COLT_IMAGE_MASK_PROB=0
export COLT_VISUAL_GROUNDING=0 COLT_KL_ANCHOR=0 COLT_BOOTSTRAP_ENABLED=0
export COLT_BATCH_AUX_DECODERS=1 COLT_AUX_MAX_BATCH_TOKENS=4096 COLT_COMPONENT_LOG_EVERY=8

log "Stage 1/3: full clean-question/final-latent curriculum (${EXPECTED_STAGE1_STEPS} optimizer steps)."
export COLT_CAUSAL_GROUNDED_LATENT=1 COLT_CAUSAL_GROUNDED_ROI_WEIGHT=0.0
export COLT_CAUSAL_GROUNDED_GEOMETRY_WEIGHT=0.0 COLT_CAUSAL_GROUNDED_INFERENCE_MEDIATED=0
export COLT_MASKED_LATENT_TRAINING=0
export COLT_LARE_REFOCUS=0 COLT_LARE_RECON_WEIGHT=0.0 COLT_COT_ATTN_ALIGN=0
unset COLT_COT_ATTN_TARGETS_PATH
export COLT_TRAIN_RUN_LABEL=two_stage_question_latent_stage1_full33328
bash scripts/lkl_8gpu/colt.sh train paper-faithful --config "$STAGE1_CONFIG" --output-dir "$STAGE1_OUTPUT" --batch-aux

python - "$STAGE1_OUTPUT" "$EXPECTED_STAGE1_STEPS" <<'PY'
import json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
if state.get("global_step") != expected or state.get("max_steps") != expected:
    raise SystemExit(f"Stage-1 full run incomplete: global_step={state.get('global_step')}, max_steps={state.get('max_steps')}")
if config.get("colt_causal_grounded_latent") is not True:
    raise SystemExit("Stage-1 final root did not persist the causal-grounded flag")
if config.get("colt_causal_grounded_roi_weight") != 0.0 or config.get("colt_causal_grounded_geometry_weight") != 0.0:
    raise SystemExit("Stage-1 final root has unexpected non-zero ROI/geometry weight")
print("Stage-1 full checkpoint contract OK.")
PY

log "Stage 2/3: fresh normal-CoLT 88K optimizer run + LaRe reconstruction=0.1, attention=0.05."
export COLT_CAUSAL_GROUNDED_LATENT=0 COLT_CAUSAL_GROUNDED_ROI_WEIGHT=0.0
export COLT_CAUSAL_GROUNDED_GEOMETRY_WEIGHT=0.0 COLT_CAUSAL_GROUNDED_INFERENCE_MEDIATED=0
export COLT_LARE_REFOCUS=1 COLT_LARE_DIM=1536 COLT_LARE_LAYERS=2 COLT_LARE_HEADS=12
export COLT_LARE_QUERIES=4 COLT_LARE_MAX_STEPS=16 COLT_LARE_DROPOUT=0.0
export COLT_LARE_VISUAL_DROPOUT=0.1 COLT_LARE_ATTN_TOPK=0 COLT_LARE_GATE_BIAS=-2.0
export COLT_LARE_RECON_WEIGHT=0.1 COLT_LARE_RECON_STEPS=1000 COLT_LARE_DETACH_VISUAL=1
export COLT_LARE_RECORD_ATTENTION=0
export COLT_COT_ATTN_ALIGN=1 COLT_COT_ATTN_ALIGN_WEIGHT=0.05 COLT_COT_ATTN_MIN_CONFIDENCE=0.05
export COLT_COT_ATTN_TARGETS_PATH="$TEACHER_TARGETS"
export COLT_TRAIN_RUN_LABEL=two_stage_question_latent_stage2_lare_full88833
bash scripts/lkl_8gpu/colt.sh train paper-faithful --config "$STAGE2_CONFIG" --output-dir "$STAGE2_OUTPUT" --batch-aux

python - "$STAGE2_OUTPUT" "$EXPECTED_STAGE2_STEPS" <<'PY'
import json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
if state.get("global_step") != expected or state.get("max_steps") != expected:
    raise SystemExit(f"Stage-2 full run incomplete: global_step={state.get('global_step')}, max_steps={state.get('max_steps')}")
if config.get("colt_lare_refocus") is not True:
    raise SystemExit("Stage-2 final root did not persist LaRe")
if config.get("colt_lare_reconstruction_weight") != 0.1 or config.get("colt_cot_attention_align_weight") != 0.05:
    raise SystemExit("Stage-2 final root did not persist requested LaRe/attention weights")
print("Stage-2 full checkpoint contract OK.")
PY

python scripts/lkl_8gpu/tools/verify_model.py --mode trained --model-dir "$STAGE2_OUTPUT" --expected-step "$EXPECTED_STAGE2_STEPS"

# Project standard all8 contract: `respect-args` is greedy, and 8192 is the
# explicit max-new-token limit. The latent transition stays training-consistent.
log "Stage 3/3: all8 evaluation, greedy +8192, training-consistent latent transition."
export COLT_EXPECTED_GLOBAL_STEP="$EXPECTED_STAGE2_STEPS"
export COLT_EVAL_LOG_LABEL=two_stage_question_latent_stage2_lare_full88833_all8_greedy8192
export COLT_EVAL_MAX_NEW_TOKENS=8192
export VLMEVAL_WORKERS_PER_GPU=3 VLMEVAL_PREFETCH=1 VLMEVAL_EMPTY_CACHE_EVERY_N=0
export VLMEVAL_DIST_BACKEND=gloo COLT_RESEED_PER_SAMPLE=1 COLT_EVAL_SEED=1234
export COLT_LATENT_INTERVENTION=none
bash scripts/lkl_8gpu/colt.sh eval paper-faithful all8 \
  --model-path "$STAGE2_OUTPUT" --gpus "$GPU_CSV" --generation respect-args \
  --latent-transition training-consistent --empty-response-policy prevent --no-reuse

log "Completed full two-stage curriculum and all8 greedy+8192 evaluation."
