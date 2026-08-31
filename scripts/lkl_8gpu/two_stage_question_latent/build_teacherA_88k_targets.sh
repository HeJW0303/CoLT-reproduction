#!/usr/bin/env bash
# Build the frozen Teacher-A visible-CoT sidecar for the canonical 88,833-row
# Stage-2 cache. This is intentionally cache-fingerprint-bound: a different
# tokenized cache must produce a different sidecar rather than being reused.

set -Eeuo pipefail

readonly REPO=/data/nvme0/lkl/CoLT-reproduction
readonly ENV_FILE="$REPO/colt-local.env"
readonly CONDA_PROFILE=/data/nvme0/lkl/miniconda3/etc/profile.d/conda.sh
readonly TOKENIZED=/data/nvme0/lkl/cache/colt/qwen_base_colt88k_lare_full_tokenized
readonly TARGETS=/data/nvme0/lkl/cache/colt/qwen_base_colt88k_lare_teacherA_full_targets
readonly SHARD_ROOT=/data/nvme0/lkl/cache/colt/qwen_base_colt88k_lare_teacherA_build_shards
readonly BUILD_LOG_ROOT="$REPO/logs/smoke/two_stage_question_latent/teacherA_88k_target_build"
readonly BASE_MODEL=/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct
readonly EXPECTED_ROWS=88833
readonly TEACHER_HEADS='23:4,21:10,26:20,19:27,23:5,29:11,25:10,24:13'
readonly GPU_CSV=0,1,2,3,4,5,6,7
readonly -a GPUS=(0 1 2 3 4 5 6 7)

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

[[ -f "$ENV_FILE" && -f "$CONDA_PROFILE" ]] || die "Missing local runtime files."
[[ -d "$TOKENIZED" && -d "$BASE_MODEL" ]] || die "Missing canonical 88K cache or base teacher model."
source "$ENV_FILE"
source "$CONDA_PROFILE"
conda activate colt
cd "$REPO"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# This utility may be invoked immediately after Stage 1. The builder uses the
# same multimodal collator, so explicitly reset the stage-specific controls
# instead of inheriting the clean-question assertion into ordinary 88K rows.
export COLT_CAUSAL_GROUNDED_LATENT=0
export COLT_CAUSAL_GROUNDED_ROI_WEIGHT=0.0
export COLT_CAUSAL_GROUNDED_GEOMETRY_WEIGHT=0.0
export COLT_CAUSAL_GROUNDED_INFERENCE_MEDIATED=0

validate_final() {
  python - "$TOKENIZED" "$TARGETS" "$EXPECTED_ROWS" <<'PY'
import json, sys
from pathlib import Path
from datasets import DatasetDict, load_from_disk

tokenized_path, target_path, expected_rows = sys.argv[1:]
expected_rows = int(expected_rows)
train = load_from_disk(tokenized_path)
train = train["train"] if isinstance(train, DatasetDict) else train
targets = load_from_disk(target_path)
targets = targets["train"] if isinstance(targets, DatasetDict) else targets
metadata = json.loads((Path(target_path) / "metadata.json").read_text(encoding="utf-8"))
heads = [[23, 4], [21, 10], [26, 20], [19, 27], [23, 5], [29, 11], [25, 10], [24, 13]]
if len(train) != expected_rows or len(targets) != expected_rows:
    raise SystemExit(f"row mismatch: cache={len(train)}, targets={len(targets)}, expected={expected_rows}")
if metadata.get("source_train_fingerprint") != train._fingerprint:
    raise SystemExit("Teacher-A sidecar/cache fingerprint mismatch")
if metadata.get("teacher_attention_mode") != "explicit_sparse_layer_head":
    raise SystemExit("Teacher-A sidecar is not explicit sparse layer/head mode")
if metadata.get("teacher_layer") is not None or metadata.get("teacher_layer_fallback") != 18:
    raise SystemExit("Teacher-A layer metadata is ambiguous")
if metadata.get("teacher_head_pairs") != heads or metadata.get("teacher_heads") != heads:
    raise SystemExit("Teacher-A head contract mismatch")
if metadata.get("query_pool") != "visual-mass":
    raise SystemExit("Teacher-A query pool mismatch")
print(f"Teacher-A 88K sidecar OK: rows={len(targets)}, fp={train._fingerprint}, coverage={metadata.get('target_coverage')}")
PY
}

if [[ -d "$TARGETS" ]]; then
  [[ -f "$TARGETS/metadata.json" ]] || die "Existing target directory lacks metadata.json: $TARGETS"
  validate_final
  exit 0
fi

if ! nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | \
  awk -F',' '{gsub(/ MiB| /,"",$2); if (($2 + 0) >= 500) bad=1} END {exit bad ? 1 : 0}'; then
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >&2
  die "Teacher-A target construction needs all 8 selected GPUs below 500 MiB."
fi
[[ ! -e "$SHARD_ROOT" ]] || die "Shard root already exists; inspect or remove it deliberately: $SHARD_ROOT"
mkdir -p "$SHARD_ROOT" "$BUILD_LOG_ROOT"

python - "$TOKENIZED" "$SHARD_ROOT" "$EXPECTED_ROWS" <<'PY'
import sys
from pathlib import Path
from datasets import DatasetDict, load_from_disk

tokenized_path, shard_root, expected_rows = sys.argv[1:]
train = load_from_disk(tokenized_path)
train = train["train"] if isinstance(train, DatasetDict) else train
if len(train) != int(expected_rows):
    raise SystemExit(f"canonical 88K cache has {len(train)} rows, expected {expected_rows}")
root = Path(shard_root)
for shard_id in range(8):
    start = len(train) * shard_id // 8
    end = len(train) * (shard_id + 1) // 8
    train.select(range(start, end)).save_to_disk(str(root / f"shard_{shard_id:02d}"))
    print(f"shard={shard_id} start={start} end={end} rows={end-start}", flush=True)
PY

log "Building eight disjoint Teacher-A shards on GPUs $GPU_CSV"
pids=()
for shard_id in "${!GPUS[@]}"; do
  shard_path="$SHARD_ROOT/shard_$(printf '%02d' "$shard_id")"
  target_path="$SHARD_ROOT/target_$(printf '%02d' "$shard_id")"
  log_path="$BUILD_LOG_ROOT/shard_${shard_id}.log"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard_id]}" \
  PYTHONPATH="$REPO/transformers-4.57.0/src:$REPO/LLaMA-Factory/src" \
  python scripts/lkl_8gpu/lare/build_cot_attention_targets.py \
    --tokenized-path "$shard_path" --output "$target_path" \
    --teacher-model-path "$BASE_MODEL" --teacher-layer 18 --teacher-heads "$TEACHER_HEADS" \
    --query-pool visual-mass --num-steps 3 --min-step-tokens 8 \
    --image-max-pixels 802816 --image-min-pixels 1024 --dtype bf16 \
    > "$log_path" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || die "Teacher-A shard construction failed; inspect $BUILD_LOG_ROOT"

python - "$TOKENIZED" "$SHARD_ROOT" "$TARGETS" "$EXPECTED_ROWS" "$TEACHER_HEADS" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
from datasets import DatasetDict, concatenate_datasets, load_from_disk

tokenized_path, shard_root, target_path, expected_rows, raw_heads = sys.argv[1:]
expected_rows = int(expected_rows)
train = load_from_disk(tokenized_path)
train = train["train"] if isinstance(train, DatasetDict) else train
heads = [[int(a), int(b)] for a, b in (pair.split(":", 1) for pair in raw_heads.split(","))]
parts, abstentions = [], Counter()
for shard_id in range(8):
    sidecar = Path(shard_root) / f"target_{shard_id:02d}"
    metadata = json.loads((sidecar / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("teacher_attention_mode"), metadata.get("teacher_layer"), metadata.get("teacher_layer_fallback")) != ("explicit_sparse_layer_head", None, 18):
        raise SystemExit(f"invalid Teacher-A metadata in shard {shard_id}")
    if metadata.get("teacher_head_pairs") != heads or metadata.get("query_pool") != "visual-mass":
        raise SystemExit(f"Teacher-A contract mismatch in shard {shard_id}")
    part = load_from_disk(str(sidecar))
    part = part["train"] if isinstance(part, DatasetDict) else part
    if part.column_names != ["cot_attention_targets"]:
        raise SystemExit(f"unexpected sidecar columns in shard {shard_id}: {part.column_names}")
    parts.append(part)
    abstentions.update(metadata.get("target_abstention_reasons", {}))
merged = concatenate_datasets(parts)
if len(merged) != expected_rows or len(train) != expected_rows:
    raise SystemExit(f"merged/cache row mismatch: merged={len(merged)}, cache={len(train)}, expected={expected_rows}")
merged.save_to_disk(target_path)
maps = merged["cot_attention_targets"]
num_steps = max((len(row) for row in maps), default=0)
hist, step_rows, nonempty_maps = Counter(), [0] * num_steps, 0
for row in maps:
    nonempty = sum(bool(step) for step in row)
    hist[str(nonempty)] += 1
    nonempty_maps += nonempty
    for index, step in enumerate(row):
        if step: step_rows[index] += 1
metadata = {
    "format": "colt_frozen_cot_attention_targets_v2_canonical_steps",
    "source_train_fingerprint": train._fingerprint,
    "source_rows": len(train),
    "teacher_model_path": "/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct",
    "teacher_attention_mode": "explicit_sparse_layer_head",
    "teacher_layer": None,
    "teacher_layer_fallback": 18,
    "teacher_head_pairs": heads,
    "teacher_heads": heads,
    "teacher_head_aggregation": "mean_over_explicit_layer_head_pairs",
    "query_pool": "visual-mass", "num_latent_steps": 3,
    "cot_step_splitter": "colt_dynamic_boundaries_v1", "min_step_tokens": 8,
    "image_max_pixels": 802816, "image_min_pixels": 1024, "template": "qwen3_vl", "dtype": "bf16",
    "causal_audit_path": None, "build_mode": "eight_disjoint_tokenized_shards",
    "target_coverage": {
        "rows_with_any_map": sum(count for key, count in hist.items() if int(key) > 0),
        "rows_with_all_steps": hist.get(str(num_steps), 0), "rows_with_zero_maps": hist.get("0", 0),
        "row_nonempty_step_histogram": dict(hist), "step_rows_with_map": step_rows, "nonempty_step_maps": nonempty_maps,
    },
    "target_abstention_reasons": dict(sorted(abstentions.items())),
}
(Path(target_path) / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(f"Merged Teacher-A 88K sidecar: rows={len(merged)}, fp={train._fingerprint}")
PY
validate_final
