# CoT-guided latent visual-attention distillation

This is a training-only addition to LaRe.  A frozen visible-CoT Qwen3-VL
teacher receives the original image, question, and each ground-truth CoT
prefix.  Its current-step text-to-image patch distribution is cached once;
the normal CoLT/LaRe run then distils that soft distribution into the existing
latent evidence-slot cross-attention.  Inference remains the ordinary hidden
three-step latent path: it neither displays nor consumes CoT text.

## Why the target cache is mandatory

Do not load a second 8B teacher inside the FSDP training process.  The target
builder writes a sidecar aligned with a **specific tokenized training cache**.
LLaMA-Factory verifies its source fingerprint and row count before attaching
it, so a resize, filtering, or tokenization mismatch fails rather than silently
matching a map to the wrong image.

The default teacher is the local frozen
`/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct`.  It shares Qwen3-VL's visual
tokenization with the student, making patch indices exactly compatible.  A
larger teacher can be used only through `--teacher-model-path` after its image
token grid has been proven compatible; otherwise its map cannot supervise the
student's native token indices without an additional spatial resampling step.

## Build the frozen teacher sidecar

Run this only after the tokenized cache selected for the next experiment exists.
It needs one otherwise-idle GPU and does not alter the original JSON dataset.

```bash
source /data/nvme0/lkl/CoLT-reproduction/colt-local.env
source /data/nvme0/lkl/miniconda3/etc/profile.d/conda.sh
conda activate colt
cd /data/nvme0/lkl/CoLT-reproduction

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/transformers-4.57.0/src:$PWD/LLaMA-Factory/src" \
python scripts/lkl_8gpu/lare/build_cot_attention_targets.py \
  --tokenized-path /absolute/path/to/tokenized_dataset \
  --output /data/nvme0/lkl/cache/colt/<experiment>_cot_attention_targets \
  --teacher-model-path /data/nvme0/lkl/models/Qwen3-VL-8B-Instruct \
  --teacher-layer 18 \
  --teacher-heads 23:4,21:10,26:20,19:27,23:5,29:11,25:10,24:13 \
  --query-pool visual-mass \
  --num-steps 3
```

When `--teacher-heads` is supplied, the target is an explicit cross-layer
`layer:head` set.  The resulting `metadata.json` records
`teacher_attention_mode=explicit_sparse_layer_head`, sets `teacher_layer` to
`null`, and stores the CLI default separately as `teacher_layer_fallback`.
Thus `teacher_layer=18` must never be read as “all heads in L18”.  Without an
explicit head list, the legacy `single_layer_all_heads` mode uses all query
heads from `--teacher-layer`.  Check teacher confidence and held-out grounding
before an expensive full run.  The `--limit` option is only for a separate,
equally truncated tokenized smoke cache; its output deliberately cannot attach
to the full cache.

### Layer/head calibration (required before the next full sidecar)

Do not select a layer by visualization alone and do not average every head by
default.  The audit below records all 36 layers × 32 query heads while keeping
two distinct quantities:

1. absolute causal-attention mass assigned to image tokens; and
2. the image-conditional spatial distribution.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/transformers-4.57.0/src:$PWD/LLaMA-Factory/src" \
python scripts/lkl_8gpu/lare/audit_cot_attention_heads.py \
  --tokenized-path /absolute/path/to/tokenized_dataset \
  --output /absolute/path/to/calibration/report.json \
  --indices <comma-separated-calibration-rows> \
  --query-pool visual-mass
```

The 2026-08-26 20-row calibration for the local Qwen3-VL-8B checkpoint ranked
the following exploratory fixed set highest:

```text
23:4,21:10,26:20,19:27,23:5,29:11,25:10,24:13
```

On 12 disjoint diagnostic rows, this set reduced mean pairwise step cosine from
0.826 (layer 18, all heads) to 0.625 and increased mean JS divergence from
0.052 to 0.148.  This is a mechanism diagnostic, not yet a full-training gate.
Before building 122K targets, require held-out semantic inspection,
causal-intervention, and a short-pilot check.  Inter-step divergence is useful
only when the CoT steps ask for different visual evidence: repeated focus is
correct for, for example, repeated OCR checks or successive bounding-box
refinement.

The target builder can consume a calibrated set without changing the default
legacy behavior:

```bash
python scripts/lkl_8gpu/lare/build_cot_attention_targets.py \
  ... \
  --teacher-heads 23:4,21:10,26:20,19:27,23:5,29:11,25:10,24:13 \
  --query-pool visual-mass
```

`visual-mass` weights current-step CoT query tokens by their true attention
mass on the image before pooling their image-conditional maps.  The metadata
records both the selected heads and pooling rule.  A cached sidecar built under
a different rule must not be silently reused.

### Causal occlusion gate (run before training)

The maps above are attention-derived hypotheses, not proof that the teacher
uses the highlighted patches.  The validation tool masks an equal number of
top-map, bottom-map, and random Qwen **merged visual cells** before the frozen
teacher's vision encoder.  It then evaluates the teacher-forced NLL of the
same canonical visible-CoT step.  A step passes only if top-cell masking raises
NLL both absolutely and above the stronger matched-size control.

This is intentionally one small safety gate: it does not add a new loss,
semantic parser, dynamic head router, or an artificial diversity objective.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/transformers-4.57.0/src:$PWD/LLaMA-Factory/src" \
python scripts/lkl_8gpu/lare/validate_cot_attention_occlusion.py \
  --tokenized-path /data/nvme0/lkl/cache/colt/qwen_base_colt122k_lare_full_tokenized \
  --output /data/nvme0/lkl/tmp/cot_attention_occlusion_heldout.json \
  --indices 44203,83575,33366,58053,24004,12465,6773,45455,106410,109495,100054,84678 \
  --teacher-heads 23:4,21:10,26:20,19:27,23:5,29:11,25:10,24:13 \
  --query-pool visual-mass \
  --mask-fraction 0.10 --num-random 1 --min-pass-rate 0.60 \
  --min-nll-increase 0.01 --min-control-margin 0.005
```

`top_nll_delta` is the increase after top-map masking.  `control_margin` is
that increase minus the larger of the bottom-map and random-mask increases.
Neither a positive NLL effect nor a passed causal gate proves semantic
correctness: a CoT can causally rely on an incorrect but textually mentioned
diagram label.  Review a stratified bilingual visualization alongside this
report, and do not launch formal training when the held-out pass rate is low or
the passed maps fail that semantic inspection.

`--min-pass-rate` is optional because the operational threshold must be fixed
before inspecting a larger validation audit.  The `0.60` example is a simple
first stop/go criterion, not a claim of a universal statistical threshold.

For a future full-dataset *per-step* filter, first run this audit over every
row of the exact tokenized cache.  Only then may the builder consume it:

```bash
python scripts/lkl_8gpu/lare/build_cot_attention_targets.py \
  ... --causal-audit-path /absolute/path/to/full_coverage_occlusion.json
```

The builder refuses partial reports, fingerprint/config mismatches, and
`--limit` with this option.  This prevents a small held-out report from being
silently applied to unvalidated training rows; failed steps become deliberate
teacher abstentions while the original CoLT objectives still train normally.

### Bilingual 12-case visualization

The visualizer defaults to 12 cases.  Its optional translation sidecar adds
larger English/Chinese text to both PNGs and HTML, while the HTML also exposes
the full three canonical spans at readable size:

The local CJK font is
`/data/nvme0/lkl/cache/fonts/NotoSansCJKsc-Regular.otf` (SHA-256
`2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b`).
Use `--font-path` on another host; bilingual PNG rendering fails explicitly if
the font is absent instead of silently writing square placeholder glyphs.

```bash
python scripts/lkl_8gpu/lare/visualize_cot_attention_maps.py \
  ... \
  --num-examples 12 \
  --teacher-heads 23:4,21:10,26:20,19:27,23:5,29:11,25:10,24:13 \
  --query-pool visual-mass \
  --translations-json scripts/lkl_8gpu/lare/translations/cot_attention_heldout12_zh.json
```

### Step-index contract

The builder uses **the same dynamic three-way CoT splitter as CoLT training**;
it does not independently regroup sentences into a second sequence of semantic
steps.  Therefore teacher map `k` and latent `k` refer to exactly the same CoT
token span used by CoLT's latent-to-CoT and CoT-to-latent losses.  If the
canonical splitter had to make a positional fallback cut because no permitted
boundary was available, the builder writes an empty target for the affected
step.  This abstains only from the attention-teacher loss: CoLT still trains
all three latent steps with its original objectives.

## Enable the loss for a matching training run

```bash
export COLT_LARE_REFOCUS=1
export COLT_COT_ATTN_ALIGN=1
export COLT_COT_ATTN_ALIGN_WEIGHT=0.05
export COLT_COT_ATTN_MIN_CONFIDENCE=0.05
export COLT_COT_ATTN_TARGETS_PATH=/data/nvme0/lkl/cache/colt/<experiment>_cot_attention_targets
```

The total objective gains

\[
\lambda_{\mathrm{attn}}\sum_k r_k\,
\mathrm{KL}(\operatorname{sg}(A_k^{T})\Vert A_k^{Z}).
\]

`r_k` is derived from normalized teacher-map concentration.  Uniform maps and
steps with fewer than two visual tokens abstain automatically; no hard top-k
target or artificial inter-step diversity penalty is used.  Training logs
`cot_attention_alignment_effective_rows` and
`cot_attention_teacher_confidence`; a near-zero effective-row count means the
teacher target needs auditing, not that the loss weight should be increased.
