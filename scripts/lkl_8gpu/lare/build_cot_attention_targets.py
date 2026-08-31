#!/usr/bin/env python3
"""Build frozen visible-CoT patch-attention targets for CoLT/LaRe training.

The output is a HuggingFace ``Dataset`` sidecar whose row order is exactly the
input tokenized training split.  It intentionally contains no images, boxes,
or model weights: only ``List[canonical_latent_step][visual_token]`` distributions and
a metadata fingerprint.  LLaMA-Factory attaches it at load time when
``COLT_COT_ATTN_TARGETS_PATH`` is set.

The teacher is a frozen Qwen3-VL checkpoint.  For each ground-truth CoT step,
we reuse CoLT's canonical dynamic three-way partition, append the visible CoT
prefix to the normal image/question prompt, and capture text-query/image-key
scores.  The default legacy mode uses all heads from ``--teacher-layer``;
Teacher-A instead supplies an explicit sparse list of ``layer:head`` pairs
that may span layers.  This is a soft relevance map, not an asserted bounding
box; diffuse maps are later confidence-weighted by the student loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "LLaMA-Factory" / "src"))

from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq  # noqa: E402
from llamafactory.data.template import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams.data_args import DataArguments  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    apply_rotary_pos_emb,
    extract_think_content_robust,
    repeat_kv,
    split_cot_by_dynamic_boundaries_with_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--teacher-model-path",
        type=Path,
        default=Path("/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument("--template", default="qwen3_vl")
    parser.add_argument("--teacher-layer", type=int, default=18)
    parser.add_argument(
        "--teacher-heads",
        default=None,
        help=(
            "Optional comma-separated layer:query-head pairs, for example "
            "23:4,21:10,26:20. When set, these heads replace the all-head "
            "average from --teacher-layer."
        ),
    )
    parser.add_argument(
        "--query-pool",
        choices=("mean", "last", "visual-mass"),
        default="mean",
        help="How to pool the current canonical CoT span within each selected head.",
    )
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument(
        "--min-step-tokens",
        type=int,
        default=8,
        help="Must match CoLT's dynamic splitter (default: 8).",
    )
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--image-min-pixels", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--causal-audit-path",
        type=Path,
        default=None,
        help=(
            "Optional full-coverage report from validate_cot_attention_occlusion.py. "
            "When supplied, only causally passed canonical steps receive a teacher target."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def select_train_split(dataset: Dataset | DatasetDict) -> Dataset:
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("Tokenized dataset must contain a train split.")
        return dataset["train"]
    return dataset


def get_teacher_attention_module(model: Any, layer_index: int) -> Any:
    layers = model.model.language_model.layers
    if not 0 <= layer_index < len(layers):
        raise ValueError(f"--teacher-layer={layer_index} is outside [0, {len(layers) - 1}].")
    return layers[layer_index].self_attn


def parse_teacher_heads(raw: str | None, model: Any, fallback_layer: int) -> list[tuple[int, int]]:
    """Resolve an explicit sparse head set or the legacy all-head layer."""
    layers = model.model.language_model.layers
    if raw is None:
        module = get_teacher_attention_module(model, fallback_layer)
        num_heads = int(module.q_proj.out_features // module.head_dim)
        return [(fallback_layer, head_index) for head_index in range(num_heads)]
    pairs: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            raise ValueError("--teacher-heads must contain comma-separated layer:head pairs.")
        layer_text, head_text = item.split(":", 1)
        layer_index, head_index = int(layer_text), int(head_text)
        if not 0 <= layer_index < len(layers):
            raise ValueError(f"Teacher layer {layer_index} is outside [0, {len(layers) - 1}].")
        module = layers[layer_index].self_attn
        num_heads = int(module.q_proj.out_features // module.head_dim)
        if not 0 <= head_index < num_heads:
            raise ValueError(
                f"Teacher query head {head_index} is outside [0, {num_heads - 1}] for layer {layer_index}."
            )
        pairs.append((layer_index, head_index))
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("--teacher-heads must contain at least one unique layer:head pair.")
    return pairs


def teacher_attention_metadata(
    fallback_layer: int,
    layer_heads: list[tuple[int, int]] | None,
    *,
    explicit_heads: bool = False,
) -> dict[str, Any]:
    """Return an unambiguous, serialisable teacher-attention contract.

    ``teacher_layer`` is retained only for the legacy all-head mode.  In the
    explicit sparse mode it is deliberately ``null``; the fallback layer is
    recorded separately so metadata cannot be misread as "all heads at layer
    18" when the actual target is a cross-layer head set.
    """
    if explicit_heads:
        if not layer_heads:
            raise ValueError("explicit_heads=True requires a non-empty layer_heads list")
        pairs = [[int(layer), int(head)] for layer, head in layer_heads]
        return {
            "teacher_attention_mode": "explicit_sparse_layer_head",
            "teacher_layer": None,
            "teacher_layer_fallback": int(fallback_layer),
            "teacher_head_pairs": pairs,
            # Kept as a read-only compatibility alias for old tooling.
            "teacher_heads": pairs,
            "teacher_head_aggregation": "mean_over_explicit_layer_head_pairs",
        }
    return {
        "teacher_attention_mode": "single_layer_all_heads",
        "teacher_layer": int(fallback_layer),
        "teacher_layer_fallback": int(fallback_layer),
        "teacher_head_pairs": None,
        "teacher_heads": None,
        "teacher_head_aggregation": "mean_over_all_query_heads_in_teacher_layer",
    }


def summarize_target_coverage(targets: list[list[list[float]]]) -> dict[str, Any]:
    """Summarise empty/partial/full rows so abstention is visible in metadata."""
    num_steps = max((len(row) for row in targets), default=0)
    step_rows = [0] * num_steps
    row_hist: dict[str, int] = {}
    nonempty_maps = 0
    for row in targets:
        nonempty = sum(bool(step) for step in row)
        row_hist[str(nonempty)] = row_hist.get(str(nonempty), 0) + 1
        nonempty_maps += nonempty
        for index, step in enumerate(row):
            if step:
                step_rows[index] += 1
    return {
        "rows_with_any_map": sum(count for key, count in row_hist.items() if int(key) > 0),
        "rows_with_all_steps": row_hist.get(str(num_steps), 0),
        "rows_with_zero_maps": row_hist.get("0", 0),
        "row_nonempty_step_histogram": row_hist,
        "step_rows_with_map": step_rows,
        "nonempty_step_maps": nonempty_maps,
    }


def make_prefix_feature(
    row: dict[str, Any],
    *,
    cot_prefix: torch.Tensor,
    think_token_id: int,
    end_think_token_id: int,
    answer_token_id: int,
) -> tuple[dict[str, Any], int, int]:
    source_ids = torch.tensor(row["input_ids"], dtype=torch.long)
    question_ids, _, _ = extract_think_content_robust(
        source_ids,
        think_token_id=think_token_id,
        end_think_token_id=end_think_token_id,
        answer_token_id=answer_token_id,
    )
    prefix = torch.cat(
        [question_ids.cpu(), torch.tensor([think_token_id], dtype=torch.long), cot_prefix.cpu()]
    )
    query_start = int(question_ids.numel() + 1 + cot_prefix.numel())
    feature = dict(row)
    feature["input_ids"] = prefix.tolist()
    feature["attention_mask"] = [1] * len(feature["input_ids"])
    feature["labels"] = [-100] * len(feature["input_ids"])
    return feature, query_start, len(feature["input_ids"])


def expand_gqa_keys_for_queries(query: torch.Tensor, key: torch.Tensor, attention_module: Any) -> torch.Tensor:
    """Match Qwen's native GQA head expansion before explicit QK inspection."""
    if query.shape[1] == key.shape[1]:
        return key
    groups = int(getattr(attention_module, "num_key_value_groups", 0))
    if groups <= 0 or key.shape[1] * groups != query.shape[1]:
        raise ValueError(
            "Cannot align Qwen query and KV heads for frozen attention extraction: "
            f"query_heads={query.shape[1]}, kv_heads={key.shape[1]}, "
            f"num_key_value_groups={groups}."
        )
    return repeat_kv(key, groups)


def image_placeholder_compatibility_error(teacher: Any, model_inputs: dict[str, Any]) -> str | None:
    """Return a cache/preprocessing mismatch before attempting the teacher forward.

    Some legacy tokenized rows retain an image-placeholder count created under
    a different resize policy.  Their image feature grid cannot be aligned to
    those placeholders, so an attention map would have no well-defined patch
    index.  The caller must abstain for that row rather than fabricate a map.
    """
    image_grid_thw = model_inputs.get("image_grid_thw")
    input_ids = model_inputs.get("input_ids")
    if image_grid_thw is None or input_ids is None:
        return "missing image_grid_thw or input_ids"
    merge_size = int(teacher.config.vision_config.spatial_merge_size)
    expected_features = int((image_grid_thw.prod(dim=-1) // (merge_size**2)).sum().item())
    actual_placeholders = int(input_ids.eq(teacher.config.image_token_id).sum().item())
    if actual_placeholders != expected_features:
        return (
            "image placeholder/feature mismatch "
            f"(tokens={actual_placeholders}, features={expected_features}, merge={merge_size})"
        )
    return None


def assert_current_cot_query_tokens(
    model_inputs: dict[str, Any], current_cot_ids: torch.Tensor, query_start: int, query_end: int
) -> None:
    """Fail if a collator transformation shifts the intended visible-CoT query."""
    actual = model_inputs["input_ids"][0, query_start:query_end]
    expected = current_cot_ids.to(device=actual.device, dtype=actual.dtype)
    if actual.numel() != expected.numel() or not torch.equal(actual, expected):
        raise ValueError(
            "Frozen CoT teacher query range no longer equals the canonical current CoT span; "
            f"range=[{query_start}, {query_end}), actual_tokens={actual.numel()}, "
            f"expected_tokens={expected.numel()}. Rebuild the tokenized cache or inspect the multimodal collator."
        )


@torch.inference_mode()
def teacher_map_for_prefix(
    teacher: Any,
    attention_module: Any,
    model_inputs: dict[str, torch.Tensor],
    *,
    query_start: int,
    query_end: int,
) -> list[float]:
    """Read one layer's exact QK scores without requesting dense all-layer attention."""
    captured: dict[str, torch.Tensor] = {}

    def capture_qk(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        position_embeddings = kwargs.get("position_embeddings")
        if hidden_states is None or position_embeddings is None:
            raise RuntimeError("Could not capture Qwen attention inputs for the frozen CoT teacher.")
        hidden_shape = (*hidden_states.shape[:-1], -1, module.head_dim)
        query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        captured["query"] = query.detach()
        captured["key"] = key.detach()

    handle = attention_module.register_forward_pre_hook(capture_qk, with_kwargs=True)
    try:
        teacher.model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            position_ids=model_inputs["position_ids"],
            pixel_values=model_inputs.get("pixel_values"),
            pixel_values_videos=model_inputs.get("pixel_values_videos"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            use_cache=False,
        )
    finally:
        handle.remove()
    if "query" not in captured:
        raise RuntimeError("Frozen CoT teacher hook did not run.")
    image_positions = torch.nonzero(
        model_inputs["input_ids"][0].eq(teacher.config.image_token_id), as_tuple=False
    ).flatten()
    if image_positions.numel() == 0:
        return []
    if not 0 <= query_start < query_end <= model_inputs["input_ids"].shape[1]:
        raise ValueError(f"Invalid current-CoT query range [{query_start}, {query_end}).")
    query = captured["query"][:, :, query_start:query_end, :].float()
    key = captured["key"][:, :, image_positions, :].float()
    # Qwen3-VL-8B uses grouped-query attention (32 query heads / 8 KV heads).
    # The real attention forward expands keys with ``repeat_kv`` before QK;
    # mirror that exact operation for the extracted diagnostic distribution.
    key = expand_gqa_keys_for_queries(query, key, attention_module)
    scores = torch.matmul(query, key.transpose(-1, -2)) * attention_module.scaling
    # Re-normalize over visual tokens only: this is a spatial relevance map,
    # not the full causal-attention mass diluted by text-prefix positions.
    attention = torch.softmax(scores, dim=-1).mean(dim=(1, 2))[0]
    return attention.to(dtype=torch.float16).cpu().tolist()


@torch.inference_mode()
def teacher_map_for_prefix_heads(
    teacher: Any,
    layer_heads: list[tuple[int, int]],
    model_inputs: dict[str, torch.Tensor],
    *,
    query_start: int,
    query_end: int,
    query_pool: str,
) -> list[float]:
    """Aggregate a calibrated sparse set of layer/head image retrievers.

    Unlike the legacy single-layer helper, ``visual-mass`` pooling first uses
    the full causal softmax to estimate whether each CoT token actually looks
    at the image.  Spatial maps remain normalized within image tokens so they
    are compatible with the LaRe cross-attention target.
    """
    if query_pool not in {"mean", "last", "visual-mass"}:
        raise ValueError(f"Unsupported query pooling rule: {query_pool!r}.")
    if not layer_heads:
        raise ValueError("At least one teacher layer/head pair is required.")
    input_ids = model_inputs["input_ids"]
    sequence_length = int(input_ids.shape[1])
    image_positions = torch.nonzero(
        input_ids[0].eq(teacher.config.image_token_id), as_tuple=False
    ).flatten()
    if image_positions.numel() == 0:
        return []
    if not 0 <= query_start < query_end <= sequence_length:
        raise ValueError(f"Invalid current-CoT query range [{query_start}, {query_end}).")

    heads_by_layer: dict[int, list[int]] = {}
    for layer_index, head_index in layer_heads:
        heads_by_layer.setdefault(layer_index, []).append(head_index)
    captured: dict[tuple[int, int], torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int, selected_heads: list[int]):
        def capture(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                raise RuntimeError("Could not capture Qwen attention inputs for sparse-head teacher.")
            hidden_shape = (*hidden_states.shape[:-1], -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            query = query[:, :, query_start:query_end, :].float()
            key = expand_gqa_keys_for_queries(query, key.float(), module)
            query = query[:, selected_heads]
            key = key[:, selected_heads]
            scores = torch.matmul(query, key.transpose(-1, -2)) * module.scaling
            image_scores = scores[..., image_positions]
            conditional = torch.softmax(image_scores, dim=-1)[0]

            if query_pool == "last":
                pooled = conditional[:, -1, :]
            elif query_pool == "mean":
                pooled = conditional.mean(dim=1)
            else:
                query_positions = torch.arange(query_start, query_end, device=scores.device)
                key_positions = torch.arange(sequence_length, device=scores.device)
                causal = key_positions[None, :] <= query_positions[:, None]
                valid_keys = model_inputs["attention_mask"][0].bool()[None, :] & causal
                full_scores = scores.masked_fill(
                    ~valid_keys[None, None, :, :], torch.finfo(scores.dtype).min
                )
                image_mass = torch.softmax(full_scores, dim=-1)[..., image_positions].sum(dim=-1)[0]
                weights = image_mass / image_mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
                pooled = (conditional * weights[..., None]).sum(dim=1)
            pooled = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            for local_index, head_index in enumerate(selected_heads):
                captured[(layer_index, head_index)] = pooled[local_index].detach().cpu()

        return capture

    layers = teacher.model.language_model.layers
    for layer_index, selected_heads in heads_by_layer.items():
        handles.append(
            layers[layer_index].self_attn.register_forward_pre_hook(
                make_hook(layer_index, selected_heads), with_kwargs=True
            )
        )
    try:
        teacher.model(
            input_ids=input_ids,
            attention_mask=model_inputs["attention_mask"],
            position_ids=model_inputs["position_ids"],
            pixel_values=model_inputs.get("pixel_values"),
            pixel_values_videos=model_inputs.get("pixel_values_videos"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    missing = [pair for pair in layer_heads if pair not in captured]
    if missing:
        raise RuntimeError(f"Sparse-head teacher hook missed selected heads: {missing}.")
    attention = torch.stack([captured[pair] for pair in layer_heads]).mean(dim=0)
    attention = attention / attention.sum().clamp_min(1e-12)
    return attention.to(dtype=torch.float16).tolist()


def move_tensor_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def load_full_coverage_causal_audit(
    path: Path,
    *,
    tokenized: Dataset,
    teacher_model_path: Path,
    teacher_layer: int,
    layer_heads: list[tuple[int, int]] | None,
    explicit_heads: bool = False,
    query_pool: str,
    num_steps: int,
    min_step_tokens: int,
    image_max_pixels: int,
    image_min_pixels: int,
) -> dict[int, list[bool]]:
    """Load a matching *full-dataset* causal gate without guessing missing rows.

    A small held-out audit is evidence for whether a teacher is promising, but
    cannot be silently repurposed as a per-example filter for a 122K sidecar.
    This deliberately accepts only a report that covers every row of the exact
    tokenized cache.  Missing or mismatched coverage is a hard error rather
    than an accidental all-abstain training set.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "colt_cot_attention_causal_occlusion_v1":
        raise ValueError("--causal-audit-path is not a supported causal-occlusion report.")
    expected = {
        "tokenized_fingerprint": getattr(tokenized, "_fingerprint", None),
        "source_rows": len(tokenized),
        "teacher_model_path": str(teacher_model_path),
        "query_pool": query_pool,
        "num_steps": num_steps,
        "min_step_tokens": min_step_tokens,
        "image_max_pixels": image_max_pixels,
        "image_min_pixels": image_min_pixels,
    }
    if "teacher_attention_mode" in payload:
        expected.update(
            teacher_attention_metadata(
                teacher_layer,
                layer_heads,
                explicit_heads=explicit_heads,
            )
        )
    else:
        # Read old audit reports so an already completed causal gate remains
        # usable; all newly written reports use the unambiguous schema above.
        expected.update({"teacher_layer": teacher_layer, "teacher_heads": layer_heads})
    for key, value in expected.items():
        actual = payload.get(key)
        if key in {"teacher_heads", "teacher_head_pairs"} and actual is not None:
            actual = [tuple(pair) for pair in actual]
            value = [tuple(pair) for pair in value] if value is not None else None
        if actual != value:
            raise ValueError(
                "Causal-occlusion report does not match this sidecar build: "
                f"{key}={actual!r}, expected {value!r}."
            )
    records = payload.get("rows")
    if not isinstance(records, list) or len(records) != len(tokenized):
        raise ValueError(
            "--causal-audit-path must cover every row of the tokenized cache. "
            "A held-out report is a gate for deciding whether to proceed, not a per-row training filter."
        )
    passed: dict[int, list[bool]] = {}
    for record in records:
        row_index = record.get("row_index")
        steps = record.get("steps")
        if not isinstance(row_index, int) or row_index in passed or not isinstance(steps, list):
            raise ValueError("Causal-occlusion report has invalid or duplicate row records.")
        if len(steps) != num_steps:
            raise ValueError("Causal-occlusion report has an unexpected number of steps.")
        passed[row_index] = [bool(step.get("passed", False)) for step in steps]
    if set(passed) != set(range(len(tokenized))):
        raise ValueError("Causal-occlusion report row indices do not exactly cover the tokenized cache.")
    return passed


def build_maps_for_row(
    row: dict[str, Any],
    *,
    collator: Any,
    teacher: Any,
    attention_module: Any,
    layer_heads: list[tuple[int, int]] | None,
    query_pool: str,
    num_steps: int,
    device: torch.device,
    think_token_id: int,
    end_think_token_id: int,
    answer_token_id: int,
    boundary_token_ids: set[int],
    eos_token_id: int,
    min_step_tokens: int,
    causal_passes: list[bool] | None = None,
    abstention_stats: Counter[str] | None = None,
) -> list[list[float]]:
    source_ids = torch.tensor(row["input_ids"], dtype=torch.long)
    _, cot_ids, _ = extract_think_content_robust(
        source_ids,
        think_token_id=think_token_id,
        end_think_token_id=end_think_token_id,
        answer_token_id=answer_token_id,
    )
    # This must be exactly CoLT's own step partition.  A teacher target for
    # semantic span B must never supervise a latent whose decoder target is
    # CoLT span A.  ``teacher_eligible`` merely abstains on forced cuts; it
    # never invents a separate semantic-step numbering.
    steps, _, split_metadata = split_cot_by_dynamic_boundaries_with_metadata(
        cot_ids,
        num_steps=num_steps,
        eos_token_id=eos_token_id,
        boundary_token_ids=boundary_token_ids,
        min_step_tokens=min_step_tokens,
    )
    maps: list[list[float]] = []
    visible_prefix = torch.empty(0, dtype=torch.long)
    for step_index, step in enumerate(steps):
        # The canonical splitter appends EOS for decoder supervision.  It is
        # not visible CoT content, so omit it from the frozen-teacher prompt.
        current = step[:-1].detach().cpu()
        visible_prefix = torch.cat([visible_prefix, current])
        if (
            not split_metadata["teacher_eligible"][step_index]
            or current.numel() == 0
            or (causal_passes is not None and not causal_passes[step_index])
        ):
            if abstention_stats is not None:
                if not split_metadata["teacher_eligible"][step_index]:
                    abstention_stats["teacher_ineligible_steps"] += 1
                if current.numel() == 0:
                    abstention_stats["empty_cot_steps"] += 1
                if causal_passes is not None and not causal_passes[step_index]:
                    abstention_stats["causal_gate_steps"] += 1
            maps.append([])
            continue
        feature, prefix_start, prefix_end = make_prefix_feature(
            row,
            cot_prefix=visible_prefix,
            think_token_id=think_token_id,
            end_think_token_id=end_think_token_id,
            answer_token_id=answer_token_id,
        )
        # Current canonical step only is the language query; prior canonical
        # steps are context.  Thus the resulting map has the same index as the
        # latent-to-CoT / CoT-to-latent target in the training forward pass.
        query_start = prefix_start - current.numel()
        batch = move_tensor_inputs(collator([feature]), device)
        compatibility_error = image_placeholder_compatibility_error(teacher, batch)
        if compatibility_error is not None:
            if abstention_stats is not None:
                abstention_stats["image_placeholder_feature_mismatch_rows"] += 1
            # Preserve sidecar row alignment while abstaining from an undefined
            # map.  The normal CoLT losses are still trained on this example.
            return [[] for _ in range(num_steps)]
        assert_current_cot_query_tokens(batch, current, query_start, prefix_end)
        if layer_heads is None:
            attention_map = teacher_map_for_prefix(
                teacher,
                attention_module,
                batch,
                query_start=query_start,
                query_end=prefix_end,
            )
            maps.append(attention_map)
        else:
            attention_map = teacher_map_for_prefix_heads(
                teacher,
                layer_heads,
                batch,
                query_start=query_start,
                query_end=prefix_end,
                query_pool=query_pool,
            )
            maps.append(attention_map)
        if not maps[-1] and abstention_stats is not None:
            abstention_stats["no_image_positions_steps"] += 1
    return maps


def main() -> None:
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if args.min_step_tokens <= 0:
        raise ValueError("--min-step-tokens must be positive.")
    if not 0 < args.image_min_pixels <= args.image_max_pixels:
        raise ValueError("image pixel limits must satisfy 0 < min <= max.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing sidecar: {args.output}")
    tokenized = select_train_split(load_from_disk(str(args.tokenized_path)))
    if args.causal_audit_path is not None and args.limit is not None:
        raise ValueError("--causal-audit-path requires the complete tokenized cache; omit --limit.")
    if args.limit is not None:
        if not 0 < args.limit <= len(tokenized):
            raise ValueError(f"--limit must be in [1, {len(tokenized)}].")
        tokenized = tokenized.select(range(args.limit))
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, use_fast=False, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.teacher_model_path, local_files_only=True)
    # Must match the SFT YAML.  A different resize policy changes the number
    # and order of image placeholders, making a cached target unsafe to use.
    processor.image_max_pixels = args.image_max_pixels
    processor.image_min_pixels = args.image_min_pixels
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=args.template))
    # The teacher only needs the base multimodal backbone.  Suppress CoLT's
    # auxiliary decoders before construction so target building neither loads
    # Qwen-0.6B decoders nor accidentally creates recursive latent modules.
    os.environ["COLT_DISABLE_LATENT_REASONING"] = "1"
    teacher = AutoModelForImageTextToText.from_pretrained(
        args.teacher_model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    explicit_heads = args.teacher_heads is not None
    teacher.eval()
    # The fallback module is only needed by legacy all-head mode.  In
    # explicit sparse mode, never even bind a single "teacher layer" object;
    # every selected layer/head is installed by teacher_map_for_prefix_heads.
    attention_module = None if explicit_heads else get_teacher_attention_module(teacher, args.teacher_layer)
    layer_heads = (
        None
        if not explicit_heads and args.query_pool == "mean"
        else parse_teacher_heads(args.teacher_heads, teacher, args.teacher_layer)
    )
    causal_passes_by_row = (
        None
        if args.causal_audit_path is None
        else load_full_coverage_causal_audit(
            args.causal_audit_path,
            tokenized=tokenized,
            teacher_model_path=args.teacher_model_path,
            teacher_layer=args.teacher_layer,
            layer_heads=layer_heads,
            explicit_heads=explicit_heads,
            query_pool=args.query_pool,
            num_steps=args.num_steps,
            min_step_tokens=args.min_step_tokens,
            image_max_pixels=args.image_max_pixels,
            image_min_pixels=args.image_min_pixels,
        )
    )
    think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
    end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    answer_token_id = tokenizer.encode("answer", add_special_tokens=False)[0]
    # Exactly mirror ``Qwen3VLForConditionalGeneration``.  This is a
    # structural contract, not a claim that every comma creates a good visual
    # reasoning unit: forced cuts abstain via ``teacher_eligible`` and diffuse
    # maps abstain later through confidence weighting.
    boundary_token_ids = {
        token_id
        for text in ("\n", "\n\n", ".", "。", "?", "？", "!", "！", ";", "；", ":", "：", ",", "，")
        for token_id in tokenizer.encode(text, add_special_tokens=False)
    }
    collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        model=teacher,
        tokenizer=tokenizer,
        processor=processor,
        label_pad_token_id=-100,
    )
    targets = []
    abstention_stats: Counter[str] = Counter()
    for index, row in enumerate(tokenized):
        targets.append(
            build_maps_for_row(
                row,
                collator=collator,
                teacher=teacher,
                attention_module=attention_module,
                layer_heads=layer_heads,
                query_pool=args.query_pool,
                num_steps=args.num_steps,
                device=device,
                think_token_id=think_token_id,
                end_think_token_id=end_think_token_id,
                answer_token_id=answer_token_id,
                boundary_token_ids=boundary_token_ids,
                eos_token_id=tokenizer.eos_token_id,
                min_step_tokens=args.min_step_tokens,
                causal_passes=(None if causal_passes_by_row is None else causal_passes_by_row[index]),
                abstention_stats=abstention_stats,
            )
        )
        if (index + 1) % 100 == 0 or index + 1 == len(tokenized):
            print(f"built frozen CoT attention maps: {index + 1}/{len(tokenized)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"cot_attention_targets": targets}).save_to_disk(str(args.output))
    metadata = {
        "format": "colt_frozen_cot_attention_targets_v2_canonical_steps",
        "source_train_fingerprint": getattr(tokenized, "_fingerprint", None),
        "source_rows": len(tokenized),
        "teacher_model_path": str(args.teacher_model_path),
        "query_pool": args.query_pool,
        "num_latent_steps": args.num_steps,
        "cot_step_splitter": "colt_dynamic_boundaries_v1",
        "min_step_tokens": args.min_step_tokens,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "template": args.template,
        "dtype": args.dtype,
        "causal_audit_path": None if args.causal_audit_path is None else str(args.causal_audit_path),
        "target_coverage": summarize_target_coverage(targets),
        "target_abstention_reasons": dict(sorted(abstention_stats.items())),
    }
    metadata.update(
        teacher_attention_metadata(
            args.teacher_layer,
            layer_heads,
            explicit_heads=explicit_heads,
        )
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
