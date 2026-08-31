#!/usr/bin/env python3
"""Audit every Qwen text-attention layer/head as a CoT-to-image retriever.

The production target builder historically averaged every head in one chosen
layer after renormalizing over image tokens.  That diagnostic cannot tell a
real visual head from a language/background head: every head is forced to
produce an image distribution.  This audit records, for every layer and query
head, both

* its *absolute* causal-attention mass on image tokens; and
* its image-conditional spatial distribution.

The output is deliberately a small calibration artifact, not a training
sidecar.  Use disjoint calibration and visualization rows before promoting a
head-selection rule into the full target builder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "LLaMA-Factory" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq  # noqa: E402
from llamafactory.data.template import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams.data_args import DataArguments  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    apply_rotary_pos_emb,
    extract_think_content_robust,
    repeat_kv,
    split_cot_by_dynamic_boundaries_with_metadata,
)

from build_cot_attention_targets import (  # noqa: E402
    assert_current_cot_query_tokens,
    image_placeholder_compatibility_error,
    make_prefix_feature,
    move_tensor_inputs,
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
    parser.add_argument("--indices", required=True, help="Comma-separated unique tokenized row indices.")
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--min-step-tokens", type=int, default=8)
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--image-min-pixels", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--query-pool",
        choices=("mean", "last", "visual-mass"),
        default="visual-mass",
        help="Pool current-step CoT query tokens per head.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Heads used by the adaptive diagnostic map.")
    return parser.parse_args()


def select_train_split(dataset: Dataset | DatasetDict) -> Dataset:
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("Tokenized dataset must contain a train split.")
        return dataset["train"]
    return dataset


def parse_indices(raw: str, total: int) -> list[int]:
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not indices:
        raise ValueError("--indices did not contain any integers.")
    if len(set(indices)) != len(indices) or any(index < 0 or index >= total for index in indices):
        raise ValueError(f"--indices must be unique values in [0, {total - 1}].")
    return indices


def canonical_boundary_token_ids(tokenizer: Any) -> set[int]:
    return {
        token_id
        for text in ("\n", "\n\n", ".", "。", "?", "？", "!", "！", ";", "；", ":", "：", ",", "，")
        for token_id in tokenizer.encode(text, add_special_tokens=False)
    }


def expand_gqa_keys(query: torch.Tensor, key: torch.Tensor, module: Any) -> torch.Tensor:
    if query.shape[1] == key.shape[1]:
        return key
    groups = int(getattr(module, "num_key_value_groups", 0))
    if groups <= 0 or key.shape[1] * groups != query.shape[1]:
        raise ValueError(
            "Cannot expand grouped-query keys: "
            f"query_heads={query.shape[1]}, kv_heads={key.shape[1]}, groups={groups}."
        )
    return repeat_kv(key, groups)


@torch.inference_mode()
def collect_all_layer_head_maps(
    teacher: Any,
    model_inputs: dict[str, torch.Tensor],
    *,
    query_start: int,
    query_end: int,
    query_pool: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(conditional_maps, image_mass, entropy)`` for all heads.

    Shapes are ``[layers, heads, visual_tokens]``, ``[layers, heads]``, and
    ``[layers, heads]``.  Image mass is computed from the full causal softmax;
    the spatial map is separately normalized within image tokens.
    """
    input_ids = model_inputs["input_ids"]
    image_positions = torch.nonzero(input_ids[0].eq(teacher.config.image_token_id), as_tuple=False).flatten()
    if image_positions.numel() == 0:
        raise ValueError("Cannot audit a row without image tokens.")
    sequence_length = int(input_ids.shape[1])
    if not 0 <= query_start < query_end <= sequence_length:
        raise ValueError(f"Invalid current-CoT query range [{query_start}, {query_end}).")

    captured_maps: dict[int, torch.Tensor] = {}
    captured_mass: dict[int, torch.Tensor] = {}
    captured_entropy: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def capture(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                raise RuntimeError("Could not capture Qwen attention inputs during layer/head audit.")
            hidden_shape = (*hidden_states.shape[:-1], -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            query = query[:, :, query_start:query_end, :].float()
            key = expand_gqa_keys(query, key.float(), module)
            scores = torch.matmul(query, key.transpose(-1, -2)) * module.scaling

            # Each visible CoT token can see only its own causal prefix.  There
            # is no padding for the batch-1 audit, but retain the input mask so
            # the invariant remains explicit if the collator changes later.
            query_positions = torch.arange(query_start, query_end, device=scores.device)
            key_positions = torch.arange(sequence_length, device=scores.device)
            causal = key_positions[None, :] <= query_positions[:, None]
            valid_keys = model_inputs["attention_mask"][0].bool()[None, :] & causal
            full_scores = scores.masked_fill(~valid_keys[None, None, :, :], torch.finfo(scores.dtype).min)
            full_attention = torch.softmax(full_scores, dim=-1)
            per_token_mass = full_attention[..., image_positions].sum(dim=-1)[0]  # [heads, query]

            image_scores = scores[..., image_positions]
            conditional = torch.softmax(image_scores, dim=-1)[0]  # [heads, query, visual]
            if query_pool == "last":
                pooled = conditional[:, -1, :]
                mass = per_token_mass[:, -1]
            elif query_pool == "mean":
                pooled = conditional.mean(dim=1)
                mass = per_token_mass.mean(dim=1)
            else:
                weights = per_token_mass / per_token_mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
                pooled = (conditional * weights[..., None]).sum(dim=1)
                mass = per_token_mass.mean(dim=1)
            pooled = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = -(pooled.clamp_min(1e-12).log() * pooled).sum(dim=-1) / math.log(pooled.shape[-1])
            captured_maps[layer_index] = pooled.to(torch.float16).cpu()
            captured_mass[layer_index] = mass.to(torch.float32).cpu()
            captured_entropy[layer_index] = entropy.to(torch.float32).cpu()

        return capture

    layers = teacher.model.language_model.layers
    for layer_index, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(layer_index), with_kwargs=True))
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
    expected = set(range(len(layers)))
    if set(captured_maps) != expected:
        raise RuntimeError(f"Layer/head audit missed layers: {sorted(expected - set(captured_maps))}.")
    maps = torch.stack([captured_maps[index] for index in range(len(layers))]).numpy()
    mass = torch.stack([captured_mass[index] for index in range(len(layers))]).numpy()
    entropy = torch.stack([captured_entropy[index] for index in range(len(layers))]).numpy()
    return maps, mass, entropy


def pairwise_step_metrics(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean pairwise ``(JS, cosine)`` per layer/head for three steps."""
    if maps.ndim != 4 or maps.shape[0] != 3:
        raise ValueError(f"Expected maps [3,layers,heads,visual], got {maps.shape}.")
    js_values = []
    cosine_values = []
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        left = maps[left_index].astype(np.float64)
        right = maps[right_index].astype(np.float64)
        midpoint = (left + right) / 2
        js_values.append(
            0.5
            * (
                np.sum(left * np.log((left + 1e-12) / (midpoint + 1e-12)), axis=-1)
                + np.sum(right * np.log((right + 1e-12) / (midpoint + 1e-12)), axis=-1)
            )
        )
        numerator = np.sum(left * right, axis=-1)
        denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
        cosine_values.append(numerator / np.maximum(denominator, 1e-12))
    return np.mean(js_values, axis=0), np.mean(cosine_values, axis=0)


def normalized_rank(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    order = np.argsort(np.argsort(flat, kind="stable"), kind="stable").astype(np.float64)
    if flat.size > 1:
        order /= flat.size - 1
    return order.reshape(values.shape)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite layer/head audit: {args.output}")
    if args.num_steps != 3:
        raise ValueError("The current audit metric expects exactly three CoLT steps.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = select_train_split(load_from_disk(str(args.tokenized_path)))
    indices = parse_indices(args.indices, len(dataset))
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, use_fast=False, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.teacher_model_path, local_files_only=True)
    processor.image_max_pixels = args.image_max_pixels
    processor.image_min_pixels = args.image_min_pixels
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=args.template))
    os.environ["COLT_DISABLE_LATENT_REASONING"] = "1"
    teacher = AutoModelForImageTextToText.from_pretrained(
        args.teacher_model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    teacher.eval()
    collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        model=teacher,
        tokenizer=tokenizer,
        processor=processor,
        label_pad_token_id=-100,
    )
    think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
    end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    answer_token_id = tokenizer.encode("answer", add_special_tokens=False)[0]
    boundary_token_ids = canonical_boundary_token_ids(tokenizer)

    rows = []
    aggregate_mass = []
    aggregate_entropy = []
    aggregate_js = []
    aggregate_cosine = []
    for ordinal, row_index in enumerate(indices, start=1):
        row = dataset[row_index]
        source_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        _, cot_ids, _ = extract_think_content_robust(
            source_ids, think_token_id, end_think_token_id, answer_token_id
        )
        steps, _, metadata = split_cot_by_dynamic_boundaries_with_metadata(
            cot_ids,
            num_steps=args.num_steps,
            eos_token_id=tokenizer.eos_token_id,
            boundary_token_ids=boundary_token_ids,
            min_step_tokens=args.min_step_tokens,
        )
        if not all(metadata["teacher_eligible"]):
            print(f"skipped row {row_index}: at least one canonical step is ineligible", flush=True)
            continue
        visible_prefix = torch.empty(0, dtype=torch.long)
        step_maps = []
        step_mass = []
        step_entropy = []
        image_grid_thw = None
        compatibility_error = None
        for step in steps:
            current = step[:-1].detach().cpu()
            visible_prefix = torch.cat([visible_prefix, current])
            feature, prefix_start, prefix_end = make_prefix_feature(
                row,
                cot_prefix=visible_prefix,
                think_token_id=think_token_id,
                end_think_token_id=end_think_token_id,
                answer_token_id=answer_token_id,
            )
            batch = move_tensor_inputs(collator([feature]), device)
            compatibility_error = image_placeholder_compatibility_error(teacher, batch)
            if compatibility_error is not None:
                break
            query_start = prefix_start - current.numel()
            assert_current_cot_query_tokens(batch, current, query_start, prefix_end)
            maps, mass, entropy = collect_all_layer_head_maps(
                teacher,
                batch,
                query_start=query_start,
                query_end=prefix_end,
                query_pool=args.query_pool,
            )
            step_maps.append(maps)
            step_mass.append(mass)
            step_entropy.append(entropy)
            image_grid_thw = batch["image_grid_thw"][0].detach().cpu().tolist()
        if compatibility_error is not None:
            print(f"skipped row {row_index}: {compatibility_error}", flush=True)
            continue
        maps_array = np.stack(step_maps)
        mass_array = np.stack(step_mass)
        entropy_array = np.stack(step_entropy)
        mean_js, mean_cosine = pairwise_step_metrics(maps_array)
        aggregate_mass.append(mass_array.mean(axis=0))
        aggregate_entropy.append(entropy_array.mean(axis=0))
        aggregate_js.append(mean_js)
        aggregate_cosine.append(mean_cosine)
        row_file = f"row_{row_index:06d}.npz"
        np.savez_compressed(
            args.output.parent / row_file,
            maps=maps_array,
            image_mass=mass_array,
            normalized_entropy=entropy_array,
        )
        rows.append(
            {
                "row_index": row_index,
                "artifact": row_file,
                "image_grid_thw": image_grid_thw,
                "split_points": metadata["split_points"],
            }
        )
        print(f"audited {ordinal}/{len(indices)}: tokenized row {row_index}", flush=True)

    if not rows:
        raise RuntimeError("No requested rows produced a complete three-step audit.")
    mean_mass = np.mean(aggregate_mass, axis=0)
    mean_entropy = np.mean(aggregate_entropy, axis=0)
    mean_js = np.mean(aggregate_js, axis=0)
    mean_cosine = np.mean(aggregate_cosine, axis=0)
    # Rank aggregation is intentionally scale-free.  A useful retrieval head
    # should really attend to the image, be spatially selective, and change
    # when the current CoT step changes.  Keep the three raw factors in the
    # report so this exploratory score is never mistaken for ground truth.
    combined = (
        normalized_rank(mean_mass)
        + normalized_rank(1.0 - mean_entropy)
        + normalized_rank(mean_js)
    ) / 3.0
    ranked = []
    for flat_index in np.argsort(combined.reshape(-1))[::-1]:
        layer_index, head_index = np.unravel_index(flat_index, combined.shape)
        ranked.append(
            {
                "layer": int(layer_index),
                "head": int(head_index),
                "combined_rank_score": float(combined[layer_index, head_index]),
                "mean_image_mass": float(mean_mass[layer_index, head_index]),
                "mean_normalized_entropy": float(mean_entropy[layer_index, head_index]),
                "mean_pairwise_js": float(mean_js[layer_index, head_index]),
                "mean_pairwise_cosine": float(mean_cosine[layer_index, head_index]),
            }
        )
    summary_npz = args.output.with_suffix(".npz")
    np.savez_compressed(
        summary_npz,
        mean_image_mass=mean_mass,
        mean_normalized_entropy=mean_entropy,
        mean_pairwise_js=mean_js,
        mean_pairwise_cosine=mean_cosine,
        combined_rank_score=combined,
    )
    report = {
        "format": "colt_cot_attention_layer_head_audit_v1",
        "tokenized_path": str(args.tokenized_path),
        "tokenized_fingerprint": getattr(dataset, "_fingerprint", None),
        "teacher_model_path": str(args.teacher_model_path),
        "num_layers": int(mean_mass.shape[0]),
        "num_query_heads": int(mean_mass.shape[1]),
        "query_pool": args.query_pool,
        "ranking_warning": "Exploratory proxy only; validate selected heads with held-out grounding and intervention.",
        "rows": rows,
        "summary_artifact": summary_npz.name,
        "top_heads": ranked[: max(args.top_k, 32)],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "top_heads": ranked[: args.top_k]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
