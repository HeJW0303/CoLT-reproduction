#!/usr/bin/env python3
"""Causally validate frozen CoT-to-image attention maps with visual occlusion.

For each requested tokenized row and canonical CoT step, this tool obtains the
same frozen Qwen3-VL teacher map used by the target builder, masks its highest
scoring visual cells *before* the vision encoder, and measures the increase in
teacher-forced NLL of the current CoT span.  It compares that effect against
equally sized bottom-map and random masks.  A map passes only when masking its
top cells damages the current CoT more than both controls by the configured
margin.

This is deliberately a validation gate, not a new training loss or a
head-selection heuristic.  It answers the narrow question needed before a
costly training run: do the displayed teacher hotspots carry causal signal for
the visible-CoT tokens they are proposed to supervise?
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "LLaMA-Factory" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq  # noqa: E402
from llamafactory.data.template import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams.data_args import DataArguments  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    extract_think_content_robust,
    split_cot_by_dynamic_boundaries_with_metadata,
)

from build_cot_attention_targets import (  # noqa: E402
    assert_current_cot_query_tokens,
    get_teacher_attention_module,
    image_placeholder_compatibility_error,
    make_prefix_feature,
    move_tensor_inputs,
    parse_teacher_heads,
    teacher_attention_metadata,
    teacher_map_for_prefix,
    teacher_map_for_prefix_heads,
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
    parser.add_argument("--teacher-heads", default=None)
    parser.add_argument("--query-pool", choices=("mean", "last", "visual-mass"), default="mean")
    parser.add_argument("--indices", required=True, help="Comma-separated unique tokenized row indices.")
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--min-step-tokens", type=int, default=8)
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--image-min-pixels", type=int, default=1024)
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=0.10,
        help="Fraction of merged visual cells masked for top/bottom/random controls.",
    )
    parser.add_argument("--num-random", type=int, default=1, help="Independent random controls per step.")
    parser.add_argument(
        "--min-nll-increase",
        type=float,
        default=0.01,
        help="Required absolute top-mask NLL increase for a passed step.",
    )
    parser.add_argument(
        "--min-control-margin",
        type=float,
        default=0.005,
        help="Required top-mask NLL increase above the stronger bottom/random control.",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help=(
            "Optional aggregate held-out gate. The report is always saved; the command then fails "
            "if the tested-step pass rate is below this value."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args()


def select_train_split(dataset: Dataset | DatasetDict) -> Dataset:
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("Tokenized dataset must contain a train split.")
        return dataset["train"]
    return dataset


def parse_indices(raw: str, total: int) -> list[int]:
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not indices:
        raise ValueError("--indices did not contain any integers.")
    if len(indices) != len(set(indices)) or any(index < 0 or index >= total for index in indices):
        raise ValueError(f"--indices must be unique values in [0, {total - 1}].")
    return indices


def canonical_boundary_token_ids(tokenizer: Any) -> set[int]:
    return {
        token_id
        for text in ("\n", "\n\n", ".", "。", "?", "？", "!", "！", ";", "；", ":", "：", ",", "，")
        for token_id in tokenizer.encode(text, add_special_tokens=False)
    }


def masked_visual_cell_count(num_cells: int, fraction: float) -> int:
    """Return a non-overlapping mask size for top and bottom controls."""
    if num_cells < 2:
        raise ValueError("Occlusion validation needs at least two visual cells.")
    if not 0.0 < fraction <= 0.5:
        raise ValueError("--mask-fraction must lie in (0, 0.5].")
    return max(1, min(num_cells // 2, int(math.ceil(num_cells * fraction))))


def choose_occlusion_cells(
    attention: list[float], *, fraction: float, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return equal-size top, bottom, and random visual-cell index sets."""
    values = torch.as_tensor(attention, dtype=torch.float32)
    if values.ndim != 1 or not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError("Teacher map must be a finite non-negative one-dimensional distribution.")
    count = masked_visual_cell_count(values.numel(), fraction)
    order = torch.argsort(values, descending=True, stable=True)
    top = order[:count]
    bottom = order[-count:]
    random_cells = torch.randperm(values.numel(), generator=generator)[:count]
    return top, bottom, random_cells


def merged_cells_to_pixel_rows(
    cells: torch.Tensor, *, raw_pixel_rows: int, merge_size: int
) -> torch.Tensor:
    """Map Qwen merged visual-token indices to its contiguous pre-merge rows.

    Qwen3-VL permutes input patch rows into spatial-merge blocks before its
    vision merger; one language-model image placeholder then consumes one
    contiguous ``merge_size ** 2`` block.  This operates in that native order,
    avoiding an error-prone conversion through rendered-image coordinates.
    """
    if merge_size <= 0:
        raise ValueError("merge_size must be positive.")
    rows_per_cell = merge_size * merge_size
    expected_cells = raw_pixel_rows // rows_per_cell
    if raw_pixel_rows % rows_per_cell or cells.numel() == 0:
        raise ValueError("Pixel rows and merged visual cells are incompatible.")
    if torch.any(cells < 0) or torch.any(cells >= expected_cells):
        raise ValueError(f"Visual-cell index is outside [0, {expected_cells - 1}].")
    offsets = torch.arange(rows_per_cell, dtype=torch.long, device=cells.device)
    return (cells[:, None].long() * rows_per_cell + offsets[None, :]).reshape(-1)


def mask_pixel_rows_with_mean(
    pixel_values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Replace selected raw vision-patch rows with the image-wide mean row."""
    if pixel_values.ndim != 2:
        raise ValueError(f"Expected flattened Qwen pixel_values [patches, features], got {tuple(pixel_values.shape)}.")
    if rows.numel() == 0 or torch.any(rows < 0) or torch.any(rows >= pixel_values.shape[0]):
        raise ValueError("Requested pixel rows are invalid.")
    masked = pixel_values.clone()
    masked[rows] = pixel_values.float().mean(dim=0).to(dtype=pixel_values.dtype)
    return masked


@torch.inference_mode()
def current_span_nll(
    teacher: Any,
    model_inputs: dict[str, torch.Tensor],
    *,
    query_start: int,
    query_end: int,
    pixel_values: torch.Tensor | None = None,
) -> float:
    """Mean teacher-forced NLL for exactly the canonical visible-CoT span."""
    if query_start <= 0 or query_end <= query_start:
        raise ValueError(f"Invalid current-CoT range [{query_start}, {query_end}).")
    outputs = teacher(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        position_ids=model_inputs["position_ids"],
        pixel_values=model_inputs.get("pixel_values") if pixel_values is None else pixel_values,
        pixel_values_videos=model_inputs.get("pixel_values_videos"),
        image_grid_thw=model_inputs.get("image_grid_thw"),
        video_grid_thw=model_inputs.get("video_grid_thw"),
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[0, query_start - 1 : query_end - 1].float()
    labels = model_inputs["input_ids"][0, query_start:query_end]
    if logits.shape[0] != labels.numel():
        raise RuntimeError("Teacher logits no longer align with the canonical current CoT span.")
    return float(F.cross_entropy(logits, labels, reduction="mean").item())


def pass_causal_gate(
    *, top_nll_delta: float, bottom_nll_delta: float, random_nll_deltas: list[float], min_nll_increase: float,
    min_control_margin: float,
) -> tuple[bool, float]:
    """Return pass/fail and margin above the strongest matched-size control."""
    control = max([bottom_nll_delta, *random_nll_deltas])
    margin = top_nll_delta - control
    passed = top_nll_delta >= min_nll_increase and margin >= min_control_margin
    return passed, margin


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite causal-occlusion report: {args.output}")
    if args.num_steps != 3:
        raise ValueError("The current CoLT validation contract expects exactly three steps.")
    if args.num_random <= 0:
        raise ValueError("--num-random must be positive.")
    if args.min_nll_increase < 0 or args.min_control_margin < 0:
        raise ValueError("Causal-gate thresholds must be non-negative.")
    if args.min_pass_rate is not None and not 0.0 <= args.min_pass_rate <= 1.0:
        raise ValueError("--min-pass-rate must lie in [0, 1].")
    if not 0 < args.image_min_pixels <= args.image_max_pixels:
        raise ValueError("image pixel limits must satisfy 0 < min <= max.")

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
    explicit_heads = args.teacher_heads is not None
    teacher.eval()
    attention_module = None if explicit_heads else get_teacher_attention_module(teacher, args.teacher_layer)
    layer_heads = (
        None
        if not explicit_heads and args.query_pool == "mean"
        else parse_teacher_heads(args.teacher_heads, teacher, args.teacher_layer)
    )
    collator = MultiModalDataCollatorForSeq2Seq(
        template=template, model=teacher, tokenizer=tokenizer, processor=processor, label_pad_token_id=-100
    )
    think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
    end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    answer_token_id = tokenizer.encode("answer", add_special_tokens=False)[0]
    boundary_token_ids = canonical_boundary_token_ids(tokenizer)
    merge_size = int(teacher.config.vision_config.spatial_merge_size)
    rows: list[dict[str, Any]] = []

    for ordinal, row_index in enumerate(indices, start=1):
        row = dataset[row_index]
        source_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        _, cot_ids, _ = extract_think_content_robust(
            source_ids, think_token_id, end_think_token_id, answer_token_id
        )
        steps, _, split_metadata = split_cot_by_dynamic_boundaries_with_metadata(
            cot_ids,
            num_steps=args.num_steps,
            eos_token_id=tokenizer.eos_token_id,
            boundary_token_ids=boundary_token_ids,
            min_step_tokens=args.min_step_tokens,
        )
        visible_prefix = torch.empty(0, dtype=torch.long)
        record_steps: list[dict[str, Any]] = []
        for step_index, step in enumerate(steps):
            current = step[:-1].detach().cpu()
            visible_prefix = torch.cat([visible_prefix, current])
            result: dict[str, Any] = {
                "step_index": step_index,
                "teacher_eligible": bool(split_metadata["teacher_eligible"][step_index]),
                "passed": False,
            }
            if not result["teacher_eligible"] or current.numel() == 0:
                result["abstain_reason"] = "ineligible canonical step"
                record_steps.append(result)
                continue
            feature, prefix_start, prefix_end = make_prefix_feature(
                row,
                cot_prefix=visible_prefix,
                think_token_id=think_token_id,
                end_think_token_id=end_think_token_id,
                answer_token_id=answer_token_id,
            )
            query_start = prefix_start - current.numel()
            batch = move_tensor_inputs(collator([feature]), device)
            compatibility_error = image_placeholder_compatibility_error(teacher, batch)
            if compatibility_error is not None:
                result["abstain_reason"] = compatibility_error
                record_steps.append(result)
                continue
            assert_current_cot_query_tokens(batch, current, query_start, prefix_end)
            attention = (
                teacher_map_for_prefix(
                    teacher, attention_module, batch, query_start=query_start, query_end=prefix_end
                )
                if layer_heads is None
                else teacher_map_for_prefix_heads(
                    teacher, layer_heads, batch, query_start=query_start, query_end=prefix_end,
                    query_pool=args.query_pool,
                )
            )
            if not attention:
                result["abstain_reason"] = "teacher produced no image attention map"
                record_steps.append(result)
                continue
            pixel_values = batch.get("pixel_values")
            if pixel_values is None:
                result["abstain_reason"] = "missing pixel_values"
                record_steps.append(result)
                continue
            expected_cells = pixel_values.shape[0] // (merge_size * merge_size)
            if pixel_values.shape[0] % (merge_size * merge_size) or len(attention) != expected_cells:
                result["abstain_reason"] = (
                    "attention/image patch mismatch "
                    f"(map={len(attention)}, merged={expected_cells}, raw={pixel_values.shape[0]})"
                )
                record_steps.append(result)
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(args.seed + row_index * 31 + step_index)
            top_cells, bottom_cells, random_cells = choose_occlusion_cells(
                attention, fraction=args.mask_fraction, generator=generator
            )
            top_rows = merged_cells_to_pixel_rows(top_cells, raw_pixel_rows=pixel_values.shape[0], merge_size=merge_size)
            bottom_rows = merged_cells_to_pixel_rows(
                bottom_cells, raw_pixel_rows=pixel_values.shape[0], merge_size=merge_size
            )
            random_rows = merged_cells_to_pixel_rows(
                random_cells, raw_pixel_rows=pixel_values.shape[0], merge_size=merge_size
            )
            baseline_nll = current_span_nll(teacher, batch, query_start=query_start, query_end=prefix_end)
            top_nll = current_span_nll(
                teacher, batch, query_start=query_start, query_end=prefix_end,
                pixel_values=mask_pixel_rows_with_mean(pixel_values, top_rows),
            )
            bottom_nll = current_span_nll(
                teacher, batch, query_start=query_start, query_end=prefix_end,
                pixel_values=mask_pixel_rows_with_mean(pixel_values, bottom_rows),
            )
            random_deltas = []
            for random_index in range(args.num_random):
                if random_index:
                    random_cells = torch.randperm(len(attention), generator=generator)[: top_cells.numel()]
                    random_rows = merged_cells_to_pixel_rows(
                        random_cells, raw_pixel_rows=pixel_values.shape[0], merge_size=merge_size
                    )
                random_nll = current_span_nll(
                    teacher, batch, query_start=query_start, query_end=prefix_end,
                    pixel_values=mask_pixel_rows_with_mean(pixel_values, random_rows),
                )
                random_deltas.append(random_nll - baseline_nll)
            top_delta = top_nll - baseline_nll
            bottom_delta = bottom_nll - baseline_nll
            passed, control_margin = pass_causal_gate(
                top_nll_delta=top_delta,
                bottom_nll_delta=bottom_delta,
                random_nll_deltas=random_deltas,
                min_nll_increase=args.min_nll_increase,
                min_control_margin=args.min_control_margin,
            )
            result.update(
                {
                    "passed": passed,
                    "attention": attention,
                    "image_grid_thw": batch["image_grid_thw"].detach().cpu().tolist(),
                    "visual_cells": len(attention),
                    "masked_cells": int(top_cells.numel()),
                    "baseline_nll": baseline_nll,
                    "top_nll_delta": top_delta,
                    "bottom_nll_delta": bottom_delta,
                    "random_nll_deltas": random_deltas,
                    "control_margin": control_margin,
                    "top_cells": top_cells.tolist(),
                    "bottom_cells": bottom_cells.tolist(),
                }
            )
            record_steps.append(result)
        rows.append(
            {
                "row_index": row_index,
                "split_points": split_metadata["split_points"],
                "steps": record_steps,
            }
        )
        print(f"validated {ordinal}/{len(indices)}: tokenized row {row_index}", flush=True)

    flat_steps = [step for row in rows for step in row["steps"]]
    tested_steps = [step for step in flat_steps if "baseline_nll" in step]
    report = {
        "format": "colt_cot_attention_causal_occlusion_v1",
        "validation_description": (
            "Top teacher-attention visual cells are mean-masked before Qwen3-VL's vision encoder; "
            "a step passes when its current-CoT NLL rises more than equal-size bottom and random masks."
        ),
        "tokenized_path": str(args.tokenized_path),
        "tokenized_fingerprint": getattr(dataset, "_fingerprint", None),
        "source_rows": len(dataset),
        "teacher_model_path": str(args.teacher_model_path),
        **teacher_attention_metadata(
            args.teacher_layer,
            layer_heads,
            explicit_heads=explicit_heads,
        ),
        "query_pool": args.query_pool,
        "num_steps": args.num_steps,
        "min_step_tokens": args.min_step_tokens,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "mask_fraction": args.mask_fraction,
        "num_random": args.num_random,
        "mask_value": "image-wide mean raw vision patch",
        "min_nll_increase": args.min_nll_increase,
        "min_control_margin": args.min_control_margin,
        "summary": {
            "requested_rows": len(indices),
            "tested_steps": len(tested_steps),
            "passed_steps": sum(bool(step["passed"]) for step in tested_steps),
            "passed_rate": (
                sum(bool(step["passed"]) for step in tested_steps) / len(tested_steps) if tested_steps else 0.0
            ),
            "mean_top_nll_delta": (
                sum(step["top_nll_delta"] for step in tested_steps) / len(tested_steps) if tested_steps else None
            ),
            "mean_control_margin": (
                sum(step["control_margin"] for step in tested_steps) / len(tested_steps) if tested_steps else None
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {args.output}", flush=True)
    if args.min_pass_rate is not None and report["summary"]["passed_rate"] < args.min_pass_rate:
        raise RuntimeError(
            "Causal-occlusion held-out gate failed: "
            f"passed_rate={report['summary']['passed_rate']:.3f} < {args.min_pass_rate:.3f}."
        )


if __name__ == "__main__":
    main()
