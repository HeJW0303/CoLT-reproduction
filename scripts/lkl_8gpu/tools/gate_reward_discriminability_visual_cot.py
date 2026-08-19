#!/usr/bin/env python3
"""Measure reward discriminability on full-label, masked Visual-CoT rows.

This is intentionally separate from the historical replay/GQA gate.  It tests
the current SFT-v2 contract: rows have answer labels, step bboxes and the
visual-CoT prompt bottleneck is forced on every measured example.  The answer
gradient is taken from the hook registered on the actual final ``latent_embd``
used by the answer decoder; it is not taken from a view of concatenated input
embeddings.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import re
import statistics

import torch
from datasets import load_from_disk
from PIL import Image


IMAGE_MAX_PIXELS = 802816
IMAGE_MIN_PIXELS = 32 * 32


def build_visual_inputs(processor, image_path: str, device: torch.device) -> dict:
    """Match LLaMA-Factory's training image resize before using cached ids."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if width * height > IMAGE_MAX_PIXELS:
        factor = math.sqrt(IMAGE_MAX_PIXELS / (width * height))
        image = image.resize((int(width * factor), int(height * factor)))
    elif width * height < IMAGE_MIN_PIXELS:
        factor = math.sqrt(IMAGE_MIN_PIXELS / (width * height))
        image = image.resize((int(width * factor), int(height * factor)))
    packed = processor(text=["<image>"], images=[image], padding=True, return_tensors="pt")
    return {
        key: packed.get(key)
        for key in (
            "pixel_values",
            "image_grid_thw",
            "video_pixel_values",
            "video_grid_thw",
        )
    }


def parse_component(stream: str, name: str) -> float:
    matches = re.findall(rf"{name} : ([-+0-9.eE]+)", stream)
    return float(matches[-1]) if matches else float("nan")


def parse_control(stream: str) -> tuple[int, int, int, int]:
    matches = re.findall(
        r"colt_control_rows : active=(\d+) visual_cot=(\d+) visual_only=(\d+) image_mask_hit=(\d+)",
        stream,
    )
    if not matches:
        raise RuntimeError("CoLT control-row log was not emitted; set COLT_COMPONENT_LOG_EVERY=1")
    return tuple(int(value) for value in matches[-1])


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0.0 or denom_y == 0.0:
        return float("nan")
    return numerator / (denom_x * denom_y)


def finite_median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenized-cache", required=True)
    parser.add_argument("--n", type=int, default=64, help="Global Visual-CoT sample count.")
    parser.add_argument("--k", type=int, default=6, help="Stochastic paths per sample and sigma.")
    parser.add_argument("--sigmas", default="0.0,0.1,0.2")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n <= 0 or args.k < 2:
        raise ValueError("--n must be positive and --k must be at least 2")
    if args.shard_id < 0 or args.n_shards < 1 or args.shard_id >= args.n_shards:
        raise ValueError("shard-id must be in [0, n-shards)")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This gate requires exactly one visible GPU per shard.")
    sigmas = [float(value) for value in args.sigmas.split(",") if value.strip()]
    if not sigmas or any(value < 0.0 for value in sigmas):
        raise ValueError("--sigmas must contain non-negative values")

    os.environ["COLT_STOCHASTIC_LATENT"] = "1"
    os.environ["COLT_VISUAL_GROUNDING"] = "1"
    os.environ["COLT_ANSWER_VISIBILITY"] = "full"
    os.environ["COLT_IMAGE_MASK_PROB"] = "1.0"
    os.environ["COLT_COMPONENT_LOG_EVERY"] = "1"
    os.environ.setdefault("COLT_RESPECT_GENERATION_ARGS", "1")
    os.environ.setdefault("COLT_LATENT_INTERVENTION", "none")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        trust_remote_code=True,
    )
    # ``out.loss`` is now exactly answer CE.  Raw grounding is still computed
    # and logged, allowing us to measure its path-to-path variation separately.
    model.forward_align_weight = 0.0
    model.backward_align_weight = 0.0
    model.prediction_weight = 0.0
    model.visual_grounding_weight = 0.0
    model.image_mask_prob = 1.0
    model.train()

    dataset = load_from_disk(args.tokenized_cache)["train"]
    rng = random.Random(args.seed)
    selected: list[int] = []
    for row_index in rng.sample(range(len(dataset)), len(dataset)):
        row = dataset[row_index]
        if (
            bool(row["visual_cot"])
            and not bool(row["visual_only"])
            and bool(row["step_bboxes"])
            and bool(row["images"])
            and any(token != -100 for token in row["labels"])
        ):
            selected.append(row_index)
            if len(selected) == args.n:
                break
    if len(selected) != args.n:
        raise RuntimeError(f"Only found {len(selected)} eligible rows, expected {args.n}.")
    local_indices = selected[args.shard_id :: args.n_shards]
    print(f"global rows={len(selected)} local rows={len(local_indices)} shard={args.shard_id}")

    vision_cache: dict[int, dict] = {}

    def run_forward(
        row_index: int, sigma: float, *, want_grad: bool
    ) -> tuple[float, float, float, torch.Tensor, tuple[int, int, int, int]]:
        row = dataset[row_index]
        model.latent_noise_std = sigma
        if row_index not in vision_cache:
            vision_cache[row_index] = build_visual_inputs(processor, row["images"][0], device)
        kwargs = {
            "input_ids": torch.tensor([row["input_ids"]], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([row["attention_mask"]], dtype=torch.long, device=device),
            "labels": torch.tensor([row["labels"]], dtype=torch.long, device=device),
            "colt_step_bboxes": [row["step_bboxes"]],
            "colt_visual_cot": torch.tensor([True], dtype=torch.bool, device=device),
            "colt_visual_only": torch.tensor([False], dtype=torch.bool, device=device),
        }
        for key, value in vision_cache[row_index].items():
            if value is not None:
                kwargs[key] = value.to(device)

        captured: dict[str, torch.Tensor] = {}

        def capture_answer_embeds(module, hook_args, hook_kwargs, output):
            embeds = hook_kwargs.get("inputs_embeds")
            if embeds is not None:
                captured["answer_embeds"] = embeds

        handle = model.model.register_forward_hook(capture_answer_embeds, with_kwargs=True)
        model.zero_grad(set_to_none=True)
        model.latent_answer_grad_norm = None
        buffer = io.StringIO()
        grad_context = torch.enable_grad() if want_grad else torch.no_grad()
        with contextlib.redirect_stdout(buffer), grad_context, torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**kwargs)
        handle.remove()
        if not torch.isfinite(output.loss):
            raise RuntimeError(f"non-finite answer CE for cache row {row_index}")
        if "answer_embeds" not in captured:
            raise RuntimeError("answer-decoder embedding hook was not reached")
        if want_grad:
            output.loss.backward()
            answer_grad = model.latent_answer_grad_norm
            if answer_grad is None:
                raise RuntimeError("final-latent answer-gradient hook was not reached")
        else:
            answer_grad = float("nan")
        stream = buffer.getvalue()
        control = parse_control(stream)
        if control != (1, 1, 0, 1):
            raise RuntimeError(f"Visual-CoT prompt isolation failed for row {row_index}: {control}")
        grounding = parse_component(stream, "grounding_loss_total")
        final_latent = captured["answer_embeds"][:, :1].detach().float().clone()
        return float(output.loss.detach()), grounding, float(answer_grad), final_latent, control

    gradients: list[float] = []
    answer_std = {f"{sigma:.2f}": [] for sigma in sigmas}
    grounding_std = {f"{sigma:.2f}": [] for sigma in sigmas}
    noise_reach = {f"{sigma:.2f}": [] for sigma in sigmas}
    coupling = {f"{sigma:.2f}": [] for sigma in sigmas}
    sample_records = []

    for local_position, row_index in enumerate(local_indices, start=1):
        torch.manual_seed(args.seed + row_index)
        _, _, grad, _, control = run_forward(row_index, 0.0, want_grad=True)
        gradients.append(grad)
        record = {
            "cache_index": row_index,
            "answer_grad_norm": grad,
            "control": {
                "active": control[0],
                "visual_cot": control[1],
                "visual_only": control[2],
                "image_mask_hit": control[3],
            },
            "by_sigma": {},
        }
        for sigma in sigmas:
            key = f"{sigma:.2f}"
            ces: list[float] = []
            groundings: list[float] = []
            final_latents: list[torch.Tensor] = []
            for repeat in range(args.k):
                torch.manual_seed(args.seed + row_index * 1000 + repeat)
                ce, grounding, _, final_latent, _ = run_forward(row_index, sigma, want_grad=False)
                ces.append(ce)
                groundings.append(grounding)
                final_latents.append(final_latent)
            ce_std = statistics.stdev(ces)
            grounding_path_std = statistics.stdev(groundings)
            base_norm = final_latents[0].norm().item()
            relative_move = (
                statistics.mean(
                    (latent - final_latents[0]).norm().item() / base_norm
                    for latent in final_latents[1:]
                )
                if base_norm > 0.0
                else float("nan")
            )
            correlation = pearson(ces, groundings)
            answer_std[key].append(ce_std)
            grounding_std[key].append(grounding_path_std)
            noise_reach[key].append(relative_move)
            coupling[key].append(correlation)
            record["by_sigma"][key] = {
                "answer_ce_std": ce_std,
                "grounding_loss_std": grounding_path_std,
                "latent_norm_relative_move": relative_move,
                "answer_grounding_pearson": correlation,
            }
        sample_records.append(record)
        print(
            f"[{local_position}/{len(local_indices)}] row={row_index} grad={grad:.3e} "
            f"ce_std@0.1={record['by_sigma'].get('0.10', {}).get('answer_ce_std', float('nan')):.3e} "
            f"ground_std@0.1={record['by_sigma'].get('0.10', {}).get('grounding_loss_std', float('nan')):.3e}",
            flush=True,
        )

    result = {
        "protocol": "visual_cot_full_label_forced_prompt_isolation_v1",
        "checkpoint": args.checkpoint,
        "tokenized_cache": args.tokenized_cache,
        "global_requested_n": args.n,
        "local_n": len(local_indices),
        "shard_id": args.shard_id,
        "n_shards": args.n_shards,
        "seed": args.seed,
        "k": args.k,
        "sigmas": sigmas,
        "answer_grad_norm_median": finite_median(gradients),
        "answer_grad_norm_mean": statistics.mean(gradients),
        "answer_grad_used_frac": sum(value > 0.0 for value in gradients) / len(gradients),
        "answer_ce_std_median_by_sigma": {key: finite_median(values) for key, values in answer_std.items()},
        "grounding_loss_std_median_by_sigma": {key: finite_median(values) for key, values in grounding_std.items()},
        "latent_norm_relative_move_median_by_sigma": {key: finite_median(values) for key, values in noise_reach.items()},
        "answer_grounding_pearson_median_by_sigma": {key: finite_median(values) for key, values in coupling.items()},
        "answer_grounding_pearson_positive_frac_by_sigma": {
            key: sum(value > 0.0 for value in values if math.isfinite(value))
            / max(sum(math.isfinite(value) for value in values), 1)
            for key, values in coupling.items()
        },
        "per_sample": sample_records,
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}, ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
