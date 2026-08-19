#!/usr/bin/env python3
"""Evaluate the CMPO visual-grounding score of a CoLT checkpoint.

For each sampled Visual-CoT row it computes:

    grounding_score = cos(z_H, z_V_roi) - cos(z_H, z_V_nonroi)

where ``z_H = mean(h_1..h_3)`` is the trajectory-level latent and ``z_V`` is
the pooled visual feature inside / outside the annotated ROI. It also computes
an image-shuffled score by matching each latent against another row's ROI; a
trained model should show a clear drop (own-image score > shuffled score).

Usage (colt env, single GPU):
  COLT_VISUAL_GROUNDING=1 \
  COLT_DECODER_MODEL_PATH=/home/dataset-local/lkl/models/Qwen3-0.6B \
  CUDA_VISIBLE_DEVICES=0 python scripts/lkl_8gpu/tools/evaluate_grounding_score.py \
      --checkpoint checkpoints/colt_paper_faithful_visual_cot_smoke \
      --n 50 --out /home/dataset-local/lkl/tmp/grounding_score.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F


def load_samples(n: int, seed: int) -> list[dict]:
    path = (
        "/home/dataset-local/lkl/datasets/LVR_Train_Dataset/"
        "visualcot_98k/lvr_gqa_merged_98k.json"
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    return rng.sample(data, min(n, len(data)))


def build_inputs(processor, sample: dict, image_root: str, device: torch.device) -> dict:
    from qwen_vl_utils import process_vision_info

    image_path = os.path.join(image_root, sample["image"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": sample["question"]},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    return inputs


def latent_hidden(model, inputs: dict) -> torch.Tensor:
    """Return the trajectory-level latent ``z_H`` of shape (batch, hidden)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model._forward_latent_reasoning(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
    hiddens = model._last_latent_hiddens
    if not hiddens:
        raise RuntimeError("no latent hiddens collected; check the checkpoint path")
    return torch.cat(hiddens, dim=1).mean(dim=1).float()


def roi_features(model, bbox: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers.models.qwen3_vl.modeling_colt_grounding import (
        pool_roi_and_non_roi_features,
    )

    visual_embeds = model.model._colt_visual_embeds
    grid = model.model._colt_image_grid_thw
    if visual_embeds is None or grid is None:
        raise RuntimeError("visual features were not cached; set COLT_VISUAL_GROUNDING=1")
    z_roi, z_nonroi = pool_roi_and_non_roi_features(visual_embeds, grid, [[list(bbox)]])
    return z_roi.float(), z_nonroi.float()


def score(z_H: torch.Tensor, z_roi: torch.Tensor, z_nonroi: torch.Tensor) -> float:
    z_H = F.normalize(z_H, dim=-1)
    z_roi = F.normalize(z_roi, dim=-1)
    z_nonroi = F.normalize(z_nonroi, dim=-1)
    return float((z_H * z_roi).sum(-1).item() - (z_H * z_nonroi).sum(-1).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-root", default="/home/dataset-local/lkl/datasets/LVR_Train_Dataset/images")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("evaluate_grounding_score requires exactly one visible GPU.")
    device = torch.device("cuda:0")
    os.environ["COLT_LATENT_INTERVENTION"] = "none"
    os.environ["COLT_RESPECT_GENERATION_ARGS"] = "1"

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=True
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
    model.eval()

    samples = load_samples(args.n, args.seed)
    own_scores = []
    shuffled_scores = []
    results = []
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            inputs = build_inputs(processor, sample, args.image_root, device)
            z_H = latent_hidden(model, inputs)
            bbox = sample["lvr_bbox"][0]
            z_roi, z_nonroi = roi_features(model, bbox)
            own = score(z_H, z_roi, z_nonroi)

            other = samples[(i + 1) % len(samples)]
            other_inputs = build_inputs(processor, other, args.image_root, device)
            other_z_H = latent_hidden(model, other_inputs)
            other_bbox = other["lvr_bbox"][0]
            other_roi, other_nonroi = roi_features(model, other_bbox)
            shuffled = score(other_z_H, z_roi, z_nonroi)

            own_scores.append(own)
            shuffled_scores.append(shuffled)
            results.append(
                {
                    "question": sample["question"][:120],
                    "bbox": bbox,
                    "own_score": own,
                    "shuffled_score": shuffled,
                }
            )

    mean_own = sum(own_scores) / len(own_scores)
    mean_shuffled = sum(shuffled_scores) / len(shuffled_scores)
    print(f"n={len(own_scores)}")
    print(f"own-image grounding score : {mean_own:+.4f}")
    print(f"shuffled-image score      : {mean_shuffled:+.4f}")
    print(f"drop (own - shuffled)     : {mean_own - mean_shuffled:+.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": len(own_scores),
                "mean_own": mean_own,
                "mean_shuffled": mean_shuffled,
                "drop": mean_own - mean_shuffled,
                "per_sample": results,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
