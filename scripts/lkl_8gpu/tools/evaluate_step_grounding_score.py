#!/usr/bin/env python3
"""Evaluate per-step visual grounding of a CoLT checkpoint on GQA step data.

The new CMPO SFT stage trains per-step contrastive grounding: each latent
``h_k`` is matched against its own step evidence pool ``V_k`` (the GQA
functional-program ROI for that reasoning step) with same-image non-evidence
and cross-image negatives.  This script mirrors that objective at eval time:

    hard_neg_score_k = cos(h_k, V_k_evidence) - cos(h_k, V_k_non_evidence)
    shuffled_score_k = cos(h_k_other, V_k_evidence) - cos(h_k_other, V_k_non_evidence)

``hard_neg_score_k`` is the same-image hard-negative margin (evidence must beat
non-evidence within the same image); ``shuffled_score_k`` measures how much of
the score is image-specific rather than a generic "ROI is salient" prior.

Usage (colt env, single GPU):
  COLT_VISUAL_GROUNDING=1 \
  COLT_DECODER_MODEL_PATH=/home/dataset-local/lkl/models/Qwen3-0.6B \
  CUDA_VISIBLE_DEVICES=0 python scripts/lkl_8gpu/tools/evaluate_step_grounding_score.py \
      --checkpoint checkpoints/colt_paper_faithful_replay_step_grounding_30k \
      --n 50 --out /home/dataset-local/lkl/tmp/step_grounding_score.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F


STEP_DATA = (
    "/home/dataset-local/lkl/datasets/CoLT_Train_Dataset/"
    "colt_sft_gqa_step_grounding_30k.json"
)
TEMPLATE_MARKER = "Please answer this question based on the visual content."


def extract_question(sample: dict) -> str:
    """Recover the plain question from the sharegpt content string.

    ``prepare_gqa_step_grounding.py`` emits ``<image>{question}\n{template}``,
    so splitting on the fixed template marker yields the original question.
    """
    content = sample["messages"][0]["content"]
    if "<image>" in content and TEMPLATE_MARKER in content:
        question = content.split("<image>", 1)[1].split(TEMPLATE_MARKER, 1)[0].strip()
        if question:
            return question
    raise ValueError(f"cannot extract question from message content: {content[:200]!r}")

def load_samples(n: int, seed: int) -> list[dict]:
    with open(STEP_DATA, encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    return rng.sample(data, min(n, len(data)))


def build_inputs(processor, sample: dict, image_root: str, device: torch.device) -> dict:
    from qwen_vl_utils import process_vision_info

    image_path = os.path.join(image_root, sample["images"][0])
    question = extract_question(sample)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    processed_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    inputs = processor(
        text=[processed_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    return inputs


def step_latent_hiddens(model, inputs: dict) -> list[torch.Tensor]:
    """Return per-step latent ``h_1..h_3`` as a list of (hidden,) tensors."""
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
    return [h[0].flatten().float() for h in hiddens]


def step_roi_features(model, step_bboxes: list[list[list[float]]]):
    from transformers.models.qwen3_vl.modeling_colt_grounding import (
        pool_step_roi_and_nonroi_features,
    )

    visual_embeds = model.model._colt_visual_embeds
    grid = model.model._colt_image_grid_thw
    if visual_embeds is None or grid is None:
        raise RuntimeError("visual features were not cached; set COLT_VISUAL_GROUNDING=1")
    z_pos, z_neg = pool_step_roi_and_nonroi_features(visual_embeds, grid, [step_bboxes])
    return z_pos[0].float(), z_neg[0].float()


def score(h: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor) -> float:
    h = F.normalize(h, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)
    z_neg = F.normalize(z_neg, dim=-1)
    return float((h * z_pos).sum(-1).item() - (h * z_neg).sum(-1).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-root",
        default="/home/dataset-local/lkl/datasets/LVR_Train_Dataset/images",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("evaluate_step_grounding_score requires exactly one visible GPU.")
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
    step_own = [0.0, 0.0, 0.0]
    step_shuffled = [0.0, 0.0, 0.0]
    results = []
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            step_bboxes = sample["step_bboxes"]
            inputs = build_inputs(processor, sample, args.image_root, device)
            hiddens = step_latent_hiddens(model, inputs)
            z_pos, z_neg = step_roi_features(model, step_bboxes)

            other = samples[(i + 1) % len(samples)]
            other_inputs = build_inputs(processor, other, args.image_root, device)
            other_hiddens = step_latent_hiddens(model, other_inputs)

            own_k = [score(hiddens[k], z_pos[k], z_neg[k]) for k in range(3)]
            shuffled_k = [
                score(other_hiddens[k], z_pos[k], z_neg[k]) for k in range(3)
            ]
            for k in range(3):
                step_own[k] += own_k[k]
                step_shuffled[k] += shuffled_k[k]
            results.append(
                {
                    "question": extract_question(sample)[:120],
                    "step_bboxes": step_bboxes,
                    "own_per_step": own_k,
                    "shuffled_per_step": shuffled_k,
                }
            )

    n = len(samples)
    mean_own = [v / n for v in step_own]
    mean_shuffled = [v / n for v in step_shuffled]
    mean_own_traj = sum(mean_own) / 3
    mean_shuffled_traj = sum(mean_shuffled) / 3
    print(f"n={n}")
    for k in range(3):
        print(
            f"step{k+1} own(hard-neg)={mean_own[k]:+.4f}  "
            f"shuffled={mean_shuffled[k]:+.4f}  drop={mean_own[k] - mean_shuffled[k]:+.4f}"
        )
    print(f"trajectory own(hard-neg) : {mean_own_traj:+.4f}")
    print(f"trajectory shuffled      : {mean_shuffled_traj:+.4f}")
    print(f"trajectory drop          : {mean_own_traj - mean_shuffled_traj:+.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": n,
                "mean_own_per_step": mean_own,
                "mean_shuffled_per_step": mean_shuffled,
                "mean_own_trajectory": mean_own_traj,
                "mean_shuffled_trajectory": mean_shuffled_traj,
                "drop_trajectory": mean_own_traj - mean_shuffled_traj,
                "per_sample": results,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
