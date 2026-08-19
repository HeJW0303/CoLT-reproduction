#!/usr/bin/env python3
"""Convert the merged Visual-CoT 98K dataset into CoLT sharegpt format.

The merged file already contains one row per sample with:
    (image, question, answer, lvr_bbox, thought/reasoning).
This script wraps each row into the CoLT conversation format
(``<think>...</think>`` + ``<answer>...</answer>``) and keeps the normalized
ROI bbox in a dedicated ``bboxes`` column that the LLaMA-Factory collator can
forward to ``colt_bboxes`` for the CMPO visual-grounding loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


ANSWER_TEMPLATE = (
    "Please answer this question based on the visual content."
    "Provide your thinking process between the <think> and </think> tags, "
    "and then give your final answer between the <answer> and </answer> tags."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
    "Please provide only your text answer within the <answer>...</answer> tags.\n"
    "Example:\n"
    "<answer>The capital of France is Paris.</answer>"
)


def convert_sample(sample: dict, image_root: str) -> dict | None:
    """Return a CoLT sharegpt sample, or None when the image is missing."""
    image_rel = sample.get("image")
    if not image_rel:
        return None
    image_path = os.path.join(image_root, image_rel)
    if not os.path.exists(image_path):
        return None

    question = (sample.get("question") or "").strip()
    thought = (sample.get("thought") or "").strip()
    answer = (sample.get("answer") or "").strip()
    if not question or not thought or not answer:
        return None

    lvr_bbox = sample.get("lvr_bbox")
    if not lvr_bbox or len(lvr_bbox) != 1:
        return None
    bbox = [float(v) for v in lvr_bbox[0]]
    if len(bbox) != 4:
        return None
    # Reject rows whose bbox is empty after clamping to [0, 1] (fully out-of-frame
    # or degenerate annotations). Such rows make the ROI pool empty and crash
    # the visual-grounding loss during training.
    x1, y1 = max(0.0, bbox[0]), max(0.0, bbox[1])
    x2, y2 = min(1.0, bbox[2]), min(1.0, bbox[3])
    if x1 >= x2 or y1 >= y2:
        return None

    user_content = f"<image>{question}\n{ANSWER_TEMPLATE}"
    assistant_content = f"<think>{thought}</think>\n<answer>{answer}</answer>"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [image_path],
        "bboxes": [bbox],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "/home/dataset-local/lkl/datasets/LVR_Train_Dataset/"
            "visualcot_98k/lvr_gqa_merged_98k.json"
        ),
    )
    parser.add_argument(
        "--image-root",
        default="/home/dataset-local/lkl/datasets/LVR_Train_Dataset/images",
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/dataset-local/lkl/datasets/CoLT_Train_Dataset/"
            "colt_sft_visual_cot_98k.json"
        ),
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"raw samples: {len(raw)}", flush=True)

    converted = []
    skipped = 0
    for sample in raw:
        item = convert_sample(sample, args.image_root)
        if item is None:
            skipped += 1
            continue
        converted.append(item)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False)
    print(f"converted: {len(converted)}", flush=True)
    print(f"skipped: {skipped}", flush=True)
    print(f"output: {args.output}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
