#!/usr/bin/env python3
"""GPU forward check of the image-mask training path on a checkpoint.

Loads a checkpoint in train() mode and runs one forward with
COLT_IMAGE_MASK_PROB forced to 1.0, verifying that masking the image span in
the answer attention mask does not crash and produces a finite loss.

Usage (colt env, single GPU):
  COLT_DECODER_MODEL_PATH=/home/dataset-local/lkl/models/Qwen3-0.6B \
  COLT_IMAGE_MASK_PROB=1.0 \
  CUDA_VISIBLE_DEVICES=0 python scripts/lkl_8gpu/tools/verify_image_mask.py \
      --checkpoint checkpoints/colt_paper_faithful_stochastic/checkpoint-300
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default="/home/dataset-local/lkl/CoLT-reproduction/eval/LMUData/images/ChartQA_TEST/1.jpg")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("verify_image_mask requires exactly one visible GPU.")
    device = torch.device("cuda:0")
    os.environ["COLT_RESPECT_GENERATION_ARGS"] = "1"
    os.environ["COLT_LATENT_TEMPERATURE"] = "0"

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

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
    model.train()
    print("image_mask_prob:", model.image_mask_prob)

    # Load one real training sample so the CoLT forward sees the exact
    # question/<think>cot</think>/answer structure it expects in train mode.
    import json

    sft_path = "/home/dataset-local/lkl/datasets/CoLT_Train_Dataset/colt_sft_image_nogrounding.json"
    with open(sft_path, encoding="utf-8") as f:
        sft_data = json.load(f)
    sample = next(
        it
        for it in sft_data
        if len(it["messages"]) >= 2 and "<think>" in it["messages"][1]["content"]
    )
    user_content = sample["messages"][0]["content"]
    image_path = os.path.join(
        "/home/dataset-local/lkl/datasets/CoLT_Train_Dataset", sample["images"][0].lstrip("./")
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_content.split("<image>", 1)[1]},
            ],
        },
        {"role": "assistant", "content": sample["messages"][1]["content"]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
    loss = outputs.loss if hasattr(outputs, "loss") else None
    print("forward OK; loss:", loss.item() if loss is not None else "n/a")
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("image-mask forward produced non-finite/None loss")
    print("IMAGE-MASK GPU FORWARD VERIFIED")


if __name__ == "__main__":
    main()
