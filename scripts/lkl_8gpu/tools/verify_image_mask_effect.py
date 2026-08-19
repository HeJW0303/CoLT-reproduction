#!/usr/bin/env python3
"""Verify that COLT_IMAGE_MASK_PROB actually changes the answer loss.

Loads a checkpoint in train() mode and runs the same forward twice: once with
image masking disabled and once forced to 1.0. If masking works, the masked
loss should be clearly higher (answer cannot see the image and must rely on
the latent trajectory).

Usage (colt env, single GPU):
  COLT_DECODER_MODEL_PATH=/home/dataset-local/lkl/models/Qwen3-0.6B \
  CUDA_VISIBLE_DEVICES=0 python scripts/lkl_8gpu/tools/verify_image_mask_effect.py \
      --checkpoint checkpoints/colt_paper_faithful_stochastic/checkpoint-600
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("verify_image_mask_effect requires exactly one visible GPU.")
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

    sft_path = "/home/dataset-local/lkl/datasets/CoLT_Train_Dataset/colt_sft_image_nogrounding.json"
    with open(sft_path, encoding="utf-8") as f:
        sft_data = json.load(f)
    samples = [
        it
        for it in sft_data
        if len(it["messages"]) >= 2 and "<think>" in it["messages"][1]["content"]
    ][:2]
    prepared = []
    for sample in samples:
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
        image_inputs, _ = process_vision_info(messages, image_patch_size=16)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(device)
        prepared.append(inputs)

    def run_forward(mask_prob: float) -> float:
        model.image_mask_prob = mask_prob
        losses = []
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for inputs in prepared:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                )
                losses.append(outputs.loss.item())
        return sum(losses) / len(losses)

    loss_off = run_forward(0.0)
    loss_on = run_forward(1.0)
    print(f"loss(mask=0)={loss_off:.4f}  loss(mask=1)={loss_on:.4f}  delta={loss_on - loss_off:+.4f}")
    if loss_on - loss_off < 0.1:
        print("WARNING: image masking has little/no effect on loss - masking may not be applied!")
    else:
        print("OK: image masking clearly affects the answer loss.")


if __name__ == "__main__":
    main()
