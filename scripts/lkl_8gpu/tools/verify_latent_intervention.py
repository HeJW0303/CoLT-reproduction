#!/usr/bin/env python3
"""Smoke-verify COLT_LATENT_INTERVENTION modes on a single GPU.

Loads the fixed-v2 checkpoint once, then runs one image+text sample through
each intervention mode (none/skip/zero/random) and prints the resolved latent
step count plus the first generated tokens, so the modes can be compared.

Usage:
  source /home/dataset-local/lkl/colt-local.env
  conda activate colt
  CUDA_VISIBLE_DEVICES=0 python scripts/lkl_8gpu/tools/verify_latent_intervention.py \
      --model checkpoints/colt_paper_faithful_v2 \
      --image eval/LMUData/images/ChartQA_TEST/60.jpg \
      --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("This verification script must see exactly one CUDA GPU.")
    device = torch.device("cuda:0")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": "What is the title of this chart?"},
            ],
        }
    ]
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    for mode in ("none", "skip", "zero", "random"):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        os.environ["COLT_LATENT_INTERVENTION"] = mode
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_k=None,
        )
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        text_out = processor.batch_decode(
            new_tokens.unsqueeze(0),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        used_steps = getattr(model, "last_oracle_k_used", None)
        print(f"[{mode}] transition_steps={used_steps} n_tokens={new_tokens.shape[0]}")
        print(f"[{mode}] text={text_out[:200]!r}")


if __name__ == "__main__":
    main()
