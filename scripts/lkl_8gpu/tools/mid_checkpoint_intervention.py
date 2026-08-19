#!/usr/bin/env python3
"""Quick latent-usage intervention check on a training checkpoint.

Loads one checkpoint, runs a small ChartQA subset under none/zero/skip latent
interventions (greedy), and reports per-mode accuracy plus per-sample output
agreement. If zero/skip barely change outputs, the latent trajectory is a
bypass at this training stage.

Usage (colt env, single GPU):
  COLT_DECODER_MODEL_PATH=/home/dataset-local/lkl/models/Qwen3-0.6B \
  CUDA_VISIBLE_DEVICES=0 python scripts/lkl_8gpu/tools/mid_checkpoint_intervention.py \
      --checkpoint checkpoints/colt_paper_faithful_stochastic/checkpoint-300 \
      --tsv eval/LMUData/ChartQA_TEST.tsv \
      --n 100 --out /home/dataset-local/lkl/tmp/mid_check_300.json
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys

import torch


def load_chartqa(tsv_path: str, n: int) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


def image_to_temp(rows: list[dict], tmp_dir: str) -> list[str]:
    os.makedirs(tmp_dir, exist_ok=True)
    paths = []
    for i, row in enumerate(rows):
        path = os.path.join(tmp_dir, f"{i}.jpg")
        if not os.path.exists(path):
            raw = row["image"]
            if raw.startswith("/9j") or raw.startswith("iVBOR"):
                with open(path, "wb") as f:
                    f.write(base64.b64decode(raw))
            else:
                path = raw
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="/home/dataset-local/lkl/tmp/mid_check.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Mid-checkpoint intervention requires exactly one visible GPU.")
    device = torch.device("cuda:0")
    # Deterministic greedy decoding for a clean intervention comparison.
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
    model.eval()

    rows = load_chartqa(args.tsv, args.n)
    image_paths = image_to_temp(rows, os.path.join(os.path.dirname(args.out), "mid_check_images"))

    results: dict[str, list[dict]] = {}
    for mode in ("none", "zero", "skip"):
        os.environ["COLT_LATENT_INTERVENTION"] = mode
        preds = []
        for i, (row, img_path) in enumerate(zip(rows, image_paths)):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": row["question"]},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[1] :]
            out_text = processor.batch_decode(
                new_tokens.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            preds.append(
                {
                    "index": row["index"],
                    "answer": row["answer"],
                    "prediction": out_text,
                    "hit": str(row["answer"]).strip() in out_text,
                }
            )
        results[mode] = preds
        acc = sum(p["hit"] for p in preds) / len(preds)
        print(f"[{mode}] acc={acc * 100:.2f}%")

    base = results["none"]
    for mode in ("zero", "skip"):
        other = results[mode]
        agree = sum(
            a["prediction"] == b["prediction"] for a, b in zip(base, other)
        ) / len(base)
        print(f"[{mode} vs none] per-sample output agreement={agree * 100:.2f}%")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
