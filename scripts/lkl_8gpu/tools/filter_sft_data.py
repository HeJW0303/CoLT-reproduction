#!/usr/bin/env python3
"""Filter spatial-grounding / segmentation samples out of the CoLT SFT corpus.

Keeps MCQ / visual-perception / chart / math / OCR / spatial-reasoning tasks and
drops referring-expression + bounding-box + segmentation-hint tasks, which are
not part of the evaluation suite. Output uses the same sharegpt JSON format.

Usage:
  python scripts/lkl_8gpu/tools/filter_sft_data.py \
      /home/dataset-local/lkl/datasets/CoLT_Train_Dataset/colt_sft_image.json \
      /home/dataset-local/lkl/datasets/CoLT_Train_Dataset/colt_sft_image_nogrounding.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter


SEG_KEYWORDS = (
    "segmentation hints",
    "positive points",
    "negative points",
    "sam2",
    "provide one bounding box",
    "one bounding box",
    "referring expression",
    "grounding",
    "provide the bounding box",
    "bounding boxes",
)


def is_grounding_segmentation(content: str) -> bool:
    lowered = content.lower()
    return any(keyword in lowered for keyword in SEG_KEYWORDS)


def image_source(images) -> str:
    for img in images or []:
        for src in (
            "CLEVRER", "Chart", "General", "Holmes", "Knowledge",
            "Math", "OCR", "Spatial", "Visulogic",
        ):
            if f"/{src}/" in img:
                return src
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()

    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)

    dropped = 0
    kept = []
    kept_by_source: Counter[str] = Counter()
    for item in data:
        content = item["messages"][0]["content"]
        if is_grounding_segmentation(content):
            dropped += 1
            continue
        kept.append(item)
        kept_by_source[image_source(item.get("images", []))] += 1

    with open(args.dst, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)

    total = len(data)
    print(f"total={total} dropped={dropped} ({dropped / total * 100:.1f}%) "
          f"kept={len(kept)} ({len(kept) / total * 100:.1f}%)")
    print("kept by source:")
    for source, count in kept_by_source.most_common():
        print(f"  {source}: {count}")
    print(f"written: {args.dst}")


if __name__ == "__main__":
    main()
