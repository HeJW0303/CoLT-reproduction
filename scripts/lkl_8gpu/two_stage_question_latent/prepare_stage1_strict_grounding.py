#!/usr/bin/env python3
"""Materialise the strict one-image/one-box Stage-1 curriculum dataset.

The original 122K CoLT JSON retains the native grounding answers but not a
separate normalized-bbox column.  This builder recovers exactly one terminal
``<answer>{"boxes": [x1,y1,x2,y2]}</answer>`` box per row, validates it
against the local image, and writes the normalized xyxy metadata required by
the Stage-1 clean-question bottleneck.  It never edits the source 122K file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_DATASET_DIR = Path("/data/nvme0/lkl/datasets/CoLT_Train_Dataset")
DEFAULT_SOURCE = "colt_sft_image.json"
DEFAULT_OUTPUT = "colt_sft_image_grounding_bbox_normalized_strict.json"
DEFAULT_DATASET_NAME = "onethinker_sft_image_grounding_bbox_normalized_strict"
ANSWER_RE = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", flags=re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--source-file", default=DEFAULT_SOURCE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terminal_answer(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    assistant_messages = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant" and isinstance(message.get("content"), str)
    ]
    return assistant_messages[-1] if assistant_messages else None


def parse_single_xyxy(answer: str) -> list[float] | None:
    matches = ANSWER_RE.findall(answer)
    if not matches:
        return None
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError:
        return None
    box = payload.get("boxes") if isinstance(payload, dict) else None
    # Stage 1 intentionally excludes multiple boxes. Some datasets use
    # [[x1,y1,x2,y2]], while the CoLT grounding replies use [x1,y1,x2,y2].
    if isinstance(box, list) and len(box) == 1 and isinstance(box[0], list):
        box = box[0]
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def normalized_box(values: list[float], image_path: Path) -> list[float] | None:
    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = values
    # Native CoLT grounding replies use Qwen's 0--1000 coordinate convention,
    # not the JPEG's raw pixel grid. A fully [0,1] record is accepted as
    # already-normalized metadata; pixel coordinates are a compatibility
    # fallback for any exceptional source row.
    if min(values) >= 0.0 and max(values) <= 1.0:
        normalized = values
    elif min(values) >= 0.0 and max(values) <= 1000.0:
        normalized = [value / 1000.0 for value in values]
    else:
        normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
    nx1, ny1, nx2, ny2 = normalized
    if not (0.0 <= nx1 < nx2 <= 1.0 and 0.0 <= ny1 < ny2 <= 1.0):
        return None
    return [round(float(value), 8) for value in normalized]


def atomic_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    source_path = dataset_dir / args.source_file
    output_path = dataset_dir / args.output_file
    registry_path = dataset_dir / "dataset_info.json"
    if not source_path.is_file() or not registry_path.is_file():
        raise SystemExit("Expected source JSON and dataset_info.json under --dataset-dir.")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {output_path}; pass --overwrite after auditing it.")

    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise SystemExit("Source dataset must be a JSON list.")
    skipped = Counter()
    strict_rows: list[dict[str, Any]] = []
    for source_index, row in enumerate(source_rows):
        images = row.get("images") if isinstance(row, dict) else None
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            skipped["not_exactly_one_image"] += 1
            continue
        answer = terminal_answer(row)
        if answer is None:
            skipped["missing_terminal_assistant"] += 1
            continue
        box = parse_single_xyxy(answer)
        if box is None:
            skipped["not_one_finite_answer_box"] += 1
            continue
        image_path = dataset_dir / images[0].lstrip("./")
        if not image_path.is_file():
            skipped["missing_image"] += 1
            continue
        try:
            normalized = normalized_box(box, image_path)
        except (OSError, ValueError):
            skipped["unreadable_image"] += 1
            continue
        if normalized is None:
            skipped["invalid_or_out_of_range_xyxy"] += 1
            continue
        strict_rows.append(
            {
                "images": images,
                "messages": row["messages"],
                "bboxes": [normalized],
                "causal_grounded": True,
                "source_index": source_index,
            }
        )

    if not strict_rows:
        raise SystemExit("No strict Stage-1 rows survived validation.")
    atomic_json_dump(output_path, strict_rows)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry[args.dataset_name] = {
        "file_name": args.output_file,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
            "bboxes": "bboxes",
            "causal_grounded": "causal_grounded",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    atomic_json_dump(registry_path, registry)
    report = {
        "format": "colt_stage1_strict_grounding_v1",
        "source_file": args.source_file,
        "source_sha256": sha256(source_path),
        "source_rows": len(source_rows),
        "strict_rows": len(strict_rows),
        "output_file": args.output_file,
        "dataset_name": args.dataset_name,
        "skipped": dict(sorted(skipped.items())),
    }
    atomic_json_dump(output_path.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
