#!/usr/bin/env python3
"""Build a strict three-step visual-CoT corpus for the current CoLT path.

The output contract is the ShareGPT-style JSON consumed by the local
LLaMA-Factory fork:

    {
      "messages": [...],
      "images": ["/absolute/path/to/rgb.jpg"],
      "step_bboxes": [[[x1, y1, x2, y2]], ...],
      "visual_cot": true
    }

Only rows with exactly three reasoning steps and exactly one valid normalized
xyxy bbox per step are emitted.  The SIF source also requires both its RGB and
depth files to exist, but the CoLT manifest intentionally passes only RGB: the
current grounding pool maps one image to one sample and has no per-sample image
index for a two-image SIF record.  The extracted depth files remain available
for a future depth-aware input contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ANSWER_INSTRUCTION = (
    "Please answer the question based on the image. Give exactly three "
    "visual reasoning steps inside <think>...</think>, then give the final "
    "answer inside <answer>...</answer>."
)

STEP_LINE_RE = re.compile(r"^\s*Step\s+(\d+)\s*:\s*(.*?)\s*$", re.IGNORECASE)
NUMBER = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
BOX_RE = re.compile(
    rf"\[\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*\]"
)
FINAL_ANSWER_RE = re.compile(r"^\s*Final Answer\s*:\s*(.*?)\s*$", re.IGNORECASE)
AREA_RE = re.compile(r"<area>\s*(.*?)\s*</area>", re.IGNORECASE | re.DOTALL)
TEXT_RE = re.compile(r"<text>\s*(.*?)\s*</text>", re.IGNORECASE | re.DOTALL)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _valid_xyxy(box: Iterable[Any]) -> list[float] | None:
    values = list(box)
    if len(values) != 4 or not all(_finite_number(value) for value in values):
        return None
    x1, y1, x2, y2 = (float(value) for value in values)
    if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        return None
    if not (x1 < x2 and y1 < y2):
        return None
    return [x1, y1, x2, y2]


def _row(question: str, thought_steps: list[str], answer: str, image: Path, step_bboxes: list[list[list[float]]]) -> dict[str, Any] | None:
    question = question.strip()
    answer = answer.strip()
    thought_steps = [step.strip() for step in thought_steps if step.strip()]
    if not question or not answer or len(thought_steps) != 3:
        return None
    if not image.is_file():
        return None
    if len(step_bboxes) != 3 or any(len(step) != 1 for step in step_bboxes):
        return None
    thought = "\n".join(f"Step {i}: {step}" for i, step in enumerate(thought_steps, 1))
    return {
        "messages": [
            {
                "role": "user",
                "content": f"<image>{question}\n{ANSWER_INSTRUCTION}",
            },
            {
                "role": "assistant",
                "content": (
                    f"<think>{thought}"
                    f"</think>\n<answer>{answer}</answer>"
                ),
            },
        ],
        "images": [str(image.resolve())],
        "step_bboxes": step_bboxes,
        "visual_cot": True,
    }


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_mm_gcot(path: Path, image_root: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = _load_json(path)
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for source in rows:
        if not str(source.get("id", "")).endswith("_cot"):
            skipped["not_cot_row"] += 1
            continue
        conversations = source.get("conversations") or []
        if len(conversations) < 2:
            skipped["missing_conversations"] += 1
            continue
        human = str(conversations[0].get("value", ""))
        assistant = str(conversations[1].get("value", ""))
        question = human.replace("<image>", "", 1)
        question = re.split(r"\n?Answer the question", question, maxsplit=1, flags=re.IGNORECASE)[0]

        parsed_steps: list[tuple[str, list[float]]] = []
        final_answer = ""
        for line in assistant.splitlines():
            step_match = STEP_LINE_RE.match(line)
            if step_match:
                step_number = step_match.group(1)
                boxes = BOX_RE.findall(line)
                if len(boxes) != 1:
                    parsed_steps.append((step_number, ""))
                    continue
                x, y, width, height = (float(value) for value in boxes[0])
                box = _valid_xyxy((x, y, x + width, y + height))
                if box is None or width <= 0.0 or height <= 0.0:
                    parsed_steps.append((step_number, ""))
                    continue
                description = step_match.group(2)
                description = description[: BOX_RE.search(description).start()].rstrip()
                description = re.sub(r"\s+at\s*$", "", description, flags=re.IGNORECASE).strip()
                parsed_steps.append((step_number, description))
                continue
            final_match = FINAL_ANSWER_RE.match(line)
            if final_match:
                final_answer = final_match.group(1).strip()

        if [step[0] for step in parsed_steps] != ["1", "2", "3"]:
            skipped["not_exactly_three_steps"] += 1
            continue
        if any(not step[1] for step in parsed_steps) or not final_answer:
            skipped["invalid_step_or_answer"] += 1
            continue

        # Re-parse the three lines to retain the validated box values.
        step_bboxes: list[list[list[float]]] = []
        for line in assistant.splitlines():
            step_match = STEP_LINE_RE.match(line)
            if not step_match:
                continue
            boxes = BOX_RE.findall(line)
            if len(boxes) != 1:
                continue
            x, y, width, height = (float(value) for value in boxes[0])
            box = _valid_xyxy((x, y, x + width, y + height))
            if box is not None:
                step_bboxes.append([box])
        if len(step_bboxes) != 3:
            skipped["invalid_bbox"] += 1
            continue

        image_name = Path(str(source.get("image", ""))).name
        image = image_root / image_name
        item = _row(
            question,
            [step[1] for step in parsed_steps],
            final_answer,
            image,
            step_bboxes,
        )
        if item is None:
            skipped["missing_image_or_text"] += 1
            continue
        selected.append(item)

    return selected, skipped


def _parse_visreason_gqa(annotation_path: Path, image_root: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = _load_json(annotation_path)
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for source in rows:
        rounds = [source.get(f"round{i}") for i in range(1, 4)]
        if not all(isinstance(round_data, dict) for round_data in rounds):
            skipped["not_exactly_three_rounds"] += 1
            continue
        width = source.get("width")
        height = source.get("height")
        if not (_finite_number(width) and _finite_number(height) and float(width) > 0 and float(height) > 0):
            skipped["invalid_image_size"] += 1
            continue

        step_bboxes: list[list[list[float]]] = []
        thought_steps: list[str] = []
        valid = True
        for index, round_data in enumerate(rounds, 1):
            answer_data = round_data.get(f"r{index}_answer") or {}
            description = str(answer_data.get("description", "")).strip()
            reasoning = str(answer_data.get("reasoning", "")).strip()
            bbox = round_data.get("bbox_xyxy")
            if not description or not reasoning or not isinstance(bbox, list) or len(bbox) != 4:
                valid = False
                break
            if not all(_finite_number(value) for value in bbox):
                valid = False
                break
            x1, y1, x2, y2 = (float(value) for value in bbox)
            normalized = _valid_xyxy((x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height)))
            if normalized is None:
                valid = False
                break
            step_bboxes.append([normalized])
            thought_steps.append(f"{description} {reasoning}")
        if not valid:
            skipped["invalid_round_or_bbox"] += 1
            continue

        image = image_root / str(source.get("image", ""))
        item = _row(str(source.get("question", "")), thought_steps, str(source.get("answer", "")), image, step_bboxes)
        if item is None:
            skipped["missing_image_or_text"] += 1
            continue
        selected.append(item)

    return selected, skipped


def _parse_sif(path: Path, image_root: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = _load_json(path)
    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for source in rows:
        metadata = source.get("normalized_solution_round") or {}
        if metadata.get("illegal_content") is not False:
            skipped["illegal_or_missing_quality_flag"] += 1
            continue
        assistant_messages = [message.get("content", "") for message in source.get("messages", []) if message.get("role") == "assistant"]
        if len(assistant_messages) != 1:
            skipped["missing_assistant_message"] += 1
            continue
        assistant = str(assistant_messages[0])
        area_blocks = AREA_RE.findall(assistant)
        text_blocks = [text.strip() for text in TEXT_RE.findall(assistant)]
        if len(area_blocks) != 3 or len(text_blocks) != 3:
            skipped["not_exactly_three_steps"] += 1
            continue

        step_bboxes: list[list[list[float]]] = []
        valid = True
        for block in area_blocks:
            try:
                entries = json.loads(block)
            except json.JSONDecodeError:
                valid = False
                break
            if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
                valid = False
                break
            bbox = _valid_xyxy(entries[0].get("bbox_2d", []))
            depth = entries[0].get("depth")
            if bbox is None or not _finite_number(depth) or not 0.0 <= float(depth) <= 1.0:
                valid = False
                break
            step_bboxes.append([bbox])
        if not valid:
            skipped["invalid_area_bbox_or_depth"] += 1
            continue

        image_rel = str(source.get("image", "")).replace("../data/", "", 1)
        depth_rel = str(source.get("depth_image", "")).replace("../data/", "", 1)
        image = image_root / image_rel
        depth_image = image_root / depth_rel
        if not image.is_file() or not depth_image.is_file():
            skipped["missing_rgb_or_depth"] += 1
            continue
        item = _row(str(source.get("problem", "")), text_blocks, str(source.get("response", "")), image, step_bboxes)
        if item is None:
            skipped["missing_image_or_text"] += 1
            continue
        selected.append(item)

    return selected, skipped


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/dataset-local/lkl/datasets/CoLT_Visual_CoT_3Step"),
    )
    parser.add_argument(
        "--mm-gcot",
        type=Path,
        default=Path("/home/dataset-local/lkl/datasets/MM-GCoT/Train/train_dataset.json"),
    )
    parser.add_argument(
        "--mm-image-root",
        type=Path,
        default=Path("/home/dataset-local/lkl/datasets/LVR_Train_Dataset/images/viscot/gqa"),
    )
    parser.add_argument(
        "--visreason-gqa",
        type=Path,
        default=Path("/home/dataset-local/lkl/datasets/VisReason/train/gqa/dataset.json"),
    )
    parser.add_argument(
        "--visreason-image-root",
        type=Path,
        default=Path("/home/dataset-local/lkl/datasets/LVR_Train_Dataset/images/viscot/gqa"),
    )
    parser.add_argument(
        "--sif",
        type=Path,
        default=Path("/home/dataset-local/lkl/datasets/SIF-50K/SIF-50K.json"),
    )
    parser.add_argument(
        "--sif-image-root",
        type=Path,
        default=Path("/home/dataset-local/lkl/datasets/SIF-50K/images"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is non-empty: {args.output_dir}; pass --overwrite to replace generated files"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mm_rows, mm_skipped = _parse_mm_gcot(args.mm_gcot, args.mm_image_root)
    visreason_rows, visreason_skipped = _parse_visreason_gqa(args.visreason_gqa, args.visreason_image_root)
    sif_rows, sif_skipped = _parse_sif(args.sif, args.sif_image_root)
    merged_rows = mm_rows + visreason_rows + sif_rows

    _write_json(args.output_dir / "mm_gcot_3step.json", mm_rows)
    _write_json(args.output_dir / "visreason_gqa_3step.json", visreason_rows)
    _write_json(args.output_dir / "sif_3step.json", sif_rows)
    _write_json(args.output_dir / "visual_cot_3step_merged.json", merged_rows)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria": {
            "exact_step_count": 3,
            "exact_bbox_count_per_step": 1,
            "bbox_format": "normalized_xyxy",
            "visual_cot": True,
            "sif_rgb_only_in_colt_manifest": True,
        },
        "selected": {
            "mm_gcot": len(mm_rows),
            "visreason_gqa": len(visreason_rows),
            "sif": len(sif_rows),
            "merged": len(merged_rows),
        },
        "skipped": {
            "mm_gcot": dict(mm_skipped),
            "visreason_gqa": dict(visreason_skipped),
            "sif": dict(sif_skipped),
        },
        "source_paths": {
            "mm_gcot": str(args.mm_gcot),
            "visreason_gqa": str(args.visreason_gqa),
            "sif": str(args.sif),
            "sif_image_root": str(args.sif_image_root),
        },
    }
    _write_json(args.output_dir / "selection_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
