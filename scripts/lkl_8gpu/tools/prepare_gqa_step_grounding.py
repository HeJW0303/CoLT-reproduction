#!/usr/bin/env python3
"""Build a high-confidence GQA step-grounding subset for CMPO visual SFT.

For every 3-step GQA functional program (select -> relate/filter -> query) it
extracts per-step visual evidence:

    V_1 = objects of step 1
    V_2 = objects of step 1 + step 2      (relate/filter input + output)
    V_3 = objects of step 2               (answer object)

Each object id is resolved through the official GQA scene graph to a normalized
bbox. Rows are filtered for:
    - complete 3-step program
    - every involved object id resolvable in the scene graph
    - valid bboxes (area within [min_area, max_area], not empty after clamp)
Then a fixed-seed random sample of ``--n`` rows is emitted in CoLT sharegpt
format with a ``step_bboxes`` column and ``visual_only: true`` marker (GQA rows
only contribute the visual grounding loss).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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


def load_scene_graphs(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalized_bbox(scene_graph: dict, img_id: str, obj_id: int) -> list[float] | None:
    image = scene_graph.get(img_id)
    if image is None:
        return None
    obj = image["objects"].get(str(obj_id))
    if obj is None:
        return None
    width, height = image["width"], image["height"]
    if not width or not height:
        return None
    x1 = max(0.0, obj["x"] / width)
    y1 = max(0.0, obj["y"] / height)
    x2 = min(1.0, (obj["x"] + obj["w"]) / width)
    y2 = min(1.0, (obj["y"] + obj["h"]) / height)
    if x1 >= x2 or y1 >= y2:
        return None
    return [x1, y1, x2, y2]


def object_ids(reasoning: list[dict]) -> list[int]:
    ids = []
    for step in reasoning:
        ids.extend(int(m) for m in re.findall(r"\((\d+)\)", step.get("argument", "")))
    return ids


def step_evidence(reasoning: list[dict]) -> tuple[list[set[int]], list[set[int]]]:
    """Return ``(all_step_objs, target_objs)``.

    ``all_step_objs[k]`` = objects involved up to step k (including dependency
    chain). ``target_objs[k]`` = the visual evidence target for latent k:
      - V_1 = step-1 objects
      - V_2 = step-1 + step-2 objects
      - V_3 = step-2 argument objects only (the answer object, not the input)
    """
    step_objs: list[set[int]] = []
    step_new: list[set[int]] = []
    for i, step in enumerate(reasoning):
        arg_objs: set[int] = set(int(m) for m in re.findall(r"\((\d+)\)", step.get("argument", "")))
        dep_objs: set[int] = set()
        for dep in step.get("dependencies", []):
            if dep < len(step_objs):
                dep_objs.update(step_objs[dep])
        step_new.append(arg_objs)
        step_objs.append(arg_objs | dep_objs)
    if len(step_objs) != 3:
        return [], []
    targets = [step_objs[0], step_objs[0] | step_objs[1], step_new[1]]
    return step_objs, targets


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
        "--scene-graph",
        default="/home/dataset-local/lkl/tmp/train_sceneGraphs.json",
    )
    parser.add_argument(
        "--image-root",
        default="/home/dataset-local/lkl/datasets/LVR_Train_Dataset/images",
    )
    parser.add_argument("--n", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=float, default=0.005)
    parser.add_argument("--max-area", type=float, default=0.95)
    parser.add_argument(
        "--output",
        default=(
            "/home/dataset-local/lkl/datasets/CoLT_Train_Dataset/"
            "colt_sft_gqa_step_grounding_15k.json"
        ),
    )
    args = parser.parse_args()

    scene_graph = load_scene_graphs(args.scene_graph)
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    dropped = {"prog_len": 0, "unresolved_obj": 0, "invalid_bbox": 0, "empty_target": 0}
    for d in data:
        reasoning = d.get("reasoning") or []
        if len(reasoning) != 3:
            dropped["prog_len"] += 1
            continue
        img_id = d["image"].split("/")[-1].replace(".jpg", "").replace(".png", "")
        if img_id not in scene_graph:
            dropped["unresolved_obj"] += 1
            continue
        _, targets = step_evidence(reasoning)
        if not targets:
            dropped["prog_len"] += 1
            continue

        target_bboxes = []
        valid = True
        for target in targets:
            bboxes = []
            for obj_id in sorted(target):
                bbox = normalized_bbox(scene_graph, img_id, obj_id)
                if bbox is None:
                    dropped["unresolved_obj"] += 1
                    valid = False
                    break
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if not (args.min_area <= area <= args.max_area):
                    dropped["invalid_bbox"] += 1
                    valid = False
                    break
                bboxes.append(bbox)
            if not valid:
                break
            if not bboxes:
                dropped["empty_target"] += 1
                valid = False
                break
            target_bboxes.append(bboxes)
        if not valid:
            continue

        image_path = os.path.join(args.image_root, d["image"])
        if not os.path.exists(image_path):
            continue
        question = (d.get("question") or "").strip()
        thought = (d.get("thought") or "").strip()
        answer = (d.get("answer") or "").strip()
        if not question or not thought or not answer:
            continue

        candidates.append(
            {
                "image": image_path,
                "question": question,
                "thought": thought,
                "answer": answer,
                "step_bboxes": target_bboxes,
            }
        )

    print(f"candidates after filtering: {len(candidates)}", flush=True)
    print(f"dropped: {dropped}", flush=True)

    rng = random.Random(args.seed)
    sample = rng.sample(candidates, min(args.n, len(candidates)))
    converted = []
    for item in sample:
        converted.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>{item['question']}\n{ANSWER_TEMPLATE}",
                    },
                    {
                        "role": "assistant",
                        "content": f"<think>{item['thought']}</think>\n<answer>{item['answer']}</answer>",
                    },
                ],
                "images": [item["image"]],
                "step_bboxes": item["step_bboxes"],
                "visual_only": True,
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False)
    print(f"output: {args.output} ({len(converted)} rows)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
