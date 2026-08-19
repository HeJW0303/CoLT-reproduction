#!/usr/bin/env python3
"""Generate a bbox<->CoT alignment audit report for the Visual-CoT 98K data.

The report samples rows stratified by question type (with extra weight on
spatial/relation rows, where a single ROI may fail to cover the full reasoning
evidence) and prints the question, the normalized bbox, its area, and the
start of the CoT so a human can check alignment.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys


def question_type(q: str) -> str:
    q = q.lower()
    if re.search(r"how many|number of|count", q):
        return "counting"
    if re.search(
        r"\b(left|right|beside|next to|above|below|behind|in front|between|near|on top|under|inside|outside)\b",
        q,
    ):
        return "spatial/relation"
    if re.search(r"\b(color|colour|what color|shape|made of|wearing|material)\b", q):
        return "attribute"
    if re.search(r"\b(where|location|position)\b", q):
        return "where/position"
    if re.search(r"\b(who|what|which|is there|are there|does)\b", q):
        return "general-vqa"
    return "other"


def bbox_area(b: list[float]) -> float:
    return max(0.0, min(1.0, b[2]) - max(0.0, b[0])) * max(
        0.0, min(1.0, b[3]) - max(0.0, b[1])
    )


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
        "--output",
        default=(
            "/home/dataset-local/lkl/CoLT-reproduction/Markdown/会话记录/"
            "20260815_visual_cot_bbox对齐检查.md"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-type", type=int, default=30)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    buckets: dict[str, list[dict]] = {}
    for d in data:
        buckets.setdefault(question_type(d["question"]), []).append(d)

    rng = random.Random(args.seed)
    lines = [
        "# Visual-CoT 98K bbox ↔ CoT 对齐人工检查",
        "",
        "> 抽样方式：按 question 类型分层随机抽样（relation/spatial 是重点风险，"
        "因为单 bbox 可能无法覆盖完整推理证据）。",
        "> bbox 为归一化 xyxy；面积越小表示 ROI 越聚焦。",
        "",
        "## 类型分布（全量 88,294 条）",
        "",
        "| 类型 | 条数 | 占比 |",
        "|---|---:|---:|",
    ]
    total = len(data)
    for qtype, rows in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {qtype} | {len(rows)} | {100.0 * len(rows) / total:.1f}% |")

    for qtype in sorted(buckets.keys()):
        rows = buckets[qtype]
        sample = rng.sample(rows, min(args.per_type, len(rows)))
        lines.append("")
        lines.append(f"## {qtype}（抽样 {len(sample)} 条）")
        lines.append("")
        for i, d in enumerate(sample):
            b = d["lvr_bbox"][0]
            area = bbox_area(b)
            flag = ""
            if qtype == "spatial/relation" or qtype == "counting":
                flag = " ⚠️ relation/counting，单 bbox 需人工确认是否覆盖全部证据"
            elif area < 0.01:
                flag = " ⚠️ 极小板"
            lines.append(f"**[{i}]** Q: {d['question'][:160]}")
            lines.append("")
            lines.append(
                f"- bbox `[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]` 面积 `{area:.3f}`{flag}"
            )
            lines.append(
                f"- CoT 开头: {d['thought'][:180].replace(chr(10), ' ')}"
            )
            lines.append("")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"audit report written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
