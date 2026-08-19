#!/usr/bin/env python3
"""Promote the complete GQA grounding manifest to visual-CoT supervision.

The original 30K manifest used ``visual_only`` as a conservative training
policy, even though every row carries exactly three grounding-box groups.
The teacher text contains between three and eight numbered sentences.  This
converter preserves every sentence while grouping the text into three
continuous ``Step 1/2/3`` segments, matching the three latent/bbox slots used
by the current training path.

All emitted rows have ``visual_cot=true`` and ``visual_only=false``.  The
source-program statistics are retained in the report so known grounding-risk
families can be audited without silently removing them from the requested
experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


NUMBERED_STEP_RE = re.compile(r"(?<!\d)([1-9])[\.:\)]\s+")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _question_key_from_raw(row: dict[str, Any]) -> tuple[str, str, str]:
    image = Path(str(row["image"])).name
    return image, str(row["question"]).strip(), str(row["answer"]).strip()


def _question_key_from_manifest(row: dict[str, Any]) -> tuple[str, str, str]:
    user = next(
        message["content"]
        for message in row["messages"]
        if message.get("role") == "user"
    )
    question = user.replace("<image>", "", 1).split("\n", 1)[0].strip()
    assistant = next(
        message["content"]
        for message in row["messages"]
        if message.get("role") == "assistant"
    )
    answer_match = ANSWER_RE.search(assistant)
    if answer_match is None:
        raise ValueError("GQA manifest row is missing an <answer>...</answer> block.")
    return Path(str(row["images"][0])).name, question, answer_match.group(1).strip()


def _extract_thought(row: dict[str, Any]) -> str:
    assistant = next(
        message["content"]
        for message in row["messages"]
        if message.get("role") == "assistant"
    )
    match = THINK_RE.search(assistant)
    if match is None:
        raise ValueError("GQA manifest row is missing a <think>...</think> block.")
    return match.group(1).strip()


def _parse_at_least_three_steps(thought: str) -> tuple[list[str] | None, int]:
    matches = list(NUMBERED_STEP_RE.finditer(thought))
    if len(matches) < 3:
        return None, len(matches)
    steps = [
        thought[matches[0].end() : matches[1].start()].strip(),
        thought[matches[1].end() : matches[2].start()].strip(),
        thought[matches[2].end() :].strip(),
    ]
    return (steps if all(steps) else None), len(matches)


def _replace_thought(row: dict[str, Any], steps: list[str]) -> dict[str, Any]:
    output = copy.deepcopy(row)
    for message in output["messages"]:
        if message.get("role") != "assistant":
            continue
        content = message["content"]
        match = THINK_RE.search(content)
        if match is None:
            raise ValueError("GQA manifest row is missing an assistant thought block.")
        normalized = "\n".join(f"Step {index}: {step}" for index, step in enumerate(steps, 1))
        message["content"] = (
            content[: match.start(1)] + normalized + content[match.end(1) :]
        )
        return output
    raise ValueError("GQA manifest row has no assistant message.")


def convert(
    manifest_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw_row in raw_rows:
        key = _question_key_from_raw(raw_row)
        if key in raw_by_key:
            previous = raw_by_key[key]
            if (
                previous.get("thought") != raw_row.get("thought")
                or previous.get("reasoning") != raw_row.get("reasoning")
            ):
                raise ValueError(f"Non-identical duplicate raw GQA key: {key!r}")
            continue
        raw_by_key[key] = raw_row

    output: list[dict[str, Any]] = []
    reasons = Counter()
    numbered_step_counts = Counter()
    source_program_step_counts = Counter()
    source_operations = Counter()
    for row_index, row in enumerate(manifest_rows):
        key = _question_key_from_manifest(row)
        raw_row = raw_by_key.get(key)
        if raw_row is None:
            raise ValueError(f"Manifest row {row_index} has no raw GQA source: {key!r}")

        reasoning = raw_row.get("reasoning") or []
        source_program_step_counts[str(len(reasoning))] += 1
        source_operations[" -> ".join(str(step.get("operation", "")) for step in reasoning)] += 1
        steps, numbered_step_count = _parse_at_least_three_steps(_extract_thought(row))
        numbered_step_counts[str(numbered_step_count)] += 1
        if steps is None:
            reasons["thought_has_fewer_than_three_numbered_steps"] += 1
            raise ValueError(
                f"GQA row {row_index} cannot be normalized to three steps: {key!r}"
            )

        row = _replace_thought(row, steps)
        row["visual_cot"] = True
        row["visual_only"] = False
        output.append(row)

    report = {
        "input_rows": len(manifest_rows),
        "output_rows": len(output),
        "visual_cot_rows": sum(bool(row["visual_cot"]) for row in output),
        "visual_only_rows": sum(bool(row["visual_only"]) for row in output),
        "residual_reasons": dict(reasons),
        "numbered_step_counts_before_normalization": dict(numbered_step_counts),
        "source_program_step_counts": dict(source_program_step_counts),
        "source_operations": dict(source_operations),
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/home/dataset-local/lkl/datasets/CoLT_Train_Dataset/"
            "colt_sft_gqa_step_grounding_30k.json"
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/home/dataset-local/lkl/datasets/LVR_Train_Dataset/"
            "visualcot_98k/lvr_gqa_merged_98k.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/dataset-local/lkl/datasets/CoLT_Train_Dataset/"
            "colt_sft_gqa_visual_cot_30k_all.json"
        ),
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    raw_rows = json.loads(args.source.read_text(encoding="utf-8"))
    output, report = convert(manifest_rows, raw_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(report_path), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
