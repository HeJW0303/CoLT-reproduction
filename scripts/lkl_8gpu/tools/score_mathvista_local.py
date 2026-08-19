#!/usr/bin/env python3
"""Local exact/robust scoring for MathVista_MINI predictions.

VLMEvalKit's MathVista evaluate path requires an OpenAI key for free-form
answer extraction. This script reproduces the local scoring rules
(multi-choice option match; integer/float extraction; list parse) so the
latent-intervention comparison can run without an external judge.

Usage (colt env):
  python scripts/lkl_8gpu/tools/score_mathvista_local.py \
      eval/results/paper-faithful/MathVista_MINI/<mode-dir>/.../Qwen3-VL-8B-Instruct-COLT_MathVista_MINI.xlsx
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd


def extract_choice(response: str) -> str | None:
    matches = re.findall(r"\b([A-E])\b", str(response))
    return matches[-1] if matches else None


def extract_number(response: str, as_float: bool) -> float | None:
    matches = re.findall(r"-?\d+\.?\d*", str(response))
    if not matches:
        return None
    raw = matches[-1]
    try:
        return float(raw)
    except ValueError:
        return None


def parse_list(response: str) -> list[float] | None:
    m = re.search(r"\[([^\]]*)\]", str(response))
    if not m:
        return None
    try:
        return [float(x.strip()) for x in m.group(1).split(",") if x.strip() != ""]
    except ValueError:
        return None


def score_row(row: pd.Series) -> bool:
    qtype = row["question_type"]
    atype = row["answer_type"]
    answer = row["answer"]
    response = row["prediction"]
    if qtype == "multi_choice":
        pred = extract_choice(response)
        return pred is not None and pred == str(row["answer_option"]).strip()
    if atype == "integer":
        pred = extract_number(response, as_float=False)
        return pred is not None and int(pred) == int(float(str(answer)))
    if atype == "float":
        pred = extract_number(response, as_float=True)
        if pred is None:
            return False
        gt = float(str(answer))
        return abs(pred - gt) <= max(1e-3, 1e-3 * abs(gt))
    if atype == "list":
        pred = parse_list(response)
        gt = parse_list(str(answer))
        return pred is not None and gt is not None and pred == gt
    pred = str(response).strip().lower()
    gt = str(answer).strip().lower()
    return pred == gt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx", nargs="+")
    parser.add_argument("--by-task", action="store_true", help="Also report per-task accuracy")
    args = parser.parse_args()

    for path in args.xlsx:
        df = pd.read_excel(path)
        hits = df.apply(score_row, axis=1)
        print(f"=== {path.split('/')[-1]}  n={len(df)}  acc={hits.mean() * 100:.2f}%")
        if args.by_task:
            by_task = df.assign(hit=hits).groupby("task").agg(
                n=("hit", "size"),
                acc=("hit", "mean"),
            )
            by_task["acc"] = by_task["acc"] * 100
            print(by_task.round(2).to_string())


if __name__ == "__main__":
    main()
