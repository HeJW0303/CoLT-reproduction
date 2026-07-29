#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("eval_id")
    args = parser.parse_args()

    result_dir = args.work_dir / args.model_name / args.eval_id
    base = f"{args.model_name}_COLT_SMOKE_MMSTAR"
    prediction = result_dir / f"{base}.xlsx"
    score = result_dir / f"{base}_acc.csv"
    if not prediction.is_file() or not score.is_file():
        raise RuntimeError(f"Missing smoke outputs under {result_dir}")
    data = pd.read_excel(prediction, keep_default_na=False)
    if len(data) != 8 or not {"index", "prediction"}.issubset(data.columns):
        raise RuntimeError(f"Invalid smoke predictions: rows={len(data)} columns={list(data.columns)}")
    responses = data["prediction"].astype(str).str.strip()
    if (responses == "").any() or responses.str.contains("Failed to obtain answer", case=False, regex=False).any():
        raise RuntimeError("Smoke predictions contain an empty or failed response.")
    print(f"Smoke evaluation validated: predictions={prediction} score={score}")


if __name__ == "__main__":
    main()
