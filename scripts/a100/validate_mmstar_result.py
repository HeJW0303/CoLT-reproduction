#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")


def normalize_indices(series: pd.Series) -> list[str]:
    def normalize(value) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    return [normalize(value) for value in series]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("data_file", type=Path)
    args = parser.parse_args()

    dataset = "MMStar"
    base = f"{args.model_name}_{dataset}"
    source = pd.read_csv(args.data_file, sep="\t", usecols=["index"])
    source_indices = normalize_indices(source["index"])

    predictions = [
        path
        for path in args.work_dir.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.stem == base
        and path.suffix.lower() in {".xlsx", ".csv", ".tsv"}
    ]
    if not predictions:
        raise RuntimeError("No MMStar prediction file was produced.")

    prediction_file = max(predictions, key=lambda path: path.stat().st_mtime)
    data = load_table(prediction_file)
    if "index" not in data or "prediction" not in data:
        raise RuntimeError(f"Invalid prediction columns: {list(data.columns)}")

    prediction_indices = normalize_indices(data["index"])
    if len(data) != len(source):
        raise RuntimeError(f"Invalid MMStar predictions: rows={len(data)}/{len(source)}")
    if len(set(prediction_indices)) != len(prediction_indices):
        raise RuntimeError("MMStar predictions contain duplicate indices.")
    if set(prediction_indices) != set(source_indices):
        missing = sorted(set(source_indices) - set(prediction_indices))[:10]
        extra = sorted(set(prediction_indices) - set(source_indices))[:10]
        raise RuntimeError(f"MMStar prediction indices mismatch: missing={missing}, extra={extra}")

    responses = data["prediction"].fillna("").astype(str)
    if (responses.str.strip() == "").any():
        raise RuntimeError("MMStar predictions contain an empty response.")
    if responses.str.contains("Failed to obtain answer", regex=False).any():
        raise RuntimeError("MMStar predictions contain a failed response.")

    score_files = [
        path
        for path in prediction_file.parent.glob(f"{base}_acc.csv")
        if path.is_file() and not path.is_symlink()
    ]
    if len(score_files) != 1:
        raise RuntimeError(f"Expected one MMStar score file beside predictions, found: {score_files}")

    score = pd.read_csv(score_files[0])
    if score.empty:
        raise RuntimeError("MMStar score file is empty.")

    print(f"Validated {len(data)} predictions: {prediction_file}")
    print(f"Validated score: {score_files[0]}")
    print(score.to_string(index=False))


if __name__ == "__main__":
    main()
