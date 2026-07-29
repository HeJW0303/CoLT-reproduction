#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import numbers
from pathlib import Path

import pandas as pd


PAPER_SCORE_PROFILES = {
    "colt": {
        "SEEDBench_IMG": 77.5,
        "MMBench_DEV_EN": 84.6,
        "ChartQA_TEST": 74.7,
        "TextVQA_VAL": 81.3,
        "ScienceQA_TEST": 92.8,
        "MMStar": 68.9,
        "AI2D_TEST": 85.4,
        "MMT-Bench_VAL": 67.4,
    },
    "qwen3vl_cot": {
        "SEEDBench_IMG": 76.4,
        "MMBench_DEV_EN": 83.4,
        "ChartQA_TEST": 65.1,
        "TextVQA_VAL": 75.2,
        "ScienceQA_TEST": 91.8,
        "MMStar": 67.1,
        "AI2D_TEST": 83.6,
        "MMT-Bench_VAL": 63.3,
    },
}

FRACTIONAL_SCORE_DATASETS = {
    "SEEDBench_IMG",
    "MMBench_DEV_EN",
    "ScienceQA_TEST",
    "MMStar",
    "AI2D_TEST",
    "MMT-Bench_VAL",
}


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path, keep_default_na=False)
    if suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    raise ValueError(f"Unsupported table format: {path}")


def normalize_index(value: object) -> str:
    if pd.isna(value):
        raise RuntimeError("Dataset or prediction file contains a missing index.")
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_indices(series: pd.Series) -> list[str]:
    return [normalize_index(value) for value in series]


def select_overall(score: pd.DataFrame, dataset: str) -> float:
    if "Overall" not in score.columns:
        raise RuntimeError(f"{dataset} score file has no Overall column: {list(score.columns)}")

    rows = score
    if "split" in score.columns:
        split = score["split"].fillna("").astype(str).str.upper()
        all_rows = score[split == "ALL"]
        if not all_rows.empty:
            rows = all_rows

    values = pd.to_numeric(rows["Overall"], errors="coerce").dropna()
    if values.empty:
        raise RuntimeError(f"{dataset} score file has no finite Overall value.")

    value = float(values.iloc[-1])
    if not math.isfinite(value):
        raise RuntimeError(f"{dataset} Overall score is not finite: {value}")
    if dataset in FRACTIONAL_SCORE_DATASETS:
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{dataset} fractional Overall score is out of range: {value}")
        return value * 100.0
    if not 0.0 <= value <= 100.0:
        raise RuntimeError(f"{dataset} percentage Overall score is out of range: {value}")
    return value


def validate_dataset(
    result_dir: Path,
    data_root: Path,
    model_name: str,
    dataset: str,
    paper_scores: dict[str, float],
) -> dict[str, object]:
    source_file = data_root / f"{dataset}.tsv"
    prediction_file = result_dir / f"{model_name}_{dataset}.xlsx"
    score_file = result_dir / f"{model_name}_{dataset}_acc.csv"

    for kind, path in (
        ("source TSV", source_file),
        ("prediction", prediction_file),
        ("score", score_file),
    ):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing regular {kind} file for {dataset}: {path}")

    source = pd.read_csv(source_file, sep="\t", usecols=["index"])
    predictions = load_table(prediction_file)
    required_columns = {"index", "prediction"}
    if not required_columns.issubset(predictions.columns):
        raise RuntimeError(
            f"Invalid {dataset} prediction columns: {list(predictions.columns)}"
        )

    source_indices = normalize_indices(source["index"])
    prediction_indices = normalize_indices(predictions["index"])
    if len(set(source_indices)) != len(source_indices):
        raise RuntimeError(f"{dataset} source TSV contains duplicate indices.")
    if len(predictions) != len(source):
        raise RuntimeError(
            f"Invalid {dataset} prediction count: {len(predictions)}/{len(source)}"
        )
    if len(set(prediction_indices)) != len(prediction_indices):
        raise RuntimeError(f"{dataset} predictions contain duplicate indices.")
    if set(prediction_indices) != set(source_indices):
        missing = sorted(set(source_indices) - set(prediction_indices))[:10]
        extra = sorted(set(prediction_indices) - set(source_indices))[:10]
        raise RuntimeError(
            f"{dataset} prediction indices mismatch: missing={missing}, extra={extra}"
        )

    responses = predictions["prediction"].fillna("").astype(str)
    if (responses.str.strip() == "").any():
        bad = predictions.loc[responses.str.strip() == "", "index"].head(10).tolist()
        raise RuntimeError(f"{dataset} predictions contain empty responses at indices {bad}")
    failed = responses.str.contains("Failed to obtain answer", case=False, regex=False)
    if failed.any():
        bad = predictions.loc[failed, "index"].head(10).tolist()
        raise RuntimeError(f"{dataset} predictions contain failed responses at indices {bad}")

    score = pd.read_csv(score_file)
    if score.empty:
        raise RuntimeError(f"{dataset} score file is empty: {score_file}")
    overall = select_overall(score, dataset)

    print(f"\n[{dataset}] validated {len(predictions)} predictions")
    print(f"Prediction: {prediction_file}")
    print(f"Score: {score_file}")
    print(score.to_string(index=False))

    paper = paper_scores[dataset]
    return {
        "dataset": dataset,
        "rows": len(predictions),
        "score": overall,
        "paper": paper,
        "gap": overall - paper,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate complete CoLT prediction and score files for an evaluation suite."
    )
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("eval_id")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument(
        "--paper-profile",
        choices=sorted(PAPER_SCORE_PROFILES),
        default="colt",
    )
    args = parser.parse_args()

    paper_scores = PAPER_SCORE_PROFILES[args.paper_profile]
    unknown = [dataset for dataset in args.datasets if dataset not in paper_scores]
    if unknown:
        parser.error(f"unsupported datasets: {unknown}")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("dataset arguments must be unique")

    result_dir = args.work_dir / args.model_name / args.eval_id
    if not result_dir.is_dir() or result_dir.is_symlink():
        raise RuntimeError(f"Missing evaluation result directory: {result_dir}")

    summaries = [
        validate_dataset(result_dir, args.data_root, args.model_name, dataset, paper_scores)
        for dataset in args.datasets
    ]
    summary = pd.DataFrame(summaries)
    macro = pd.DataFrame(
        [
            {
                "dataset": "MACRO_AVG",
                "rows": int(summary["rows"].sum()),
                "score": float(summary["score"].mean()),
                "paper": float(summary["paper"].mean()),
                "gap": float(summary["gap"].mean()),
            }
        ]
    )
    summary = pd.concat([summary, macro], ignore_index=True)
    summary["score"] = summary["score"].map(lambda value: f"{value:.2f}")
    summary["paper"] = summary["paper"].map(lambda value: f"{value:.2f}")
    summary["gap"] = summary["gap"].map(lambda value: f"{value:+.2f}")

    print(f"\nValidated evaluation suite against paper profile {args.paper_profile} (scores are percentages):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
