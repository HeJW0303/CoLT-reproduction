#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_DATASETS = ("MathVista_MINI", "MathVerse_MINI", "MMVet")
FAILURE_MARKERS = ("Failed to obtain answer via API", "All 5 retries failed")


def normalize_index(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("index contains a null value")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and float(value).is_integer():
        return str(int(value))
    return str(value)


def normalized_indices(values: Any, label: str) -> list[str]:
    indices = [normalize_index(value) for value in values]
    if len(indices) != len(set(indices)):
        duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
        raise RuntimeError(f"{label} contains duplicate indices: {duplicates[:5]}")
    return indices


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing {label}: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Empty {label}: {path}")
    return path


def load_xlsx(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    data = pd.read_excel(path)
    if data.empty:
        raise RuntimeError(f"{label} contains no rows: {path}")
    return data


def validate_index_contract(data: pd.DataFrame, expected: set[str], label: str) -> None:
    if "index" not in data.columns:
        raise RuntimeError(f"{label} has no index column")
    actual = normalized_indices(data["index"], label)
    if len(actual) != len(expected):
        raise RuntimeError(f"{label} row count is {len(actual)}, expected {len(expected)}")
    actual_set = set(actual)
    if actual_set != expected:
        missing = sorted(expected - actual_set)[:5]
        extra = sorted(actual_set - expected)[:5]
        raise RuntimeError(f"{label} index mismatch; missing={missing}, extra={extra}")


def validate_prediction(
    path: Path, expected: set[str], allow_empty_predictions: bool = False
) -> pd.DataFrame:
    data = load_xlsx(path, "prediction file")
    validate_index_contract(data, expected, "prediction file")
    if "prediction" not in data.columns:
        raise RuntimeError("prediction file has no prediction column")
    predictions = data["prediction"]
    empty = predictions.isna() | predictions.astype(str).str.strip().eq("")
    if empty.any() and not allow_empty_predictions:
        raise RuntimeError(f"prediction file contains {int(empty.sum())} empty predictions")
    failed = predictions.astype(str).str.contains("Failed to obtain answer", regex=False)
    if failed.any():
        raise RuntimeError(f"prediction file contains {int(failed.sum())} failed predictions")
    return data


def validate_pickle(path: Path, expected: set[str], required_fields: tuple[str, ...], label: str) -> None:
    require_file(path, label)
    data = pd.read_pickle(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} must contain an index-keyed dictionary")
    indices = normalized_indices(data.keys(), label)
    if set(indices) != expected:
        missing = sorted(expected - set(indices))[:5]
        extra = sorted(set(indices) - expected)[:5]
        raise RuntimeError(f"{label} index mismatch; missing={missing}, extra={extra}")
    for index, record in data.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"{label} record {index!r} is not a dictionary")
        missing_fields = [field for field in required_fields if field not in record]
        if missing_fields:
            raise RuntimeError(f"{label} record {index!r} is missing {missing_fields}")


def validate_logs(data: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        if column not in data.columns:
            raise RuntimeError(f"{label} has no {column} column")
        logs = data[column].fillna("").astype(str)
        failed = logs.map(lambda log: any(marker in log for marker in FAILURE_MARKERS))
        if failed.any():
            raise RuntimeError(f"{label} contains {int(failed.sum())} failed judge records in {column}")


def read_overall_row(path: Path, key_column: str, value_column: str, label: str) -> pd.Series:
    require_file(path, label)
    data = pd.read_csv(path)
    if key_column not in data.columns or value_column not in data.columns:
        raise RuntimeError(f"{label} must contain {key_column!r} and {value_column!r}")
    overall = data[data[key_column].astype(str) == "Overall"]
    if len(overall) != 1:
        raise RuntimeError(f"{label} must contain exactly one Overall row, found {len(overall)}")
    row = overall.iloc[0]
    value = float(row[value_column])
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise RuntimeError(f"{label} Overall score is outside [0, 100]: {value}")
    return row


def artifact(base: Path, suffix: str, extension: str) -> Path:
    return base.with_name(f"{base.stem}{suffix}.{extension}")


def validate_mathvista(base: Path, expected: set[str], judge_model: str) -> float:
    storage = artifact(base, f"_{judge_model}", "xlsx")
    temporary = artifact(base, f"_{judge_model}", "pkl")
    score = artifact(base, f"_{judge_model}_score", "csv")
    judged = load_xlsx(storage, "MathVista judge result")
    validate_index_contract(judged, expected, "MathVista judge result")
    validate_logs(judged, ("log",), "MathVista judge result")
    validate_pickle(temporary, expected, ("log", "res"), "MathVista judge checkpoint")
    overall = read_overall_row(score, "Task&Skill", "acc", "MathVista accuracy artifact")
    if "tot" not in overall.index or "hit" not in overall.index:
        raise RuntimeError("MathVista accuracy artifact must contain tot and hit")
    total = int(overall["tot"])
    hit = int(overall["hit"])
    reported = float(overall["acc"])
    if total != len(expected) or not 0 <= hit <= total:
        raise RuntimeError(f"MathVista Overall hit/tot is invalid: {hit}/{total}")
    computed = hit / total * 100
    if not math.isclose(reported, computed, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError(f"MathVista accuracy {reported} does not match hit/tot {computed}")
    return reported


def validate_mathverse(base: Path, expected: set[str], judge_model: str) -> float:
    extract = artifact(base, f"_{judge_model}_extract", "xlsx")
    extract_tmp = artifact(base, f"_{judge_model}_extract", "pkl")
    scored = artifact(base, f"_{judge_model}_score", "xlsx")
    score_tmp = artifact(base, f"_{judge_model}_score", "pkl")
    score = artifact(base, f"_{judge_model}_score", "csv")

    extracted = load_xlsx(extract, "MathVerse extraction result")
    validate_index_contract(extracted, expected, "MathVerse extraction result")
    validate_logs(extracted, ("log_extract",), "MathVerse extraction result")
    validate_pickle(
        extract_tmp, expected, ("log_extract", "extract"), "MathVerse extraction checkpoint"
    )

    judged = load_xlsx(scored, "MathVerse score result")
    validate_index_contract(judged, expected, "MathVerse score result")
    validate_logs(judged, ("log_extract", "log_score"), "MathVerse score result")
    validate_pickle(score_tmp, expected, ("log_score", "score"), "MathVerse score checkpoint")
    if "score" not in judged.columns:
        raise RuntimeError("MathVerse score result has no score column")
    numeric_scores = pd.to_numeric(judged["score"], errors="raise")
    if not numeric_scores.isin([0, 1]).all():
        raise RuntimeError("MathVerse judge scores must all be boolean or 0/1")
    if "problem_version" not in judged.columns:
        raise RuntimeError("MathVerse score result has no problem_version column")

    require_file(score, "MathVerse accuracy artifact")
    score_data = pd.read_csv(score)
    required_columns = {"split", "Overall"}
    missing_columns = required_columns - set(score_data.columns)
    if missing_columns:
        raise RuntimeError(
            f"MathVerse accuracy artifact is missing required columns: {sorted(missing_columns)}"
        )
    if score_data["split"].isna().any() or score_data["split"].astype(str).duplicated().any():
        raise RuntimeError("MathVerse accuracy artifact split values must be present and unique")

    reported_scores = pd.to_numeric(score_data["Overall"], errors="raise")
    if not reported_scores.map(lambda value: math.isfinite(float(value)) and 0 <= value <= 100).all():
        raise RuntimeError("MathVerse accuracy artifact Overall values must be finite percentages in [0, 100]")

    problem_versions = judged["problem_version"].astype(str)
    artifact_splits = set(score_data["split"].astype(str))
    expected_splits = set(problem_versions)
    if artifact_splits != expected_splits:
        missing = sorted(expected_splits - artifact_splits)
        extra = sorted(artifact_splits - expected_splits)
        raise RuntimeError(
            f"MathVerse accuracy artifact split mismatch; missing={missing}, extra={extra}"
        )

    for _, row in score_data.iterrows():
        split = str(row["split"])
        computed = float(numeric_scores.mean() * 100) if split == "Overall" else float(
            numeric_scores[problem_versions == split].mean() * 100
        )
        reported = float(row["Overall"])
        if not math.isclose(reported, computed, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(
                f"MathVerse {split!r} accuracy {reported} does not match judge mean {computed}"
            )

    return float(numeric_scores.mean() * 100)


def validate_mmvet(base: Path, expected: set[str], judge_model: str) -> float:
    storage = artifact(base, f"_{judge_model}", "xlsx")
    temporary = artifact(base, f"_{judge_model}", "pkl")
    score = artifact(base, f"_{judge_model}_score", "csv")
    score_fine = artifact(base, f"_{judge_model}_score_fine", "csv")
    judged = load_xlsx(storage, "MMVet judge result")
    validate_index_contract(judged, expected, "MMVet judge result")
    validate_logs(judged, ("log",), "MMVet judge result")
    validate_pickle(temporary, expected, ("log", "score"), "MMVet judge checkpoint")
    if "score" not in judged.columns:
        raise RuntimeError("MMVet judge result has no score column")
    numeric_scores = pd.to_numeric(judged["score"], errors="raise")
    if not numeric_scores.map(lambda value: math.isfinite(float(value)) and 0 <= value <= 1).all():
        raise RuntimeError("MMVet judge scores must all be finite values in [0, 1]")
    overall = read_overall_row(score, "Category", "acc", "MMVet score artifact")
    if "tot" not in overall.index or int(overall["tot"]) != len(expected):
        raise RuntimeError("MMVet score artifact Overall total does not match the dataset")
    fine_overall = read_overall_row(
        score_fine, "Category", "acc", "MMVet fine-grained score artifact"
    )
    if "tot" not in fine_overall.index or int(fine_overall["tot"]) != len(expected):
        raise RuntimeError("MMVet fine-grained score artifact Overall total does not match the dataset")
    reported = float(overall["acc"])
    continuous_average = float(numeric_scores.mean() * 100)
    if not math.isclose(reported, continuous_average, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError(
            f"MMVet Overall score {reported} does not match continuous mean {continuous_average}"
        )
    return continuous_average


def validate_dataset(
    result_dir: Path,
    data_root: Path,
    model_name: str,
    dataset: str,
    judge_model: str,
    allow_empty_predictions: bool,
) -> dict[str, Any]:
    source = require_file(data_root / f"{dataset}.tsv", f"{dataset} source dataset")
    source_indices = normalized_indices(
        pd.read_csv(source, sep="\t", usecols=["index"])["index"], source.name
    )
    expected = set(source_indices)
    base = result_dir / f"{model_name}_{dataset}.xlsx"
    validate_prediction(base, expected, allow_empty_predictions)

    if dataset == "MathVista_MINI":
        score = validate_mathvista(base, expected, judge_model)
    elif dataset == "MathVerse_MINI":
        score = validate_mathverse(base, expected, judge_model)
    elif dataset == "MMVet":
        score = validate_mmvet(base, expected, judge_model)
    else:
        raise RuntimeError(f"Unsupported external-judge dataset: {dataset}")
    return {"dataset": dataset, "rows": len(expected), "score": score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--allow-empty-predictions",
        action="store_true",
        help="Allow empty model outputs that the selected evaluation policy treats as scored answers.",
    )
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()
    unknown = [dataset for dataset in args.datasets if dataset not in SUPPORTED_DATASETS]
    if unknown:
        parser.error(f"unsupported datasets: {', '.join(unknown)}")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("dataset arguments must be unique")

    result_dir = args.work_dir / args.model_name / args.eval_id
    if not result_dir.is_dir() or result_dir.is_symlink():
        raise RuntimeError(f"Missing evaluation result directory: {result_dir}")
    rows = [
        validate_dataset(
            result_dir,
            args.data_root,
            args.model_name,
            dataset,
            args.judge_model,
            args.allow_empty_predictions,
        )
        for dataset in args.datasets
    ]
    summary = pd.DataFrame(rows)
    summary["score"] = summary["score"].map(lambda value: f"{value:.6f}")
    print("\nValidated external-judge evaluation (scores are percentages):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
