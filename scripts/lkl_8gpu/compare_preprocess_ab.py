#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import numbers
from pathlib import Path

import pandas as pd

from validate_eval_suite import PAPER_SCORE_PROFILES


FRACTIONAL_SCORE_DATASETS = {"AI2D_TEST"}
PAPER_BASELINE_SCORES = PAPER_SCORE_PROFILES["qwen3vl_cot"]
LEGACY_PROFILE = "legacy14_processor_resize"


def normalize_index(value: object) -> str:
    if pd.isna(value):
        raise RuntimeError("A prediction or result row contains a missing index.")
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def load_overall(path: Path, dataset: str) -> float:
    score = pd.read_csv(path, keep_default_na=False)
    if "Overall" not in score.columns:
        raise RuntimeError(f"Missing Overall column: {path}")
    rows = score
    if "split" in score.columns:
        all_rows = score[score["split"].astype(str).str.upper() == "ALL"]
        if not all_rows.empty:
            rows = all_rows
    values = pd.to_numeric(rows["Overall"], errors="coerce").dropna()
    if values.empty:
        raise RuntimeError(f"No numeric Overall score: {path}")
    value = float(values.iloc[-1])
    return value * 100.0 if dataset in FRACTIONAL_SCORE_DATASETS else value


def load_variant(result_dir: Path, model_name: str, dataset: str) -> pd.DataFrame:
    prefix = f"{model_name}_{dataset}"
    prediction_path = result_dir / f"{prefix}.xlsx"
    score_path = result_dir / f"{prefix}_acc.csv"
    detail_candidates = [
        result_dir / f"{prefix}_exact_matching_result.xlsx",
        result_dir / f"{prefix}_results.xlsx",
    ]
    detail_path = next((path for path in detail_candidates if path.is_file()), None)
    for path in (prediction_path, score_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if detail_path is None:
        raise FileNotFoundError(f"No detailed result workbook found for {prefix} under {result_dir}")

    predictions = pd.read_excel(prediction_path, keep_default_na=False)
    details = pd.read_excel(detail_path, keep_default_na=False)
    score_column = "hit" if "hit" in details.columns else "eval_score"
    if not {"index", "prediction"}.issubset(predictions.columns) or score_column not in details.columns:
        raise RuntimeError(f"Invalid prediction or detail columns for {prefix}")

    predictions = predictions[["index", "prediction"]].copy()
    predictions["key"] = predictions["index"].map(normalize_index)
    if "prediction" not in details.columns:
        raise RuntimeError(f"Detailed result workbook has no prediction column for {prefix}")
    details = details[["index", "prediction", score_column]].copy()
    details["key"] = details["index"].map(normalize_index)
    if predictions["key"].duplicated().any() or details["key"].duplicated().any():
        raise RuntimeError(f"Duplicate indices in {prefix}")

    if dataset == "MMBench_DEV_EN":
        predictions = predictions[predictions["key"].isin(set(details["key"]))]
    merged = predictions.merge(
        details[["key", "prediction", score_column]],
        on="key",
        suffixes=("_prediction", "_detail"),
        validate="one_to_one",
    )
    if len(merged) != len(predictions) or len(merged) != len(details):
        raise RuntimeError(f"Prediction/detail index mismatch for {prefix}")
    merged["prediction_prediction"] = merged["prediction_prediction"].astype(str)
    merged["prediction_detail"] = merged["prediction_detail"].astype(str)
    if (merged["prediction_prediction"] != merged["prediction_detail"]).any():
        raise RuntimeError(f"Prediction/detail answer mismatch for {prefix}")
    merged["item_score"] = pd.to_numeric(merged[score_column], errors="raise").astype(float)

    overall = load_overall(score_path, dataset)
    recomputed = float(merged["item_score"].mean() * 100.0)
    if not math.isclose(overall, recomputed, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"Score mismatch for {prefix}: score_csv={overall:.12f}, item_mean={recomputed:.12f}"
        )
    merged["prediction"] = merged["prediction_prediction"]
    return merged[["key", "prediction", "item_score"]].sort_values("key").reset_index(drop=True)


def parse_variant(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("--variant must use NAME=/absolute/result/directory")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare isolated Qwen3-VL preprocessing evaluation profiles.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    variants = args.variant
    if len(variants) < 2 or len({name for name, _ in variants}) != len(variants):
        parser.error("provide at least two uniquely named --variant arguments")
    variant_names = {name for name, _ in variants}
    if LEGACY_PROFILE not in variant_names:
        parser.error(f"missing required reference profile: {LEGACY_PROFILE}")
    unknown_datasets = [dataset for dataset in args.datasets if dataset not in PAPER_BASELINE_SCORES]
    if unknown_datasets:
        parser.error(f"unsupported datasets for the paper baseline comparison: {unknown_datasets}")

    reference_name = LEGACY_PROFILE
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        loaded = {
            name: load_variant(result_dir, args.model_name, dataset)
            for name, result_dir in variants
        }
        reference = loaded[reference_name]
        reference_score = float(reference["item_score"].mean() * 100.0)
        paper_score = PAPER_BASELINE_SCORES[dataset]
        reference_keys = set(reference["key"])
        for name, _ in variants:
            current = loaded[name]
            if set(current["key"]) != reference_keys:
                raise RuntimeError(f"Index mismatch between {reference_name} and {name} for {dataset}")
            paired = reference.merge(current, on="key", suffixes=("_reference", "_current"), validate="one_to_one")
            score_delta = paired["item_score_current"] - paired["item_score_reference"]
            changed = paired["prediction_reference"] != paired["prediction_current"]
            current_score = float(paired["item_score_current"].mean() * 100.0)
            lengths = paired["prediction_current"].astype(str).str.len()
            rows.append(
                {
                    "dataset": dataset,
                    "reference": reference_name,
                    "variant": name,
                    "rows": len(paired),
                    "score": current_score,
                    "paper_score": paper_score,
                    "gap_vs_paper": current_score - paper_score,
                    "legacy_score": reference_score,
                    "delta_vs_legacy": float(score_delta.mean() * 100.0),
                    "changed_predictions": int(changed.sum()),
                    "changed_prediction_rate": float(changed.mean()),
                    "improved_rows": int((score_delta > 1e-12).sum()),
                    "regressed_rows": int((score_delta < -1e-12).sum()),
                    "unchanged_score_rows": int((score_delta.abs() <= 1e-12).sum()),
                    "prediction_chars_max": int(lengths.max()),
                    "prediction_chars_ge_100": int((lengths >= 100).sum()),
                    "prediction_chars_ge_1000": int((lengths >= 1000).sum()),
                    "zero_score_long_ge_100": int(
                        ((lengths >= 100) & (paired["item_score_current"] <= 1e-12)).sum()
                    ),
                    "zero_score_long_ge_1000": int(
                        ((lengths >= 1000) & (paired["item_score_current"] <= 1e-12)).sum()
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    summary.to_csv(temporary, index=False)
    temporary.replace(args.output)

    display = summary.copy()
    display["score"] = display["score"].map(lambda value: f"{value:.4f}")
    display["paper_score"] = display["paper_score"].map(lambda value: f"{value:.2f}")
    display["gap_vs_paper"] = display["gap_vs_paper"].map(lambda value: f"{value:+.4f}")
    display["legacy_score"] = display["legacy_score"].map(lambda value: f"{value:.4f}")
    display["delta_vs_legacy"] = display["delta_vs_legacy"].map(lambda value: f"{value:+.4f}")
    display["changed_prediction_rate"] = display["changed_prediction_rate"].map(lambda value: f"{value:.2%}")
    print(f"\nPreprocessing A/B comparison; reference={reference_name}")
    print(display.to_string(index=False))
    print(f"\nSaved comparison: {args.output}")


if __name__ == "__main__":
    main()
