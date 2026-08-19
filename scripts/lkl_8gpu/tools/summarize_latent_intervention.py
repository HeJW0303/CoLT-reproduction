#!/usr/bin/env python3
"""Summarize paired accuracy changes from a latent-intervention evaluation.

The four intervention modes evaluate the same examples with deterministic
decoding.  This tool therefore aligns rows by ``index`` and uses paired
statistics; independent-sample tests would overstate uncertainty incorrectly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


MODES = ("none", "zero", "random", "skip")
RESULT_PAT = re.compile(
    r"(?:^|/)paper-faithful/(?:chart-text|mmstar)/"
    r"replicas[^/]*_li(?P<mode>none|zero|random|skip)"
    r"(?:_mn\d+)?_seed[^/]*/[^/]+/"
    r"Qwen3-VL-8B-Instruct-COLT_(?P<dataset>.+?)_results\.xlsx$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("eval/results/paper-faithful"),
        help="Root containing paper-faithful result profiles.",
    )
    parser.add_argument(
        "--path-filter",
        default="",
        help="Require this literal substring in every selected result path.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--permutation-replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    score_column = "eval_score" if "eval_score" in frame.columns else "hit"
    required = {"index", "prediction", score_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame["index"].duplicated().any():
        raise ValueError(f"{path} contains duplicate example indices.")

    rows = frame.loc[:, ["index", "prediction", score_column]].copy()
    rows = rows.rename(columns={score_column: "score"})
    rows["score"] = pd.to_numeric(rows["score"], errors="raise")
    if rows["score"].isna().any():
        raise ValueError(f"{path} has missing evaluation scores.")
    return rows.set_index("index", verify_integrity=True).sort_index()


def discover_results(result_root: Path, path_filter: str) -> dict[str, dict[str, tuple[Path, pd.DataFrame]]]:
    discovered: dict[str, dict[str, tuple[Path, pd.DataFrame]]] = {}
    for path in sorted(result_root.rglob("*results.xlsx")):
        path_text = str(path)
        if path_filter and path_filter not in path_text:
            continue
        match = RESULT_PAT.search(path_text)
        if not match:
            continue
        dataset, mode = match.group("dataset"), match.group("mode")
        modes = discovered.setdefault(dataset, {})
        if mode in modes:
            raise ValueError(
                f"Multiple {mode!r} result files for {dataset!r}: {modes[mode][0]} and {path}. "
                "Use --path-filter to select one experiment."
            )
        modes[mode] = (path, load_rows(path))

    if not discovered:
        raise FileNotFoundError(f"No matching result workbooks under {result_root}.")
    for dataset, modes in discovered.items():
        missing = sorted(set(MODES).difference(modes))
        if missing:
            raise ValueError(f"{dataset} is missing intervention modes: {missing}")
    return discovered


def assert_aligned(baseline: pd.DataFrame, candidate: pd.DataFrame, dataset: str, mode: str) -> None:
    if not baseline.index.equals(candidate.index):
        missing_from_mode = baseline.index.difference(candidate.index).tolist()[:10]
        missing_from_none = candidate.index.difference(baseline.index).tolist()[:10]
        raise ValueError(
            f"{dataset}/{mode} does not align with none by index; "
            f"missing_from_mode={missing_from_mode}, missing_from_none={missing_from_none}."
        )


def paired_bootstrap_ci(differences: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    if replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive.")
    count = len(differences)
    estimates = np.empty(replicates, dtype=np.float64)
    batch_size = min(1_000, replicates)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        sampled = rng.integers(0, count, size=(stop - start, count), dtype=np.int32)
        estimates[start:stop] = differences[sampled].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def is_binary(scores: np.ndarray) -> bool:
    return bool(np.all(np.isin(scores, (0.0, 1.0))))


def paired_randomization_pvalue(
    differences: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> float:
    if replicates <= 0:
        raise ValueError("--permutation-replicates must be positive.")
    observed = abs(float(differences.mean()))
    if observed == 0.0:
        return 1.0
    exceed = 0
    batch_size = min(1_000, replicates)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        signs = rng.integers(0, 2, size=(stop - start, len(differences)), dtype=np.int8)
        signs = signs * 2 - 1
        null_means = (signs @ differences) / len(differences)
        exceed += int(np.count_nonzero(np.abs(null_means) >= observed - 1e-15))
    return float((exceed + 1) / (replicates + 1))


def compare(
    baseline: np.ndarray,
    candidate: np.ndarray,
    rng: np.random.Generator,
    bootstrap_replicates: int,
    permutation_replicates: int,
) -> dict[str, Any]:
    differences = candidate - baseline
    result: dict[str, Any] = {
        "n": int(len(baseline)),
        "none_accuracy": float(baseline.mean()),
        "mode_accuracy": float(candidate.mean()),
        "difference": float(differences.mean()),
        "difference_ci95": list(paired_bootstrap_ci(differences, rng, bootstrap_replicates)),
    }
    if is_binary(baseline) and is_binary(candidate):
        none_wins = int(np.count_nonzero((baseline == 1.0) & (candidate == 0.0)))
        mode_wins = int(np.count_nonzero((baseline == 0.0) & (candidate == 1.0)))
        result.update(
            {
                "test": "exact_mcnemar",
                "none_wins": none_wins,
                "mode_wins": mode_wins,
                "p_value": float(stats.binomtest(min(none_wins, mode_wins), none_wins + mode_wins, 0.5).pvalue)
                if none_wins + mode_wins
                else 1.0,
            }
        )
    else:
        result.update(
            {
                "test": "paired_randomization",
                "p_value": paired_randomization_pvalue(differences, rng, permutation_replicates),
            }
        )
    return result


def summarize_dataset(
    dataset: str,
    modes: dict[str, tuple[Path, pd.DataFrame]],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    baseline = modes["none"][1]
    score_arrays = {"none": baseline["score"].to_numpy(dtype=np.float64)}
    summary: dict[str, Any] = {
        "paths": {mode: str(modes[mode][0]) for mode in MODES},
        "n": int(len(baseline)),
        "modes": {
            "none": {
                "n": int(len(baseline)),
                "accuracy": float(score_arrays["none"].mean()),
            }
        },
    }
    for mode in MODES[1:]:
        candidate = modes[mode][1]
        assert_aligned(baseline, candidate, dataset, mode)
        score_arrays[mode] = candidate["score"].to_numpy(dtype=np.float64)
        summary["modes"][mode] = compare(
            score_arrays["none"],
            score_arrays[mode],
            rng,
            args.bootstrap_replicates,
            args.permutation_replicates,
        )

    zero, skip = modes["zero"][1], modes["skip"][1]
    assert_aligned(zero, skip, dataset, "zero-vs-skip")
    summary["zero_skip_control"] = {
        "prediction_equal_fraction": float(
            (zero["prediction"].fillna("").astype(str) == skip["prediction"].fillna("").astype(str)).mean()
        ),
        "score_equal_fraction": float(np.isclose(zero["score"], skip["score"]).mean()),
    }
    return summary, score_arrays


def print_summary(summary: dict[str, Any]) -> None:
    for dataset, dataset_summary in summary["datasets"].items():
        print(f"\n--- {dataset} (n={dataset_summary['n']}) ---")
        print(f"  none   {dataset_summary['modes']['none']['accuracy'] * 100:7.3f}")
        for mode in MODES[1:]:
            metrics = dataset_summary["modes"][mode]
            ci_low, ci_high = metrics["difference_ci95"]
            print(
                f"  {mode:6s} {metrics['mode_accuracy'] * 100:7.3f} "
                f"diff={metrics['difference'] * 100:+7.3f}pp "
                f"CI=[{ci_low * 100:+.3f}, {ci_high * 100:+.3f}] "
                f"{metrics['test']} p={metrics['p_value']:.3g}"
            )
        control = dataset_summary.get("zero_skip_control")
        if control is not None:
            print(
                "  zero/skip equality: "
                f"predictions={control['prediction_equal_fraction']:.3%}, "
                f"scores={control['score_equal_fraction']:.3%}"
            )


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0 or args.permutation_replicates <= 0:
        raise ValueError("Bootstrap and permutation replicate counts must be positive.")
    discovered = discover_results(args.result_root, args.path_filter)
    rng = np.random.default_rng(args.seed)
    datasets: dict[str, dict[str, Any]] = {}
    pooled_scores = {mode: [] for mode in MODES}
    for dataset in sorted(discovered):
        dataset_summary, score_arrays = summarize_dataset(dataset, discovered[dataset], rng, args)
        datasets[dataset] = dataset_summary
        for mode, scores in score_arrays.items():
            pooled_scores[mode].append(scores)

    if len(datasets) > 1:
        pooled = {mode: np.concatenate(chunks) for mode, chunks in pooled_scores.items()}
        pooled_summary: dict[str, Any] = {
            "n": int(len(pooled["none"])),
            "modes": {"none": {"n": int(len(pooled["none"])), "accuracy": float(pooled["none"].mean())}},
        }
        for mode in MODES[1:]:
            pooled_summary["modes"][mode] = compare(
                pooled["none"], pooled[mode], rng, args.bootstrap_replicates, args.permutation_replicates
            )
        datasets["ALL_POOLED"] = pooled_summary

    summary = {
        "path_filter": args.path_filter,
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "seed": args.seed,
        "datasets": datasets,
    }
    print_summary(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if args.output_csv:
        rows = []
        for dataset, dataset_summary in datasets.items():
            for mode, metrics in dataset_summary["modes"].items():
                rows.append({"dataset": dataset, "mode": mode, **metrics})
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
