#!/usr/bin/env python3
"""Prepare external-judge checkpoints for a safe resumed evaluation."""

from __future__ import annotations

import argparse
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Any


FAILURE_MARKERS = ("Failed to obtain answer via API", "All 5 retries failed")
_LEGACY_SCORE_RESPONSE = re.compile(r"res is (.*?), failed to parse\.\n", re.DOTALL)
_BARE_BINARY_SCORE = re.compile(r"^\s*([01])\s*$")
_LABELLED_BINARY_SCORE = re.compile(
    r"^\s*(?:final\s+)?judg(?:e)?ment\s*(?::|is)?\s*([01])\s*[.!]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def artifact(base: Path, suffix: str, extension: str) -> Path:
    return base.with_name(f"{base.stem}{suffix}.{extension}")


def is_failed_record(record: dict[str, Any], log_field: str) -> bool:
    return any(marker in str(record[log_field]) for marker in FAILURE_MARKERS)


def parse_mathverse_score_response(response: str) -> int | None:
    bare_match = _BARE_BINARY_SCORE.fullmatch(response)
    if bare_match:
        return int(bare_match.group(1))

    labelled_scores = [int(match.group(1)) for match in _LABELLED_BINARY_SCORE.finditer(response)]
    if labelled_scores and len(set(labelled_scores)) == 1:
        return labelled_scores[0]
    return None


def recover_legacy_mathverse_score_records(
    checkpoint: Path, completed_artifacts: tuple[Path, ...]
) -> int:
    if checkpoint.is_symlink():
        raise RuntimeError(f"Refusing to read a symlinked judge checkpoint: {checkpoint}")
    if not checkpoint.exists():
        return 0

    data = load_checkpoint(checkpoint, ("log_score", "score"))
    recovered = 0
    for record in data.values():
        log_score = str(record["log_score"])
        if "All 5 retries failed" not in log_score:
            continue
        recovered_score = next(
            (
                score
                for response in _LEGACY_SCORE_RESPONSE.findall(log_score)
                if (score := parse_mathverse_score_response(response)) is not None
            ),
            None,
        )
        if recovered_score is None:
            continue
        record["log_score"] = "Recovered from a label-compatible legacy judge response"
        record["score"] = recovered_score == 1
        recovered += 1

    if not recovered:
        return 0
    archive_completed_artifacts(completed_artifacts)
    atomic_write_checkpoint(checkpoint, data)
    print(f"Recovered {recovered} MathVerse score records from legacy judge logs")
    return recovered


def load_checkpoint(path: Path, required_fields: tuple[str, ...]) -> dict[Any, dict[str, Any]]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Judge checkpoint must contain an index-keyed dictionary: {path}")
    for index, record in data.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"Judge checkpoint record {index!r} is not a dictionary: {path}")
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise RuntimeError(f"Judge checkpoint record {index!r} is missing {missing}: {path}")
    return data


def atomic_write_checkpoint(path: Path, data: dict[Any, dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def archive_completed_artifacts(paths: tuple[Path, ...]) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"Refusing to archive a symlinked judge artifact: {path}")
        if path.exists():
            if not path.is_file():
                raise RuntimeError(f"Expected a judge artifact file, found: {path}")
            archive = path.with_name(f"{path.name}.failed-judge-{timestamp}.bak")
            if archive.exists():
                raise RuntimeError(f"Judge artifact archive already exists: {archive}")
            os.replace(path, archive)


def retry_failed_records(
    checkpoint: Path,
    required_fields: tuple[str, ...],
    log_field: str,
    completed_artifacts: tuple[Path, ...],
) -> int:
    if checkpoint.is_symlink():
        raise RuntimeError(f"Refusing to read a symlinked judge checkpoint: {checkpoint}")
    if not checkpoint.exists():
        return 0
    data = load_checkpoint(checkpoint, required_fields)
    failed_indices = [index for index, record in data.items() if is_failed_record(record, log_field)]
    if not failed_indices:
        return 0

    for index in failed_indices:
        del data[index]
    archive_completed_artifacts(completed_artifacts)
    atomic_write_checkpoint(checkpoint, data)
    print(f"Prepared {checkpoint.name}: will retry {len(failed_indices)} failed judge records")
    return len(failed_indices)


def prepare_dataset(result_dir: Path, model_name: str, judge_model: str, dataset: str) -> int:
    base = result_dir / f"{model_name}_{dataset}.xlsx"
    if dataset == "MathVista_MINI":
        checkpoint = artifact(base, f"_{judge_model}", "pkl")
        return retry_failed_records(
            checkpoint,
            ("log", "res"),
            "log",
            (artifact(base, f"_{judge_model}", "xlsx"), artifact(base, f"_{judge_model}_score", "csv")),
        )
    if dataset == "MathVerse_MINI":
        extract_checkpoint = artifact(base, f"_{judge_model}_extract", "pkl")
        extract_failed = retry_failed_records(
            extract_checkpoint,
            ("log_extract", "extract"),
            "log_extract",
            (
                artifact(base, f"_{judge_model}_extract", "xlsx"),
                artifact(base, f"_{judge_model}_score", "xlsx"),
                artifact(base, f"_{judge_model}_score", "pkl"),
                artifact(base, f"_{judge_model}_score", "csv"),
            ),
        )
        if extract_failed:
            return extract_failed
        score_checkpoint = artifact(base, f"_{judge_model}_score", "pkl")
        score_artifacts = (
            artifact(base, f"_{judge_model}_score", "xlsx"),
            artifact(base, f"_{judge_model}_score", "csv"),
        )
        recover_legacy_mathverse_score_records(score_checkpoint, score_artifacts)
        retried = retry_failed_records(
            score_checkpoint,
            ("log_score", "score"),
            "log_score",
            score_artifacts,
        )
        return retried
    if dataset == "MMVet":
        checkpoint = artifact(base, f"_{judge_model}", "pkl")
        return retry_failed_records(
            checkpoint,
            ("log", "score"),
            "log",
            (
                artifact(base, f"_{judge_model}", "xlsx"),
                artifact(base, f"_{judge_model}_score", "csv"),
                artifact(base, f"_{judge_model}_score_fine", "csv"),
            ),
        )
    raise ValueError(f"Unsupported external-judge dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retain successful judge checkpoints and retry only terminal API failures."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()

    if not args.result_dir.is_dir() or args.result_dir.is_symlink():
        raise RuntimeError(f"Missing evaluation result directory: {args.result_dir}")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("dataset arguments must be unique")

    retried = sum(
        prepare_dataset(args.result_dir, args.model_name, args.judge_model, dataset)
        for dataset in args.datasets
    )
    print(f"Judge resume preparation complete: retry_records={retried}")


if __name__ == "__main__":
    main()
