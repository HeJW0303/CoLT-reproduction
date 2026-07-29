#!/usr/bin/env python3

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import transformers


def require_version(distribution: str, expected: str, prefix: bool = False) -> None:
    actual = version(distribution)
    valid = actual.startswith(expected) if prefix else actual == expected
    if not valid:
        raise RuntimeError(f"{distribution}: expected {expected}, found {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--adapter", choices=("colt", "baseline"), required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    transformers_file = Path(transformers.__file__).resolve()
    if transformers.__version__ != "4.57.0":
        raise RuntimeError(f"transformers: expected 4.57.0, found {transformers.__version__}")
    if not transformers_file.is_relative_to(repo / "transformers-4.57.0"):
        raise RuntimeError(f"Transformers is not loaded from the vendored tree: {transformers_file}")

    require_version("torch", "2.6.0", prefix=True)
    require_version("flash-attn", "2.7.4.post1")
    require_version("qwen-vl-utils", "0.0.14")
    require_version("numpy", "1.26.4")
    require_version("opencv-python-headless", "4.11.0.86")
    for distribution in ("openpyxl", "xlsxwriter"):
        try:
            version(distribution)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Missing evaluation package: {distribution}") from error

    if args.adapter == "baseline":
        from vlmeval.vlm import Qwen3VLBaseChat as adapter

        expected_suffix = "qwen3_vl_baseline"
    else:
        from vlmeval.vlm import Qwen3VLChat as adapter

        expected_suffix = "colt_qwen3_vl"
    if not adapter.__module__.endswith(expected_suffix):
        raise RuntimeError(f"Unexpected adapter module: {adapter.__module__}")
    print(f"Evaluation environment verified: adapter={args.adapter} transformers={transformers_file}")


if __name__ == "__main__":
    main()
