#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAT_MODULE_PATH = (
    REPO_ROOT / "transformers-4.57.0/src/transformers/models/qwen3_vl/oracle_k.py"
)
spec = importlib.util.spec_from_file_location("colt_oracle_k_format", FORMAT_MODULE_PATH)
oracle_k_format = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = oracle_k_format
spec.loader.exec_module(oracle_k_format)

get_assistant_cot = oracle_k_format.get_assistant_cot
parse_oracle_k_cot = oracle_k_format.parse_oracle_k_cot
remove_assistant_annotation = oracle_k_format.remove_assistant_annotation


def find_annotated_message(record: dict[str, Any]) -> tuple[int, str]:
    matches = []
    for index, message in enumerate(record.get("messages", [])):
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, str) and "<thought_segments>" in content:
            matches.append((index, content))
    if len(matches) != 1:
        raise ValueError(f"Expected one annotated assistant message, found {len(matches)}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate reversible Oracle-K dataset annotations.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--original-start", type=int, default=0)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=8)
    args = parser.parse_args()
    if args.original_start < 0:
        parser.error("--original-start must be non-negative")
    if args.expected_records is not None and args.expected_records < 1:
        parser.error("--expected-records must be positive")
    if args.min_k < 1 or args.max_k < args.min_k:
        parser.error("Require 1 <= min-k <= max-k")
    return args


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("Annotated dataset must be a non-empty JSON list")
    if args.expected_records is not None and len(records) != args.expected_records:
        raise ValueError(f"Expected {args.expected_records} records, found {len(records)}")

    original_records = None
    if args.original:
        with args.original.open(encoding="utf-8") as file:
            original_records = json.load(file)
        if args.original_start + len(records) > len(original_records):
            raise ValueError("Annotated range exceeds the original dataset")

    counts = Counter()
    errors = []
    for index, record in enumerate(records):
        try:
            message_index, content = find_annotated_message(record)
            annotation = parse_oracle_k_cot(
                get_assistant_cot(content),
                min_k=args.min_k,
                max_k=args.max_k,
            )
            counts[annotation.k] += 1
            if original_records is not None:
                restored = json.loads(json.dumps(record, ensure_ascii=False))
                restored["messages"][message_index]["content"] = remove_assistant_annotation(
                    content,
                    min_k=args.min_k,
                    max_k=args.max_k,
                )
                original = original_records[args.original_start + index]
                if restored != original:
                    raise ValueError("Removing Oracle-K markers does not exactly restore the original record")
        except Exception as error:
            if len(errors) < 100:
                errors.append(f"record {index}: {error}")

    if errors:
        raise ValueError("Oracle-K validation failed:\n" + "\n".join(errors))
    distribution = ", ".join(f"K={k}:{counts[k]}" for k in sorted(counts))
    print(f"Validated {len(records)} Oracle-K records. {distribution}")


if __name__ == "__main__":
    main()
