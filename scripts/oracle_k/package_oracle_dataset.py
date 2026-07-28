#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


DATASET_REVISION = "7f65a2088bd486b38c24a58c699013d008533388"
REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAT_MODULE_PATH = REPO_ROOT / "transformers-4.57.0/src/transformers/models/qwen3_vl/oracle_k.py"
spec = importlib.util.spec_from_file_location("colt_oracle_k_package_format", FORMAT_MODULE_PATH)
oracle_k_format = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = oracle_k_format
spec.loader.exec_module(oracle_k_format)

SEGMENTER_MODULE_PATH = REPO_ROOT / "scripts/oracle_k/segment_teacher_blocks.py"
segmenter_spec = importlib.util.spec_from_file_location("colt_oracle_k_package_segmenter", SEGMENTER_MODULE_PATH)
segmenter = importlib.util.module_from_spec(segmenter_spec)
assert segmenter_spec.loader is not None
sys.modules[segmenter_spec.name] = segmenter
segmenter_spec.loader.exec_module(segmenter)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def load_annotation_state(path: Path, expected_indices: set[int]) -> tuple[dict, dict[int, dict]]:
    with path.open(encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line:
            raise ValueError(f"Annotation state is empty: {path}")
        meta = json.loads(first_line)
        supported_protocols = {(3, 1), (4, 2), (5, 3)}
        if meta.get("type") != "meta" or (meta.get("format_version"), meta.get("user_prompt_version")) not in supported_protocols:
            raise ValueError("Annotation state uses an unsupported segmentation protocol")

        results = {}
        for line_number, line in enumerate(file, start=2):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") != "result":
                raise ValueError(f"Unexpected state payload at line {line_number}")
            index = payload.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(f"Invalid state index at line {line_number}")
            if index in results:
                raise ValueError(f"Duplicate state result for index {index}")
            results[index] = payload

    if set(results) != expected_indices:
        missing = sorted(expected_indices - set(results))[:10]
        extra = sorted(set(results) - expected_indices)[:10]
        raise ValueError(f"Annotation state range mismatch; missing={missing}, extra={extra}")
    return meta, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a validated Oracle-K JSON and create a transfer manifest.")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--annotated", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="onethinker_sft_image_oracle_k")
    parser.add_argument("--original-start", type=int, default=0)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=8)
    args = parser.parse_args()
    if args.original_start < 0:
        raise ValueError("--original-start must be non-negative")
    if args.min_k < 1 or args.max_k < args.min_k:
        raise ValueError("Require 1 <= min-k <= max-k")

    with args.annotated.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("Annotated dataset must be a non-empty JSON list")
    with args.original.open(encoding="utf-8") as file:
        original_records = json.load(file)
    if not isinstance(original_records, list) or not original_records:
        raise ValueError("Original dataset must be a non-empty JSON list")
    if args.original_start < 0 or args.original_start + len(records) > len(original_records):
        raise ValueError("Annotated range exceeds the original dataset")

    original_sha256 = sha256_file(args.original)
    state_path = args.state_file or args.annotated.with_suffix(args.annotated.suffix + ".state.jsonl")
    expected_indices = set(range(args.original_start, args.original_start + len(records)))
    state_meta, state_results = load_annotation_state(state_path, expected_indices)
    required_state_meta = {
        "model",
        "endpoint",
        "system_prompt_sha256",
        "user_prompt_version",
        "unit_boundary_pattern",
    }
    missing_state_meta = sorted(key for key in required_state_meta if not state_meta.get(key))
    if missing_state_meta:
        raise ValueError(f"Annotation state lacks provenance fields: {missing_state_meta}")
    expected_state_meta = {
        "input_sha256": original_sha256,
        "start": args.original_start,
        "stop": args.original_start + len(records),
        "min_k": args.min_k,
        "max_k": args.max_k,
    }
    for key, expected in expected_state_meta.items():
        if state_meta.get(key) != expected:
            raise ValueError(f"Annotation state {key}={state_meta.get(key)!r}, expected {expected!r}")

    k_distribution = Counter()
    for index, record in enumerate(records):
        matches = [
            (message_index, message["content"])
            for message_index, message in enumerate(record.get("messages", []))
            if message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and "<thought_segments>" in message["content"]
        ]
        if len(matches) != 1:
            raise ValueError(f"record {index}: expected exactly one Oracle-K assistant message")
        message_index, content = matches[0]
        cot = oracle_k_format.get_assistant_cot(content)
        annotation = oracle_k_format.parse_oracle_k_cot(cot, min_k=args.min_k, max_k=args.max_k)
        source_index = args.original_start + index
        state_result = state_results[source_index]
        if state_result.get("message_index") != message_index:
            raise ValueError(f"record {index}: state assistant index does not match annotated record")
        if state_result.get("k") != annotation.k:
            raise ValueError(f"record {index}: state K does not match annotated K")
        boundaries = state_result.get("boundary_after_units")
        if not isinstance(boundaries, list) or len(boundaries) + 1 != annotation.k:
            raise ValueError(f"record {index}: invalid compact boundaries in annotation state")
        original_cot = oracle_k_format.get_assistant_cot(
            original_records[source_index]["messages"][message_index]["content"]
        )
        expected_segmented = segmenter.build_segmented_cot(
            segmenter.split_cot_units(original_cot), boundaries, min_k=args.min_k, max_k=args.max_k
        )
        if expected_segmented != oracle_k_format.CONTINUE_THINK.join(annotation.blocks):
            raise ValueError(f"record {index}: annotated boundaries do not match the compact state")
        k_distribution[annotation.k] += 1
        restored = json.loads(json.dumps(record, ensure_ascii=False))
        restored["messages"][message_index]["content"] = oracle_k_format.remove_assistant_annotation(
            content, min_k=args.min_k, max_k=args.max_k
        )
        if restored != original_records[args.original_start + index]:
            raise ValueError(f"record {index}: Oracle-K removal does not restore the original record")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / f"{args.annotated.name}.gz"
    archive_temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    annotated_digest = hashlib.sha256()
    with args.annotated.open("rb") as source, gzip.open(archive_temporary, "wb", compresslevel=9) as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            annotated_digest.update(chunk)
            target.write(chunk)
    archive_temporary.replace(archive_path)

    manifest = {
        "schema_version": 1,
        "dataset_revision": DATASET_REVISION,
        "dataset_name": args.dataset_name,
        "original_start": args.original_start,
        "min_k": args.min_k,
        "max_k": args.max_k,
        "annotation": {
            "protocol_version": state_meta["format_version"],
            "model": state_meta["model"],
            "endpoint": state_meta["endpoint"],
            "system_prompt_sha256": state_meta["system_prompt_sha256"],
            "user_prompt_version": state_meta["user_prompt_version"],
            "unit_boundary_pattern": state_meta["unit_boundary_pattern"],
            "reasoning_effort": state_meta.get("reasoning_effort"),
            "state_file_name": state_path.name,
            "state_sha256": sha256_file(state_path),
        },
        "original": {
            "file_name": args.original.name,
            "size": args.original.stat().st_size,
            "sha256": original_sha256,
        },
        "annotated": {
            "file_name": args.annotated.name,
            "size": args.annotated.stat().st_size,
            "sha256": annotated_digest.hexdigest(),
            "records": len(records),
            "k_distribution": {str(key): k_distribution[key] for key in sorted(k_distribution)},
        },
        "archive": {
            "file_name": archive_path.name,
            "size": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
    }
    manifest_path = archive_path.with_suffix(archive_path.suffix + ".manifest.json")
    write_json_atomic(manifest_path, manifest)
    print(f"Archive: {archive_path}")
    print(f"Manifest: {manifest_path}")
    print(f"K distribution: {dict(sorted(k_distribution.items()))}")


if __name__ == "__main__":
    main()
