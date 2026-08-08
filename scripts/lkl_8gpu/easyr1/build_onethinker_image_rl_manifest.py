#!/usr/bin/env python3
"""Build the media-complete image subset of OneThinker RL annotations."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


EXPECTED_IMAGE_RECORDS = 189645


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_media_path(media_root, relative_path):
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("image entries must be non-empty strings")
    if Path(relative_path).is_absolute():
        raise ValueError("image entries must be relative paths: {!r}".format(relative_path))

    candidate = (media_root / relative_path).resolve()
    try:
        candidate.relative_to(media_root)
    except ValueError:
        raise ValueError("image path escapes media root: {!r}".format(relative_path))
    return candidate


def load_image_records(source_file, media_root, expected_records):
    with source_file.open(encoding="utf-8") as handle:
        source_records = json.load(handle)
    if not isinstance(source_records, list):
        raise ValueError("source JSON must contain a top-level list")

    image_records = []
    for index, record in enumerate(source_records):
        if not isinstance(record, dict):
            raise ValueError("source record {} is not an object".format(index))
        if record.get("data_type") != "image":
            continue

        images = record.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("image record {} has no image paths".format(index))
        for image_path in images:
            resolved_path = resolve_media_path(media_root, image_path)
            if not resolved_path.is_file():
                raise FileNotFoundError(
                    "image record {} references missing media: {}".format(index, resolved_path)
                )
        image_records.append(record)

    if len(image_records) != expected_records:
        raise ValueError(
            "expected {} image records, found {}; refuse to write a partial manifest".format(
                expected_records, len(image_records)
            )
        )
    return source_records, image_records


def write_json_atomically(output_file, records, overwrite):
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            "output already exists: {} (pass --overwrite after reviewing it)".format(output_file)
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(output_file.parent), prefix=output_file.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary_file = Path(handle.name)
            json.dump(records, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_file), str(output_file))
    finally:
        if temporary_file is not None and temporary_file.exists():
            temporary_file.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--expected-records", type=int, default=EXPECTED_IMAGE_RECORDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_file = args.source_file.resolve()
    media_root = args.media_root.resolve()
    output_file = args.output_file.resolve()

    if not source_file.is_file():
        raise FileNotFoundError("source file does not exist: {}".format(source_file))
    if not media_root.is_dir():
        raise NotADirectoryError("media root does not exist: {}".format(media_root))
    if args.expected_records <= 0:
        raise ValueError("--expected-records must be positive")

    source_records, image_records = load_image_records(source_file, media_root, args.expected_records)
    write_json_atomically(output_file, image_records, args.overwrite)
    print(
        json.dumps(
            {
                "source_file": str(source_file),
                "source_records": len(source_records),
                "source_sha256": sha256_file(source_file),
                "media_root": str(media_root),
                "manifest_file": str(output_file),
                "image_records": len(image_records),
                "manifest_sha256": sha256_file(output_file),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
