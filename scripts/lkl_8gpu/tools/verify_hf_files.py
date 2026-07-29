#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=["model", "dataset"], required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--suffix", action="append", default=[])
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    api = HfApi()
    if args.repo_type == "dataset":
        info = api.dataset_info(args.repo_id, revision=args.revision, files_metadata=True)
    else:
        info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True)

    expected = {}
    explicit_files = set(args.file)
    for sibling in info.siblings:
        name = sibling.rfilename
        selected = name in explicit_files or any(name.endswith(suffix) for suffix in args.suffix)
        lfs = sibling.lfs
        if isinstance(lfs, dict):
            expected_hash = lfs.get("sha256")
        else:
            expected_hash = getattr(lfs, "sha256", None)
        if selected and expected_hash:
            expected[name] = expected_hash

    if not expected:
        raise RuntimeError("The Hub API returned no selected LFS SHA-256 metadata.")
    if args.expected_count is not None and len(expected) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} selected LFS files, Hub metadata returned {len(expected)}.")

    failures = []
    for name, expected_hash in sorted(expected.items()):
        path = args.local_dir / name
        if not path.is_file():
            failures.append(f"missing: {name}")
            continue
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            failures.append(f"sha256 mismatch: {name}")
        else:
            print(f"OK {name}")

    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Verified {len(expected)} LFS files against Hub metadata.")


if __name__ == "__main__":
    main()
