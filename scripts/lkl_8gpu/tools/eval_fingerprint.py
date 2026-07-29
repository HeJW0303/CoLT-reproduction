#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def add_text(digest: "hashlib._Hash", label: str, value: str) -> None:
    digest.update(label.encode())
    digest.update(b"\0")
    digest.update(value.encode())
    digest.update(b"\0")


def add_file(digest: "hashlib._Hash", path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    add_text(digest, "path", label)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def add_model(digest: "hashlib._Hash", root: Path) -> None:
    root = root.resolve(strict=True)
    add_text(digest, "resolved_model", str(root))
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            stat = path.stat()
            add_text(digest, "weight", f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        elif path.suffix in {".json", ".model", ".py", ".txt"} or path.name.startswith("tokenizer"):
            add_file(digest, path, f"model/{path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--setting", action="append", default=[])
    args = parser.parse_args()

    root = args.repo_root.resolve(strict=True)
    digest = hashlib.sha256()
    for setting in sorted(args.setting):
        add_text(digest, "setting", setting)
    for distribution in (
        "torch",
        "transformers",
        "flash-attn",
        "qwen-vl-utils",
        "numpy",
        "pandas",
        "Pillow",
        "opencv-python-headless",
        "accelerate",
        "safetensors",
        "openpyxl",
        "xlsxwriter",
    ):
        try:
            package_version = version(distribution)
        except PackageNotFoundError:
            package_version = "missing"
        add_text(digest, "package", f"{distribution}=={package_version}")

    source_roots = (
        root / "scripts/lkl_8gpu",
        root / "Evaluation/VLMEvalKit/vlmeval",
        root / "transformers-4.57.0/src/transformers/models/qwen3_vl",
        root / "transformers-4.57.0/src/transformers/generation",
    )
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*")):
            if path.is_file() and path.suffix in {".sh", ".py", ".txt"}:
                add_file(digest, path, str(path.relative_to(root)))
    add_file(digest, root / "Evaluation/VLMEvalKit/run.py", "Evaluation/VLMEvalKit/run.py")
    add_model(digest, args.model_dir)
    print(digest.hexdigest()[:12])


if __name__ == "__main__":
    main()
