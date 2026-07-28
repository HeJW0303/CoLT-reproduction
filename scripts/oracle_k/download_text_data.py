#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path


DATASET_REPO = "hulianyuyy/CoLT_Train_Dataset"
DATASET_REVISION = "7f65a2088bd486b38c24a58c699013d008533388"
DATASET_FILE = "colt_sft_image.json"
EXPECTED_SIZE = 284614647
EXPECTED_SHA256 = "737054d823716c172c0bea7c8e32dc1e998d8fd7383179762cd043de14511eca"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> None:
    if path.stat().st_size != EXPECTED_SIZE:
        raise ValueError(f"Unexpected file size for {path}: {path.stat().st_size}")
    digest = sha256_file(path)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Unexpected SHA-256 for {path}: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only the fixed CoLT training JSON, without image ZIPs.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or repo_root / "data/oracle_k_source"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / DATASET_FILE

    if output_path.is_file() and not args.force:
        verify(output_path)
        print(f"Already verified: {output_path}")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("Install huggingface_hub before downloading the CoLT JSON") from error

        endpoints = args.endpoints or [
            os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            "https://hf-mirror.com",
        ]
        errors = []
        for endpoint in dict.fromkeys(endpoints):
            try:
                downloaded = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename=DATASET_FILE,
                    repo_type="dataset",
                    revision=DATASET_REVISION,
                    endpoint=endpoint,
                    local_dir=output_dir,
                    force_download=args.force,
                )
                output_path = Path(downloaded)
                verify(output_path)
                print(f"Downloaded from {endpoint}: {output_path}")
                break
            except Exception as error:
                errors.append(f"{endpoint}: {type(error).__name__}: {error}")
        else:
            raise RuntimeError("All Hugging Face endpoints failed:\n" + "\n".join(errors))

    manifest = {
        "repo_id": DATASET_REPO,
        "revision": DATASET_REVISION,
        "file_name": DATASET_FILE,
        "size": EXPECTED_SIZE,
        "sha256": EXPECTED_SHA256,
    }
    manifest_path = output_dir / "source_manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    temporary.replace(manifest_path)
    print(f"Source manifest: {manifest_path}")


if __name__ == "__main__":
    main()
