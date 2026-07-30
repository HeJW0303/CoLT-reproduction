#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a separate Oracle-K dataset in dataset_info.json.")
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--dataset-name", default="onethinker_sft_image_oracle_k")
    args = parser.parse_args()

    with args.dataset_info.open(encoding="utf-8") as file:
        dataset_info = json.load(file)
    legacy_entry = {
        "file_name": args.file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
    }
    entry = {
        **legacy_entry,
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    existing = dataset_info.get(args.dataset_name)
    if existing is not None and existing not in (entry, legacy_entry):
        raise ValueError(f"Existing {args.dataset_name} registration differs: {existing}")
    dataset_info[args.dataset_name] = entry

    temporary = args.dataset_info.with_name(f".{args.dataset_info.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(dataset_info, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(args.dataset_info)
    print(f"Registered {args.dataset_name} -> {args.file_name}")


if __name__ == "__main__":
    main()
