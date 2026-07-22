#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    json_path = args.data_root / "colt_sft_image.json"
    info_path = args.data_root / "dataset_info.json"
    if not json_path.is_file() or not info_path.is_file():
        raise FileNotFoundError("colt_sft_image.json or dataset_info.json is missing")

    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    if "onethinker_sft_image" not in info:
        raise ValueError("dataset_info.json does not register onethinker_sft_image")

    with json_path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("The training JSON is not a non-empty list")

    missing = []
    malformed = []
    image_count = 0
    for index, record in enumerate(records):
        messages = record.get("messages")
        images = record.get("images")
        if not isinstance(messages, list) or not messages:
            malformed.append(f"record {index}: invalid messages")
        else:
            for message in messages:
                if message.get("role") not in {"user", "assistant", "system"} or not isinstance(
                    message.get("content"), str
                ):
                    malformed.append(f"record {index}: invalid message")
                    break
        if not isinstance(images, list) or not images:
            malformed.append(f"record {index}: invalid images")
            continue
        for image in images:
            if not isinstance(image, str):
                malformed.append(f"record {index}: image path is not a string")
                continue
            image_count += 1
            image_path = args.data_root / image.removeprefix("./")
            if not image_path.is_file() and len(missing) < 100:
                missing.append(f"record {index}: {image}")

    if malformed:
        raise ValueError("Malformed records:\n" + "\n".join(malformed[:100]))
    if missing:
        raise FileNotFoundError("Missing image files (first 100):\n" + "\n".join(missing))

    print(f"Validated {len(records)} records and {image_count} image references; missing=0.")


if __name__ == "__main__":
    main()
