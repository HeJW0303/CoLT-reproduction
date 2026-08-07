#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("datasets", nargs="+")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    missing = [dataset for dataset in args.datasets if dataset not in config]
    if missing:
        parser.error(f"datasets missing judge configuration: {', '.join(missing)}")

    selected = [config[dataset] for dataset in args.datasets]
    first = selected[0]
    if any(item != first for item in selected[1:]):
        parser.error("selected datasets require different judge configurations; run them separately")

    required = {
        "model": str,
        "wire_api": str,
        "reasoning_effort": str,
        "thinking": str,
    }
    for key, expected_type in required.items():
        if key not in first or not isinstance(first[key], expected_type):
            parser.error(f"judge configuration field {key!r} must be {expected_type.__name__}")
    if first["wire_api"] != "chat_completions":
        parser.error("DeepSeek external-judge evaluation requires wire_api=chat_completions")
    if first["thinking"] not in {"enabled", "disabled"}:
        parser.error("thinking must be enabled or disabled")
    if first["reasoning_effort"] not in {"low", "high", "max"}:
        parser.error("reasoning_effort must be low, high, or max")

    judge_args = {
        "wire_api": first["wire_api"],
        "reasoning_effort": first["reasoning_effort"],
        "thinking": {"type": first["thinking"]},
    }
    if first["model"] != "deepseek-v4-flash":
        parser.error("DeepSeek external-judge evaluation requires model=deepseek-v4-flash")
    print(first["model"])
    print(json.dumps(judge_args, separators=(",", ":"), sort_keys=True))
    print(f'{first["model"]}_{first["wire_api"]}_{first["reasoning_effort"]}')


if __name__ == "__main__":
    main()
