#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing metadata file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_expected_step(state: dict, explicit_step: int | None) -> int:
    if explicit_step is not None:
        return explicit_step

    max_steps = state.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise RuntimeError(
            "Trained model metadata does not contain a positive max_steps; "
            "pass --expected-step explicitly."
        )
    return max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an evaluation model without changing it.")
    parser.add_argument("--mode", choices=("trained", "official", "base"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-step",
        type=int,
        help="Expected trained global step; defaults to trainer_state.max_steps.",
    )
    parser.add_argument("--expected-revision")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve(strict=True)
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Model index contains no weight_map entries.")

    shards = sorted(set(weight_map.values()))
    missing = [name for name in shards if not (model_dir / name).is_file()]
    empty = [name for name in shards if (model_dir / name).is_file() and (model_dir / name).stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(f"Invalid model shards: missing={missing}, empty={empty}")

    expected_keys_by_shard = {name: set() for name in shards}
    for tensor_name, shard_name in weight_map.items():
        expected_keys_by_shard[shard_name].add(tensor_name)
    for shard_name, expected_keys in expected_keys_by_shard.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"Model index/shard mismatch for {shard_name}: "
                f"missing={sorted(expected_keys - actual_keys)[:5]}, "
                f"extra={sorted(actual_keys - expected_keys)[:5]}"
            )

    config = load_json(model_dir / "config.json")
    if config.get("model_type") != "qwen3_vl":
        raise RuntimeError(f"Unexpected model_type: {config.get('model_type')!r}")
    for name in ("generation_config.json", "preprocessor_config.json", "tokenizer_config.json"):
        if not (model_dir / name).is_file():
            raise RuntimeError(f"Missing model file: {model_dir / name}")

    if args.mode == "trained":
        state = load_json(model_dir / "trainer_state.json")
        actual_step = state.get("global_step")
        expected_step = resolve_expected_step(state, args.expected_step)
        if actual_step != expected_step:
            raise RuntimeError(
                f"Incomplete or unexpected trained model: global_step={actual_step}, "
                f"expected={expected_step}"
            )
        for name in ("train_results.json",):
            if not (model_dir / name).is_file():
                raise RuntimeError(f"Missing trained-model metadata: {model_dir / name}")

    if args.mode == "base":
        if args.expected_revision:
            marker = model_dir / ".colt_verified_revision"
            actual_revision = marker.read_text().strip() if marker.is_file() else None
            if actual_revision != args.expected_revision:
                raise RuntimeError(
                    f"Base model revision mismatch: expected={args.expected_revision}, "
                    f"found={actual_revision}"
                )
        forbidden_parts = {
            "decoder",
            "backward_decoder",
            "prj",
            "latent_predictor",
            "pj_in",
            "pj_back",
            "pj_out",
            "alpha",
            "latent_to_decoder_scale",
        }
        unexpected = sorted(
            name for name in weight_map if forbidden_parts.intersection(name.split("."))
        )
        if unexpected:
            raise RuntimeError(f"Base model contains CoLT parameters: {unexpected[:10]}")

    total_bytes = sum((model_dir / name).stat().st_size for name in shards)
    print(
        f"Model verified: mode={args.mode} path={model_dir} "
        f"shards={len(shards)} size_gib={total_bytes / 2**30:.2f}"
    )


if __name__ == "__main__":
    main()
