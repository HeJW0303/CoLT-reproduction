#!/usr/bin/env python3
"""Gate a latent-only smoke checkpoint using mid-checkpoint intervention JSON.

The smoke checkpoint must show both zero and skip latent interventions reduce
ChartQA accuracy by more than the configured threshold. This is a stronger
signal than the training-only answer_grad_norm metric.

Usage (colt env):
  python scripts/lkl_8gpu/tools/check_latent_only_smoke_gate.py \
      --input /home/dataset-local/lkl/tmp/causal_latent_only_smoke_intervention.json \
      --min-drop 5.0
"""

from __future__ import annotations

import argparse
import json


def accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    return 100.0 * sum(bool(record["hit"]) for record in records) / len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--min-drop", type=float, default=5.0)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        results = json.load(f)

    missing = {"none", "zero", "skip"} - set(results)
    if missing:
        raise SystemExit(f"Intervention result is missing modes: {sorted(missing)}")

    scores = {mode: accuracy(results[mode]) for mode in ("none", "zero", "skip")}
    drop_zero = scores["none"] - scores["zero"]
    drop_skip = scores["none"] - scores["skip"]

    print(
        f"latent_only_smoke_gate none={scores['none']:.2f} "
        f"zero={scores['zero']:.2f} (drop={drop_zero:+.2f}) "
        f"skip={scores['skip']:.2f} (drop={drop_skip:+.2f}) "
        f"threshold={args.min_drop:+.2f}"
    )

    if drop_zero <= args.min_drop:
        raise SystemExit(
            f"Gate failed: zero intervention dropped only {drop_zero:+.2f} points "
            f"(required > {args.min_drop:+.2f})."
        )
    if drop_skip <= args.min_drop:
        raise SystemExit(
            f"Gate failed: skip intervention dropped only {drop_skip:+.2f} points "
            f"(required > {args.min_drop:+.2f})."
        )
    print("latent_only_smoke_gate PASSED")


if __name__ == "__main__":
    main()
