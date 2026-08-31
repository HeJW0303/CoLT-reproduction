#!/usr/bin/env python3
"""Render comparable maps from multiple causal-occlusion candidate reports.

The reports are produced by ``validate_cot_attention_occlusion.py`` and carry
the exact raw maps, visual grid, and pass/fail result.  This renderer never
rescales candidates independently: all candidates and all three steps for one
row share one color range, so a candidate cannot look better merely because it
was contrast-stretched more aggressively.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict, load_from_disk

from visualize_cot_attention_maps import attention_to_spatial_grid, display_normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=REPORT.json",
        help="Repeat exactly three times; report must be a causal-occlusion v1 report.",
    )
    parser.add_argument("--indices", required=True, help="Comma-separated row indices to render.")
    parser.add_argument("--merge-size", type=int, default=2)
    return parser.parse_args()


def select_train_split(dataset: Dataset | DatasetDict) -> Dataset:
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("Tokenized dataset must contain a train split.")
        return dataset["train"]
    return dataset


def parse_candidates(raw: list[str]) -> list[tuple[str, dict[str, Any]]]:
    if len(raw) != 3:
        raise ValueError("Provide exactly three --candidate NAME=REPORT.json arguments.")
    result = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Invalid candidate {item!r}; expected NAME=REPORT.json.")
        name, path_text = item.split("=", 1)
        if not name or not path_text:
            raise ValueError(f"Invalid candidate {item!r}.")
        report = json.loads(Path(path_text).read_text(encoding="utf-8"))
        if report.get("format") != "colt_cot_attention_causal_occlusion_v1":
            raise ValueError(f"Candidate {name!r} is not a causal-occlusion v1 report.")
        result.append((name, report))
    return result


def row_records(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = report.get("rows")
    if not isinstance(records, list):
        raise ValueError("Candidate report has no rows list.")
    return {int(record["row_index"]): record for record in records}


def render_row(
    *,
    row_index: int,
    row: dict[str, Any],
    candidates: list[tuple[str, dict[str, Any]]],
    output_path: Path,
    merge_size: int,
) -> None:
    image_paths = row.get("images") or []
    if not image_paths:
        raise ValueError(f"Row {row_index} has no image path.")
    with Image.open(image_paths[0]) as opened:
        image = opened.convert("RGB").copy()

    candidate_rows = [(name, row_records(report)[row_index]) for name, report in candidates]
    maps: list[list[np.ndarray | None]] = []
    passed: list[list[bool | None]] = []
    for _, record in candidate_rows:
        step_maps = []
        step_passed = []
        for step in record["steps"]:
            values = step.get("attention")
            if not values:
                step_maps.append(None)
                step_passed.append(None)
                continue
            grid = torch_grid(step["image_grid_thw"])
            step_maps.append(attention_to_spatial_grid(values, grid, merge_size))
            step_passed.append(bool(step.get("passed", False)))
        maps.append(step_maps)
        passed.append(step_passed)

    valid = [spatial for row_maps in maps for spatial in row_maps if spatial is not None]
    if not valid:
        raise ValueError(f"Row {row_index} has no non-empty candidate maps.")
    all_values = np.concatenate([spatial.reshape(-1) for spatial in valid])
    lower = float(np.min(all_values))
    upper = float(np.quantile(all_values, 0.99))
    fig, axes = plt.subplots(4, 3, figsize=(18, 18), squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.02, wspace=0.02, hspace=0.10)
    for col in range(3):
        axes[0, col].imshow(image)
        axes[0, col].set_title(f"Original / 原图 | step {col + 1}", fontsize=14)
        axes[0, col].axis("off")
    for row_pos, (name, _) in enumerate(candidate_rows, start=1):
        for col in range(3):
            axis = axes[row_pos, col]
            axis.imshow(image)
            spatial = maps[row_pos - 1][col]
            if spatial is not None:
                axis.imshow(
                    display_normalize(spatial, lower=lower, upper=upper),
                    cmap="jet",
                    alpha=0.50,
                    interpolation="bilinear",
                    extent=(0, image.width, image.height, 0),
                )
            status = "abstain" if spatial is None else ("causal PASS" if passed[row_pos - 1][col] else "causal fail")
            axis.set_title(f"{name} | step {col + 1} | {status}", fontsize=13)
            axis.axis("off")
    fig.suptitle(
        f"Candidate comparison / 候选对比 | tokenized row {row_index}\n"
        "Shared per-row scale / 同一行共享色阶",
        fontsize=17,
    )
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def torch_grid(raw: Any):
    """Avoid importing torch at module import time for HTML-only inspection."""
    import torch

    grid = raw
    if isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    return torch.tensor(grid, dtype=torch.long)


def main() -> None:
    args = parse_args()
    indices = [int(item.strip()) for item in args.indices.split(",") if item.strip()]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("--indices must contain unique row indices.")
    candidates = parse_candidates(args.candidate)
    dataset = select_train_split(load_from_disk(str(args.tokenized_path)))
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("An --indices value is outside the tokenized dataset.")
    candidate_maps = [row_records(report) for _, report in candidates]
    if any(any(index not in records for index in indices) for records in candidate_maps):
        raise ValueError("Every candidate report must contain every requested row.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_rows = []
    for row_index in indices:
        output_path = args.output_dir / f"row_{row_index:06d}.png"
        render_row(
            row_index=row_index,
            row=dataset[row_index],
            candidates=candidates,
            output_path=output_path,
            merge_size=args.merge_size,
        )
        html_rows.append(
            f'<figure><img src="{html.escape(output_path.name)}" alt="row {row_index}">'
            f"<figcaption>row {row_index}</figcaption></figure>"
        )
    (args.output_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>CoT teacher candidate comparison</title>"
        "<style>body{font-family:Arial,sans-serif}figure{margin:24px 0}img{max-width:100%}</style>"
        + "\n".join(html_rows),
        encoding="utf-8",
    )
    print(f"rendered {len(indices)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
