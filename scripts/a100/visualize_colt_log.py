#!/usr/bin/env python3
"""Parse and visualize a CoLT/LLaMA-Factory training log.

The log is written concurrently by eight distributed workers, so multiple
``name : value`` records can be glued onto the same physical line.  This parser
therefore searches the complete text with regular expressions and groups CoLT
component losses by the Trainer metric records that terminate each optimizer
step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "logs" / "colt_train_20260713_130119.log"
DEFAULT_OUTPUT = ROOT / "Vis"

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TRAINER_RE = re.compile(
    rf"\{{'loss':\s*({NUMBER}),\s*'grad_norm':\s*({NUMBER}),\s*"
    rf"'learning_rate':\s*({NUMBER}),\s*'epoch':\s*({NUMBER})\}}"
)
COMPONENTS = (
    "ce_loss_total",
    "forward_loss_total",
    "backward_loss_total",
    "prediction_loss_total",
)
COMPONENT_LABELS = {
    "ce_loss_total": "Answer CE",
    "forward_loss_total": "Forward alignment",
    "backward_loss_total": "Backward alignment",
    "prediction_loss_total": "Latent prediction",
}
COMPONENT_WEIGHTS = {
    "ce_loss_total": 1.0,
    "forward_loss_total": 0.2,
    "backward_loss_total": 0.2,
    "prediction_loss_total": 0.2,
}
COLORS = {
    "loss": "#0072B2",
    "ce_loss_total": "#D55E00",
    "forward_loss_total": "#009E73",
    "backward_loss_total": "#CC79A7",
    "prediction_loss_total": "#E69F00",
    "grad": "#8B1A1A",
    "lr": "#56B4E9",
    "speed": "#6A3D9A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Trainer and CoLT component metrics from a training log."
    )
    parser.add_argument(
        "--log",
        type=Path,
        nargs="+",
        default=[DEFAULT_LOG],
        help=(
            "Input log file(s), in chronological order. For resumed runs, the earlier "
            "log is automatically truncated at the checkpoint named by the next log."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for PNG/CSV/JSON outputs"
    )
    parser.add_argument(
        "--rolling-window", type=int, default=30, help="Rolling window for smoothed curves"
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution")
    return parser.parse_args()


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#FAFAFA",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#555555",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "savefig.bbox": "tight",
            "savefig.facecolor": "#FAFAFA",
        }
    )


def extract_int(text: str, pattern: str, default: int | None = None) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1).replace(",", "")) if match else default


def duration_to_seconds(value: str) -> float:
    parts = [float(piece) for piece in value.split(":")]
    total = 0.0
    for piece in parts:
        total = total * 60.0 + piece
    return total


def component_values(segment: str, name: str) -> np.ndarray:
    pattern = re.compile(rf"{re.escape(name)}\s*:\s*({NUMBER})")
    return np.asarray([float(value) for value in pattern.findall(segment)], dtype=float)


def summarize_values(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p10": math.nan,
            f"{prefix}_p90": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_count": int(finite.size),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p10": float(np.quantile(finite, 0.10)),
        f"{prefix}_p90": float(np.quantile(finite, 0.90)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def parse_trainer_and_components(
    text: str, step_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    matches = list(TRAINER_RE.finditer(text))
    trainer_rows: list[dict[str, float | int]] = []
    component_rows: list[dict[str, float | int | bool]] = []
    raw_parts: dict[str, list[np.ndarray]] = {name: [] for name in COMPONENTS}

    segment_start = text.find("***** Running training *****")
    segment_start = max(segment_start, 0)
    for local_step, match in enumerate(matches, start=1):
        step = step_offset + local_step
        segment = text[segment_start : match.start()]
        row: dict[str, float | int | bool] = {"step": step, "complete_step": True}
        for name in COMPONENTS:
            values = component_values(segment, name)
            raw_parts[name].append(values)
            row.update(summarize_values(values, name))
        row["micro_records"] = min(int(row[f"{name}_count"]) for name in COMPONENTS)
        component_rows.append(row)

        loss, grad_norm, learning_rate, epoch = map(float, match.groups())
        trainer_rows.append(
            {
                "step": step,
                "loss": loss,
                "grad_norm": grad_norm,
                "learning_rate": learning_rate,
                "epoch": epoch,
            }
        )
        segment_start = match.end()

    trailing = text[segment_start:]
    trailing_arrays = {name: component_values(trailing, name) for name in COMPONENTS}
    if any(values.size for values in trailing_arrays.values()):
        row = {"step": step_offset + len(matches) + 1, "complete_step": False}
        for name, values in trailing_arrays.items():
            raw_parts[name].append(values)
            row.update(summarize_values(values, name))
        row["micro_records"] = min(int(row[f"{name}_count"]) for name in COMPONENTS)
        component_rows.append(row)

    raw = {
        name: np.concatenate(parts) if parts else np.asarray([], dtype=float)
        for name, parts in raw_parts.items()
    }
    return pd.DataFrame(trainer_rows), pd.DataFrame(component_rows), raw


def parse_progress(text: str, expected_total_steps: int | None = None) -> pd.DataFrame:
    progress_re = re.compile(
        r"(?P<percent>\d{1,3})%\|[^\r\n]*?\|\s*(?P<step>\d+)/(?P<total>\d+)\s*"
        r"\[(?P<elapsed>[^<\]]+)<(?P<eta>[^,\]]+),\s*"
        r"(?P<rate>[0-9.]+)(?P<unit>s/it|it/s)\]"
    )
    latest: dict[int, dict[str, float | int | str]] = {}
    for match in progress_re.finditer(text):
        data = match.groupdict()
        total = int(data["total"])
        # Ignore unrelated tqdm bars such as "Loading checkpoint shards: 1/4".
        if expected_total_steps is not None and total != expected_total_steps:
            continue
        rate = float(data["rate"])
        seconds_per_step = rate if data["unit"] == "s/it" else 1.0 / rate
        latest[int(data["step"])] = {
            "step": int(data["step"]),
            "total_steps": total,
            "percent": int(data["percent"]),
            "raw_elapsed_seconds": duration_to_seconds(data["elapsed"].strip()),
            "eta_seconds": duration_to_seconds(data["eta"].strip()),
            "seconds_per_step": seconds_per_step,
        }
    frame = pd.DataFrame([latest[key] for key in sorted(latest)])
    if frame.empty:
        return frame

    # tqdm elapsed time restarts after resume. Convert it to a continuous active
    # runtime while retaining the raw per-process elapsed time in the CSV.
    offset = 0.0
    previous_raw: float | None = None
    continuous: list[float] = []
    segments: list[int] = []
    segment = 1
    for raw_elapsed in frame.raw_elapsed_seconds:
        raw_elapsed = float(raw_elapsed)
        if previous_raw is not None and raw_elapsed + 60.0 < previous_raw:
            offset += previous_raw
            segment += 1
        continuous.append(raw_elapsed + offset)
        segments.append(segment)
        previous_raw = raw_elapsed
    frame["elapsed_seconds"] = continuous
    frame["run_segment"] = segments
    return frame


def parse_metadata(
    text: str, log_paths: list[Path], resume_boundaries: list[int]
) -> dict[str, object]:
    total_steps = extract_int(text, r"Total optimization steps\s*=\s*([\d,]+)")
    examples = extract_int(text, r"Num examples\s*=\s*([\d,]+)")
    batch_size = extract_int(
        text, r"Total train batch size \(w\. parallel, distributed & accumulation\)\s*=\s*([\d,]+)"
    )
    trainable = extract_int(text, r"Number of trainable parameters\s*=\s*([\d,]+)")
    world_size = extract_int(text, r"world size:\s*(\d+)")
    checkpoint_steps = sorted(
        {int(value) for value in re.findall(r"Saving model checkpoint to .*?checkpoint-(\d+)", text)}
        | set(resume_boundaries)
    )

    warning_counts = Counter()
    warning_counts["PIL transparency"] = text.count("Palette images with Transparency")
    warning_counts["NCCL barrier mapping"] = text.count("perform barrier as devices used")
    warning_counts["Kernel cache"] = text.count("kernel cache directory could not be created")
    warning_counts["Tokenizer token update"] = text.count("new PAD/BOS/EOS tokens")
    warning_counts["trust_remote_code"] = text.count("`trust_remote_code` is not supported anymore")
    warning_counts["Traceback blocks"] = text.count("Traceback (most recent call last):")
    warning_counts["NFS cleanup error"] = text.count("Device or resource busy")

    return {
        "log_files": [str(path.resolve()) for path in log_paths],
        "log_bytes": sum(path.stat().st_size for path in log_paths),
        "log_modified": max(
            datetime.fromtimestamp(path.stat().st_mtime) for path in log_paths
        ).isoformat(timespec="seconds"),
        "resume_boundaries": resume_boundaries,
        "num_examples": examples,
        "num_epochs": extract_int(text, r"Num Epochs\s*=\s*(\d+)"),
        "world_size": world_size,
        "total_batch_size": batch_size,
        "gradient_accumulation_steps": extract_int(text, r"Gradient Accumulation steps\s*=\s*(\d+)"),
        "total_steps": total_steps,
        "trainable_parameters": trainable,
        "checkpoint_steps": checkpoint_steps,
        "training_completed": "Training completed" in text or "train_runtime" in text,
        "warning_counts": dict(warning_counts),
    }


def resume_checkpoint(text: str) -> int | None:
    matches = re.findall(r"Resuming training from .*?checkpoint-(\d+)", text)
    return int(matches[-1]) if matches else None


def merge_resumed_logs(log_paths: list[Path]) -> tuple[str, list[int]]:
    """Build one logical log, replacing overlapping post-checkpoint records.

    Example: when log B resumes from checkpoint-1000, all Trainer/component
    records after step 1000 in log A are discarded and all records from log B
    are appended. This remains correct even if log A kept running after the
    checkpoint and therefore overlaps with the resumed branch.
    """
    texts = [path.read_text(encoding="utf-8", errors="replace") for path in log_paths]
    merged = texts[0]
    boundaries: list[int] = []
    for path, next_text in zip(log_paths[1:], texts[1:]):
        boundary = resume_checkpoint(next_text)
        if boundary is None:
            raise ValueError(
                f"Cannot merge {path}: no 'Resuming training from ... checkpoint-N' record was found"
            )
        existing_records = list(TRAINER_RE.finditer(merged))
        if boundary < 1 or boundary > len(existing_records):
            raise ValueError(
                f"Cannot merge {path}: checkpoint-{boundary} is outside the "
                f"{len(existing_records)} Trainer records parsed so far"
            )
        merged = merged[: existing_records[boundary - 1].end()] + "\n" + next_text
        boundaries.append(boundary)
    return merged, boundaries


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(1, window // 5), center=True).median()


def stable_speed(progress: pd.DataFrame, window: int) -> pd.Series:
    """Mask tqdm rate warm-up after start/resume before smoothing it."""
    if progress.empty:
        return pd.Series(dtype=float)
    segment_position = progress.groupby("run_segment").cumcount()
    result = progress.seconds_per_step.where(segment_position >= window).copy()
    # tqdm can print a synthetic final average after checkpoint saving. Remove
    # such isolated rate artifacts without altering the cumulative time curve.
    for _, indices in progress.groupby("run_segment").groups.items():
        median = result.loc[indices].median()
        if pd.notna(median) and median > 0:
            valid = result.loc[indices].between(0.6 * median, 1.5 * median)
            result.loc[indices] = result.loc[indices].where(valid)
    return result


def add_checkpoint_lines(ax: plt.Axes, checkpoint_steps: Iterable[int]) -> None:
    for step in checkpoint_steps:
        ax.axvline(step, color="#666666", lw=1.0, ls="--", alpha=0.65)
        ymin, ymax = ax.get_ylim()
        ax.text(step, ymax, f" ckpt {step}", va="top", ha="left", fontsize=8, color="#555555")


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_overview(
    trainer: pd.DataFrame,
    components: pd.DataFrame,
    progress: pd.DataFrame,
    metadata: dict[str, object],
    output: Path,
    window: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle("CoLT Training Overview", fontsize=18)
    checkpoint_steps = metadata["checkpoint_steps"]

    ax = axes[0, 0]
    ax.plot(trainer.step, trainer.loss, color=COLORS["loss"], alpha=0.22, lw=0.8, label="per-step")
    ax.plot(trainer.step, rolling(trainer.loss, window), color=COLORS["loss"], lw=2.0, label=f"rolling median ({window})")
    ax.set(title="Trainer loss", xlabel="Optimizer step", ylabel="Loss")
    ax.legend(frameon=False)
    add_checkpoint_lines(ax, checkpoint_steps)

    ax = axes[0, 1]
    for name in COMPONENTS:
        column = f"{name}_mean"
        values = components[column].clip(lower=1e-8)
        ax.plot(components.step, rolling(values, window), lw=1.8, color=COLORS[name], label=COMPONENT_LABELS[name])
    ax.set_yscale("log")
    ax.set(title="CoLT component losses (step mean)", xlabel="Optimizer step", ylabel="Loss (log scale)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    add_checkpoint_lines(ax, checkpoint_steps)

    ax = axes[1, 0]
    positive_grad = trainer.grad_norm.clip(lower=1e-8)
    ax.plot(trainer.step, positive_grad, color=COLORS["grad"], alpha=0.3, lw=0.8)
    ax.plot(trainer.step, rolling(positive_grad, window), color=COLORS["grad"], lw=1.8)
    ax.set_yscale("log")
    ax.set(title="Gradient norm", xlabel="Optimizer step", ylabel="L2 norm (log scale)")
    add_checkpoint_lines(ax, checkpoint_steps)

    ax = axes[1, 1]
    ax.plot(trainer.step, trainer.learning_rate, color=COLORS["lr"], lw=2.0)
    peak_idx = int(trainer.learning_rate.idxmax())
    ax.scatter([trainer.loc[peak_idx, "step"]], [trainer.loc[peak_idx, "learning_rate"]], color="#222222", s=28, zorder=3)
    ax.annotate(
        f"peak {trainer.loc[peak_idx, 'learning_rate']:.2e}\nstep {int(trainer.loc[peak_idx, 'step'])}",
        (trainer.loc[peak_idx, "step"], trainer.loc[peak_idx, "learning_rate"]),
        xytext=(12, -35), textcoords="offset points", fontsize=8,
    )
    ax.set(title="Learning-rate schedule", xlabel="Optimizer step", ylabel="Learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    add_checkpoint_lines(ax, checkpoint_steps)

    ax = axes[2, 0]
    if not progress.empty:
        speed = stable_speed(progress, window)
        ax.plot(progress.step, speed, color=COLORS["speed"], alpha=0.25, lw=0.8)
        ax.plot(progress.step, rolling(speed, window), color=COLORS["speed"], lw=1.8, label="seconds / step")
        batch_size = metadata.get("total_batch_size") or 1
        throughput = batch_size / speed
        twin = ax.twinx()
        twin.plot(progress.step, rolling(throughput, window), color=COLORS["forward_loss_total"], lw=1.5, label="samples / second")
        twin.set_ylabel("Samples / second")
        lines = ax.get_lines()[-1:] + twin.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")
    ax.set(title="Training speed", xlabel="Optimizer step", ylabel="Seconds / step")
    add_checkpoint_lines(ax, checkpoint_steps)

    ax = axes[2, 1]
    total_steps = int(metadata.get("total_steps") or max(trainer.step))
    last_step = int(trainer.step.max())
    ax.barh(["Run"], [total_steps], color="#D9D9D9", height=0.42)
    ax.barh(["Run"], [last_step], color=COLORS["loss"], height=0.42)
    ax.axvline(last_step, color="#222222", lw=1)
    pct = 100.0 * last_step / total_steps
    status = "complete" if metadata["training_completed"] else "log snapshot / still incomplete"
    ax.text(last_step / 2, 0, f"{last_step} / {total_steps} ({pct:.1f}%)", ha="center", va="center", color="white", fontweight="bold")
    ax.text(total_steps, 0.28, status, ha="right", va="bottom", fontsize=9)
    ax.set(title="Recorded progress", xlabel="Optimizer step", xlim=(0, total_steps * 1.02))
    ax.grid(axis="y", visible=False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output / "01_training_overview.png", dpi)


def plot_component_details(components: pd.DataFrame, output: Path, window: int, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("CoLT Component Loss Details", fontsize=18)
    for ax, name in zip(axes.flat, COMPONENTS):
        step = components.step
        median = components[f"{name}_median"]
        mean = components[f"{name}_mean"]
        p10 = components[f"{name}_p10"]
        p90 = components[f"{name}_p90"]
        ax.fill_between(step, p10, p90, color=COLORS[name], alpha=0.17, label="micro-record p10-p90")
        ax.plot(step, rolling(mean, window), color=COLORS[name], lw=1.9, label=f"mean, rolling {window}")
        ax.plot(step, rolling(median, window), color="#222222", lw=1.2, ls="--", label=f"median, rolling {window}")
        if name != "forward_loss_total":
            ax.set_yscale("log")
            ax.set_ylabel("Loss (log scale)")
        else:
            ax.set_ylabel("Loss")
        ax.set_title(COMPONENT_LABELS[name])
        ax.set_xlabel("Optimizer step")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output / "02_component_loss_details.png", dpi)


def plot_distributions(raw: dict[str, np.ndarray], output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Raw CoLT Loss Distributions", fontsize=18)
    for ax, name in zip(axes.flat, COMPONENTS):
        values = raw[name]
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        upper = float(np.quantile(finite, 0.995))
        shown = finite[finite <= upper]
        ax.hist(shown, bins=80, color=COLORS[name], alpha=0.78, edgecolor="none")
        ax.axvline(np.median(finite), color="#222222", lw=1.5, ls="--", label=f"median {np.median(finite):.3g}")
        ax.axvline(np.mean(finite), color="#666666", lw=1.2, ls=":", label=f"mean {np.mean(finite):.3g}")
        ax.set_yscale("log")
        ax.set(title=f"{COMPONENT_LABELS[name]} (<= p99.5)", xlabel="Loss", ylabel="Count (log scale)")
        ax.legend(frameon=False, fontsize=8)
        ax.text(
            0.98, 0.95,
            f"n={finite.size:,}\nmin={finite.min():.3g}\np90={np.quantile(finite, .9):.3g}\nmax={finite.max():.3g}",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output / "03_component_distributions.png", dpi)


def plot_performance(progress: pd.DataFrame, metadata: dict[str, object], output: Path, window: int, dpi: int) -> None:
    if progress.empty:
        return
    clean = progress[progress.step >= 10].copy()
    clean["seconds_per_step"] = stable_speed(progress, window).loc[clean.index]
    batch_size = int(metadata.get("total_batch_size") or 1)
    clean["samples_per_second"] = batch_size / clean.seconds_per_step

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Runtime and Throughput", fontsize=18)
    plots = (
        ("seconds_per_step", "Seconds per optimizer step", "Seconds"),
        ("samples_per_second", "Effective sample throughput", "Samples / second"),
        ("elapsed_seconds", "Cumulative active runtime", "Hours"),
        ("eta_seconds", "Reported remaining time", "Hours"),
    )
    for ax, (column, title, ylabel) in zip(axes.flat, plots):
        values = clean[column] / 3600.0 if column in {"elapsed_seconds", "eta_seconds"} else clean[column]
        ax.plot(clean.step, values, color=COLORS["speed"], alpha=0.22, lw=0.8)
        ax.plot(clean.step, rolling(values, window), color=COLORS["speed"], lw=1.8)
        ax.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output / "04_runtime_throughput.png", dpi)


def plot_correlations(
    trainer: pd.DataFrame, components: pd.DataFrame, progress: pd.DataFrame, output: Path, dpi: int
) -> None:
    columns = ["step", "loss", "grad_norm", "learning_rate"]
    merged = trainer[columns].copy()
    component_cols = [f"{name}_mean" for name in COMPONENTS]
    merged = merged.merge(components[["step", *component_cols]], on="step", how="left")
    if not progress.empty:
        progress_for_corr = progress[["step"]].copy()
        progress_for_corr["seconds_per_step"] = stable_speed(progress, 30)
        merged = merged.merge(progress_for_corr, on="step", how="left")
    labels = {
        "loss": "Trainer loss",
        "grad_norm": "Grad norm",
        "learning_rate": "Learning rate",
        "ce_loss_total_mean": "Answer CE",
        "forward_loss_total_mean": "Forward align",
        "backward_loss_total_mean": "Backward align",
        "prediction_loss_total_mean": "Prediction",
        "seconds_per_step": "Sec / step",
    }
    metric_columns = [column for column in labels if column in merged]
    corr = merged[metric_columns].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    display = [labels[column] for column in metric_columns]
    ax.set_xticks(range(len(display)), display, rotation=40, ha="right")
    ax.set_yticks(range(len(display)), display)
    for row in range(len(display)):
        for col in range(len(display)):
            value = corr.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > 0.55 else "#222222")
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_title("Step-Level Metric Correlations")
    fig.tight_layout()
    save_figure(fig, output / "05_metric_correlations.png", dpi)
    corr.rename(index=labels, columns=labels).to_csv(output / "metric_correlations.csv", encoding="utf-8")


def plot_events(metadata: dict[str, object], output: Path, dpi: int) -> None:
    counts = metadata["warning_counts"]
    names = [name for name, count in counts.items() if count]
    values = [counts[name] for name in names]
    if not names:
        return
    order = np.argsort(values)
    names = [names[index] for index in order]
    values = [values[index] for index in order]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(names, values, color="#7A7A7A")
    ax.bar_label(bars, padding=4)
    ax.set(title="Repeated Log Warnings and Exceptions", xlabel="Occurrences", xlim=(0, max(values) * 1.12))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, output / "06_log_events.png", dpi)


def export_raw_components(raw: dict[str, np.ndarray], output: Path) -> None:
    # Each column is independently ordered because distributed stdout can glue
    # records together; do not interpret a row as a guaranteed per-sample tuple.
    max_length = max((len(values) for values in raw.values()), default=0)
    with (output / "raw_component_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_index", *COMPONENTS])
        for index in range(max_length):
            writer.writerow(
                [index + 1]
                + [raw[name][index] if index < len(raw[name]) else "" for name in COMPONENTS]
            )


def finite_number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_summary(
    trainer: pd.DataFrame,
    components: pd.DataFrame,
    progress: pd.DataFrame,
    raw: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> dict[str, object]:
    total_steps = int(metadata.get("total_steps") or len(trainer))
    last_step = int(trainer.step.max())
    complete_components = components[components.complete_step.astype(bool)]
    last_window = min(100, len(trainer))
    summary: dict[str, object] = {
        **metadata,
        "recorded_optimizer_steps": last_step,
        "progress_percent": 100.0 * last_step / total_steps,
        "remaining_optimizer_steps": max(0, total_steps - last_step),
        "trainer_records": len(trainer),
        "component_complete_steps": len(complete_components),
        "component_partial_step": bool((~components.complete_step.astype(bool)).any()),
        "partial_step_micro_records": int(components.loc[~components.complete_step.astype(bool), "micro_records"].max())
        if (~components.complete_step.astype(bool)).any()
        else 0,
        "nonstandard_component_record_steps": [
            {
                "step": int(row.step),
                "micro_records": int(row.micro_records),
            }
            for row in complete_components.itertuples()
            if int(row.micro_records) != int(metadata.get("total_batch_size") or row.micro_records)
        ],
        "latest_trainer_metrics": {
            key: finite_number(trainer.iloc[-1][key])
            for key in ("loss", "grad_norm", "learning_rate", "epoch")
        },
        "last_100_step_medians": {
            "loss": finite_number(trainer.loss.tail(last_window).median()),
            "grad_norm": finite_number(trainer.grad_norm.tail(last_window).median()),
            **{
                name: finite_number(complete_components[f"{name}_mean"].tail(last_window).median())
                for name in COMPONENTS
            },
        },
        "gradient_norm_peak": {
            "step": int(trainer.loc[trainer.grad_norm.idxmax(), "step"]),
            "value": finite_number(trainer.grad_norm.max()),
        },
        "learning_rate_peak": {
            "step": int(trainer.loc[trainer.learning_rate.idxmax(), "step"]),
            "value": finite_number(trainer.learning_rate.max()),
        },
        "raw_component_counts": {name: len(values) for name, values in raw.items()},
        "raw_component_statistics": {
            name: {
                "mean": finite_number(np.mean(values)),
                "median": finite_number(np.median(values)),
                "p10": finite_number(np.quantile(values, 0.10)),
                "p90": finite_number(np.quantile(values, 0.90)),
                "min": finite_number(np.min(values)),
                "max": finite_number(np.max(values)),
            }
            for name, values in raw.items()
            if len(values)
        },
    }
    if not progress.empty:
        stable = progress[progress.step >= 10]
        median_seconds = float(stable.seconds_per_step.tail(min(100, len(stable))).median())
        summary["runtime"] = {
            "latest_elapsed_hours": finite_number(progress.elapsed_seconds.iloc[-1] / 3600.0),
            "latest_reported_eta_hours": finite_number(progress.eta_seconds.iloc[-1] / 3600.0),
            "last_100_median_seconds_per_step": finite_number(median_seconds),
            "last_100_effective_samples_per_second": finite_number(
                (metadata.get("total_batch_size") or 1) / median_seconds
            ),
            "estimated_remaining_hours_at_recent_median": finite_number(
                (total_steps - last_step) * median_seconds / 3600.0
            ),
        }
    return summary


def write_report(summary: dict[str, object], output: Path) -> None:
    latest = summary["latest_trainer_metrics"]
    medians = summary["last_100_step_medians"]
    runtime = summary.get("runtime", {})
    warnings = summary["warning_counts"]
    state = "已完成" if summary["training_completed"] else "日志快照未训练完成"
    lines = [
        "# CoLT 训练日志可视化摘要",
        "",
        f"- 状态：{state}",
        f"- 训练进度：{summary['recorded_optimizer_steps']} / {summary['total_steps']} "
        f"({summary['progress_percent']:.2f}%)",
        f"- 数据规模：{summary['num_examples']:,} 条；全局 batch size = {summary['total_batch_size']}；"
        f"world size = {summary['world_size']}",
        f"- 最新 Trainer 指标：loss = {latest['loss']:.6g}，grad_norm = {latest['grad_norm']:.6g}，"
        f"learning_rate = {latest['learning_rate']:.6g}，epoch = {latest['epoch']:.4g}",
        f"- 最近 100 step 中位数：loss = {medians['loss']:.6g}，grad_norm = {medians['grad_norm']:.6g}",
        f"- 梯度范数峰值：{summary['gradient_norm_peak']['value']:.6g} "
        f"(step {summary['gradient_norm_peak']['step']})",
        f"- 学习率峰值：{summary['learning_rate_peak']['value']:.6g} "
        f"(step {summary['learning_rate_peak']['step']})",
    ]
    if runtime:
        lines.extend(
            [
                f"- 最近 100 step 中位耗时：{runtime['last_100_median_seconds_per_step']:.2f} 秒/step",
                f"- 按近期速度估计剩余时间：{runtime['estimated_remaining_hours_at_recent_median']:.2f} 小时",
            ]
        )
    if summary["component_partial_step"]:
        lines.append(
            f"- 尾部存在未完成的 step {summary['recorded_optimizer_steps'] + 1}："
            f"已写入 {summary['partial_step_micro_records']} / {summary['total_batch_size']} 条细粒度记录。"
        )
    short_steps = summary["nonstandard_component_record_steps"]
    if short_steps:
        details = "，".join(
            f"step {item['step']} 有 {item['micro_records']} 条" for item in short_steps
        )
        lines.append(
            f"- 完整但不足 64 条的尾 batch：{details}。这是数据集末尾的正常短 batch，不是日志缺失。"
        )
    lines.extend(
        [
            "",
            "## CoLT 四项损失（最近 100 个完整 step 的 step-mean 中位数）",
            "",
            f"- 最终答案 CE：{medians['ce_loss_total']:.6g}",
            f"- 前向对齐损失：{medians['forward_loss_total']:.6g}",
            f"- 反向对齐损失：{medians['backward_loss_total']:.6g}",
            f"- latent prediction 损失：{medians['prediction_loss_total']:.6g}",
            "",
            "## 日志事件计数",
            "",
        ]
    )
    lines.extend(f"- {name}: {count}" for name, count in warnings.items())
    lines.extend(
        [
            "",
            "## 解析说明",
            "",
            "日志由 8 个分布式进程并发写入，四项 CoLT 损失经常粘连在同一物理行。脚本采用全文正则解析，"
            "并以每条 Trainer 指标作为一个优化 step 的结束边界。常规完整优化 step 含 64 条细粒度记录"
            "（8 卡 × 8 次梯度累积），数据集尾部允许出现短 batch。`raw_component_metrics.csv` 的各列分别保持日志"
            "出现顺序，但由于 stdout 并发"
            "交错，同一行的四个值不能严格解释为同一个样本的配对记录；step 级统计不受这一点影响。",
            "",
            "训练初期数据转换阶段出现的 `SystemExit: 0` 后仍成功完成 122,179 条数据转换并进入训练，因此不能"
            "把该 traceback 单独解读为训练失败。是否完整结束以 `training_completed` 和最终 step 为准。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.rolling_window < 1:
        raise ValueError("--rolling-window must be >= 1")

    log_paths = [path.resolve() for path in args.log]
    missing = [path for path in log_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Log file(s) not found: {', '.join(map(str, missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    text, resume_boundaries = merge_resumed_logs(log_paths)
    step_offset = 0
    if len(log_paths) == 1:
        single_resume_boundary = resume_checkpoint(text)
        if single_resume_boundary is not None:
            step_offset = single_resume_boundary
            resume_boundaries = [single_resume_boundary]
    trainer, components, raw = parse_trainer_and_components(text, step_offset=step_offset)
    metadata = parse_metadata(text, log_paths, resume_boundaries)
    progress = parse_progress(text, metadata.get("total_steps"))
    if trainer.empty:
        raise RuntimeError("No Trainer metric dictionaries were found in the log")

    trainer.to_csv(args.output_dir / "trainer_metrics.csv", index=False)
    components.to_csv(args.output_dir / "component_step_metrics.csv", index=False)
    progress.to_csv(args.output_dir / "progress_metrics.csv", index=False)
    export_raw_components(raw, args.output_dir)

    summary = build_summary(trainer, components, progress, raw, metadata)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(summary, args.output_dir)

    set_plot_style()
    plot_overview(trainer, components, progress, metadata, args.output_dir, args.rolling_window, args.dpi)
    plot_component_details(components, args.output_dir, args.rolling_window, args.dpi)
    plot_distributions(raw, args.output_dir, args.dpi)
    plot_performance(progress, metadata, args.output_dir, args.rolling_window, args.dpi)
    plot_correlations(trainer, components, progress, args.output_dir, args.dpi)
    plot_events(metadata, args.output_dir, args.dpi)

    if resume_boundaries:
        print(
            "Merged resumed logs at checkpoint step(s): "
            + ", ".join(map(str, resume_boundaries))
        )
    print(f"Parsed {len(trainer):,} Trainer steps and {sum(len(v) for v in raw.values()):,} component values.")
    print(f"Wrote visualizations and data to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
