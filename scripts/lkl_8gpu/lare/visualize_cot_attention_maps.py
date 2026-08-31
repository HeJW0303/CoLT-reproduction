#!/usr/bin/env python3
"""Render frozen-teacher CoT-to-image attention maps for audited examples.

This is a qualitative diagnostic, not a training-data writer.  It reuses the
canonical CoLT dynamic three-step partition used by both decoder supervision
and the attention-target builder.  For each selected sample it writes a PNG
containing the original image and one overlay per latent step, together with a
JSON manifest and a browsable ``index.html``.

Run on one otherwise-idle GPU.  The 8B frozen teacher makes CPU execution
technically possible but impractically slow for a few dozen examples.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "LLaMA-Factory" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq  # noqa: E402
from llamafactory.data.template import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams.data_args import DataArguments  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    extract_think_content_robust,
    split_cot_by_dynamic_boundaries_with_metadata,
)

from build_cot_attention_targets import (  # noqa: E402
    assert_current_cot_query_tokens,
    get_teacher_attention_module,
    image_placeholder_compatibility_error,
    make_prefix_feature,
    move_tensor_inputs,
    parse_teacher_heads,
    teacher_attention_metadata,
    teacher_map_for_prefix,
    teacher_map_for_prefix_heads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-model-path",
        type=Path,
        default=Path("/data/nvme0/lkl/models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument("--template", default="qwen3_vl")
    parser.add_argument("--teacher-layer", type=int, default=18)
    parser.add_argument(
        "--teacher-heads",
        default=None,
        help="Optional comma-separated calibrated layer:query-head pairs.",
    )
    parser.add_argument(
        "--query-pool", choices=("mean", "last", "visual-mass"), default="mean"
    )
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--min-step-tokens", type=int, default=8)
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--min-maps-per-example", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--indices", default=None, help="Comma-separated tokenized row indices; overrides sampling.")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--image-min-pixels", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--translations-json",
        type=Path,
        default=None,
        help="Optional row-keyed Chinese translations for bilingual rendering.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("/data/nvme0/lkl/cache/fonts/NotoSansCJKsc-Regular.otf"),
        help="CJK-capable font used inside PNGs; HTML uses browser fonts.",
    )
    return parser.parse_args()


def select_train_split(dataset: Dataset | DatasetDict) -> Dataset:
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("Tokenized dataset must contain a train split.")
        return dataset["train"]
    return dataset


def parse_indices(raw: str | None, total: int, seed: int, max_candidates: int) -> list[int]:
    if raw:
        indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
        if not indices:
            raise ValueError("--indices did not contain any integers.")
        if len(set(indices)) != len(indices) or any(index < 0 or index >= total for index in indices):
            raise ValueError(f"--indices must be unique values in [0, {total - 1}].")
        return indices
    rng = random.Random(seed)
    return rng.sample(range(total), k=min(total, max_candidates))


def canonical_boundary_token_ids(tokenizer: Any) -> set[int]:
    return {
        token_id
        for text in ("\n", "\n\n", ".", "。", "?", "？", "!", "！", ";", "；", ":", "：", ",", "，")
        for token_id in tokenizer.encode(text, add_special_tokens=False)
    }


def configure_bilingual_font(font_path: Path, translations_enabled: bool) -> None:
    """Register a CJK font before any PNG text is measured or rendered."""
    if not translations_enabled:
        return
    if not font_path.is_file():
        raise FileNotFoundError(
            f"Bilingual PNG rendering requires a CJK font, but it is missing: {font_path}"
        )
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


def load_translations(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("rows", payload)
    if not isinstance(raw_rows, dict):
        raise ValueError("Translation JSON must be a row-keyed object or contain a 'rows' object.")
    translations: dict[int, dict[str, Any]] = {}
    for key, value in raw_rows.items():
        if not isinstance(value, dict):
            raise ValueError(f"Translation entry {key!r} must be an object.")
        row_index = int(key)
        cot_steps_zh = value.get("cot_steps_zh")
        if not isinstance(value.get("question_zh"), str) or not isinstance(cot_steps_zh, list):
            raise ValueError(f"Translation entry {key!r} needs question_zh and cot_steps_zh.")
        if len(cot_steps_zh) != 3 or not all(isinstance(text, str) for text in cot_steps_zh):
            raise ValueError(f"Translation entry {key!r} must contain exactly three Chinese CoT steps.")
        translations[row_index] = value
    return translations


def attention_to_spatial_grid(attention: list[float], image_grid_thw: torch.Tensor, merge_size: int) -> np.ndarray:
    """Convert one image's flattened Qwen visual-token distribution to HxW."""
    if image_grid_thw.ndim != 1 or image_grid_thw.numel() != 3:
        raise ValueError(f"Expected one image grid [T,H,W], got {tuple(image_grid_thw.shape)}.")
    t, h, w = (int(value) for value in image_grid_thw.tolist())
    if h % merge_size or w % merge_size:
        raise ValueError(f"Image grid {(t, h, w)} is not divisible by spatial merge size {merge_size}.")
    expected = t * (h // merge_size) * (w // merge_size)
    if len(attention) != expected:
        raise ValueError(
            "Attention length does not match one image grid: "
            f"map={len(attention)}, expected={expected}, grid={(t, h, w)}, merge={merge_size}."
        )
    return np.asarray(attention, dtype=np.float32).reshape(t, h // merge_size, w // merge_size).mean(axis=0)


def display_normalize(spatial: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
    """Normalize using a shared per-example scale, not a per-step color scale."""
    if not np.isfinite(upper) or upper <= lower:
        return np.zeros_like(spatial)
    return np.clip((spatial - lower) / (upper - lower), 0.0, 1.0)


def normalized_entropy(attention: list[float]) -> float:
    values = np.asarray(attention, dtype=np.float64)
    values = values[values > 0]
    if values.size < 2:
        return 0.0
    values = values / values.sum()
    return float(-(values * np.log(values)).sum() / np.log(values.size))


def map_pair_metrics(left: list[float], right: list[float]) -> dict[str, float] | None:
    """Report raw-map agreement before any image-display normalization."""
    if not left or not right or len(left) != len(right):
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    pearson = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else 1.0
    midpoint = (x + y) / 2
    js_divergence = float(
        0.5
        * (
            np.sum(x * np.log((x + 1e-12) / (midpoint + 1e-12)))
            + np.sum(y * np.log((y + 1e-12) / (midpoint + 1e-12)))
        )
    )
    topk = max(1, round(0.05 * x.size))
    top_x = set(np.argpartition(x, -topk)[-topk:].tolist())
    top_y = set(np.argpartition(y, -topk)[-topk:].tolist())
    return {
        "cosine": cosine,
        "pearson": pearson,
        "js_divergence": js_divergence,
        "top5pct_overlap": len(top_x & top_y) / topk,
    }


def pairwise_metrics(maps: list[list[float]]) -> dict[str, dict[str, float] | None]:
    return {
        "step1_step2": map_pair_metrics(maps[0], maps[1]),
        "step1_step3": map_pair_metrics(maps[0], maps[2]),
        "step2_step3": map_pair_metrics(maps[1], maps[2]),
    }


def png_display_text(text: str) -> str:
    """Keep dataset text literal when Matplotlib measures wrapped captions.

    Some Matplotlib wrapping paths still invoke mathtext for ``$`` even with
    ``parse_math=False``.  A bracketed dollar marker is visually unambiguous,
    broadly supported by the available font, and prevents a malformed
    question/CoT expression aborting an entire run.  The manifest and HTML
    retain the original, unmodified text.
    """
    return text.replace("$", "[$]")


def render_example(
    *,
    output_path: Path,
    image_path: str,
    question: str,
    question_zh: str | None,
    cot_steps: list[str],
    cot_steps_zh: list[str] | None,
    maps: list[list[float]],
    grids: list[np.ndarray | None],
    row_index: int,
) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB").copy()
    valid_spatials = [spatial for spatial in grids if spatial is not None]
    common_lower = float(min(np.min(spatial) for spatial in valid_spatials))
    common_upper = float(np.quantile(np.concatenate([spatial.ravel() for spatial in valid_spatials]), 0.99))
    fig, panels = plt.subplots(
        2,
        4,
        figsize=(28, 12),
        gridspec_kw={"height_ratios": [5, 2.4]},
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.04, wspace=0.04, hspace=0.02)
    image_axes, caption_axes = panels
    image_axes[0].imshow(image)
    image_axes[0].set_title("Original image / 原始图像", fontsize=15)
    image_axes[0].axis("off")
    caption_axes[0].axis("off")
    caption_axes[0].text(
        0.5, 0.95, "Input image\n输入图像", transform=caption_axes[0].transAxes,
        ha="center", va="top", fontsize=13
    )
    for step_index, axis in enumerate(image_axes[1:]):
        axis.imshow(image)
        spatial = grids[step_index]
        if spatial is None:
            axis.text(
                0.5,
                0.5,
                "Teacher abstained / 教师弃权\n(forced canonical cut / 强制规范切分)",
                transform=axis.transAxes,
                ha="center",
                va="center",
                parse_math=False,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            title = f"Latent step {step_index + 1} / 隐式步骤 {step_index + 1}: abstain / 弃权"
        else:
            axis.imshow(
                display_normalize(spatial, lower=common_lower, upper=common_upper),
                cmap="jet",
                alpha=0.48,
                interpolation="bilinear",
                extent=(0, image.width, image.height, 0),
            )
            title = (
                f"Latent step {step_index + 1} / 隐式步骤 {step_index + 1}: "
                f"H={normalized_entropy(maps[step_index]):.3f}, "
                f"pmax={max(maps[step_index]):.4f}"
            )
        axis.set_title(title, fontsize=14)
        axis.axis("off")
        cot_text = png_display_text(cot_steps[step_index].replace("\n", " ").strip())
        caption_axis = caption_axes[step_index + 1]
        caption_axis.axis("off")
        english_caption = textwrap.fill(
            (cot_text[:300] + "…") if len(cot_text) > 300 else cot_text, width=52
        )
        chinese_caption = ""
        if cot_steps_zh is not None:
            chinese_text = png_display_text(cot_steps_zh[step_index].replace("\n", " ").strip())
            chinese_caption = "\n\n中文：" + textwrap.fill(
                (chinese_text[:180] + "…") if len(chinese_text) > 180 else chinese_text, width=30
            )
        caption_axis.text(
            0.5,
            0.95,
            f"English: {english_caption}{chinese_caption}",
            transform=caption_axis.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            wrap=True,
            parse_math=False,
        )
    question_short = png_display_text(question.replace("\n", " ").strip())
    question_lines = "English: " + textwrap.fill(
        (question_short[:420] + "…") if len(question_short) > 420 else question_short,
        width=175,
    )
    if question_zh:
        question_lines += "\n中文：" + textwrap.fill(question_zh, width=85)
    fig.suptitle(
        f"CoT-to-image attention / CoT 到图像注意力 | tokenized row / 分词行 {row_index}\n"
        + question_lines,
        fontsize=15,
        parse_math=False,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_difference_example(
    *,
    output_path: Path,
    image_path: str,
    grids: list[np.ndarray | None],
    metrics: dict[str, dict[str, float] | None],
    row_index: int,
) -> None:
    """Make raw step-to-step changes visible without per-step color rescaling."""
    if any(spatial is None for spatial in grids):
        return
    with Image.open(image_path) as opened:
        image = opened.convert("RGB").copy()
    deltas = [grids[1] - grids[0], grids[2] - grids[0], grids[2] - grids[1]]
    limit = float(np.quantile(np.abs(np.concatenate([delta.ravel() for delta in deltas])), 0.99))
    limit = max(limit, np.finfo(np.float32).eps)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.08, wspace=0.04)
    axes[0].imshow(image)
    axes[0].set_title("Original image / 原始图像", fontsize=14)
    axes[0].axis("off")
    for axis, delta, label, metric_key in zip(
        axes[1:], deltas,
        ("Step 2 − Step 1 / 步骤2−步骤1", "Step 3 − Step 1 / 步骤3−步骤1", "Step 3 − Step 2 / 步骤3−步骤2"),
        ("step1_step2", "step1_step3", "step2_step3"),
    ):
        axis.imshow(image)
        axis.imshow(
            delta,
            cmap="coolwarm",
            alpha=0.68,
            interpolation="bilinear",
            extent=(0, image.width, image.height, 0),
            vmin=-limit,
            vmax=limit,
        )
        metric = metrics[metric_key]
        axis.set_title(
            f"{label}\ncos={metric['cosine']:.3f}, JS={metric['js_divergence']:.3f}, "
            f"top5%={metric['top5pct_overlap']:.2f}",
            fontsize=13,
        )
        axis.axis("off")
    fig.suptitle(
        "Canonical CoT teacher / 规范 CoT 教师：raw conditional image-map changes / 原始条件图变化\n"
        f"red = later step gains probability / 红色表示后一步概率增加 | row / 行 {row_index}",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_index(output_dir: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        filename = html.escape(record["png"], quote=True)
        difference = record.get("difference_png")
        difference_link = (
            f"<p><a href=\"./{html.escape(difference, quote=True)}\">"
            "Open raw step-difference map / 打开原始步骤差分图</a></p>"
            if difference
            else "<p>Difference map unavailable because at least one step abstained. / 至少一步弃权，无法生成差分图。</p>"
        )
        question = html.escape(record.get("question_display_en", record["question"]))
        question_zh = html.escape(record.get("question_zh") or "（未提供中文翻译）")
        eligible = ", ".join("yes" if value else "no" for value in record["teacher_eligible"])
        eligible_zh = ", ".join("是" if value else "否" for value in record["teacher_eligible"])
        step_cards = []
        cot_steps_zh = record.get("cot_steps_zh") or ["（未提供中文翻译）"] * 3
        for step_index, (english_step, chinese_step) in enumerate(zip(record["cot_steps"], cot_steps_zh), start=1):
            step_cards.append(
                f"<section><h3>Step {step_index} / 步骤 {step_index}</h3>"
                f"<p><strong>English:</strong> {html.escape(english_step)}</p>"
                f"<p><strong>中文：</strong>{html.escape(chinese_step)}</p></section>"
            )
        rows.append(
            f'<article><h2>row / 行 {record["row_index"]} | eligible / 可用: [{eligible}] / [{eligible_zh}]</h2>'
            f"<div class=\"question\"><p><strong>English:</strong> {question}</p>"
            f"<p><strong>中文：</strong>{question_zh}</p></div>"
            f"<div class=\"steps\">{''.join(step_cards)}</div>"
            f"<a href=\"./{filename}\"><img src=\"./{filename}\" alt=\"attention map / 注意力图\"></a>"
            f"{difference_link}</article>"
        )
    body = "\n".join(rows) or "<p>No requested examples produced a usable map.</p>"
    (output_dir / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>CoT attention maps / CoT 注意力图</title>"
        "<style>body{font-family:'Noto Sans CJK SC','Microsoft YaHei','PingFang SC',system-ui,sans-serif;"
        "margin:32px;background:#f5f5f5;color:#202124;font-size:20px;line-height:1.65}"
        "article{background:white;padding:28px;margin:28px 0;border-radius:14px}"
        "img{max-width:100%;height:auto}h1{font-size:36px}h2{font-size:28px;margin:0 0 16px;color:#202124}"
        "h3{font-size:22px;margin:0 0 8px}.question{background:#f7f9fc;padding:10px 18px;border-radius:8px}"
        ".steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:18px 0}"
        ".steps section{border:1px solid #d9dee7;padding:14px;border-radius:8px}.steps p{font-size:18px}"
        "p{color:#303134;margin:8px 0}@media(max-width:1000px){.steps{grid-template-columns:1fr}}</style>"
        f"<h1>Canonical CoLT CoT-to-image attention maps / 规范 CoLT CoT 到图像注意力图</h1>{body}",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    translations = load_translations(args.translations_json)
    configure_bilingual_font(args.font_path, bool(translations))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        # A failed render can leave several PNGs but no complete manifest.
        # It is safe to deterministically re-render those named rows; a fully
        # completed directory remains protected against accidental overwrite.
        if (args.output_dir / "manifest.json").exists() or (args.output_dir / "index.html").exists():
            raise FileExistsError(f"Refusing to overwrite completed output directory: {args.output_dir}")
        print(f"Resuming incomplete visualization directory: {args.output_dir}", flush=True)
    if args.num_steps != 3:
        raise ValueError("This diagnostic is for CoLT's current fixed three-step contract; use --num-steps 3.")
    if args.num_examples <= 0 or args.min_maps_per_example <= 0 or args.min_maps_per_example > args.num_steps:
        raise ValueError("--num-examples and --min-maps-per-example must be positive and compatible.")
    if args.min_step_tokens <= 0 or not 0 < args.image_min_pixels <= args.image_max_pixels:
        raise ValueError("Invalid step-token or image-pixel limits.")

    dataset = select_train_split(load_from_disk(str(args.tokenized_path)))
    max_candidates = args.max_candidates or max(args.num_examples * 10, args.num_examples)
    candidates = parse_indices(args.indices, len(dataset), args.seed, max_candidates)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, use_fast=False, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.teacher_model_path, local_files_only=True)
    processor.image_max_pixels = args.image_max_pixels
    processor.image_min_pixels = args.image_min_pixels
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=args.template))
    os.environ["COLT_DISABLE_LATENT_REASONING"] = "1"
    teacher = AutoModelForImageTextToText.from_pretrained(
        args.teacher_model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    explicit_heads = args.teacher_heads is not None
    teacher.eval()
    attention_module = None if explicit_heads else get_teacher_attention_module(teacher, args.teacher_layer)
    layer_heads = (
        None
        if not explicit_heads and args.query_pool == "mean"
        else parse_teacher_heads(args.teacher_heads, teacher, args.teacher_layer)
    )
    collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        model=teacher,
        tokenizer=tokenizer,
        processor=processor,
        label_pad_token_id=-100,
    )
    think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
    end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    answer_token_id = tokenizer.encode("answer", add_special_tokens=False)[0]
    boundary_token_ids = canonical_boundary_token_ids(tokenizer)
    merge_size = int(teacher.config.vision_config.spatial_merge_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row_index in candidates:
        if len(records) >= args.num_examples:
            break
        row = dataset[row_index]
        image_paths = row.get("images") or []
        if len(image_paths) != 1 or not isinstance(image_paths[0], str) or not Path(image_paths[0]).is_file():
            continue
        source_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        question_ids, cot_ids, _ = extract_think_content_robust(
            source_ids, think_token_id, end_think_token_id, answer_token_id
        )
        steps, _, split_metadata = split_cot_by_dynamic_boundaries_with_metadata(
            cot_ids,
            num_steps=args.num_steps,
            eos_token_id=tokenizer.eos_token_id,
            boundary_token_ids=boundary_token_ids,
            min_step_tokens=args.min_step_tokens,
        )
        maps: list[list[float]] = []
        grids: list[np.ndarray | None] = []
        compatibility_error: str | None = None
        visible_prefix = torch.empty(0, dtype=torch.long)
        for step_index, step in enumerate(steps):
            current = step[:-1].detach().cpu()
            visible_prefix = torch.cat([visible_prefix, current])
            if not split_metadata["teacher_eligible"][step_index] or current.numel() == 0:
                maps.append([])
                grids.append(None)
                continue
            feature, prefix_start, prefix_end = make_prefix_feature(
                row,
                cot_prefix=visible_prefix,
                think_token_id=think_token_id,
                end_think_token_id=end_think_token_id,
                answer_token_id=answer_token_id,
            )
            batch = move_tensor_inputs(collator([feature]), device)
            compatibility_error = image_placeholder_compatibility_error(teacher, batch)
            if compatibility_error is not None:
                maps = [[] for _ in range(args.num_steps)]
                grids = [None for _ in range(args.num_steps)]
                break
            assert_current_cot_query_tokens(batch, current, prefix_start - current.numel(), prefix_end)
            image_grids = batch.get("image_grid_thw")
            if image_grids is None or image_grids.shape[0] != 1:
                raise ValueError("Only single-image rows are supported by this visualizer.")
            if layer_heads is None:
                attention = teacher_map_for_prefix(
                    teacher,
                    attention_module,
                    batch,
                    query_start=prefix_start - current.numel(),
                    query_end=prefix_end,
                )
            else:
                attention = teacher_map_for_prefix_heads(
                    teacher,
                    layer_heads,
                    batch,
                    query_start=prefix_start - current.numel(),
                    query_end=prefix_end,
                    query_pool=args.query_pool,
                )
            maps.append(attention)
            grids.append(attention_to_spatial_grid(attention, image_grids[0].cpu(), merge_size))
        if sum(bool(attention) for attention in maps) < args.min_maps_per_example:
            if compatibility_error is not None:
                print(f"skipped tokenized row {row_index}: {compatibility_error}", flush=True)
            continue
        filename = f"row_{row_index:06d}.png"
        difference_filename = f"row_{row_index:06d}_difference.png" if all(grid is not None for grid in grids) else None
        raw_filename = f"row_{row_index:06d}_maps.npz"
        cot_text = [tokenizer.decode(step[:-1], skip_special_tokens=True) for step in steps]
        question = tokenizer.decode(question_ids, skip_special_tokens=True)
        translation = translations.get(row_index)
        if translations and translation is None:
            raise ValueError(f"Missing bilingual translation for selected row {row_index}.")
        question_display_en = translation.get("question_en", question) if translation else question
        question_zh = translation.get("question_zh") if translation else None
        cot_steps_zh = translation.get("cot_steps_zh") if translation else None
        metrics = pairwise_metrics(maps)
        render_example(
            output_path=args.output_dir / filename,
            image_path=image_paths[0],
            question=question_display_en,
            question_zh=question_zh,
            cot_steps=cot_text,
            cot_steps_zh=cot_steps_zh,
            maps=maps,
            grids=grids,
            row_index=row_index,
        )
        if difference_filename is not None:
            render_difference_example(
                output_path=args.output_dir / difference_filename,
                image_path=image_paths[0],
                grids=grids,
                metrics=metrics,
                row_index=row_index,
            )
        np.savez_compressed(
            args.output_dir / raw_filename,
            **{f"step_{step_index + 1}": np.asarray(values, dtype=np.float32) for step_index, values in enumerate(maps)},
        )
        records.append(
            {
                "row_index": row_index,
                "png": filename,
                "difference_png": difference_filename,
                "raw_maps_npz": raw_filename,
                "image_path": image_paths[0],
                "question": question,
                "question_display_en": question_display_en,
                "question_zh": question_zh,
                "cot_steps": cot_text,
                "cot_steps_zh": cot_steps_zh,
                "split_points": split_metadata["split_points"],
                "teacher_eligible": split_metadata["teacher_eligible"],
                "map_stats": [
                    None
                    if not attention
                    else {
                        "tokens": len(attention),
                        "max_probability": float(max(attention)),
                        "normalized_entropy": normalized_entropy(attention),
                    }
                    for attention in maps
                ],
                "pairwise_metrics": metrics,
            }
        )
        print(f"rendered {len(records)}/{args.num_examples}: tokenized row {row_index}", flush=True)

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "colt_cot_attention_visualization_v1",
                "tokenized_path": str(args.tokenized_path),
                "tokenized_fingerprint": getattr(dataset, "_fingerprint", None),
                "teacher_model_path": str(args.teacher_model_path),
                **teacher_attention_metadata(
                    args.teacher_layer,
                    layer_heads,
                    explicit_heads=explicit_heads,
                ),
                "query_pool": args.query_pool,
                "translations_json": str(args.translations_json) if args.translations_json else None,
                "num_steps": args.num_steps,
                "min_step_tokens": args.min_step_tokens,
                "seed": args.seed,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_index(args.output_dir, records)
    print(f"wrote {len(records)} examples to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
