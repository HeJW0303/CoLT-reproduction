#!/usr/bin/env python3
"""Gate experiment: is there differentiable reward signal for multi-path RL?

Before committing to multi-path latent RL, this script measures whether the
two CMPO rewards actually respond to latent-path perturbations on the final
checkpoint:

  A. Answer reward sensitivity (OneThinker rows, real labels)
     - d(answer CE) / d(h_3) via the model's ``latent_answer_grad_norm`` hook
       registered on the actual final ``latent_embd`` consumed by the answer
       decoder (all auxiliary loss weights zeroed -> pure answer CE gradient)
     - empirical answer log-prob spread: with COLT_STOCHASTIC_LATENT=1 and
       latent noise std sigma, K teacher-forced runs -> std(ce_loss_total)
  B. Grounding reward sensitivity (GQA step-grounding rows)
     - with the same path noise, K runs -> std(grounding_loss_total)
  C. Grounding <-> answer coupling (GQA rows with reconstructed labels)
     - per-sample Pearson r between answer CE and grounding loss across runs

Usage (colt env, single GPU):
  COLT_VISUAL_GROUNDING=1 COLT_STOCHASTIC_LATENT=1 \
  COLT_COMPONENT_LOG_EVERY=1 COLT_BATCH_AUX_DECODERS=1 \
  CUDA_VISIBLE_DEVICES=0 python scripts/lkl_8gpu/tools/gate_reward_discriminability.py \
      --checkpoint checkpoints/colt_paper_faithful_replay_step_grounding_30k/checkpoint-1986 \
      --out /home/dataset-local/lkl/tmp/gate_result.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import re
import statistics
import sys

import torch

from datasets import load_from_disk
from PIL import Image


TOKENIZED_CACHE = (
    "/home/dataset-local/lkl/cache/colt/replay_step_grounding_30k_tokenized"
)
ASSISTANT_MARKER = "<|im_start|>assistant"
# Must match the SFT tokenization (training yaml image_max_pixels + LLaMA-Factory defaults).
IMAGE_MAX_PIXELS = 802816
IMAGE_MIN_PIXELS = 32 * 32


def build_visual_inputs(processor, image_path: str, device: torch.device) -> dict:
    # Replicate LLaMA-Factory mm_plugin._preprocess_image so the pixel grid
    # matches the image_pad token count in the cached input_ids.
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if width * height > IMAGE_MAX_PIXELS:
        factor = math.sqrt(IMAGE_MAX_PIXELS / (width * height))
        width, height = int(width * factor), int(height * factor)
        image = image.resize((width, height))
    elif width * height < IMAGE_MIN_PIXELS:
        factor = math.sqrt(IMAGE_MIN_PIXELS / (width * height))
        width, height = int(width * factor), int(height * factor)
        image = image.resize((width, height))
    # Qwen3-VL processor requires non-None ``text`` aligned with ``images``;
    # only pixel_values / grid tensors are used here (input_ids come from cache).
    out = processor(
        text=["<image>"],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return {
        "pixel_values": out.get("pixel_values"),
        "image_grid_thw": out.get("image_grid_thw"),
        "video_pixel_values": out.get("video_pixel_values"),
        "video_grid_thw": out.get("video_grid_thw"),
    }


def assistant_start(ids: list[int], marker_ids: list[int]) -> int:
    for p in range(len(ids) - len(marker_ids) + 1):
        if ids[p : p + len(marker_ids)] == marker_ids:
            return p
    raise ValueError("assistant marker not found in tokenized row")


def make_labels_for_gqa_row(row: dict, marker_ids: list[int]) -> list[int]:
    ids = row["input_ids"]
    start = assistant_start(ids, marker_ids)
    labels = [-100] * start + ids[start:]
    return labels


def parse_component(stream: str, name: str) -> float:
    matches = re.findall(rf"{name} : ([-+0-9.eE]+)", stream)
    return float(matches[-1]) if matches else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    vy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-answer", type=int, default=60, help="OneThinker rows")
    parser.add_argument("--n-ground", type=int, default=60, help="GQA rows")
    parser.add_argument("--k", type=int, default=6, help="paths per sample per sigma")
    parser.add_argument("--sigmas", default="0.0,0.1,0.2", help="latent noise std scan")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shard-id", type=int, default=0, help="this process's shard index")
    parser.add_argument("--n-shards", type=int, default=1, help="total shards (one per GPU)")
    parser.add_argument("--out", default="/home/dataset-local/lkl/tmp/gate_result.json")
    args = parser.parse_args()

    if args.shard_id < 0 or args.n_shards < 1 or args.shard_id >= args.n_shards:
        raise ValueError("shard_id must be in [0, n_shards)")
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    if not sigmas or any(sigma < 0.0 for sigma in sigmas) or not any(sigma > 0.0 for sigma in sigmas):
        raise ValueError("--sigmas must contain non-negative values and at least one positive value")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("gate_reward_discriminability requires exactly one visible GPU.")
    device = torch.device("cuda:0")
    os.environ.setdefault("COLT_LATENT_INTERVENTION", "none")
    os.environ.setdefault("COLT_RESPECT_GENERATION_ARGS", "1")
    os.environ["COLT_STOCHASTIC_LATENT"] = "1"
    os.environ["COLT_VISUAL_GROUNDING"] = "1"
    os.environ["COLT_ANSWER_VISIBILITY"] = "full"
    os.environ["COLT_COMPONENT_LOG_EVERY"] = "1"
    # The backward probe must be d(answer CE)/d(final latent), not the
    # gradient of an optional side objective that happens to share the latent.
    # These are read while the model is constructed, so override any ambient
    # training environment before importing/loading the checkpoint.
    os.environ["COLT_ORACLE_K_PREDICTOR_ENABLED"] = "0"
    os.environ["COLT_KL_ANCHOR"] = "0"

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        trust_remote_code=True,
    )
    # Isolate the pure answer CE for the gradient hook.
    model.forward_align_weight = 0.0
    model.backward_align_weight = 0.0
    model.prediction_weight = 0.0
    model.visual_grounding_weight = 0.0
    model.oracle_k_predictor_loss_weight = 0.0
    model.kl_anchor_weight = 0.0
    loss_isolation = {
        "forward_align_weight": model.forward_align_weight,
        "backward_align_weight": model.backward_align_weight,
        "prediction_weight": model.prediction_weight,
        "visual_grounding_weight": model.visual_grounding_weight,
        "oracle_k_predictor_loss_weight": model.oracle_k_predictor_loss_weight,
        "kl_anchor_weight": model.kl_anchor_weight,
    }
    if any(weight != 0.0 for weight in loss_isolation.values()):
        raise RuntimeError(f"answer-CE loss isolation failed: {loss_isolation}")
    model.eval()

    ds = load_from_disk(TOKENIZED_CACHE)["train"]
    vis_cache: dict[int, dict] = {}
    marker_ids = processor.tokenizer(
        ASSISTANT_MARKER, add_special_tokens=False
    )["input_ids"]

    rng = random.Random(args.seed)
    answer_rows = []
    ground_rows = []
    # Reading full Arrow rows decodes image paths and every unused feature for
    # each example.  Selection depends only on these two columns, so keep the
    # original order and seed while avoiding that repeated decode cost.
    labels_column = ds["labels"]
    step_bboxes_column = ds["step_bboxes"]
    for i, (labels, step_bboxes) in enumerate(zip(labels_column, step_bboxes_column)):
        has_labels = bool(labels) and any(token != -100 for token in labels)
        has_bbox = bool(step_bboxes)
        if has_labels and not has_bbox:
            answer_rows.append(i)
        elif has_bbox and not has_labels:
            ground_rows.append(i)
    rng.shuffle(answer_rows)
    rng.shuffle(ground_rows)
    answer_rows = answer_rows[: args.n_answer][args.shard_id :: args.n_shards]
    ground_rows = ground_rows[: args.n_ground][args.shard_id :: args.n_shards]
    print(f"answer rows: {len(answer_rows)}, ground rows: {len(ground_rows)}")

    def run_forward(
        row: dict, ridx: int, labels: list[int], sigma: float, train_mode: bool, want_grad: bool = False
    ) -> tuple[float, float, float, torch.Tensor]:
        """Return (ce_loss_total, grounding_loss_total, answer_grad_norm, final_latent)."""
        model.latent_noise_std = sigma
        model.train(mode=train_mode)
        if ridx not in vis_cache:
            vis_cache[ridx] = build_visual_inputs(processor, row["images"][0], device)
        vis = vis_cache[ridx]
        input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor([row["attention_mask"]], dtype=torch.long, device=device)
        labels_t = torch.tensor([labels], dtype=torch.long, device=device)
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_t,
            colt_step_bboxes=[row["step_bboxes"]] if row["step_bboxes"] else None,
        )
        for key in ("pixel_values", "image_grid_thw", "video_pixel_values", "video_grid_thw"):
            v = vis.get(key)
            if v is not None:
                kwargs[key] = v.to(device)
        captured: dict[str, torch.Tensor] = {}

        def capture_embeds(module, args, kwargs, output):
            # The answer decode calls self.model(inputs_embeds=concat([h_3, answer])).
            # This capture is used only to compare forward values across noisy
            # paths.  Its ``[:, :1]`` view is not the graph tensor used for the
            # answer-gradient measurement below.
            embeds = kwargs.get("inputs_embeds")
            if embeds is not None:
                captured["answer_embeds"] = embeds

        handle = model.model.register_forward_hook(capture_embeds, with_kwargs=True)
        model.zero_grad(set_to_none=True)
        model.latent_answer_grad_norm = None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with torch.inference_mode(False):
                # Emulate training numerics: the training-style label forward
                # mixes float32 activations with bf16 weights outside autocast.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**kwargs)
        handle.remove()
        if "answer_embeds" not in captured:
            raise RuntimeError("answer-decoder embedding hook was not reached")
        final_latent = captured["answer_embeds"][:, :1].detach().clone()
        grad = float("nan")
        if want_grad:
            # ``latent_embd`` registers this hook immediately before it is
            # concatenated with answer token embeddings.  Calling backward on
            # the pure answer CE therefore measures the graph edge that the
            # decoder actually consumes.  Do not use autograd.grad on a view of
            # the post-concatenation tensor: that view is not necessarily a
            # graph input and can incorrectly report ``None``.
            out.loss.backward()
            if model.latent_answer_grad_norm is None:
                raise RuntimeError(
                    "final-latent answer-gradient hook was not reached; "
                    "cannot classify the answer path as bypassed"
                )
            grad = float(model.latent_answer_grad_norm)
        stream = buf.getvalue()
        ce = parse_component(stream, "ce_loss_total")
        gr = parse_component(stream, "grounding_loss_total")
        return ce, gr, grad, final_latent

    # ---- A. answer gradient (sigma=0, train mode) ----
    grads = []
    grad_used = 0
    latent_0: dict[int, torch.Tensor] = {}
    noise_reach = {f"{s:.2f}": [] for s in sigmas}
    for i, ridx in enumerate(answer_rows):
        row = ds[ridx]
        ce, gr, grad, lat = run_forward(row, ridx, row["labels"], 0.0, train_mode=True, want_grad=True)
        if grad is not None:
            grads.append(grad)
            if grad > 0:
                grad_used += 1
        print(f"[grad {i+1}/{len(answer_rows)}] grad_norm={grad if grad is not None else 'n/a'}", flush=True)
    grad_median = statistics.median(grads) if grads else float("nan")
    grad_mean = statistics.mean(grads) if grads else float("nan")
    grad_used_frac = grad_used / max(len(grads), 1)

    # ---- A/B/C. spread + coupling ----
    answer_spread = {f"{s:.2f}": [] for s in sigmas}
    ground_spread = {f"{s:.2f}": [] for s in sigmas}
    answer_spread_raw = {f"{s:.2f}": [] for s in sigmas}
    ground_spread_raw = {f"{s:.2f}": [] for s in sigmas}
    couplings = {f"{sigma:.2f}": [] for sigma in sigmas}
    for sigma in sigmas:
        key = f"{sigma:.2f}"
        for i, ridx in enumerate(ground_rows):
            row = ds[ridx]
            labels = make_labels_for_gqa_row(row, marker_ids)
            ces = []
            grs = []
            lats = []
            for r in range(args.k):
                torch.manual_seed(args.seed + ridx * 1000 + r)
                ce, gr, _, lat = run_forward(row, ridx, labels, sigma, train_mode=False)
                ces.append(ce)
                grs.append(gr)
                lats.append(lat)
            ces = [c for c in ces if not math.isnan(c)]
            grs = [g for g in grs if not math.isnan(g)]
            # noise reach: how far does sigma noise move the final latent?
            if len(lats) >= 2:
                base_norm = lats[0].float().norm().item()
                if base_norm > 0:
                    rel_moves = [
                        (lat.float() - lats[0].float()).norm().item() / base_norm for lat in lats[1:]
                    ]
                    noise_reach[key].append(statistics.mean(rel_moves))
            if len(ces) >= 2:
                stdev_ce = statistics.stdev(ces)
                answer_spread[key].append(stdev_ce)
                answer_spread_raw[key].append(stdev_ce)
            if len(grs) >= 2:
                stdev_gr = statistics.stdev(grs)
                ground_spread[key].append(stdev_gr)
                ground_spread_raw[key].append(stdev_gr)
            if len(ces) >= 3 and len(grs) >= 3:
                r_val = pearson(ces, grs)
                couplings[key].append(r_val)
            print(
                f"[{key} gqa {i+1}/{len(ground_rows)}] ce_std={answer_spread[key][-1]:.4f} "
                f"gr_std={ground_spread[key][-1]:.4f}",
                flush=True,
            )

    def median_or_nan(vals: list[float]) -> float:
        return statistics.median(vals) if vals else float("nan")

    answer_spread_med = {k: median_or_nan(v) for k, v in answer_spread.items()}
    ground_spread_med = {k: median_or_nan(v) for k, v in ground_spread.items()}
    noise_reach_med = {k: median_or_nan(v) for k, v in noise_reach.items()}
    coupling_mean_by_sigma = {
        key: statistics.mean(values) if values else float("nan")
        for key, values in couplings.items()
    }
    coupling_pos_by_sigma = {
        key: sum(1 for value in values if value > 0) / max(len(values), 1)
        for key, values in couplings.items()
    }

    # ---- decision ----
    sigma_ref = next(sigma for sigma in sigmas if sigma > 0.0)
    sigma_ref_key = f"{sigma_ref:.2f}"
    expected_delta = grad_median * sigma_ref * math.sqrt(2 / math.pi)
    # Graph connectivity and reward-scale adequacy are different questions.
    # In particular, a nonzero direct hook must never be labelled as a bypass
    # merely because it is below an arbitrary absolute norm threshold.
    answer_graph_connected = grad_used_frac > 0.0
    answer_response_present = (
        answer_spread_med[sigma_ref_key] > 1e-8
        and noise_reach_med[sigma_ref_key] > 1e-4
    )
    grounding_response_present = ground_spread_med[sigma_ref_key] > 1e-8
    coupling_mean = coupling_mean_by_sigma[sigma_ref_key]
    coupling_pos = coupling_pos_by_sigma[sigma_ref_key]
    coupling_ok = (coupling_mean > 0.1) or (coupling_pos > 0.55)
    answer_alive = answer_graph_connected and answer_response_present
    grounding_alive = grounding_response_present and coupling_ok
    if args.n_shards == 1 and not answer_graph_connected:
        decision = "SFT: answer CE is graph-disconnected from the final latent"
    elif args.n_shards == 1 and not answer_response_present:
        decision = "SFT: final latent is graph-connected but answer CE is insensitive to validated path noise"
    elif args.n_shards == 1 and not grounding_alive:
        decision = "SFT: answer path responds, but grounding reward lacks stable answer-coupled discrimination"
    elif args.n_shards == 1:
        decision = "RL candidate: answer and grounding rewards both respond under the gate"
    else:
        decision = "sharded run: use merge_gate_shards.py for the final decision"

    result = {
        "protocol": "replay_gqa_direct_final_latent_hook_v2",
        "gradient_capture": "model.latent_answer_grad_norm hook on final latent_embd",
        "loss_isolation": loss_isolation,
        "checkpoint": args.checkpoint,
        "n_answer": len(answer_rows),
        "n_ground": len(ground_rows),
        "shard_id": args.shard_id,
        "n_shards": args.n_shards,
        "k": args.k,
        "sigmas": sigmas,
        "answer_grad_norm_median": grad_median,
        "answer_grad_norm_mean": grad_mean,
        "answer_grad_used_frac": grad_used_frac,
        "answer_graph_connected": answer_graph_connected,
        "answer_response_present": answer_response_present,
        "grounding_response_present": grounding_response_present,
        "answer_response_sigma": sigma_ref,
        "expected_answer_delta_at_response_sigma": expected_delta,
        "answer_ce_std_by_sigma": answer_spread_med,
        "grounding_loss_std_by_sigma": ground_spread_med,
        "noise_reach_median_by_sigma": noise_reach_med,
        "coupling_pearson_mean": coupling_mean,
        "coupling_positive_frac": coupling_pos,
        "coupling_pearson_mean_by_sigma": coupling_mean_by_sigma,
        "coupling_positive_frac_by_sigma": coupling_pos_by_sigma,
        "per_sample_grads": grads,
        "per_sample_answer_ce_std": answer_spread_raw,
        "per_sample_grounding_loss_std": ground_spread_raw,
        "per_sample_noise_reach": noise_reach,
        "per_sample_couplings": couplings[sigma_ref_key],
        "per_sample_couplings_by_sigma": couplings,
        "answer_alive": answer_alive,
        "grounding_alive": grounding_alive,
        "coupling_ok": coupling_ok,
        "decision": decision,
    }
    print(json.dumps(result, indent=1, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
