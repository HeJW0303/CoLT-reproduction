#!/usr/bin/env python3
"""Compare online CoLT rollout log-probs with latent teacher-forced recomputation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]


def build_easyr1_position_ids(processor, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    module_path = REPO_ROOT / "EasyR1/verl/models/transformers/qwen3_vl.py"
    spec = importlib.util.spec_from_file_location("colt_easyr1_qwen3_positions", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    input_ids = model_inputs["input_ids"][0]
    attention_mask = model_inputs["attention_mask"][0]
    vision_position_ids = module.get_rope_index(
        processor,
        input_ids=input_ids,
        image_grid_thw=model_inputs.get("image_grid_thw"),
        video_grid_thw=model_inputs.get("video_grid_thw"),
        attention_mask=attention_mask,
    )
    text_position_ids = torch.arange(input_ids.shape[0], dtype=input_ids.dtype).unsqueeze(0)
    return torch.cat((text_position_ids, vision_position_ids), dim=0).unsqueeze(1)


def left_pad_prompt_inputs(
    model_inputs: dict[str, torch.Tensor],
    left_pad_tokens: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    if left_pad_tokens == 0:
        return model_inputs
    if left_pad_tokens < 0:
        raise ValueError("left_pad_tokens must be non-negative.")

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    position_ids = model_inputs["position_ids"]
    input_padding = torch.full(
        (input_ids.shape[0], left_pad_tokens),
        pad_token_id,
        dtype=input_ids.dtype,
    )
    attention_padding = torch.zeros(
        (attention_mask.shape[0], left_pad_tokens),
        dtype=attention_mask.dtype,
    )
    position_padding = torch.zeros(
        (*position_ids.shape[:-1], left_pad_tokens),
        dtype=position_ids.dtype,
    )
    return {
        **model_inputs,
        "input_ids": torch.cat((input_padding, input_ids), dim=1),
        "attention_mask": torch.cat((attention_padding, attention_mask), dim=1),
        "position_ids": torch.cat((position_padding, position_ids), dim=-1),
    }


def build_text_batch_inputs(processor, prompts: list[str], left_pad_tokens: int) -> dict[str, torch.Tensor]:
    records = []
    for prompt in prompts:
        prompt_text = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        record = processor(text=[prompt_text], add_special_tokens=False, return_tensors="pt")
        record["position_ids"] = build_easyr1_position_ids(processor, record)
        records.append(record)

    target_length = max(record["input_ids"].shape[1] for record in records) + left_pad_tokens
    pad_token_id = processor.tokenizer.pad_token_id
    batched_input_ids = []
    batched_attention_masks = []
    batched_position_ids = []
    for record in records:
        padding = target_length - record["input_ids"].shape[1]
        padded = left_pad_prompt_inputs(record, padding, pad_token_id)
        batched_input_ids.append(padded["input_ids"])
        batched_attention_masks.append(padded["attention_mask"])
        batched_position_ids.append(padded["position_ids"])
    return {
        "input_ids": torch.cat(batched_input_ids, dim=0),
        "attention_mask": torch.cat(batched_attention_masks, dim=0),
        "position_ids": torch.cat(batched_position_ids, dim=1),
    }


def make_response_mask(response_ids: torch.Tensor, eos_token_id: int | None) -> torch.Tensor:
    if eos_token_id is None:
        return torch.ones_like(response_ids, dtype=torch.long)
    eos_positions = response_ids.eq(eos_token_id).long()
    return torch.logical_not((torch.cumsum(eos_positions, dim=1) - eos_positions).bool()).long()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt", default="What is 2 + 2? Return only the final answer.")
    parser.add_argument("--image", type=Path, help="Optional image for a multimodal parity check.")
    parser.add_argument(
        "--batch-secondary-prompt",
        help="Optional second text prompt. Exercises actor recomputation with unequal prompt lengths.",
    )
    parser.add_argument(
        "--left-pad-tokens",
        type=int,
        default=0,
        help="Extra EasyR1-style left padding for each prompt before actor recomputation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--num-hidden-generations", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance", type=float, default=5e-2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Invalid model checkpoint: {model_path}")
    if args.max_new_tokens <= 0 or args.num_hidden_generations < 0 or args.left_pad_tokens < 0:
        raise ValueError("max-new-tokens must be positive and num-hidden-generations must be non-negative.")
    if args.tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    os.environ["COLT_RL_MODE"] = "1"
    os.environ["COLT_RL_TOKENIZER_PATH"] = str(model_path)
    os.environ["COLT_RESPECT_GENERATION_ARGS"] = "1"

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.requires_grad_(False)
    model.eval()
    if not hasattr(model, "latent_reasoning_generate"):
        raise TypeError("Loaded model is not the vendored CoLT implementation.")
    if getattr(model, "decoder", None) is not None or getattr(model, "backward_decoder", None) is not None:
        raise RuntimeError("COLT_RL_MODE failed to omit SFT auxiliary decoders.")

    if args.image is not None and args.batch_secondary_prompt is not None:
        raise ValueError("Multimodal parity currently supports one prompt; omit --batch-secondary-prompt.")

    image = None
    if args.image is not None:
        from PIL import Image

        image_path = args.image.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        image = Image.open(image_path).convert("RGB")
        message_content = [{"type": "image"}, {"type": "text", "text": args.prompt}]
    else:
        image_path = None
        message_content = args.prompt

    if args.batch_secondary_prompt is not None:
        model_inputs = build_text_batch_inputs(
            processor,
            [args.prompt, args.batch_secondary_prompt],
            args.left_pad_tokens,
        )
    else:
        messages = [{"role": "user", "content": message_content}]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        processor_kwargs = {
            "text": [prompt_text],
            "add_special_tokens": False,
            "return_tensors": "pt",
        }
        if image is not None:
            processor_kwargs["images"] = [image]
        model_inputs = processor(**processor_kwargs)
        model_inputs["position_ids"] = build_easyr1_position_ids(processor, model_inputs)
        model_inputs = left_pad_prompt_inputs(
            model_inputs,
            args.left_pad_tokens,
            processor.tokenizer.pad_token_id,
        )
    model_inputs = {key: value.to(args.device) for key, value in model_inputs.items()}
    input_ids = model_inputs["input_ids"]
    prompt_attention_mask = model_inputs["attention_mask"]
    prompt_position_ids = model_inputs["position_ids"]
    multimodal_inputs = {
        key: value
        for key, value in model_inputs.items()
        if key not in {"input_ids", "attention_mask", "position_ids"}
    }
    if input_ids.shape[0] > 1 and multimodal_inputs:
        raise RuntimeError("Batched parity only supports text prompts.")

    response_rows = []
    online_log_prob_rows = []
    for row in range(input_ids.shape[0]):
        rollout_prompt_ids = input_ids[row : row + 1]
        rollout_prompt_attention_mask = prompt_attention_mask[row : row + 1]
        rollout_prompt_position_ids = prompt_position_ids[:, row : row + 1, :]
        sequences, online_generation_log_probs = model.latent_reasoning_generate(
            input_ids=rollout_prompt_ids,
            attention_mask=rollout_prompt_attention_mask,
            position_ids=rollout_prompt_position_ids,
            num_hidden_generations=args.num_hidden_generations,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            eos_token_id=processor.tokenizer.eos_token_id,
            return_token_log_probs=True,
            **multimodal_inputs,
        )
        responses = sequences[:, rollout_prompt_ids.shape[1] :]
        response_rows.append(responses[0])
        online_log_prob_rows.append(online_generation_log_probs[0])

    response_ids = torch.full(
        (input_ids.shape[0], args.max_new_tokens),
        processor.tokenizer.pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    online_generation_log_probs = torch.zeros(
        (input_ids.shape[0], args.max_new_tokens),
        dtype=torch.float32,
        device=input_ids.device,
    )
    for row, (response, online_log_probs) in enumerate(zip(response_rows, online_log_prob_rows)):
        response_ids[row, : response.shape[0]] = response
        online_generation_log_probs[row, : online_log_probs.shape[0]] = online_log_probs.float()

    response_mask = make_response_mask(response_ids, processor.tokenizer.eos_token_id)
    full_attention_mask = torch.cat((prompt_attention_mask, response_mask), dim=1)
    response_position_delta = torch.arange(
        1,
        response_ids.shape[1] + 1,
        device=prompt_position_ids.device,
    ).view(1, 1, -1)
    response_position_ids = prompt_position_ids[..., -1:] + response_position_delta
    full_position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
    actor_sequences = torch.cat((input_ids, response_ids), dim=1)
    rollout_scoring_output = model(
        input_ids=actor_sequences,
        attention_mask=full_attention_mask,
        position_ids=full_position_ids,
        colt_rl_response_length=response_ids.shape[1],
        num_hidden_generations=args.num_hidden_generations,
        **multimodal_inputs,
    )
    rollout_log_probs = torch.log_softmax(rollout_scoring_output.logits.float(), dim=-1).gather(
        dim=-1,
        index=response_ids.unsqueeze(-1),
    ).squeeze(-1)
    actor_scoring_output = model(
        input_ids=actor_sequences,
        attention_mask=full_attention_mask,
        position_ids=full_position_ids,
        colt_rl_response_length=response_ids.shape[1],
        num_hidden_generations=args.num_hidden_generations,
        **multimodal_inputs,
    )
    recomputed_log_probs = torch.log_softmax(actor_scoring_output.logits.float(), dim=-1).gather(
        dim=-1,
        index=response_ids.unsqueeze(-1),
    ).squeeze(-1)
    valid_response_mask = response_mask.bool()
    rollout_scoring_max_abs_diff = float(
        torch.max(torch.abs(online_generation_log_probs - rollout_log_probs)[valid_response_mask]).item()
    )
    max_abs_diff = float(torch.max(torch.abs(rollout_log_probs - recomputed_log_probs)[valid_response_mask]).item())
    result = {
        "model_path": str(model_path),
        "image": str(image_path) if image_path is not None else None,
        "batch_size": input_ids.shape[0],
        "left_pad_tokens": args.left_pad_tokens,
        "response_token_ids": response_ids.detach().cpu().tolist(),
        "response_text": [processor.tokenizer.decode(row, skip_special_tokens=True) for row in response_ids],
        "online_generation_log_probs": online_generation_log_probs.detach().cpu().tolist(),
        "rollout_log_probs": rollout_log_probs.detach().cpu().tolist(),
        "recomputed_log_probs": recomputed_log_probs.detach().cpu().tolist(),
        "rollout_scoring_max_abs_diff": rollout_scoring_max_abs_diff,
        "max_abs_diff": max_abs_diff,
        "tolerance": args.tolerance,
        "passed": max_abs_diff <= args.tolerance,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
