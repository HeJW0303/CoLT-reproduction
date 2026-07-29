#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SYSTEM_PROMPT = (
    "Please answer this question based on the visual content."
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer "
    "between the <answer> and </answer> tags."
    "At the end, you must output the final answer in the format:\n"
    "<think>\nyour_thinking_process\n</think>\n\n<answer>\nyour_answer_here\n</answer>\n\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture token-level evidence for CoLT empty responses.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", default="ChartQA_TEST")
    parser.add_argument("--indices", required=True, help="Comma-separated dataset index values.")
    parser.add_argument("--generation", choices=("official", "respect-args"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--empty-response-policy", choices=("allow", "prevent"), default="allow")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    if not indices:
        raise ValueError("--indices must contain at least one integer")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)

    os.environ["COLT_EVAL_MODEL_PATH"] = str(args.model_path.resolve())
    os.environ["COLT_RESPECT_GENERATION_ARGS"] = "1" if args.generation == "respect-args" else "0"
    os.environ["COLT_RESEED_PER_SAMPLE"] = "1"
    os.environ["COLT_PREVENT_EMPTY_RESPONSE"] = "1" if args.empty_response_policy == "prevent" else "0"

    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.colt_qwen3_vl import Qwen3VLChat

    dataset = build_dataset(args.dataset)
    if dataset is None:
        raise RuntimeError(f"Unable to build dataset: {args.dataset}")

    model = Qwen3VLChat(
        model_path=str(args.model_path.resolve()),
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        temperature=0.6,
        top_k=20,
        prevent_empty_response=args.empty_response_policy == "prevent",
        max_pixels=256 * 32 * 32,
        system_prompt=SYSTEM_PROMPT,
        post_process=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    empty_count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index in indices:
            matches = dataset.data[dataset.data["index"] == index]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one row for index {index}, found {len(matches)}")
            row = matches.iloc[0]
            prompt = dataset.build_prompt(row)
            prepared = model.prepare_request(prompt, args.dataset)
            diagnostic = model.diagnose_prepared(prepared)
            record = {
                "index": index,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "empty_response_policy": args.empty_response_policy,
                **diagnostic,
            }
            empty_count += int(not diagnostic["final_response"].strip())
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"index={index} tokens={len(diagnostic['token_ids'])} "
                f"first_tokens={diagnostic['token_ids'][:8]} "
                f"empty={not diagnostic['final_response'].strip()}",
                flush=True,
            )

    print(f"completed={len(indices)} empty={empty_count} output={args.output}")


if __name__ == "__main__":
    main()
