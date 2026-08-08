"""Outcome-only reward for CoLT hidden latent reasoning."""

from __future__ import annotations

import re
from typing import Any

from verl.reward_function.onethinker_reward import accuracy_reward, answer_structure_bonus, extract_answer


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_FULL_PATTERN = re.compile(r"\A\s*<answer>\s*(.*?)\s*</answer>\s*\Z", re.DOTALL | re.IGNORECASE)


def parse_hidden_reasoning_answer(response: str) -> tuple[str, float]:
    """Return the final answer and a format score without requiring visible thought text."""
    if not isinstance(response, str):
        raise TypeError(f"response must be a string, received {type(response).__name__}")
    stripped = response.strip()
    if not stripped:
        return "", 0.0

    full_match = ANSWER_FULL_PATTERN.fullmatch(stripped)
    if full_match is not None:
        answer = full_match.group(1).strip()
        return answer, 1.0 if answer else 0.0

    matches = ANSWER_PATTERN.findall(stripped)
    if matches:
        return matches[-1].strip(), 0.0
    if "<answer" in stripped.lower() or "</answer" in stripped.lower():
        return "", 0.0
    return stripped, 1.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("CoLT outcome reward requires worker.reward.reward_type=batch.")
    if not 0.0 <= format_weight <= 1.0:
        raise ValueError("format_weight must be in [0, 1].")

    results = []
    for item in reward_inputs:
        response = item["response"]
        ground_truth = item["ground_truth"]
        data_type = item["data_type"]
        problem_type = item["problem_type"]
        answer, format_score = parse_hidden_reasoning_answer(response)
        reference = extract_answer(ground_truth) or ground_truth
        wrapped_answer = f"<answer>{answer}</answer>" if answer else ""
        accuracy = accuracy_reward(wrapped_answer, reference, data_type, problem_type)
        structure = answer_structure_bonus(answer, reference, data_type, problem_type) if format_score else 0.0
        overall = (1.0 - format_weight) * accuracy + format_weight * format_score + structure
        results.append(
            {
                "overall": float(overall),
                "format": float(format_score),
                "accuracy": float(accuracy),
                "structure_reward": float(structure),
            }
        )
    return results
