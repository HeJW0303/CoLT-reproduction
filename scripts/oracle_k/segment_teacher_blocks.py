#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAT_MODULE_PATH = (
    REPO_ROOT / "transformers-4.57.0/src/transformers/models/qwen3_vl/oracle_k.py"
)
spec = importlib.util.spec_from_file_location("colt_oracle_k_format", FORMAT_MODULE_PATH)
oracle_k_format = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = oracle_k_format
spec.loader.exec_module(oracle_k_format)

CONTINUE_THINK = oracle_k_format.CONTINUE_THINK
OracleKFormatError = oracle_k_format.OracleKFormatError
annotate_assistant_content = oracle_k_format.annotate_assistant_content
get_assistant_cot = oracle_k_format.get_assistant_cot


SYSTEM_PROMPT = """You segment an existing chain of thought into a small number of semantically complete reasoning blocks.

Rules:
1. Return JSON only, with exactly one array field named boundary_after_units.
2. The input is an ordered list of immutable CoT units. Do not rewrite or repeat their text.
3. boundary_after_units contains the 1-based unit indices where the current block ends and the next block starts.
4. Do not include the final unit index: the final block always ends at the final unit.
5. Do not minimize or maximize K. Choose a compact but complete set of blocks so that each major reasoning subgoal is represented.
6. Insert a boundary when the reasoning changes its objective, operation, intermediate quantity, or evidence role. Do not merge distinct subgoals merely to reduce K.
7. Each block must represent a substantive and relatively complete reasoning stage, not a fixed token chunk, sentence, option, or formatting fragment.
8. Merge generic setup or transition text (for example, "Got it", "Let's see", or "Now") into the adjacent substantive block.
9. The final answer is supervised separately downstream. Merge an answer-only conclusion such as "So the answer is C" into the preceding reasoning or verification block.
10. Merge repeated attempts, self-corrections, coordinate revisions, and restatements only when they pursue the same subgoal. Repetition is not a new stage, but a genuinely new approach or subproblem is.
11. For multiple-choice reasoning, evidence extraction and option comparison may be separate stages, but checking similar options one by one is normally one comparison stage.
12. For grounding tasks, target identification, box localization, positive-point selection, negative-point selection, and consistency validation are distinct stages when they are substantively present. Merge only repeated refinements within the same stage.
13. For mathematics and statistics, preserve distinct setup/modeling, independent subcalculations or quantities, synthesis/comparison, and substantive verification stages when present.
14. Calibration: short direct reasoning commonly needs 1-2 blocks; ordinary multi-step reasoning commonly needs 2-4; long reasoning with several real subgoals commonly needs 3-6. These are semantic guides, not quotas.
15. A long CoT with several different operations or intermediate results should not be compressed into one or two blocks solely because its prose is repetitive.
16. Create a separate verification or conclusion block only when it contains substantive new reasoning, synthesis, or error checking.
17. Return strictly increasing unique integers and keep the resulting block count within the requested range.
"""

UNIT_BOUNDARY_PATTERN = re.compile(r"(?:[。！？；]|[.!?;](?=\s|$)|\n+)")
USER_PROMPT_VERSION = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_assistant_message(record: dict[str, Any]) -> tuple[int, str, str]:
    matches = []
    for index, message in enumerate(record.get("messages", [])):
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, str) and "<think>" in content:
            matches.append((index, content))
    if len(matches) != 1:
        raise OracleKFormatError(f"Expected exactly one assistant <think> message, found {len(matches)}")
    message_index, content = matches[0]
    return message_index, content, get_assistant_cot(content)


def collect_text_context(record: dict[str, Any], assistant_index: int) -> str:
    context = []
    for message in record.get("messages", [])[:assistant_index]:
        content = message.get("content")
        if isinstance(content, str) and content:
            context.append({"role": message.get("role", "unknown"), "content": content})
    return json.dumps(context, ensure_ascii=False)


def split_cot_units(cot: str) -> list[str]:
    if not cot:
        raise OracleKFormatError("Cannot segment an empty CoT")
    units = []
    start = 0
    for match in UNIT_BOUNDARY_PATTERN.finditer(cot):
        if match.group() == ".":
            line_start = cot.rfind("\n", 0, match.start()) + 1
            if cot[line_start : match.start()].strip().isdigit():
                continue
        end = match.end()
        while end < len(cot) and cot[end].isspace():
            end += 1
        if end > start:
            units.append(cot[start:end])
            start = end
    if start < len(cot):
        units.append(cot[start:])
    if not units or "".join(units) != cot or any(unit == "" for unit in units):
        raise OracleKFormatError("Internal CoT unit splitting was not reversible")
    return units


def build_segmented_cot(
    units: list[str],
    boundary_after_units: list[int],
    min_k: int,
    max_k: int,
) -> str:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in boundary_after_units):
        raise OracleKFormatError("Teacher boundaries must be integers")
    if boundary_after_units != sorted(set(boundary_after_units)):
        raise OracleKFormatError("Teacher boundaries must be strictly increasing and unique")
    if any(value < 1 or value >= len(units) for value in boundary_after_units):
        raise OracleKFormatError(f"Teacher boundaries must be in [1, {len(units) - 1}]")

    k = len(boundary_after_units) + 1
    if not min_k <= k <= max_k:
        raise OracleKFormatError(f"Teacher K={k} is outside [{min_k}, {max_k}]")

    blocks = []
    start = 0
    for end in boundary_after_units + [len(units)]:
        blocks.append("".join(units[start:end]))
        start = end
    segmented_cot = CONTINUE_THINK.join(blocks)
    if segmented_cot.replace(CONTINUE_THINK, "") != "".join(units):
        raise OracleKFormatError("Boundary reconstruction changed the original CoT")
    return segmented_cot


def normalize_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(content[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Teacher response is not a JSON object")
    return payload


def request_segmentation(
    units: list[str],
    text_context: str,
    model: str,
    endpoint: str,
    api_key: str,
    min_k: int,
    max_k: int,
    timeout: float,
    max_output_tokens: int,
    reasoning_effort: str | None,
    use_response_format: bool,
) -> list[int]:
    effective_max_k = min(max_k, len(units))
    if min_k > effective_max_k:
        raise OracleKFormatError(
            f"CoT has only {len(units)} candidate units, fewer than requested min_k={min_k}"
        )
    user_prompt = (
        f"Segment the following {len(units)} immutable CoT units into {min_k} to "
        f"{effective_max_k} semantic blocks.\n"
        "The problem context is read-only and may contain an <image> placeholder; no image is provided.\n"
        "Choose a compact but complete set of major semantic stages; neither minimize nor maximize the block count.\n"
        "Return boundary_after_units only. Example: [2, 5] means units 1-2, 3-5, and 6-final.\n"
        f"problem_context={text_context}\n"
        f"units={json.dumps(units, ensure_ascii=False)}"
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_output_tokens,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    content = response_payload["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Teacher response content must be a string")
    teacher_payload = parse_json_response(content)
    if set(teacher_payload) != {"boundary_after_units"}:
        raise ValueError("Teacher JSON must contain only boundary_after_units")
    boundaries = teacher_payload["boundary_after_units"]
    if not isinstance(boundaries, list):
        raise ValueError("Teacher boundary_after_units must be an array")
    return boundaries


def segment_record(index: int, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    message_index, _, cot = find_assistant_message(record)
    text_context = collect_text_context(record, message_index)
    units = split_cot_units(cot)
    if len(units) == 1:
        if args.min_k > 1:
            raise OracleKFormatError(f"record {index}: one candidate unit cannot satisfy min_k={args.min_k}")
        return {
            "type": "result",
            "index": index,
            "message_index": message_index,
            "boundary_after_units": [],
            "k": 1,
        }
    last_error = None
    for attempt in range(args.max_retries + 1):
        try:
            boundaries = request_segmentation(
                units=units,
                text_context=text_context,
                model=args.model,
                endpoint=args.endpoint,
                api_key=args.api_key,
                min_k=args.min_k,
                max_k=args.max_k,
                timeout=args.timeout,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
                use_response_format=not args.no_response_format,
            )
            build_segmented_cot(
                units,
                boundaries,
                min_k=args.min_k,
                max_k=args.max_k,
            )
            return {
                "type": "result",
                "index": index,
                "message_index": message_index,
                "boundary_after_units": boundaries,
                "k": len(boundaries) + 1,
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as error:
            last_error = error
            if attempt < args.max_retries:
                time.sleep(args.retry_backoff * (2**attempt))
    raise RuntimeError(f"record {index}: {type(last_error).__name__}: {last_error}")


def append_state(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def load_state(path: Path, expected_meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line or json.loads(first_line) != expected_meta:
            raise ValueError(f"State metadata does not match this run: {path}")
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") == "result":
                results[int(payload["index"])] = payload
    return results


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reversible Think-in-Blocks Oracle-K annotations.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("ORACLE_K_TEACHER_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default=os.environ.get("ORACLE_K_REASONING_EFFORT") or None,
    )
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--failures-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--state-flush-every", type=int, default=100)
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or ORACLE_K_TEACHER_MODEL is required")
    if not args.base_url:
        parser.error("--base-url or OPENAI_BASE_URL is required")
    if args.min_k < 1 or args.max_k < args.min_k:
        parser.error("Require 1 <= min-k <= max-k")
    if args.start < 0 or args.limit is not None and args.limit < 1:
        parser.error("--start must be non-negative and --limit must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_retries < 0 or args.retry_backoff < 0:
        parser.error("--max-retries and --retry-backoff must be non-negative")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if args.state_flush_every < 1:
        parser.error("--state-flush-every must be positive")
    args.endpoint = normalize_endpoint(args.base_url)
    args.api_key = os.environ.get(args.api_key_env, "")
    args.state_file = args.state_file or args.output.with_suffix(args.output.suffix + ".state.jsonl")
    args.failures_file = args.failures_file or args.output.with_suffix(args.output.suffix + ".failures.jsonl")
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.state_file.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"State already exists; use --resume: {args.state_file}")

    with args.input.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("Input dataset must be a non-empty JSON list")

    stop = len(records) if args.limit is None else min(len(records), args.start + args.limit)
    selected_indices = list(range(args.start, stop))
    if not selected_indices:
        raise ValueError("Selected range is empty")

    meta = {
        "type": "meta",
        "format_version": 5,
        "input_sha256": sha256_file(args.input),
        "start": args.start,
        "stop": stop,
        "model": args.model,
        "endpoint": args.endpoint,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "user_prompt_version": USER_PROMPT_VERSION,
        "unit_boundary_pattern": UNIT_BOUNDARY_PATTERN.pattern,
        "reasoning_effort": args.reasoning_effort,
        "min_k": args.min_k,
        "max_k": args.max_k,
    }
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    if args.state_file.exists() and args.resume:
        completed = load_state(args.state_file, meta)
    else:
        if args.state_file.exists():
            args.state_file.unlink()
        append_state(args.state_file, meta)
        completed = {}

    pending = [index for index in selected_indices if index not in completed]
    failures = []
    print(f"Selected={len(selected_indices)} completed={len(completed)} pending={len(pending)}")
    processed = 0
    unsynced_results = 0
    submission_window = max(args.workers * 4, 1)
    with args.state_file.open("a", encoding="utf-8") as state_file, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        for window_start in range(0, len(pending), submission_window):
            window = pending[window_start : window_start + submission_window]
            futures = {executor.submit(segment_record, index, records[index], args): index for index in window}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    payload = future.result()
                    completed[index] = payload
                    state_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    unsynced_results += 1
                except Exception as error:
                    failures.append({"index": index, "error": str(error)})
                processed += 1
                if processed % 10 == 0 or processed == len(pending):
                    print(
                        f"Processed={processed}/{len(pending)} "
                        f"successes={len(completed)} failures={len(failures)}"
                    )
                if unsynced_results >= args.state_flush_every:
                    state_file.flush()
                    os.fsync(state_file.fileno())
                    unsynced_results = 0
        state_file.flush()
        os.fsync(state_file.fileno())

    if failures:
        failures_temporary = args.failures_file.with_name(f".{args.failures_file.name}.tmp")
        args.failures_file.parent.mkdir(parents=True, exist_ok=True)
        with failures_temporary.open("w", encoding="utf-8") as file:
            for failure in failures:
                file.write(json.dumps(failure, ensure_ascii=False) + "\n")
        failures_temporary.replace(args.failures_file)
        raise RuntimeError(
            f"{len(failures)} records failed. Successful records remain in {args.state_file}; rerun with --resume."
        )
    if args.failures_file.exists():
        args.failures_file.unlink()

    output_records = []
    for index in selected_indices:
        result = completed[index]
        record = copy.deepcopy(records[index])
        message_index, content, cot = find_assistant_message(record)
        if message_index != result["message_index"]:
            raise RuntimeError(f"record {index}: assistant message index changed during reconstruction")
        segmented_cot = build_segmented_cot(
            split_cot_units(cot),
            result["boundary_after_units"],
            min_k=args.min_k,
            max_k=args.max_k,
        )
        record["messages"][message_index]["content"] = annotate_assistant_content(
            content,
            segmented_cot,
            min_k=args.min_k,
            max_k=args.max_k,
        )
        output_records.append(record)
    write_json_atomic(args.output, output_records)
    print(f"Wrote {len(output_records)} Oracle-K records to {args.output}")


if __name__ == "__main__":
    main()
