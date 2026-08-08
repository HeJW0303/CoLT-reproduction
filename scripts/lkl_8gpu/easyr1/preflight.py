#!/usr/bin/env python3
"""Audit the local contract required for a correct CoLT/EasyR1 integration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


REQUIRED_DATASET_FIELDS = {
    "problem",
    "answer",
    "data_type",
    "problem_type",
    "problem_id",
}
REQUIRED_MODEL_WEIGHT_PREFIXES = ("alpha", "prj.")
RUNTIME_MODULES = {
    "accelerate": "accelerate",
    "codetiming": "codetiming",
    "datasets": "datasets",
    "einops": "einops",
    "flash-attn": "flash_attn",
    "math-verify": "math_verify",
    "mathruler": "mathruler",
    "omegaconf": "omegaconf",
    "Pillow": "PIL",
    "pylatexenc": "pylatexenc",
    "pyarrow": "pyarrow",
    "qwen-vl-utils": "qwen_vl_utils",
    "ray[default]": "ray",
    "rouge-score": "rouge_score",
    "tensordict": "tensordict",
    "torch": "torch",
    "torchdata": "torchdata",
    "transformers": "transformers",
}


@dataclass(frozen=True)
class Finding:
    code: str
    message: str


@dataclass
class AuditReport:
    facts: dict[str, Any] = field(default_factory=dict)
    blockers: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers

    def add_blocker(self, code: str, message: str) -> None:
        self.blockers.append(Finding(code, message))

    def add_warning(self, code: str, message: str) -> None:
        self.warnings.append(Finding(code, message))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "facts": self.facts,
            "blockers": [asdict(item) for item in self.blockers],
            "warnings": [asdict(item) for item in self.warnings],
        }


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"File is not valid UTF-8: {path}: {error}") from error


def read_upstream_commit(easyr1_root: Path) -> str | None:
    provenance_path = easyr1_root / "UPSTREAM.md"
    if not provenance_path.is_file():
        return None
    match = re.search(r"^- Upstream commit: `([0-9a-f]{40})`$", read_text(provenance_path), re.MULTILINE)
    return match.group(1) if match is not None else None


def audit_easyr1(easyr1_root: Path, report: AuditReport) -> None:
    report.facts["easyr1_root"] = str(easyr1_root)
    if not easyr1_root.is_dir():
        report.add_blocker("EASYR1_ROOT_MISSING", f"EasyR1 root does not exist: {easyr1_root}")
        return

    required_paths = {
        "worker": easyr1_root / "verl/workers/fsdp_workers.py",
        "actor": easyr1_root / "verl/workers/actor/dp_actor.py",
        "reward": easyr1_root / "verl/workers/reward/function.py",
        "dataset": easyr1_root / "verl/utils/dataset.py",
        "prompt": easyr1_root / "verl/utils/reasoning_prompt.py",
        "main": easyr1_root / "verl/trainer/main.py",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        report.add_blocker("EASYR1_SOURCE_INCOMPLETE", "Missing EasyR1 files: " + ", ".join(missing))
        return

    upstream_commit = read_upstream_commit(easyr1_root)
    report.facts["easyr1_upstream_commit"] = upstream_commit
    if upstream_commit is None:
        report.add_warning(
            "EASYR1_PROVENANCE_MISSING",
            "EasyR1/UPSTREAM.md does not contain a fixed 40-character upstream commit.",
        )
    worker_source = read_text(required_paths["worker"])
    actor_source = read_text(required_paths["actor"])
    reward_source = read_text(required_paths["reward"])
    dataset_source = read_text(required_paths["dataset"])
    prompt_source = read_text(required_paths["prompt"])

    has_colt_rollout = (
        (easyr1_root / "verl/workers/rollout/colt_transformers_rollout.py").is_file()
        and (easyr1_root / "verl/workers/sharding_manager/fsdp_transformers.py").is_file()
        and "self.config.rollout.name == \"colt_transformers\"" in worker_source
        and "FSDPTransformersShardingManager" in worker_source
    )
    vllm_is_fixed = not has_colt_rollout and (
        "from .rollout import vLLMRollout" in worker_source
        and "self.rollout = vLLMRollout(" in worker_source
        and "FSDPVLLMShardingManager(" in worker_source
    )
    report.facts["easyr1_rollout_is_vllm_only"] = vllm_is_fixed
    report.facts["easyr1_has_colt_transformers_rollout"] = has_colt_rollout
    if vllm_is_fixed:
        report.add_blocker(
            "ROLLOUT_VLLM_ONLY",
            "EasyR1 hard-codes vLLM rollout and FSDP-vLLM weight sync; this bypasses CoLT latent_reasoning_generate.",
        )

    colt_reward_path = easyr1_root / "verl/reward_function/colt_outcome.py"
    has_colt_outcome_reward = colt_reward_path.is_file() and "parse_hidden_reasoning_answer" in read_text(
        colt_reward_path
    )
    report.facts["easyr1_has_colt_outcome_reward"] = has_colt_outcome_reward
    if not has_colt_outcome_reward:
        report.add_blocker(
            "COLT_OUTCOME_REWARD_MISSING",
            "EasyR1 does not contain the hidden-reasoning-compatible CoLT outcome reward.",
        )

    colt_config_path = easyr1_root / "examples/colt_fixed_v2_outcome_grpo.yaml"
    report.facts["easyr1_has_default_colt_config"] = colt_config_path.is_file()
    if not colt_config_path.is_file():
        report.add_blocker(
            "COLT_RL_CONFIG_MISSING",
            "EasyR1 does not contain examples/colt_fixed_v2_outcome_grpo.yaml.",
        )

    actor_is_standard = "colt_rl_response_length" not in actor_source
    report.facts["actor_has_colt_latent_logprob"] = not actor_is_standard
    if actor_is_standard:
        report.add_blocker(
            "ACTOR_LOGPROB_NOT_LATENT_AWARE",
            "EasyR1 recomputes response log-prob with the standard full-sequence forward, which enters CoLT's SFT-only <think> parser.",
        )

    final_token_reward = "reward_tensor[i, cur_response_length - 1] = score[\"overall\"]" in reward_source
    report.facts["outcome_reward_written_to_final_token"] = final_token_reward
    if not final_token_reward:
        report.add_warning(
            "REWARD_MANAGER_SEMANTICS_UNKNOWN",
            "The audited EasyR1 reward manager no longer matches the expected final-token assignment; re-audit its semantics.",
        )

    visible_think_required = (
        "Provide your thinking process between the <think> and </think> tags" in prompt_source
        and "<answer><your_answer_here></answer>" in prompt_source
    )
    latent_prompt_supported = (
        "LATENT_REASONING_TEMPLATE" in prompt_source
        and 'reasoning_mode == "latent"' in prompt_source
        and "format_reasoning_prompt" in dataset_source
    )
    report.facts["dataset_prompt_requires_visible_think"] = visible_think_required
    report.facts["dataset_supports_latent_prompt"] = latent_prompt_supported
    if visible_think_required and not latent_prompt_supported:
        report.add_blocker(
            "PROMPT_REQUIRES_VISIBLE_THINK",
            "The OneThinker prompt template requests visible <think> text, conflicting with CoLT hidden latent reasoning.",
        )


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object")
                records.append(value)
        return records

    value = json.loads(read_text(path))
    if not isinstance(value, list):
        raise ValueError("JSON dataset root must be an array")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("Every dataset record must be an object")
    return value


def missing_fields(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(REQUIRED_DATASET_FIELDS - record.keys())
    return counts


def audit_dataset(train_file: Path, report: AuditReport) -> None:
    report.facts["train_file"] = str(train_file)
    if not train_file.is_file():
        report.add_blocker("RL_DATASET_MISSING", f"RL training dataset does not exist: {train_file}")
        return

    try:
        records = load_json_records(train_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        report.add_blocker("RL_DATASET_INVALID", f"Unable to parse RL dataset: {error}")
        return

    report.facts["dataset_records"] = len(records)
    if not records:
        report.add_blocker("RL_DATASET_EMPTY", "RL training dataset contains no records.")
        return

    missing = missing_fields(records)
    if missing:
        details = ", ".join(f"{key} ({count})" for key, count in sorted(missing.items()))
        report.add_blocker("RL_DATASET_SCHEMA_MISMATCH", f"Required fields are missing: {details}")

    report.facts["problem_type_counts"] = dict(
        sorted(Counter(str(item.get("problem_type", "<missing>")) for item in records).items())
    )
    report.facts["data_type_counts"] = dict(
        sorted(Counter(str(item.get("data_type", "<missing>")) for item in records).items())
    )
    duplicate_ids = len(records) - len({str(item.get("problem_id")) for item in records})
    report.facts["duplicate_problem_ids"] = duplicate_ids
    if duplicate_ids:
        report.add_warning(
            "DUPLICATE_PROBLEM_IDS",
            f"The dataset contains {duplicate_ids} duplicate problem_id values; verify grouping and leakage semantics.",
        )


def audit_model(repo_root: Path, model_path: Path, report: AuditReport) -> None:
    report.facts["repo_root"] = str(repo_root)
    report.facts["model_path"] = str(model_path)
    vendored_model = repo_root / "transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py"
    if not vendored_model.is_file():
        report.add_blocker("COLT_TRANSFORMERS_MISSING", f"Vendored CoLT model source is missing: {vendored_model}")
    elif "def latent_reasoning_generate(" not in read_text(vendored_model):
        report.add_blocker(
            "COLT_LATENT_GENERATE_MISSING",
            f"Vendored Qwen3-VL does not define latent_reasoning_generate: {vendored_model}",
        )
    else:
        report.facts["vendored_latent_generate"] = True

    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not model_path.is_dir():
        report.add_blocker("RL_MODEL_MISSING", f"CoLT model directory does not exist: {model_path}")
        return
    if not config_path.is_file() or not index_path.is_file():
        report.add_blocker(
            "RL_MODEL_INCOMPLETE",
            f"CoLT model requires config.json and model.safetensors.index.json: {model_path}",
        )
        return

    try:
        config = json.loads(read_text(config_path))
        weight_index = json.loads(read_text(index_path))
        weight_names = tuple(weight_index["weight_map"].keys())
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        report.add_blocker("RL_MODEL_METADATA_INVALID", f"Invalid checkpoint metadata: {error}")
        return

    report.facts["model_architectures"] = config.get("architectures")
    report.facts["model_type"] = config.get("model_type")
    missing_prefixes = [
        prefix for prefix in REQUIRED_MODEL_WEIGHT_PREFIXES if not any(name.startswith(prefix) for name in weight_names)
    ]
    if missing_prefixes:
        report.add_blocker(
            "RL_MODEL_NOT_COLT",
            "Checkpoint is missing CoLT latent weights with prefixes: " + ", ".join(missing_prefixes),
        )
    else:
        report.facts["colt_latent_weights"] = {
            prefix: sum(name.startswith(prefix) for name in weight_names)
            for prefix in REQUIRED_MODEL_WEIGHT_PREFIXES
        }

    auto_map = config.get("auto_map")
    report.facts["checkpoint_has_auto_map"] = bool(auto_map)
    if not auto_map:
        report.add_warning(
            "CHECKPOINT_NO_AUTO_MAP",
            "Checkpoint has no auto_map; imports must resolve to the repository's vendored Transformers implementation.",
        )


def audit_runtime(report: AuditReport) -> None:
    missing_packages = [
        package_name
        for package_name, module_name in RUNTIME_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    report.facts["runtime_python"] = sys.executable
    report.facts["runtime_missing_packages"] = missing_packages
    if missing_packages:
        report.add_blocker(
            "RL_RUNTIME_DEPENDENCIES_MISSING",
            "The selected Python environment is missing required packages: " + ", ".join(missing_packages),
        )


def render_text(report: AuditReport) -> str:
    lines = [f"CoLT/EasyR1 readiness: {'READY' if report.ready else 'BLOCKED'}", "", "Facts:"]
    lines.extend(f"  {key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}" for key, value in report.facts.items())
    lines.append("")
    lines.append("Blockers:")
    lines.extend(f"  [{item.code}] {item.message}" for item in report.blockers)
    if not report.blockers:
        lines.append("  none")
    lines.append("")
    lines.append("Warnings:")
    lines.extend(f"  [{item.code}] {item.message}" for item in report.warnings)
    if not report.warnings:
        lines.append("  none")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--easyr1-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = AuditReport()
    audit_easyr1(args.easyr1_root.resolve(), report)
    audit_model(args.repo_root.resolve(), args.model_path.resolve(), report)
    audit_dataset(args.train_file.resolve(), report)
    if args.check_runtime:
        audit_runtime(report)
    if args.json_output:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0 if report.ready or args.allow_incomplete else 2


if __name__ == "__main__":
    sys.exit(main())
