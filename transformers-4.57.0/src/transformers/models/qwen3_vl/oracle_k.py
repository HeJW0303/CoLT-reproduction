import math
import os
from dataclasses import dataclass
from typing import Mapping, Optional


THOUGHT_SEGMENTS_OPEN = "<thought_segments>"
THOUGHT_SEGMENTS_CLOSE = "</thought_segments>"
CONTINUE_THINK = "<continue_think>"
COLT_INFERENCE_TRANSITION_OFFICIAL = "official"
COLT_INFERENCE_TRANSITION_TRAINING_CONSISTENT = "training-consistent"
COLT_INFERENCE_TRANSITIONS = {
    COLT_INFERENCE_TRANSITION_OFFICIAL,
    COLT_INFERENCE_TRANSITION_TRAINING_CONSISTENT,
}


class OracleKFormatError(ValueError):
    pass


@dataclass(frozen=True)
class OracleKAnnotation:
    k: int
    blocks: tuple[str, ...]

    @property
    def original_cot(self) -> str:
        return "".join(self.blocks)


@dataclass(frozen=True)
class OracleKSettings:
    enabled: bool
    max_k: int
    budget_conditioning: bool
    predictor_enabled: bool
    predictor_loss_weight: float
    dynamic_inference: bool


@dataclass(frozen=True)
class OracleKTrainingStep:
    active: bool
    forward_index: int
    backward_index: Optional[int]


@dataclass(frozen=True)
class OracleKInferencePlan:
    transition_steps: int
    conditioning_k: int


def resolve_colt_inference_latent_transition(environ: Optional[Mapping[str, str]] = None) -> str:
    environ = os.environ if environ is None else environ
    mode = environ.get("COLT_INFERENCE_LATENT_TRANSITION", COLT_INFERENCE_TRANSITION_OFFICIAL)
    mode = mode.strip().lower()
    if mode not in COLT_INFERENCE_TRANSITIONS:
        choices = ", ".join(sorted(COLT_INFERENCE_TRANSITIONS))
        raise ValueError(f"COLT_INFERENCE_LATENT_TRANSITION must be one of {choices}, got {mode!r}")
    return mode


def initialize_colt_inference_latent(hidden_states, projector, mode: str):
    """Match the historical inference pre-loop transform or the training initial state."""
    if mode == COLT_INFERENCE_TRANSITION_OFFICIAL:
        return projector(hidden_states)
    if mode == COLT_INFERENCE_TRANSITION_TRAINING_CONSISTENT:
        return hidden_states
    raise ValueError(f"Unsupported CoLT inference latent transition: {mode!r}")


def advance_colt_inference_latent(hidden_states, projector, alpha, mode: str):
    """Advance one latent step without changing the historical mode's arithmetic."""
    projected = projector(hidden_states)
    if mode == COLT_INFERENCE_TRANSITION_OFFICIAL:
        return projected
    if mode == COLT_INFERENCE_TRANSITION_TRAINING_CONSISTENT:
        return hidden_states + alpha * projected
    raise ValueError(f"Unsupported CoLT inference latent transition: {mode!r}")


def build_oracle_k_training_plan(local_k: int, synchronized_k: int) -> tuple[OracleKTrainingStep, ...]:
    """Describe real and zero-loss steps for distributed Oracle-K training."""
    if local_k < 1:
        raise ValueError(f"local_k must be at least 1, got {local_k}")
    if synchronized_k < local_k:
        raise ValueError(
            f"synchronized_k must be at least local_k, got local_k={local_k}, synchronized_k={synchronized_k}"
        )

    return tuple(
        OracleKTrainingStep(
            active=step < local_k,
            forward_index=min(step, local_k - 1),
            backward_index=None if step == 0 else min(step - 1, local_k - 1),
        )
        for step in range(synchronized_k)
    )


def parse_oracle_k_cot(cot_text: str, min_k: int = 1, max_k: Optional[int] = None) -> OracleKAnnotation:
    if not cot_text.startswith(THOUGHT_SEGMENTS_OPEN):
        raise OracleKFormatError(f"Oracle-K CoT must start with {THOUGHT_SEGMENTS_OPEN}")

    count_end = cot_text.find(THOUGHT_SEGMENTS_CLOSE, len(THOUGHT_SEGMENTS_OPEN))
    if count_end < 0:
        raise OracleKFormatError(f"Missing {THOUGHT_SEGMENTS_CLOSE}")

    count_text = cot_text[len(THOUGHT_SEGMENTS_OPEN) : count_end]
    if not count_text.isdigit():
        raise OracleKFormatError(f"Invalid Oracle K: {count_text!r}")

    k = int(count_text)
    if k < min_k:
        raise OracleKFormatError(f"Oracle K={k} is below min_k={min_k}")
    if max_k is not None and k > max_k:
        raise OracleKFormatError(f"Oracle K={k} exceeds max_k={max_k}")

    block_text = cot_text[count_end + len(THOUGHT_SEGMENTS_CLOSE) :]
    forbidden_markers = (THOUGHT_SEGMENTS_OPEN, THOUGHT_SEGMENTS_CLOSE)
    if any(marker in block_text for marker in forbidden_markers):
        raise OracleKFormatError("Duplicate thought-segment metadata found inside CoT blocks")

    blocks = tuple(block_text.split(CONTINUE_THINK))
    if len(blocks) != k:
        raise OracleKFormatError(f"Declared Oracle K={k}, but found {len(blocks)} blocks")
    if any(block == "" for block in blocks):
        raise OracleKFormatError("Oracle-K blocks must not be empty")

    return OracleKAnnotation(k=k, blocks=blocks)


def annotate_segmented_cot(segmented_cot: str, min_k: int = 1, max_k: Optional[int] = None) -> str:
    if THOUGHT_SEGMENTS_OPEN in segmented_cot or THOUGHT_SEGMENTS_CLOSE in segmented_cot:
        raise OracleKFormatError("Teacher output must not contain thought-segment metadata")

    blocks = tuple(segmented_cot.split(CONTINUE_THINK))
    k = len(blocks)
    if any(block == "" for block in blocks):
        raise OracleKFormatError("Teacher produced an empty block")
    if k < min_k:
        raise OracleKFormatError(f"Teacher K={k} is below min_k={min_k}")
    if max_k is not None and k > max_k:
        raise OracleKFormatError(f"Teacher K={k} exceeds max_k={max_k}")

    return f"{THOUGHT_SEGMENTS_OPEN}{k}{THOUGHT_SEGMENTS_CLOSE}{segmented_cot}"


def find_think_span(content: str) -> tuple[int, int]:
    think_open = "<think>"
    think_close = "</think>"
    start = content.rfind(think_open)
    if start < 0:
        raise OracleKFormatError("Assistant content has no <think> section")
    content_start = start + len(think_open)
    end = content.find(think_close, content_start)
    if end < 0:
        raise OracleKFormatError("Assistant content has no closing </think>")
    return content_start, end


def get_assistant_cot(content: str) -> str:
    start, end = find_think_span(content)
    return content[start:end]


def annotate_assistant_content(content: str, segmented_cot: str, min_k: int = 1, max_k: Optional[int] = None) -> str:
    start, end = find_think_span(content)
    original_cot = content[start:end]
    recovered_cot = segmented_cot.replace(CONTINUE_THINK, "")
    if recovered_cot != original_cot:
        raise OracleKFormatError("Teacher output changed the original CoT text")

    annotated_cot = annotate_segmented_cot(segmented_cot, min_k=min_k, max_k=max_k)
    return content[:start] + annotated_cot + content[end:]


def remove_assistant_annotation(content: str, min_k: int = 1, max_k: Optional[int] = None) -> str:
    start, end = find_think_span(content)
    annotation = parse_oracle_k_cot(content[start:end], min_k=min_k, max_k=max_k)
    return content[:start] + annotation.original_cot + content[end:]


def _read_bool_env(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _read_float_env(environ: Mapping[str, str], name: str, default: float) -> float:
    value = environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a floating-point value, got {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {parsed}")
    return parsed


def _config_has(config, name: str) -> bool:
    return name in getattr(config, "__dict__", {})


def resolve_oracle_k_settings(config, environ: Optional[Mapping[str, str]] = None) -> OracleKSettings:
    environ = os.environ if environ is None else environ

    if _config_has(config, "colt_oracle_k_enabled"):
        enabled = bool(config.colt_oracle_k_enabled)
    else:
        enabled = _read_bool_env(environ, "COLT_ORACLE_K_ENABLED", False)

    if _config_has(config, "colt_oracle_k_max"):
        max_k = int(config.colt_oracle_k_max)
    else:
        max_k = int(environ.get("COLT_ORACLE_K_MAX", "8"))

    if _config_has(config, "colt_oracle_k_budget_conditioning"):
        budget_conditioning = bool(config.colt_oracle_k_budget_conditioning)
    else:
        budget_conditioning = _read_bool_env(environ, "COLT_ORACLE_K_BUDGET_CONDITIONING", True)

    if _config_has(config, "colt_oracle_k_predictor_enabled"):
        predictor_enabled = bool(config.colt_oracle_k_predictor_enabled)
    else:
        # Keep existing Oracle-K checkpoints at the Oracle-K-only behavior unless
        # the second-stage predictor is explicitly enabled.
        predictor_enabled = _read_bool_env(environ, "COLT_ORACLE_K_PREDICTOR_ENABLED", False)

    if _config_has(config, "colt_oracle_k_predictor_loss_weight"):
        predictor_loss_weight = float(config.colt_oracle_k_predictor_loss_weight)
    else:
        predictor_loss_weight = _read_float_env(environ, "COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT", 0.2)

    if _config_has(config, "colt_oracle_k_dynamic_inference"):
        dynamic_inference = bool(config.colt_oracle_k_dynamic_inference)
    else:
        dynamic_inference = _read_bool_env(environ, "COLT_ORACLE_K_DYNAMIC_INFERENCE", False)

    if max_k < 1:
        raise ValueError(f"COLT_ORACLE_K_MAX must be at least 1, got {max_k}")
    if not math.isfinite(predictor_loss_weight) or predictor_loss_weight < 0:
        raise ValueError(
            "COLT_ORACLE_K_PREDICTOR_LOSS_WEIGHT must be finite and non-negative, "
            f"got {predictor_loss_weight}"
        )
    if predictor_enabled and not enabled:
        raise ValueError("COLT_ORACLE_K_PREDICTOR_ENABLED requires COLT_ORACLE_K_ENABLED")
    if dynamic_inference and not predictor_enabled:
        raise ValueError("COLT_ORACLE_K_DYNAMIC_INFERENCE requires COLT_ORACLE_K_PREDICTOR_ENABLED")

    if enabled or _config_has(config, "colt_oracle_k_enabled"):
        config.colt_oracle_k_enabled = enabled
        config.colt_oracle_k_max = max_k
        config.colt_oracle_k_budget_conditioning = budget_conditioning
        config.colt_oracle_k_predictor_enabled = predictor_enabled
        config.colt_oracle_k_predictor_loss_weight = predictor_loss_weight
        config.colt_oracle_k_dynamic_inference = dynamic_inference
    return OracleKSettings(
        enabled=enabled,
        max_k=max_k,
        budget_conditioning=budget_conditioning,
        predictor_enabled=predictor_enabled,
        predictor_loss_weight=predictor_loss_weight,
        dynamic_inference=dynamic_inference,
    )


def resolve_forced_inference_k(max_k: int, environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    environ = os.environ if environ is None else environ
    value = environ.get("COLT_INFERENCE_K")
    if value is None or value.strip() == "":
        return None
    forced_k = int(value)
    if not 1 <= forced_k <= max_k:
        raise ValueError(f"COLT_INFERENCE_K must be in [1, {max_k}], got {forced_k}")
    return forced_k


def resolve_forced_inference_transition_steps(
    max_k: int,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[int]:
    environ = os.environ if environ is None else environ
    value = environ.get("COLT_INFERENCE_TRANSITION_STEPS")
    if value is None or value.strip() == "":
        return None
    forced_steps = int(value)
    if not 1 <= forced_steps <= max_k:
        raise ValueError(
            f"COLT_INFERENCE_TRANSITION_STEPS must be in [1, {max_k}], got {forced_steps}"
        )
    return forced_steps


def select_oracle_k_inference_steps(
    max_k: int,
    default_k: int,
    num_hidden_generations: int = 0,
    forced_k: Optional[int] = None,
    predicted_k: Optional[int] = None,
    dynamic_inference: bool = False,
) -> int:
    if num_hidden_generations > 0:
        latent_steps = num_hidden_generations
    elif forced_k is not None:
        latent_steps = forced_k
    elif dynamic_inference:
        if predicted_k is None:
            raise ValueError("Dynamic Oracle-K inference requires one predicted K value")
        latent_steps = predicted_k
    else:
        latent_steps = default_k
    if not 1 <= latent_steps <= max_k:
        raise ValueError(f"Oracle-K inference steps must be in [1, {max_k}], got {latent_steps}")
    return latent_steps


def select_oracle_k_inference_plan(
    max_k: int,
    default_k: int,
    num_hidden_generations: int = 0,
    forced_k: Optional[int] = None,
    forced_transition_steps: Optional[int] = None,
    predicted_k: Optional[int] = None,
    dynamic_inference: bool = False,
) -> OracleKInferencePlan:
    if forced_transition_steps is None:
        latent_steps = select_oracle_k_inference_steps(
            max_k=max_k,
            default_k=default_k,
            num_hidden_generations=num_hidden_generations,
            forced_k=forced_k,
            predicted_k=predicted_k,
            dynamic_inference=dynamic_inference,
        )
        return OracleKInferencePlan(
            transition_steps=latent_steps,
            conditioning_k=latent_steps,
        )

    if num_hidden_generations > 0:
        raise ValueError(
            "COLT_INFERENCE_TRANSITION_STEPS cannot be combined with num_hidden_generations"
        )
    if forced_k is not None:
        raise ValueError(
            "COLT_INFERENCE_TRANSITION_STEPS cannot be combined with COLT_INFERENCE_K"
        )
    if not 1 <= forced_transition_steps <= max_k:
        raise ValueError(f"Forced transition steps must be in [1, {max_k}], got {forced_transition_steps}")
    if predicted_k is None:
        raise ValueError("Decoupled transition-step inference requires one predicted semantic K value")
    if not 1 <= predicted_k <= max_k:
        raise ValueError(f"Predicted semantic K must be in [1, {max_k}], got {predicted_k}")
    return OracleKInferencePlan(
        transition_steps=forced_transition_steps,
        conditioning_k=predicted_k,
    )


def select_oracle_k_conditioning_step(transition_step: int, conditioning_k: int) -> int:
    if transition_step < 1:
        raise ValueError(f"Transition step must be at least 1, got {transition_step}")
    if conditioning_k < 1:
        raise ValueError(f"Conditioning K must be at least 1, got {conditioning_k}")
    return min(transition_step, conditioning_k)
