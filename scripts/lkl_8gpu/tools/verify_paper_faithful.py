#!/usr/bin/env python3

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "transformers-4.57.0" / "src"))
sys.path.insert(0, str(REPO_ROOT / "LLaMA-Factory" / "src"))


def verify_forward_causal_alignment() -> None:
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _compute_colt_forward_cot_loss

    vocab_size = 8
    logits = torch.full((1, 4, vocab_size), -20.0, requires_grad=True)
    preferred = torch.tensor([1, 2, 3, 0]).view(1, 4, 1)
    logits = logits.scatter(2, preferred, 20.0)
    labels = torch.tensor([[-100, 1, 2, 3]])

    loss = _compute_colt_forward_cot_loss(ForCausalLMLoss, logits, labels, vocab_size)
    if not torch.isfinite(loss) or loss.item() > 1e-4:
        raise RuntimeError(f"B1 verification failed: expected one-shift causal loss near zero, got {loss.item()}")


def verify_empty_forward_targets() -> None:
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _compute_colt_forward_cot_loss

    logits = torch.randn(1, 4, 8, requires_grad=True)
    labels = torch.full((1, 4), -100)
    loss = _compute_colt_forward_cot_loss(
        ForCausalLMLoss, logits, labels, vocab_size=8, has_targets=False
    )

    loss.backward()
    if not torch.isfinite(loss) or loss.item() != 0.0:
        raise RuntimeError(f"B1 empty-target verification failed: expected zero loss, got {loss.item()}")
    if logits.grad is None or logits.grad.abs().sum().item() != 0.0:
        raise RuntimeError("B1 empty-target verification failed: expected a zero gradient connected to logits.")


def verify_visual_partial_adaptation() -> None:
    from llamafactory.model.adapter import _assert_paper_faithful_visual_policy, _setup_full_tuning
    from llamafactory.model.model_utils import visual

    visual = importlib.reload(visual)

    config = SimpleNamespace(model_type="qwen3_vl")
    finetuning_args = SimpleNamespace(
        freeze_vision_tower=True,
        freeze_multi_modal_projector=True,
        freeze_language_model=False,
    )
    forbidden = visual.get_forbidden_modules(config, finetuning_args)
    expected_frozen = {"visual.patch_embed", "visual.blocks", "visual.merger"}
    expected_trainable = {"visual.pos_embed", "visual.deepstack_merger_list"}
    missing = expected_frozen - forbidden
    unexpectedly_frozen = expected_trainable & forbidden
    if missing or unexpectedly_frozen:
        raise RuntimeError(
            "Visual policy verification failed: "
            f"missing frozen keys={sorted(missing)}, unexpectedly frozen={sorted(unexpectedly_frozen)}"
        )

    class ToyVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Linear(2, 2)
            self.pos_embed = torch.nn.Embedding(2, 2)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.merger = torch.nn.Linear(2, 2)
            self.deepstack_merger_list = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.visual = ToyVisual()

    model = ToyModel()
    _setup_full_tuning(model, finetuning_args, is_trainable=True, cast_trainable_params_to_fp32=False)
    _assert_paper_faithful_visual_policy(model, finetuning_args)
    for name, parameter in model.named_parameters():
        if name.startswith(("visual.patch_embed.", "visual.blocks.", "visual.merger.")):
            if parameter.requires_grad:
                raise RuntimeError(f"Visual policy verification failed: {name} should be frozen.")
        elif name.startswith(("visual.pos_embed.", "visual.deepstack_merger_list.")):
            if not parameter.requires_grad:
                raise RuntimeError(f"Visual policy verification failed: {name} should remain trainable.")


def verify_backward_gradient_ownership() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _compute_colt_backward_alignment_loss

    torch.manual_seed(7)
    cot_hidden = torch.randn(2, 3, 4, requires_grad=True)
    latent = torch.randn(2, 1, 6, requires_grad=True)
    projector = torch.nn.Linear(4, 6)

    loss = _compute_colt_backward_alignment_loss(cot_hidden, latent, projector)
    loss.backward()

    decoder_grad = cot_hidden.grad
    latent_grad = latent.grad
    projector_grad = projector.weight.grad
    if decoder_grad is not None:
        raise RuntimeError("Backward verification failed: the official textual semantic anchor received a gradient.")
    if projector_grad is None or not torch.isfinite(projector_grad).all() or projector_grad.abs().sum().item() == 0:
        raise RuntimeError("Backward verification failed: pj_back did not receive a finite nonzero gradient.")
    if latent_grad is None or not torch.isfinite(latent_grad).all() or latent_grad.abs().sum().item() == 0:
        raise RuntimeError("Backward verification failed: latent state did not receive a finite nonzero gradient.")


def verify_backward_decoder_official_behavior() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    init_source = inspect.getsource(Qwen3VLForConditionalGeneration.__init__)
    forward_source = inspect.getsource(Qwen3VLForConditionalGeneration.forward)
    forbidden = [
        "self.backward_decoder.gradient_checkpointing_enable()",
        "enable_gradient_checkpointing=self.paper_faithful",
    ]
    present = [snippet for snippet in forbidden if snippet in init_source or snippet in forward_source]
    if present:
        raise RuntimeError(f"Backward decoder no longer preserves official initialization behavior: {present}")


def verify_aux_batching_switch() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _is_colt_aux_batching_enabled

    previous = os.environ.pop("COLT_BATCH_AUX_DECODERS", None)
    try:
        if _is_colt_aux_batching_enabled():
            raise RuntimeError("Auxiliary decoder batching must default to disabled.")
        for value in ("1", "true", "yes", "on"):
            os.environ["COLT_BATCH_AUX_DECODERS"] = value
            if not _is_colt_aux_batching_enabled():
                raise RuntimeError(f"Auxiliary decoder batching did not accept opt-in value {value!r}.")
        os.environ["COLT_BATCH_AUX_DECODERS"] = "0"
        if _is_colt_aux_batching_enabled():
            raise RuntimeError("Auxiliary decoder batching did not honor explicit disable value '0'.")
    finally:
        if previous is None:
            os.environ.pop("COLT_BATCH_AUX_DECODERS", None)
        else:
            os.environ["COLT_BATCH_AUX_DECODERS"] = previous


def main() -> None:
    if os.environ.get("COLT_PAPER_FAITHFUL", "0") != "1":
        raise RuntimeError("Set COLT_PAPER_FAITHFUL=1 before running this verifier.")

    verify_forward_causal_alignment()
    print("B1 OK: forward CoT targets use exactly one causal shift.")
    verify_empty_forward_targets()
    print("B1 OK: empty forward-CoT targets produce a finite graph-connected zero loss.")
    verify_visual_partial_adaptation()
    print("Visual OK: official partial adaptation keeps pos_embed/deepstack trainable.")
    verify_backward_gradient_ownership()
    print("Backward OK: gradients flow to latent/pj_back and stop at the textual semantic anchor.")
    verify_backward_decoder_official_behavior()
    print("Backward decoder OK: official initialization/checkpointing behavior is preserved.")
    verify_aux_batching_switch()
    print("Batching OK: auxiliary decoder batching defaults off and supports explicit opt-in.")


if __name__ == "__main__":
    main()
