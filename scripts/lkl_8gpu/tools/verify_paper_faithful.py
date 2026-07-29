#!/usr/bin/env python3

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
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


def verify_visual_freeze_keys() -> None:
    from llamafactory.model.model_utils import visual

    visual = importlib.reload(visual)

    config = SimpleNamespace(model_type="qwen3_vl")
    finetuning_args = SimpleNamespace(
        freeze_vision_tower=True,
        freeze_multi_modal_projector=True,
        freeze_language_model=False,
    )
    forbidden = visual.get_forbidden_modules(config, finetuning_args)
    expected = {
        "visual.patch_embed",
        "visual.pos_embed",
        "visual.blocks",
        "visual.merger",
        "visual.deepstack_merger_list",
    }
    missing = expected - forbidden
    if missing:
        raise RuntimeError(f"B2 verification failed: missing visual freeze keys: {sorted(missing)}")


def verify_backward_gradient_ownership() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _compute_colt_backward_alignment_loss

    torch.manual_seed(7)
    cot_hidden = torch.randn(2, 3, 4, requires_grad=True)
    latent = torch.randn(2, 1, 6, requires_grad=True)
    projector = torch.nn.Linear(4, 6)

    loss = _compute_colt_backward_alignment_loss(cot_hidden, latent, projector)
    loss.backward()

    decoder_grad = cot_hidden.grad
    projector_grad = projector.weight.grad
    if decoder_grad is None or not torch.isfinite(decoder_grad).all() or decoder_grad.abs().sum().item() == 0:
        raise RuntimeError("B3 verification failed: decoder branch did not receive a finite nonzero gradient.")
    if projector_grad is None or not torch.isfinite(projector_grad).all() or projector_grad.abs().sum().item() == 0:
        raise RuntimeError("B3 verification failed: pj_back did not receive a finite nonzero gradient.")
    if latent.grad is not None:
        raise RuntimeError("B3 verification failed: latent target unexpectedly received a gradient.")


def main() -> None:
    if os.environ.get("COLT_PAPER_FAITHFUL", "0") != "1":
        raise RuntimeError("Set COLT_PAPER_FAITHFUL=1 before running this verifier.")

    verify_forward_causal_alignment()
    print("B1 OK: forward CoT targets use exactly one causal shift.")
    verify_empty_forward_targets()
    print("B1 OK: empty forward-CoT targets produce a finite graph-connected zero loss.")
    verify_visual_freeze_keys()
    print("B2 OK: Qwen3-VL visual freeze covers all paper-faithful modules.")
    verify_backward_gradient_ownership()
    print("B3 OK: gradients flow to decoder/pj_back and stop at the latent target.")


if __name__ == "__main__":
    main()
