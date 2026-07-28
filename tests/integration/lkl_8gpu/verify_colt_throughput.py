#!/usr/bin/env python3

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "transformers-4.57.0" / "src"))


class ToyForwardDecoder(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.float64)
        self.forward_calls = 0

    def forward(self, inputs_embeds, attention_mask, position_ids, use_cache=False):
        del use_cache
        self.forward_calls += 1
        masked = inputs_embeds * attention_mask.unsqueeze(-1)
        hidden = torch.cumsum(masked, dim=1) + position_ids.unsqueeze(-1) * 0.01
        return SimpleNamespace(logits=self.projection(hidden))


class ToyBackwardBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size, dtype=torch.float64)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask, use_cache=False):
        del use_cache
        self.forward_calls += 1
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        hidden = torch.cumsum(hidden, dim=1)
        return (hidden,)


class ToyBackwardDecoder(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.backbone = ToyBackwardBackbone(vocab_size, hidden_size)

    def get_decoder(self):
        return self.backbone


class ToyReloadDecoder(torch.nn.Module):
    def __init__(self, zero: bool):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 3)
        torch.nn.init.constant_(self.embedding.weight, 0.0 if zero else 1.0)
        self.gradient_checkpointing_calls = 0

    def get_input_embeddings(self):
        return self.embedding

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_calls += 1


def _causal_loss(logits, labels, vocab_size):
    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def _assert_close(name, actual, expected, atol=1e-10, rtol=1e-9):
    if actual is None or expected is None:
        if actual is expected:
            return
        raise RuntimeError(f"{name}: one gradient is None and the other is not")
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_diff = (actual - expected).abs().max().item()
        raise RuntimeError(f"{name}: tensors differ (max abs diff {max_diff})")


def verify_decoder_check_runs_once() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _ensure_colt_decoder_initialized

    loader_calls = []

    def loader(*args, **kwargs):
        loader_calls.append((args, kwargs))
        return ToyReloadDecoder(zero=False)

    decoder = ToyReloadDecoder(zero=False)
    decoder, checked = _ensure_colt_decoder_initialized(
        decoder, False, "toy", torch.float32, torch.device("cpu"), loader=loader
    )
    if not checked or loader_calls:
        raise RuntimeError("A nonzero decoder was not marked checked exactly once.")

    decoder.embedding.weight.data.zero_()
    decoder, checked = _ensure_colt_decoder_initialized(
        decoder, checked, "toy", torch.float32, torch.device("cpu"), loader=loader
    )
    if loader_calls:
        raise RuntimeError("A decoder marked checked was scanned/reloaded again.")

    zero_decoder = ToyReloadDecoder(zero=True)
    reloaded, checked = _ensure_colt_decoder_initialized(
        zero_decoder,
        False,
        "toy",
        torch.float32,
        torch.device("cpu"),
        enable_gradient_checkpointing=True,
        loader=loader,
    )
    if not checked or len(loader_calls) != 1 or reloaded is zero_decoder:
        raise RuntimeError("A zero decoder was not reloaded exactly once.")
    if reloaded.gradient_checkpointing_calls != 1:
        raise RuntimeError("The reloaded backward decoder did not restore gradient checkpointing.")

    _ensure_colt_decoder_initialized(
        reloaded, checked, "toy", torch.float32, torch.device("cpu"), loader=loader
    )
    if len(loader_calls) != 1:
        raise RuntimeError("A zero decoder reload was attempted more than once.")


def _make_forward_case(seed: int):
    torch.manual_seed(seed)
    hidden_size = 5
    vocab_size = 7
    decoder = ToyForwardDecoder(hidden_size, vocab_size)
    pj_in = torch.nn.Linear(4, hidden_size, bias=False, dtype=torch.float64)
    pj_out = torch.nn.Linear(vocab_size, vocab_size, bias=False, dtype=torch.float64)
    latents = [torch.randn(2, 1, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    step_lengths = [4, 6, 3]
    records = []
    for step, (latent, length) in enumerate(zip(latents, step_lengths)):
        token_embeds = torch.randn(2, length - 1, hidden_size, dtype=torch.float64)
        inputs_embeds = torch.cat([pj_in(latent), token_embeds], dim=1)
        attention_mask = torch.ones(2, length, dtype=torch.long)
        if step == 1:
            attention_mask[1, -2:] = 0
        labels = torch.randint(0, vocab_size, (2, length), dtype=torch.long)
        labels[:, 0] = -100
        labels = labels.masked_fill(attention_mask == 0, -100)
        records.append(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": torch.arange(length).unsqueeze(0).expand(2, -1),
                "labels": labels,
                "has_targets": True,
            }
        )
    return decoder, pj_in, pj_out, latents, records, vocab_size


def _collect_named_gradients(modules, tensors):
    gradients = {}
    for module_name, module in modules:
        for name, parameter in module.named_parameters():
            gradients[f"{module_name}.{name}"] = None if parameter.grad is None else parameter.grad.detach().clone()
    for index, tensor in enumerate(tensors):
        gradients[f"latent.{index}"] = None if tensor.grad is None else tensor.grad.detach().clone()
    return gradients


def _run_forward_case(seed: int, batched: bool):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _run_colt_forward_decoder_steps

    decoder, pj_in, pj_out, latents, records, vocab_size = _make_forward_case(seed)
    losses = _run_colt_forward_decoder_steps(
        decoder, pj_out, _causal_loss, records, vocab_size, batched=batched
    )
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    gradients = _collect_named_gradients(
        [("decoder", decoder), ("pj_in", pj_in), ("pj_out", pj_out)], latents
    )
    return torch.stack([loss.detach() for loss in losses]), total_loss.detach(), gradients, decoder.forward_calls


def verify_forward_batch_equivalence() -> None:
    sequential_losses, sequential_total, sequential_gradients, sequential_calls = _run_forward_case(
        31, batched=False
    )
    batched_losses, batched_total, batched_gradients, batched_calls = _run_forward_case(31, batched=True)
    if sequential_calls != 3 or batched_calls != 1:
        raise RuntimeError(f"Expected forward decoder calls 3 -> 1, got {sequential_calls} -> {batched_calls}.")
    _assert_close("forward per-step losses", batched_losses, sequential_losses)
    _assert_close("forward step-equal total", batched_total, sequential_total)
    for name in sequential_gradients:
        _assert_close(f"forward gradient {name}", batched_gradients[name], sequential_gradients[name])


def _make_backward_case(seed: int):
    torch.manual_seed(seed)
    decoder = ToyBackwardDecoder(vocab_size=11, hidden_size=5)
    pj_back = torch.nn.Linear(5, 4, bias=False, dtype=torch.float64)
    latents = [torch.randn(2, 1, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    records = []
    for index, (length, latent) in enumerate(zip([3, 6], latents)):
        input_ids = torch.randint(1, 10, (2, length), dtype=torch.long)
        input_ids[:, -1] = 10
        attention_mask = torch.ones(2, length, dtype=torch.long)
        if index == 1:
            attention_mask[1, 3:5] = 0
        records.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "probe_positions": torch.full((2,), length - 1, dtype=torch.long),
                "latent_embd": latent,
            }
        )
    return decoder, pj_back, latents, records


def _run_backward_case(seed: int, batched: bool):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _run_colt_backward_decoder_steps

    decoder, pj_back, latents, records = _make_backward_case(seed)
    losses = _run_colt_backward_decoder_steps(
        decoder, pj_back, records, pad_token_id=0, batched=batched
    )
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    gradients = _collect_named_gradients([("decoder", decoder), ("pj_back", pj_back)], latents)
    return (
        torch.stack([loss.detach() for loss in losses]),
        total_loss.detach(),
        gradients,
        decoder.backbone.forward_calls,
    )


def verify_backward_batch_equivalence() -> None:
    sequential_losses, sequential_total, sequential_gradients, sequential_calls = _run_backward_case(
        47, batched=False
    )
    batched_losses, batched_total, batched_gradients, batched_calls = _run_backward_case(47, batched=True)
    if sequential_calls != 2 or batched_calls != 1:
        raise RuntimeError(f"Expected backward decoder calls 2 -> 1, got {sequential_calls} -> {batched_calls}.")
    _assert_close("backward per-step losses", batched_losses, sequential_losses)
    _assert_close("backward step-equal total", batched_total, sequential_total)
    for name in sequential_gradients:
        _assert_close(f"backward gradient {name}", batched_gradients[name], sequential_gradients[name])
    if any(sequential_gradients[f"latent.{index}"] is not None for index in range(2)):
        raise RuntimeError("Backward alignment leaked gradients into a latent target.")


def verify_hot_path_has_no_tensor_conditions_or_tensor_prints() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    source = inspect.getsource(Qwen3VLForConditionalGeneration.forward)
    forbidden = [
        "torch.any(cot_step_attention_mask)",
        "(ref_labels != -100).sum()",
        "(shift_ref_labels != -100).sum()",
        'print(f"ce_loss_total : {ce_loss_total}")',
    ]
    present = [snippet for snippet in forbidden if snippet in source]
    if present:
        raise RuntimeError(f"Hot-path CUDA synchronization patterns remain: {present}")


def main() -> None:
    verify_decoder_check_runs_once()
    print("T1 OK: each decoder embedding is checked/reloaded at most once.")
    verify_forward_batch_equivalence()
    print("T2 OK: Bx3 forward losses and decoder/pj_in/pj_out/latent gradients match sequential execution.")
    verify_backward_batch_equivalence()
    print("T3 OK: Bx2 backward losses and decoder/pj_back gradients match; latent targets stay detached.")
    verify_hot_path_has_no_tensor_conditions_or_tensor_prints()
    print("T4 OK: training forward has no known per-step tensor conditions or per-microbatch tensor prints.")


if __name__ == "__main__":
    main()
