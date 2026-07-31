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
        self.call_shapes = []

    def forward(self, inputs_embeds, attention_mask, position_ids, use_cache=False):
        del use_cache
        self.forward_calls += 1
        self.call_shapes.append(tuple(inputs_embeds.shape[:2]))
        masked = inputs_embeds * attention_mask.unsqueeze(-1)
        hidden = torch.cumsum(masked, dim=1) + position_ids.unsqueeze(-1) * 0.01
        return SimpleNamespace(logits=self.projection(hidden))


class ToyBackwardBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size, dtype=torch.float64)
        self.forward_calls = 0
        self.call_shapes = []
        self.grad_enabled_calls = []

    def forward(self, input_ids, attention_mask, use_cache=False):
        del use_cache
        self.forward_calls += 1
        self.call_shapes.append(tuple(input_ids.shape))
        self.grad_enabled_calls.append(torch.is_grad_enabled())
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
    projector_calls = []
    pj_out.register_forward_hook(lambda *_: projector_calls.append(1))
    losses = _run_colt_forward_decoder_steps(
        decoder, pj_out, _causal_loss, records, vocab_size, batched=batched
    )
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    gradients = _collect_named_gradients(
        [("decoder", decoder), ("pj_in", pj_in), ("pj_out", pj_out)], latents
    )
    return (
        torch.stack([loss.detach() for loss in losses]),
        total_loss.detach(),
        gradients,
        decoder.forward_calls,
        len(projector_calls),
    )


def verify_forward_batch_equivalence() -> None:
    sequential_losses, sequential_total, sequential_gradients, sequential_calls, sequential_projector_calls = (
        _run_forward_case(31, batched=False)
    )
    batched_losses, batched_total, batched_gradients, batched_calls, batched_projector_calls = (
        _run_forward_case(31, batched=True)
    )
    if sequential_calls != 3 or batched_calls != 1:
        raise RuntimeError(f"Expected forward decoder calls 3 -> 1, got {sequential_calls} -> {batched_calls}.")
    if sequential_projector_calls != 3 or batched_projector_calls != 1:
        raise RuntimeError(
            "Expected forward projector calls 3 -> 1, got "
            f"{sequential_projector_calls} -> {batched_projector_calls}."
        )
    _assert_close("forward per-step losses", batched_losses, sequential_losses)
    _assert_close("forward step-equal total", batched_total, sequential_total)
    for name in sequential_gradients:
        _assert_close(f"forward gradient {name}", batched_gradients[name], sequential_gradients[name])


def _run_forward_dummy_case(seed: int, batched: bool):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _run_colt_forward_decoder_steps

    decoder, pj_in, pj_out, latents, records, vocab_size = _make_forward_case(seed)
    projector_calls = []
    pj_out.register_forward_hook(lambda *_: projector_calls.append(1))
    records[2]["has_targets"] = False
    records[2]["labels"].fill_(-100)
    losses = _run_colt_forward_decoder_steps(
        decoder, pj_out, _causal_loss, records, vocab_size, batched=batched
    )
    torch.stack(losses).sum().backward()
    return losses, latents, decoder.forward_calls, decoder.call_shapes, len(projector_calls)


def verify_dummy_forward_step_has_zero_gradient() -> None:
    for batched, expected_calls in ((False, 3), (True, 1)):
        losses, latents, calls, call_shapes, projector_calls = _run_forward_dummy_case(41, batched=batched)
        if calls != expected_calls:
            raise RuntimeError(
                f"Dummy forward call count mismatch for batched={batched}: {calls} != {expected_calls}."
            )
        if projector_calls != expected_calls:
            raise RuntimeError(
                f"Dummy forward projector call count mismatch for batched={batched}: "
                f"{projector_calls} != {expected_calls}."
            )
        if losses[2].detach().abs().item() != 0.0:
            raise RuntimeError(f"Dummy forward loss was nonzero for batched={batched}.")
        if batched and call_shapes != [(4, 6)]:
            raise RuntimeError(f"Inactive forward record entered the decoder batch: {call_shapes}")
        dummy_gradient = latents[2].grad
        if dummy_gradient is None or dummy_gradient.abs().sum().item() != 0.0:
            raise RuntimeError(f"Dummy forward step changed gradients for batched={batched}.")
        for index in (0, 1):
            active_gradient = latents[index].grad
            if active_gradient is None or active_gradient.abs().sum().item() == 0.0:
                raise RuntimeError(f"Active forward step {index} lost its gradient for batched={batched}.")


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
    projector_calls = []
    pj_back.register_forward_hook(lambda *_: projector_calls.append(1))
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
        decoder.backbone.grad_enabled_calls,
        len(projector_calls),
    )


def verify_backward_batch_equivalence() -> None:
    (
        sequential_losses,
        sequential_total,
        sequential_gradients,
        sequential_calls,
        sequential_grad_enabled,
        sequential_projector_calls,
    ) = _run_backward_case(47, batched=False)
    (
        batched_losses,
        batched_total,
        batched_gradients,
        batched_calls,
        batched_grad_enabled,
        batched_projector_calls,
    ) = _run_backward_case(47, batched=True)
    if sequential_calls != 2 or batched_calls != 1:
        raise RuntimeError(f"Expected backward decoder calls 2 -> 1, got {sequential_calls} -> {batched_calls}.")
    if sequential_projector_calls != 2 or batched_projector_calls != 1:
        raise RuntimeError(
            "Expected backward projector calls 2 -> 1, got "
            f"{sequential_projector_calls} -> {batched_projector_calls}."
        )
    _assert_close("backward per-step losses", batched_losses, sequential_losses)
    _assert_close("backward step-equal total", batched_total, sequential_total)
    for name in sequential_gradients:
        _assert_close(f"backward gradient {name}", batched_gradients[name], sequential_gradients[name])
    decoder_gradient_names = [name for name in sequential_gradients if name.startswith("decoder.")]
    if any(sequential_gradients[name] is not None for name in decoder_gradient_names):
        raise RuntimeError("Backward alignment leaked gradients into the fixed textual semantic anchor.")
    if any(sequential_grad_enabled) or any(batched_grad_enabled):
        raise RuntimeError("Backward decoder backbone did not run under torch.no_grad().")
    for index in range(2):
        gradient = sequential_gradients[f"latent.{index}"]
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum().item() == 0:
            raise RuntimeError(f"Backward alignment did not update latent state {index}.")


def _run_backward_dummy_case(seed: int, batched: bool):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _run_colt_backward_decoder_steps

    decoder, pj_back, latents, records = _make_backward_case(seed)
    projector_calls = []
    pj_back.register_forward_hook(lambda *_: projector_calls.append(1))
    records[1]["active"] = False
    losses = _run_colt_backward_decoder_steps(
        decoder, pj_back, records, pad_token_id=0, batched=batched
    )
    torch.stack(losses).sum().backward()
    return losses, latents, decoder.backbone.forward_calls, len(projector_calls)


def verify_dummy_backward_step_has_zero_gradient() -> None:
    for batched, expected_calls in ((False, 2), (True, 1)):
        losses, latents, calls, projector_calls = _run_backward_dummy_case(53, batched=batched)
        if calls != expected_calls:
            raise RuntimeError(
                f"Dummy backward call count mismatch for batched={batched}: {calls} != {expected_calls}."
            )
        if projector_calls != expected_calls:
            raise RuntimeError(
                f"Dummy backward projector call count mismatch for batched={batched}: "
                f"{projector_calls} != {expected_calls}."
            )
        if losses[1].detach().abs().item() != 0.0:
            raise RuntimeError(f"Dummy backward loss was nonzero for batched={batched}.")
        dummy_gradient = latents[1].grad
        if dummy_gradient is None or dummy_gradient.abs().sum().item() != 0.0:
            raise RuntimeError(f"Dummy backward step changed gradients for batched={batched}.")
        active_gradient = latents[0].grad
        if active_gradient is None or active_gradient.abs().sum().item() == 0.0:
            raise RuntimeError(f"Active backward step lost its gradient for batched={batched}.")


def _mock_synchronized_chunk_count(global_count: int):
    distributed = torch.distributed
    originals = {
        "is_available": distributed.is_available,
        "is_initialized": distributed.is_initialized,
        "all_reduce": distributed.all_reduce,
    }

    def fake_all_reduce(value, op):
        if op != distributed.ReduceOp.MAX:
            raise RuntimeError(f"Chunk synchronization used the wrong reduction: {op}")
        value.fill_(global_count)

    distributed.is_available = lambda: True
    distributed.is_initialized = lambda: True
    distributed.all_reduce = fake_all_reduce
    return originals


def _restore_distributed(originals) -> None:
    for name, original in originals.items():
        setattr(torch.distributed, name, original)


def verify_inactive_long_backward_records_are_excluded() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _run_colt_backward_decoder_steps

    decoder, pj_back, latents, records = _make_backward_case(59)
    projector_calls = []
    pj_back.register_forward_hook(lambda *_: projector_calls.append(1))
    records = [records[0]]
    records[0]["active"] = True
    for _ in range(6):
        latent = torch.randn(2, 1, 4, dtype=torch.float64, requires_grad=True)
        latents.append(latent)
        records.append(
            {
                "input_ids": torch.ones((2, 3297), dtype=torch.long),
                "attention_mask": torch.ones((2, 3297), dtype=torch.long),
                "probe_positions": torch.full((2,), 3296, dtype=torch.long),
                "latent_embd": latent,
                "active": False,
            }
        )

    losses = _run_colt_backward_decoder_steps(
        decoder,
        pj_back,
        records,
        pad_token_id=0,
        batched=True,
        max_batch_tokens=4096,
    )
    torch.stack(losses).sum().backward()
    if decoder.backbone.call_shapes != [(2, 3)]:
        raise RuntimeError(f"Inactive long records entered the decoder batch: {decoder.backbone.call_shapes}")
    if len(projector_calls) != 1:
        raise RuntimeError(f"Inactive records changed projector call count: {len(projector_calls)}")
    for index, latent in enumerate(latents[2:], start=2):
        if latent.grad is None or latent.grad.abs().sum().item() != 0.0:
            raise RuntimeError(f"Inactive long record {index} changed gradients.")


def verify_minimal_dummy_for_zero_active_backward_rank() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import _run_colt_backward_decoder_steps

    decoder, pj_back, latents, records = _make_backward_case(61)
    projector_calls = []
    pj_back.register_forward_hook(lambda *_: projector_calls.append(1))
    for record in records:
        record["active"] = False
    originals = _mock_synchronized_chunk_count(1)
    try:
        losses = _run_colt_backward_decoder_steps(
            decoder,
            pj_back,
            records,
            pad_token_id=0,
            batched=True,
            max_batch_tokens=4096,
        )
    finally:
        _restore_distributed(originals)

    torch.stack(losses).sum().backward()
    if decoder.backbone.call_shapes != [(1, 1)]:
        raise RuntimeError(f"Zero-active rank did not use a 1x1 dummy: {decoder.backbone.call_shapes}")
    if len(projector_calls) != 1:
        raise RuntimeError(f"Zero-active rank did not issue one projector call: {len(projector_calls)}")
    if any(decoder.backbone.grad_enabled_calls):
        raise RuntimeError("Minimal backward dummy did not run under torch.no_grad().")
    if any(parameter.grad is not None for parameter in decoder.parameters()):
        raise RuntimeError("Minimal backward dummy created decoder parameter gradients.")
    for parameter in pj_back.parameters():
        if parameter.grad is None or parameter.grad.abs().sum().item() != 0.0:
            raise RuntimeError("Minimal backward dummy did not produce an exact zero projector gradient.")
    for index, latent in enumerate(latents):
        if latent.grad is None or latent.grad.abs().sum().item() != 0.0:
            raise RuntimeError(f"Minimal backward dummy changed latent gradient {index}.")


def verify_adaptive_chunking_and_dummy_fill() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        _chunk_colt_records,
        _run_colt_backward_decoder_steps,
    )

    torch.manual_seed(67)
    decoder = ToyBackwardDecoder(vocab_size=11, hidden_size=5)
    pj_back = torch.nn.Linear(5, 4, bias=False, dtype=torch.float64)
    projector_calls = []
    pj_back.register_forward_hook(lambda *_: projector_calls.append(1))
    records = []
    latents = []
    for _ in range(3):
        latent = torch.randn(1, 1, 4, dtype=torch.float64, requires_grad=True)
        latents.append(latent)
        records.append(
            {
                "input_ids": torch.ones((1, 2000), dtype=torch.long),
                "attention_mask": torch.ones((1, 2000), dtype=torch.long),
                "probe_positions": torch.full((1,), 1999, dtype=torch.long),
                "latent_embd": latent,
                "active": True,
            }
        )
    chunks = _chunk_colt_records(records, "input_ids", 4096)
    if [len(chunk) for chunk in chunks] != [2, 1]:
        raise RuntimeError(f"Unexpected padded-token chunks: {[len(chunk) for chunk in chunks]}")

    originals = _mock_synchronized_chunk_count(3)
    try:
        losses = _run_colt_backward_decoder_steps(
            decoder,
            pj_back,
            records,
            pad_token_id=0,
            batched=True,
            max_batch_tokens=4096,
        )
    finally:
        _restore_distributed(originals)
    torch.stack(losses).sum().backward()
    expected_shapes = [(2, 2000), (1, 2000), (1, 1)]
    if decoder.backbone.call_shapes != expected_shapes:
        raise RuntimeError(
            f"Adaptive chunks or synchronized dummy calls were incorrect: {decoder.backbone.call_shapes}"
        )
    if len(projector_calls) != 3:
        raise RuntimeError(f"Projector calls did not match synchronized chunk count: {len(projector_calls)}")
    for index, latent in enumerate(latents):
        if latent.grad is None or not torch.isfinite(latent.grad).all() or latent.grad.abs().sum().item() == 0.0:
            raise RuntimeError(f"Adaptive chunking lost active latent gradient {index}.")


def verify_oracle_k_global_max_sync() -> None:
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    distributed = torch.distributed
    originals = {
        "is_available": distributed.is_available,
        "is_initialized": distributed.is_initialized,
        "all_reduce": distributed.all_reduce,
    }

    def fake_all_reduce(value, op):
        if op != distributed.ReduceOp.MAX:
            raise RuntimeError(f"Oracle-K synchronization used the wrong reduction: {op}")
        value.fill_(7)

    try:
        distributed.is_available = lambda: True
        distributed.is_initialized = lambda: True
        distributed.all_reduce = fake_all_reduce
        synchronized = modeling_qwen3_vl._synchronize_colt_oracle_k(2, torch.device("cpu"))
    finally:
        for name, original in originals.items():
            setattr(distributed, name, original)

    if synchronized != 7:
        raise RuntimeError(f"Oracle-K synchronization did not return the global maximum: {synchronized}")


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
    print("T3 OK: Bx2 backward losses and latent/pj_back gradients match; textual anchors stay detached.")
    verify_hot_path_has_no_tensor_conditions_or_tensor_prints()
    print("T4 OK: training forward has no known per-step tensor conditions or per-microbatch tensor prints.")
    verify_dummy_backward_step_has_zero_gradient()
    print("T5 OK: synchronized dummy backward steps execute but contribute zero loss and gradient.")
    verify_oracle_k_global_max_sync()
    print("T6 OK: Oracle-K distributed synchronization uses the global maximum K.")
    verify_dummy_forward_step_has_zero_gradient()
    print("T7 OK: synchronized dummy forward steps execute but contribute zero loss and gradient.")
    verify_inactive_long_backward_records_are_excluded()
    print("T8 OK: inactive 3297-token backward records never enter the real decoder batch.")
    verify_minimal_dummy_for_zero_active_backward_rank()
    print("T9 OK: a zero-active rank uses one graph-connected 1x1 backward dummy call.")
    verify_adaptive_chunking_and_dummy_fill()
    print("T10 OK: padded-token chunking and cross-rank minimal dummy fill preserve gradients.")


if __name__ == "__main__":
    main()
