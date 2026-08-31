# coding=utf-8
# Copyright 2026 CoLT-reproduction contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Latent transition heads used by CoLT.

This module hosts the small, self-contained ``nn.Module`` heads that turn the
backbone hidden state into a stochastic or low-rank latent action. Keeping them
here (instead of inside the 4k-line ``modeling_qwen3_vl.py``) keeps the main
model file focused on wiring.
"""

import contextlib
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_visual_tokens_by_sample(
    image_embeds: Optional[torch.Tensor],
    sample_token_counts: torch.Tensor,
    *,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack flattened Qwen3-VL image tokens into a padded per-sample batch.

    Qwen3-VL concatenates the projected tokens of every image into one
    ``(total_tokens, hidden)`` tensor.  The image placeholders in each prompt
    provide the authoritative number of projected tokens belonging to that
    sample, including the multi-image case.  Returning an all-masked dummy
    token for text-only rows keeps the LaRe module call order identical across
    ZeRO-3 ranks while guaranteeing a zero visual residual for those rows.
    """
    if sample_token_counts.ndim != 1:
        raise ValueError(
            "sample_token_counts must have shape (batch,), "
            f"got {tuple(sample_token_counts.shape)}"
        )
    if torch.any(sample_token_counts < 0):
        raise ValueError("sample_token_counts must be non-negative")

    counts = [int(value) for value in sample_token_counts.detach().cpu().tolist()]
    batch_size = len(counts)
    total_tokens = sum(counts)
    if image_embeds is None:
        if total_tokens:
            raise RuntimeError(
                "The prompt contains image placeholders but Qwen3-VL did not cache image features. "
                "Set COLT_LARE_REFOCUS=1 before model construction and provide pixel_values."
            )
    else:
        if image_embeds.ndim != 2:
            raise ValueError(
                "image_embeds must have shape (total_tokens, hidden), "
                f"got {tuple(image_embeds.shape)}"
            )
        if image_embeds.shape[-1] != hidden_size:
            raise ValueError(
                f"image hidden size {image_embeds.shape[-1]} does not match model hidden size {hidden_size}"
            )
        if image_embeds.shape[0] != total_tokens:
            raise ValueError(
                "Flattened image-token count does not match prompt placeholders: "
                f"cached={image_embeds.shape[0]}, prompt={total_tokens}."
            )

    max_tokens = max(max(counts, default=0), 1)
    padded = torch.zeros((batch_size, max_tokens, hidden_size), dtype=dtype, device=device)
    valid_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)
    if image_embeds is None or total_tokens == 0:
        return padded, valid_mask

    source = image_embeds.to(device=device, dtype=dtype)
    offset = 0
    for row, count in enumerate(counts):
        if count:
            padded[row, :count] = source[offset : offset + count]
            valid_mask[row, :count] = True
            offset += count
    return padded, valid_mask


def cot_attention_alignment_loss(
    student_attention: torch.Tensor,
    teacher_targets: Optional[torch.Tensor],
    teacher_target_mask: Optional[torch.Tensor],
    visual_mask: torch.Tensor,
    *,
    min_confidence: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distil a frozen CoT-guided patch distribution into LaRe attention.

    ``teacher_targets`` are deliberately supplied by the data pipeline rather
    than generated in the training forward.  They come from a frozen visible-
    CoT Qwen checkpoint and can therefore be cached, fingerprinted, inspected,
    and replaced without adding a second 8B model to every training rank.

    The teacher is a *soft, unreliable* pseudo target, not a box annotation.
    We consequently (1) stop gradients at the call site/data boundary, (2)
    renormalize only over the sample's real image tokens, and (3) weight each
    row by its normalized concentration ``1 - H(A)/log(P)``.  Diffuse maps
    carry no directional spatial information and are skipped below
    ``min_confidence``; no hard top-k target is ever used.

    Args:
        student_attention: final LaRe cross attention, ``[B, slots, P]``.
        teacher_targets: cached teacher scores/probabilities, ``[B, P]``.
        teacher_target_mask: cache-presence/padding mask, ``[B, P]``.
        visual_mask: valid native Qwen visual tokens, ``[B, P]``.

    Returns:
        ``(loss, effective_rows, mean_confidence)``.  ``effective_rows`` is a
        scalar tensor so callers can aggregate it across recurrent steps.
    """
    if student_attention.ndim != 3:
        raise ValueError(
            "student_attention must have shape (batch, slots, visual_tokens), "
            f"got {tuple(student_attention.shape)}"
        )
    if visual_mask.shape != student_attention.shape[:1] + student_attention.shape[2:]:
        raise ValueError(
            "visual_mask must have shape (batch, visual_tokens), "
            f"got {tuple(visual_mask.shape)} for attention={tuple(student_attention.shape)}"
        )
    zero = student_attention.float().sum() * 0.0
    if teacher_targets is None or teacher_target_mask is None:
        # Some distributed microbatches legitimately have no cached teacher
        # map.  Still execute the complete student-attention graph with an
        # all-false target mask: this contributes exactly zero supervision
        # while keeping the LaRe autograd/ZeRO-3 hook topology identical to
        # ranks whose samples do have text-guided targets.
        teacher_targets = torch.zeros_like(visual_mask, dtype=student_attention.dtype)
        teacher_target_mask = torch.zeros_like(visual_mask, dtype=torch.bool)
    if teacher_targets.shape != visual_mask.shape or teacher_target_mask.shape != visual_mask.shape:
        raise ValueError(
            "cached CoT attention targets/mask must exactly match packed visual tokens; "
            f"targets={tuple(teacher_targets.shape)}, target_mask={tuple(teacher_target_mask.shape)}, "
            f"visual_mask={tuple(visual_mask.shape)}. Rebuild the sidecar cache for this tokenized dataset."
        )
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence!r}")

    valid = visual_mask.bool() & teacher_target_mask.bool()
    valid_count = valid.sum(dim=-1)
    has_target = valid_count >= 2
    target = teacher_targets.detach().float().clamp_min(0.0) * valid
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)

    # A teacher that is uniform over P patches has zero confidence.  For a
    # one-patch image there is no spatial choice, so it is excluded above.
    entropy = -(target.clamp_min(eps).log() * target).sum(dim=-1)
    max_entropy = valid_count.clamp_min(2).float().log()
    confidence = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
    effective = has_target & (confidence >= min_confidence)

    # Keep the autograd topology identical on every ZeRO-3 rank. Whether a
    # local microbatch has any confident teacher rows is data-dependent; a
    # Python early return here changes the backward hook order for the LaRe
    # extractor and makes DeepSpeed's first trace disagree across ranks.
    student = student_attention.float().mean(dim=1) * valid
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = (target * (target.clamp_min(eps).log() - student.clamp_min(eps).log())).sum(dim=-1)
    effective_rows = effective.to(confidence.dtype).sum()
    row_weight = confidence * effective.to(confidence.dtype)
    loss = (kl * row_weight).sum() / row_weight.sum().clamp_min(eps)
    mean_confidence = (confidence * effective.to(confidence.dtype)).sum() / effective_rows.clamp_min(1.0)
    return loss, effective_rows, mean_confidence


class LaReRefocusingLayer(nn.Module):
    """One LaRe extractor block: logic self-attention then visual cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_topk: int = 0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError(
                "LaRe extractor hidden_size must be positive and divisible by num_heads; "
                f"got hidden_size={hidden_size}, num_heads={num_heads}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        if attention_topk < 0:
            raise ValueError(f"attention_topk must be non-negative, got {attention_topk!r}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_topk = attention_topk
        self.dropout = dropout

        self.logic_projection = nn.Linear(hidden_size, hidden_size)
        self.logic_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # ``from_pretrained`` initializes missing parameters after loading a
        # base Qwen checkpoint.  Under ZeRO-3, the generic Transformers
        # initializer sees this module's ``in_proj_weight`` as a partitioned
        # (one-dimensional) view and calling ``_reset_parameters`` then raises
        # in Xavier fan-in/fan-out calculation.  This attention is already
        # correctly initialized by its own constructor, before DeepSpeed
        # partitions it; the Qwen wrapper therefore leaves it alone during the
        # later missing-key pass.
        self.self_attention._colt_lare_constructor_initialized = True
        self.cross_query_norm = nn.LayerNorm(hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.cross_q = nn.Linear(hidden_size, hidden_size)
        self.cross_k = nn.Linear(hidden_size, hidden_size)
        self.cross_v = nn.Linear(hidden_size, hidden_size)
        self.cross_out = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # LaRe is a newly initialized router attached to a bf16 backbone.
        # Running its LayerNorm/MHA/projections under autocast can overflow
        # before the residual gate has had a chance to close.  Keep the
        # router's small internal computation in fp32 and return the residual
        # in the caller's dtype; this preserves gradients while preventing a
        # non-finite attention map from poisoning the next latent step.
        output_dtype = queries.dtype
        queries = queries.float()
        visual_tokens = visual_tokens.float()
        debug = os.environ.get("COLT_DEBUG_NONFINITE", "0").strip().lower() in {"1", "true", "yes", "on"}
        rank_zero = not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        def debug_stats(name, values):
            if debug and rank_zero:
                print(
                    f"[colt-debug-lare-internal] {name} finite={bool(torch.isfinite(values.float()).all().item())} "
                    f"absmax={float(torch.nan_to_num(values.float(), nan=0.0).abs().max().item())}",
                    flush=True,
                )

        debug_stats("queries_input", queries)
        debug_stats("visual_input", visual_tokens)
        def linear_fp32(module, values):
            weight = module.weight.float()
            bias = None if module.bias is None else module.bias.float()
            return F.linear(values.float(), weight, bias)

        def layer_norm_fp32(values, module):
            return F.layer_norm(
                values.float(),
                module.normalized_shape,
                module.weight.float(),
                module.bias.float(),
                module.eps,
            )

        logic = layer_norm_fp32(linear_fp32(self.logic_projection, queries), self.logic_norm)
        logic = torch.nan_to_num(logic, nan=0.0, posinf=0.0, neginf=0.0).clamp(-32.0, 32.0)
        # Implement the small slot self-attention explicitly in fp32.  Calling
        # nn.MultiheadAttention under bf16 autocast can overflow even when the
        # surrounding Qwen activations are finite.
        mha = self.self_attention
        qkv = F.linear(logic, mha.in_proj_weight.float(), mha.in_proj_bias.float())
        qkv = torch.nan_to_num(qkv, nan=0.0, posinf=0.0, neginf=0.0).clamp(-32.0, 32.0)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        self_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        self_scores = torch.nan_to_num(self_scores, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        self_attention = torch.softmax(self_scores, dim=-1)
        self_attention = torch.nan_to_num(self_attention, nan=0.0, posinf=0.0, neginf=0.0)
        logic_update = torch.matmul(self_attention, v).transpose(1, 2).contiguous().view_as(logic)
        logic_update = F.linear(
            logic_update,
            mha.out_proj.weight.float(),
            mha.out_proj.bias.float() if mha.out_proj.bias is not None else None,
        )
        logic_update = torch.nan_to_num(logic_update, nan=0.0, posinf=0.0, neginf=0.0).clamp(-32.0, 32.0)
        queries = queries + F.dropout(logic_update, p=self.dropout, training=self.training)

        q = self._split_heads(linear_fp32(self.cross_q, layer_norm_fp32(queries, self.cross_query_norm)))
        visual = layer_norm_fp32(visual_tokens, self.visual_norm)
        k = self._split_heads(linear_fp32(self.cross_k, visual))
        v = self._split_heads(linear_fp32(self.cross_v, visual))
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).clamp(-32.0, 32.0)
        k = torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0).clamp(-32.0, 32.0)
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-32.0, 32.0)

        # Keep score/softmax accumulation in float32.  This avoids the bf16
        # overflow that has already affected CoLT's latent path while retaining
        # bf16 projections and value aggregation under autocast.
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        expanded_mask = visual_mask[:, None, None, :]
        scores = scores.masked_fill(~expanded_mask, torch.finfo(scores.dtype).min)

        if 0 < self.attention_topk < visual_tokens.shape[1]:
            topk = min(self.attention_topk, visual_tokens.shape[1])
            topk_indices = scores.topk(topk, dim=-1).indices
            sparse_mask = torch.zeros_like(scores, dtype=torch.bool)
            sparse_mask.scatter_(-1, topk_indices, True)
            sparse_mask &= expanded_mask
            scores = scores.masked_fill(~sparse_mask, torch.finfo(scores.dtype).min)

        attention = torch.softmax(scores, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
        dropped_attention = F.dropout(attention, p=self.dropout, training=self.training)
        refocused = torch.matmul(dropped_attention, v)
        refocused = refocused.transpose(1, 2).contiguous().view_as(queries)
        queries = queries + F.dropout(linear_fp32(self.cross_out, refocused), p=self.dropout, training=self.training)
        queries = torch.nan_to_num(queries, nan=0.0, posinf=0.0, neginf=0.0).clamp(-32.0, 32.0)
        return queries.to(output_dtype), attention.mean(dim=1)


class LaReLatentRefocusingExtractor(nn.Module):
    """LaRe-style dynamic visual extractor adapted to CoLT's one-token state.

    The paper uses one query per LLM layer.  CoLT exposes one recurrent latent
    token per reasoning step, so the state is expanded into a small bank of
    dynamic evidence slots.  The slots self-attend, refocus over native
    Qwen3-VL image features, pass through GLU+FFN, and are pooled back to one
    gated residual.  The output projection is reset to zero after HF
    ``post_init`` so enabling the module starts exactly at the trained CoLT
    transition rather than perturbing it with random features.
    """

    def __init__(
        self,
        model_hidden_size: int,
        extractor_hidden_size: int = 1536,
        num_layers: int = 2,
        num_heads: int = 12,
        num_queries: int = 4,
        max_steps: int = 16,
        dropout: float = 0.0,
        visual_dropout: float = 0.1,
        attention_topk: int = 0,
        gate_bias: float = -2.0,
        reconstruction_steps: int = 1000,
    ) -> None:
        super().__init__()
        if model_hidden_size <= 0 or extractor_hidden_size <= 0:
            raise ValueError("model and extractor hidden sizes must be positive")
        if num_layers <= 0 or num_queries <= 0 or max_steps <= 0:
            raise ValueError("num_layers, num_queries and max_steps must be positive")
        if extractor_hidden_size % num_heads:
            raise ValueError(
                f"extractor_hidden_size={extractor_hidden_size} must be divisible by num_heads={num_heads}"
            )
        if not 0.0 <= visual_dropout < 1.0:
            raise ValueError(f"visual_dropout must be in [0, 1), got {visual_dropout!r}")
        if reconstruction_steps <= 0:
            raise ValueError("reconstruction_steps must be positive")

        self.model_hidden_size = model_hidden_size
        self.extractor_hidden_size = extractor_hidden_size
        self.num_queries = num_queries
        self.max_steps = max_steps
        self.visual_dropout = visual_dropout
        self.gate_bias = gate_bias
        self.reconstruction_steps = reconstruction_steps

        self.latent_norm = nn.LayerNorm(model_hidden_size)
        self.visual_input_norm = nn.LayerNorm(model_hidden_size)
        self.latent_projection = nn.Linear(model_hidden_size, extractor_hidden_size)
        self.visual_projection = nn.Linear(model_hidden_size, extractor_hidden_size)
        self.query_slots = nn.Parameter(torch.empty(num_queries, extractor_hidden_size))
        self.step_embeddings = nn.Parameter(torch.empty(max_steps, extractor_hidden_size))
        self.layers = nn.ModuleList(
            [
                LaReRefocusingLayer(
                    extractor_hidden_size,
                    num_heads,
                    dropout=dropout,
                    attention_topk=attention_topk,
                )
                for _ in range(num_layers)
            ]
        )
        self.glu_projection = nn.Linear(extractor_hidden_size, extractor_hidden_size * 2)
        self.ffn_norm = nn.LayerNorm(extractor_hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(extractor_hidden_size, extractor_hidden_size * 4),
            nn.GELU(),
            nn.Linear(extractor_hidden_size * 4, extractor_hidden_size),
        )
        self.slot_score = nn.Linear(extractor_hidden_size, 1)
        self.output_norm = nn.LayerNorm(extractor_hidden_size)
        self.output_projection = nn.Linear(extractor_hidden_size, model_hidden_size)
        self.gate_projection = nn.Linear(model_hidden_size, 1)

        # Lightweight native-feature analogue of LaRe's diffusion semantic
        # augmentation.  It reconstructs injected noise in frozen projected
        # Qwen visual features, conditioned on the refocused latent state.
        self.time_mlp = nn.Sequential(
            nn.Linear(1, extractor_hidden_size),
            nn.SiLU(),
            nn.Linear(extractor_hidden_size, extractor_hidden_size),
        )
        self.reconstruction_target_projection = nn.Linear(
            model_hidden_size, extractor_hidden_size, bias=False
        )
        self.reconstruction_target_projection.requires_grad_(False)
        self.denoiser = nn.Sequential(
            nn.LayerNorm(extractor_hidden_size),
            nn.Linear(extractor_hidden_size, extractor_hidden_size * 2),
            nn.SiLU(),
            nn.Linear(extractor_hidden_size * 2, extractor_hidden_size),
        )
        betas = torch.linspace(1e-4, 0.02, reconstruction_steps, dtype=torch.float32)
        self.register_buffer("reconstruction_alpha_cumprod", torch.cumprod(1.0 - betas, dim=0), persistent=True)

    def _safe_init_parameters(self) -> list[torch.Tensor]:
        parameters = [
            self.query_slots,
            self.step_embeddings,
            self.output_projection.weight,
            self.gate_projection.weight,
        ]
        # ``nn.MultiheadAttention`` is a native PyTorch module and its
        # constructor initialization is not reliable when the surrounding
        # Qwen model is instantiated on the meta device (the usual
        # low-memory ``from_pretrained`` path).  In that case the later HF
        # missing-key pass can leave the packed QKV bias with arbitrary
        # bit-patterns.  Include every packed-QKV/output tensor in the same
        # gathered safe-init transaction so both ordinary DDP and ZeRO-3 get
        # a deterministic finite initialization.
        for layer in self.layers:
            parameters.extend(
                parameter
                for parameter in layer.self_attention.parameters()
                if parameter is not None
            )
        if self.output_projection.bias is not None:
            parameters.append(self.output_projection.bias)
        if self.gate_projection.bias is not None:
            parameters.append(self.gate_projection.bias)
        return parameters

    @staticmethod
    def _is_distributed_rank_zero() -> bool:
        return not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    def _gather_safe_init_parameters(self):
        """Gather the parameters that must be initialized atomically under ZeRO-3."""
        parameters = self._safe_init_parameters()
        if not any(hasattr(parameter, "ds_id") for parameter in parameters):
            return contextlib.nullcontext()
        import deepspeed

        return deepspeed.zero.GatheredParameters(parameters, modifier_rank=0)

    def reset_safe_output(self, initializer_range: float = 0.02) -> None:
        """Initialize new checkpoints with a zero residual and a mostly closed gate.

        A normal ``torch.no_grad`` reset is insufficient in a DeepSpeed ZeRO-3
        construction context: a parameter can be an empty or rank-local shard.
        Gather the complete tensors, let rank 0 initialize them, then let
        ``GatheredParameters`` repartition and broadcast the result.
        """
        with self._gather_safe_init_parameters(), torch.no_grad():
            if not self._is_distributed_rank_zero():
                return
            nn.init.normal_(self.query_slots, mean=0.0, std=initializer_range)
            nn.init.normal_(self.step_embeddings, mean=0.0, std=initializer_range)
            for layer in self.layers:
                mha = layer.self_attention
                # Match torch.nn.MultiheadAttention's constructor defaults,
                # but perform them here while the full tensors are gathered.
                if mha.in_proj_weight is not None:
                    nn.init.xavier_uniform_(mha.in_proj_weight)
                if mha.in_proj_bias is not None:
                    nn.init.zeros_(mha.in_proj_bias)
                nn.init.xavier_uniform_(mha.out_proj.weight)
                if mha.out_proj.bias is not None:
                    nn.init.zeros_(mha.out_proj.bias)
            nn.init.zeros_(self.output_projection.weight)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)
            nn.init.zeros_(self.gate_projection.weight)
            if self.gate_projection.bias is not None:
                nn.init.constant_(self.gate_projection.bias, self.gate_bias)

    def safe_initialization_stats(self) -> Optional[tuple[float, float]]:
        """Return gathered ``(output_norm, gate_bias)`` on rank 0 for audit logs."""
        with self._gather_safe_init_parameters():
            if not self._is_distributed_rank_zero():
                return None
            output_norm = self.output_projection.weight.float().norm().item()
            if self.gate_projection.bias is None:
                return output_norm, float("nan")
            return output_norm, self.gate_projection.bias.float().mean().item()

    def _apply_visual_dropout(self, visual_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.visual_dropout == 0.0:
            return visual_mask
        kept = visual_mask & (torch.rand_like(visual_mask, dtype=torch.float32) >= self.visual_dropout)
        # Restore one real token whenever dropout removed every valid token.
        needs_restore = visual_mask.any(dim=-1) & ~kept.any(dim=-1)
        for row in torch.nonzero(needs_restore, as_tuple=False).flatten().tolist():
            first_valid = int(torch.nonzero(visual_mask[row], as_tuple=False)[0].item())
            kept[row, first_valid] = True
        return kept

    def _denoising_loss(
        self,
        raw_visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if not visual_mask.any():
            return condition.sum() * 0.0
        # A separate frozen projection plays the role of LaRe's frozen image
        # tokenizer.  Using the trainable router projection as its own target
        # would permit a slow collapse-to-zero shortcut across optimizer steps.
        with torch.no_grad():
            normalized_visual = F.layer_norm(
                raw_visual_tokens,
                (self.model_hidden_size,),
            )
            clean = self.reconstruction_target_projection(normalized_visual).float()
        batch = clean.shape[0]
        timesteps = torch.randint(
            0,
            self.reconstruction_steps,
            (batch,),
            device=clean.device,
        )
        # DeepSpeed ZeRO-3 can retain non-parameter buffers on CPU while the
        # extractor activations live on a rank-local CUDA device.  Indexing a
        # CPU schedule with CUDA timesteps is invalid, so materialize this
        # immutable schedule on the activation device for the lookup.  It has
        # no gradients or optimizer state.
        alpha_cumprod = self.reconstruction_alpha_cumprod.to(device=clean.device)
        alpha_bar = alpha_cumprod[timesteps].view(batch, 1, 1)
        noise = torch.randn_like(clean)
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        time_value = timesteps.float().view(batch, 1) / max(self.reconstruction_steps - 1, 1)
        time_embedding = self.time_mlp(time_value.to(condition.dtype)).float().unsqueeze(1)
        denoiser_input = noisy + condition.float().unsqueeze(1) + time_embedding
        predicted_noise = self.denoiser(denoiser_input.to(condition.dtype)).float()
        token_loss = (predicted_noise - noise).pow(2).mean(dim=-1)
        mask = visual_mask.to(token_loss.dtype)
        return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(
        self,
        latent_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor,
        *,
        step_index: int,
        compute_reconstruction: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output_dtype = latent_state.dtype
        if latent_state.ndim != 3 or latent_state.shape[1] != 1:
            raise ValueError(
                "latent_state must have shape (batch, 1, hidden), "
                f"got {tuple(latent_state.shape)}"
            )
        if visual_tokens.ndim != 3 or visual_mask.shape != visual_tokens.shape[:2]:
            raise ValueError(
                "visual_tokens/visual_mask must have shapes (batch, tokens, hidden)/(batch, tokens); "
                f"got {tuple(visual_tokens.shape)} and {tuple(visual_mask.shape)}"
            )
        if visual_tokens.shape[0] != latent_state.shape[0]:
            raise ValueError("latent and visual batch sizes must match")
        if visual_tokens.shape[-1] != self.model_hidden_size:
            raise ValueError(
                f"visual hidden size {visual_tokens.shape[-1]} does not match {self.model_hidden_size}"
            )
        if not 0 <= step_index < self.max_steps:
            raise ValueError(
                f"step_index={step_index} is outside configured COLT_LARE_MAX_STEPS={self.max_steps}"
            )

        debug = os.environ.get("COLT_DEBUG_NONFINITE", "0").strip().lower() in {"1", "true", "yes", "on"}
        rank_zero = not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        def debug_stats(name, values):
            if debug and rank_zero:
                print(
                    f"[colt-debug-lare-internal] {name} finite={bool(torch.isfinite(values.float()).all().item())} "
                    f"absmax={float(torch.nan_to_num(values.float(), nan=0.0).abs().max().item())}",
                    flush=True,
                )

        row_has_visual = visual_mask.any(dim=-1)
        safe_mask = self._apply_visual_dropout(visual_mask.bool())
        # Avoid all-masked softmax rows.  Their final residual and gate are
        # explicitly zeroed below, so the dummy token cannot affect answers.
        no_visual = ~safe_mask.any(dim=-1)
        if no_visual.any():
            safe_mask = safe_mask.clone()
            safe_mask[no_visual, 0] = True

        latent_query = self.latent_projection(self.latent_norm(latent_state))
        queries = latent_query.expand(-1, self.num_queries, -1)
        queries = (
            queries
            + self.query_slots.unsqueeze(0).to(queries.dtype)
            + self.step_embeddings[step_index].view(1, 1, -1).to(queries.dtype)
        )
        visual = self.visual_projection(self.visual_input_norm(visual_tokens))
        debug_stats("latent_query", latent_query)
        debug_stats("visual_projected", visual)

        attention = visual.new_zeros((visual.shape[0], self.num_queries, visual.shape[1]), dtype=torch.float32)
        for layer in self.layers:
            queries, attention = layer(queries, visual, safe_mask)
            debug_stats("layer_queries", queries)
            debug_stats("layer_attention", attention)

        glu_left, glu_right = self.glu_projection(queries).chunk(2, dim=-1)
        refined = F.silu(glu_left) * glu_right
        refined = refined + self.ffn(self.ffn_norm(refined))
        slot_weights = torch.softmax(self.slot_score(refined).float(), dim=1).to(refined.dtype)
        condition = (slot_weights * refined).sum(dim=1)
        debug_stats("condition", condition)
        raw_delta = self.output_projection(self.output_norm(condition)).unsqueeze(1)
        raw_delta = torch.nan_to_num(raw_delta.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-8.0, 8.0).to(output_dtype)
        gate = torch.sigmoid(self.gate_projection(latent_state).float()).to(raw_delta.dtype)
        present = row_has_visual.view(-1, 1, 1).to(raw_delta.dtype)
        delta = raw_delta * gate * present

        if compute_reconstruction:
            reconstruction_loss = self._denoising_loss(visual_tokens, visual_mask, condition)
        else:
            reconstruction_loss = condition.sum() * 0.0
        return delta, attention, gate * present, reconstruction_loss


class LatentStochasticHead(nn.Module):
    """CoLaR-style Gaussian latent head used for stochastic latent transitions.

    Maps the backbone hidden state to a per-dimension Gaussian ``(mean, log_std)``
    whose samples add exploration noise to the CoLT residual latent update:
    ``z_{t+1} = h + alpha * prj(h) + sigma(h) * eps``.
    """

    def __init__(self, hidden_size: int, log_std_min: float = -4.0, log_std_max: float = 0.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(hidden_states).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max) + torch.log(
            torch.as_tensor(max(temperature, 1e-5), dtype=log_std.dtype, device=log_std.device)
        )
        std = log_std.exp()
        if not sample:
            return torch.zeros_like(mean), mean, std
        noise = torch.randn_like(mean)
        return noise * std, mean, std


class LowRankExplorationHead(nn.Module):
    """Low-rank stochastic action head for multi-path latent exploration.

    Instead of adding full-dimensional Gaussian noise to the latent hidden
    state, this head samples a compact action ``a_k in R^rank`` and maps it back
    through a learned matrix ``U in R^{hidden x rank}``:

        a_k ~ N(mu(h_k), sigma(h_k)^2),   h~_k = h_k + U @ a_k.

    The small ``rank`` keeps the added parameter count negligible while still
    producing genuinely distinct latent trajectories for the same ``(I, Q)``.
    Because ``a_k`` follows an explicit Gaussian, its log-probability is
    available for policy-gradient / preference optimization.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 16,
        log_std_min: float = -4.0,
        log_std_max: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank!r}")
        self.hidden_size = hidden_size
        self.rank = rank
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, rank * 2),
        )
        self.U = nn.Parameter(torch.empty(hidden_size, rank))
        nn.init.normal_(self.U, mean=0.0, std=1.0 / math.sqrt(hidden_size))

    def _distribution(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(hidden_states).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def forward(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, mean, std, delta)``.

        ``delta`` is the perturbation to add to ``hidden_states`` (``U @ action``).
        With ``sample=False`` the action is deterministic (``action = mean``).
        """
        mean, log_std = self._distribution(hidden_states)
        log_std = log_std + math.log(max(temperature, 1e-5))
        std = log_std.exp()
        if sample:
            action = mean + std * torch.randn_like(mean)
        else:
            action = mean
        delta = action @ self.U.t()
        return action, mean, std, delta

    def log_prob(self, hidden_states: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return per-sample ``log p(action | hidden_states)`` of shape ``(B,)``."""
        mean, log_std = self._distribution(hidden_states)
        std = log_std.exp()
        var = std * std
        log_prob = -0.5 * ((action - mean) ** 2 / var + 2.0 * log_std + math.log(2.0 * math.pi))
        return log_prob.sum(dim=-1)
