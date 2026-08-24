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

import math
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
        logic = self.logic_norm(self.logic_projection(queries))
        logic_update, _ = self.self_attention(logic, logic, logic, need_weights=False)
        queries = queries + F.dropout(logic_update, p=self.dropout, training=self.training)

        q = self._split_heads(self.cross_q(self.cross_query_norm(queries)))
        visual = self.visual_norm(visual_tokens)
        k = self._split_heads(self.cross_k(visual))
        v = self._split_heads(self.cross_v(visual))

        # Keep score/softmax accumulation in float32.  This avoids the bf16
        # overflow that has already affected CoLT's latent path while retaining
        # bf16 projections and value aggregation under autocast.
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
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
        dropped_attention = F.dropout(attention, p=self.dropout, training=self.training)
        refocused = torch.matmul(dropped_attention.to(v.dtype), v)
        refocused = refocused.transpose(1, 2).contiguous().view_as(queries)
        queries = queries + F.dropout(
            self.cross_out(refocused), p=self.dropout, training=self.training
        )
        return queries, attention.mean(dim=1)


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

    def reset_safe_output(self, initializer_range: float = 0.02) -> None:
        """Initialize new checkpoints with a zero residual and a mostly closed gate."""
        with torch.no_grad():
            nn.init.normal_(self.query_slots, mean=0.0, std=initializer_range)
            nn.init.normal_(self.step_embeddings, mean=0.0, std=initializer_range)
            nn.init.zeros_(self.output_projection.weight)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)
            nn.init.zeros_(self.gate_projection.weight)
            if self.gate_projection.bias is not None:
                nn.init.constant_(self.gate_projection.bias, self.gate_bias)

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
        alpha_bar = self.reconstruction_alpha_cumprod[timesteps].view(batch, 1, 1)
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

        attention = visual.new_zeros((visual.shape[0], self.num_queries, visual.shape[1]), dtype=torch.float32)
        for layer in self.layers:
            queries, attention = layer(queries, visual, safe_mask)

        glu_left, glu_right = self.glu_projection(queries).chunk(2, dim=-1)
        refined = F.silu(glu_left) * glu_right
        refined = refined + self.ffn(self.ffn_norm(refined))
        slot_weights = torch.softmax(self.slot_score(refined).float(), dim=1).to(refined.dtype)
        condition = (slot_weights * refined).sum(dim=1)
        raw_delta = self.output_projection(self.output_norm(condition)).unsqueeze(1)
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
