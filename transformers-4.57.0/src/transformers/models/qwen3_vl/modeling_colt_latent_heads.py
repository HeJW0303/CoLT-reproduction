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

import torch
import torch.nn as nn


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
