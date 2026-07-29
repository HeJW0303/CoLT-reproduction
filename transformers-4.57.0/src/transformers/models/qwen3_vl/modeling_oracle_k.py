from typing import Optional, Union

import torch
from torch import nn


def pool_last_valid_hidden(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected hidden states with shape [batch, sequence, hidden], got {hidden_states.shape}")
    if attention_mask is None:
        return hidden_states[:, -1, :]
    if attention_mask.ndim != 2 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            "Question attention mask must match hidden-state batch/sequence dimensions, "
            f"got mask={attention_mask.shape}, hidden={hidden_states.shape}"
        )
    valid_positions = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    token_positions = torch.arange(hidden_states.shape[1], device=hidden_states.device).expand_as(valid_positions)
    last_valid = token_positions.masked_fill(~valid_positions, -1).amax(dim=-1)
    if torch.any(last_valid < 0):
        raise ValueError("Every question must contain at least one valid token")
    gather_index = last_valid[:, None, None].expand(-1, 1, hidden_states.shape[-1])
    return hidden_states.gather(dim=1, index=gather_index).squeeze(1)


class OracleKBudgetConditioner(nn.Module):
    def __init__(self, max_k: int, hidden_size: int):
        super().__init__()
        self.max_k = max_k
        self.budget_embedding = nn.Embedding(max_k + 1, hidden_size)
        self.step_embedding = nn.Embedding(max_k + 1, hidden_size)

    def forward(
        self,
        latent_embd: torch.Tensor,
        oracle_k: Union[int, torch.Tensor],
        step_index: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = latent_embd.shape[0]
        budget_index = torch.as_tensor(oracle_k, dtype=torch.long, device=latent_embd.device).reshape(-1)
        step_index_tensor = torch.as_tensor(step_index, dtype=torch.long, device=latent_embd.device).reshape(-1)
        if budget_index.numel() == 1:
            budget_index = budget_index.expand(batch_size)
        if step_index_tensor.numel() == 1:
            step_index_tensor = step_index_tensor.expand(batch_size)
        if budget_index.numel() != batch_size or step_index_tensor.numel() != batch_size:
            raise ValueError("Oracle-K conditioning indices must be scalar or have one value per sample")
        if torch.any((budget_index < 1) | (budget_index > self.max_k)):
            raise ValueError(f"Oracle K must be in [1, {self.max_k}]")
        if torch.any((step_index_tensor < 1) | (step_index_tensor > budget_index)):
            raise ValueError("Oracle-K step must be in [1, K] for every sample")
        budget_embd = self.budget_embedding(budget_index).unsqueeze(1).to(latent_embd.dtype)
        step_embd = self.step_embedding(step_index_tensor).unsqueeze(1).to(latent_embd.dtype)
        return latent_embd + budget_embd + step_embd


class OracleKPredictor(nn.Module):
    """Predicts a one-based latent budget class from the question/image hidden state."""

    def __init__(self, max_k: int, hidden_size: int):
        super().__init__()
        self.max_k = max_k
        bottleneck_size = max(hidden_size // 2, 1)
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck_size),
            nn.GELU(),
            nn.Linear(bottleneck_size, max_k),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        if pooled_hidden.ndim != 2:
            raise ValueError(f"K predictor expects [batch, hidden], got {tuple(pooled_hidden.shape)}")
        return self.network(pooled_hidden)
