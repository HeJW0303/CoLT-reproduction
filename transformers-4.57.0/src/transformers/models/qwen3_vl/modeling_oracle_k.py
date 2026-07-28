from typing import Union

import torch
from torch import nn


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
