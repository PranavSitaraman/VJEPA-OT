from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MetricNet(nn.Module):
    """Learns a low-rank Mahalanobis metric A(z)^T A(z)."""

    def __init__(self, latent_dim: int = 256, rank: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank * latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        weight = self.backbone(z)
        return weight.view(z.shape[0], self.rank, self.latent_dim)

    def diagonal(self, z: torch.Tensor) -> torch.Tensor:
        factors = self.forward(z)
        return (factors ** 2).sum(dim=1)

    def cost(self, z_ref: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        factors = self.forward(z_ref)
        delta = (z_a - z_b).unsqueeze(1)
        weighted = (factors * delta).sum(dim=-1)  # (B, r)
        return (weighted ** 2).sum(dim=1)
