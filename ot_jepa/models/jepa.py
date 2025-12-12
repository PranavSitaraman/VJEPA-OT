from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class TemporalPredictor(nn.Module):
    """Transformer-based predictor that extrapolates latent futures."""

    def __init__(self, latent_dim: int = 256, depth: int = 4, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            batch_first=True,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )

    def forward(self, z_hist: torch.Tensor, horizon: int) -> torch.Tensor:
        # z_hist: (B, K+1, d)
        h = self.encoder(z_hist)
        base = h[:, -1, :]
        preds = []
        z_t = base
        for _ in range(horizon):
            z_t = self.head(z_t)
            preds.append(z_t)
        return torch.stack(preds, dim=1)


@dataclass
class EMABundle:
    params: dict
    decay: float

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name not in self.params:
                self.params[name] = param.detach().clone()
            else:
                self.params[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.params, strict=False)


class GoalDistributionHead(nn.Module):
    """Predicts goal latent distribution conditioned on instruction embedding."""

    def __init__(self, latent_dim: int = 256, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mean_proj = nn.Linear(hidden, latent_dim)
        self.logvar_proj = nn.Linear(hidden, latent_dim)

    def forward(self, z_lang: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z_lang)
        mean = self.mean_proj(h)
        logvar = torch.tanh(self.logvar_proj(h))  # keep variance bounded
        return mean, logvar
