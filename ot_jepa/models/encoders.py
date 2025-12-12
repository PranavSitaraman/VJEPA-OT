from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Image to patch embeddings with learnable positional encodings."""

    def __init__(self, img_channels: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(img_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        b, d, h, w = x.shape
        return x.flatten(2).transpose(1, 2)


class VisionEncoder(nn.Module):
    """ViT-style encoder optionally augmented with stochastic depth and dropout."""

    def __init__(
        self,
        latent_dim: int = 256,
        embed_dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        patch_size: int = 16,
        img_channels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_channels, embed_dim, patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))  # supports up to 1024 patches
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(x)
        b, n, d = patches.shape
        if n > self.pos_embed.shape[1] - 1:
            raise ValueError(f"Number of patches {n} exceeds maximum supported {self.pos_embed.shape[1]-1}")
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        pos = self.pos_embed[:, : n + 1, :]
        x = x + pos
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.layer_norm(x[:, 0])  # CLS token
        return self.proj(x)


class StateEncoder(nn.Module):
    def __init__(self, in_dim: int = 7, latent_dim: int = 256, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class LangEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 512,
        emb_dim: int = 256,
        latent_dim: int = 256,
        max_len: int = 32,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=4,
            dim_feedforward=emb_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        self.proj = nn.Linear(emb_dim, latent_dim)
        self.max_len = max_len
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        if token_ids.size(1) > self.max_len:
            token_ids = token_ids[:, : self.max_len]
        x = self.embedding(token_ids)
        pos = self.pos_embed[:, : x.size(1), :]
        x = self.transformer(x + pos)
        x = x.mean(dim=1)
        return self.proj(x)


def update_target_network(source: nn.Module, target: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p_src, p_tgt in zip(source.parameters(), target.parameters()):
            p_tgt.data.mul_(decay).add_(p_src.data, alpha=1.0 - decay)


def pairwise_fusion(z_list: list[torch.Tensor]) -> torch.Tensor:
    """Fuse multiple latent representations via concatenation and projection."""

    assert len(z_list) > 0
    z = torch.cat(z_list, dim=-1)
    return z
