from __future__ import annotations

import os
import sys
import math
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Ensure vjepa2 src is importable regardless of working directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VJEPA2 = os.path.join(_ROOT, "vjepa2")
_VJEPA2_SRC = os.path.join(_VJEPA2, "src")
for p in (_VJEPA2, _VJEPA2_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from vjepa2.src.models.vision_transformer import VisionTransformer

class VJEPA2VisionEncoder(nn.Module):
    """Wrapper around V-JEPA2 VisionTransformer to produce a single latent vector per image.

    This wraps the official VisionTransformer and mean-pools patch tokens, followed by an optional
    projection to a desired latent dimension.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        img_size: tuple[int, int] = (256, 256),
        patch_size: int = 16,
        depth: int = 12,
        heads: int = 8,
        in_chans: int = 3,
        proj_to_latent: bool = False,
    ) -> None:
        super().__init__()
        
        # Use the VisionTransformer with the same embedding dim as latent_dim by default to avoid extra projection
        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=1,
            in_chans=in_chans,
            embed_dim=latent_dim,
            depth=depth,
            num_heads=heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            handle_nonsquare_inputs=True,
            use_rope=False,
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Identity() if not proj_to_latent else nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B, C, T, H, W)
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise RuntimeError(f"VJEPA2VisionEncoder expected input of shape (B,C,H,W) or (B,C,T,H,W), got {x.shape}")
        if x.shape[2] == 1:
            # Backbone was trained with tubelet_size=2; duplicate single frame along time
            x = torch.cat([x, x], dim=2)
        tokens = self.backbone(x)
        if tokens.ndim == 4:
            # (B, T, N, D) -> average over time
            tokens = tokens.mean(dim=1)
        if tokens.ndim != 3:
            raise RuntimeError(f"VJEPA2VisionEncoder expected tokens of shape (B,N,D), got {tokens.shape}")
        z = tokens.mean(dim=1)
        z = self.norm(z)
        return self.proj(z)


class VJEPA2HubEncoder(nn.Module):
    """Load a pretrained V-JEPA2 encoder via torch.hub and expose pooled tokens as a latent.

    Args:
        latent_dim: Desired output latent dimension. If it differs from encoder.embed_dim, a linear proj is used.
        variant: Name of the torch.hub entrypoint (default: ``"vjepa2_ac_vit_giant"`` trained on robotics data).
        pretrained: Whether to load pretrained weights (passed to hub entrypoint when supported).
        freeze: If True, freezes the encoder parameters (minimal fine-tuning mode).
        hub_repo: Torch hub repo name to load from (default ``"facebookresearch/vjepa2"``).
        cache_dir: Optional hub cache directory override.
        
    Note:
        Meta's pretrained encoder expects ImageNet-normalized inputs. This class automatically
        applies ImageNet normalization to inputs in [0, 1] range.
    """
    
    # ImageNet normalization constants (Meta VJEPA2-AC uses ImageNet-normalized inputs)
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        latent_dim: int = 256,
        variant: str = "vjepa2_ac_vit_giant",
        pretrained: bool = True,
        freeze: bool = True,
        hub_repo: str = "facebookresearch/vjepa2",
        cache_dir: str | None = None,
        img_size: tuple[int, int] | int = 256,
        patch_size: int = 16,
        tubelet_size: int = 2,
        num_frames: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        # Register ImageNet normalization buffers (not parameters, just constants)
        self.register_buffer('_imagenet_mean', self.IMAGENET_MEAN, persistent=False)
        self.register_buffer('_imagenet_std', self.IMAGENET_STD, persistent=False)

        load_kwargs: dict[str, object] = {}
        if cache_dir is not None:
            load_kwargs["hub_dir"] = cache_dir
        load_kwargs["patch_size"] = int(patch_size)
        load_kwargs["tubelet_size"] = int(tubelet_size)
        if num_frames is not None:
            load_kwargs["num_frames"] = int(num_frames)
        # Some hub entrypoints do not accept a `pretrained` kwarg; fall back gracefully.
        def _hub_load() -> tuple[torch.nn.Module, torch.nn.Module | None]:
            try:
                return torch.hub.load(hub_repo, variant, pretrained=pretrained, **load_kwargs)
            except TypeError:
                return torch.hub.load(hub_repo, variant, **load_kwargs)

        # Avoid concurrent extraction/rmtree conflicts when multiple ranks download
        ddp_world = 1
        ddp_rank = 0
        if dist is not None and dist.is_available() and dist.is_initialized():
            ddp_world = dist.get_world_size()
            ddp_rank = dist.get_rank()

        if ddp_world > 1:
            if ddp_rank == 0:
                encoder, predictor = _hub_load()
            dist.barrier()
            if ddp_rank != 0:
                encoder, predictor = _hub_load()
            dist.barrier()
        else:
            encoder, predictor = _hub_load()
        self.backbone = encoder
        # Register predictor as a module if it's an nn.Module (for proper .to(device) handling)
        # This ensures it gets moved to the correct device automatically
        if predictor is not None and isinstance(predictor, nn.Module):
            self.add_module("predictor", predictor)
            self.predictor = predictor
        else:
            self.predictor = predictor

        embed_dim = getattr(self.backbone, "embed_dim", latent_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Identity() if embed_dim == latent_dim else nn.Linear(embed_dim, latent_dim)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.norm.parameters():
                p.requires_grad = False
            for p in self.proj.parameters():
                p.requires_grad = False
            self.backbone.eval()
    
    def _imagenet_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization to input tensor.
        
        Meta's pretrained VJEPA2-AC encoder expects ImageNet-normalized inputs.
        Input: tensor in [0, 1] range with shape (..., C, H, W)
        Output: normalized tensor with same shape
        """
        # Handle different input shapes by normalizing along channel dimension
        # _imagenet_mean and _imagenet_std are (1, 3, 1, 1)
        if x.ndim == 4:  # (B, C, H, W)
            return (x - self._imagenet_mean) / self._imagenet_std
        elif x.ndim == 5:  # (B, T, C, H, W)
            # Reshape mean/std to broadcast correctly
            mean = self._imagenet_mean.unsqueeze(0)  # (1, 1, 3, 1, 1)
            std = self._imagenet_std.unsqueeze(0)    # (1, 1, 3, 1, 1)
            return (x - mean) / std
        else:
            raise RuntimeError(f"_imagenet_normalize expects 4D or 5D input, got {x.ndim}D")

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # (B, C, H, W) -> (B, 1, C, H, W)
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise RuntimeError(
                f"VJEPA2HubEncoder.encode_patches expected input of shape (B,T,C,H,W) or (B,C,H,W), got {x.shape}"
            )
        b, t, c, h, w = x.shape
        # Apply ImageNet normalization (Meta's encoder expects this)
        x = self._imagenet_normalize(x)
        # Flatten time into batch: (B*T, C, H, W)
        frames = x.reshape(b * t, c, h, w)
        # Upsample frames to the backbone's native resolution (e.g., 256x256) so that
        # the patch grid (H/patch_size x W/patch_size) matches the predictor's
        # grid_height * grid_width.
        target_h = int(getattr(self.backbone, "img_height", h))
        target_w = int(getattr(self.backbone, "img_width", w))
        if (h, w) != (target_h, target_w):
            frames = F.interpolate(
                frames,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        # Match upstream world_model_wrapper: duplicate each frame along time to satisfy tubelet_size=2
        clip = frames.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # (B*T, C, 2, H', W')
        # Avoid building autograd graph when encoder backbone is frozen
        requires_grad = any(p.requires_grad for p in self.backbone.parameters())
        ctx = nullcontext() if requires_grad else torch.no_grad()
        with ctx:
            tokens = self.backbone(clip)
        # Backbone returns (B*T, N, D) or (B*T, T_p, N_p, D); flatten any temporal dim
        if tokens.ndim == 4:
            tokens = tokens.flatten(1, 2)
        if tokens.ndim != 3:
            raise RuntimeError(
                f"VJEPA2HubEncoder.encode_patches expected tokens of shape (B*T,N,D), got {tokens.shape}"
            )
        # Reshape back to (B, T, N_patches, D)
        return tokens.view(b, t, -1, tokens.size(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise RuntimeError(f"VJEPA2HubEncoder expected input of shape (B,C,H,W) or (B,C,T,H,W), got {x.shape}")
        # Apply ImageNet normalization (Meta's encoder expects this)
        x = self._imagenet_normalize(x)
        if x.shape[2] == 1:
            x = torch.cat([x, x], dim=2)
        tokens = self.backbone(x)
        if tokens.ndim == 4:
            tokens = tokens.mean(dim=1)
        if tokens.ndim != 3:
            raise RuntimeError(f"VJEPA2HubEncoder expected tokens of shape (B,N,D), got {tokens.shape}")
        z = tokens.mean(dim=1)
        z = self.norm(z)
        return self.proj(z)