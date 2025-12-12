from __future__ import annotations
import os
import time
import random
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
import torch.backends.cudnn as cudnn
from torch import amp
from contextlib import nullcontext
from tqdm import tqdm

from ot_jepa.models.encoders import (
    VisionEncoder,
    StateEncoder,
    LangEncoder,
    update_target_network,
)
from ot_jepa.models.metric import MetricNet
from ot_jepa.models.jepa import TemporalPredictor, GoalDistributionHead
from ot_jepa.models.ot_losses import sliced_w2, sinkhorn_w2, string_prior, cross_modal_ot, goal_w2
from ot_jepa.models.gromov_wasserstein import (
    representation_alignment_loss,
    batch_ot_coupling,
    bilevel_ot_contrastive_loss,
    entropic_gromov_wasserstein,
)
from ot_jepa.models.flow_matching import FlowMatchingHead
from ot_jepa.data.buffers import EpisodeDataset, WindowSpec


class OTJEPAModel(nn.Module):
    def __init__(self, state_dim: int, cfg: dict, act_dim: int = None):
        super().__init__()
        d = cfg["model"]["latent_dim"]
        # Use dataset-inferred action dim (7D for VJEPA2-AC with ee_delta)
        # Falls back to config if not provided (for backward compatibility)
        if act_dim is None:
            act_dim = int(cfg["model"].get("act_dim", 7))
        self.act_dim = act_dim
        # Online encoders
        v_patch = int(cfg["model"].get("patch_size", 16))
        v_depth = int(cfg["model"].get("vision_depth", 4))
        v_heads = int(cfg["model"].get("vision_heads", 4))
        vision_backbone = str(cfg["model"].get("vision_backbone", "internal")).lower()
        if vision_backbone == "vjepa2":
            # Lazy import to avoid requiring vjepa2 when not used
            from ot_jepa.models.vjepa2_backbone import VJEPA2VisionEncoder
            self.E_v = VJEPA2VisionEncoder(
                latent_dim=d,
                img_size=tuple(cfg.get("data", {}).get("image_size", (256, 256))),
                patch_size=v_patch,
                depth=v_depth,
                heads=v_heads,
            )
        elif vision_backbone == "vjepa2_hub":
            from ot_jepa.models.vjepa2_backbone import VJEPA2HubEncoder
            v2 = cfg.get("vjepa2", {})
            self.E_v = VJEPA2HubEncoder(
                latent_dim=d,
                variant=str(v2.get("variant", "vjepa2_ac_vit_giant")),
                pretrained=bool(v2.get("pretrained", True)),
                freeze=bool(v2.get("freeze_encoder", True)),
                img_size=tuple(cfg.get("data", {}).get("image_size", (256, 256))),
                patch_size=v_patch,
            )
        else:
            self.E_v = VisionEncoder(latent_dim=d, patch_size=v_patch, depth=v_depth, heads=v_heads)
        self.E_s = StateEncoder(in_dim=state_dim, latent_dim=d)
        self.E_l = LangEncoder(vocab_size=512, emb_dim=d, latent_dim=d)
        # Multi-view fusion: front + wrist + state -> d
        self.Fusion = nn.Sequential(nn.Linear(d * 3, d), nn.ReLU(), nn.Linear(d, d))
        # Single-view fusion: vision + state -> d (for independent camera training)
        self.Fusion_single = nn.Sequential(nn.Linear(d * 2, d), nn.ReLU(), nn.Linear(d, d))
        # Prediction and control heads
        # V-JEPA2-AC variants use the official hub action-conditioned predictor attached to E_v.
        # OT-JEPA and related architectures use a simple temporal predictor over latents.
        arch = cfg.get("model", {}).get("architecture", "ot-jepa").lower()
        self.use_action_conditioning = arch in ("vjepa2ac-baseline", "vjepa2ac-continued", "vjepa2ac-unfreeze", "vjepa2ac-ot")
        # VJEPA2-AC hub predictor operates on patch tokens for MPC planning
        self.use_patch_token_mpc = self.use_action_conditioning
        # Only non action-conditioned JEPA/OT architectures rely on this latent-level predictor.
        # For V-JEPA2-AC variants, self.Pred is not used in training.
        if self.use_action_conditioning:
            self.Pred = None
        else:
            self.Pred = TemporalPredictor(latent_dim=d)
        self.Metric = MetricNet(d, cfg["model"]["metric_rank"])
        self.FM = FlowMatchingHead(d, act_dim)
        self.GoalHead = GoalDistributionHead(latent_dim=d)
        # Projection head for bilevel batch-OT (R^d -> R^64)
        self.OTProj = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 64))


def set_seed(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_latest_checkpoint(arch: str, directory: str = "checkpoints") -> tuple[str | None, int]:
    if not os.path.isdir(directory):
        return None, 0
    prefix = f"{arch}_"
    latest_step = -1
    latest_path: str | None = None
    for name in os.listdir(directory):
        if not name.startswith(prefix) or not name.endswith(".pt"):
            continue
        step_str = name[len(prefix):-3]
        try:
            step_val = int(step_str)
        except ValueError:
            continue
        if step_val > latest_step:
            latest_step = step_val
            latest_path = os.path.join(directory, name)
    return latest_path, max(latest_step, 0)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    # Optional overrides in dot-notation, e.g.:
    #   --override train.total_steps=100 device=cpu
    ap.add_argument("--override", type=str, nargs="*", default=[])
    return ap.parse_args()


def _parse_override_value(raw: str):
    """Best-effort scalar parsing for override values.

    Attempts bool -> int -> float -> str, so that common CLI values
    like "true", "123", or "1e-4" map to the expected Python types.
    """
    text = str(raw).strip()
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply --override key=value updates into the nested config dict.

    Keys use dot-notation, e.g. "train.total_steps". Missing intermediate
    dictionaries are created on demand.
    """
    for item in overrides:
        if "=" not in item:
            continue
        key_str, value_str = item.split("=", 1)
        key_str = key_str.strip()
        if not key_str:
            continue
        keys = key_str.split(".")
        d: dict = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        leaf = keys[-1]
        d[leaf] = _parse_override_value(value_str)


def init_distributed() -> tuple[int, int, int]:
    # Prefer torchrun/torch.distributed.run env, fallback to SLURM when needed
    has_torchrun = "WORLD_SIZE" in os.environ and "RANK" in os.environ
    has_slurm = "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ
    if has_torchrun or has_slurm:
        if has_torchrun:
            rank = int(os.environ["RANK"])  # global rank
            world_size = int(os.environ["WORLD_SIZE"])  # total processes
            local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
        else:
            # Single srun per node; use SLURM env as a fallback
            rank = int(os.environ.get("SLURM_PROCID", 0))
            world_size = int(os.environ.get("SLURM_NTASKS", 1))
            local_rank = int(os.environ.get("SLURM_LOCALID", rank % max(1, torch.cuda.device_count())))
            # Provide sane defaults if not set
            os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1"))
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("RANK", str(rank))
            os.environ.setdefault("WORLD_SIZE", str(world_size))
            os.environ.setdefault("LOCAL_RANK", str(local_rank))

        # Set device before creating process group
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
        return rank, world_size, local_rank
    return 0, 1, 0


def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config))
    # Apply any command-line overrides after reading the base config
    if getattr(args, "override", None):
        _apply_overrides(cfg, args.override)
    rank, world_size, local_rank = init_distributed()
    is_distributed = world_size > 1
    set_seed(cfg["seed"], rank)
    is_main = (rank == 0)

    if cfg["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda", local_rank if torch.cuda.device_count() > 0 else 0)
        cudnn.benchmark = True
        # Enable TF32 for faster matmuls on Ampere+ GPUs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        device = torch.device("cpu")

    # Data - aligned with Meta VJEPA2-AC clip lengths
    os.makedirs(cfg["data"]["episode_dir"], exist_ok=True)
    window = WindowSpec(k=cfg["data"]["window_k"], H=cfg["data"]["horizon_H"])
    
    # V-JEPA 2 data augmentation parameters from config
    random_resize_scale = tuple(cfg["data"].get("random_resize_scale", [0.3, 1.0]))
    random_resize_aspect_ratio = tuple(cfg["data"].get("random_resize_aspect_ratio", [0.75, 1.35]))
    horizontal_flip = bool(cfg["data"].get("horizontal_flip", False))
    augment = bool(cfg["data"].get("augment", True))  # Enable augmentation by default
    
    ds = EpisodeDataset(
        cfg["data"]["episode_dir"],
        window,
        cfg["data"].get("image_size", (256, 256)),
        rank=rank,
        world_size=world_size,
        use_embeddings=bool(cfg["data"].get("use_embeddings", False)),
        # V-JEPA 2 data augmentation
        random_resize_scale=random_resize_scale,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        horizontal_flip=horizontal_flip,
        augment=augment,
    )
    if len(ds)==0:
        raise RuntimeError("No episodes found; run data.py first.")
    
    # Detect dataset format for logging
    dataset_format = "unknown"
    if ds.episodes:
        import pandas as pd
        first_ep = ds.episodes[0]
        parquet_path = os.path.join(first_ep, "episode.parquet")
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            has_ee_state = any(c.startswith("state_ee_state") for c in df.columns)
            has_ee_delta = any(c.startswith("act_ee_delta") for c in df.columns)
            if has_ee_state and has_ee_delta:
                dataset_format = "7D end-effector (VJEPA2-AC format)"
            elif any(c.startswith("state_q") for c in df.columns):
                dataset_format = "9D joint-space (legacy format)"
    
    if is_main:
        print(f"Dataset: {len(ds)} episodes, state_dim={ds.state_dim}, action_dim={ds.action_dim}")
        print(f"Dataset format: {dataset_format}")
        if ds.state_dim == 9 or ds.action_dim == 9:
            print(f"Using legacy 9D format. Consider regenerating dataset for 7D end-effector format.")
        # Log V-JEPA 2 augmentation settings
        if augment:
            print(f"V-JEPA 2 Augmentation: ENABLED")
            print(f"RandomResizedCrop scale: {random_resize_scale}")
            print(f"RandomResizedCrop aspect ratio: {random_resize_aspect_ratio}")
            print(f"Horizontal flip: {horizontal_flip}")
        else:
            print(f"V-JEPA 2 Augmentation: DISABLED")
    
    # Synchronize inferred dims across ranks to avoid model shape mismatches
    if is_distributed:
        sd = torch.tensor([int(ds.state_dim)], device=device)
        ad = torch.tensor([int(ds.action_dim)], device=device)
        dist.all_reduce(sd, op=dist.ReduceOp.MAX)
        dist.all_reduce(ad, op=dist.ReduceOp.MAX)
        ds.state_dim = int(sd.item())
        ds.action_dim = int(ad.item())
    # Select architecture
    arch = cfg.get("model", {}).get("architecture", "ot-jepa").lower()
    # Optional fast-mode shortcuts for compute-constrained runs. These only
    # adjust hyperparameters (not loss structure) and are fully controlled by config.
    fast_mode = bool(cfg.get("train", {}).get("fast_mode", False))
    if fast_mode:
        mconf = cfg.setdefault("model", {})
        oconf = cfg.setdefault("ot", {})
        # Smaller action-conditioned predictor for V-JEPA2-AC variants
        if arch in ("vjepa2ac-baseline", "vjepa2ac-continued", "vjepa2ac-unfreeze", "vjepa2ac-ot"):
            mconf.setdefault("pred_hidden_dim", 384)
            mconf.setdefault("pred_layers", 6)
            mconf.setdefault("pred_heads", 6)
        # OT hyperparameters: fewer projections / iterations by default
        oconf.setdefault("num_projections_time", 32)
        oconf.setdefault("num_projections_xmod", 16)
        oconf.setdefault("iters", 10)
        oconf.setdefault("batch_ot_iters", 10)
        oconf.setdefault("bilevel_iters", 10)
        oconf.setdefault("gw_iters", 10)
    # Ensure act_dim aligns with dataset
    cfg.setdefault("model", {})
    cfg["model"]["act_dim"] = int(ds.action_dim)
    # Sensible default backbones per arch
    vb = str(cfg["model"].get("vision_backbone", "internal")).lower()
    if arch in ("vjepa", "ot-vjepa") and vb == "internal":
        cfg["model"]["vision_backbone"] = "vjepa2_hub"
        vb = "vjepa2_hub"
    # For all V-JEPA2-AC variants, force the official robotics-trained hub backbone
    # so that continued pretraining always starts from vjepa2_ac_vit_giant weights.
    if arch in ("vjepa2ac-baseline", "vjepa2ac-continued", "vjepa2ac-unfreeze", "vjepa2ac-ot"):
        cfg["model"]["vision_backbone"] = "vjepa2_hub"
        v2 = cfg.setdefault("vjepa2", {})
        v2.setdefault("variant", "vjepa2_ac_vit_giant")
        v2.setdefault("pretrained", True)
    # Model (online)
    if is_main:
        print(f"Building model (arch='{arch}', backbone='{cfg['model'].get('vision_backbone','internal')}') ...")
    # JEPA or OT-JEPA use the same backbone; OT only affects losses
    model = OTJEPAModel(ds.state_dim, cfg, act_dim=ds.action_dim).to(device)
    
    if is_main:
        print(f"Model created with state_dim={ds.state_dim}, action_dim={ds.action_dim}")
    if is_main:
        print("Model built; moving to optional compile if enabled ...")
    # Variant-specific freezing for VJEPA2-AC
    def _freeze(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False
    def _unfreeze(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = True
    if arch in ("vjepa2ac-baseline", "vjepa2ac-continued", "vjepa2ac-unfreeze", "vjepa2ac-ot"):
        ev = getattr(model, "E_v", None)
        has_hub_predictor = (ev is not None and hasattr(ev, "predictor") and ev.predictor is not None)
        
        if arch == "vjepa2ac-baseline":
            if is_main:
                print("Variant vjepa2ac-baseline: training FM head only; freezing encoders and predictors.")
            _freeze(model)
            _unfreeze(model.FM)
            # Unfreeze fusion layers as they are new and random initialized
            _unfreeze(model.Fusion)
            _unfreeze(model.Fusion_single)
            _unfreeze(model.E_s) # State encoder is also new/random
            _unfreeze(model.E_l) # Lang encoder is also new/random
            # Keep hub predictor frozen (no training)
            if has_hub_predictor:
                _freeze(ev.predictor)
                ev.predictor.eval()
        elif arch == "vjepa2ac-continued":
            if is_main:
                print("Variant vjepa2ac-continued: freezing vision backbone; training hub predictor/FM/fusion/state.")
            # Canonical VJEPA2-AC continued training: freeze encoder, train predictor
            if ev is not None and hasattr(ev, "backbone"):
                _freeze(ev.backbone)
                _freeze(ev.norm)
                _freeze(ev.proj)
                ev.backbone.eval()
            # Unfreeze hub predictor for training
            if has_hub_predictor:
                _unfreeze(ev.predictor)
                ev.predictor.train()
            _unfreeze(model.E_s)
            _unfreeze(model.E_l)
            _unfreeze(model.Fusion)
            _unfreeze(model.Fusion_single)  # Unfreeze single-view fusion layer
            _unfreeze(model.FM)
        elif arch == "vjepa2ac-unfreeze":
            if is_main:
                print("Variant vjepa2ac-unfreeze: training all components (encoder + predictor + FM).")
            _unfreeze(model)
            # Unfreeze hub encoder backbone and predictor
            if ev is not None and hasattr(ev, "backbone"):
                _unfreeze(ev.backbone)
                ev.backbone.train()
            if has_hub_predictor:
                _unfreeze(ev.predictor)
                ev.predictor.train()
        elif arch == "vjepa2ac-ot":
            if is_main:
                print("Variant vjepa2ac-ot: training all + OT losses enabled.")
            _unfreeze(model)
            # For OT variant: unfreeze encoder (for GW alignment) and predictor
            if ev is not None and hasattr(ev, "backbone"):
                _unfreeze(ev.backbone)
                ev.backbone.train()
            if has_hub_predictor:
                _unfreeze(ev.predictor)
                ev.predictor.train()
    # Optional compile for speed (PyTorch 2.x); enabled by default but configurable
    # Use 'reduce-overhead' for better iteration time vs 'max-autotune' for peak throughput
    if bool(cfg.get("train", {}).get("compile", True)) and hasattr(torch, "compile"):
        try:
            compile_mode = str(cfg.get("train", {}).get("compile_mode", "reduce-overhead"))
            model = torch.compile(model, mode=compile_mode)
            if is_main:
                print(f"Model compiled with mode='{compile_mode}'")
        except Exception as e:
            if is_main:
                print(f"torch.compile failed: {e}")
    # Target encoders/fuser (EMA) for JEPA/OT-JEPA
    d = cfg["model"]["latent_dim"]
    v_patch = int(cfg["model"].get("patch_size", 16))
    v_depth = int(cfg["model"].get("vision_depth", 4))
    v_heads = int(cfg["model"].get("vision_heads", 4))
    if arch in ("jepa", "ot-jepa", "ot-vjepa", "vjepa2ac-ot"):
        if is_main:
            print("Building target encoders/fuser (EMA) ...")
        vision_backbone = str(cfg["model"].get("vision_backbone", "internal")).lower()
        if vision_backbone == "vjepa2":
            from ot_jepa.models.vjepa2_backbone import VJEPA2VisionEncoder
            E_v_t = VJEPA2VisionEncoder(
                latent_dim=d,
                img_size=tuple(cfg.get("data", {}).get("image_size", (256, 256))),
                patch_size=v_patch,
                depth=v_depth,
                heads=v_heads,
            ).to(device)
        elif vision_backbone == "vjepa2_hub":
            from ot_jepa.models.vjepa2_backbone import VJEPA2HubEncoder
            v2 = cfg.get("vjepa2", {})
            E_v_t = VJEPA2HubEncoder(
                latent_dim=d,
                variant=str(v2.get("variant", "vjepa2_ac_vit_giant")),
                pretrained=bool(v2.get("pretrained", True)),
                freeze=True,
                img_size=tuple(cfg.get("data", {}).get("image_size", (256, 256))),
                patch_size=v_patch,
            ).to(device)
        else:
            E_v_t = VisionEncoder(latent_dim=d, patch_size=v_patch, depth=v_depth, heads=v_heads).to(device)
        E_s_t = StateEncoder(in_dim=ds.state_dim, latent_dim=d).to(device)
        Fusion_t = nn.Sequential(nn.Linear(d * 3, d), nn.ReLU(), nn.Linear(d, d)).to(device)
        # Single-view fusion target (for independent camera trajectory training)
        Fusion_single_t = nn.Sequential(nn.Linear(d * 2, d), nn.ReLU(), nn.Linear(d, d)).to(device)
        # Initialize targets with online weights
        E_v_t.load_state_dict(model.E_v.state_dict())
        E_s_t.load_state_dict(model.E_s.state_dict())
        Fusion_t.load_state_dict(model.Fusion.state_dict())
        Fusion_single_t.load_state_dict(model.Fusion_single.state_dict())
        if is_main:
            print("Target encoders/fuser ready (including Fusion_single_t for single-view).")
    else:
        E_v_t = E_s_t = Fusion_t = Fusion_single_t = None
    
    # GW alignment infrastructure
    # For OT architectures (including V-JEPA2-AC OT) we keep the
    # original representation-based GW alignment that uses a frozen pretrained encoder.
    E_v_pretrained = None
    gw_ref_weights = None
    use_gw_alignment = bool(cfg.get("ot", {}).get("use_gw_alignment", False))
    # Ensure vision_backbone is defined even for architectures that skip EMA targets
    vision_backbone = str(cfg["model"].get("vision_backbone", "internal")).lower()
    # Representation-based GW path: only for non V-JEPA2-AC OT architectures
    if (
        use_gw_alignment
        and vision_backbone in ("vjepa2_hub")
        # Explicitly exclude vjepa2ac-ot (using EMA target instead of frozen copy)
        and arch not in ("vjepa2ac-ot",)
    ):
        if is_main:
            print(f"Creating frozen pretrained encoder for GW alignment (backbone={vision_backbone}) ...")
        if vision_backbone == "vjepa2_hub":
            from ot_jepa.models.vjepa2_backbone import VJEPA2HubEncoder
            v2 = cfg.get("vjepa2", {})
            E_v_pretrained = VJEPA2HubEncoder(
                latent_dim=d,
                variant=str(v2.get("variant", "vjepa2_ac_vit_giant")),
                pretrained=bool(v2.get("pretrained", True)),
                freeze=True,  # Always frozen for alignment reference
                img_size=tuple(cfg.get("data", {}).get("image_size", (256, 256))),
                patch_size=v_patch,
            ).to(device).eval()
        if is_main:
            print("Frozen pretrained encoder ready for GW alignment.")

    # Weight-Space GW path removed as per user request.
    
    # BEFORE DDP wrapping: ensure hub predictor is on correct device for this rank
    # This ensures DDP replicates the predictor correctly on each GPU
    if hasattr(model, 'E_v') and hasattr(model.E_v, 'predictor') and model.E_v.predictor is not None:
        predictor = model.E_v.predictor
        # Recursively move predictor and ALL its submodules to device
        predictor = predictor.to(device)
        # Explicitly move all named submodules to ensure they're on device
        for name, module in predictor.named_modules():
            if module is not predictor:  # Don't move the root module twice
                if hasattr(module, 'to'):
                    module.to(device)
        # Also check for direct attributes that might be modules (state_encoder, action_encoder, etc.)
        for attr_name in dir(predictor):
            if not attr_name.startswith('_'):
                try:
                    attr = getattr(predictor, attr_name)
                    if isinstance(attr, nn.Module) and attr is not predictor:
                        attr.to(device)
                except (AttributeError, RuntimeError):
                    pass  # Skip attributes that can't be accessed
        model.E_v.predictor = predictor
        if is_main:
            print(f"Hub predictor moved to {device} (before DDP wrapping)")
    
    if is_distributed:
        # gradient_as_bucket_view reduces memory copies
        # static_graph=True enables extra optimizations if model structure is fixed
        find_unused = arch not in ("vjepa2ac-baseline", "vjepa2ac-continued", "vjepa2ac-unfreeze", "vjepa2ac-ot")
        # Only pass device_ids when model is on GPU; for CPU training, DDP handles it automatically
        ddp_kwargs = {
            "find_unused_parameters": find_unused,
            "gradient_as_bucket_view": True,
        }
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
        model = DDP(model, **ddp_kwargs)
        # Ensure all ranks complete DDP wrapping before moving on
        dist.barrier()
        print(f"Rank {rank}: DDP wrapped and synchronized.")

    model_for_losses = model.module if isinstance(model, DDP) else model

    lr = float(cfg["train"]["lr"])  # YAML scientific notation can load as str; ensure optimizer receives float
    # Prefer fused AdamW when available
    try:
        opt = optim.AdamW(model.parameters(), lr=lr, fused=True, weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    except TypeError:
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    
    # Learning rate warmup for faster convergence within 100 iterations
    warmup_steps = int(cfg["train"].get("lr_warmup_steps", 0))
    total_steps = int(cfg["train"]["total_steps"])
    start_lr = float(cfg["train"].get("start_lr", lr * 0.1))
    final_lr = float(cfg["train"].get("final_lr", 1.0e-6))
    
    def get_lr_scale(step):
        """Linear warmup then Cosine decay (V-JEPA 2 schedule)"""
        # 1. Linear Warmup: start_lr -> lr
        if warmup_steps > 0 and step < warmup_steps:
            alpha = step / warmup_steps
            # scale = current_lr / base_lr
            # current_lr = start_lr + alpha * (lr - start_lr)
            return (start_lr + alpha * (lr - start_lr)) / lr
            
        # 2. Cosine Decay: lr -> final_lr
        if step >= total_steps:
            return final_lr / lr
            
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        current_lr = final_lr + (lr - final_lr) * cosine_decay
        return current_lr / lr
    
    # AMP dtype: default bfloat16 for stability; choose via config train.amp_dtype: [bfloat16|float16]
    amp_dtype_name = str(cfg.get("train", {}).get("amp_dtype", "bfloat16")).lower()
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(amp_dtype_name, torch.bfloat16)
    scaler = GradScaler(enabled=(device.type == "cuda" and amp_dtype is torch.float16))

    # Auto-resume from latest checkpoint if available
    latest_ckpt_path, latest_ckpt_step = _find_latest_checkpoint(arch)
    start_step = 0
    if latest_ckpt_path is not None:
        map_location = device
        ckpt = torch.load(latest_ckpt_path, map_location=map_location, weights_only=False)
        model_state = ckpt.get("model")
        if model_state is not None:
            model_for_losses.load_state_dict(model_state, strict=False)
        if E_v_t is not None and "E_v_t" in ckpt:
            E_v_t.load_state_dict(ckpt["E_v_t"], strict=False)
        if E_s_t is not None and "E_s_t" in ckpt:
            E_s_t.load_state_dict(ckpt["E_s_t"], strict=False)
        if Fusion_t is not None and "Fusion_t" in ckpt:
            Fusion_t.load_state_dict(ckpt["Fusion_t"], strict=False)
        if Fusion_single_t is not None and "Fusion_single_t" in ckpt:
            Fusion_single_t.load_state_dict(ckpt["Fusion_single_t"], strict=False)
        opt_state = ckpt.get("optimizer")
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        scaler_state = ckpt.get("scaler")
        if scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except Exception:
                pass
        start_step = int(ckpt.get("step", latest_ckpt_step))
        if is_main:
            print(f"Loaded checkpoint '{latest_ckpt_path}' (step={start_step})")
    else:
        start_step = 0

    per_rank_cfg = cfg["data"].get("per_device_batch_size", None)
    if per_rank_cfg is not None:
        per_rank_batch = int(per_rank_cfg)
        global_batch = per_rank_batch * world_size
    else:
        global_batch = int(cfg["data"]["batch_size"])
        if global_batch % world_size != 0 and is_main:
            print(f"data.batch_size {global_batch} not divisible by world_size {world_size}; using floor per-rank batch.")
        per_rank_batch = max(1, global_batch // world_size)
    if is_main:
        print(f"world_size={world_size}, per_rank_batch={per_rank_batch}, effective_global_batch={per_rank_batch*world_size}")
    K = window.k; H = window.H
    w = cfg["loss_weights"]; total = cfg["train"]["total_steps"]
    iterator = range(start_step, total)
    if is_main:
        iterator = tqdm(iterator, desc="train", initial=start_step, total=total)

    grad_accum = int(cfg.get("train", {}).get("grad_accum_steps", 1))
    teacher_stride = max(1, int(cfg.get("train", {}).get("fm_teacher_stride", 4)))
    if is_main:
        print(f"FM teacher stride={teacher_stride} (teacher runs every {teacher_stride} steps)")

    # Initialize OT-CFM Loss
    fm_loss_fn = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    # Camera view selection: Meta VJEPA2 trains on single camera view (left exocentric)
    # If front_camera_only=True, always use front camera; otherwise randomly select front/wrist
    front_camera_only = cfg.get("data", {}).get("front_camera_only", False)
    if is_main:
        if front_camera_only:
            print("Using FRONT camera only (matching Meta's single-view training)")
        else:
            print("Using random front/wrist view selection (50/50)")

    cumulative_step_time = 0.0
    optimizer_step = 0
    group_start_time = None
    for step in iterator:
        step_start = time.perf_counter()
        if is_main and step == start_step == 0:
            print("Sampling first batch ...", flush=True)
        # Sample batch with Meta VJEPA2-AC aligned parameters:
        # Clips enforced to 5-10 seconds (20-40 frames at 4fps)
        # Goal images sampled with fractional conditioning (0.7, 0.2, 0.1)
        # Per-scene subgoal images used when available (front_goal_grasp.png, 
        #   front_goal_near.png, front_goal_final.png)
        batch = ds.sample_batch(per_rank_batch, sample_goal_images=True, 
                                max_goal_offset=20, use_fractional_goals=True,
                                use_scene_subgoals=True)
        if is_main and step == start_step == 0:
            print("First batch ready; entering forward pass ...", flush=True)
        # Keep CPU copy for teacher to avoid GPU->CPU sync
        cpu_imgs_front = batch["imgs_front"]
        imgs_front = batch["imgs_front"].to(device, non_blocking=True).float().contiguous()  # (B, K+H+1, C, H, W)
        imgs_wrist = batch["imgs_wrist"].to(device, non_blocking=True).float().contiguous()
        state = batch["state"].to(device, non_blocking=True).float()
        token = batch["token"].to(device, non_blocking=True)
        actions = batch.get("actions", torch.zeros(imgs_front.size(0), K+H+1, cfg["model"]["act_dim"], device=device)).to(device, non_blocking=True).float()

        # Batch shape: images (B, T, C, H, W) or precomputed embeddings (B, T, D)
        if imgs_front.ndim == 5:
            B, T, C, H_img, W_img = imgs_front.shape
        else:
            B, T, _ = imgs_front.shape
            C = H_img = W_img = None
        # Encode sequence with ONLINE encoders or precomputed embeddings (multi-view fusion)
        # ImageNet normalization is now handled internally by VJEPA2HubEncoder
        def encode_sequence(Ev, Es, Fusion, imgs_f, imgs_w, s):
            if imgs_f.ndim == 3:
                # Precomputed embeddings: imgs_f/imgs_w are (B, T, D)
                b, t, d = imgs_f.shape
                zf = imgs_f.reshape(b * t, d)
                zw = imgs_w.reshape(b * t, d)
            else:
                b, t, c, hh, ww = imgs_f.shape
                f = imgs_f.reshape(b * t, c, hh, ww)
                wv = imgs_w.reshape(b * t, c, hh, ww)
                # Apply channels_last only to 4D image tensors for better throughput
                f = f.contiguous(memory_format=torch.channels_last)
                wv = wv.contiguous(memory_format=torch.channels_last)
                # If encoder is frozen, avoid building autograd graph for its forward
                ctx = torch.no_grad() if not next(Ev.parameters()).requires_grad else nullcontext()
                with ctx:
                    zf = Ev(f)
                    zw = Ev(wv)
            zs = Es(s.reshape(b * t, -1))
            fused = torch.cat([zf, zw, zs], dim=-1)
            z = Fusion(fused)
            return z.reshape(b, t, -1)
        
        # Encode single camera view (V-JEPA 2 style: independent trajectories)
        # ImageNet normalization is now handled internally by VJEPA2HubEncoder
        def encode_single_view(Ev, Es, Fusion_single, imgs, s):
            """Encode a single camera view.
            Used for training on independent front/wrist trajectories.
            For V-JEPA 2 AC: returns vision only (d), state passed to predictor
            For other architectures: fuses vision+state using single-view fusion (2*d -> d)
            """
            if imgs.ndim == 3:
                # Precomputed embeddings: imgs are (B, T, D)
                b, t, d = imgs.shape
                zv = imgs.reshape(b * t, d)
            else:
                b, t, c, hh, ww = imgs.shape
                imgs_flat = imgs.reshape(b * t, c, hh, ww)
                imgs_flat = imgs_flat.contiguous(memory_format=torch.channels_last)
                ctx = torch.no_grad() if not next(Ev.parameters()).requires_grad else nullcontext()
                with ctx:
                    zv = Ev(imgs_flat)
            # For single-view: fuse vision + state using a 2-input fusion
            # This gives us d-dimensional output matching expected latent_dim
            zs = Es(s.reshape(b * t, -1))
            z = Fusion_single(torch.cat([zv, zs], dim=-1))
            return z.reshape(b, t, -1)

        teacher_start = None
        with amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=(device.type == "cuda")):
            # Initialize logging variables to avoid UnboundLocalError
            gw_weight = 0.0
            
            if arch in ("jepa-hf", "vjepa", "vjepa2ac-baseline"):
                # FM-only fine-tuning on top of a chosen backbone; teacher supervision
                # V-JEPA 2 style: randomly select front OR wrist trajectory (independent training)
                # If front_camera_only=True, always use front camera (matching Meta's single-view training)
                use_wrist_view = False if front_camera_only else (random.random() < 0.5)
                imgs_selected = imgs_wrist if use_wrist_view else imgs_front
                
                # Encode only the final context frame K from selected view
                z_seq_online = encode_single_view(
                    model_for_losses.E_v,
                    model_for_losses.E_s,
                    model_for_losses.Fusion_single,
                    imgs_selected[:, K : K + 1, :, :, :],
                    state[:, K : K + 1, :],
                )
                z_t = z_seq_online[:, -1, :]
                z_l = model_for_losses.E_l(token)
                # FM supervision: OT-CFM
                B = imgs_front.size(0)
                # Infer action dim from FM head's output layer
                fm_net = model_for_losses.FM.net
                pred_dim = fm_net[-1].out_features
                
                # Get x1 (target)
                # For VJEPA2-AC baseline: use ee_delta (7D end-effector deltas) if available
                ee_delta_batch = batch.get("ee_delta")
                if ee_delta_batch is not None:
                    ee_delta_batch = ee_delta_batch.to(device, non_blocking=True)
                    
                    # Zero rotation component to match MPC behavior
                    # Meta's MPC implementation zeros rotation (simplified kinematics)
                    ee_delta_batch = ee_delta_batch.clone()
                    ee_delta_batch[..., 3:6] = 0.0  # Zero rotation deltas
                
                # Use ee_delta for VJEPA2-AC, fall back to joint actions otherwise
                if ee_delta_batch is not None and arch in ("vjepa2ac-baseline",):
                    # Use K+1 for departure action (target)
                    # ee_delta[K] is arrival action (s_K - s_{K-1})
                    # ee_delta[K+1] is departure action (s_{K+1} - s_K)
                    gt = ee_delta_batch[:, K+1, :]  # (B, 7)
                else:
                    gt = actions[:, K, :]
                
                if gt.shape[1] != pred_dim:
                    if gt.shape[1] > pred_dim:
                        gt = gt[:, :pred_dim]
                    else:
                        pad = torch.zeros(B, pred_dim - gt.shape[1], device=device, dtype=gt.dtype)
                        gt = torch.cat([gt, pad], dim=1)
                x1 = gt
                do_teacher = True

                if do_teacher:
                    # 4. Compute Flow State xt and Target Velocity ut using torchcfm
                    x0 = torch.randn_like(x1)
                    t, xt, ut = fm_loss_fn.sample_location_and_conditional_flow(x0, x1)
                    # 5. Predict Velocity
                    # Use goal image for conditioning (Meta's hindsight relabeling)
                    # Use goal corresponding to the selected camera view
                    if use_wrist_view and "goal_imgs_wrist" in batch and batch["goal_imgs_wrist"] is not None:
                        goal_imgs = batch["goal_imgs_wrist"].to(device, non_blocking=True)
                        z_goal = model_for_losses.E_v(goal_imgs)
                    elif not use_wrist_view and "goal_imgs_front" in batch and batch["goal_imgs_front"] is not None:
                        goal_imgs = batch["goal_imgs_front"].to(device, non_blocking=True)
                        z_goal = model_for_losses.E_v(goal_imgs)
                    else:
                        # Fallback to final frame from selected view
                        z_goal = model_for_losses.E_v(imgs_selected[:, K+H, :, :, :])
                    vt = model_for_losses.FM(xt, z_t, z_goal, t)
                    fm_loss = ((vt - ut) ** 2).mean()
                else:
                    fm_loss = torch.zeros(1, device=device)
                    
                if teacher_start is not None:
                    elapsed = time.perf_counter() - teacher_start
                    if is_main:
                        print(f"step={step}: teacher finished in {elapsed:.2f}s", flush=True)
                loss = w.get("fm", 0.5) * fm_loss
            else:
                # JEPA / OT pipelines
                # Also used by vjepa2ac-{continued,unfreeze,ot}
                # V-JEPA 2 style: randomly select front OR wrist trajectory (independent training)
                # If front_camera_only=True, always use front camera (matching Meta's single-view training)
                use_wrist_view = False if front_camera_only else (random.random() < 0.5)
                imgs_selected = imgs_wrist if use_wrist_view else imgs_front
                
                z_seq_online = encode_single_view(
                    model_for_losses.E_v,
                    model_for_losses.E_s,
                    model_for_losses.Fusion_single,
                    imgs_selected[:, : K + 1, :, :, :],
                    state[:, : K + 1, :],
                )
                z_hist = z_seq_online  # (B, K+1, d)
                z_t = z_hist[:, -1, :]
                rollout_loss = torch.tensor(0.0, device=device)

                # For V-JEPA 2-AC variants: use action-conditioned prediction
                if model_for_losses.use_action_conditioning:
                    # Get end-effector data from batch (B, K+H+1, 7)
                    ee_state_batch = batch.get("ee_state")  # (B, K+H+1, 7)
                    ee_delta_batch = batch.get("ee_delta")  # (B, K+H+1, 7)
                    
                    # Ensure end-effector tensors are on the correct device
                    if ee_state_batch is not None:
                        ee_state_batch = ee_state_batch.to(device, non_blocking=True)
                    if ee_delta_batch is not None:
                        ee_delta_batch = ee_delta_batch.to(device, non_blocking=True)
                        
                        # Zero rotation component to match MPC behavior
                        # Meta's MPC implementation zeros rotation (simplified kinematics)
                        # Training must match this to avoid distribution shift
                        ee_delta_batch = ee_delta_batch.clone()
                        ee_delta_batch[..., 3:6] = 0.0  # Zero rotation deltas
                        
                    
                    # Validate dimensions
                    if ee_state_batch is not None and ee_state_batch.shape[-1] != 7:
                        raise ValueError(
                            f"Invalid ee_state dimensions: expected (B, T, 7), got {ee_state_batch.shape}. "
                            f"EE state must be 7D: [pos(3), rot_rpy(3), gripper(1)]"
                        )
                    if ee_delta_batch is not None and ee_delta_batch.shape[-1] != 7:
                        raise ValueError(
                            f"Invalid ee_delta dimensions: expected (B, T, 7), got {ee_delta_batch.shape}. "
                            f"EE delta must be 7D: [dpos(3), drot_rpy(3), dgripper(1)]"
                        )

                    # Decide whether to use the hub AC predictor on patch tokens
                    use_hub_predictor = (
                        hasattr(model_for_losses.E_v, "encode_patches")
                        and getattr(model_for_losses.E_v, "predictor", None) is not None
                        and ee_state_batch is not None
                        and ee_delta_batch is not None
                        and imgs_selected.ndim == 5
                        and not bool(cfg["data"].get("use_embeddings", False))
                    )

                    if use_hub_predictor:
                        # Patch-token-level action-conditioned prediction with hub predictor
                        B_tokens = imgs_selected.size(0)
                        total_T = K + H + 1
                        img_seq = imgs_selected[:, : total_T, :, :, :]  # (B, K+H+1, C, H, W) from selected view
                        patch_tokens = model_for_losses.E_v.encode_patches(img_seq)  # (B, T, N_p, D_enc)
                        Bp, T_p, N_p, D_enc = patch_tokens.shape
                        if Bp != B_tokens or T_p != total_T:
                            raise RuntimeError(
                                f"encode_patches returned shape {(Bp, T_p)} but expected {(B_tokens, total_T)}"
                            )
                        
                        # Apply layer_norm to encoder output (Meta's normalize_reps=True)
                        # Meta's WorldModel.encode() applies F.layer_norm(h, (h.size(-1),)) to encoder output
                        # This must be done BEFORE passing to predictor for consistency with MPC testing
                        patch_tokens = F.layer_norm(patch_tokens, (patch_tokens.size(-1),))

                        # Meta's loss is computed on PATCH TOKENS, not pooled representations:
                        #   loss_fn(z, h) = torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp
                        # where z and h are both (B, T*N_p, D) patch token tensors.
                        #
                        # We implement both:
                        # 1. Teacher-forcing: predict all future frames given all context
                        # 2. Autoregressive: predict step-by-step, feeding predictions back
                        
                        # Get target tokens (from target encoder, already layer_normed)
                        # We need targets for frames K+1 to K+H
                        with torch.no_grad():
                            # Target tokens are the encoder outputs for future frames
                            # patch_tokens shape: (B, T, N_p, D_enc) - already layer_normed
                            target_tokens = patch_tokens[:, K+1:K+1+H, :, :]  # (B, H, N_p, D_enc)
                            target_tokens_flat = target_tokens.reshape(B_tokens, H * N_p, D_enc)
                        
                        # Teacher-Forcing Prediction (Meta's z_tf)
                        # Meta uses ALL frames except the last for TF input:
                        #   _z = z[:, :-tokens_per_frame]  (T-1 frames)
                        #   actions = all T-1 actions
                        #   states = states[:, :-1] (T-1 states)
                        # The predictor predicts the NEXT token for each position
                        
                        # Use T-1 frames (all except last) as input
                        T_tf = total_T - 1  # Number of input frames for TF
                        context_tokens = patch_tokens[:, :T_tf, :, :]  # (B, T-1, N_p, D_enc)
                        x_in = context_tokens.reshape(B_tokens, T_tf * N_p, D_enc)
                        
                        # Actions and states for T-1 frames
                        # Shift actions by +1 (departure action alignment)
                        # ee_delta_batch[:, 1:T_tf+1, :] gives T_tf departure actions
                        actions_in = ee_delta_batch[:, 1:T_tf+1, :]  # (B, T-1, 7)
                        states_in = ee_state_batch[:, :T_tf, :]       # (B, T-1, 7)
                        
                        # Run predictor (teacher-forcing: single forward pass)
                        pred_tf = model_for_losses.E_v.predictor(x_in, actions_in, states_in)
                        # pred_tf shape: (B, (T-1)*N_p, D_enc)
                        
                        # Apply layer_norm to predictor output (Meta's normalize_reps=True)
                        pred_tf = F.layer_norm(pred_tf, (pred_tf.size(-1),))
                        
                        # Teacher-forcing loss: compare pred_tf with target_tokens
                        # Meta: _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
                        # This means: target is frames 1 to T-1 (shifted by 1 frame)
                        # pred_tf[i] predicts the token at position i+1 in the sequence
                        tf_target = patch_tokens[:, 1:T_tf, :, :].reshape(B_tokens, (T_tf-1)*N_p, D_enc)
                        tf_loss = torch.mean(torch.abs(pred_tf[:, N_p:, :] - tf_target))
                        
                        # Autoregressive Prediction (Meta's z_ar)
                        # Meta starts with [frame0_GT, frame1_from_TF] and continues from there
                        # Use the TF prediction for frame 1, not a fresh prediction
                        auto_steps = min(H, total_T - 1)  # Number of AR steps (predict frames 2, 3, ...)
                        
                        # Start with frame 0 (GT) + frame 1 (from TF prediction)
                        # pred_tf[:, :N_p] is the TF prediction for frame 1
                        pred_frame1_tf = pred_tf[:, :N_p, :].reshape(B_tokens, 1, N_p, D_enc)
                        curr_tokens = torch.cat([
                            patch_tokens[:, :1, :, :],  # Frame 0 (GT)
                            pred_frame1_tf              # Frame 1 (TF prediction)
                        ], dim=1)  # (B, 2, N_p, D_enc)
                        
                        # Continue autoregressive rollout for frames 2, 3, ...
                        for n in range(1, auto_steps):
                            T_curr = curr_tokens.shape[1]
                            x_in_ar = curr_tokens.reshape(B_tokens, T_curr * N_p, D_enc)
                            # Actions: need T_curr actions for T_curr frames
                            # ee_delta_batch[:, 1:T_curr+1, :] gives actions [a_1, a_2, ..., a_{T_curr}]
                            # which are departure actions from states [s_0, s_1, ..., s_{T_curr-1}]
                            actions_ar = ee_delta_batch[:, 1:T_curr+1, :]  # (B, T_curr, 7)
                            states_ar = ee_state_batch[:, :T_curr, :]      # (B, T_curr, 7)
                            pred_ar = model_for_losses.E_v.predictor(x_in_ar, actions_ar, states_ar)
                            pred_ar = F.layer_norm(pred_ar, (pred_ar.size(-1),))
                            pred_next = pred_ar[:, -N_p:, :]  # (B, N_p, D_enc)
                            curr_tokens = torch.cat([curr_tokens, pred_next.unsqueeze(1)], dim=1)
                        
                        # Autoregressive loss: compare predicted tokens with targets
                        # curr_tokens[:, 1:] are the predictions (skip frame 0 which is GT)
                        ar_pred_tokens = curr_tokens[:, 1:, :, :].reshape(B_tokens, -1, D_enc)
                        ar_tgt_tokens = patch_tokens[:, 1:1+auto_steps, :, :].reshape(B_tokens, -1, D_enc)
                        ar_loss = torch.mean(torch.abs(ar_pred_tokens - ar_tgt_tokens))
                        
                        # Combined JEPA loss (Meta: loss = jloss + sloss)
                        jepa_patch_loss = tf_loss + ar_loss
                        
                        # For compatibility with existing code, also compute pooled representations
                        # (used for FM conditioning and other downstream tasks)
                        z_future_pred_frames = []
                        z_future_tgt_frames = []
                        for h in range(1, H + 1):
                            t_pred = K + h
                            if t_pred < total_T:
                                # Predicted: use autoregressive predictions
                                if h <= auto_steps:
                                    pred_tokens_h = curr_tokens[:, h, :, :]  # (B, N_p, D_enc)
                                else:
                                    pred_tokens_h = curr_tokens[:, -1, :, :]  # Use last prediction
                                z_pred = pred_tokens_h.mean(dim=1)
                                z_pred = model_for_losses.E_v.norm(z_pred)
                                z_pred = model_for_losses.E_v.proj(z_pred)
                                z_future_pred_frames.append(z_pred)
                                
                                # Target: ground truth
                                with torch.no_grad():
                                    tgt_tokens_h = patch_tokens[:, t_pred, :, :]
                                    z_tgt = tgt_tokens_h.mean(dim=1)
                                    z_tgt = model_for_losses.E_v.norm(z_tgt)
                                    z_tgt = model_for_losses.E_v.proj(z_tgt)
                                z_future_tgt_frames.append(z_tgt)
                        
                        # Stack pooled representations for FM conditioning
                        if len(z_future_pred_frames) > 0:
                            z_future_pred = torch.stack(z_future_pred_frames, dim=1)  # (B, H, d)
                            z_future_tgt = torch.stack(z_future_tgt_frames, dim=1)    # (B, H, d)
                        else:
                            # Fallback if no predictions were made
                            z_future_pred = torch.zeros(B_tokens, H, D_enc, device=device)
                            z_future_tgt = torch.zeros(B_tokens, H, D_enc, device=device)
                    else:
                        # Action-conditioned variant but hub predictor not available
                        # This can happen if ee_state/ee_delta are missing or using embeddings
                        # Fall back to simple latent-space prediction using the fused latents
                        if is_main and step == start_step:
                            print("use_action_conditioning=True but hub predictor unavailable. "
                                  "Falling back to latent-space prediction. Check that ee_state/ee_delta are in your data.")
                        
                        # Use the fused latent z_t as current state and predict futures
                        # Since we don't have a proper predictor, use a simple identity + noise baseline
                        # This ensures training doesn't crash, but JEPA loss will be high
                        B_fallback = z_t.shape[0]
                        d_fallback = z_t.shape[-1]
                        z_future_pred = z_t.unsqueeze(1).expand(-1, H, -1).clone()  # (B, H, d)
                        
                        # Targets: encode future frames (single-view to match online encoding)
                        with torch.no_grad():
                            z_future_tgt = encode_single_view(
                                model_for_losses.E_v,
                                model_for_losses.E_s,
                                model_for_losses.Fusion_single,
                                imgs_selected[:, K + 1 : K + 1 + H, :, :, :],
                                state[:, K + 1 : K + 1 + H, :],
                            )  # (B, H, d)
                else:
                    # Non action-conditioned JEPA/OT: temporal predictor over latents
                    # z_seq_online was already computed with encode_single_view above (lines 1025-1031)
                    # We reuse that encoding here; do NOT recompute with multi-view encode_sequence
                    z_future_pred = model_for_losses.Pred(z_hist, horizon=H)  # (B, H, d)

                    # Encode targets with EMA TARGET encoders ONLY for future frames
                    # Use single-view encoding to match online path
                    with torch.no_grad():
                        z_future_tgt = encode_single_view(
                            E_v_t,
                            E_s_t,
                            Fusion_single_t,
                            imgs_selected[:, K + 1 : K + 1 + H, :, :, :],
                            state[:, K + 1 : K + 1 + H, :],
                        )  # (B, H, d)

                # Language embedding (once per batch)
                z_l = model_for_losses.E_l(token)

                # Losses
                # For VJEPA2-AC with hub predictor: use PATCH-LEVEL loss (jepa_patch_loss)
                # This matches Meta's training exactly: loss = jloss + sloss on patch tokens
                # For other variants: use pooled representation loss
                if 'jepa_patch_loss' in locals():
                    # Meta VJEPA2-AC: patch-level L1 loss (tf_loss + ar_loss)
                    jepa_loss = jepa_patch_loss
                else:
                    # Non-hub predictor variants: pooled representation loss
                    jepa_loss_mode = str(cfg.get("train", {}).get("jepa_loss", "l2")).lower()
                    if jepa_loss_mode == "l1":
                        jepa_loss = (z_future_pred - z_future_tgt).abs().mean()
                    else:
                        jepa_loss = ((z_future_pred - z_future_tgt) ** 2).mean()
                ot_conf = cfg.get("ot", {})
                method = str(ot_conf.get("method", "sliced")).lower()
                num_proj_time = int(ot_conf.get("num_projections_time", 128))
                num_proj_xmod = int(ot_conf.get("num_projections_xmod", 64))
                # Optional stride to compute OT only every N steps (default 1). This can
                # substantially reduce wall-clock time while still providing OT gradients
                # periodically when enabled.
                ot_step_stride = max(1, int(ot_conf.get("step_stride", 1)))
                do_ot_step = (step % ot_step_stride) == 0
                is_ot = arch in ("ot-jepa", "ot-vjepa", "vjepa2ac-ot")
                if is_ot and do_ot_step:
                    ot_time = torch.zeros(1, device=device)
                    ot_xmod = torch.zeros(1, device=device)
                    string_loss = torch.zeros(1, device=device)
                    goal_loss = torch.zeros(1, device=device)
                    batch_ot_loss = torch.zeros(1, device=device)
                    bilevel_ot_loss = torch.zeros(1, device=device)
                    ot_diag_metrics = {}

                    # Temporal OT (JEPA-style)
                    if w.get("ot_time", 0.0) != 0.0:
                        if method == "sinkhorn":
                            eps = float(ot_conf.get("epsilon", 0.1))
                            iters = int(ot_conf.get("iters", 30))
                            ot_time = sinkhorn_w2(
                                model_for_losses.Metric,
                                z_future_pred,
                                z_future_tgt,
                                epsilon=eps,
                                n_iters=iters,
                            ).mean()
                        else:
                            ot_time = sliced_w2(
                                model_for_losses.Metric,
                                z_future_pred,
                                z_future_tgt,
                                num_projections=num_proj_time,
                            ).mean()

                    # Cross-modal alignment at current time (encode single frame)
                    if w.get("ot_xmod", 0.0) != 0.0:
                        z_v_last = model_for_losses.E_v(imgs_front[:, K, :, :, :])
                        z_s_last = model_for_losses.E_s(state[:, K, :])
                        ot_xmod = cross_modal_ot(
                            model_for_losses.Metric,
                            z_v_last,
                            z_s_last,
                            num_projections=num_proj_xmod,
                        )

                    # String prior on predicted futures
                    if w.get("string", 0.0) != 0.0:
                        length, curvature = string_prior(model_for_losses.Metric, z_future_pred)
                        string_loss = length + 0.1 * curvature

                    # Goal latent alignment
                    if w.get("goal", 0.0) != 0.0:
                        mean_g, logvar_g = model_for_losses.GoalHead(z_l)
                        goal_loss = goal_w2(
                            model_for_losses.Metric,
                            z_future_pred[:, -1, :],
                            mean_g,
                            logvar_g,
                            num_samples=4,
                        )

                    # Batch-level OT and Bilevel OT
                    ot_diag_metrics = {}
                    if w.get("batch_ot", 0.0) != 0.0:
                        batch_ot_eps = float(ot_conf.get("batch_ot_epsilon", 0.1))
                        batch_ot_iters = int(ot_conf.get("batch_ot_iters", 50))
                        # Scale embeddings by 1/sqrt(d) to normalize Euclidean cost to O(1)
                        # This preserves LayerNorm geometry (unlike cosine) while fixing Sinkhorn stability
                        scale_ot = 1.0 / (z_t.shape[-1] ** 0.5)
                        coupling, batch_ot_loss = batch_ot_coupling(
                            z_t * scale_ot,
                            z_future_pred[:, 0, :] * scale_ot,
                            epsilon=batch_ot_eps,
                            n_iters=batch_ot_iters,
                        )
                        ot_diag_metrics = coupling._ot_diagnostics if hasattr(coupling, "_ot_diagnostics") else {}

                    if w.get("bilevel_ot", 0.0) != 0.0:
                        bl_eps = float(ot_conf.get("bilevel_epsilon", 0.1))
                        bl_iters = int(ot_conf.get("bilevel_iters", 50))
                        bl_lvar = float(ot_conf.get("lambda_var", 1.0))
                        bl_lcov = float(ot_conf.get("lambda_cov", 0.5))
                        bl_lunif = float(ot_conf.get("lambda_unif", 0.0))  # Uniformity loss (IOT-CL)
                        bl_gamma = float(ot_conf.get("gamma", 1.0))
                        Xp = model_for_losses.OTProj(z_t)
                        Yp = model_for_losses.OTProj(z_future_pred[:, 0, :])
                        # Scale projections by 1/sqrt(d) for stability (preserves geometry vs normalize)
                        scale_bl = 1.0 / (Xp.shape[-1] ** 0.5)
                        Xp = Xp * scale_bl
                        Yp = Yp * scale_bl
                        
                        # Force float32 for OT stability (logsumexp sensitive to precision)
                        bilevel_ot_loss, bilevel_diag = bilevel_ot_contrastive_loss(
                            Xp.float(),
                            Yp.float(),
                            epsilon=bl_eps,
                            n_iters=bl_iters,
                            lambda_var=bl_lvar,
                            lambda_cov=bl_lcov,
                            lambda_unif=bl_lunif,
                            gamma=bl_gamma,
                        )
                else:
                    ot_time = torch.zeros(1, device=device)
                    ot_xmod = torch.zeros(1, device=device)
                    string_loss = torch.zeros(1, device=device)
                    goal_loss = torch.zeros(1, device=device)
                    batch_ot_loss = torch.zeros(1, device=device)
                    bilevel_ot_loss = torch.zeros(1, device=device)

                # Gromov-Wasserstein alignment
                gw_align_loss = torch.zeros(1, device=device)
                if is_ot and w.get("gw_align", 0.0) != 0.0:
                    gw_eps = float(ot_conf.get("gw_epsilon", 0.05))
                    gw_iters = int(ot_conf.get("gw_iters", 50))
                    if E_v_pretrained is not None:
                        with torch.no_grad():
                            z_v_pretrained = E_v_pretrained(imgs_selected[:, K, :, :, :])
                        z_v_online = model_for_losses.E_v(imgs_selected[:, K, :, :, :])
                        gw_align_loss = representation_alignment_loss(
                            z_v_online.float(),
                            z_v_pretrained.float(),
                            epsilon=gw_eps,
                            n_iters=gw_iters,
                        )
                    elif arch == "vjepa2ac-ot" and E_v_t is not None:
                        # Learned GW Alignment (Self-Distillation):
                        # Project new data onto domain of current encoder (represented by stable EMA target)
                        # and recursively improve alignment (Sinkhorn) to regularize encoder changes.
                        with torch.no_grad():
                            # Use EMA target encoder as the "geometry anchor"
                            z_v_anchor = E_v_t(imgs_selected[:, K, :, :, :])
                        z_v_online = model_for_losses.E_v(imgs_selected[:, K, :, :, :])
                        
                        gw_align_loss = representation_alignment_loss(
                            z_v_online.float(),
                            z_v_anchor.float(),
                            epsilon=gw_eps,
                            n_iters=gw_iters,
                        )

                # Flow-matching supervision to teacher actions (rank 0 only, batched, then broadcast)
                # IMPORTANT: FM training is INDEPENDENT of encoder/predictor training
                # FM loss does NOT backpropagate to encoder/predictor (detached inputs)
                B = imgs_front.size(0)
                # Infer action dim from FM head's output layer
                fm_net = model_for_losses.FM.net
                pred_dim = fm_net[-1].out_features  # Last linear layer's output = act_dim
                # infer from actions if available
                if 'actions' in dir() and actions is not None:
                    pred_dim = actions.shape[-1]

                # For VJEPA2-AC variants: use ee_delta (7D end-effector deltas)
                # For legacy variants: use actions (joint velocities)
                if model_for_losses.use_action_conditioning and ee_delta_batch is not None:
                    # Use K+1 for departure action (same as baseline branch)
                    # ee_delta[K] is arrival action (s_K - s_{K-1})
                    # ee_delta[K+1] is departure action (s_{K+1} - s_K) - what FM should predict
                    gt = ee_delta_batch[:, K+1, :]  # (B, 7)
                else:
                    # Legacy: use joint velocity actions
                    gt = actions[:, K, :]
                
                pred_dim = gt.shape[-1]
                if gt.shape[1] != pred_dim:
                    if gt.shape[1] > pred_dim:
                        gt = gt[:, :pred_dim]
                    else:
                        pad = torch.zeros(B, pred_dim - gt.shape[1], device=device, dtype=gt.dtype)
                        gt = torch.cat([gt, pad], dim=1)
                x1 = gt
                do_teacher = True

                if do_teacher:
                    # Detach latents to prevent FM from affecting encoder/predictor
                    # FM learns independently as a "side policy" using frozen representations
                    z_t_detached = z_t.detach()
                    
                    # Use goal image for conditioning (Meta's hindsight relabeling approach)
                    # Use goal corresponding to the selected camera view
                    if use_wrist_view and "goal_imgs_wrist" in batch and batch["goal_imgs_wrist"] is not None:
                        goal_imgs = batch["goal_imgs_wrist"].to(device, non_blocking=True)
                        z_goal = model_for_losses.E_v(goal_imgs)
                    elif not use_wrist_view and "goal_imgs_front" in batch and batch["goal_imgs_front"] is not None:
                        # Use sampled future goal images (Meta's approach)
                        goal_imgs = batch["goal_imgs_front"].to(device, non_blocking=True)  # (B, C, H, W)
                        z_goal = model_for_losses.E_v(goal_imgs)
                    else:
                        # use final frame from selected view
                        z_goal = model_for_losses.E_v(imgs_selected[:, K+H, :, :, :])
                    z_goal_detached = z_goal.detach()
                    
                    # Compute Flow State xt and Target Velocity ut using torchcfm
                    x0 = torch.randn_like(x1)
                    t, xt, ut = fm_loss_fn.sample_location_and_conditional_flow(x0, x1)
                    # Predict Velocity (FM only - no gradients to encoder/predictor)
                    vt = model_for_losses.FM(xt, z_t_detached, z_goal_detached, t)
                    fm_loss = ((vt - ut) ** 2).mean()
                else:
                    fm_loss = torch.zeros(1, device=device)

                # Metric regularization (only applicable when metric exists)
                metric_reg = (
                    1e-6 * sum((p ** 2).sum() for p in model_for_losses.Metric.parameters())
                    if is_ot
                    else torch.zeros(1, device=device)
                )

                # GW Annealing Strategy
                gw_anneal_strategy = str(ot_conf.get("gw_anneal_strategy", "linear")).lower()
                base_gw_weight = w.get("gw_align", 0.0)
                
                if gw_anneal_strategy == "none":
                    gw_weight = base_gw_weight
                elif gw_anneal_strategy == "cosine":
                    # Cosine decay from base_gw_weight to 0
                    progress = step / total
                    gw_weight = base_gw_weight * 0.5 * (1.0 + np.cos(np.pi * progress))
                else:  # linear (default)
                    # Linear decay from base_gw_weight to 0
                    gw_weight = base_gw_weight * (1.0 - step / total)

                loss = (
                    w.get("jepa", 1.0) * jepa_loss
                    + w.get("ot_time", 0.0) * ot_time
                    + w.get("ot_xmod", 0.0) * ot_xmod
                    + w.get("string", 0.0) * string_loss
                    + w.get("goal", 0.0) * goal_loss
                    + w.get("batch_ot", 0.0) * batch_ot_loss
                    + w.get("bilevel_ot", 0.0) * bilevel_ot_loss
                    + gw_weight * gw_align_loss
                    + w.get("fm", 0.5) * fm_loss  # FM included but inputs are detached
                    + w.get("metric", 0.0) * metric_reg
                    + w.get("rollout", 0.0) * rollout_loss
                )

        # Gradient accumulation and DDP no_sync to reduce all-reduce overhead
        commit = ((step + 1) % grad_accum) == 0
        if (step % grad_accum) == 0:
            opt.zero_grad(set_to_none=True)
            group_start_time = step_start
        sync_ctx = nullcontext()
        if isinstance(model, DDP) and not commit:
            sync_ctx = model.no_sync()
        
        # Single backward pass for all losses
        # FM uses detached inputs (z_t_detached, z_l_detached) so even though
        # fm_loss is in the combined loss, gradients from FM cannot flow to encoder/predictor
        with sync_ctx:
            scaler.scale(loss / grad_accum).backward()
        
        if commit:
            scaler.unscale_(opt)
            # Optional gradient clipping (prevents exploding gradients)
            max_grad_norm = float(cfg.get("train", {}).get("max_grad_norm", 1.0))
            if max_grad_norm > 0:
                grad_norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            else:
                # Compute grad norm for logging even without clipping
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm_value = total_norm ** 0.5
            scaler.step(opt)
            scaler.update()
            
            # Learning rate warmup: scale LR for first warmup_steps
            lr_scale = get_lr_scale(step)
            for param_group in opt.param_groups:
                param_group['lr'] = lr * lr_scale
            
            # EMA updates for target encoders and fusion (both multi-view and single-view)
            if arch in ("jepa", "ot-jepa", "ot-vjepa", "vjepa2ac-ot"):
                decay = float(cfg["model"].get("ema_decay", 0.996))
                update_target_network(model_for_losses.E_v, E_v_t, decay)
                update_target_network(model_for_losses.E_s, E_s_t, decay)
                update_target_network(model_for_losses.Fusion, Fusion_t, decay)
                update_target_network(model_for_losses.Fusion_single, Fusion_single_t, decay)

            optimizer_step += 1
            loss_log = loss.detach().clone()
            fm_log = fm_loss.detach().clone() if isinstance(fm_loss, torch.Tensor) else torch.tensor(float(fm_loss), device=device)
            if is_distributed:
                # Use SUM + divide instead of AVG (Gloo backend doesn't support ReduceOp.AVG)
                dist.all_reduce(loss_log, op=dist.ReduceOp.SUM)
                dist.all_reduce(fm_log, op=dist.ReduceOp.SUM)
                loss_log /= world_size
                fm_log /= world_size

            step_time = time.perf_counter() - (group_start_time if group_start_time is not None else step_start)
            cumulative_step_time += step_time
            avg_step_time = cumulative_step_time / optimizer_step
            total_optimizer_steps = (total + grad_accum - 1) // grad_accum
            remaining_optimizer_steps = max(total_optimizer_steps - optimizer_step, 0)
            eta_minutes = (remaining_optimizer_steps * avg_step_time) / 60.0
            if is_main and (step + 1) % 10 == 0:
                lr = opt.param_groups[0]["lr"]
                stats = (
                    f"step={step + 1}/{total} "
                    f"loss={float(loss_log.cpu()):.5f} "
                    f"fm={float(fm_log.cpu()):.5f} "
                    f"lr={lr:.2e} "
                    f"grad_norm={grad_norm_value:.3f} "
                    f"step_s={step_time:.2f} "
                    f"eta_min={eta_minutes:.1f}"
                )
                # Add OT soft contrastive diagnostics if available
                if 'ot_diag_metrics' in locals() and ot_diag_metrics:
                    ot_stats = (
                        f" | OT: diag_dom={ot_diag_metrics.get('diagonal_dominance', 0):.2f} "
                        f"sep_ratio={ot_diag_metrics.get('separation_ratio', 0):.2f} "
                        f"entropy={ot_diag_metrics.get('coupling_entropy', 0):.3f}"
                    )
                    stats += ot_stats
                print(f"stats {stats}", flush=True)
                iterator.set_postfix(loss=float(loss_log.cpu()), fm=float(fm_log.cpu()), step_s=step_time, grad_norm=grad_norm_value)

        if is_main and (step + 1) % cfg["train"]["save_every"] == 0:
            os.makedirs("checkpoints", exist_ok=True)
            # Move state_dict to CPU to free GPU memory immediately
            state = {
                "model": {k: v.cpu() for k, v in model_for_losses.state_dict().items()},
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step + 1,
            }
            if arch in ("jepa", "ot-jepa", "ot-vjepa", "vjepa2ac-ot"):
                state.update({
                    "E_v_t": {k: v.cpu() for k, v in E_v_t.state_dict().items()},
                    "E_s_t": {k: v.cpu() for k, v in E_s_t.state_dict().items()},
                    "Fusion_t": {k: v.cpu() for k, v in Fusion_t.state_dict().items()},
                    "Fusion_single_t": {k: v.cpu() for k, v in Fusion_single_t.state_dict().items()},
                })
            # Save with atomic write (tmp file + rename)
            ckpt_path = f"checkpoints/{arch}_{step+1}.pt"
            ckpt_tmp = ckpt_path + ".tmp"
            torch.save(state, ckpt_tmp)
            os.rename(ckpt_tmp, ckpt_path)  # Atomic on POSIX systems
            print(f"Saved checkpoint: {ckpt_path}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
