from __future__ import annotations
import os, argparse, yaml, time, numpy as np, torch
import torch.nn.functional as F
import torch.distributed as dist
from pydrake.all import StartMeshcat
import imageio
from collections import deque

from drake_env.scenes import build_scene, sample_randomization
from drake_env.planners import compute_subgoals, plan_and_rollout
from ot_jepa.models.encoders import VisionEncoder, StateEncoder, LangEncoder
from ot_jepa.models.metric import MetricNet
from ot_jepa.models.jepa import TemporalPredictor, GoalDistributionHead
from ot_jepa.models.action_conditioned_predictor import ActionConditionedPredictor
from ot_jepa.models.flow_matching import FlowMatchingHead
from ot_jepa.models.mpc_planner import MPCPlanner

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--randomize", type=float, default=0.5)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--hz", type=int, default=4)
    ap.add_argument("--horizon_H", type=int, default=4)
    ap.add_argument("--window_k", type=int, default=2)
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", type=str, default="pick_place")
    ap.add_argument("--skip-planner-baseline", action="store_true", help="Skip planner baseline (faster)")
    ap.add_argument("--no-meshcat", action="store_true", help="Disable Meshcat visualization (faster)")
    ap.add_argument("--use-mpc", action="store_true", help="Use MPC planning instead of flow matching for action generation")
    ap.add_argument("--mpc-horizon", type=int, default=2, help="MPC planning horizon (Meta default: 2)")
    ap.add_argument("--mpc-samples", type=int, default=256, help="MPC number of samples per iteration (Meta default: 400)")
    ap.add_argument("--mpc-iterations", type=int, default=5, help="MPC number of CEM iterations (Optimized with Warm Start: 5)")
    ap.add_argument("--mpc-top-k", type=int, default=10, help="MPC top-k trajectories for updating (Meta default: 10)")
    ap.add_argument("--ac-hidden-dim", type=int, default=1024, help="Action-conditioned predictor hidden dim (default: 1024)")
    ap.add_argument("--eval-clip-seconds", type=float, default=10.0, 
                    help="Evaluation clip length in seconds (default: 10.0)")
    ap.add_argument("--ac-layers", type=int, default=24, help="Action-conditioned predictor layers (default: 24)")
    ap.add_argument("--ac-heads", type=int, default=16, help="Action-conditioned predictor attention heads (default: 16)")
    ap.add_argument("--position-guidance", type=float, default=0.0, 
                    help="Position guidance weight for MPC (0.0=pure latent, 1.0=pure position, default: 0.5 for pg)")
    ap.add_argument("--position-guidance-phases", type=str, default="all",
                    help="Phases to apply position guidance: 'all' or 'none' (default: all for test-place)")
    # test-place.py: Placement-only evaluation (starts with letter grasped)
    ap.add_argument("--near-fraction", type=float, default=0.70,
                    help="Fraction of trajectory to use NEAR goal (default: 0.70 = 70%%)")
    ap.add_argument("--place-fraction", type=float, default=0.30,
                    help="Fraction of trajectory to use FINAL goal (default: 0.30 = 30%%)")
    ap.add_argument("--start-height", type=float, default=0.40,
                    help="Starting height for gripper with grasped letter (default: 0.40m)")
    return ap.parse_args()


class EvalOTJEPA(torch.nn.Module):
    def __init__(
        self,
        state_dim: int,
        d: int,
        act_dim: int,
        vision_backbone: str = "internal",
        image_size=(256, 256),
        patch_size: int = 16,
        depth: int = 4,
        heads: int = 4,
        vjepa2_cfg: dict | None = None,
        metric_rank: int = 64,
    ):
        super().__init__()
        if str(vision_backbone).lower() == "vjepa2":
            from ot_jepa.models.vjepa2_backbone import VJEPA2VisionEncoder
            self.E_v = VJEPA2VisionEncoder(
                latent_dim=d,
                img_size=tuple(image_size),
                patch_size=int(patch_size),
                depth=int(depth),
                heads=int(heads),
            )
        elif str(vision_backbone).lower() == "vjepa2_hub":
            from ot_jepa.models.vjepa2_backbone import VJEPA2HubEncoder
            cfg = dict(vjepa2_cfg or {})
            variant = str(cfg.get("variant", "vjepa2_ac_vit_giant"))
            pretrained = bool(cfg.get("pretrained", True))
            freeze = bool(cfg.get("freeze_encoder", True))
            hub_repo = str(cfg.get("hub_repo", "facebookresearch/vjepa2"))
            cache_dir = cfg.get("cache_dir")
            self.E_v = VJEPA2HubEncoder(
                latent_dim=d,
                variant=variant,
                pretrained=pretrained,
                freeze=freeze,
                hub_repo=hub_repo,
                cache_dir=cache_dir,
                img_size=tuple(image_size),
                patch_size=int(patch_size),
            )
        else:
            self.E_v = VisionEncoder(latent_dim=d)
        self.E_s = StateEncoder(in_dim=state_dim, latent_dim=d)
        self.E_l = LangEncoder(vocab_size=512, emb_dim=d, latent_dim=d)
        # Multi-view fusion: front + wrist + state -> d
        self.Fusion = torch.nn.Sequential(torch.nn.Linear(d * 3, d), torch.nn.ReLU(), torch.nn.Linear(d, d))
        # Single-view fusion: vision + state -> d (for independent camera training)
        self.Fusion_single = torch.nn.Sequential(torch.nn.Linear(d * 2, d), torch.nn.ReLU(), torch.nn.Linear(d, d))
        # Placeholder for Pred - will be set based on architecture
        self.Pred = None
        self.use_action_conditioning = False
        self.Metric = MetricNet(d, rank=metric_rank)
        self.FM = FlowMatchingHead(d, act_dim)
        self.GoalHead = GoalDistributionHead(latent_dim=d)

def _find_latest_checkpoint(arch: str, directory: str = "checkpoints") -> str | None:
    prefix = f"{arch}_"
    best_step = -1
    best_path: str | None = None
    if not os.path.isdir(directory):
        return None
    for fname in os.listdir(directory):
        if not (fname.startswith(prefix) and fname.endswith(".pt")):
            continue
        step_str = fname[len(prefix) : -3]
        try:
            step = int(step_str)
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_path = os.path.join(directory, fname)
    return best_path

def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config))
    
    # Initialize distributed for multi-GPU MPC inference
    # torch.distributed.run sets RANK, WORLD_SIZE, LOCAL_RANK env vars
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    # For single episode evaluation, distributed overhead isn't worth it
    # Force single-GPU mode for simplicity
    if args.episodes >= 1 and world_size > 1:
        if rank == 0:
            print(f"Single episode evaluation detected")
            print(f"Using single GPU (rank 0) only for simplicity")
            print(f"For multi-episode evaluation, use --episodes > 1 to leverage distributed MPC")
        # Non-rank-0 processes exit immediately
        if rank != 0:
            return
        world_size = 1
    
    # Initialize process group if not already initialized and world_size > 1
    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        # Use NCCL backend for GPU communication
        dist.init_process_group(backend="nccl", init_method="env://")
        if rank == 0:
            print(f"Initialized distributed process group: {world_size} GPUs")
    
    is_main = (rank == 0)
    
    # Verify CUDA availability (only rank 0 prints to avoid clutter)
    if is_main:
        print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA NOT AVAILABLE. Running on CPU!")
        if world_size > 1:
            print(f"Distributed MPC: Each of {world_size} ranks will evaluate {args.mpc_samples // world_size} samples")

    # Set device to this rank's GPU
    if torch.cuda.is_available() and cfg["device"] == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    d = int(cfg["model"]["latent_dim"])
    cfg_act_dim = int(cfg["model"].get("act_dim", 7))
    state_dim = int(cfg["model"].get("state_dim", 7))
    arch = cfg.get("model", {}).get("architecture", "ot-jepa").lower()

    # Model selection (FM-only at eval for all architectures)
    ckpt_path = args.ckpt
    if not ckpt_path:
        ckpt_path = _find_latest_checkpoint(arch)
        if ckpt_path and is_main:
            print(f"loading latest checkpoint: {ckpt_path}")

    # Enforce strict checkpoint loading semantics
    explicit_ckpt = bool(args.ckpt)
    if explicit_ckpt and not ckpt_path:
        if is_main:
            print(f"--ckpt was provided but no checkpoint path could be resolved")
        raise SystemExit(1)
    if explicit_ckpt and not os.path.exists(ckpt_path):
        if is_main:
            print(f"Checkpoint file not found: {ckpt_path}")
        raise SystemExit(1)
    if not explicit_ckpt and not ckpt_path:
        if is_main:
            print(f"No checkpoint found in directory 'checkpoints' for arch='{arch}'")
        raise SystemExit(1)

    ckpt = None
    ckpt_state_dim = None
    ckpt_act_dim = None
    ckpt_metric_rank = None
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_state = ckpt.get("model", ckpt)
        if isinstance(model_state, dict):
            # Handle both compiled (_orig_mod. prefix) and non-compiled checkpoints
            es_weight = model_state.get("E_s.net.0.weight") or model_state.get("_orig_mod.E_s.net.0.weight")
            if isinstance(es_weight, torch.Tensor) and es_weight.ndim == 2:
                ckpt_state_dim = int(es_weight.shape[1])
            if ckpt_state_dim is None:
                for key, tensor in model_state.items():
                    # Strip _orig_mod. prefix if present for matching
                    clean_key = key.replace("_orig_mod.", "")
                    if not (isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and clean_key.startswith("E_s")):
                        continue
                    ckpt_state_dim = int(tensor.shape[1])
                    break
            fm_weight = model_state.get("FM.net.4.weight") or model_state.get("_orig_mod.FM.net.4.weight")
            if isinstance(fm_weight, torch.Tensor) and fm_weight.ndim == 2:
                ckpt_act_dim = int(fm_weight.shape[0])
            if ckpt_act_dim is None:
                for key, tensor in model_state.items():
                    # Strip _orig_mod. prefix if present for matching
                    clean_key = key.replace("_orig_mod.", "")
                    if not (isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and clean_key.startswith("FM")):
                        continue
                    ckpt_act_dim = int(tensor.shape[0])
                    break
            # Infer metric_rank from Metric.backbone.4.weight shape
            # Shape is [rank * latent_dim, hidden], where hidden=256, latent_dim=d
            metric_weight = model_state.get("Metric.backbone.4.weight") or model_state.get("_orig_mod.Metric.backbone.4.weight")
            if isinstance(metric_weight, torch.Tensor) and metric_weight.ndim == 2:
                rank_times_d = int(metric_weight.shape[0])
                # latent_dim (d) will be determined from config, typically 256
                # So we need to defer this calculation until after d is known
                ckpt_metric_rank = rank_times_d  # Store the product for now
    if ckpt_state_dim is not None:
        state_dim = ckpt_state_dim
        if is_main:
            print(f"Inferred state_dim={state_dim} from checkpoint (was {cfg.get('model', {}).get('state_dim', 7)} in config)")
    act_dim = ckpt_act_dim if ckpt_act_dim is not None else cfg_act_dim
    if ckpt_act_dim is not None and act_dim != cfg_act_dim and is_main:
        print(f"Inferred act_dim={act_dim} from checkpoint (was {cfg_act_dim} in config)")
    
    # Infer metric_rank from checkpoint if available
    metric_rank = 64  # Default
    if ckpt_metric_rank is not None:
        # ckpt_metric_rank is rank * latent_dim, so divide by d to get rank
        metric_rank = int(ckpt_metric_rank // d)
        if is_main:
            print(f"Inferred metric_rank={metric_rank} from checkpoint (rank*d={ckpt_metric_rank}, d={d})")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Multi-GPU inference handled via torch.distributed, not DataParallel
    # Each rank has its own model copy on its own GPU
    # Default backbone; override sensible defaults for convenience
    vb = str(cfg.get("model", {}).get("vision_backbone", "internal")).lower()
    if arch in ("vjepa", "ot-vjepa") and vb == "internal":
        vb = "vjepa2_hub"
    elif arch in ("vjepa2ac-baseline", "vjepa2ac-continued", "vjepa2ac-unfreeze", "vjepa2ac-ot") and vb == "internal":
        vb = "vjepa2_hub"
    ps = int(cfg.get("model", {}).get("patch_size", 16))
    vd = int(cfg.get("model", {}).get("vision_depth", 4))
    vh = int(cfg.get("model", {}).get("vision_heads", 4))
    imsz = tuple(cfg.get("data", {}).get("image_size", (256, 256)))
    vjepa2_cfg = cfg.get("vjepa2", {})
    model = EvalOTJEPA(
        state_dim,
        d,
        act_dim,
        vision_backbone=vb,
        image_size=imsz,
        patch_size=ps,
        depth=vd,
        heads=vh,
        vjepa2_cfg=vjepa2_cfg,
        metric_rank=metric_rank,
    ).to(device).eval()
    
    # No DataParallel wrapping needed - each rank has its own model copy in distributed mode
    # Set predictor for eval
    # MPC requires action-conditioned predictor for full rollout capability
    # FM-only uses basic TemporalPredictor
    model_unwrapped = model  # No wrapping in distributed mode
    
    if args.use_mpc:
        # VJEPA2-AC MPC: Use hub predictor with patch-token-level planning (Meta's approach)
        if hasattr(model_unwrapped.E_v, 'predictor') and model_unwrapped.E_v.predictor is not None:
            model_unwrapped.Pred = model_unwrapped.E_v.predictor  # Use continued-pretrained VJEPA2-AC predictor
            model_unwrapped.use_action_conditioning = True
            model_unwrapped.use_patch_token_mpc = True  # Flag for patch-token-based MPC
            if is_main:
                print(f"MPC with hub predictor (VisionTransformerPredictorAC)")
                print(f"Planning operates on patch tokens (Meta's VJEPA2-AC approach)")
                if ckpt is not None:
                    print(f"Using continued-pretrained weights from checkpoint")
                else:
                    print(f"Using hub pretrained weights only (no checkpoint loaded)")
        else:
            # custom predictor for non-VJEPA2-AC architectures
            model_unwrapped.Pred = ActionConditionedPredictor(
                latent_dim=d,
                action_dim=act_dim,
                state_dim=state_dim,
                hidden_dim=args.ac_hidden_dim,
                num_layers=args.ac_layers,
                num_heads=args.ac_heads,
            ).to(device).eval()
            model_unwrapped.use_action_conditioning = True
            model_unwrapped.use_patch_token_mpc = False  # Latent-space MPC
            if is_main:
                print(f"MPC with custom ActionConditionedPredictor (latent-space)")
                print(f"Config: hidden_dim={args.ac_hidden_dim}, layers={args.ac_layers}, heads={args.ac_heads}")
                print(f"Predictor will have random weights unless loaded from checkpoint")
    else:
        # Use basic temporal predictor for FM
        model_unwrapped.Pred = TemporalPredictor(latent_dim=d).to(device).eval()
        model_unwrapped.use_action_conditioning = False
        model_unwrapped.use_patch_token_mpc = False

    if ckpt and "model" in ckpt:
        # Handle torch.compile() prefixes in checkpoint keys
        ckpt_state = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in ckpt_state.keys()):
            # Checkpoint was saved with compiled model; strip _orig_mod. prefix
            ckpt_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt_state.items()}
            if is_main:
                print("Detected compiled checkpoint; stripped _orig_mod. prefixes")
        
        # Check what keys are in checkpoint vs what model expects
        if is_main:
            ckpt_keys = set(ckpt_state.keys())
            model_keys = set(model_unwrapped.state_dict().keys())
            ev_pred_keys = [k for k in ckpt_keys if k.startswith("E_v.predictor")]
            pred_keys = [k for k in ckpt_keys if k.startswith("Pred.")]
            print(f"Checkpoint E_v.predictor.* keys: {len(ev_pred_keys)}")
            print(f"Checkpoint Pred.* keys: {len(pred_keys)}")
            if len(ev_pred_keys) > 0:
                print(f"Sample E_v.predictor keys: {ev_pred_keys[:3]}")

        ckpt_has_predictor = any(k.startswith("E_v.predictor") for k in ckpt_state.keys())
        
        # Capture predictor weights before loading to verify they change
        pred_hash_before = None
        if args.use_mpc and hasattr(model_unwrapped, 'Pred') and model_unwrapped.Pred is not None:
            pred_param = next(model_unwrapped.Pred.parameters())
            pred_hash_before = pred_param.data.sum().item()
        
        missing_keys, unexpected_keys = model_unwrapped.load_state_dict(ckpt_state, strict=False)
        
        # Filter out Pred.* missing keys if they're duplicates of E_v.predictor (same module)
        # This happens because we assign model.Pred = model.E_v.predictor
        if missing_keys:
            filtered_missing = [k for k in missing_keys if not k.startswith("Pred.")]
            pred_missing = [k for k in missing_keys if k.startswith("Pred.")]
            if is_main and pred_missing:
                print(f"{len(pred_missing)} 'Pred.*' keys missing because Pred=E_v.predictor (same module)")
            missing_keys = filtered_missing
        
        if is_main and (missing_keys or unexpected_keys):
            print(f"Checkpoint loading summary:")
            if missing_keys:
                print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
        
        # Verify predictor weights actually changed after loading (only when we expect them to)
        if args.use_mpc and hasattr(model_unwrapped, 'Pred') and model_unwrapped.Pred is not None and pred_hash_before is not None:
            pred_param = next(model_unwrapped.Pred.parameters())
            pred_hash_after = pred_param.data.sum().item()
            delta = abs(pred_hash_after - pred_hash_before)

            # For continued / unfreeze / OT variants, we EXPECT predictor updates if the
            # checkpoint actually contains predictor weights. For baseline, it's
            # acceptable (and expected) that the predictor remains equal to the hub.
            expect_predictor_change = (
                explicit_ckpt
                and ckpt_has_predictor
                and arch in ("vjepa2ac-continued", "vjepa2ac-unfreeze", "vjepa2ac-ot")
            )

            if is_main:
                if delta > 1e-6:
                    print(f"Predictor weights LOADED from checkpoint (param sum: {pred_hash_before:.4f} -> {pred_hash_after:.4f})")
                else:
                    if expect_predictor_change:
                        print(f"Predictor weights UNCHANGED after loading checkpoint (expected updates for arch='{arch}')")
                    else:
                        print(f"Predictor weights unchanged after loading checkpoint (arch='{arch}', frozen hub predictor)")

            if delta <= 1e-6 and expect_predictor_change:
                raise SystemExit(1)
    
    # Optimize model for inference
    model.eval()  # Ensure all modules in eval mode
    # Compile for faster inference (PyTorch 2.0+)
    if hasattr(torch, "compile") and not args.no_meshcat:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            if is_main:
                print("Model compiled for faster inference")
        except Exception as e:
            if is_main:
                print(f"torch.compile failed: {e}")
    
    # Initialize MPC planner if requested (mutually exclusive with FM)
    mpc_planner = None
    if args.use_mpc:
        # ACTION SCALE: The action encoder was trained with larger action magnitudes.
        # Our actions are in [-0.05, 0.05] meters, but the action encoder expects
        # inputs that produce embeddings comparable to visual token embeddings.
        # Visual tokens after predictor_embed have std ≈ 4.6, while action tokens
        # have std ≈ 0.03 with our current action magnitudes.
        # With action_scale=20, visual/action ratio drops from ~160 to ~23.
        # Try action_scale=35 to get ratio closer to ~13 for stronger action influence.
        mpc_planner = MPCPlanner(
            act_dim=act_dim,
            planning_horizon=args.mpc_horizon,
            num_samples=args.mpc_samples,
            num_iterations=args.mpc_iterations,
            top_k=args.mpc_top_k,
            action_l1_max=0.08,  # Larger box constraint for more exploration (8cm)
            action_std_init=0.06, # Larger initial std for better coverage (6cm)
            action_scale=20.0,   # Scale actions to match training distribution
        ).to(device).eval()
        
        # Enable verbose debug for first few MPC calls to diagnose energy landscape
        mpc_planner._verbose_debug = True  # Enable for diagnostics
        
        # MPC planner uses distributed communication (not DataParallel)
        # Each rank evaluates a subset of samples on its own GPU
        
        if is_main:
            print("Policy: Model Predictive Control (MPC)")
            print(f"MPC uses predictor for planning (FM ignored)")
            print(f"Horizon={args.mpc_horizon}, samples={args.mpc_samples}, "
                  f"iterations={args.mpc_iterations}, top_k={args.mpc_top_k}")
            if world_size > 1:
                print(f"Distributed MPC: {args.mpc_samples} samples split across {world_size} GPUs (~{args.mpc_samples // world_size} samples/GPU)")
    else:
        if is_main:
            print("Policy: Flow Matching (FM)")
            print("FM generates actions directly (predictor ignored)")



    def render_rgb(port, root_context):
        sys = port.get_system()
        port_context = sys.GetMyContextFromRoot(root_context)
        img = port.Eval(port_context).data
        arr = np.asarray(img)
        # Match EpisodeLogger._prepare_image EXACTLY to keep train/eval aligned
        # NO FLIPS: Meta's pretrained VJEPA2 expects unflipped images
        # Flipping would destroy the pretrained representations
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            # Drake camera format: (C, H, W) -> (H, W, C)
            arr = np.transpose(arr[:3], (1, 2, 0))
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            # Drop alpha channel for (H, W, 4) -> (H, W, 3)
            arr = arr[..., :3]
        arr = np.clip(arr, 0, 255)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    base_seed = args.seed if args.seed is not None else 4321
    if args.scene:
        scene_seed = abs(hash(args.scene)) % (2**32)
        base_seed = (base_seed + scene_seed) % (2**32)
    rng = np.random.default_rng(base_seed)
    if is_main and args.scene:
        print(f"scene={args.scene} seed={base_seed}")
    elif is_main and args.seed is not None:
        print(f"seed={base_seed}")
    successes_policy, successes_planner = 0, 0
    # Define all available scenes with their configurations
    ALL_SCENES = {
        "scene_a": {
            "scene_type": "franka_reference",
            "drake_task_name": "scene_a",
            "language_eval": "Place the grasped letter P on the top shelf",
            "letter": "P",
        },
        "scene_b": {
            "scene_type": "franka_reference",
            "drake_task_name": "scene_b",
            "language_eval": "Place the grasped letter L on the top shelf",
            "letter": "L",
        },
        "scene_c": {
            "scene_type": "franka_reference",
            "drake_task_name": "scene_c",
            "language_eval": "Place the grasped letter A on the top shelf",
            "letter": "A",
        },
        "scene_d": {
            "scene_type": "franka_reference",
            "drake_task_name": "scene_d",
            "language_eval": "Place the grasped letter C on the top shelf",
            "letter": "C",
        },
    }
    
    # Determine which scenes to run
    if args.scene:
        scene_name = str(args.scene).lower()
        # Normalize scene name
        if scene_name in ("a",):
            scene_name = "scene_a"
        elif scene_name in ("b",):
            scene_name = "scene_b"
        elif scene_name in ("c",):
            scene_name = "scene_c"
        elif scene_name in ("d",):
            scene_name = "scene_d"
        
        if scene_name in ALL_SCENES:
            scenes_to_run = [scene_name]
        else:
            if is_main:
                print(f"unknown scene '{args.scene}', defaulting to all scenes")
            scenes_to_run = list(ALL_SCENES.keys())
    else:
        # No scene specified: run all 4 scenes
        scenes_to_run = list(ALL_SCENES.keys())
        if is_main:
            print(f"No --scene specified, running all {len(scenes_to_run)} scenes: {scenes_to_run}")
            print(f"Total episodes: {len(scenes_to_run) * args.episodes} ({args.episodes} per scene)")
    
    # Determine suffix for output files
    if len(scenes_to_run) == 1:
        suffix = f"{scenes_to_run[0]}_{'mpc' if args.use_mpc else 'fm'}"
    else:
        suffix = f"all_scenes_{'mpc' if args.use_mpc else 'fm'}"
    
    html_path = os.path.join(results_dir, f"{arch}_{suffix}.html")
    mp4_path = os.path.join(results_dir, f"{arch}_{suffix}.mp4")
    
    # Global tracking across all scenes
    all_episode_results = []  # All episodes from all scenes
    global_successes_policy = 0
    global_successes_planner = 0
    global_total_episodes = 0
    
    # Global best episode tracking (across all scenes)
    global_best_episode_idx = -1
    global_best_episode_scene = None
    global_best_episode_min_dist = float('inf')
    global_best_episode_frames: list[np.ndarray] | None = None
    
    # Per-scene results for detailed reporting
    per_scene_results = {}
    
    meshcat = None
    meshcat_can_record = False
    # Only enable meshcat for single-scene evaluation (multi-scene would record all episodes which isn't useful)
    single_scene_mode = len(scenes_to_run) == 1
    if is_main and not args.no_meshcat and single_scene_mode:
        try:
            meshcat = StartMeshcat()
            if hasattr(meshcat, "DeleteRecording"):
                meshcat.DeleteRecording()
            if all(hasattr(meshcat, attr) for attr in ("StartRecording", "StopRecording", "PublishRecording", "StaticHtml")):
                meshcat.StartRecording(set_visualizations_while_recording=False)
                meshcat_can_record = True
            else:
                print("Meshcat version lacks recording support; HTML will be static.")
        except Exception as exc:
            meshcat = None
            print(f"Meshcat start failed: {exc}. Proceeding without HTML export.")
    elif is_main and args.no_meshcat:
        print("Meshcat disabled for faster evaluation")
    elif is_main and not single_scene_mode:
        print("Meshcat disabled for multi-scene evaluation (use --scene to enable)")

    # Episode generation loop
    # Only rank 0 runs Drake simulation; other ranks participate when MPC is called
    if rank == 0:
        # Outer loop over scenes
        for scene_idx, current_scene_name in enumerate(scenes_to_run):
            scene_config = ALL_SCENES[current_scene_name]
            scene_type = scene_config["scene_type"]
            drake_task_name = scene_config["drake_task_name"]
            language_eval = scene_config["language_eval"]
            
            if is_main:
                print(f"\nScene {scene_idx+1}/{len(scenes_to_run)}: {current_scene_name}")
                print(f"Task: {language_eval}")
            
            # Per-scene tracking
            scene_episode_results = []
            scene_successes_policy = 0
            scene_successes_planner = 0
            scene_best_episode_idx = -1
            scene_best_episode_min_dist = float('inf')
            scene_best_episode_frames = None
            
            for ep in range(args.episodes):
                ep_start_time = time.time()
                rand = sample_randomization(args.randomize, rng)
                
                # Planner baseline (optional)
                planner_success = None
                planner_min_distance = None
                if not args.skip_planner_baseline:
                    t_planner_start = time.time()
                    scene = build_scene(
                        drake_task_name,
                        rand,
                        image_size=(256, 256),
                        camera_fps=float(args.hz),
                        meshcat=meshcat,
                        scene_type=scene_type,
                    )
                    subgoals = compute_subgoals(scene, drake_task_name)
                    if scene_type != "franka_reference":
                        baseline = plan_and_rollout(scene, subgoals, episode_length_sec=10.0, hz=args.hz)
                        planner_success = baseline["contacts"][-1]["min_distance"] > 0.005
                        planner_min_distance = baseline["contacts"][-1]["min_distance"]
                        if planner_success:
                            scene_successes_planner += 1
                    if is_main:
                        if scene_type == "franka_reference":
                            print("Skipping planner baseline for franka_reference scene")
                        else:
                            print(f"Planner baseline took {time.time() - t_planner_start:.2f}s")
                else:
                    scene = build_scene(
                        drake_task_name,
                        rand,
                        image_size=(256, 256),
                        camera_fps=float(args.hz),
                        meshcat=meshcat,
                        scene_type=scene_type,
                    )
                    subgoals = compute_subgoals(scene, drake_task_name)
            
                # Track per-episode metrics
                ep_data = {
                    "scene": current_scene_name,
                    "episode": ep,
                    "global_episode_idx": global_total_episodes,
                    "randomize_factor": args.randomize,
                    "randomization": {
                        "light_intensity": rand.light_intensity,
                        "camera_jitter": rand.camera_jitter,
                        "clutter_count": rand.clutter_count,
                        "friction": rand.friction,
                        "block_color": rand.block_color,
                    },
                    "planner_success": planner_success,
                    "planner_min_distance": planner_min_distance,
                }

                # Reset scene for policy rollout
                scene = build_scene(
                    drake_task_name,
                    rand,
                    image_size=(256, 256),
                    camera_fps=float(args.hz),
                    meshcat=meshcat,
                    scene_type=scene_type,
                )
                plant = scene.plant
                sim = scene.simulator
                root_context = scene.context
                plant_context = plant.GetMyMutableContextFromRoot(root_context)
                
                # Reset per-episode frame collection
                if is_main:
                    current_episode_frames = []

                K = int(cfg["data"].get("window_k", args.window_k))
                H = int(cfg["data"].get("horizon_H", args.horizon_H))
                dt = 1.0 / float(args.hz)
                steps = int(10.0 * args.hz)  # Default: 40 timesteps (10.0 s at 4fps)

                # Helper to access model attributes (handles DataParallel wrapping)
                def get_model_attr(model, attr):
                    """Access model attribute, handling DataParallel wrapping."""
                    if isinstance(model, torch.nn.DataParallel):
                        return getattr(model.module, attr)
                    return getattr(model, attr)
                
                # Language/instruction -> token/embedding (for FM policy)
                language = language_eval
                token = torch.tensor([hash(language) % 512], dtype=torch.long, device=device)
                z_l = get_model_attr(model, 'E_l')(token)
                
                # Load goal image for both MPC and FM (Meta's VJEPA2-AC approach)
                # Goal image is stored per scene (not per episode)
                goal_img_front = None
                goal_patches_front = None
                z_goals = []  # List of goal encodings for multi-goal planning
                goal_timesteps = []  # Timesteps per goal
                multi_goal_mode = False
                prev_mpc_mean = None # Warm start state for MPC
                
                # Determine scene directory from current scene being evaluated
                # Use current_scene_name (from scene loop) instead of args.scene
                # This ensures correct goal loading in multi-scene mode
                goal_dir = f"episodes/{current_scene_name}"
                
                # test-place.py: Placement-only - skip GRASP goal, only load NEAR and FINAL
                # Load both front and wrist goals for multi-view conditioning
                multi_goal_files = [
                    # Skip grasp goal - letter is already grasped
                    (("front_goal_near.png", "wrist_goal_near.png"), "near target sub-goal"),
                    (("front_goal_final.png", "wrist_goal_final.png"), "final goal"),
                ]
                
                import imageio
                model_unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
                use_patch_mpc = getattr(model_unwrapped, 'use_patch_token_mpc', False)
                
                # Try to load multi-goal images (both front and wrist views)
                for (goal_file_front, goal_file_wrist), goal_desc in multi_goal_files:
                    goal_path_front = os.path.join(goal_dir, goal_file_front)
                    goal_path_wrist = os.path.join(goal_dir, goal_file_wrist)
                    
                    z_goal_front = None
                    z_goal_wrist = None
                    
                    # Load and encode front goal if available
                    if os.path.exists(goal_path_front):
                        goal_img = imageio.imread(goal_path_front)
                        if is_main:
                            print(f"Found front goal: {goal_path_front}, shape={goal_img.shape}")
                        if goal_img.shape[-1] == 4:
                            goal_img = goal_img[:, :, :3]
                        
                        # ImageNet normalization is handled internally by VJEPA2HubEncoder
                        goal_img_tensor = torch.from_numpy(goal_img / 255.0).to(
                            device=device, dtype=torch.float32, non_blocking=True
                        ).unsqueeze(0).permute(0, 3, 1, 2)  # (1, C, H, W)
                        
                        # Encode as patches or pooled latent based on model type
                        if use_patch_mpc and hasattr(get_model_attr(model, 'E_v'), 'encode_patches'):
                            goal_img_batch = goal_img_tensor.unsqueeze(1)  # (1, 1, C, H, W)
                            z_goal_front = get_model_attr(model, 'E_v').encode_patches(goal_img_batch).squeeze(1)
                            # Apply layer_norm to encoder output (Meta's normalize_reps=True)
                            z_goal_front = F.layer_norm(z_goal_front, (z_goal_front.size(-1),))
                        else:
                            z_goal_front = get_model_attr(model, 'E_v')(goal_img_tensor)
                    elif is_main:
                        print(f"Missing front goal: {goal_path_front}")
                    
                    # Load and encode wrist goal if available
                    if os.path.exists(goal_path_wrist):
                        goal_img = imageio.imread(goal_path_wrist)
                        if is_main:
                            print(f"Found wrist goal: {goal_path_wrist}, shape={goal_img.shape}")
                        if goal_img.shape[-1] == 4:
                            goal_img = goal_img[:, :, :3]
                        
                        # ImageNet normalization is handled internally by VJEPA2HubEncoder
                        goal_img_tensor = torch.from_numpy(goal_img / 255.0).to(
                            device=device, dtype=torch.float32, non_blocking=True
                        ).unsqueeze(0).permute(0, 3, 1, 2)  # (1, C, H, W)
                        
                        # Encode as patches or pooled latent based on model type
                        if use_patch_mpc and hasattr(get_model_attr(model, 'E_v'), 'encode_patches'):
                            goal_img_batch = goal_img_tensor.unsqueeze(1)  # (1, 1, C, H, W)
                            z_goal_wrist = get_model_attr(model, 'E_v').encode_patches(goal_img_batch).squeeze(1)
                            # Apply layer_norm to encoder output (Meta's normalize_reps=True)
                            z_goal_wrist = F.layer_norm(z_goal_wrist, (z_goal_wrist.size(-1),))
                        else:
                            z_goal_wrist = get_model_attr(model, 'E_v')(goal_img_tensor)
                    elif is_main:
                        print(f"Missing wrist goal: {goal_path_wrist}")
                    
                    # Use front goal if available, otherwise fall back to wrist
                    # At test time, front camera is typically the primary view
                    if z_goal_front is not None:
                        goal_encoded = z_goal_front
                        if is_main:
                            print(f"Loaded {goal_desc} from {goal_file_front}")
                    elif z_goal_wrist is not None:
                        goal_encoded = z_goal_wrist
                        if is_main:
                            print(f"Loaded {goal_desc} from {goal_file_wrist} (fallback)")
                    else:
                        continue  # Skip this goal if neither view is available
                    
                    z_goals.append(goal_encoded)
                
                # test-place.py: If we loaded 2 goals (NEAR + FINAL), use multi-goal mode
                if is_main:
                    print(f"Total goals loaded: {len(z_goals)}. Multi-goal mode: {len(z_goals) == 2}")
                if len(z_goals) == 2:
                    multi_goal_mode = True
                    # test-place.py: Placement-only fractional goal conditioning
                    # NEAR goal: first 70% of episode (args.near_fraction)
                    # FINAL goal: last 30% of episode (args.place_fraction)
                    eval_clip_seconds = getattr(args, 'eval_clip_seconds', 10.0)
                    eval_episode_frames = int(eval_clip_seconds * args.hz)
                    
                    # Use configurable fractions (default 70/30)
                    near_frames = int(args.near_fraction * eval_episode_frames)
                    place_frames = eval_episode_frames - near_frames
                    goal_timesteps = [near_frames, place_frames]
                    
                    if is_main:
                        print(f"Placement-only multi-goal planning (letter starts grasped)")
                        print(f"Clip duration: {eval_clip_seconds:.1f}s ({eval_episode_frames} frames at {args.hz}fps)")
                        print(f"Goal fractions: [{args.near_fraction:.0%}, {args.place_fraction:.0%}] = {goal_timesteps} timesteps")
                        print(f"Phases: NEAR={near_frames/args.hz:.1f}s, FINAL={place_frames/args.hz:.1f}s")
                else:
                    # Fallback to single goal (try both front and wrist)
                    goal_img_path_front = os.path.join(goal_dir, "front_goal.png")
                    goal_img_path_wrist = os.path.join(goal_dir, "wrist_goal.png")
                    
                    z_goal_front = None
                    z_goal_wrist = None
                    
                    # Load and encode front goal if available
                    if os.path.exists(goal_img_path_front):
                        goal_img_front = imageio.imread(goal_img_path_front)
                        # Ensure RGB (drop alpha if present)
                        if goal_img_front.shape[-1] == 4:
                            goal_img_front = goal_img_front[:, :, :3]
                        
                        # Encode goal image as patch tokens or pooled latent
                        goal_img_tensor = torch.from_numpy(goal_img_front / 255.0).to(
                            device=device, dtype=torch.float32, non_blocking=True
                        ).unsqueeze(0).permute(0, 3, 1, 2)  # (1, C, H, W)
                        
                        # For patch-token MPC: encode as patches
                        if use_patch_mpc and hasattr(get_model_attr(model, 'E_v'), 'encode_patches'):
                            goal_img_batch = goal_img_tensor.unsqueeze(1)  # (1, 1, C, H, W)
                            z_goal_front = get_model_attr(model, 'E_v').encode_patches(goal_img_batch).squeeze(1)
                            # Apply layer_norm to encoder output (Meta's normalize_reps=True)
                            z_goal_front = F.layer_norm(z_goal_front, (z_goal_front.size(-1),))
                        else:
                            # For latent-space MPC or FM: encode as pooled latent
                            z_goal_front = get_model_attr(model, 'E_v')(goal_img_tensor)
                    
                    # Load and encode wrist goal if available
                    if os.path.exists(goal_img_path_wrist):
                        goal_img_wrist = imageio.imread(goal_img_path_wrist)
                        # Ensure RGB (drop alpha if present)
                        if goal_img_wrist.shape[-1] == 4:
                            goal_img_wrist = goal_img_wrist[:, :, :3]
                        
                        # Encode goal image as patch tokens or pooled latent
                        goal_img_tensor = torch.from_numpy(goal_img_wrist / 255.0).to(
                            device=device, dtype=torch.float32, non_blocking=True
                        ).unsqueeze(0).permute(0, 3, 1, 2)  # (1, C, H, W)
                        
                        # For patch-token MPC: encode as patches
                        if use_patch_mpc and hasattr(get_model_attr(model, 'E_v'), 'encode_patches'):
                            goal_img_batch = goal_img_tensor.unsqueeze(1)  # (1, 1, C, H, W)
                            z_goal_wrist = get_model_attr(model, 'E_v').encode_patches(goal_img_batch).squeeze(1)
                            # Apply layer_norm to encoder output (Meta's normalize_reps=True)
                            z_goal_wrist = F.layer_norm(z_goal_wrist, (z_goal_wrist.size(-1),))
                        else:
                            # For latent-space MPC or FM: encode as pooled latent
                            z_goal_wrist = get_model_attr(model, 'E_v')(goal_img_tensor)
                    
                    # Use front goal if available, otherwise fall back to wrist
                    # At test time, front camera is typically the primary view
                    if z_goal_front is not None:
                        z_goals = [z_goal_front]
                        if is_main:
                            print(f"Loaded goal from {goal_img_path_front}")
                            print(f"Goal shape: {z_goal_front.shape}")
                    elif z_goal_wrist is not None:
                        z_goals = [z_goal_wrist]
                        if is_main:
                            print(f"Loaded goal from {goal_img_path_wrist} (fallback)")
                            print(f"Goal shape: {z_goal_wrist.shape}")
                    else:
                        if is_main:
                            print(f"No goal images found in {goal_dir}")
                            if not args.use_mpc:
                                print(f"FM will use language embedding as fallback")
                            else:
                                print(f"MPC will use language embedding as fallback (not recommended)")
                    
                    if z_goals:
                        # Single goal: use configurable clip length (default: 10 seconds at 4fps)
                        eval_clip_seconds = getattr(args, 'eval_clip_seconds', 10.0)
                        goal_timesteps = [int(eval_clip_seconds * args.hz)]
                
                # Adjust episode length based on goal mode
                if multi_goal_mode:
                    # Use the planned duration for multi-goal tasks
                    steps = sum(goal_timesteps)
                    if is_main:
                        print(f"Adjusted episode length to {steps} timesteps ({steps/args.hz:.1f}s) for multi-goal task")

                z_hist = deque(maxlen=K + 1)
                # For action-conditioned variants: track end-effector states and actions
                if get_model_attr(model, 'use_action_conditioning'):
                    ee_state_hist = deque(maxlen=K + 1)
                    ee_delta_hist = deque(maxlen=K + 1)
                
                # Boot with the first observation
                img_f = render_rgb(scene.ports["rgb_front"], root_context)
                img_w = render_rgb(scene.ports["rgb_wrist"], root_context)
                if current_episode_frames is not None:
                    current_episode_frames.append(img_f[:, :, :3].copy())
                q0 = plant.GetPositions(plant_context).copy()
                # Match evaluation state to checkpoint's expected state_dim (e.g., 9-DoF state_q)
                if q0.shape[0] > state_dim:
                    q0_state = q0[:state_dim]
                elif q0.shape[0] < state_dim:
                    q0_state = np.zeros(state_dim, dtype=q0.dtype)
                    q0_state[: q0.shape[0]] = q0
                else:
                    q0_state = q0
                
                # Encode front view only (test time uses front camera, no multi-view fusion)
                # During training, each view is treated independently; at test time we use front
                # ImageNet normalization is handled internally by VJEPA2HubEncoder
                img_f_tensor = torch.from_numpy(img_f / 255.0).to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(0).permute(0, 3, 1, 2)
                z_v0 = get_model_attr(model, 'E_v')(img_f_tensor)
                z_s0 = get_model_attr(model, 'E_s')(torch.from_numpy(q0_state).float().to(device).unsqueeze(0))
                # Fuse vision and state using single-view fusion (2*d -> d)
                z_t = get_model_attr(model, 'Fusion_single')(torch.cat([z_v0, z_s0], dim=-1))
                for _ in range(K + 1):
                    z_hist.append(z_t)
                
                # Initialize end-effector tracking for action-conditioned variants
                if get_model_attr(model, 'use_action_conditioning'):
                    from pydrake.all import JacobianWrtVariable, RotationMatrix, RigidTransform
                    if plant.HasBodyNamed("panda_hand"):
                        eeF = plant.GetBodyByName("panda_hand").body_frame()
                    elif plant.HasBodyNamed("body"):
                        eeF = plant.GetBodyByName("body").body_frame()
                    else:
                        eeF = plant.GetBodyByName("panda_link8").body_frame()
                    X_WE_0 = plant.CalcRelativeTransform(plant_context, plant.world_frame(), eeF)
                    ee_pos_0 = X_WE_0.translation()
                    ee_rot_0 = X_WE_0.rotation().ToRollPitchYaw().vector()
                    ee_gripper_0 = q0[7] if len(q0) >= 9 else 0.0
                    
                    # Use World Frame directly - training data uses plant.world_frame()
                    # No coordinate transformation needed (verified in episode_generation.py)
                    ee_state_0 = np.concatenate([ee_pos_0, ee_rot_0, [ee_gripper_0]])
                    ee_state_0_t = torch.from_numpy(ee_state_0).float().to(device).unsqueeze(0)  # (1, 7)
                    
                    for _ in range(K + 1):
                        ee_state_hist.append(ee_state_0_t)
                        ee_delta_hist.append(torch.zeros(1, 7, device=device))  # Zero action initially

                # Closed-loop control with simple safety fallback
                from pydrake.all import JacobianWrtVariable, RotationMatrix, RigidTransform
                if plant.HasBodyNamed("panda_hand"):
                    eeF = plant.GetBodyByName("panda_hand").body_frame()
                elif plant.HasBodyNamed("body"):
                    eeF = plant.GetBodyByName("body").body_frame()
                else:
                    eeF = plant.GetBodyByName("panda_link8").body_frame()
                hazard_thresh = -0.005
                inspector = scene.scene_graph.model_inspector()
                def _is_manip_geom(geom_id):
                    name = inspector.GetName(geom_id)
                    return ("panda" in name) or ("wsg" in name) or ("hand" in name) or ("gripper" in name)
                fallback_goal = None
                panda_port = None
                hand_port = None  # Franka hand or WSG gripper
                use_station_control = (scene_type == "franka_reference")
                if use_station_control:
                    diagram = scene.diagram
                    try:
                        num_in_ports = diagram.num_input_ports()
                    except AttributeError:
                        num_in_ports = diagram.NumInputPorts()
                    for i in range(num_in_ports):
                        port = diagram.get_input_port(i)
                        port_name = port.get_name()
                        if "panda.position" in port_name:
                            panda_port = port
                        elif "hand.position" in port_name:
                            hand_port = port
                        elif "wsg.position" in port_name:
                            hand_port = port  # Backward compatibility with WSG
                    if panda_port is None and hand_port is None:
                        use_station_control = False
                
                # Track trajectory metrics
                min_distances = []
                actions_taken = []
                fallback_activations = 0
                
                t_rollout_start = time.time()
                # Prepare final goal for distance checking
                final_goal = subgoals[-1]
                p_WG_final = np.asarray(final_goal.pose_world[:3, 3], dtype=np.float64)
                min_dist_to_goal = float('inf')
                
                letter_body = None
                letter_initial_pose = None
                letter_welded = False  # Letter moves with gripper (not welded to table)
                letter_contacted = True  # Already contacted (grasped)
                min_dist_to_letter = 0.0  # Letter is in gripper
                
                # Determine letter name from scene config
                letter_name = scene_config.get("letter") if scene_type == "franka_reference" else None
                
                # Get letter body and set up grasped initial state
                # ALWAYS set letter_pos_tensor to final goal for position guidance
                letter_pos_tensor = torch.tensor(p_WG_final, dtype=torch.float32, device=device).unsqueeze(0)
                
                if letter_name is not None:
                    try:
                        from pydrake.all import InverseKinematics, Solve
                        letter_body_name = f"{letter_name}_body_link"
                        letter_body = plant.GetBodyByName(letter_body_name)
                        plant_context_init = plant.GetMyMutableContextFromRoot(root_context)
                        
                        # Get letter's initial position on table
                        letter_table_pose = plant.EvalBodyPoseInWorld(plant_context_init, letter_body)
                        letter_table_pos = letter_table_pose.translation()
                        
                        # Compute gripper pose for holding letter at start_height
                        # Gripper pointing down, holding letter
                        R_grasp = RotationMatrix.MakeXRotation(-np.pi/2)
                        gripper_start_pos = np.array([letter_table_pos[0], letter_table_pos[1], args.start_height])
                        X_WG_start = RigidTransform(R_grasp, gripper_start_pos)
                        
                        # Solve IK for the starting configuration
                        panda_model = plant.GetModelInstanceByName("panda")
                        q_current = plant.GetPositions(plant_context_init, panda_model)[:7]
                        
                        ik = InverseKinematics(plant, plant_context_init)
                        q_vars = ik.q()[:7]
                        prog = ik.prog()
                        ik.AddOrientationConstraint(eeF, RotationMatrix(), plant.world_frame(), X_WG_start.rotation(), 0.1)
                        p_target = X_WG_start.translation()
                        ik.AddPositionConstraint(eeF, np.zeros(3), plant.world_frame(), p_target - 0.03, p_target + 0.03)
                        prog.AddQuadraticErrorCost(np.eye(7), q_current, q_vars)
                        prog.SetInitialGuess(q_vars, q_current)
                        ik_result = Solve(prog)
                        
                        if ik_result.is_success():
                            q_start = ik_result.GetSolution(q_vars)
                            # Set arm to IK solution using full plant positions
                            # Plant has 9 DOF for robot (7 arm + 2 gripper) + letter DOF
                            q_full = plant.GetPositions(plant_context_init).copy()
                            q_full[:7] = q_start  # Arm joints
                            q_full[7:9] = [0.02, 0.02]  # Close gripper (grasped)
                            plant.SetPositions(plant_context_init, q_full)
                            
                            # Move letter to gripper position (attached)
                            # Letter is a free body with 7 DOF (quaternion + position)
                            letter_held_pos = gripper_start_pos - np.array([0, 0, 0.05])  # Letter center below gripper
                            try:
                                # Set letter position via free body pose
                                from pydrake.math import RigidTransform as RT
                                letter_pose = RT(letter_held_pos)
                                plant.SetFreeBodyPose(plant_context_init, letter_body, letter_pose)
                            except Exception as e2:
                                if is_main:
                                    print(f"Could not set letter pose: {e2}")
                            
                            if is_main:
                                print(f"Letter '{letter_name}' ALREADY GRASPED at start")
                                print(f"Gripper starts at z={args.start_height:.2f}m (holding letter)")
                                print(f"Using NEAR goal for {args.near_fraction*100:.0f}% → FINAL goal for {args.place_fraction*100:.0f}%")
                        else:
                            if is_main:
                                print(f"IK failed for start pose, using default")
                        
                    except Exception as e:
                        if is_main:
                            print(f"Could not set up grasped state: {e}")
                            import traceback
                            traceback.print_exc()
                        letter_body = None
                
                # Print initial EE position and goal position
                if is_main:
                    plant_context_debug = plant.GetMyMutableContextFromRoot(root_context)
                    X_WE_init = plant.CalcRelativeTransform(plant_context_debug, plant.world_frame(), eeF)
                    init_ee = X_WE_init.translation()
                    init_dist = np.linalg.norm(init_ee - p_WG_final)
                    print(f"Initial EE pos: [{init_ee[0]:.4f}, {init_ee[1]:.4f}, {init_ee[2]:.4f}]")
                    print(f"Goal position:  [{p_WG_final[0]:.4f}, {p_WG_final[1]:.4f}, {p_WG_final[2]:.4f}]")
                    print(f"Initial distance to goal: {init_dist*100:.1f}cm")
                    if args.position_guidance > 0:
                        print(f"Position guidance ENABLED: weight={args.position_guidance}, phases={args.position_guidance_phases}")
                        print(f"Target for position guidance: [{p_WG_final[0]:.4f}, {p_WG_final[1]:.4f}, {p_WG_final[2]:.4f}] (shelf)")
                    else:
                        print(f"Position guidance DISABLED (pure latent MPC)")
                
                # Use mixed precision for faster inference
                act = None
                import time as _time
                mpc_times = []  # Track MPC timing for speed analysis
                
                if is_main:
                    print(f"\n{'='*70}")
                    print(f"Starting rollout: {steps} timesteps @ {args.hz}Hz = {steps/args.hz:.1f}s")
                    print(f"MPC config: horizon={args.mpc_horizon}, samples={args.mpc_samples}, iters={args.mpc_iterations}")
                    print(f"{'='*70}")
                
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    for t in range(steps):
                        # Predictor warmup (JEPA/OT-only architectures)
                        # For VJEPA2-AC with hub predictor, skip this step since the hub predictor
                        # operates on patch tokens (not pooled latents). The predictor will be called
                        # properly inside MPC with patch tokens if --use-mpc is enabled.
                        if arch in ("jepa", "ot-jepa", "ot-vjepa", "vjepa2ac-continued", "vjepa2ac-unfreeze", "vjepa2ac-ot"):
                            # Skip predictor call when using hub predictor (operates on patch tokens, not latents)
                            # This is critical because the warmup block below uses ee_delta_hist (arrival actions),
                            # but VJEPA2-AC predictor expects departure actions. Since this block is skipped for
                            # VJEPA2-AC (skip_predictor_warmup=True), no mismatch occurs.
                            # For VJEPA2-AC, model.Pred is None - the predictor is in E_v.predictor
                            skip_predictor_warmup = (
                                get_model_attr(model, 'use_action_conditioning') 
                                and hasattr(get_model_attr(model, 'E_v'), 'predictor') 
                                and get_model_attr(model, 'E_v').predictor is not None
                            )
                            
                            if not skip_predictor_warmup:
                                z_hist_tensor = torch.stack(list(z_hist), dim=1)  # (1, K+1, d)
                                if get_model_attr(model, 'use_action_conditioning') and hasattr(get_model_attr(model, 'Pred'), 'forward'):
                                    # Custom ActionConditionedPredictor (operates on pooled latents)
                                    ee_state_tensor = torch.stack(list(ee_state_hist), dim=1)  # (1, K+1, 7)
                                    ee_delta_tensor = torch.stack(list(ee_delta_hist), dim=1)  # (1, K+1, 7)
                                    _ = get_model_attr(model, 'Pred')(
                                        actions=ee_delta_tensor,
                                        states=ee_state_tensor,
                                        latents=z_hist_tensor,
                                        return_all_timesteps=False,
                                    )
                                else:
                                    # Non action-conditioned predictor
                                    _ = get_model_attr(model, 'Pred')(z_hist_tensor, horizon=H)

                        # Policy action generation (placeholder - will be set after observations)
                        # MPC/FM action generation happens after encoding observations


                        # Safety: check min distance and if below threshold, blend with Jacobian step to 'pre_place'
                        sg_context = scene.scene_graph.GetMyContextFromRoot(root_context)
                        query = scene.scene_graph.get_query_output_port().Eval(sg_context)
                        signed = query.ComputeSignedDistancePairwiseClosestPoints()
                        if len(signed) == 0:
                            min_dist = 0.2
                        else:
                            dists = [p.distance for p in signed if _is_manip_geom(p.id_A) or _is_manip_geom(p.id_B)]
                            if dists:
                                min_dist = float(min(dists))
                            else:
                                min_dist = 0.2
                        min_distances.append(min_dist)
                        blend = 0.0
                        v_plan = None
                        if min_dist < hazard_thresh:
                            fallback_activations += 1
                            if fallback_goal is None:
                                # pick a mid-way safe subgoal
                                for sg in subgoals:
                                    if sg.name == "pre_place":
                                        fallback_goal = sg
                                        break
                                if fallback_goal is None:
                                    fallback_goal = subgoals[-1]
                            plant_context = plant.GetMyMutableContextFromRoot(root_context)
                            R_WG = RotationMatrix(np.asarray(fallback_goal.pose_world[:3, :3], dtype=np.float64))
                            p_WG = np.asarray(fallback_goal.pose_world[:3, 3], dtype=np.float64)
                            T_WG = RigidTransform(R_WG, p_WG.tolist())
                            X_WE = plant.CalcRelativeTransform(plant_context, plant.world_frame(), eeF)
                            p_err = T_WG.translation() - X_WE.translation()
                            J = plant.CalcJacobianSpatialVelocity(
                                plant_context,
                                JacobianWrtVariable.kV,
                                eeF,
                                [0, 0, 0],
                                plant.world_frame(),
                                plant.world_frame(),
                            )
                            nv = plant.num_velocities()
                            Jv = J[:3, :nv]
                            qdot, *_ = np.linalg.lstsq(Jv, 0.5 * p_err, rcond=1e-3)
                            v_plan = np.clip(qdot, -0.3, 0.3)
                            blend = 0.5

                        # Apply blended velocity
                        nv = plant.num_velocities()
                        v = np.zeros(nv)
                        # On first iteration, act may be None (generated after observations)
                        if act is not None:
                            # Convert 7D end-effector action to joint velocities using Jacobian
                            # Following Meta's VJEPA2-AC: actions are end-effector DELTAS (not velocities)
                            # act = [Δpos_x, Δpos_y, Δpos_z, Δrot_x, Δrot_y, Δrot_z, Δgripper] (7D)
                            # The low-level controller converts these to joint commands using operational space control
                            # In simulation, we approximate OSC with Jacobian-based velocity control
                            if act.shape[0] == 7:
                                # Get current plant context for Jacobian calculation
                                plant_context_jacobian = plant.GetMyMutableContextFromRoot(root_context)
                                
                                # Compute spatial Jacobian for end-effector
                                # J maps joint velocities to end-effector spatial velocities
                                J = plant.CalcJacobianSpatialVelocity(
                                    plant_context_jacobian,
                                    JacobianWrtVariable.kV,
                                    eeF,
                                    [0, 0, 0],
                                    plant.world_frame(),
                                    plant.world_frame(),
                                )
                                # J is (6, nv): [angular_vel(3), linear_vel(3)] x nv
                                # Reorder to [linear_vel(3), angular_vel(3)] to match action format [Δpos, Δrot]
                                J_reordered = np.vstack([J[3:6, :], J[0:3, :]])  # (6, nv)
                                J_arm = J_reordered[:, :7]  # Only arm joints (6x7)
                                
                                # Meta's VJEPA2-AC: actions are position DELTAS, not velocities
                                # Convert to desired velocity: v_desired = Δpos / dt
                                # This gives the velocity needed to achieve the position delta in one timestep
                                #
                                # Meta's MPC only optimizes position (xyz) + gripper.
                                # Rotation deltas are set to ZERO in their implementation (see mpc_utils.py).
                                # Using rotation deltas from MPC causes "wrong shape" issues because they're
                                # essentially random noise. We follow Meta's approach: only use position velocity.
                                #
                                # ACTION OUTPUT: Use World Frame directly
                                # Training data (episode_generation.py) uses plant.world_frame() with NO transformation
                                # Model predicts in World Frame, execute in World Frame
                                ee_pos_delta = np.array([act[0], act[1], act[2]])
                                
                                # Rotation: Zero (Meta's approach)
                                ee_rot_delta = np.zeros(3)
                                ee_delta = np.concatenate([ee_pos_delta, ee_rot_delta])
                                ee_vel_desired = ee_delta / dt  # Convert delta to velocity
                                
                                # Solve for joint velocities using damped least squares (more stable near singularities)
                                # This is the kinematic approximation of operational space control (OSC)
                                # Full OSC would use: τ = J^T @ Λ @ (x_ddot_des - J_dot @ q_dot) + h(q, q_dot)
                                # where Λ = (J @ M^-1 @ J^T)^-1 is the task-space inertia matrix
                                # For simulation, damped pseudoinverse is sufficient since Drake handles dynamics
                                damping = 0.01  # Damping factor for numerical stability
                                JJT = J_arm @ J_arm.T + damping * np.eye(6)
                                J_pinv = J_arm.T @ np.linalg.inv(JJT)  # Damped pseudoinverse
                                qdot_arm = J_pinv @ ee_vel_desired
                                
                                # Safety limits on joint velocities
                                qdot_arm = np.clip(qdot_arm, -0.5, 0.5)
                                
                                # Gripper: convert delta to velocity
                                # Δgripper is in [0, 1] range (closedness), scale to gripper position change
                                gripper_delta = act[6]  # Change in gripper closedness
                                gripper_vel = np.clip(gripper_delta / dt, -0.1, 0.1)  # Safety limits
                                
                                # Assemble full joint velocity vector
                                v[:7] = qdot_arm
                                if nv >= 9:
                                    v[7:9] = [gripper_vel, gripper_vel]  # Both gripper fingers
                                
                                # Apply blending if fallback planner is active
                                if v_plan is not None:
                                    v[:7] = (1.0 - blend) * v[:7] + blend * v_plan[:7]
                                    # Gripper uses policy only (no blend)
                            
                            elif act.shape[0] == 9:
                                # Legacy 9D joint space actions (backward compatibility)
                                if v_plan is None:
                                    # Apply policy action directly
                                    n_act = min(act.shape[0], nv)
                                    v[:n_act] = np.clip(act[:n_act], -0.5, 0.5)
                                    # Gripper actions get smaller velocity limits for safety
                                    if nv >= 9 and act.shape[0] >= 9:
                                        v[7:9] = np.clip(act[7:9], -0.1, 0.1)
                                else:
                                    # Blend policy action with planner baseline
                                    v_fm = np.zeros(nv)
                                    n_act = min(act.shape[0], nv)
                                    v_fm[:n_act] = np.clip(act[:n_act], -0.5, 0.5)
                                    if nv >= 9 and act.shape[0] >= 9:
                                        v_fm[7:9] = np.clip(act[7:9], -0.1, 0.1)
                                    # Blend only affects arm (first 7 DOF), gripper uses policy directly
                                    v[:7] = (1.0 - blend) * v_fm[:7] + blend * v_plan[:7]
                                    if nv >= 9:
                                        v[7:9] = v_fm[7:9]  # Gripper from policy only
                            else:
                                # Unsupported action dimension
                                if is_main:
                                    print(f"Unsupported action dimension {act.shape[0]}, skipping action")
                        
                        actions_taken.append(v.copy())
                        plant_context = plant.GetMyMutableContextFromRoot(root_context)
                        if use_station_control:
                            q_current = plant.GetPositions(plant_context)
                            qdot_conf = plant.MapVelocityToQDot(plant_context, v)
                            q = q_current + qdot_conf * dt
                            if q.shape[0] >= 9:
                                width = 0.5 * (q[7] + q[8])
                                width = np.clip(width, 0.0, 0.107)
                                q[7] = width
                                q[8] = width
                            if panda_port is not None:
                                panda_port.FixValue(root_context, q[:7])
                            if hand_port is not None and q.shape[0] >= 9:
                                # WSG gripper uses width command [0, 0.107]
                                # Convert from finger positions to width
                                width = float(np.clip(q[7] + q[8], 0.0, 0.107))
                                hand_port.FixValue(root_context, np.array([width], dtype=float))
                            sim.AdvanceTo(sim.get_context().get_time() + dt)
                            plant_context = plant.GetMyMutableContextFromRoot(root_context)
                            q = plant.GetPositions(plant_context).copy()
                        else:
                            plant.SetVelocities(plant_context, v)
                            qdot_conf = plant.MapVelocityToQDot(plant_context, v)
                            q = plant.GetPositions(plant_context) + qdot_conf * dt
                            if q.shape[0] >= 9:
                                width = 0.5 * (q[7] + q[8])
                                width = np.clip(width, 0.0, 0.107)
                                q[7] = width
                                q[8] = width
                            plant.SetPositions(plant_context, q)
                            if scene_type != "franka_reference":
                                sim.AdvanceTo(sim.get_context().get_time() + dt)
                        if scene_type == "franka_reference" and meshcat is not None:
                            scene.diagram.ForcedPublish(root_context)

                        # Track min distance to goal for success metric (EE to goal position)
                        plant_context_eval = plant.GetMyMutableContextFromRoot(root_context)
                        X_WE_eval = plant.CalcRelativeTransform(plant_context_eval, plant.world_frame(), eeF)
                        dist_curr = np.linalg.norm(p_WG_final - X_WE_eval.translation())
                        if dist_curr < min_dist_to_goal:
                            min_dist_to_goal = dist_curr
                        
                        if letter_body is not None:
                            # Move letter to follow gripper position
                            gripper_pos = X_WE_eval.translation()
                            try:
                                # Position letter slightly below gripper (held in gripper)
                                letter_held_pos = gripper_pos - np.array([0, 0, 0.05])
                                letter_model = plant.GetModelInstanceByName(letter_name)
                                letter_q = plant.GetPositions(plant_context_eval, letter_model)
                                if len(letter_q) >= 3:
                                    letter_q[:3] = letter_held_pos
                                plant.SetPositions(plant_context_eval, letter_model, letter_q)
                            except Exception:
                                pass  # Ignore if we can't update letter position

                        # Observe next frame and update z_hist / recording
                        img_f = render_rgb(scene.ports["rgb_front"], root_context)
                        img_w = render_rgb(scene.ports["rgb_wrist"], root_context)
                        if current_episode_frames is not None:
                            current_episode_frames.append(img_f[:, :, :3].copy())
                        img_f_t = torch.from_numpy(img_f).float().to(device) / 255.0
                        img_w_t = torch.from_numpy(img_w).float().to(device) / 255.0
                        img_f_t = img_f_t.permute(2, 0, 1).unsqueeze(0)
                        img_w_t = img_w_t.permute(2, 0, 1).unsqueeze(0)
                        # ImageNet normalization is handled internally by VJEPA2HubEncoder
                        # Encode front camera only (test time uses front view, no multi-view fusion)
                        z_v = get_model_attr(model, 'E_v')(img_f_t)
                        # Match runtime state to checkpoint's expected state_dim
                        if q.shape[0] > state_dim:
                            q_state = q[:state_dim]
                        elif q.shape[0] < state_dim:
                            q_state = np.zeros(state_dim, dtype=q.dtype)
                            q_state[: q.shape[0]] = q
                        else:
                            q_state = q
                        z_s = get_model_attr(model, 'E_s')(torch.from_numpy(q_state).float().to(device).unsqueeze(0))
                        # Fuse vision and state using single-view fusion (2*d -> d)
                        z_t = get_model_attr(model, 'Fusion_single')(torch.cat([z_v, z_s], dim=-1))
                        z_hist.append(z_t)
                        
                        # Policy action generation (now that we have current observations)
                        # Reset act so we re-plan at every timestep using latest observations (receding-horizon)
                        act = None
                        if act is None:  # Generate action if not already set
                            if args.use_mpc:
                                # MPC planning: minimize L1 distance to GOAL IMAGE using CEM (Meta's VJEPA2-AC approach)
                                # Get current end-effector state
                                state_current = ee_state_hist[-1] if len(ee_state_hist) > 0 and get_model_attr(model, 'use_action_conditioning') else None
                                
                                # test-place.py: Multi-goal switching (NEAR → FINAL)
                                # Determine which goal to use based on timestep (70/30 split)
                                current_goal_idx = 0  # Default to NEAR goal
                                if multi_goal_mode and len(z_goals) > 0:
                                    cumulative = 0
                                    for goal_idx, duration in enumerate(goal_timesteps):
                                        if t < cumulative + duration:
                                            current_goal_idx = goal_idx
                                            break
                                        cumulative += duration
                                    else:
                                        # After all scheduled timesteps, use final goal (FINAL)
                                        current_goal_idx = len(z_goals) - 1
                                    
                                    z_goal_current = z_goals[current_goal_idx]
                                    
                                    # Reset warm start on goal switch (NEAR → FINAL transition)
                                    goal_switch_time = goal_timesteps[0]  # Switch at end of NEAR phase
                                    if t == goal_switch_time:
                                        prev_mpc_mean = None  # Reset warm start on goal switch
                                        if is_main:
                                            print(f"t={t}: Goal switch NEAR → FINAL")
                                elif len(z_goals) > 0:
                                    # Single goal mode
                                    z_goal_current = z_goals[0]
                                else:
                                    # Fallback to language (not recommended)
                                    z_goal_current = None
                                
                                # Check if using patch-token-based MPC (VJEPA2-AC hub predictor)
                                # Access the underlying model to check the attribute
                                model_unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
                                use_patch_mpc = getattr(model_unwrapped, 'use_patch_token_mpc', False)
                                
                                if use_patch_mpc and hasattr(get_model_attr(model, 'E_v'), 'encode_patches'):
                                    # Patch-token-based MPC (Meta's VJEPA2-AC approach)
                                    # Encode current frame as patch tokens
                                    img_current_batch = img_f_t.unsqueeze(1)  # (1, 1, C, H, W) - add time dimension
                                    z_current_patches = get_model_attr(model, 'E_v').encode_patches(img_current_batch)  # (1, 1, N_p, D_enc)
                                    z_current_patches = z_current_patches.squeeze(1)  # (1, N_p, D_enc)
                                    # layer_norm is applied inside MPC planner to z_current_expanded
                                    # Do NOT apply it here to avoid double-normalization
                                    
                                    # Use selected goal
                                    if z_goal_current is not None:
                                        z_goal = z_goal_current
                                    else:
                                        # Fallback to language (not recommended - prints warning above)
                                        tokens_per_frame = z_current_patches.shape[1]
                                        z_goal = z_l.unsqueeze(1).expand(-1, tokens_per_frame, -1)  # (1, N_p, d)
                                    
                                    tokens_per_frame = z_current_patches.shape[1]
                                    
                                    # Compute latent distance for progress display (only first step)
                                    # Only valid when z_goal has same shape as patch tokens
                                    # (e.g., when using image goals). When falling back to language
                                    # embeddings, dimensionality may differ; in that case we skip.
                                    latent_dist = None
                                    if t == 0:
                                        z_curr_normed = F.layer_norm(z_current_patches, (z_current_patches.size(-1),))
                                        z_curr_flat = z_curr_normed.flatten(1)
                                        z_goal_flat = z_goal.flatten(1)
                                        if z_curr_flat.shape == z_goal_flat.shape:
                                            latent_dist = torch.mean(torch.abs(z_curr_flat - z_goal_flat)).item()
                                            if is_main:
                                                print(f"Initial latent distance to goal: {latent_dist:.4f}")
                                        elif is_main:
                                            print(f"Skipping latent distance debug (shape mismatch): "
                                                    f"z_curr={z_curr_flat.shape}, z_goal={z_goal_flat.shape}")
                                    
                                    # For VJEPA2-AC, the predictor is stored in E_v.predictor
                                    # (loaded from Meta's hub), NOT in model.Pred (which is None for AC variants)
                                    ac_predictor = get_model_attr(model, 'E_v').predictor
                                    
                                    # Determine position guidance weight based on current phase
                                    # Position guidance helps when latent energy landscape is flat
                                    pos_weight = 0.0
                                    target_pos_for_mpc = None
                                    if letter_pos_tensor is not None and args.position_guidance > 0:
                                        if args.position_guidance_phases == "all":
                                            pos_weight = args.position_guidance
                                            target_pos_for_mpc = letter_pos_tensor
                                        # "none" or other phases: pos_weight stays 0.0
                                    
                                    _mpc_start = _time.time()
                                    act_tensor, plan_mean = mpc_planner(
                                        z_current=z_current_patches,
                                        z_goal=z_goal,
                                        predictor=ac_predictor,
                                        state_current=state_current,
                                        use_patch_tokens=True,
                                        tokens_per_frame=tokens_per_frame,
                                        prev_action_mean=prev_mpc_mean,
                                        encoder=get_model_attr(model, 'E_v'),
                                        target_pos=target_pos_for_mpc,
                                        position_weight=pos_weight,
                                    )
                                    mpc_times.append(_time.time() - _mpc_start)
                                else:
                                    # Standard latent-space MPC (non-VJEPA2-AC architectures)
                                    z_current = z_hist[-1]
                                    # Use selected goal
                                    z_goal = z_goal_current if z_goal_current is not None else z_l
                                    
                                    # Get predictor - for VJEPA2-AC fallback, use E_v.predictor
                                    predictor_for_mpc = get_model_attr(model, 'Pred')
                                    if predictor_for_mpc is None and hasattr(get_model_attr(model, 'E_v'), 'predictor'):
                                        predictor_for_mpc = get_model_attr(model, 'E_v').predictor
                                    
                                    _mpc_start = _time.time()
                                    act_tensor, plan_mean = mpc_planner(
                                        z_current=z_current,
                                        z_goal=z_goal,
                                        predictor=predictor_for_mpc,
                                        state_current=state_current,
                                        use_patch_tokens=False,
                                        tokens_per_frame=None,
                                        prev_action_mean=prev_mpc_mean,
                                    )
                                    mpc_times.append(_time.time() - _mpc_start)
                                
                                # Update warm start state for next step (Receding Horizon)
                                # Shift plan left by 1: [t+1, t+2, ...] -> [t, t+1, ...]
                                # plan_mean shape: [B, T, act_dim]
                                # New last step initialized to zeros (or could duplicate last action)
                                if plan_mean.shape[1] > 1:
                                    next_mean = plan_mean[:, 1:, :].clone()
                                    # Pad with zeros for the new last step
                                    padding = torch.zeros_like(plan_mean[:, -1:, :])
                                    prev_mpc_mean = torch.cat([next_mean, padding], dim=1)
                                else:
                                    prev_mpc_mean = None # Horizon 1, no warm start useful
                                
                                act = act_tensor[0].detach().cpu().numpy()
                                
                                # test-place.py: Progress output showing distance to final goal (shelf)
                                if is_main:
                                    ee_pos = state_current[0, :3].cpu().numpy()
                                    goal_names = ["near", "final"]
                                    phase = goal_names[current_goal_idx] if current_goal_idx < len(goal_names) else "?"
                                    
                                    # Compute distance to final goal (shelf position)
                                    goal_dist_str = ""
                                    direction_str = ""
                                    goal_pos = p_WG_final  # Final goal position (shelf)
                                    goal_dist = np.linalg.norm(ee_pos - goal_pos)
                                    goal_dist_str = f"  goal={goal_dist*100:5.1f}cm"
                                    # Add visual indicator
                                    if goal_dist < 0.05:
                                        goal_dist_str += " ★"
                                    elif goal_dist < 0.10:
                                        goal_dist_str += " ●"
                                    elif goal_dist < 0.20:
                                        goal_dist_str += " ○"
                                    
                                    # Show direction to target vs action direction (first 5 steps only)
                                    if t < 5:
                                        to_target = goal_pos - ee_pos
                                        to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-8)
                                        act_norm = act[:3] / (np.linalg.norm(act[:3]) + 1e-8)
                                        alignment = np.dot(to_target_norm, act_norm)
                                        direction_str = f" align={alignment:+.2f}"
                                    
                                    pos_guide = f" [POS:{pos_weight:.1f}]" if pos_weight > 0 else ""
                                    action_mag = np.linalg.norm(act[:3])
                                    
                                    print(f"[t={t:2d}/{steps}] {phase:5s}{pos_guide} | "
                                            f"EE=[{ee_pos[0]:+.3f},{ee_pos[1]:+.3f},{ee_pos[2]:+.3f}] | "
                                            f"act={action_mag*100:.1f}cm{goal_dist_str}{direction_str}")
                                
                            else:
                                # Flow Matching policy (uses goal image latent)
                                # Multi-goal support: use appropriate goal based on timestep
                                if multi_goal_mode and len(z_goals) > 0:
                                    cumulative = 0
                                    current_goal_idx = 0
                                    for goal_idx, duration in enumerate(goal_timesteps):
                                        if t < cumulative + duration:
                                            current_goal_idx = goal_idx
                                            break
                                        cumulative += duration
                                    else:
                                        current_goal_idx = len(z_goals) - 1
                                    z_goal_fm = z_goals[current_goal_idx]
                                elif len(z_goals) > 0:
                                    z_goal_fm = z_goals[0]
                                else:
                                    # Fallback to language
                                    z_goal_fm = z_l
                                    if is_main and t == 0:
                                        print("FM using language fallback (goal image not found)")
                                
                                # For patch-based goal, pool to latent dimension
                                if z_goal_fm.dim() == 3:  # (1, N_p, D_enc) - patch tokens
                                    z_goal_fm = z_goal_fm.mean(dim=1)  # Pool to (1, D_enc)
                                    # Project to match latent dimension if needed
                                    if z_goal_fm.shape[-1] != z_t.shape[-1]:
                                        # Simple average pooling to match dimensions
                                        ratio = z_goal_fm.shape[-1] // z_t.shape[-1]
                                        z_goal_fm = z_goal_fm.view(z_goal_fm.shape[0], -1, ratio).mean(dim=-1)
                                
                                act_tensor = get_model_attr(model, 'FM').sample(z_t, z_goal_fm, 5)
                                act = act_tensor[0].detach().cpu().numpy()
                                
                                # Apply box constraint (matching MPC's constraint for consistency)
                                # Meta VJEPA2-AC uses maxnorm=0.05m for XYZ positions
                                act[:3] = np.clip(act[:3], -0.05, 0.05)  # XYZ position deltas
                                act[3:6] = 0.0  # Zero rotation (Meta's simplification)
                                act[6] = np.clip(act[6], -0.75, 0.75)  # Gripper delta
                        
                        # Update end-effector tracking for action-conditioned variants
                        if get_model_attr(model, 'use_action_conditioning'):
                            plant_context_current = plant.GetMyMutableContextFromRoot(root_context)
                            X_WE_current = plant.CalcRelativeTransform(plant_context_current, plant.world_frame(), eeF)
                            ee_pos = X_WE_current.translation()
                            ee_rot = X_WE_current.rotation().ToRollPitchYaw().vector()
                            ee_gripper = q[7] if len(q) >= 9 else 0.0
                            
                            # Use World Frame directly - training data uses plant.world_frame()
                            ee_state = np.concatenate([ee_pos, ee_rot, [ee_gripper]])
                            ee_state_t = torch.from_numpy(ee_state).float().to(device).unsqueeze(0)  # (1, 7)
                            
                            # Compute delta from previous state (both in World Frame)
                            ee_delta = ee_state_t - ee_state_hist[-1]
                            
                            ee_state_hist.append(ee_state_t)
                            ee_delta_hist.append(ee_delta)
                
                if is_main:
                    print(f"Policy rollout took {time.time() - t_rollout_start:.2f}s")
                    if mpc_times:
                        avg_mpc = sum(mpc_times) / len(mpc_times)
                        print(f"MPC timing: avg={avg_mpc*1000:.0f}ms/step, total={sum(mpc_times):.1f}s ({len(mpc_times)} calls)")

                # Success proxy: Did we reach the goal?
                # We define success as getting the EE within 5cm (0.05m) of the final goal pose
                # ideally we check object pose, but we track EE->Goal
                policy_success = min_dist_to_goal < 0.05
                
                # Also compute collision clearance for stats
                sg_context = scene.scene_graph.GetMyContextFromRoot(root_context)
                query = scene.scene_graph.get_query_output_port().Eval(sg_context)
                signed = query.ComputeSignedDistancePairwiseClosestPoints()
                final_min_dist = 0.2 if len(signed) == 0 else float(min([p.distance for p in signed]))
                
                if policy_success:
                    scene_successes_policy += 1
                
                # Compute trajectory statistics
                actions_array = np.array(actions_taken)
                action_magnitudes = np.linalg.norm(actions_array, axis=1)
                
                # Compute jerk (third derivative of position, approximated from velocities)
                if len(actions_array) >= 3:
                    vel_diff = np.diff(actions_array, axis=0)  # acceleration
                    acc_diff = np.diff(vel_diff, axis=0)  # jerk
                    mean_jerk = float(np.mean(np.linalg.norm(acc_diff, axis=1)))
                else:
                    mean_jerk = 0.0
                
                ep_duration = time.time() - ep_start_time
                
                # Update episode data with policy results
                ep_data.update({
                    "policy_success": policy_success,
                    "policy_final_min_distance": final_min_dist,
                    "policy_min_distance_mean": float(np.mean(min_distances)),
                    "policy_min_distance_min": float(np.min(min_distances)),
                    # test-place.py: Minimum Euclidean distance from EE to GOAL (shelf) over trajectory
                    # This uses min_dist_to_goal, which is updated each timestep from EE pose -> p_WG_final
                    "min_dist_to_goal": float(min_dist_to_goal) if min_dist_to_goal != float('inf') else None,
                    "fallback_activations": fallback_activations,
                    "fallback_rate": fallback_activations / len(min_distances) if len(min_distances) > 0 else 0.0,
                    "trajectory_length": len(actions_taken),
                    "action_magnitude_mean": float(np.mean(action_magnitudes)),
                    "action_magnitude_max": float(np.max(action_magnitudes)),
                    "mean_jerk": mean_jerk,
                    "episode_duration_sec": ep_duration,
                    # Letter contact tracking
                    "letter_contacted": letter_contacted,
                    "min_dist_to_letter": float(min_dist_to_letter) if min_dist_to_letter != float('inf') else None,
                })
                
                scene_episode_results.append(ep_data)
                global_total_episodes += 1
                
                if is_main:
                    planner_str = f"planner={'✓' if planner_success else '✗'}" if planner_success is not None else "planner=skipped"
                    letter_str = ""
                    if letter_body is not None:
                        contact_symbol = '✓' if letter_contacted else '✗'
                        min_dist_cm = min_dist_to_letter * 100 if min_dist_to_letter != float('inf') else float('inf')
                        letter_str = f" letter_contact={contact_symbol} min_dist_letter={min_dist_cm:.2f}cm"
                    print(f"Episode {ep}/{args.episodes}: policy={'✓' if policy_success else '✗'} "
                        f"{planner_str} "
                        f"min_dist={final_min_dist:.4f} fallback={fallback_activations}/{len(min_distances)}{letter_str}")
                    
                    # Use min_dist_to_letter if available (letter scenes), otherwise use min_dist_to_goal
                    episode_metric = min_dist_to_letter if min_dist_to_letter != float('inf') else min_dist_to_goal
                    
                    # Track scene-level best
                    if episode_metric < scene_best_episode_min_dist:
                        scene_best_episode_min_dist = episode_metric
                        scene_best_episode_idx = ep
                        scene_best_episode_frames = [frame.copy() for frame in current_episode_frames] if current_episode_frames else None
                        if letter_body is not None:
                            print(f"New scene best: ep {ep} (min_dist_to_letter={episode_metric*100:.2f}cm)")
                        else:
                            print(f"New scene best: ep {ep} (min_dist_to_goal={episode_metric*100:.2f}cm)")
                    
                    # Track global best (across all scenes)
                    if episode_metric < global_best_episode_min_dist:
                        global_best_episode_min_dist = episode_metric
                        global_best_episode_idx = global_total_episodes - 1  # Already incremented
                        global_best_episode_scene = current_scene_name
                        global_best_episode_frames = [frame.copy() for frame in current_episode_frames] if current_episode_frames else None
                        if letter_body is not None:
                            print(f"New GLOBAL best: {current_scene_name} ep {ep} (min_dist_to_letter={episode_metric*100:.2f}cm)")
                        else:
                            print(f"New GLOBAL best: {current_scene_name} ep {ep} (min_dist_to_goal={episode_metric*100:.2f}cm)")
            
            # Store per-scene results
            per_scene_results[current_scene_name] = {
                "episodes": scene_episode_results,
                "successes_policy": scene_successes_policy,
                "successes_planner": scene_successes_planner,
                "best_episode_idx": scene_best_episode_idx,
                "best_episode_min_dist": float(scene_best_episode_min_dist) if scene_best_episode_min_dist != float('inf') else None,
            }
            
            # Accumulate to global counters
            all_episode_results.extend(scene_episode_results)
            global_successes_policy += scene_successes_policy
            global_successes_planner += scene_successes_planner
            
            if is_main:
                print(f"\nScene {current_scene_name} summary:")
                print(f"Policy success: {scene_successes_policy}/{args.episodes}")
                if scene_best_episode_min_dist != float('inf'):
                    print(f"Best episode: {scene_best_episode_idx} (min_dist={scene_best_episode_min_dist*100:.2f}cm)")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if is_main:
        html_content = ""
        if meshcat is not None and meshcat_can_record:
            try:
                meshcat.StopRecording()
                meshcat.PublishRecording()
                html_content = meshcat.StaticHtml()
            except Exception as exc:
                print(f"Meshcat export failed: {exc}")
        elif meshcat is not None and hasattr(meshcat, "StaticHtml"):
            try:
                html_content = meshcat.StaticHtml()
            except Exception as exc:
                print(f"Meshcat StaticHtml failed: {exc}")

        if html_content:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"saved html to {html_path}")
            # Meshcat records all episodes continuously, so HTML contains all episodes
            # For single-episode HTML, would need per-episode recording (complex with Drake)
        
        # Save MP4 from global best episode (minimum distance to letter/goal across all scenes)
        if global_best_episode_frames:
            try:
                imageio.mimwrite(mp4_path, global_best_episode_frames, fps=args.hz, macro_block_size=None)
                print(f"saved mp4 to {mp4_path} (global best: {global_best_episode_scene} ep {global_best_episode_idx}, min_dist={global_best_episode_min_dist*100:.2f}cm)")
            except Exception as e:
                print(f"failed to write mp4: {e}")
        elif global_best_episode_idx < 0:
            print(f"no episodes completed, skipping mp4 export")
        
        print(f"\nGLOBAL SUMMARY")
        print(f"Scenes evaluated: {len(scenes_to_run)} ({', '.join(scenes_to_run)})")
        print(f"Total episodes: {global_total_episodes}")
        print(f"Planner baseline: {global_successes_planner}/{global_total_episodes}")
        policy_method = "MPC" if args.use_mpc else "FM"
        print(f"Learned policy ({policy_method}): {global_successes_policy}/{global_total_episodes}")
        if global_best_episode_scene:
            print(f"Global best episode: {global_best_episode_scene} (min_dist={global_best_episode_min_dist*100:.2f}cm)")
        
        # Per-scene breakdown
        if len(scenes_to_run) > 1:
            print(f"\nPer-scene breakdown:")
            for scene_name in scenes_to_run:
                scene_data = per_scene_results.get(scene_name, {})
                policy_rate = scene_data.get("successes_policy", 0) / args.episodes if args.episodes > 0 else 0
                print(f"{scene_name}: {scene_data.get('successes_policy', 0)}/{args.episodes} ({policy_rate*100:.1f}%)")

        # Compute contact rate and average minimum distance to letter
        letter_contact_episodes = [ep for ep in all_episode_results if ep.get("letter_contacted") is not None]
        letter_stats = None
        if letter_contact_episodes:
            num_contacted = sum(1 for ep in letter_contact_episodes if ep["letter_contacted"])
            contact_rate = num_contacted / len(letter_contact_episodes)
            
            # Compute average minimum distance (only for episodes where tracking was active)
            valid_distances = [ep["min_dist_to_letter"] for ep in letter_contact_episodes 
                              if ep.get("min_dist_to_letter") is not None]
            if valid_distances:
                avg_min_dist = np.mean(valid_distances)
                std_min_dist = np.std(valid_distances)
                min_min_dist = np.min(valid_distances)
                max_min_dist = np.max(valid_distances)
            else:
                avg_min_dist = std_min_dist = min_min_dist = max_min_dist = float('nan')
            
            print(f"\nLETTER CONTACT STATISTICS")
            print(f"Contact rate: {num_contacted}/{len(letter_contact_episodes)} ({contact_rate*100:.1f}%)")
            print(f"Min distance to letter:")
            print(f"Average: {avg_min_dist*100:.2f}cm (±{std_min_dist*100:.2f}cm)")
            print(f"Range:   [{min_min_dist*100:.2f}cm, {max_min_dist*100:.2f}cm]")
            
            # Prepare letter stats for JSON summary
            letter_stats = {
                "num_episodes_tracked": len(letter_contact_episodes),
                "num_contacted": num_contacted,
                "contact_rate": float(contact_rate),
                "min_distance_to_letter": {
                    "mean": float(avg_min_dist) if not np.isnan(avg_min_dist) else None,
                    "std": float(std_min_dist) if not np.isnan(std_min_dist) else None,
                    "min": float(min_min_dist) if not np.isnan(min_min_dist) else None,
                    "max": float(max_min_dist) if not np.isnan(max_min_dist) else None,
                }
            }
        
        goal_distance_episodes = [ep for ep in all_episode_results if ep.get("min_dist_to_goal") is not None]
        goal_stats = None
        if goal_distance_episodes:
            goal_distances = [ep["min_dist_to_goal"] for ep in goal_distance_episodes]
            goal_avg = np.mean(goal_distances)
            goal_std = np.std(goal_distances)
            goal_min = np.min(goal_distances)
            goal_max = np.max(goal_distances)
            
            print(f"\nGOAL DISTANCE STATISTICS")
            print(f"Min distance to GOAL (EE->shelf):")
            print(f"Average: {goal_avg*100:.2f}cm (±{goal_std*100:.2f}cm)")
            print(f"Range:   [{goal_min*100:.2f}cm, {goal_max*100:.2f}cm]")
            
            goal_stats = {
                "num_episodes_tracked": len(goal_distance_episodes),
                "min_distance_to_goal": {
                    "mean": float(goal_avg),
                    "std": float(goal_std),
                    "min": float(goal_min),
                    "max": float(goal_max),
                },
            }
        
        # Save detailed results
        import json
        
        # Helper to convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        results_json_path = os.path.join(results_dir, f"{arch}_{suffix}.json")
        
        results_summary = {
            "architecture": arch,
            "checkpoint": ckpt_path,
            "randomize_factor": args.randomize,
            "episodes_per_scene": args.episodes,
            "scenes_evaluated": scenes_to_run,
            "total_episodes": global_total_episodes,
            "policy_method": "MPC" if args.use_mpc else "FlowMatching",
            "mpc_config": {
                "enabled": args.use_mpc,
                "horizon": args.mpc_horizon if args.use_mpc else None,
                "samples": args.mpc_samples if args.use_mpc else None,
                "iterations": args.mpc_iterations if args.use_mpc else None,
                "top_k": args.mpc_top_k if args.use_mpc else None,
            },
            "global_planner_success_rate": global_successes_planner / global_total_episodes if global_total_episodes > 0 else 0,
            "global_policy_success_rate": global_successes_policy / global_total_episodes if global_total_episodes > 0 else 0,
            "letter_contact_stats": letter_stats,
            "goal_distance_stats": goal_stats,
            "global_best_episode": {
                "scene": global_best_episode_scene,
                "episode_idx": global_best_episode_idx,
                "min_distance": float(global_best_episode_min_dist) if global_best_episode_min_dist != float('inf') else None,
            },
            "per_scene_results": {
                scene_name: {
                    "successes_policy": data.get("successes_policy", 0),
                    "successes_planner": data.get("successes_planner", 0),
                    "policy_success_rate": data.get("successes_policy", 0) / args.episodes if args.episodes > 0 else 0,
                    "best_episode_idx": data.get("best_episode_idx"),
                    "best_episode_min_dist": data.get("best_episode_min_dist"),
                }
                for scene_name, data in per_scene_results.items()
            },
            "episodes": all_episode_results,
        }
        
        # Convert numpy types to native Python types for JSON serialization
        results_summary = convert_to_native(results_summary)
        
        with open(results_json_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"saved detailed results to {results_json_path}")
        
        # Also save as CSV for easy analysis
        try:
            import csv
            csv_path = os.path.join(results_dir, f"{arch}_{suffix}.csv")
            if all_episode_results:
                # Flatten nested dicts
                csv_rows = []
                for ep in all_episode_results:
                    row = {
                        "scene": ep.get("scene", "unknown"),
                        "episode": ep["episode"],
                        "global_episode_idx": ep.get("global_episode_idx", ep["episode"]),
                        "randomize_factor": ep["randomize_factor"],
                        "planner_success": ep["planner_success"],
                        "policy_success": ep["policy_success"],
                        "policy_final_min_distance": ep["policy_final_min_distance"],
                        "policy_min_distance_mean": ep["policy_min_distance_mean"],
                        "policy_min_distance_min": ep["policy_min_distance_min"],
                        # Minimum EE->goal (shelf) distance in centimeters
                        "min_dist_to_goal_cm": ep.get("min_dist_to_goal") * 100 if ep.get("min_dist_to_goal") is not None else None,
                        "fallback_activations": ep["fallback_activations"],
                        "fallback_rate": ep["fallback_rate"],
                        "trajectory_length": ep["trajectory_length"],
                        "action_magnitude_mean": ep["action_magnitude_mean"],
                        "action_magnitude_max": ep["action_magnitude_max"],
                        "mean_jerk": ep["mean_jerk"],
                        "episode_duration_sec": ep["episode_duration_sec"],
                        "light_intensity": ep["randomization"]["light_intensity"],
                        "camera_jitter": ep["randomization"]["camera_jitter"],
                        "clutter_count": ep["randomization"]["clutter_count"],
                        "friction": ep["randomization"]["friction"],
                        # Letter contact tracking
                        "letter_contacted": ep.get("letter_contacted"),
                        "min_dist_to_letter_cm": ep.get("min_dist_to_letter") * 100 if ep.get("min_dist_to_letter") is not None else None,
                    }
                    csv_rows.append(row)
                
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_rows)
                print(f"saved CSV results to {csv_path}")
        except Exception as e:
            print(f"failed to save CSV: {e}")
    
    # Synchronize all ranks before exiting
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            print(f"All ranks synchronized. Exiting.")

if __name__ == "__main__":
    main()
