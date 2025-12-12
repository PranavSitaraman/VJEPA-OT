from __future__ import annotations
import os, glob, random, json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch

@dataclass
class WindowSpec:
    k: int
    H: int

class EpisodeDataset:
    """Episode dataset aligned with Meta VJEPA2-AC paper.
    
    Key alignment with Meta's implementation:
    - Clips are 5-10 seconds at 4fps (20-40 frames) - enforced via min/max clip length
    - Goal images sampled uniformly up to 20 timesteps forward
    - Fractional goal conditioning: 0.7 grasp, 0.2 transport, 0.1 place
    """
    
    def __init__(
        self,
        root_dir: str,
        window: WindowSpec,
        image_size=(256, 256),
        rank: int = 0,
        world_size: int = 1,
        use_embeddings: bool = False,
        min_clip_frames: int = 20,  # 5 seconds at 4fps
        max_clip_frames: int = 40,  # 10 seconds at 4fps
        # V-JEPA 2 data augmentation parameters
        random_resize_scale: Tuple[float, float] = (0.3, 1.0),  # Meta VJEPA2: RandomResizedCrop scale
        random_resize_aspect_ratio: Tuple[float, float] = (0.75, 1.35),  # Meta VJEPA2: aspect ratio range
        horizontal_flip: bool = False,  # Meta VJEPA2: horizontal flip (disabled by default for robotics)
        augment: bool = True,  # Enable/disable augmentation
    ):
        self.root_dir = root_dir
        self.window = window
        self.image_size = tuple(image_size)
        self.use_embeddings = use_embeddings
        self.min_clip_frames = min_clip_frames
        self.max_clip_frames = max_clip_frames
        # V-JEPA 2 augmentation settings
        self.random_resize_scale = random_resize_scale
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.horizontal_flip = horizontal_flip
        self.augment = augment
        # Simple LRU cache for decoded images (CHW float32). Capacity tuned for memory.
        self._img_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._img_cache_cap = 128
        # LRU cache for per-episode DataFrames to avoid repeated parquet reads.
        self._df_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._df_cache_cap = 16
        # LRU cache for precomputed embeddings (front/wrist) when enabled.
        self._emb_cache: OrderedDict[str, dict] = OrderedDict()
        self._emb_cache_cap = 64
        episode_files = glob.glob(os.path.join(root_dir, "**", "episode.parquet"), recursive=True)
        if episode_files:
            all_episodes = sorted({os.path.dirname(p) for p in episode_files})
        else:
            all_episodes = sorted([p for p in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(p)])
        if world_size > 1 and len(all_episodes) >= world_size:
            shard = all_episodes[rank :: world_size]
            self.episodes = shard if shard else all_episodes
        else:
            self.episodes = all_episodes
        self.state_dim = self._infer_state_dim()
        self.action_dim = self._infer_action_dim()
        if len(self.episodes)==0:
            print(f"No episodes in {root_dir} yet.")

    def __len__(self):
        return len(self.episodes)

    def _infer_state_dim(self) -> int:
        default_dim = 7
        for ep in self.episodes:
            parquet_path = os.path.join(ep, "episode.parquet")
            if not os.path.exists(parquet_path):
                continue
            df = pd.read_parquet(parquet_path)
            
            # Prioritize ee_state (end-effector state) for state dimension
            # This is 7D: [x, y, z, roll, pitch, yaw, gripper]
            # Matches Meta's VJEPA2-AC which uses Cartesian space states
            ee_state_cols = [c for c in df.columns if c.startswith("state_ee_state")]
            if ee_state_cols and len(df) > 0:
                first_col = ee_state_cols[0]
                series = df[first_col]
                if series.dtype == object:
                    first_value = series.iloc[0]
                    if hasattr(first_value, "__len__"):
                        return len(first_value)
                return len(ee_state_cols)
            
            # Fallback to state_q (joint positions) for legacy datasets
            s_cols = [c for c in df.columns if c.startswith("state_q")]
            if not s_cols or len(df)==0:
                continue
            if len(s_cols)==1 and df[s_cols[0]].dtype == object:
                first_value = df[s_cols[0]].iloc[0]
                if hasattr(first_value, "__len__"):
                    return len(first_value)
            return len(s_cols)
        return default_dim

    def _load_df(self, ep: str) -> pd.DataFrame:
        parquet_path = os.path.join(ep, "episode.parquet")
        if ep in self._df_cache:
            df = self._df_cache.pop(ep)
            self._df_cache[ep] = df
            return df
        df = pd.read_parquet(parquet_path)
        self._df_cache[ep] = df
        if len(self._df_cache) > self._df_cache_cap:
            self._df_cache.popitem(last=False)
        return df

    def _infer_action_dim(self) -> int:
        default_dim = 7
        for ep in self.episodes:
            parquet_path = os.path.join(ep, "episode.parquet")
            if not os.path.exists(parquet_path):
                continue
            df = pd.read_parquet(parquet_path)
            # Prioritize ee_delta (end-effector deltas) for action dimension
            # This is 7D: [dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]
            # Matches Meta's VJEPA2-AC which uses Cartesian space actions
            ee_delta_cols = [c for c in df.columns if c.startswith("act_ee_delta")]
            if ee_delta_cols and len(df) > 0:
                first_col = ee_delta_cols[0]
                series = df[first_col]
                if series.dtype == object:
                    first_value = series.iloc[0]
                    if hasattr(first_value, "__len__"):
                        return len(first_value)
                return len(ee_delta_cols)
            # Fallback to qdot (joint velocities) for legacy datasets
            qdot_cols = [c for c in df.columns if c.startswith("act_qdot")]
            if qdot_cols and len(df) > 0:
                first_col = qdot_cols[0]
                series = df[first_col]
                if series.dtype == object:
                    first_value = series.iloc[0]
                    if hasattr(first_value, "__len__"):
                        return len(first_value)
                return len(qdot_cols)
            a_cols = [c for c in df.columns if c.startswith("act_")]
            if not a_cols or len(df)==0:
                continue
            first_col = a_cols[0]
            series = df[first_col]
            if series.dtype == object:
                first_value = series.iloc[0]
                if hasattr(first_value, "__len__"):
                    return len(first_value)
            return len(a_cols)
        return default_dim

    def sample_crop_params(self, img_width: int, img_height: int) -> Tuple[int, int, int, int, bool]:
        """Sample random crop parameters for V-JEPA 2 style RandomResizedCrop.
        
        Returns (x, y, crop_w, crop_h, flip) where:
        - (x, y) is the top-left corner of the crop
        - (crop_w, crop_h) is the size of the crop region
        - flip indicates whether to apply horizontal flip
        
        The same parameters should be applied to all frames in a sequence.
        """
        if not self.augment:
            # No augmentation: use full image
            return 0, 0, img_width, img_height, False
        
        scale_min, scale_max = self.random_resize_scale
        ar_min, ar_max = self.random_resize_aspect_ratio
        
        # Try up to 10 times to find valid crop (following torchvision implementation)
        for _ in range(10):
            # Sample scale and aspect ratio
            scale = random.uniform(scale_min, scale_max)
            aspect_ratio = random.uniform(ar_min, ar_max)
            
            # Compute crop dimensions
            area = img_width * img_height * scale
            crop_w = int(round((area * aspect_ratio) ** 0.5))
            crop_h = int(round((area / aspect_ratio) ** 0.5))
            
            # Check if crop fits within image
            if crop_w <= img_width and crop_h <= img_height:
                x = random.randint(0, img_width - crop_w)
                y = random.randint(0, img_height - crop_h)
                flip = self.horizontal_flip and random.random() < 0.5
                return x, y, crop_w, crop_h, flip
        
        # center crop with minimum scale
        crop_size = min(img_width, img_height)
        x = (img_width - crop_size) // 2
        y = (img_height - crop_size) // 2
        flip = self.horizontal_flip and random.random() < 0.5
        return x, y, crop_size, crop_size, flip
    
    def _load_img_augmented(self, path: str, crop_params: Tuple[int, int, int, int, bool] = None) -> np.ndarray:
        """Load and augment an image with given crop parameters.
        
        Args:
            path: Path to image file
            crop_params: (x, y, crop_w, crop_h, flip) from sample_crop_params()
                        If None, loads without augmentation (uses full image)
        
        Returns:
            Image array (C, H, W) normalized to [0, 1]
        """
        img = Image.open(path).convert("RGB")
        
        if crop_params is not None:
            x, y, crop_w, crop_h, flip = crop_params
            # Apply crop
            img = img.crop((x, y, x + crop_w, y + crop_h))
            # Apply horizontal flip if needed
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Resize to target size
        img = img.resize(self.image_size, Image.BILINEAR)
        
        # Convert to numpy array (C, H, W) normalized
        arr = np.array(img)
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0
        return arr
    
    def _load_img(self, path, crop_params: Tuple[int, int, int, int, bool] = None):
        """Load image with optional augmentation.
        
        For cached (non-augmented) loading, use crop_params=None.
        For augmented loading, pass crop_params from sample_crop_params().
        
        Augmented images are NOT cached since each sequence uses different crops.
        """
        # If augmentation requested, bypass cache and load with crop
        if crop_params is not None:
            return self._load_img_augmented(path, crop_params)
        
        # LRU cache lookup (for non-augmented loading)
        if path in self._img_cache:
            arr = self._img_cache.pop(path)
            self._img_cache[path] = arr
            return arr
        img = Image.open(path).convert("RGB").resize(self.image_size)
        arr = np.array(img)
        arr = np.transpose(arr, (2,0,1)).astype(np.float32) / 255.0
        # Insert into cache
        self._img_cache[path] = arr
        if len(self._img_cache) > self._img_cache_cap:
            self._img_cache.popitem(last=False)
        return arr

    def _load_embeddings(self, emb_path: str) -> Optional[dict]:
        if emb_path in self._emb_cache:
            embs = self._emb_cache.pop(emb_path)
            self._emb_cache[emb_path] = embs
            return embs
        try:
            embs = torch.load(emb_path, map_location="cpu")
        except Exception:
            return None
        if not isinstance(embs, dict) or "front" not in embs or "wrist" not in embs:
            return None
        self._emb_cache[emb_path] = embs
        if len(self._emb_cache) > self._emb_cache_cap:
            self._emb_cache.popitem(last=False)
        return embs

    def sample_batch(self, batch_size=8, sample_goal_images=True, max_goal_offset=20, 
                     use_fractional_goals=True, use_scene_subgoals=True) -> Dict[str, torch.Tensor]:
        """Sample a batch of trajectory segments aligned with Meta VJEPA2-AC.
        
        Args:
            batch_size: Number of episodes to sample
            sample_goal_images: If True, sample goal images from future timesteps (Meta's approach)
            max_goal_offset: Maximum timesteps forward to sample goal from (Meta uses 20)
            use_fractional_goals: If True, sample goals based on fractional position in clip
                                  (0.7 grasp, 0.2 transport, 0.1 place)
            use_scene_subgoals: If True, use per-scene subgoal images (front_goal_grasp.png,
                               front_goal_near.png, front_goal_final.png) instead of random
                               future frames. This aligns with Meta's pick-and-place approach.
        
        Returns:
            Dictionary with images, states, actions, and optionally goal images
            
        Meta VJEPA2-AC Alignment:
            - Clips are 5-10 seconds at 4fps (20-40 frames)
            - Goal images sampled uniformly up to 20 timesteps forward
            - For pick-and-place: fractional conditioning (0.7, 0.2, 0.1) using scene subgoals
        """
        batch = []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            df = self._load_df(ep)
            T = len(df)
            k, H = self.window.k, self.window.H
            
            # Enforce Meta VJEPA2-AC clip length constraints
            # Clips should be 5-10 seconds (20-40 frames at 4fps)
            min_clip = max(k + H + 1, self.min_clip_frames)
            max_clip = min(T, self.max_clip_frames)
            
            if T < min_clip:
                # Episode too short - use entire episode (will be handled by augmentation)
                clip_start = 0
                clip_end = T
            elif max_clip <= min_clip:
                # Episode within acceptable range - use entire episode
                clip_start = 0
                clip_end = T
            else:
                # Sample a clip of appropriate length from the episode
                # This "speeds up" long episodes by taking shorter segments
                clip_length = random.randint(min_clip, max_clip)
                max_start = T - clip_length
                clip_start = random.randint(0, max(0, max_start))
                clip_end = clip_start + clip_length
            
            # Effective clip length for this sample
            clip_T = clip_end - clip_start
            
            # Sample t0 within the clip (relative to clip_start)
            # Ensure: t0 - k >= clip_start (need k frames before t0)
            # Ensure: t0 + H < clip_end (need H frames after t0)
            # So: t0_relative must be in [k, clip_T - H - 1]
            min_t0_rel = k
            max_t0_rel = clip_T - H - 1
            
            # If clip is too short for the window, adjust bounds
            if max_t0_rel < min_t0_rel:
                # Clip is too short - center the window as best as possible
                max_t0_rel = min_t0_rel
                # Also need to ensure we don't exceed clip bounds
                if min_t0_rel + H + 1 > clip_T:
                    # Shift t0 back to fit
                    min_t0_rel = max(0, clip_T - H - 1)
                    max_t0_rel = min_t0_rel
            
            t0_relative = random.randint(min_t0_rel, max_t0_rel)
            t0 = clip_start + t0_relative  # Convert to absolute index
            
            # Safety clamp to ensure indices are valid for the full dataframe
            t0 = max(k, min(t0, T - H - 1))

            imgs_f, imgs_w = [], []
            goal_img_f, goal_img_w = None, None
            emb_f = emb_w = None
            
            # Sample goal image based on fractional position in clip (Meta's approach)
            # For pick-and-place: 0.7 grasp, 0.2 transport, 0.1 place
            # Use per-scene subgoal images if available (front_goal_grasp.png, front_goal_near.png, front_goal_final.png)
            if sample_goal_images:
                # Try to use per-scene subgoal images first (Meta's approach for pick-and-place)
                scene_subgoal_loaded = False
                if use_scene_subgoals and use_fractional_goals:
                    # Determine which phase we're in based on position in clip
                    relative_pos = (t0 - clip_start) / clip_T if clip_T > 0 else 0.0
                    
                    # Select appropriate subgoal image based on phase
                    # 0.0-0.7: grasp phase -> front_goal_grasp.png
                    # 0.7-0.9: transport phase -> front_goal_near.png
                    # 0.9-1.0: place phase -> front_goal_final.png
                    if relative_pos < 0.7:
                        subgoal_suffix = "grasp"
                    elif relative_pos < 0.9:
                        subgoal_suffix = "near"
                    else:
                        subgoal_suffix = "final"
                    
                    subgoal_file_f = f"front_goal_{subgoal_suffix}.png"
                    subgoal_file_w = f"wrist_goal_{subgoal_suffix}.png"
                    
                    # Look for subgoal image in episode directory first
                    subgoal_path_f = os.path.join(ep, subgoal_file_f)
                    subgoal_path_w = os.path.join(ep, subgoal_file_w)
                    
                    # Also check parent scene directory (e.g., episodes/scene_a/)
                    # Episode paths are like: episodes/scene_a/scene_a_123456/
                    parent_dir = os.path.dirname(ep)
                    subgoal_path_f_parent = os.path.join(parent_dir, subgoal_file_f)
                    subgoal_path_w_parent = os.path.join(parent_dir, subgoal_file_w)
                    
                    # Load front subgoal
                    if os.path.exists(subgoal_path_f):
                        goal_img_f = self._load_img(subgoal_path_f)
                        scene_subgoal_loaded = True
                    elif os.path.exists(subgoal_path_f_parent):
                        goal_img_f = self._load_img(subgoal_path_f_parent)
                        scene_subgoal_loaded = True
                    
                    # Load wrist subgoal if available
                    if os.path.exists(subgoal_path_w):
                        goal_img_w = self._load_img(subgoal_path_w)
                    elif os.path.exists(subgoal_path_w_parent):
                        goal_img_w = self._load_img(subgoal_path_w_parent)
                
                # Fallback to random future frame sampling if scene subgoals not available
                if not scene_subgoal_loaded:
                    if use_fractional_goals and clip_T >= self.min_clip_frames:
                        # Determine which phase we're in based on position in clip
                        relative_pos = (t0 - clip_start) / clip_T if clip_T > 0 else 0.0
                        
                        if relative_pos < 0.7:
                            # Grasp phase (first 70%): goal is near end of grasp phase
                            goal_fraction = 0.35 + random.uniform(0, 0.05)
                        elif relative_pos < 0.9:
                            # Transport phase (middle 20%): goal is near end of transport
                            goal_fraction = 0.75 + random.uniform(0, 0.05)
                        else:
                            # Place phase (final 10%): goal is final frame
                            goal_fraction = 0.95 + random.uniform(0, 0.05)
                        
                        goal_t = clip_start + int(goal_fraction * clip_T)
                        goal_t = min(goal_t, clip_end - 1)  # Ensure within bounds
                    else:
                        # Standard uniform sampling up to max_goal_offset
                        max_goal_t = min(t0 + max_goal_offset, clip_end - 1)
                        if max_goal_t > t0:
                            goal_t = random.randint(t0 + 1, max_goal_t)
                        else:
                            goal_t = clip_end - 1  # Fallback to last frame
                    
                    goal_path_f = os.path.join(ep, f"front_{goal_t:06d}.png")
                    goal_path_w = os.path.join(ep, f"wrist_{goal_t:06d}.png")
                    if os.path.exists(goal_path_f):
                        goal_img_f = self._load_img(goal_path_f)
                    if os.path.exists(goal_path_w):
                        goal_img_w = self._load_img(goal_path_w)
            
            if self.use_embeddings:
                emb_path = os.path.join(ep, "embeddings.pt")
                if os.path.exists(emb_path):
                    embs = self._load_embeddings(emb_path)
                    if embs is not None:
                        ef_all = embs["front"]
                        ew_all = embs["wrist"]
                        indices = list(range(t0-k, t0+H+1))
                        indices = [min(max(i, 0), len(ef_all)-1) for i in indices]
                        emb_f = ef_all[indices]
                        emb_w = ew_all[indices]

            if emb_f is None:
                # V-JEPA 2 style: sample crop parameters ONCE per sequence
                # The same crop is applied to all frames to maintain temporal consistency
                # First, get source image dimensions from first frame
                first_frame_path = os.path.join(ep, f"front_{t0:06d}.png")
                if os.path.exists(first_frame_path):
                    with Image.open(first_frame_path) as probe_img:
                        src_width, src_height = probe_img.size
                else:
                    # Fallback to target size if we can't probe
                    src_width, src_height = self.image_size
                
                # Sample augmentation parameters once for this sequence
                # Use same crop for front and wrist (spatial consistency across views)
                crop_params = self.sample_crop_params(src_width, src_height) if self.augment else None
                
                for t in range(t0-k, t0+H+1):
                    fpath_f = os.path.join(ep, f"front_{t:06d}.png")
                    fpath_w = os.path.join(ep, f"wrist_{t:06d}.png")
                    if not os.path.exists(fpath_f): fpath_f = os.path.join(ep, f"front_{t0:06d}.png")
                    if not os.path.exists(fpath_w): fpath_w = os.path.join(ep, f"wrist_{t0:06d}.png")
                    # Apply same crop to all frames in sequence
                    imgs_f.append(self._load_img(fpath_f, crop_params))
                    imgs_w.append(self._load_img(fpath_w, crop_params))
                imgs_f = np.stack(imgs_f,0); imgs_w = np.stack(imgs_w,0)
            else:
                # Use latent embeddings (T_win, D); keep as numpy for consistency
                if isinstance(emb_f, torch.Tensor):
                    imgs_f = emb_f.numpy()
                    imgs_w = emb_w.numpy()
                else:
                    imgs_f = np.asarray(emb_f)
                    imgs_w = np.asarray(emb_w)

            # Extract joint states (q)
            s_cols = [c for c in df.columns if c.startswith("state_q")]
            if len(s_cols)==0:
                state = np.zeros((T, self.state_dim), dtype=np.float32)
            else:
                if len(s_cols)==1 and df[s_cols[0]].dtype == object:
                    state = np.stack([np.asarray(row, dtype=np.float32) for row in df[s_cols[0]].to_numpy()], axis=0)
                else:
                    state = df[s_cols].to_numpy(dtype=np.float32)
                if state.ndim == 1:
                    state = state[:, None]
            if self.state_dim and state.shape[1] != self.state_dim:
                if state.shape[1] > self.state_dim:
                    state = state[:, :self.state_dim]
                else:
                    pad = self.state_dim - state.shape[1]
                    state = np.pad(state, ((0,0),(0,pad)), mode="constant")
            s_window = state[t0-k:t0+H+1]
            
            # Extract end-effector states (for V-JEPA 2-AC conditioning)
            # EE state format: [pos_x, pos_y, pos_z, roll, pitch, yaw, gripper] (7D)
            ee_cols = [c for c in df.columns if c.startswith("state_ee_state")]
            if len(ee_cols) > 0:
                if df[ee_cols[0]].dtype == object:
                    ee_state = np.stack([np.asarray(row, dtype=np.float32) for row in df[ee_cols[0]].to_numpy()], axis=0)
                else:
                    ee_state = df[ee_cols].to_numpy(dtype=np.float32)
                if ee_state.ndim == 1:
                    ee_state = ee_state[:, None]
                # Validate shape (relaxed for backward compatibility)
                if ee_state.shape[1] != 7:
                    import warnings
                    warnings.warn(
                        f"ee_state has {ee_state.shape[1]}D instead of expected 7D. "
                        f"This is likely a legacy dataset. Consider regenerating for optimal performance."
                    )
                    # Pad or truncate to 7D for consistency
                    if ee_state.shape[1] > 7:
                        ee_state = ee_state[:, :7]
                    else:
                        pad_width = ((0, 0), (0, 7 - ee_state.shape[1]))
                        ee_state = np.pad(ee_state, pad_width, mode='constant')
                ee_window = ee_state[t0-k:t0+H+1]  # (window_len, 7)
            else:
                # use zeros if ee_state not available (legacy datasets)
                ee_window = np.zeros((k+H+1, 7), dtype=np.float32)

            # Extract joint velocity actions (qdot) - LEGACY ONLY
            # For VJEPA2-AC, use ee_delta instead! This is kept only for backward
            # compatibility with old datasets. Meta's VJEPA2-AC uses 7D end-effector deltas.
            a_cols = [c for c in df.columns if c.startswith("act_qdot")]
            if len(a_cols)==0:
                actions = np.zeros((T, self.action_dim), dtype=np.float32)
            else:
                first_col = a_cols[0]
                series = df[first_col]
                if series.dtype == object:
                    actions = np.stack([np.asarray(row, dtype=np.float32) for row in series.to_numpy()], axis=0)
                else:
                    actions = df[a_cols].to_numpy(dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[:, None]
            if self.action_dim and actions.shape[1] != self.action_dim:
                if actions.shape[1] > self.action_dim:
                    actions = actions[:, :self.action_dim]
                else:
                    pad = self.action_dim - actions.shape[1]
                    actions = np.pad(actions, ((0,0),(0,pad)), mode="constant")
            a_window = actions[t0-k:t0+H+1]  # Not used for VJEPA2-AC (uses ee_delta)
            
            # Extract end-effector delta actions (for V-JEPA 2-AC training)
            # EE delta format: [dpos_x, dpos_y, dpos_z, droll, dpitch, dyaw, dgripper] (7D)
            ee_delta_cols = [c for c in df.columns if c.startswith("act_ee_delta")]
            if len(ee_delta_cols) > 0:
                if df[ee_delta_cols[0]].dtype == object:
                    ee_delta = np.stack([np.asarray(row, dtype=np.float32) for row in df[ee_delta_cols[0]].to_numpy()], axis=0)
                else:
                    ee_delta = df[ee_delta_cols].to_numpy(dtype=np.float32)
                if ee_delta.ndim == 1:
                    ee_delta = ee_delta[:, None]
                # Validate shape (relaxed for backward compatibility)
                if ee_delta.shape[1] != 7:
                    import warnings
                    warnings.warn(
                        f"ee_delta has {ee_delta.shape[1]}D instead of expected 7D. "
                        f"This is likely a legacy dataset. Consider regenerating for optimal performance."
                    )
                    # Pad or truncate to 7D for consistency
                    if ee_delta.shape[1] > 7:
                        ee_delta = ee_delta[:, :7]
                    else:
                        pad_width = ((0, 0), (0, 7 - ee_delta.shape[1]))
                        ee_delta = np.pad(ee_delta, pad_width, mode='constant')
                ee_delta_window = ee_delta[t0-k:t0+H+1]  # (window_len, 7)
            else:
                # use zeros if ee_delta not available (legacy datasets)
                ee_delta_window = np.zeros((k+H+1, 7), dtype=np.float32)

            import json
            meta_path = os.path.join(ep, "meta.json")
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path))
                language = meta.get("language", "place the block")
                subgoals = meta.get("subgoals", [])
            else:
                language = "place the block"
                subgoals = []
            token = hash(language) % 512

            batch.append({
                "imgs_front": imgs_f,
                "imgs_wrist": imgs_w,
                "goal_img_front": goal_img_f,  # Sampled future frame as goal
                "goal_img_wrist": goal_img_w,  # Sampled future frame as goal
                "state": s_window,
                "actions": a_window,
                "ee_state": ee_window,       # End-effector states (7D)
                "ee_delta": ee_delta_window,  # End-effector delta actions (7D)
                "token": token,
                "language": language,
                "subgoals": subgoals,
            })

        def to_tensor(x): import torch, numpy as np; return torch.from_numpy(np.ascontiguousarray(x))
        imgs_front = to_tensor(np.stack([b["imgs_front"] for b in batch],0))
        imgs_wrist = to_tensor(np.stack([b["imgs_wrist"] for b in batch],0))
        
        # Stack goal images (may be None for some samples)
        # If goal image sampling is enabled, add to batch
        goal_imgs_f_list = [b.get("goal_img_front") for b in batch]
        goal_imgs_w_list = [b.get("goal_img_wrist") for b in batch]
        has_goals = any(g is not None for g in goal_imgs_f_list)
        if has_goals and sample_goal_images:
            # Replace None with zeros of appropriate shape
            img_shape = imgs_front.shape[2:]  # (C, H, W)
            goal_imgs_f_np = [g if g is not None else np.zeros(img_shape, dtype=np.float32) 
                              for g in goal_imgs_f_list]
            goal_imgs_w_np = [g if g is not None else np.zeros(img_shape, dtype=np.float32) 
                              for g in goal_imgs_w_list]
            goal_imgs_front = to_tensor(np.stack(goal_imgs_f_np, 0))
            goal_imgs_wrist = to_tensor(np.stack(goal_imgs_w_np, 0))
        else:
            goal_imgs_front = None
            goal_imgs_wrist = None
        
        state = to_tensor(np.stack([b["state"] for b in batch],0))
        actions = to_tensor(np.stack([b["actions"] for b in batch],0))
        ee_state = to_tensor(np.stack([b["ee_state"] for b in batch],0))
        ee_delta = to_tensor(np.stack([b["ee_delta"] for b in batch],0))
        token = torch.tensor([b["token"] for b in batch], dtype=torch.long)
        
        result = {
            "imgs_front": imgs_front,
            "imgs_wrist": imgs_wrist,
            "state": state,           # Joint positions (legacy, not used by VJEPA2-AC)
            "actions": actions,       # Joint velocities (legacy, not used by VJEPA2-AC)
            # PRIMARY FIELDS FOR VJEPA2-AC (Meta's implementation):
            "ee_state": ee_state,     # (B, K+H+1, 7) end-effector states [pos(3), rot(3), gripper(1)]
            "ee_delta": ee_delta,     # (B, K+H+1, 7) end-effector deltas [dpos(3), drot(3), dgripper(1)]
            "token": token,
            # Keep raw language strings in Python list for adapters that need text prompts.
            # Length is B; element i corresponds to the i-th item in the batch.
            "language": [b.get("language", "place the block") for b in batch],
        }
        
        # Add goal images if available (Meta's hindsight relabeling approach)
        if goal_imgs_front is not None:
            result["goal_imgs_front"] = goal_imgs_front  # (B, C, H, W)
        if goal_imgs_wrist is not None:
            result["goal_imgs_wrist"] = goal_imgs_wrist  # (B, C, H, W)
        
        return result
