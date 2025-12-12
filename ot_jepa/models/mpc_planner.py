from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation

def compute_new_pose(pose: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Compute new end-effector pose from current pose and action delta.
    
    Following Meta's VJEPA2-AC implementation (vjepa2/notebooks/utils/mpc_utils.py):
    - Position: simple addition (new_xyz = pose_xyz + action_xyz)
    - Rotation: proper rotation composition using rotation matrices
    - Gripper: simple addition with clipping to [0, 1]
    
    Args:
        pose: Current end-effector pose [B, T=1, 7] or [B, 7]
              Format: [x, y, z, roll, pitch, yaw, gripper]
        action: Action delta [B, T=1, 7] or [B, 7]
                Format: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]
    
    Returns:
        new_pose: Updated pose [B, T=1, 7] or [B, 7]
    """
    # Handle both [B, T, 7] and [B, 7] inputs
    squeeze_output = False
    if pose.dim() == 2:
        pose = pose.unsqueeze(1)
        action = action.unsqueeze(1)
        squeeze_output = True
    elif pose.dim() == 3 and pose.shape[1] > 1:
        # Take last timestep
        pose = pose[:, -1:, :]
        action = action[:, -1:, :]
    
    device, dtype = pose.device, pose.dtype
    B = pose.shape[0]
    
    # Simple addition for position (same as Meta)
    new_xyz = pose[..., :3] + action[..., :3]
    
    # Use scipy for accurate rotation composition
    # Detach to avoid graph retention for numpy conversion if needed
    # But in MPC rollout we need gradients? 
    # Meta's code uses .cpu().numpy(), so it likely DOES NOT backprop through dynamics during MPC sampling/rollout?
    # CEM is derivative-free. So we don't need gradients here.
    pose_np = pose[:, 0].detach().cpu().numpy()
    action_np = action[:, 0].detach().cpu().numpy()
    
    thetas = pose_np[:, 3:6]  # [B, 3] - current euler angles
    delta_thetas = action_np[:, 3:6]  # [B, 3] - delta euler angles
    
    # Convert to rotation matrices and compose
    new_angles = []
    for i in range(B):
        R_current = Rotation.from_euler("xyz", thetas[i], degrees=False).as_matrix()
        R_delta = Rotation.from_euler("xyz", delta_thetas[i], degrees=False).as_matrix()
        R_new = R_delta @ R_current  # Compose: apply delta to current
        new_euler = Rotation.from_matrix(R_new).as_euler("xyz", degrees=False)
        new_angles.append(new_euler)
    
    new_angle = torch.from_numpy(np.stack(new_angles, axis=0)).to(device).to(dtype)
    new_angle = new_angle.unsqueeze(1)  # [B, 1, 3]
    
    # Gripper: simple addition with clipping
    new_gripper = pose[..., 6:7] + action[..., 6:7]
    new_gripper = torch.clip(new_gripper, 0.0, 1.0)
    
    # Assemble new pose
    new_pose = torch.cat([new_xyz, new_angle, new_gripper], dim=-1)
    
    if squeeze_output:
        new_pose = new_pose.squeeze(1)
    
    return new_pose


class MPCPlanner(nn.Module):
    """Model Predictive Control planner using Cross-Entropy Method.
    
    Given current observation and goal image, plans an action sequence by:
    1. Sampling action trajectories from Gaussian distributions
    2. Using the predictor to imagine future states for each trajectory (on PATCH TOKENS)
    3. Computing energy (L1 distance to goal) for each trajectory
    4. Updating Gaussian parameters based on top-k trajectories
    5. Returning the first action from the optimized sequence
    
    This follows Meta's VJEPA2-AC implementation which operates on patch tokens
    throughout the planning process, only pooling for the final energy computation.
    """
    
    def __init__(
        self,
        act_dim: int = 7,
        planning_horizon: int = 2,  # Meta VJEPA2-AC: rollout=2
        num_samples: int = 400,     # Meta VJEPA2-AC: samples=400
        num_iterations: int = 10,   # Meta VJEPA2-AC: cem_steps=10
        top_k: int = 10,            # Meta VJEPA2-AC: topk=10
        action_std_init: float = 0.05,  # Initial sampling std (Meta uses maxnorm as scale ref)
        action_mean_init: float = 0.0,
        momentum_mean: float = 0.15,  # Meta VJEPA2-AC: 0.15
        momentum_std: float = 0.15,   # Meta VJEPA2-AC: 0.15
        action_l1_max: float = 0.05,  # Meta VJEPA2-AC: maxnorm=0.05 (Box constraint)
        temperature: float = 1.0,     # Not used in Meta's simple update, kept for compatibility
        action_scale: float = 1.0,    # Scale actions before passing to predictor (e.g. 35.0)
    ):
        """Initialize MPC planner.
        
        Args:
            act_dim: Action dimension (7 for end-effector control)
            planning_horizon: Number of timesteps to plan ahead (rollout in paper)
            num_samples: Number of action trajectories to sample per iteration
            num_iterations: Number of CEM optimization iterations
            top_k: Number of best trajectories to use for updating distributions
            action_std_init: Initial standard deviation for action sampling
            action_mean_init: Initial mean for action sampling
            momentum_mean: Momentum for updating mean
            momentum_std: Momentum for updating std
            action_l1_max: Box constraint limit for position deltas (maxnorm)
            action_scale: Scaling factor for actions passed to predictor
        """
        super().__init__()
        self.act_dim = act_dim
        self.planning_horizon = planning_horizon
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.top_k = top_k
        self.action_std_init = action_std_init
        self.action_mean_init = action_mean_init
        self.momentum_mean = momentum_mean
        self.momentum_std = momentum_std
        self.action_l1_max = action_l1_max
        self.temperature = temperature
        self.action_scale = action_scale
    
    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Clip actions to box constraint (Meta VJEPA2-AC approach).
        
        Args:
            actions: Action tensor of shape [..., act_dim]
                     [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]
        
        Returns:
            Clipped actions
        """
        # Meta clips XYZ to [-maxnorm, maxnorm]
        xyz = torch.clamp(actions[..., :3], min=-self.action_l1_max, max=self.action_l1_max)
        
        # Rotation is zeroed in Meta's implementation (simplification)
        rot = torch.zeros_like(actions[..., 3:6])
        
        # Meta clips gripper to [-0.75, 0.75]
        gripper = torch.clamp(actions[..., 6:7], min=-0.75, max=0.75)
        
        return torch.cat([xyz, rot, gripper], dim=-1)
    
    def forward(
        self,
        z_current: torch.Tensor,
        z_goal: torch.Tensor,
        predictor: nn.Module,
        state_current: Optional[torch.Tensor] = None,
        use_patch_tokens: bool = False,
        tokens_per_frame: Optional[int] = None,
        prev_action_mean: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        target_pos: Optional[torch.Tensor] = None,
        position_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan action sequence using CEM optimization.
        
        Args:
            z_current: Current observation encoding
                       - If use_patch_tokens=False: [B, D] pooled latent
                       - If use_patch_tokens=True: [B, N_patches, D] patch tokens
            z_goal: Goal observation encoding (same format as z_current)
            predictor: Predictor network P_phi that predicts future states
            state_current: Current end-effector state [B, 7] (optional)
            use_patch_tokens: Whether to use patch-token-level planning (Meta's approach)
            tokens_per_frame: Number of patch tokens per frame (required if use_patch_tokens=True)
            prev_action_mean: [B, T, act_dim] Mean from previous step (for warm start)
            encoder: Optional encoder to pool+project tokens (for training-aligned energy)
            
        Returns:
            (action, full_plan): 
                - action: [B, act_dim] First action to execute
                - full_plan: [B, T, act_dim] Optimized trajectory mean
        """
        if use_patch_tokens and tokens_per_frame is not None:
            # Use Meta's patch-token-based MPC (correct VJEPA2-AC approach)
            return self._plan_with_patch_tokens(
                z_current, z_goal, predictor, state_current, tokens_per_frame, prev_action_mean, encoder,
                target_pos=target_pos, position_weight=position_weight
            )
        else:
            # simplified latent-space MPC
            return self._plan_with_pooled_latents(
                z_current, z_goal, predictor, state_current, prev_action_mean
            )
    
    def _plan_with_pooled_latents(
        self,
        z_current: torch.Tensor,
        z_goal: torch.Tensor,
        predictor: nn.Module,
        state_current: Optional[torch.Tensor] = None,
        prev_action_mean: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simplified latent-space planning (for non-VJEPA2-AC predictors).
        """
        batch_size = z_current.shape[0]
        device = z_current.device
        
        # Initialize Gaussian distributions
        if prev_action_mean is not None:
            action_mean = prev_action_mean.clone()
        else:
            action_mean = torch.full(
                (batch_size, self.planning_horizon, self.act_dim),
                self.action_mean_init,
                device=device,
            )
        
        action_std = torch.full(
            (batch_size, self.planning_horizon, self.act_dim),
            self.action_std_init,
            device=device,
        )
        
        # CEM optimization loop
        for iteration in range(self.num_iterations):
            eps = torch.randn(
                batch_size,
                self.num_samples,
                self.planning_horizon,
                self.act_dim,
                device=device,
            )
            action_samples = action_mean.unsqueeze(1) + action_std.unsqueeze(1) * eps
            
            # Apply Box Constraint
            action_samples = self._clip_actions(action_samples)
            
            # Compute energy
            energies = self._compute_energy(
                action_samples, z_current, z_goal, predictor, state_current
            )
            
            # Select top-k
            top_k_indices = torch.topk(energies, self.top_k, dim=1, largest=False).indices
            
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, self.top_k, self.planning_horizon, self.act_dim
            )
            top_k_actions = torch.gather(action_samples, 1, top_k_indices_expanded)
            
            # Update mean and std (simple mean of top-k)
            mean_selected = top_k_actions.mean(dim=1)  # [B, T, act_dim]
            std_selected = top_k_actions.std(dim=1)    # [B, T, act_dim]
            
            action_mean = (1.0 - self.momentum_mean) * mean_selected + self.momentum_mean * action_mean
            action_std = (1.0 - self.momentum_std) * std_selected + self.momentum_std * action_std
        
        return action_mean[:, 0, :], action_mean
    
    def _plan_with_patch_tokens(
        self,
        z_current: torch.Tensor,
        z_goal: torch.Tensor,
        predictor: nn.Module,
        state_current: torch.Tensor,
        tokens_per_frame: int,
        prev_action_mean: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        target_pos: Optional[torch.Tensor] = None,
        position_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan using patch-token-level prediction (Meta's VJEPA2-AC approach).
        
        Args:
            target_pos: Optional [B, 3] target position for position-guided MPC
            position_weight: Weight for position-based energy term (0.0 = pure latent MPC)
        """
        batch_size = z_current.shape[0]
        device = z_current.device
        
        # Initialize Gaussian distributions
        # Only optimize position (xyz) + gripper. Rotation is zero.
        if prev_action_mean is not None:
            action_mean = prev_action_mean.clone()
        else:
            action_mean = torch.zeros(
                (batch_size, self.planning_horizon, self.act_dim),
                device=device,
            )
            
        # Initialize std (Meta uses maxnorm=0.05 as reference scale)
        action_std = torch.ones(
            (batch_size, self.planning_horizon, self.act_dim),
            device=device,
        ) * self.action_std_init
        
        # Track if MPC is actually optimizing
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        # Pre-compute tokens_per_frame to avoid repeated computation
        tokens_per_frame_cached = tokens_per_frame
        
        for iteration in range(self.num_iterations):
            # Distributed sampling logic
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                samples_per_rank = self.num_samples // world_size
                if rank == 0:
                    samples_this_rank = self.num_samples - (samples_per_rank * (world_size - 1))
                else:
                    samples_this_rank = samples_per_rank
            else:
                world_size = 1
                rank = 0
                samples_this_rank = self.num_samples
            
            # Sample actions
            eps = torch.randn(
                batch_size,
                samples_this_rank,
                self.planning_horizon,
                self.act_dim,
                device=device,
            )
            action_samples = action_mean.unsqueeze(1) + action_std.unsqueeze(1) * eps
            
            # Apply Box Constraint (Meta approach)
            action_samples = self._clip_actions(action_samples)
            
            # Compute latent energy (Meta's VJEPA2-AC approach)
            energies = self._compute_energy_patch_tokens(
                action_samples, z_current, z_goal, predictor, 
                state_current, tokens_per_frame, encoder
            )
            
            # Add position-based energy term if target position is provided
            # This helps guide the robot when the latent energy landscape is flat
            if target_pos is not None and position_weight > 0.0:
                # Compute predicted final position from action trajectory
                # state_current: [B, 7] = [x, y, z, roll, pitch, yaw, gripper]
                # action_samples: [B, N, H, 7] = action deltas
                
                # Sum position deltas over horizon to get final position offset
                pos_deltas = action_samples[..., :3].sum(dim=2)  # [B, N, 3]
                
                # Predicted final position = current position + sum of deltas
                current_pos = state_current[:, :3].unsqueeze(1)  # [B, 1, 3]
                pred_final_pos = current_pos + pos_deltas  # [B, N, 3]
                
                # Target position: [B, 3] -> [B, 1, 3]
                target_pos_expanded = target_pos.unsqueeze(1)  # [B, 1, 3]
                
                # L2 distance to target (normalized by typical workspace scale ~0.5m)
                pos_energy = torch.norm(pred_final_pos - target_pos_expanded, dim=-1) / 0.5  # [B, N]
                
                # Combine latent and position energies
                # Normalize latent energy to similar scale (typical range ~0.01-0.03)
                latent_scale = 30.0  # Scale latent energy to ~0.3-0.9 range
                energies = (1.0 - position_weight) * (energies * latent_scale) + position_weight * pos_energy
            
            # Gather results
            if world_size > 1:
                # (Same gathering logic as before...)
                max_samples = self.num_samples - (samples_per_rank * (world_size - 1))
                if samples_this_rank < max_samples:
                    pad_size = max_samples - samples_this_rank
                    energies = F.pad(energies, (0, pad_size), value=float('inf'))
                    action_samples = F.pad(
                        action_samples.reshape(batch_size, samples_this_rank, -1),
                        (0, 0, 0, pad_size),
                        value=0.0
                    ).reshape(batch_size, max_samples, self.planning_horizon, self.act_dim)
                
                gathered_energies = [torch.zeros(batch_size, max_samples, device=device) for _ in range(world_size)]
                gathered_actions = [torch.zeros(batch_size, max_samples, self.planning_horizon, self.act_dim, device=device) for _ in range(world_size)]
                
                dist.all_gather(gathered_energies, energies)
                dist.all_gather(gathered_actions, action_samples)
                
                energies_list = []
                actions_list = []
                for i in range(world_size):
                    if i == 0:
                        energies_list.append(gathered_energies[i])
                        actions_list.append(gathered_actions[i])
                    else:
                        energies_list.append(gathered_energies[i][:, :samples_per_rank])
                        actions_list.append(gathered_actions[i][:, :samples_per_rank, :, :])
                
                energies = torch.cat(energies_list, dim=1)
                action_samples = torch.cat(actions_list, dim=1)
            
            # Select top-k
            top_k_indices = torch.topk(energies, self.top_k, dim=1, largest=False).indices
            
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, self.top_k, self.planning_horizon, self.act_dim
            )
            top_k_actions = torch.gather(action_samples, 1, top_k_indices_expanded)
            
            # IMPROVED: Use energy-weighted mean to amplify small energy differences
            # When energy landscape is flat, unweighted mean treats all top-k equally.
            # Softmax weighting with low temperature gives more influence to best trajectories.
            top_k_energies = torch.gather(energies, 1, top_k_indices)  # [B, top_k]
            
            # Use temperature scaling to amplify differences (lower temp = sharper weights)
            # Energy range is typically ~0.001-0.01, so use high inverse temperature
            cem_temperature = 0.001  # Low temperature amplifies small differences
            weights = torch.softmax(-top_k_energies / cem_temperature, dim=1)  # [B, top_k]
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, top_k, 1, 1]
            
            # Weighted mean and std
            mean_selected = (top_k_actions * weights).sum(dim=1)  # [B, T, act_dim]
            diff_squared = (top_k_actions - mean_selected.unsqueeze(1)) ** 2
            std_selected = torch.sqrt((diff_squared * weights).sum(dim=1) + 1e-6)  # [B, T, act_dim]
            
            action_mean = (1.0 - self.momentum_mean) * mean_selected + self.momentum_mean * action_mean
            action_std = (1.0 - self.momentum_std) * std_selected + self.momentum_std * action_std
            
            # Verbose debug (disabled by default - set _verbose_debug=True to enable)
            if getattr(self, '_verbose_debug', False):
                is_first_iter = (iteration == 0)
                is_last_iter = (iteration == self.num_iterations - 1)
                if self._debug_counter <= 5 and (is_first_iter or is_last_iter):
                    energy_range = energies.max().item() - energies.min().item()
                    top_weight = weights[0, 0, 0, 0].item() if weights.numel() > 0 else 0
                    print(f"Call #{self._debug_counter}, Iter {iteration}:")
                    print(f"Energy: min={energies.min().item():.6f}, max={energies.max().item():.6f}, range={energy_range:.6f}")
                    print(f"Top-k energies: {top_k_energies[0].tolist()[:5]}")
                    print(f"Top weight (best traj): {top_weight:.4f}")
                    print(f"Action mean (xyz): {action_mean[0, 0, :3].detach().cpu().numpy()}")
                    print(f"Action std (xyz): {action_std[0, 0, :3].detach().cpu().numpy()}")
        
        # Return the MEAN action (Meta's approach) for smooth, consistent behavior
        # The mean is more stable than the best sample, which can be noisy and cause oscillation
        # Meta's code: new_action = torch.cat([mean[..., :3], ...])
        
        # Apply gripper deadband (Meta's round_small_elements with threshold=0.25)
        # This zeros out small gripper actions to prevent jitter
        first_action = action_mean[:, 0, :].clone()
        gripper_mask = torch.abs(first_action[:, 6:7]) < 0.25
        first_action[:, 6:7] = torch.where(gripper_mask, torch.zeros_like(first_action[:, 6:7]), first_action[:, 6:7])
        
        return first_action, action_mean
        
    def _compute_energy_patch_tokens(
        self,
        action_sequences: torch.Tensor,
        z_current: torch.Tensor,
        z_goal: torch.Tensor,
        predictor: nn.Module,
        state_current: torch.Tensor,
        tokens_per_frame: int,
        encoder: Optional[nn.Module] = None,  # Kept for API compatibility but not used
    ) -> torch.Tensor:
        """Compute energy using patch-token-level prediction (Meta's VJEPA2-AC approach).
        
        The predictor receives layer_normed tokens and outputs tokens which are then layer_normed.
        Energy is computed as L1 distance on flattened patch tokens (matching Meta's MPC).
        """
        batch_size, num_samples, horizon, _ = action_sequences.shape
        
        # Apply layer_norm to encoder outputs (Meta's normalize_reps=True)
        # Meta's WorldModel.encode() applies F.layer_norm(h, (h.size(-1),)) to encoder output
        # This must be done BEFORE passing to predictor for autoregressive rollout
        
        # Expand current state and tokens
        # z_current shape: (B, N_p, D) - patch tokens for current frame
        # We need (B*N, 1, N_p, D) for consistent 4D format in z_traj_frames
        z_current_expanded = z_current.unsqueeze(1).unsqueeze(1).expand(
            batch_size, num_samples, 1, -1, -1
        ).reshape(batch_size * num_samples, 1, z_current.shape[-2], z_current.shape[-1])
        
        # Apply layer_norm to encoder output (Meta's normalize_reps=True)
        # Meta's code: if self.normalize_reps: h = F.layer_norm(h, (h.size(-1),))
        
        z_current_expanded = F.layer_norm(z_current_expanded, (z_current_expanded.size(-1),))
        
        state_current_expanded = state_current.unsqueeze(1).expand(
            batch_size, num_samples, -1
        ).reshape(batch_size * num_samples, 1, -1)
        
        z_traj_frames = [z_current_expanded]  # List of (B*N, 1, N_p, D) tensors (layer_normed)
        state_traj = state_current_expanded
        
        # Align action/state pairing with training
        # Training uses "departure actions": state s_t is paired with action a_{t+1}
        # (the action that will be taken FROM state s_t to reach s_{t+1})
        #
        # In MPC rollout:
        # We have states [s_0] and want to predict z_1, z_2, ..., z_H
        # For predicting z_1: use (z_0, s_0, a_0) where a_0 is the action FROM s_0
        # For predicting z_2: use (z_0, z_1, s_0, s_1, a_0, a_1)
        #
        # The key insight: action a_h is the "departure action" from state s_h
        # So we pass actions [a_0, ..., a_{h-1}] with states [s_0, ..., s_{h-1}]
        # This matches training where actions_in = ee_delta_batch[:, 1:t_ctx+1, :]
        # (shifted by +1 to get departure actions)
        
        for h in range(horizon):
            action_h = action_sequences[:, :, h, :].reshape(batch_size * num_samples, 1, -1)
            
            # Build action sequence for predictor
            # At step h, we have h actions [a_0, ..., a_{h-1}] and h states [s_0, ..., s_{h-1}]
            # But we also need to include the current action a_h as the "departure action" from s_{h-1}
            # Wait - let me re-read training code more carefully...
            #
            # Training (line 1123): actions_in = ee_delta_batch[:, 1:t_ctx+1, :]
            # For t_ctx frames of context (z_0 to z_{t_ctx-1}), we use actions a_1 to a_{t_ctx}
            # This means: for state s_i, we use action a_{i+1}
            #
            # So in MPC:
            # At h=0: we have z_0, s_0, and want to predict z_1
            #   Training would use: context=[z_0], states=[s_0], actions=[a_1]
            #   But we're sampling a_0 as the action to take from s_0...
            #
            # The issue is: in training, a_i = s_i - s_{i-1} (arrival action)
            # But we shift to get a_{i+1} paired with s_i (departure action)
            # In MPC, we're sampling the action to take FROM current state, which IS the departure action
            #
            # So the fix is: we should NOT accumulate actions, but use the CURRENT action
            # as the departure action for the CURRENT state
            
            T_curr = len(z_traj_frames)
            z_seq = torch.cat(z_traj_frames, dim=1)  # (B*N, T_curr, N_p, D)
            
            # Flatten time and patches to match training input format
            # Training does: x_in = context_tokens.reshape(B, t_ctx * N_p, D_enc)
            # So we need: (B*N, T_curr, N_p, D) -> (B*N, T_curr * N_p, D)
            BN, T_c, N_p, D = z_seq.shape
            z_seq_flat = z_seq.reshape(BN, T_c * N_p, D)
            
            # Build action sequence: all actions up to and including current
            # MPC's a_i is the action to take FROM state s_i (departure action)
            # This matches training where actions_in = ee_delta_batch[:, 1:t_ctx+1, :]
            # (shifted by +1 to pair s_i with action FROM s_i)
            if h == 0:
                actions_seq = action_h  # Just a_0
            else:
                # Accumulate all previous actions plus current
                prev_actions = action_sequences[:, :, :h, :].reshape(batch_size * num_samples, h, -1)
                actions_seq = torch.cat([prev_actions, action_h], dim=1)  # [a_0, ..., a_h]
            
            states_seq = state_traj  # [s_0, ..., s_h]
            
            actions_for_pred = actions_seq * self.action_scale
            
            with torch.no_grad():
                pred_all = predictor(z_seq_flat, actions_for_pred, states_seq)
            
            # Extract predicted tokens for the next frame
            # pred_all shape: (B*N, T_curr * N_p, D) - flattened output
            # We want the last N_p tokens which correspond to the predicted next frame
            pred_tokens = pred_all[:, -tokens_per_frame:, :]  # (B*N, N_p, D)
            
            # Apply layer_norm to predictor output (Meta's normalize_reps=True)
            # Meta's step_predictor: if self.normalize_reps: next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            # This is essential for autoregressive rollout - each step's output must be normalized
            # before being fed back as input to the next step
            pred_tokens = F.layer_norm(pred_tokens, (pred_tokens.size(-1),))
            
            # Add time dimension for concatenation: (B*N, N_p, D) -> (B*N, 1, N_p, D)
            pred_tokens_4d = pred_tokens.unsqueeze(1)
            
            # Compute next state for next iteration
            state_next = compute_new_pose(state_traj[:, -1:, :], action_h)
            z_traj_frames.append(pred_tokens_4d)
            state_traj = torch.cat([state_traj, state_next], dim=1)
        
        # z_traj_frames[-1] has shape (B*N, 1, N_p, D), squeeze time dim
        z_final = z_traj_frames[-1].squeeze(1)  # [B*N, N_p, D] - already layer_normed
        z_current_raw = z_traj_frames[0].squeeze(1)  # [B*N, N_p, D] - NOT normalized yet
        
        # Apply layer_norm to z_current to match z_goal and z_final
        # z_goal was normalized during encoding, z_final was normalized after predictor
        # z_current must also be normalized for consistent direction computation
        z_current_for_energy = F.layer_norm(z_current_raw, (z_current_raw.size(-1),))
        
        # Expand goal to match samples: (B, N_p, D) -> (B*N, N_p, D)
        z_goal_expanded = z_goal.unsqueeze(1).expand(
            batch_size, num_samples, -1, -1
        ).reshape(batch_size * num_samples, z_goal.shape[-2], z_goal.shape[-1])
        
        # Energy computation: all three tensors are now consistently layer-normalized
        # z_current_for_energy: normalized ✓
        # z_final: normalized ✓ (from predictor output)
        # z_goal_expanded: normalized ✓ (from goal encoding)
        
        # Flatten for computation
        z_final_flat = z_final.flatten(1)  # [B*N, N_p*D]
        z_goal_flat = z_goal_expanded.flatten(1)  # [B*N, N_p*D]
        z_curr_flat = z_current_for_energy.flatten(1)  # [B*N, N_p*D]
        
        # Simple and correct energy: L1 distance from predicted final state to goal
        # This is Meta's original VJEPA2-AC approach, now with consistent normalization
        # Lower distance = lower energy = better action
        energy = torch.mean(torch.abs(z_final_flat - z_goal_flat), dim=-1)  # [B*N]
        
        energy = energy.view(batch_size, num_samples)
        
        # Check energy range and predictor sensitivity
        if getattr(self, '_verbose_debug', False) and getattr(self, '_debug_counter', 0) <= 3:
            # How much did predictions differ from current state?
            travel_dist = torch.mean(torch.abs(z_final_flat - z_curr_flat), dim=-1)
            # How far is current from goal?
            curr_to_goal = torch.mean(torch.abs(z_curr_flat - z_goal_flat), dim=-1)
            print(f"Curr→Goal dist: {curr_to_goal.mean():.6f}")
            print(f"Travel dist: min={travel_dist.min():.6f}, max={travel_dist.max():.6f}")
            print(f"Final→Goal (energy): min={energy.min():.6f}, max={energy.max():.6f}")
        
        return energy
    
    def _compute_energy(
        self,
        action_sequences: torch.Tensor,
        z_current: torch.Tensor,
        z_goal: torch.Tensor,
        predictor: nn.Module,
        state_current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute energy for action sequences."""
        # Simplified for L1 energy
        batch_size, num_samples, horizon, act_dim = action_sequences.shape
        device = action_sequences.device
        
        action_flat = action_sequences.reshape(batch_size * num_samples, horizon, act_dim)
        z_current_expanded = z_current.unsqueeze(1).expand(batch_size, num_samples, -1)
        z_current_flat = z_current_expanded.reshape(batch_size * num_samples, -1)
        
        with torch.no_grad():
            if hasattr(predictor, 'rollout'):
                 state_flat = torch.zeros(batch_size * num_samples, self.act_dim, device=device)
                 z_predicted_seq = predictor.rollout(z_current_flat, state_flat, action_flat)
                 z_predicted = z_predicted_seq[:, -1, :]
            else:
                 z_current_3d = z_current_flat.unsqueeze(1)
                 z_predicted = predictor(z_current_3d, actions=action_flat, horizon=horizon)
                 if z_predicted.dim() == 3: z_predicted = z_predicted[:, -1, :]
        
        z_predicted = z_predicted.reshape(batch_size, num_samples, -1)
        z_goal_expanded = z_goal.unsqueeze(1).expand(batch_size, num_samples, -1)
        energies = torch.abs(z_predicted - z_goal_expanded).sum(dim=-1)
        return energies
    
    def plan_sequence(
        self,
        z_current: torch.Tensor,
        z_goal: torch.Tensor,
        predictor: nn.Module,
        state_current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Plan full action sequence (not just first action).
        
        Args:
            z_current: Current observation encoding [B, D]
            z_goal: Goal observation encoding [B, D]
            predictor: Predictor network
            state_current: Current end-effector state [B, state_dim]
            
        Returns:
            action_sequence: Optimized action sequence [B, T, act_dim]
        """
        batch_size = z_current.shape[0]
        device = z_current.device
        
        # Initialize distributions
        action_mean = torch.full(
            (batch_size, self.planning_horizon, self.act_dim),
            self.action_mean_init,
            device=device,
        )
        action_std = torch.full(
            (batch_size, self.planning_horizon, self.act_dim),
            self.action_std_init,
            device=device,
        )
        
        # CEM optimization
        for iteration in range(self.num_iterations):
            eps = torch.randn(
                batch_size,
                self.num_samples,
                self.planning_horizon,
                self.act_dim,
                device=device,
            )
            action_samples = action_mean.unsqueeze(1) + action_std.unsqueeze(1) * eps
            
            # Apply box constraint (Meta VJEPA2-AC: maxnorm=0.05 for XYZ, zero rotation)
            original_shape = action_samples.shape  # [B, N, T, act_dim]
            action_samples_flat = action_samples.reshape(-1, self.act_dim)  # [B*N*T, act_dim]
            action_samples_flat = self._clip_actions(action_samples_flat)
            action_samples = action_samples_flat.reshape(original_shape)  # [B, N, T, act_dim]
            
            energies = self._compute_energy(
                action_samples, z_current, z_goal, predictor, state_current
            )
            
            top_k_indices = torch.topk(energies, self.top_k, dim=1, largest=False).indices
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(
                batch_size, self.top_k, self.planning_horizon, self.act_dim
            )
            top_k_actions = torch.gather(action_samples, 1, top_k_indices_expanded)
            
            top_k_energies = torch.gather(energies, 1, top_k_indices)
            weights = torch.softmax(-top_k_energies / self.temperature, dim=1)
            weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
            
            action_mean = (top_k_actions * weights_expanded).sum(dim=1)
            action_var = ((top_k_actions - action_mean.unsqueeze(1)) ** 2 * weights_expanded).sum(dim=1)
            action_std = torch.sqrt(action_var + 1e-6)
        
        return action_mean
