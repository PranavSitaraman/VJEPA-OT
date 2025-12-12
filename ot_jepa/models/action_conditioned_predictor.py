from __future__ import annotations
import torch
import torch.nn as nn
import math


class ActionConditionedPredictor(nn.Module):
    """
    Transformer-based predictor for action-conditioned world modeling.
    
    Matches VJEPA2-AC pretrained model architecture from Meta:
    - Uses end-effector representation (not joint angles)
    - Both actions and states are 7D: [position (3), rotation (3), gripper (1)]
    
    Args:
        latent_dim: Dimension of visual latent features from encoder
        action_dim: Dimension of action vectors (7 for end-effector delta)
        state_dim: Dimension of proprioceptive state vectors (7 for end-effector pose)
        hidden_dim: Hidden dimension of transformer (default: 1024)
        num_layers: Number of transformer layers (default: 24)
        num_heads: Number of attention heads (default: 16)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        latent_dim: int = 1408,  # ViT-G embedding dim
        action_dim: int = 7,  # End-effector delta: [pos_delta (3), rot_delta (3), gripper_delta (1)]
        state_dim: int = 7,  # End-effector pose: [pos (3), rot (3), gripper (1)]
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Input projections (map to hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Transformer with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (map back to latent_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Token type embeddings to distinguish action/state/latent tokens
        self.token_type_embed = nn.Embedding(3, hidden_dim)  # 0=action, 1=state, 2=latent
        
        # Temporal position embeddings
        self.register_buffer("pos_encoding", self._create_positional_encoding(max_len=256, d_model=hidden_dim))
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def _create_block_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Create block-causal attention mask for action-conditioned prediction.
        
        Each timestep has 3 tokens: [action_t, state_t, latent_t]
        Block-causal means: tokens at time t can attend to all tokens at time <= t
        
        Args:
            T: Number of timesteps
            device: Device for the mask
            
        Returns:
            Attention mask of shape (T*3, T*3) where True = masked (not attend)
        """
        # Total sequence length = T timesteps * 3 tokens per timestep
        seq_len = T * 3
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for t_query in range(T):
            for t_key in range(T):
                if t_key > t_query:
                    # Future timesteps are masked
                    query_start = t_query * 3
                    query_end = (t_query + 1) * 3
                    key_start = t_key * 3
                    key_end = (t_key + 1) * 3
                    mask[query_start:query_end, key_start:key_end] = True
        
        return mask
    
    def forward(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
        latents: torch.Tensor,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            actions: (B, T, action_dim) - action sequence
            states: (B, T, state_dim) - proprioceptive state sequence  
            latents: (B, T, latent_dim) - visual latent sequence from frozen encoder
            return_all_timesteps: If True, return predictions for all T timesteps;
                                 if False, return only last timestep (default)
        
        Returns:
            If return_all_timesteps=True: (B, T, latent_dim) - predicted next latents
            If return_all_timesteps=False: (B, latent_dim) - predicted next latent at last timestep
        """
        B, T, _ = latents.shape
        device = latents.device
        
        # Project inputs to hidden_dim
        a_tokens = self.action_proj(actions)  # (B, T, hidden_dim)
        s_tokens = self.state_proj(states)    # (B, T, hidden_dim)
        z_tokens = self.latent_proj(latents)  # (B, T, hidden_dim)
        
        # Interleave tokens: [a_0, s_0, z_0, a_1, s_1, z_1, ...]
        # Shape: (B, T*3, hidden_dim)
        tokens = torch.stack([a_tokens, s_tokens, z_tokens], dim=2).view(B, T*3, self.hidden_dim)
        
        # Add token type embeddings
        # 0=action, 1=state, 2=latent, repeated T times
        token_types = torch.tensor([0, 1, 2] * T, device=device)  # (T*3,)
        type_embeds = self.token_type_embed(token_types).unsqueeze(0).expand(B, -1, -1)  # (B, T*3, hidden_dim)
        tokens = tokens + type_embeds
        
        # Add positional encoding
        pos_embeds = self.pos_encoding[:T*3].unsqueeze(0).expand(B, -1, -1)  # (B, T*3, hidden_dim)
        tokens = tokens + pos_embeds
        
        # Create block-causal mask
        attn_mask = self._create_block_causal_mask(T, device)
        
        # Apply transformer
        h = self.transformer(tokens, mask=attn_mask, is_causal=False)  # (B, T*3, hidden_dim)
        
        # Extract latent tokens (every 3rd token, offset by 2: indices 2, 5, 8, ...)
        latent_token_indices = torch.arange(2, T*3, 3, device=device)  # [2, 5, 8, ...]
        h_latents = h[:, latent_token_indices, :]  # (B, T, hidden_dim)
        
        # Project back to latent_dim to predict next latent
        pred_latents = self.output_proj(h_latents)  # (B, T, latent_dim)
        
        if return_all_timesteps:
            return pred_latents  # (B, T, latent_dim)
        else:
            return pred_latents[:, -1, :]  # (B, latent_dim) - last timestep only
    
    def rollout(
        self,
        initial_latent: torch.Tensor,
        initial_state: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Autoregressive rollout for planning (no teacher forcing).
        
        Args:
            initial_latent: (B, latent_dim) - starting visual latent
            initial_state: (B, state_dim) - starting proprioceptive state
            action_sequence: (B, T, action_dim) - planned action sequence
        
        Returns:
            pred_latents: (B, T, latent_dim) - predicted future latents
        """
        B, T, _ = action_sequence.shape
        device = action_sequence.device
        
        # Initialize sequences
        latents = [initial_latent]  # List of (B, latent_dim)
        states = [initial_state]     # List of (B, state_dim)
        
        # Autoregressively predict future
        for t in range(T):
            # Current action
            a_t = action_sequence[:, t, :]  # (B, action_dim)
            
            # Compute next state (integrate action)
            # For VJEPA2: action and state are BOTH 7D end-effector representations
            # action = ee_delta = [pos_delta, rot_delta, gripper_delta]
            # state = ee_pose = [pos, rot, gripper]
            # Integration: s_{t+1} = s_t + a_t
            s_next = states[-1] + a_t  # (B, state_dim)
            states.append(s_next)
            
            # Stack history for transformer
            # Convert lists to tensors
            latent_hist = torch.stack(latents, dim=1)  # (B, t+1, latent_dim)
            state_hist = torch.stack(states[:-1], dim=1)  # (B, t+1, state_dim) - exclude s_next
            action_hist = action_sequence[:, :t+1, :]   # (B, t+1, action_dim)
            
            # Predict next latent using full history
            with torch.no_grad():
                z_next_pred = self.forward(
                    actions=action_hist,
                    states=state_hist,
                    latents=latent_hist,
                    return_all_timesteps=False,  # Only need last prediction
                )  # (B, latent_dim)
            
            latents.append(z_next_pred)
        
        # Return all predicted latents (excluding initial)
        pred_latents = torch.stack(latents[1:], dim=1)  # (B, T, latent_dim)
        return pred_latents
