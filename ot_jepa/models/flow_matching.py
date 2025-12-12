from __future__ import annotations
import torch, torch.nn as nn
import math


#A less generalized version of : https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_riemannian_flow_matching_flat_torus.ipynb

class SinusoidalPositionalEmbedding(nn.Module):
    #Transformer-like positional embedding for time = t 
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B, 1)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FlowMatchingHead(nn.Module):
    def __init__(self, latent_dim=256, act_dim=7, time_embed_dim=64, token_width=256):
        super().__init__()
        self.act_dim = act_dim

        # π0-style time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Action token projection that mixes action + time (cf. π0 action expert)
        self.action_proj = nn.Sequential(
            nn.Linear(act_dim, token_width),
            nn.SiLU(),
        )
        self.action_time_proj = nn.Sequential(
            nn.Linear(token_width + time_embed_dim, token_width),
            nn.SiLU(),
            nn.Linear(token_width, token_width),
        )

        # Separate projections for state/vision latent and goal latent (late fusion)
        # "goal" can be either goal image latent or language latent depending on use case
        self.state_proj = nn.Sequential(
            nn.Linear(latent_dim, token_width),
            nn.SiLU(),
        )
        self.goal_proj = nn.Sequential(
            nn.Linear(latent_dim, token_width),
            nn.SiLU(),
        )

        fused_dim = token_width * 3
        self.fuse_norm = nn.LayerNorm(fused_dim)
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x, z, z_goal, t):
        # x: noisy action, z: fused visual/state latent, z_goal: goal latent (image or language)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t_emb = self.time_mlp(t)
        action_token = self.action_proj(x)
        action_token = self.action_time_proj(torch.cat([action_token, t_emb], dim=-1))

        state_token = self.state_proj(z)
        goal_token = self.goal_proj(z_goal)

        fused = torch.cat([action_token, state_token, goal_token], dim=-1)
        fused = self.fuse_norm(fused)
        return self.net(fused)
    
    @torch.no_grad()
    def sample(self, z, z_goal, num_steps: int = 5):
        """Sample action using Euler integration of the learned flow.
        
        Args:
            z: Current observation latent [B, D]
            z_goal: Goal latent (from goal image or language) [B, D]
            num_steps: Number of integration steps (default: 5)
            
        Returns:
            action: Sampled action [B, act_dim]
        """
        device = z.device
        batch_size = z.shape[0]
        act_dim = self.net[-1].out_features  # Infer from final layer
        
        # Start from noise: x_0 ~ N(0, I)
        x = torch.randn(batch_size, act_dim, device=device)
        
        # Euler integration: x_{t+dt} = x_t + dt * v_theta(x_t, z, z_goal, t)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((batch_size, 1), step * dt, device=device)
            v_t = self.forward(x, z, z_goal, t)
            x = x + dt * v_t
        
        return x


class MLP_FlowMatchingHead(nn.Module):
    def __init__(self, latent_dim=256, act_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim*2 + act_dim + 1, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, act_dim)
        )
    def forward(self, x, z, z_goal, t):
        """
        x: (B, D) - noisy action at t
        z: (B, D) - visual/state latent 
        z_goal: (B, D) - goal latent (image or language)
        t: (B, 1) - time 
        """
        return self.net(torch.cat([x, z, z_goal, t], dim=-1))
