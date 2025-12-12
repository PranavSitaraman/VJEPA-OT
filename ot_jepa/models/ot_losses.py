from __future__ import annotations

from typing import Tuple

import torch


def sliced_w2(metric, X: torch.Tensor, Y: torch.Tensor, num_projections: int = 128) -> torch.Tensor:
    """Sliced Wasserstein-2 distance using learned projections from metric.

    Args:
        metric: MetricNet-like module taking anchors (B,d) -> factors (B,r,d).
        X, Y: (B, N, d) sets with same N (time steps or samples).
        num_projections: number of random 1-D projections per batch.
    Returns:
        (B,) tensor of sliced W2 distances.
    """

    B, N, d = X.shape
    device = X.device
    anchors = 0.5 * (X.mean(dim=1) + Y.mean(dim=1))
    factors = metric(anchors)  # (B, r, d)
    if factors.shape[1] == 0:
        raise ValueError("Metric rank must be > 0")

    # Vectorized sampling of all random projections at once to avoid Python loops.
    # factors: (B, r, d) -> F: (B, d, r)
    F = factors.transpose(1, 2)  # (B, d, r)
    r = factors.shape[1]
    P = int(num_projections)
    # Random directions in the metric factor space
    rand = torch.randn(B, r, P, device=device)
    rand = rand / (rand.norm(dim=1, keepdim=True) + 1e-8)  # normalize over r
    # Project to latent space directions: (B,d,P)
    projections = torch.einsum("bdr,brp->bdp", F, rand)
    # Do NOT normalize projections here! We want to preserve the metric scaling.
    # projections = projections / (projections.norm(dim=1, keepdim=True) + 1e-8)

    # Project X,Y onto all directions at once: x_proj,y_proj: (B,N,P)
    x_proj = torch.einsum("bnd,bdp->bnp", X, projections)
    y_proj = torch.einsum("bnd,bdp->bnp", Y, projections)

    # Sort along the sample dimension for each projection
    x_sorted, _ = torch.sort(x_proj, dim=1)
    y_sorted, _ = torch.sort(y_proj, dim=1)

    # Average squared distance over samples and projections
    total = ((x_sorted - y_sorted) ** 2).mean(dim=1).mean(dim=1)  # (B,)
    return total


def sinkhorn_w2(metric, X: torch.Tensor, Y: torch.Tensor, epsilon: float = 0.1, n_iters: int = 50) -> torch.Tensor:
    """Entropic OT (Sinkhorn) approximation of W2 between equal-sized sets.

    This is a batched implementation assuming X and Y are (B,N,d) with same N.
    The ground metric is induced by a learned low-rank Mahalanobis metric G via
    c_ij = ||A(z_anchor) (x_i - y_j)||_2^2.

    Args:
        metric: MetricNet-like, called on anchors (B,d) to produce (B,r,d) factors.
        X, Y: (B,N,d) sets.
        epsilon: entropic regularization strength.
        n_iters: number of Sinkhorn iterations.
    Returns:
        (B,) tensor with approximate W2 values.
    """
    B, N, d = X.shape
    device = X.device
    anchors = 0.5 * (X.mean(dim=1) + Y.mean(dim=1))  # (B,d)
    factors = metric(anchors)  # (B,r,d)
    # Compute pairwise costs C[b,i,j]
    # Efficiently compute A*(x_i - y_j) by expanding dims.
    # Xb: (B,N,1,d), Yb: (B,1,N,d), diff: (B,N,N,d)
    Xb = X.unsqueeze(2)
    Yb = Y.unsqueeze(1)
    diff = Xb - Yb  # (B,N,N,d)
    # Apply factors: (B,r,d) * (B,N,N,d) -> (B,N,N,r,d) then sum over d
    # We use broadcasting by reshaping
    Ff = factors.unsqueeze(1).unsqueeze(1)  # (B,1,1,r,d)
    # Apply factors: (B,r,d) * (B,N,N,d) -> (B,N,N,r)
    # We want to project the difference vector onto the factor directions
    projected = (Ff * diff.unsqueeze(3)).sum(dim=-1)  # (B,N,N,r)
    C = (projected ** 2).sum(dim=-1)  # (B,N,N)

    # Sinkhorn iterations in log-space for stability
    log_K = -C / max(epsilon, 1e-6)
    log_u = torch.zeros(B, N, device=device)
    log_v = torch.zeros(B, N, device=device)
    log_a = torch.zeros(B, N, device=device)  # uniform marginals in log-space
    log_b = torch.zeros(B, N, device=device)

    for _ in range(n_iters):
        # u <- a / (K v)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
        # v <- b / (K^T u)
        log_v = log_b - torch.logsumexp(log_K.transpose(1, 2) + log_u.unsqueeze(1), dim=2)

    # Transport plan in log-space
    log_P = log_K + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    P = torch.exp(log_P)
    cost = (P * C).sum(dim=(1, 2))
    return cost


def string_prior(metric, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Metric-aware length and curvature along a latent sequence Z (B,T,d).

    We flatten time to call MetricNet.cost on (N,d) pairs while using a per-batch
    anchor replicated across time, matching the design G(z)=A(z)^T A(z).
    """
    B, T, d = Z.shape
    if T < 3:
        # Fall back to Euclidean for very short sequences
        d1 = Z[:, 1:, :] - Z[:, :-1, :]
        length = (d1**2).sum(-1).mean()
        curvature = torch.zeros_like(length)
        return length, curvature

    # Anchors: one per batch, replicate across time steps as needed
    anchors = Z[:, 1:-1, :].mean(dim=1)  # (B,d)

    # Length term between consecutive steps (T-1 pairs)
    ref_len = anchors.unsqueeze(1).expand(-1, T - 1, -1).reshape(-1, d)  # (B*(T-1),d)
    z_next = Z[:, 1:, :].reshape(-1, d)
    z_prev = Z[:, :-1, :].reshape(-1, d)
    length = metric.cost(ref_len, z_next, z_prev).mean()

    # Curvature term comparing z_{t+1} to 2 z_t - z_{t-1} (T-2 pairs)
    ref_cur = anchors.unsqueeze(1).expand(-1, T - 2, -1).reshape(-1, d)
    z_p1 = Z[:, 2:, :].reshape(-1, d)
    z_mid = (2 * Z[:, 1:-1, :] - Z[:, :-2, :]).reshape(-1, d)
    curvature = metric.cost(ref_cur, z_p1, z_mid).mean()

    return length, curvature


def cross_modal_ot(metric, z_a: torch.Tensor, z_b: torch.Tensor, num_projections: int = 64) -> torch.Tensor:
    z = torch.stack([z_a, z_b], dim=1).mean(dim=1)
    return sliced_w2(metric, z_a.unsqueeze(1), z_b.unsqueeze(1), num_projections=num_projections).mean()


def goal_w2(metric, z_pred: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, num_samples: int = 4) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(z_pred.shape[0], num_samples, z_pred.shape[-1], device=z_pred.device)
    samples = mean.unsqueeze(1) + std.unsqueeze(1) * eps
    preds = z_pred.unsqueeze(1).repeat(1, num_samples, 1)
    return sliced_w2(metric, preds, samples, num_projections=64).mean()
