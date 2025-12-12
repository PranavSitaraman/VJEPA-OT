from __future__ import annotations

import torch
import torch.nn.functional as F


def entropic_gromov_wasserstein(
    C_X: torch.Tensor,
    C_Y: torch.Tensor,
    p: torch.Tensor | None = None,
    q: torch.Tensor | None = None,
    epsilon: float = 0.05,
    n_iters: int = 100,
    loss_type: str = "square_loss",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
            LOG DOMAIN SINKHORN    
    **More stable although possibly more costly**
    Entropic Gromov-Wasserstein distance between two metric spaces.
    
    Aligns the geometry of space X (with cost matrix C_X) to space Y (with cost matrix C_Y)
    by finding a soft coupling that preserves pairwise distances.
    
    Args:
        C_X: (n, n) cost/distance matrix in source space X
        C_Y: (m, m) cost/distance matrix in target space Y
        p: (n,) source distribution (uniform if None)
        q: (m,) target distribution (uniform if None)
        epsilon: entropic regularization strength
        n_iters: number of Sinkhorn iterations
        loss_type: "square_loss" or "kl_loss"
    
    Returns:
        coupling: (n, m) soft assignment matrix (transport plan)
        gw_dist: scalar Gromov-Wasserstein distance
    """
    n, m = C_X.shape[0], C_Y.shape[0]
    device = C_X.device
    
    if p is None:
        p = torch.ones(n, device=device) / n
    if q is None:
        q = torch.ones(m, device=device) / m
    
    # Initialize coupling as outer product of marginals
    T = p.unsqueeze(1) @ q.unsqueeze(0)  # (n, m)
    
    for _ in range(n_iters):
        # Compute tensor product cost: L(C_X, C_Y, T)
        # For square loss: L_ij = sum_kl (C_X_ik - C_Y_jl)^2 * T_kl
        # Efficient computation via matrix operations
        if loss_type == "square_loss":
            # f1(C_X) = C_X^2 @ p
            f1 = (C_X ** 2) @ p  # (n,) 
            # f2(C_Y) = (C_Y^2).T @ q = C_Y^2 @ q (if symmetric)
            # We use (C_Y**2).T @ q to be safe for asymmetric costs
            f2 = (C_Y ** 2).T @ q  # (m,)
            # Cross term: -2 * C_X @ T @ C_Y^T
            cross = -2.0 * C_X @ T @ C_Y.T  # (n, m)
            
            # Total cost matrix
            cost = f1.unsqueeze(1) + f2.unsqueeze(0) + cross  # (n, m)
        else:
            raise NotImplementedError(f"loss_type={loss_type} not implemented")
        
        # Entropic regularization: log_K = -cost / epsilon
        log_K = -cost / epsilon
        
        # Sinkhorn iterations in log-space for stability
        # We run multiple iterations to solve the inner OT problem for the current cost
        log_p = torch.log(p + 1e-8)
        log_q = torch.log(q + 1e-8)
        log_u = torch.zeros(n, device=device)
        log_v = torch.zeros(m, device=device)
        
        for _ in range(20):  # Inner Sinkhorn loop
            # log_u = log_p - logsumexp(log_K + log_v)
            log_u = log_p - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            # log_v = log_q - logsumexp(log_K.T + log_u)
            log_v = log_q - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        
        # Transport plan in log-space
        log_T = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
        T = torch.exp(log_T)
    
    # Compute final GW distance
    if loss_type == "square_loss":
        f1 = (C_X ** 2) @ p
        f2 = (C_Y ** 2).T @ q
        cross = -2.0 * C_X @ T @ C_Y.T
        cost = f1.unsqueeze(1) + f2.unsqueeze(0) + cross
    
    gw_dist = (T * cost).sum()
    
    return T, gw_dist


def batch_ot_coupling(
    X: torch.Tensor,
    Y: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 50,
    cost_fn: str = "euclidean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute entropic OT coupling between batched point clouds with soft contrastive properties.
    
    This function implements batch-level entropic optimal transport that acts as soft contrastive
    learning. Unlike pointwise MSE which only minimizes ||X[i] - Y[i]||², entropic OT:
    
    1. **Entropy Maximization**: The entropic term ε·H(T) spreads out the coupling mass,
       preventing deterministic (sharp) assignments. This forces embeddings to be well-separated.
    
    2. **Implicit Repulsion**: To minimize transport cost while maximizing entropy, embeddings
       must be geometrically structured such that incorrect pairings (i≠j) have high cost.
       The Sinkhorn iterations naturally balance attraction (low diagonal cost) with repulsion
       (entropy-driven mass spreading).
    
    3. **Soft Contrastive Learning**: The optimal coupling T* has high diagonal values T[i,i]
       (correct pairs) and low off-diagonal values T[i,j] (incorrect pairs), achieving
       contrastive separation WITHOUT explicit negative sampling.
    
    Mathematical intuition:
        T* = argmin_T <C, T> - ε·H(T)  s.t. marginal constraints
        
        - Small ε: Nearly deterministic matching (T ≈ permutation matrix)
        - Large ε: Diffuse coupling (T ≈ uniform), weak contrastive signal
        - Optimal ε (0.1): Balanced soft contrastive structure
    
    Args:
        X: (B, d) context embeddings
        Y: (B, d) target embeddings (predicted or ground truth)
        epsilon: entropic regularization (default 0.1 balances structure and smoothness)
        n_iters: Sinkhorn iterations (default 50 for convergence)
        cost_fn: "euclidean" or "cosine" distance metric
    
    Returns:
        coupling: (B, B) soft assignment matrix T (rows/cols sum to 1/B)
        ot_loss: scalar identifiability loss = -log(diag(T)).mean()
                 Lower loss → stronger diagonal dominance → better contrastive structure
    """
    B = X.shape[0]
    device = X.device
    
    # Compute pairwise cost matrix
    if cost_fn == "euclidean":
        # ||X[i] - Y[j]||^2
        X_norm = (X ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        Y_norm = (Y ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        cost = X_norm + Y_norm.T - 2.0 * X @ Y.T  # (B, B)
    elif cost_fn == "cosine":
        # 1 - cosine_similarity
        X_norm = F.normalize(X, dim=1)
        Y_norm = F.normalize(Y, dim=1)
        cost = 1.0 - X_norm @ Y_norm.T  # (B, B)
    else:
        raise ValueError(f"Unknown cost_fn: {cost_fn}")
    
    # Uniform marginals
    p = torch.ones(B, device=device) / B
    q = torch.ones(B, device=device) / B
    
    # Sinkhorn in log-space for stability
    log_K = -cost / epsilon
    log_u = torch.zeros(B, device=device)
    log_v = torch.zeros(B, device=device)
    
    for _ in range(n_iters):
        log_u = torch.log(p + 1e-8) - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = torch.log(q + 1e-8) - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
    
    # Coupling matrix
    log_T = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    T = torch.exp(log_T)
    
    # Identifiability loss: maximize diagonal coupling (minimize negative log-likelihood)
    # We want T[i,i] to be large (X[i] couples to Y[i])
    diagonal_coupling = torch.diag(T)  # (B,)
    ot_loss = -torch.log(diagonal_coupling + 1e-8).mean()
    
    # ===== Soft Contrastive Properties Diagnostics =====
    # These metrics verify that entropic OT creates the desired contrastive structure:
    
    # 1. Diagonal dominance: T[i,i] should be >> 1/B (uniform baseline)
    #    Uniform coupling would have all entries = 1/B
    #    Good contrastive structure: diagonal >> off-diagonal
    with torch.no_grad():
        uniform_baseline = 1.0 / B
        diagonal_mean = diagonal_coupling.mean().item()
        diagonal_dominance_ratio = diagonal_mean / (uniform_baseline + 1e-8)
        
        # 2. Off-diagonal suppression: mean(T[i,j] for i≠j) should be << 1/B
        #    This shows the implicit repulsive force between incorrect pairs
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        off_diagonal_mean = T[mask].mean().item()
        off_diagonal_ratio = off_diagonal_mean / (uniform_baseline + 1e-8)
        
        # 3. Coupling entropy: H(T) = -sum(T * log(T))
        #    Lower entropy → more concentrated (deterministic) coupling
        #    Balanced with transport cost via epsilon
        entropy = -(T * torch.log(T + 1e-8)).sum().item()
        
        # 4. Separation metric: ratio of diagonal to off-diagonal
        #    High values indicate strong contrastive separation
        separation_ratio = diagonal_mean / (off_diagonal_mean + 1e-8)
    
    # Store diagnostics in the coupling tensor as attributes for logging
    # (these are detached and won't affect gradients)
    if not hasattr(T, '_ot_diagnostics'):
        T._ot_diagnostics = {}
    T._ot_diagnostics = {
        'diagonal_mean': diagonal_mean,
        'diagonal_dominance': diagonal_dominance_ratio,  # Should be >> 1.0
        'off_diagonal_mean': off_diagonal_mean,
        'off_diagonal_ratio': off_diagonal_ratio,  # Should be << 1.0
        'coupling_entropy': entropy,
        'separation_ratio': separation_ratio,  # Should be >> 1.0
        'epsilon': epsilon,
    }
    
    return T, ot_loss


def bilevel_ot_contrastive_loss(
    X: torch.Tensor,
    Y: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 50,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.5,
    lambda_unif: float = 0.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Bilevel OT for contrastive learning following IOT-CL (ICML 2023) formulation.
    
    This implements the mathematically rigorous formulation from:
    - "Understanding and Generalizing Contrastive Learning from IOT Perspective" (ICML 2023)
    - "Your contrastive learning problem is secretly a distribution alignment problem" (NeurIPS 2024)
    
    BILEVEL OPTIMIZATION:
        Outer: min_θ KL(P_target || P^θ) + λ_var·L_var + λ_cov·L_cov + λ_unif·L_unif
        Inner: P^θ = argmin_{P ∈ U(a,b)} ⟨C^θ, P⟩ - ε·H(P)
    
    UNIFORMITY LOSS (from IOT-CL):
        L_unif = log Σ_{i,j} exp(-2·||z_i - z_j'||²)
        Pushes negative pairs to be uniformly spread apart.
    
    ANTI-COLLAPSE GUARANTEE:
        Under collapse (X[i] = Y[j] = c for all i,j):
        - P^θ → uniform (1/B everywhere)
        - KL(I || uniform) = B·log(B) >> 0 (HIGH LOSS)
        - ∇L ≠ 0 due to variance/uniformity term (NON-ZERO GRADIENT)
        ∴ Collapse is NOT a minimum!
    
    CONNECTION TO InfoNCE:
        When using U(a) constraints (row-only) and single Sinkhorn iteration:
        - This reduces to InfoNCE loss
        - Temperature τ ≡ entropic regularization ε
        - Our full Sinkhorn (U(a,b)) is STRONGER than InfoNCE
    
    Args:
        X: (B, d) context embeddings
        Y: (B, d) target embeddings
        epsilon: entropic regularization (0.1 recommended, equivalent to temperature)
        n_iters: Sinkhorn iterations (50 for doubly stochastic convergence)
        lambda_var: variance regularization weight (1.0 recommended, set 0 to disable)
        lambda_cov: covariance regularization weight (0.5 recommended, set 0 to disable)
        lambda_unif: uniformity loss weight (0.0 default = disabled; try 1.0 to enable)
        gamma: target std per dimension (1.0 recommended)
    
    Returns:
        total_loss: combined bilevel OT loss
        diagnostics: dict with component losses and coupling metrics
    """
    B, d = X.shape
    device = X.device
    
    # ===== INNER: Solve Entropic OT with U(a,b) constraints =====
    # Cost matrix
    X_norm = (X ** 2).sum(dim=1, keepdim=True)
    Y_norm = (Y ** 2).sum(dim=1, keepdim=True)
    C = (X_norm + Y_norm.T - 2.0 * X @ Y.T).clamp(min=0.0)
    
    # Sinkhorn algorithm (doubly stochastic)
    p = torch.ones(B, device=device) / B
    q = torch.ones(B, device=device) / B
    log_K = -C / epsilon
    log_u = torch.zeros(B, device=device)
    log_v = torch.zeros(B, device=device)
    
    for _ in range(n_iters):
        log_u = torch.log(p + 1e-8) - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = torch.log(q + 1e-8) - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
    
    log_T = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    T = torch.exp(log_T)
    
    # ===== OUTER: Supervise Coupling with Target Plan =====
    # KL(P_target || P_learned) where P_target = I (identity)
    # For identity target, this simplifies to: -Σᵢ log(P_learned[i,i])
    # This is EXACTLY the InfoNCE objective when using U(a) constraints!
    diagonal = torch.diag(T)
    kl_loss = -(torch.log(diagonal + 1e-8)).sum()
    
    # ===== Variance Regularization (Explicit Anti-Collapse) =====
    X_std = X.std(dim=0)
    Y_std = Y.std(dim=0)
    var_loss = F.relu(gamma - X_std).pow(2).mean() + F.relu(gamma - Y_std).pow(2).mean()
    
    # ===== Covariance Regularization (Decorrelation) =====
    X_centered = X - X.mean(dim=0)
    Y_centered = Y - Y.mean(dim=0)
    cov_X = (X_centered.T @ X_centered) / B
    cov_Y = (Y_centered.T @ Y_centered) / B
    
    if d > 1:
        mask = ~torch.eye(d, dtype=torch.bool, device=device)
        cov_loss = (cov_X[mask] ** 2).mean() + (cov_Y[mask] ** 2).mean()
    else:
        cov_loss = torch.zeros(1, device=device)
    
    # ===== Uniformity Loss (IOT-CL formulation) =====
    # KL(Q_bar || P) where Q_bar has:
    #   - Diagonal: same as P (positive pairs keep their probabilities)
    #   - Off-diagonal: mean of all off-diagonal values (negative pairs uniform)
    # This forces negative matching probabilities to be uniform and less volatile.
    if lambda_unif > 0.0:
        off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        # Q_bar: target distribution
        Q_bar = T.clone()
        off_diag_mean = T[off_diag_mask].mean()
        Q_bar[off_diag_mask] = off_diag_mean
        # KL(Q_bar || P) = Σ Q_bar * log(Q_bar / P)
        # Only compute where Q_bar > 0 to avoid log(0)
        unif_loss = (Q_bar * (torch.log(Q_bar + 1e-8) - torch.log(T + 1e-8))).sum()
    else:
        unif_loss = torch.zeros(1, device=device)
    
    # ===== Combined Bilevel Loss =====
    total_loss = kl_loss + lambda_var * var_loss + lambda_cov * cov_loss + lambda_unif * unif_loss
    
    # ===== Diagnostics =====
    with torch.no_grad():
        uniform_baseline = 1.0 / B
        diagonal_mean = diagonal.mean().item()
        off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        off_diagonal_mean = T[off_diag_mask].mean().item() if B > 1 else 0.0
        
        diagnostics = {
            'kl_loss': kl_loss.item(),
            'var_loss': var_loss.item(),
            'cov_loss': cov_loss.item(),
            'unif_loss': unif_loss.item() if isinstance(unif_loss, torch.Tensor) else 0.0,
            'diagonal_mean': diagonal_mean,
            'diagonal_dominance': diagonal_mean / (uniform_baseline + 1e-8),
            'off_diagonal_mean': off_diagonal_mean,
            'separation_ratio': diagonal_mean / (off_diagonal_mean + 1e-8),
            'min_std_X': X_std.min().item(),
            'min_std_Y': Y_std.min().item(),
            'coupling_entropy': -(T * torch.log(T + 1e-8)).sum().item(),
            'collapse_risk': kl_loss.item() / (B * torch.log(torch.tensor(B, dtype=torch.float32))),
        }
    
    return total_loss, diagnostics


def representation_alignment_loss(
    z_online: torch.Tensor,
    z_pretrained: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 50,
) -> torch.Tensor:
    """Align fine-tuned representations to pretrained ones via Gromov-Wasserstein.
    
    Prevents catastrophic forgetting by preserving the geometry of pretrained embeddings.
    Uses cosine distance cost matrices and entropic GW.
    
    Args:
        z_online: (B, d) embeddings from fine-tuned encoder (Student)
        z_pretrained: (B, d) embeddings from frozen pretrained encoder (Teacher)
        epsilon: entropic regularization
        n_iters: GW iterations
    
    Returns:
        gw_dist: scalar Gromov-Wasserstein distance (lower = better alignment)
    """
    # 0) Normalize
    # eps inside normalize prevents div by zero
    z_teacher_n = F.normalize(z_pretrained, p=2, dim=1).detach()
    z_student_n = F.normalize(z_online, p=2, dim=1)

    # 1) Cost Matrices (Cosine Distance)
    # Clamp is vital for numerical stability of Sinkhorn
    C_teacher = (1.0 - (z_teacher_n @ z_teacher_n.t())).clamp(min=0.0)
    C_student = (1.0 - (z_student_n @ z_student_n.t())).clamp(min=0.0)
    
    # Symmetrize
    C_teacher = 0.5 * (C_teacher + C_teacher.t())
    C_student = 0.5 * (C_student + C_student.t())

    # 2) Uniform Distributions
    N = z_online.size(0)
    device = z_online.device
    p = torch.full((N,), 1.0 / N, device=device, dtype=z_online.dtype)
    q = torch.full((N,), 1.0 / N, device=device, dtype=z_online.dtype)

    # 3) Entropic GW
    # We use our internal entropic_gromov_wasserstein implementation
    _, gw_dist = entropic_gromov_wasserstein(
        C_X=C_teacher,
        C_Y=C_student,
        p=p,
        q=q,
        epsilon=epsilon,
        n_iters=n_iters,
        loss_type="square_loss"
    )

    return gw_dist
