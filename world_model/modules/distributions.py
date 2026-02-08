# world_model/modules/distributions.py

from __future__ import annotations
import math
import torch


def clamp_logstd(logstd: torch.Tensor, min_logstd: float, max_logstd: float) -> torch.Tensor:
    return torch.clamp(logstd, min=min_logstd, max=max_logstd)


def gaussian_diag_sample(mu: torch.Tensor, logstd: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
    if eps is None:
        eps = torch.randn_like(mu)
    return mu + torch.exp(logstd) * eps


def gaussian_diag_logprob(x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    """
    Returns log N(x | mu, diag(exp(2*logstd))) for each batch element.
    Shapes: (B,D) -> (B,)
    """
    D = x.shape[-1]
    inv_std = torch.exp(-logstd)
    z = (x - mu) * inv_std
    return -0.5 * (torch.sum(z * z, dim=-1) + 2.0 * torch.sum(logstd, dim=-1) + D * math.log(2.0 * math.pi))


# --- Low-rank + diagonal Gaussian: Sigma = diag(exp(2*logstd)) + U U^T ---

def gaussian_lowrank_diag_sample(
    mu: torch.Tensor,
    logstd: torch.Tensor,
    U: torch.Tensor,
    eps_diag: torch.Tensor | None = None,
    eps_rank: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sample from N(mu, diag(exp(2*logstd)) + U U^T).

    Args:
      mu:      (B, D)
      logstd:  (B, D)
      U:       (B, D, r)
      eps_diag:(B, D)  standard normal for diagonal part (optional)
      eps_rank:(B, r)  standard normal for low-rank part (optional)

    Returns:
      x: (B, D)
    """
    B, D = mu.shape
    r = U.shape[-1]
    if eps_diag is None:
        eps_diag = torch.randn_like(mu)
    if eps_rank is None:
        eps_rank = torch.randn(B, r, device=mu.device, dtype=mu.dtype)

    std = torch.exp(logstd)  # (B,D)
    diag_part = std * eps_diag  # (B,D)
    lowrank_part = torch.einsum("bdr,br->bd", U, eps_rank)  # (B,D)
    return mu + diag_part + lowrank_part


def gaussian_lowrank_diag_logprob(
    x: torch.Tensor,
    mu: torch.Tensor,
    logstd: torch.Tensor,
    U: torch.Tensor,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Log prob for N(mu, Sigma) where Sigma = D + U U^T, D = diag(exp(2*logstd)).
    Uses Woodbury + matrix determinant lemma.

    Shapes:
      x, mu, logstd: (B, D)
      U:             (B, D, r)
    Returns:
      logp: (B,)
    """
    B, D = x.shape
    r = U.shape[-1]

    # Residual
    y = x - mu  # (B,D)

    # D^{-1} via diag precision
    inv_var = torch.exp(-2.0 * logstd)  # (B,D)
    # Compute D^{-1} y
    Dy = inv_var * y  # (B,D)

    # A = I_r + U^T D^{-1} U   (B,r,r)
    # First scale rows of U by inv_var (since D^{-1} is diagonal)
    U_scaled = inv_var.unsqueeze(-1) * U  # (B,D,r) = D^{-1} U
    A = torch.eye(r, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, r, r) \
        + torch.einsum("bdr,bdk->brk", U, U_scaled)  # U^T (D^{-1}U)

    # Cholesky for logdet(A) and solving
    # Add jitter for numerical stability
    A = A + jitter * torch.eye(r, device=x.device, dtype=x.dtype).unsqueeze(0)

    L = torch.linalg.cholesky(A)  # (B,r,r)

    # Compute Woodbury quadratic form:
    # y^T Sigma^{-1} y = y^T D^{-1} y - v^T A^{-1} v
    # where v = U^T D^{-1} y
    v = torch.einsum("bdr,bd->br", U, Dy)  # (B,r)

    # Solve A^{-1} v using Cholesky: A^{-1}v = solve(L^T, solve(L, v))
    # torch.cholesky_solve expects (B,r,1)
    v_col = v.unsqueeze(-1)  # (B,r,1)
    Ainv_v = torch.cholesky_solve(v_col, L).squeeze(-1)  # (B,r)

    y_Dinv_y = torch.sum(y * Dy, dim=-1)               # (B,)
    v_Ainv_v = torch.sum(v * Ainv_v, dim=-1)           # (B,)
    quad = y_Dinv_y - v_Ainv_v                         # (B,)

    # logdet(Sigma) = logdet(D) + logdet(A)
    # logdet(D) = sum log var = sum 2*logstd
    logdet_D = torch.sum(2.0 * logstd, dim=-1)         # (B,)
    # logdet(A) from Cholesky: 2 * sum log diag(L)
    logdet_A = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)  # (B,)
    logdet = logdet_D + logdet_A

    return -0.5 * (quad + logdet + D * math.log(2.0 * math.pi))
