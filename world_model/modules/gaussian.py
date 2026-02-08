from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from .mlp import MLP
from .distributions import (
    gaussian_diag_logprob,
    gaussian_diag_sample,
    gaussian_lowrank_diag_logprob,
    gaussian_lowrank_diag_sample,
    clamp_logstd,
)


@dataclass
class GaussianDiagParams:
    mu: torch.Tensor      # (B,D)
    logstd: torch.Tensor  # (B,D)


@dataclass
class GaussianLowRankDiagParams:
    mu: torch.Tensor      # (B,D)
    logstd: torch.Tensor  # (B,D)  (diagonal part)
    U: torch.Tensor       # (B,D,r) (low-rank factors)


class GaussianDiagHead(nn.Module):
    """
    Head that outputs diagonal Gaussian params.
    cov_type:
      - "fixed_diag": learn a global logstd parameter (vector) independent of input
      - "diag": predict logstd from input
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        cov_type: str = "diag",
        min_logstd: float = -8.0,
        max_logstd: float = 3.0,
        init_logstd: float = -1.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.cov_type = cov_type
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)

        # Mean network
        self.mean_net = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )

        if cov_type == "fixed_diag":
            self.logstd_param = nn.Parameter(torch.full((out_dim,), float(init_logstd)))
            self.logstd_net = None
        elif cov_type == "diag":
            self.logstd_param = None
            self.logstd_net = MLP(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation=activation,
            )
        else:
            raise ValueError(
                f"GaussianDiagHead only supports cov_type in {{'fixed_diag','diag'}}, got {cov_type}."
            )

    def forward(self, h: torch.Tensor) -> GaussianDiagParams:
        mu = self.mean_net(h)

        if self.cov_type == "fixed_diag":
            logstd = self.logstd_param.unsqueeze(0).expand_as(mu)
        else:
            logstd = self.logstd_net(h)

        logstd = clamp_logstd(logstd, self.min_logstd, self.max_logstd)
        return GaussianDiagParams(mu=mu, logstd=logstd)

    @staticmethod
    def sample(params: GaussianDiagParams, eps: torch.Tensor | None = None) -> torch.Tensor:
        return gaussian_diag_sample(params.mu, params.logstd, eps=eps)

    @staticmethod
    def log_prob(x: torch.Tensor, params: GaussianDiagParams) -> torch.Tensor:
        return gaussian_diag_logprob(x, params.mu, params.logstd)


class GaussianLowRankDiagHead(nn.Module):
    """
    Head that outputs Gaussian params with covariance:
        Sigma = diag(exp(2*logstd)) + U U^T
    cov_type is implicitly "lowrank".

    Args:
      rank: low-rank factor size r
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        min_logstd: float = -8.0,
        max_logstd: float = 3.0,
        init_logstd: float = -1.0,
        U_scale: float = 0.05,
    ):
        super().__init__()
        assert rank >= 1
        self.out_dim = out_dim
        self.rank = rank
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)
        self.U_scale = float(U_scale)

        # Mean network
        self.mean_net = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )

        # Diagonal part network (logstd)
        self.logstd_net = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )

        # Low-rank factor network: outputs flattened U of size D*r
        self.U_net = MLP(
            in_dim=in_dim,
            out_dim=out_dim * rank,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
        )

        # optional init bias for logstd
        with torch.no_grad():
            # If you want a slightly stable init: push logstd bias toward init_logstd
            # only if last layer is Linear
            pass

        self.init_logstd = float(init_logstd)

    def forward(self, h: torch.Tensor) -> GaussianLowRankDiagParams:
        B = h.shape[0]
        mu = self.mean_net(h)

        # logstd: clamp and optionally shift toward init_logstd
        logstd = self.logstd_net(h) + self.init_logstd
        logstd = clamp_logstd(logstd, self.min_logstd, self.max_logstd)

        U_flat = self.U_net(h)  # (B, D*r)
        U = U_flat.view(B, self.out_dim, self.rank)  # (B,D,r)

        # Scale U down so covariance starts close to diagonal (helps stability)
        U = self.U_scale * U

        return GaussianLowRankDiagParams(mu=mu, logstd=logstd, U=U)

    @staticmethod
    def sample(
        params: GaussianLowRankDiagParams,
        eps_diag: torch.Tensor | None = None,
        eps_rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return gaussian_lowrank_diag_sample(
            params.mu, params.logstd, params.U,
            eps_diag=eps_diag, eps_rank=eps_rank
        )

    @staticmethod
    def log_prob(
        x: torch.Tensor,
        params: GaussianLowRankDiagParams,
        jitter: float = 1e-6,
    ) -> torch.Tensor:
        return gaussian_lowrank_diag_logprob(x, params.mu, params.logstd, params.U, jitter=jitter)
