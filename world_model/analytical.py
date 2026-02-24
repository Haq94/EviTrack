# world_model/analytical.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from .base import WorldModelConfig
from .markov import MarkovWorldModel
from .modules.gaussian import GaussianDiagParams


ArrayFn = Callable[[torch.Tensor, Optional[Dict]], torch.Tensor]


class AnalyticalGaussianDiagHead(nn.Module):
    """
    Adapts analytic mean/cov functions into the same output type as GaussianDiagHead.
    """
    def __init__(
        self,
        mean_fn: Callable[[torch.Tensor, Optional[Dict]], torch.Tensor],
        cov_fn: Callable[[torch.Tensor, Optional[Dict]], torch.Tensor],
        *,
        min_logstd: float = -8.0,
        max_logstd: float = 3.0,
    ):
        super().__init__()
        self.mean_fn = mean_fn
        self.cov_fn = cov_fn
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)

    def forward(self, ctx: torch.Tensor, extras: Optional[Dict] = None) -> GaussianDiagParams:
        mu = self.mean_fn(ctx, extras)

        cov = self.cov_fn(ctx, extras)
        # cov can be:
        #  - (B, d, d)
        #  - (d, d)
        #  - (B, d)   [diag]
        #  - (d,)     [diag]
        if cov.ndim == 3:
            diag = torch.diagonal(cov, dim1=-2, dim2=-1)
        elif cov.ndim == 2:
            diag = torch.diagonal(cov, dim1=-2, dim2=-1).unsqueeze(0).expand(mu.shape[0], -1)
        elif cov.ndim == 1:
            diag = cov.unsqueeze(0).expand(mu.shape[0], -1)
        else:
            raise ValueError(f"Unsupported cov shape: {tuple(cov.shape)}")

        # std = sqrt(diag), logstd = log(std)
        logstd = 0.5 * torch.log(torch.clamp(diag, min=1e-12))
        logstd = torch.clamp(logstd, self.min_logstd, self.max_logstd)

        return GaussianDiagParams(mu=mu, logstd=logstd)


class AnalyticalWorldModel(MarkovWorldModel):
    """
    Markov analytical world model:
      z1 ~ N(mu0, cov0)
      zt|z_{t-1} ~ N(mu(z_{t-1}), cov(z_{t-1}))
      xt|z_state_curr,x_state_prev ~ N(mu(...), cov(...))
    """

    def __init__(
        self,
        cfg: WorldModelConfig,
        *,
        prior_mu0: torch.Tensor,
        prior_cov0: torch.Tensor,
        trans_mean: Callable[[torch.Tensor, Optional[Dict]], torch.Tensor],
        trans_cov: Callable[[torch.Tensor, Optional[Dict]], torch.Tensor],
        emit_mean: Callable[[torch.Tensor, Optional[Dict]], torch.Tensor],
        emit_cov: Callable[[torch.Tensor, Optional[Dict]], torch.Tensor],
    ):
        super().__init__(cfg)

        # ---- prior: overwrite learned parameters with fixed analytic values ----
        # base.py stores prior_mu/prior_logstd as Parameters【turn27file10†base.py†L76-L79】
        with torch.no_grad():
            self.prior_mu.copy_(prior_mu0.reshape(-1))
            diag0 = torch.diagonal(prior_cov0, dim1=-2, dim2=-1).reshape(-1)
            self.prior_logstd.copy_(0.5 * torch.log(torch.clamp(diag0, min=1e-12)))
        self.prior_mu.requires_grad_(False)
        self.prior_logstd.requires_grad_(False)

        # ---- transition + emission: replace neural heads with analytic heads ----
        self._transition_head = AnalyticalGaussianDiagHead(
            trans_mean, trans_cov,
            min_logstd=self.cfg.transition.min_logstd,
            max_logstd=self.cfg.transition.max_logstd,
        )
        self._emission_head = AnalyticalGaussianDiagHead(
            emit_mean, emit_cov,
            min_logstd=self.cfg.emission.min_logstd,
            max_logstd=self.cfg.emission.max_logstd,
        )