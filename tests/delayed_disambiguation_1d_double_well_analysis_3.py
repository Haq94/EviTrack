# tests/delayed_disambiguation_1d_double_well_analysis_3.py
#
# Analysis script for 1D double-well delayed-disambiguation synthetic.
# Generates one trajectory, runs a bootstrap PF, and saves:
#  - observation channels
#  - predictive heatmap p(z_t | x_{1:t-1})
#  - filtering heatmap p(z_t | x_{1:t})
#  - branch-mass curves
#  - ESS vs time
#  - collapse-time sensitivity to resampling frequency
#  - NEW: evidence accumulation plots:
#       * per-step log p(x_t | x_{1:t-1})
#       * cumulative log evidence sum_t log p(x_t | x_{1:t-1})
#       * optional: log likelihood ratio from branch masses
#
# Outputs are saved to: results/_synthetic_1d_double_well_analysis/

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from data.synthetic_generator import (
    GaussianPrior, GaussianTransition, GaussianEmission, generate_sequences
)


# ----------------------------
# 1) Stationary nonlinear Gaussian-conditionals (transition + emission)
# ----------------------------

def trans_mean(z_prev: np.ndarray, extras: Dict) -> np.ndarray:
    """
    Drift for a classic double-well-ish dynamics using a tanh nonlinearity.

    z_prev: (B,1) or (N,1)
    """
    a = float(extras.get("a", 0.15))
    b = float(extras.get("b", 0.10))
    return (1.0 + a) * z_prev - b * (z_prev ** 3)


def trans_cov(z_prev: np.ndarray, extras: Dict) -> np.ndarray:
    dz = z_prev.shape[-1]
    q = float(extras.get("process_noise", 0.20))
    return (q ** 2) * np.eye(dz)


def emit_mean(z: np.ndarray, extras: Dict) -> np.ndarray:
    """
    2D emission (dx=2), stationary and nonlinear.

    x1 = z^2  (many-to-one, kills sign)
    x2 = "gated sign" channel: informative only in a band |z| < eps, else ~0
         This is the key to delayed disambiguation:
           - when z is outside the band, x2 does not reveal sign
           - when z passes through/near the band, x2 reveals sign
    """
    z1 = z[:, :1]  # (N,1) or (B,1)

    eps = float(extras.get("eps", 0.6))
    # sign-informative near zero; otherwise suppressed
    x2 = np.where(np.abs(z1) < eps, z1, 0.0)

    x1 = z1 ** 2
    return np.concatenate([x1, x2], axis=1)  # (N,2)


def emit_cov(z: np.ndarray, extras: Dict) -> np.ndarray:
    r = float(extras.get("emit_noise", 0.25))
    # infer dx robustly
    dx = int(emit_mean(z, extras).shape[-1])
    return (r ** 2) * np.eye(dx)


# ----------------------------
# 2) Gaussian logpdf utility
# ----------------------------

def log_gaussian_full(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    x, mu: (N, d)
    cov: (d,d) or (N,d,d)
    returns: (N,)
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    d = x.shape[-1]

    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 2:
        cov = np.broadcast_to(cov, (x.shape[0], d, d))

    out = np.empty((x.shape[0],), dtype=float)
    const = -0.5 * d * np.log(2.0 * np.pi)
    for i in range(x.shape[0]):
        L = np.linalg.cholesky(cov[i])
        y = np.linalg.solve(L, x[i] - mu[i])
        maha = float(y @ y)
        logdet = float(2.0 * np.sum(np.log(np.diag(L))))
        out[i] = const - 0.5 * (logdet + maha)
    return out


def systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = len(w)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(w)
    return np.searchsorted(cumsum, positions)


# ----------------------------
# 3) Bootstrap PF with evidence logging
# ----------------------------

@dataclass
class PFOutputs:
    z_filt: List[np.ndarray]          # list length T: (N,dz) particles for p(z_t|x_{1:t}) (pre-resample)
    w_filt: List[np.ndarray]          # list length T: (N,) weights for p(z_t|x_{1:t}) (pre-resample)
    z_pred: List[np.ndarray]          # list length T-1: (N,dz) particles for p(z_{t+1}|x_{1:t}) (predictive)
    ess: np.ndarray                   # (T,)
    log_evidence: np.ndarray          # (T,) approx log p(x_t | x_{1:t-1})
    cum_log_evidence: np.ndarray      # (T,) cumulative sum
    log_lr_pred: np.ndarray           # (T,) log((1-m_left_pred)/m_left_pred) with eps
    log_lr_filt: np.ndarray           # (T,) log((1-m_left_filt)/m_left_filt) with eps


def bootstrap_pf(
    x_seq: np.ndarray,
    prior: GaussianPrior,
    transition: GaussianTransition,
    emission: GaussianEmission,
    *,
    N: int = 20000,
    seed: int = 0,
    extras: Optional[Dict] = None,
    resample_every: int = 1,
) -> PFOutputs:
    """
    Filtering:
      z_t^i ~ q(z_t|z_{t-1}^i) = p(z_t|z_{t-1}^i)  (bootstrap)
      w_t^i ∝ w_{t-1}^i p(x_t|z_t^i)

    We also compute:
      log p(x_t | x_{1:t-1}) = log sum_i w_{t-1}^i p(x_t|z_t^i)
    where w_{t-1} are the normalized weights BEFORE seeing x_t (here, since we resample, they’re uniform
    right after resampling; but we compute this generally using current w).
    """
    rng = np.random.default_rng(seed)
    extras = dict(extras or {})

    T, dx = x_seq.shape
    dz = int(np.asarray(prior.mu0).shape[0])

    # init particles from prior
    z = rng.multivariate_normal(mean=np.asarray(prior.mu0), cov=np.asarray(prior.cov0), size=N)  # (N,dz)
    w = np.ones(N, dtype=float) / N

    z_filt: List[np.ndarray] = []
    w_filt: List[np.ndarray] = []
    z_pred: List[np.ndarray] = []
    ess = np.zeros((T,), dtype=float)

    log_evidence = np.zeros((T,), dtype=float)
    log_lr_pred = np.zeros((T,), dtype=float)
    log_lr_filt = np.zeros((T,), dtype=float)

    # We'll also maintain "predictive particles" for p(z_t|x_{<t}) to compute predictive branch mass.
    # At t=1, predictive is just prior samples.
    z_pred_curr = z.copy()  # p(z_1|x_{<1}) = p(z_1)

    eps_lr = 1e-12

    for t in range(T):
        # --- Predictive branch mass at time t: p(z_t < 0 | x_{<t}) via z_pred_curr ---
        mleft_pred = float(np.mean(z_pred_curr[:, 0] < 0.0))
        log_lr_pred[t] = np.log((1.0 - mleft_pred + eps_lr) / (mleft_pred + eps_lr))

        # --- Weight by likelihood p(x_t | z_t) ---
        mu_x = np.asarray(emission.mean_fn(z, extras), dtype=float)  # (N,dx)
        cov_x = np.asarray(emission.cov_fn(z, extras), dtype=float)  # (dx,dx) or (N,dx,dx)
        xt_rep = np.repeat(x_seq[t:t + 1], N, axis=0)                # (N,dx)

        loglik = log_gaussian_full(xt_rep, mu_x, cov_x)              # (N,)

        # log evidence increment: log sum_i w_i * exp(loglik_i)
        maxlog = float(np.max(loglik))
        log_evidence[t] = maxlog + np.log(np.sum(w * np.exp(loglik - maxlog)) + 1e-12)

        # weight update (numerically stable)
        w = w * np.exp(loglik - maxlog)
        w = w / (w.sum() + 1e-12)

        ess[t] = 1.0 / np.sum(w ** 2)

        # store filtering cloud BEFORE resample
        z_filt.append(z.copy())
        w_filt.append(w.copy())

        # filtering branch mass (using weights)
        mleft_filt = float(np.sum(w * (z[:, 0] < 0.0)))
        log_lr_filt[t] = np.log((1.0 - mleft_filt + eps_lr) / (mleft_filt + eps_lr))

        # --- resample ---
        if resample_every > 0 and ((t + 1) % resample_every == 0):
            idx = systematic_resample(w, rng)
            z = z[idx]
            w = np.ones(N, dtype=float) / N

        # --- propagate to next time step: z_{t+1} ~ p(z_{t+1}|z_t) ---
        if t < T - 1:
            mu_z = np.asarray(transition.mean_fn(z, extras), dtype=float)    # (N,dz)
            cov_z = np.asarray(transition.cov_fn(z, extras), dtype=float)    # (dz,dz) or (N,dz,dz)
            if cov_z.ndim == 2:
                cov_z = np.broadcast_to(cov_z, (N, dz, dz))

            z_next = np.empty((N, dz), dtype=float)
            for i in range(N):
                z_next[i] = rng.multivariate_normal(mean=mu_z[i], cov=cov_z[i])
            z_pred.append(z_next.copy())

            # next predictive particles for time t+1 given x_{<=t} are z_next
            z_pred_curr = z_next.copy()
            z = z_next

    cum_log_evidence = np.cumsum(log_evidence)

    return PFOutputs(
        z_filt=z_filt,
        w_filt=w_filt,
        z_pred=z_pred,
        ess=ess,
        log_evidence=log_evidence,
        cum_log_evidence=cum_log_evidence,
        log_lr_pred=log_lr_pred,
        log_lr_filt=log_lr_filt,
    )


# ----------------------------
# 4) Heatmaps + branch mass curves
# ----------------------------

def predictive_heatmap(z_pred: List[np.ndarray], z_true: np.ndarray, *, zmin=-2.5, zmax=2.5, nbins=200) -> Tuple[np.ndarray, np.ndarray]:
    """
    z_pred: list length T-1 of (N,1) for p(z_{t+1} | x_{<=t})
    We return heatmap for t=1..T-1 (index 1..T-1 on y-axis).
    """
    Tm1 = len(z_pred)
    edges = np.linspace(zmin, zmax, nbins + 1)
    H = np.zeros((Tm1, nbins), dtype=float)
    for t in range(Tm1):
        z = z_pred[t][:, 0]
        hist, _ = np.histogram(z, bins=edges, density=True)
        H[t] = hist
    return H, edges


def filtering_heatmap(z_filt: List[np.ndarray], w_filt: List[np.ndarray], *, zmin=-2.5, zmax=2.5, nbins=200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted filtering heatmap p(z_t | x_{1:t}) using particle weights.
    """
    T = len(z_filt)
    edges = np.linspace(zmin, zmax, nbins + 1)
    H = np.zeros((T, nbins), dtype=float)
    for t in range(T):
        z = z_filt[t][:, 0]
        w = w_filt[t]
        hist, _ = np.histogram(z, bins=edges, weights=w, density=False)
        # convert to density by dividing by bin width
        bw = (zmax - zmin) / nbins
        H[t] = hist / (bw + 1e-12)
    return H, edges


def branch_mass_from_pf(pf: PFOutputs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      mleft_pred[t] = P(z_t < 0 | x_{<t})
      mleft_filt[t] = P(z_t < 0 | x_{<=t})
    with t indexed 1..T, but arrays are length T with 0-based indices.
    """
    T = len(pf.z_filt)

    # predictive left mass:
    # at t=0 (z1|x_<1) isn't stored directly in outputs; approximate using z_filt[0] before weighting is not available.
    # We'll approximate predictive mass at t=0 by using z_filt[0] particle signs with uniform weights AFTER init.
    # Better: use log_lr_pred already computed in PF.
    mleft_pred = np.zeros((T,), dtype=float)
    mleft_filt = np.zeros((T,), dtype=float)

    for t in range(T):
        # from log_lr: m = 1/(1+exp(log_lr)) where log_lr = log((1-m)/m)
        # -> exp(log_lr) = (1-m)/m -> m = 1/(1+exp(log_lr))
        mleft_pred[t] = 1.0 / (1.0 + np.exp(pf.log_lr_pred[t]))
        mleft_filt[t] = 1.0 / (1.0 + np.exp(pf.log_lr_filt[t]))

    return mleft_pred, mleft_filt


def find_collapse_time(mleft: np.ndarray, thresh: float = 0.05) -> int:
    """
    Return first time index (1-based in plots) where left mass < thresh.
    If never collapses, return T.
    """
    idx = np.where(mleft < thresh)[0]
    if len(idx) == 0:
        return int(len(mleft))
    return int(idx[0] + 1)  # convert to 1-based "t"


# ----------------------------
# 5) Main experiment + saving
# ----------------------------

def main():
    # Output directory
    out_dir = Path("results") / "_synthetic_1d_double_well_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic config
    T = 90
    dz = 1
    seed = 0

    # Model knobs
    extras = {
        "a": 0.15,             # affects drift strength
        "b": 0.10,             # cubic term strength
        "process_noise": 0.20,
        "emit_noise": 0.25,
        "eps": 0.60,           # sign-informative band for x2
    }

    # Build stationary generator objects
    prior = GaussianPrior(mu0=np.zeros(dz), cov0=np.eye(dz))
    transition = GaussianTransition(mean_fn=trans_mean, cov_fn=trans_cov)
    emission = GaussianEmission(mean_fn=emit_mean, cov_fn=emit_cov)

    # Generate ONE sequence (B=1)
    batch = generate_sequences(
        T=T, B=1,
        prior=prior, transition=transition, emission=emission,
        seed=seed,
        return_logp=False,
        extras=extras,
    )
    z_true = batch.z[0, :, 0]   # (T,)
    x = batch.x[0, :, :]        # (T,dx)

    # Save observation sanity plot
    fig = plt.figure(figsize=(12, 4))
    plt.plot(np.arange(1, T + 1), x[:, 0], label="x1")
    plt.plot(np.arange(1, T + 1), x[:, 1], label="x2")
    plt.xlabel("t")
    plt.title("Observation channels (sanity)")
    plt.grid(True)
    plt.legend()
    fig.savefig(out_dir / "observations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Run PF
    pf = bootstrap_pf(
        x_seq=x,
        prior=prior,
        transition=transition,
        emission=emission,
        N=int(extras.get("N", 30000)),
        seed=123,
        extras=extras,
        resample_every=1,
    )

    # Heatmaps
    zmin, zmax, nbins = -2.5, 2.5, 220
    H_pred, edges = predictive_heatmap(pf.z_pred, z_true, zmin=zmin, zmax=zmax, nbins=nbins)
    H_filt, edges_f = filtering_heatmap(pf.z_filt, pf.w_filt, zmin=zmin, zmax=zmax, nbins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Predictive heatmap plot (t=1..T-1 on y axis)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(
        H_pred,
        aspect="auto",
        origin="lower",
        extent=[centers[0], centers[-1], 2, T],  # predictive z_{t} stored as z_pred[t-1]
        interpolation="nearest",
    )
    plt.plot(z_true, np.arange(1, T + 1), lw=2)
    plt.xlabel("z")
    plt.ylabel("t")
    plt.title("Predictive heatmap: p(z_t | x_{1:t-1}) + true z_t")
    cbar = plt.colorbar()
    cbar.set_label("density (hist)")
    fig.savefig(out_dir / "predictive_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Filtering heatmap plot (t=1..T)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(
        H_filt,
        aspect="auto",
        origin="lower",
        extent=[centers[0], centers[-1], 1, T],
        interpolation="nearest",
    )
    plt.plot(z_true, np.arange(1, T + 1), lw=2)
    plt.xlabel("z")
    plt.ylabel("t")
    plt.title("Filtering heatmap: p(z_t | x_{1:t}) (weighted) + true z_t")
    cbar = plt.colorbar()
    cbar.set_label("density (weighted hist)")
    fig.savefig(out_dir / "filtering_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Branch-mass curves
    mleft_pred, mleft_filt = branch_mass_from_pf(pf)
    fig = plt.figure(figsize=(12, 4))
    plt.plot(np.arange(1, T + 1), mleft_pred, label="predictive mass: P(z_t<0 | x_{<t})")
    plt.plot(np.arange(1, T + 1), mleft_filt, label="filtering mass: P(z_t<0 | x_{<=t})")
    plt.axhline(0.05, ls="--")
    plt.xlabel("t")
    plt.ylabel("left-branch mass")
    plt.title("Branch-mass curves (evidence-driven collapse should show smooth decay)")
    plt.grid(True)
    plt.legend()
    fig.savefig(out_dir / "branch_mass_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ESS plot
    fig = plt.figure(figsize=(12, 4))
    plt.plot(np.arange(1, T + 1), pf.ess)
    plt.xlabel("t")
    plt.ylabel("ESS")
    plt.title("Effective sample size (ESS) after weighting")
    plt.grid(True)
    fig.savefig(out_dir / "ess.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # NEW: Evidence accumulation plots
    fig = plt.figure(figsize=(12, 4))
    plt.plot(np.arange(1, T + 1), pf.log_evidence)
    plt.xlabel("t")
    plt.ylabel("log p(x_t | x_{1:t-1})")
    plt.title("Per-step log evidence increment")
    plt.grid(True)
    fig.savefig(out_dir / "log_evidence_per_step.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4))
    plt.plot(np.arange(1, T + 1), pf.cum_log_evidence)
    plt.xlabel("t")
    plt.ylabel("sum_{s<=t} log p(x_s | x_{1:s-1})")
    plt.title("Cumulative log evidence (evidence accumulation)")
    plt.grid(True)
    fig.savefig(out_dir / "log_evidence_cumulative.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Optional: log-likelihood ratio proxies from branch masses
    fig = plt.figure(figsize=(12, 4))
    plt.plot(np.arange(1, T + 1), pf.log_lr_pred, label="predictive log((1-m)/m)")
    plt.plot(np.arange(1, T + 1), pf.log_lr_filt, label="filtering log((1-m)/m)")
    plt.xlabel("t")
    plt.ylabel("log odds (right vs left)")
    plt.title("Log-odds from branch masses (proxy likelihood-ratio accumulation)")
    plt.grid(True)
    plt.legend()
    fig.savefig(out_dir / "branch_log_odds.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Collapse-time sensitivity to resampling frequency
    resample_grid = [0, 1, 2, 3, 5, 10]
    collapse_times = []
    for re in resample_grid:
        pf_re = bootstrap_pf(
            x_seq=x,
            prior=prior,
            transition=transition,
            emission=emission,
            N=int(extras.get("N", 30000)),
            seed=123,
            extras=extras,
            resample_every=re,
        )
        _, mleft_f = branch_mass_from_pf(pf_re)
        collapse_times.append(find_collapse_time(mleft_f, thresh=0.05))

    fig = plt.figure(figsize=(10, 4))
    plt.plot(resample_grid, collapse_times, marker="o")
    plt.xlabel("resample_every (0 = never)")
    plt.ylabel("collapse time (t where left-mass<0.05)")
    plt.title("Collapse-time sensitivity to resampling frequency")
    plt.grid(True)
    fig.savefig(out_dir / "collapse_time_vs_resample_every.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save a small results text file
    txt = out_dir / "summary.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Synthetic 1D double-well analysis\n")
        f.write(f"T={T}, dz={dz}\n")
        f.write(f"seed={seed}\n")
        f.write(f"extras={extras}\n")
        f.write("\n")
        f.write("Saved plots:\n")
        for p in sorted(out_dir.glob("*.png")):
            f.write(f"  - {p.name}\n")
        f.write("\n")
        f.write("Notes:\n")
        f.write("  - log_evidence_per_step.png shows log p(x_t|x_<t)\n")
        f.write("  - log_evidence_cumulative.png shows cumulative evidence accumulation\n")
        f.write("  - branch_log_odds.png shows log-odds from predictive/filtering branch masses\n")

    print(f"[done] wrote results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()