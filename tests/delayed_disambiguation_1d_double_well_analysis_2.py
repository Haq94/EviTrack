# tests/delayed_disambiguation_1d_double_well_analysis_2.py
# Runs analysis + saves plots/results to: results/_synthetic_1d_double_well_analysis/

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from data.synthetic_generator import (
    GaussianPrior, GaussianTransition, GaussianEmission, generate_sequences
)

# ============================================================
# 1) Synthetic model (copied from your script for consistency)
# ============================================================

def trans_mean(z_prev, extras):
    a = float(extras.get("a", 0.10))
    b = float(extras.get("b", 0.10))
    return z_prev + a * z_prev - b * (z_prev ** 3)

def trans_cov(z_prev, extras):
    q = float(extras.get("process_noise", 0.12))
    return (q ** 2) * np.eye(1)

def emit_mean(z, extras):
    # 2D emission:
    # x1 = z^2
    # x2 = piecewise: inside band -> even power (sign-destroying), outside -> z (sign-revealing)
    z1 = z[:, :1]
    x1 = z1 ** 2
    eps, n = float(extras.get("eps", 1.3)), int(extras.get("n", 2))
    inside = (np.abs(z1) < eps)
    x2_inside = (z1 / (eps + 1e-12)) ** (2 * n)     # even power -> loses sign
    x2_outside = z1                                  # reveals sign
    x2 = np.where(inside, x2_inside, x2_outside)
    return np.concatenate([x1, x2], axis=1)

def emit_cov(z, extras):
    r = float(extras.get("emit_noise", 0.25))
    return (r ** 2) * np.eye(2)

# ============================================================
# 2) PF utilities
# ============================================================

def log_gaussian_full(x, mu, cov):
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

def systematic_resample(w, rng):
    N = len(w)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(w)
    return np.searchsorted(cumsum, positions)

@dataclass
class PFOutputs:
    z_filt: List[np.ndarray]     # list length T, each (N,1) BEFORE resample, AFTER weighting
    w_filt: List[np.ndarray]     # list length T, each (N,)
    z_pred: List[np.ndarray]     # list length T-1, each (N,1): predictive p(z_{t+1}|x_{1:t})
    ess: np.ndarray              # (T,)

def bootstrap_pf(
    x_seq: np.ndarray,
    prior: GaussianPrior,
    transition: GaussianTransition,
    emission: GaussianEmission,
    *,
    N: int = 30000,
    seed: int = 0,
    extras: Optional[Dict] = None,
    resample_every: int = 1,
) -> PFOutputs:
    """
    Bootstrap PF (proposal = transition). Stores weights so filtering is meaningful.

    Returns:
      z_filt[t], w_filt[t] approximate p(z_t | x_{1:t})
      z_pred[t] approximates p(z_{t+1} | x_{1:t})
    """
    rng = np.random.default_rng(seed)
    extras = dict(extras or {})

    T, dx = x_seq.shape
    dz = int(np.asarray(prior.mu0).shape[0])

    z = rng.multivariate_normal(mean=np.asarray(prior.mu0), cov=np.asarray(prior.cov0), size=N)  # (N,1)
    w = np.ones(N, dtype=float) / N

    z_filt, w_filt, z_pred = [], [], []
    ess = np.zeros((T,), dtype=float)

    for t in range(T):
        mu_x = np.asarray(emission.mean_fn(z, extras), dtype=float)  # (N,dx)
        cov_x = np.asarray(emission.cov_fn(z, extras), dtype=float)  # (dx,dx) or (N,dx,dx)
        xt_rep = np.repeat(x_seq[t:t+1], N, axis=0)

        logw = log_gaussian_full(xt_rep, mu_x, cov_x)
        logw = logw - logw.max()
        w = w * np.exp(logw)
        w = w / (w.sum() + 1e-12)

        ess[t] = 1.0 / np.sum(w ** 2)

        # Store BEFORE resample; this is the (weighted) filtering posterior at time t
        z_filt.append(z.copy())
        w_filt.append(w.copy())

        # Resample
        if resample_every > 0 and ((t + 1) % resample_every == 0):
            idx = systematic_resample(w, rng)
            z = z[idx]
            w = np.ones(N, dtype=float) / N

        # Predict next
        if t < T - 1:
            mu_z = np.asarray(transition.mean_fn(z, extras), dtype=float)
            cov_z = np.asarray(transition.cov_fn(z, extras), dtype=float)
            if cov_z.ndim == 2:
                cov_z = np.broadcast_to(cov_z, (N, dz, dz))

            z_next = np.empty((N, dz), dtype=float)
            for i in range(N):
                z_next[i] = rng.multivariate_normal(mean=mu_z[i], cov=cov_z[i])
            z_pred.append(z_next)
            z = z_next

    return PFOutputs(z_filt=z_filt, w_filt=w_filt, z_pred=z_pred, ess=ess)

# ============================================================
# 3) Density helpers + branch mass + collapse time
# ============================================================

def hist_density(samples_1d, *, edges, weights=None, eps=1e-12):
    samples_1d = np.asarray(samples_1d, dtype=float)
    if weights is None:
        hist, _ = np.histogram(samples_1d, bins=edges, density=False)
        hist = hist.astype(float)
    else:
        w = np.asarray(weights, dtype=float)
        hist, _ = np.histogram(samples_1d, bins=edges, weights=w, density=False)
        hist = hist.astype(float)
    hist = hist / (hist.sum() + eps)
    return hist

def build_common_edges(z_pred_list, nbins=180):
    allz = np.concatenate([zp[:, 0] for zp in z_pred_list], axis=0)
    z_min = float(np.percentile(allz, 0.5))
    z_max = float(np.percentile(allz, 99.5))
    pad = 0.20 * (z_max - z_min + 1e-9)
    z_min -= pad
    z_max += pad
    edges = np.linspace(z_min, z_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers

def predictive_heatmap(z_pred, *, nbins=180):
    edges, centers = build_common_edges(z_pred, nbins=nbins)
    H = np.zeros((len(z_pred), nbins), dtype=float)
    for t, zp in enumerate(z_pred):
        H[t] = hist_density(zp[:, 0], edges=edges)
    return H, centers

def filtering_heatmap(z_filt, w_filt, *, nbins=180, z_min=None, z_max=None):
    # Use filter samples to set grid if desired; simplest: use percentiles from all filter samples
    allz = np.concatenate([zf[:, 0] for zf in z_filt], axis=0)
    if z_min is None:
        z_min = float(np.percentile(allz, 0.5))
    if z_max is None:
        z_max = float(np.percentile(allz, 99.5))
    pad = 0.20 * (z_max - z_min + 1e-9)
    z_min -= pad
    z_max += pad
    edges = np.linspace(z_min, z_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    H = np.zeros((len(z_filt), nbins), dtype=float)
    for t, (zf, wf) in enumerate(zip(z_filt, w_filt)):
        H[t] = hist_density(zf[:, 0], edges=edges, weights=wf)
    return H, centers

def branch_mass(samples_1d, weights=None):
    s = np.asarray(samples_1d, dtype=float)
    if weights is None:
        return float(np.mean(s < 0.0))
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    return float(np.sum(w[s < 0.0]))

def estimate_collapse_time(mass_left: np.ndarray, *, thresh=0.05):
    """
    Return first t index (1-based) where left-mass drops below thresh and stays there.
    If never, return None.
    """
    below = mass_left < thresh
    if not np.any(below):
        return None
    # require persistence: once below, stay below (simple)
    for i in range(len(mass_left)):
        if below[i] and np.all(below[i:]):
            return i + 1  # 1-based time index
    # fallback: first below
    return int(np.argmax(below)) + 1

# ============================================================
# 4) Forecast divergence test (mixture vs forced-right)
# ============================================================

def rollout_forecast_from_particles(
    z_particles: np.ndarray,          # (N,1)
    weights: Optional[np.ndarray],    # (N,) or None
    transition: GaussianTransition,
    emission: GaussianEmission,
    extras: Dict,
    *,
    H: int = 20,
    seed: int = 0,
    use_weights_resample: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Roll out H steps, returning samples of x_{t+1:t+H} for each particle.
    For simplicity: one rollout per particle (can extend later).
    """
    rng = np.random.default_rng(seed)
    z = z_particles.copy()
    N = z.shape[0]
    dz = z.shape[1]

    if weights is not None and use_weights_resample:
        w = weights / (weights.sum() + 1e-12)
        idx = rng.choice(N, size=N, replace=True, p=w)
        z = z[idx]

    xs = []
    for h in range(H):
        mu_z = np.asarray(transition.mean_fn(z, extras), dtype=float)  # (N,1)
        cov_z = np.asarray(transition.cov_fn(z, extras), dtype=float)  # (1,1) or (N,1,1)
        if cov_z.ndim == 2:
            cov_z = np.broadcast_to(cov_z, (N, dz, dz))
        z_next = np.empty_like(z)
        for i in range(N):
            z_next[i] = rng.multivariate_normal(mean=mu_z[i], cov=cov_z[i])
        z = z_next

        mu_x = np.asarray(emission.mean_fn(z, extras), dtype=float)  # (N,2)
        cov_x = np.asarray(emission.cov_fn(z, extras), dtype=float)  # (2,2) or (N,2,2)
        if cov_x.ndim == 2:
            cov_x = np.broadcast_to(cov_x, (N, 2, 2))
        x_next = np.empty((N, 2), dtype=float)
        for i in range(N):
            x_next[i] = rng.multivariate_normal(mean=mu_x[i], cov=cov_x[i])
        xs.append(x_next)

    X = np.stack(xs, axis=1)  # (N,H,2)
    mean = X.mean(axis=0)     # (H,2)
    var = X.var(axis=0)       # (H,2)
    return {"X": X, "mean": mean, "var": var}

# ============================================================
# 5) Main analysis
# ============================================================

def main():
    # ------------------------------
    # Config (edit these as needed)
    # ------------------------------
    T = 90
    dz = 1
    seed_data = 87
    seed_pf = 66

    extras = {
        "a": 0.15,
        "b": 0.10,
        "process_noise": 0.10,
        "emit_noise": 0.15,
        # emission band params (you used fixed eps,n inside emit_mean; expose them here too)
        "eps": 1.4,
        "n": 2,
    }

    N_main = 30000
    resample_every_main = 3

    # Forecast divergence params
    t_star = 55   # choose near-but-before branch collapse (adjust as you like)
    H_forecast = 20
    left_thresh = 0.05

    # Sensitivity grids
    resample_grid = [1, 2, 3, 5, 10, 0]
    N_grid = [5000, 10000, 30000, 60000]

    # Output directory
    out_dir = Path("results") / "_synthetic_1d_double_well_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    cfg = dict(
        T=T, dz=dz, seed_data=seed_data, seed_pf=seed_pf,
        extras=extras,
        N_main=N_main, resample_every_main=resample_every_main,
        t_star=t_star, H_forecast=H_forecast, left_thresh=left_thresh,
        resample_grid=resample_grid, N_grid=N_grid,
    )
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ------------------------------
    # Build generator
    # ------------------------------
    prior = GaussianPrior(mu0=np.zeros(dz), cov0=np.eye(dz))
    transition = GaussianTransition(mean_fn=trans_mean, cov_fn=trans_cov)
    emission = GaussianEmission(mean_fn=emit_mean, cov_fn=emit_cov)

    batch = generate_sequences(
        T=T, B=1,
        prior=prior, transition=transition, emission=emission,
        seed=seed_data, return_logp=False,
        extras=extras
    )
    x = batch.x[0]        # (T,2)
    z_true = batch.z[0]   # (T,1)

    # ------------------------------
    # Run PF (main)
    # ------------------------------
    pf = bootstrap_pf(
        x, prior, transition, emission,
        N=N_main, seed=seed_pf, extras=extras,
        resample_every=resample_every_main
    )

    # ------------------------------
    # Compute heatmaps
    # ------------------------------
    Hpred, zgrid_pred = predictive_heatmap(pf.z_pred, nbins=180)
    Hfilt, zgrid_filt = filtering_heatmap(pf.z_filt, pf.w_filt, nbins=180)

    # Predictive heatmap plot
    fig = plt.figure(figsize=(11, 4))
    plt.imshow(
        Hpred, aspect="auto", origin="lower",
        extent=[zgrid_pred[0], zgrid_pred[-1], 1, T - 1],
        interpolation="nearest"
    )
    plt.colorbar(label="density (hist)")
    plt.plot(z_true[1:, 0], np.arange(1, T), linewidth=2)
    plt.xlabel("z")
    plt.ylabel("t")
    plt.title("Predictive heatmap: p(z_t | x_{1:t-1}) + true z_t")
    plt.grid(False)
    fig.savefig(out_dir / "heatmap_predictive.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Filtering heatmap plot (weighted!)
    fig = plt.figure(figsize=(11, 4))
    plt.imshow(
        Hfilt, aspect="auto", origin="lower",
        extent=[zgrid_filt[0], zgrid_filt[-1], 1, T],
        interpolation="nearest"
    )
    plt.colorbar(label="density (weighted hist)")
    plt.plot(z_true[:, 0], np.arange(1, T + 1), linewidth=2)
    plt.xlabel("z")
    plt.ylabel("t")
    plt.title("Filtering heatmap: p(z_t | x_{1:t}) (weighted) + true z_t")
    plt.grid(False)
    fig.savefig(out_dir / "heatmap_filtering.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------
    # ESS curve
    # ------------------------------
    fig = plt.figure(figsize=(10, 3))
    plt.plot(np.arange(1, T + 1), pf.ess)
    plt.xlabel("t")
    plt.ylabel("ESS")
    plt.title("ESS after weighting with x_t")
    plt.grid(True)
    fig.savefig(out_dir / "ess_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------
    # Branch mass curves
    # ------------------------------
    # predictive left mass uses z_pred[t-1] ~ p(z_t|x_{<t})
    mleft_pred = np.array([branch_mass(pf.z_pred[t-1][:, 0]) for t in range(1, T)], dtype=float)
    # filtering left mass uses weights
    mleft_filt = np.array([branch_mass(zf[:, 0], wf) for zf, wf in zip(pf.z_filt, pf.w_filt)], dtype=float)

    t_pred = np.arange(1, T)
    t_filt = np.arange(1, T + 1)

    fig = plt.figure(figsize=(10, 3))
    plt.plot(t_pred, mleft_pred, label="predictive mass: P(z_t<0 | x_{<t})")
    plt.plot(t_filt, mleft_filt, label="filtering mass: P(z_t<0 | x_{<=t})")
    plt.axhline(left_thresh, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("left-branch mass")
    plt.title("Branch-mass curves (evidence-driven collapse should show smooth decay)")
    plt.grid(True)
    plt.legend()
    fig.savefig(out_dir / "branch_mass_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    collapse_pred_t = estimate_collapse_time(mleft_pred, thresh=left_thresh)
    collapse_filt_t = estimate_collapse_time(mleft_filt, thresh=left_thresh)

    # ------------------------------
    # Emission sanity plots
    # ------------------------------
    fig = plt.figure(figsize=(10, 3))
    plt.plot(np.arange(1, T + 1), x[:, 0], label="x1")
    plt.plot(np.arange(1, T + 1), x[:, 1], label="x2")
    plt.xlabel("t")
    plt.title("Observation channels (sanity)")
    plt.grid(True)
    plt.legend()
    fig.savefig(out_dir / "observations_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(z_true[:, 0], x[:, 1], s=14)
    plt.xlabel("z_true")
    plt.ylabel("x2")
    plt.title("Emission sanity: x2 vs z_true")
    plt.grid(True)
    fig.savefig(out_dir / "emission_x2_vs_z.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------
    # Resampling sensitivity: collapse time vs resample_every
    # ------------------------------
    resample_results = []
    for re in resample_grid:
        pf_re = bootstrap_pf(
            x, prior, transition, emission,
            N=N_main, seed=seed_pf, extras=extras,
            resample_every=re
        )
        mleft_filt_re = np.array([branch_mass(zf[:, 0], wf) for zf, wf in zip(pf_re.z_filt, pf_re.w_filt)], dtype=float)
        c = estimate_collapse_time(mleft_filt_re, thresh=left_thresh)
        resample_results.append((re, -1 if c is None else c))

    fig = plt.figure(figsize=(8, 3))
    xs = [r for r, _ in resample_results]
    ys = [c for _, c in resample_results]
    plt.plot(xs, ys, marker="o")
    plt.xlabel("resample_every (0 = never)")
    plt.ylabel(f"collapse time (t where left-mass<{left_thresh})")
    plt.title("Collapse-time sensitivity to resampling frequency")
    plt.grid(True)
    fig.savefig(out_dir / "sensitivity_resample_every.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------
    # Particle sensitivity: collapse time vs N
    # ------------------------------
    N_results = []
    for N in N_grid:
        pf_N = bootstrap_pf(
            x, prior, transition, emission,
            N=N, seed=seed_pf, extras=extras,
            resample_every=resample_every_main
        )
        mleft_filt_N = np.array([branch_mass(zf[:, 0], wf) for zf, wf in zip(pf_N.z_filt, pf_N.w_filt)], dtype=float)
        c = estimate_collapse_time(mleft_filt_N, thresh=left_thresh)
        N_results.append((N, -1 if c is None else c))

    fig = plt.figure(figsize=(8, 3))
    xs = [n for n, _ in N_results]
    ys = [c for _, c in N_results]
    plt.plot(xs, ys, marker="o")
    plt.xlabel("N particles")
    plt.ylabel(f"collapse time (t where left-mass<{left_thresh})")
    plt.title("Collapse-time sensitivity to particle count")
    plt.grid(True)
    fig.savefig(out_dir / "sensitivity_particle_count.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------
    # Forecast divergence test: mixture vs forced-right-branch
    # ------------------------------
    # Use filtering posterior at t_star (1-based index)
    t_idx = max(1, min(T, int(t_star))) - 1  # 0-based
    zt = pf.z_filt[t_idx]
    wt = pf.w_filt[t_idx]

    # Mixture forecast (resample by weights)
    mix = rollout_forecast_from_particles(
        zt, wt, transition, emission, extras,
        H=H_forecast, seed=999, use_weights_resample=True
    )

    # Forced right-branch forecast (z>=0)
    mask_right = (zt[:, 0] >= 0.0)
    z_right = zt[mask_right]
    w_right = wt[mask_right]
    if z_right.shape[0] < 10:
        # fallback: if too few, don't crash; just use mixture (shouldn't happen unless branch fully dead)
        z_right = zt
        w_right = wt

    right = rollout_forecast_from_particles(
        z_right, w_right, transition, emission, extras,
        H=H_forecast, seed=1001, use_weights_resample=True
    )

    # Plot mean forecasts for x2 (more interpretable here)
    hgrid = np.arange(1, H_forecast + 1)
    fig = plt.figure(figsize=(10, 3))
    plt.plot(hgrid, mix["mean"][:, 1], label="mixture forecast mean (x2)")
    plt.plot(hgrid, right["mean"][:, 1], label="forced-right forecast mean (x2)")
    plt.xlabel("horizon step h")
    plt.ylabel("E[x2_{t+h} | ...]")
    plt.title(f"Forecast divergence at t*={t_star}: mixture vs forced-right (x2)")
    plt.grid(True)
    plt.legend()
    fig.savefig(out_dir / "forecast_divergence_x2_mean.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save a scalar divergence summary too
    # (simple L2 between mean forecast curves for x2)
    forecast_div = float(np.sqrt(np.mean((mix["mean"][:, 1] - right["mean"][:, 1]) ** 2)))

    # ------------------------------
    # Save results arrays
    # ------------------------------
    np.savez(
        out_dir / "results.npz",
        z_true=z_true,
        x=x,
        ess=pf.ess,
        mleft_pred=mleft_pred,
        mleft_filt=mleft_filt,
        collapse_pred_t=-1 if collapse_pred_t is None else collapse_pred_t,
        collapse_filt_t=-1 if collapse_filt_t is None else collapse_filt_t,
        resample_results=np.array(resample_results, dtype=float),
        N_results=np.array(N_results, dtype=float),
        forecast_mix_mean=mix["mean"],
        forecast_mix_var=mix["var"],
        forecast_right_mean=right["mean"],
        forecast_right_var=right["var"],
        forecast_divergence_x2_mean_l2=forecast_div,
    )

    # Small text summary
    summary = [
        f"collapse_pred_t (left-mass<thresh) = {collapse_pred_t}",
        f"collapse_filt_t (left-mass<thresh) = {collapse_filt_t}",
        f"forecast_divergence_x2_mean_l2 @ t*={t_star}, H={H_forecast} = {forecast_div:.6f}",
        "",
        "Interpretation tips:",
        "- If collapse time changes wildly with resample_every or N, branch death may be PF artifact.",
        "- If filtering left-mass decays smoothly but predictive is jumpy, that's normal: conditioning on x_t matters.",
        "- If forecast_divergence is ~0, branches don't matter for forecasting (bad synthetic for EviTrack).",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary))

    print(f"[done] wrote analysis to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()