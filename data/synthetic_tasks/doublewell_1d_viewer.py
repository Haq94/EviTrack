# data/synthetic_tasks/doublewell_1d_viewer.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

from data.synthetic_tasks.doublewell_1d_dataset import DoubleWell1DDatasetArtifact


def _transition_mean_np(z: np.ndarray, *, a: float, V: float, dt: float) -> np.ndarray:
    dU = (4.0 * V / (a ** 4)) * z * (z ** 2 - a ** 2)
    return z - dU * dt


def _emission_mean_np(z: np.ndarray, *, d: float, n: int) -> np.ndarray:
    z = np.asarray(z)
    inside = np.abs(z) <= d
    out = np.empty_like(z, dtype=float)
    out[inside] = z[inside] ** (2 * n)
    out[~inside] = z[~inside]
    return out


def compute_quadrature_products(
    x_obs: torch.Tensor,
    *,
    T: int,
    H: int,
    a: float,
    V: float,
    dt: float,
    sigma_z: float,
    d: float,
    n: int,
    sigma_x: float,
    z0_std: float,
    threshold: float = 0.8,
    zmin: float = -4.0,
    zmax: float = 4.0,
    G: int = 1000,
    Gx: int = 2000,
):
    """
    Recompute the same quadrature objects from a saved observation trajectory x_obs:
      - p_pred[t] ~ p(z_t | x_<t)
      - p_filt[t] ~ p(z_t | x_<=t)
      - mass_diff[t] = P(z_t>0|x_<t) - P(z_t<0|x_<t)
      - pzH_heat[t] ~ p(z_{t+H} | x_{1:t})
      - pxH_heat[t] ~ p(x_{t+H} | x_{1:t})
      - mass_diff_H[t]
      - disamb_time from threshold crossing of |mass_diff|
    """
    x_obs_np = x_obs.detach().cpu().numpy().reshape(T)

    # z-grid
    z_grid = np.linspace(zmin, zmax, G)
    dz = z_grid[1] - z_grid[0]

    # prior p(z1)
    p_pred = np.zeros((T, G), dtype=float)
    p_filt = np.zeros((T, G), dtype=float)

    p0 = (1.0 / (np.sqrt(2.0 * np.pi) * z0_std)) * np.exp(-0.5 * (z_grid / z0_std) ** 2)
    p0 = p0 / (np.sum(p0) * dz + 1e-12)
    p_pred[0] = p0

    # transition kernel
    mu_zj = _transition_mean_np(z_grid, a=a, V=V, dt=dt)
    var_z = sigma_z ** 2
    diff = z_grid[:, None] - mu_zj[None, :]
    K = (1.0 / np.sqrt(2.0 * np.pi * var_z)) * np.exp(-0.5 * (diff ** 2) / var_z)

    # emission on z-grid
    mu_x_grid = _emission_mean_np(z_grid, d=d, n=n)
    var_x = sigma_x ** 2

    # x-grid
    xmin = float(np.min(mu_x_grid) - 4.0 * np.sqrt(var_x))
    xmax = float(np.max(mu_x_grid) + 4.0 * np.sqrt(var_x))
    x_grid = np.linspace(xmin, xmax, Gx)
    dx = x_grid[1] - x_grid[0]

    # filtering / predictive recursion
    for t in range(T):
        xt = x_obs_np[t]

        logL = -0.5 * ((xt - mu_x_grid) ** 2) / var_x - 0.5 * np.log(2.0 * np.pi * var_x)
        logL -= np.max(logL)
        L = np.exp(logL)

        pf = L * p_pred[t]
        pf = pf / (np.sum(pf) * dz + 1e-12)
        p_filt[t] = pf

        if t < T - 1:
            pn = K @ (pf * dz)
            pn = pn / (np.sum(pn) * dz + 1e-12)
            p_pred[t + 1] = pn

    # marginal mass diff for predictive density p(z_t | x_<t)
    pos_mask = z_grid >= 0
    neg_mask = z_grid < 0

    mass_diff = np.zeros(T, dtype=float)
    for t in range(T):
        pt = p_pred[t]
        m_pos = np.sum(pt[pos_mask]) * dz
        m_neg = np.sum(pt[neg_mask]) * dz
        mass_diff[t] = m_pos - m_neg

    idx = np.where(np.abs(mass_diff) > threshold)[0]
    disamb_time = int(idx[0] + 1) if len(idx) > 0 else -1

    # H-step forecast heatmaps
    TT = T - H
    pzH_heat = np.zeros((TT, G), dtype=float)
    pxH_heat = np.zeros((TT, Gx), dtype=float)

    X = x_grid[:, None]
    MU = mu_x_grid[None, :]
    G_emit = (1.0 / np.sqrt(2.0 * np.pi * var_x)) * np.exp(-0.5 * ((X - MU) ** 2) / var_x)

    for t in range(1, TT + 1):
        pz = p_filt[t - 1].copy()

        for _ in range(H):
            pz = K @ (pz * dz)
            pz = pz / (np.sum(pz) * dz + 1e-12)

        pzH_heat[t - 1] = pz

        px = G_emit @ (pz * dz)
        px = px / (np.sum(px) * dx + 1e-12)
        pxH_heat[t - 1] = px

    mass_diff_H = np.zeros(TT, dtype=float)
    for t in range(TT):
        pz = pzH_heat[t]
        m_pos = np.sum(pz[pos_mask]) * dz
        m_neg = np.sum(pz[neg_mask]) * dz
        mass_diff_H[t] = m_pos - m_neg

    return {
        "z_grid": z_grid,
        "x_grid": x_grid,
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "p_pred": p_pred,
        "p_filt": p_filt,
        "mass_diff": mass_diff,
        "disamb_time": disamb_time,
        "pzH_heat": pzH_heat,
        "pxH_heat": pxH_heat,
        "mass_diff_H": mass_diff_H,
        "TT": TT,
        "H": H,
    }


def plot_dataset_index(
    artifact: DoubleWell1DDatasetArtifact,
    *,
    index: int,
    H: int = 30,
    threshold: Optional[float] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    G: Optional[int] = None,
    Gx: int = 2000,
    show: bool = True,
):
    meta = artifact.meta
    x_obs = artifact.x[index]   # [T, dx]
    z_true = artifact.z[index].reshape(-1).detach().cpu().numpy()

    T = int(meta["T"])
    threshold = float(meta["delay_threshold"] if threshold is None else threshold)
    zmin = float(meta["quadrature_zmin"] if zmin is None else zmin)
    zmax = float(meta["quadrature_zmax"] if zmax is None else zmax)
    G = int(meta["quadrature_G"] if G is None else G)

    out = compute_quadrature_products(
        x_obs,
        T=T,
        H=H,
        a=float(meta["a"]),
        V=float(meta["V"]),
        dt=float(meta["dt"]),
        sigma_z=float(meta["sigma_z"]),
        d=float(meta["d"]),
        n=int(meta["n"]),
        sigma_x=float(meta["sigma_x"]),
        z0_std=float(meta["z0_std"]),
        threshold=threshold,
        zmin=zmin,
        zmax=zmax,
        G=G,
        Gx=Gx,
    )

    seed = int(artifact.data_seed_ids[index].item())
    delayed_flag_saved = bool(artifact.delayed_flag[index].item())
    disamb_time_saved = int(artifact.disamb_time[index].item())
    disamb_time_recomputed = int(out["disamb_time"])

    print("=" * 80)
    print(f"dataset index      : {index}")
    print(f"data_seed          : {seed}")
    print(f"saved delayed_flag : {delayed_flag_saved}")
    print(f"saved disamb_time  : {disamb_time_saved}")
    print(f"recomputed disamb  : {disamb_time_recomputed}")
    print(f"horizon H          : {H}")

    # 1) mass difference
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(1, T + 1), out["mass_diff"], linewidth=2)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.ylim([-1.05, 1.05])
    plt.xlabel("t")
    plt.ylabel("m⁺ - m⁻")
    plt.title(f"Predictive mass difference | idx={index}, seed={seed}")
    plt.tight_layout()
    if show:
        plt.show()

    # 2) predictive latent heatmap p(z_t | x_<t)
    plt.figure(figsize=(10, 4))
    plt.imshow(
        out["p_pred"].T,
        origin="lower",
        aspect="auto",
        extent=[1, T, out["zmin"], out["zmax"]],
        interpolation="nearest",
    )
    plt.plot(np.arange(1, T + 1), z_true, linewidth=1.5)
    plt.colorbar(label="density")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.title(f"Predictive latent heatmap p(z_t | x_<t) | idx={index}, seed={seed}")
    plt.tight_layout()
    if show:
        plt.show()

    # 3) H-step latent forecast heatmap
    TT = out["TT"]
    plt.figure(figsize=(10, 4))
    plt.imshow(
        out["pzH_heat"].T,
        origin="lower",
        aspect="auto",
        extent=[1, TT, out["zmin"], out["zmax"]],
        interpolation="nearest",
    )
    plt.plot(np.arange(1, TT + 1), z_true[H:], linewidth=1.5)
    plt.colorbar(label="density")
    plt.xlabel("conditioning time t")
    plt.ylabel(f"z (forecast at t+H, H={H})")
    plt.title(f"Latent forecast p(z_(t+H)|x_1:t) | idx={index}, seed={seed}")
    plt.tight_layout()
    if show:
        plt.show()

    # 4) H-step observation forecast heatmap
    x_true = x_obs.reshape(-1).detach().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(
        out["pxH_heat"].T,
        origin="lower",
        aspect="auto",
        extent=[1, TT, out["xmin"], out["xmax"]],
        interpolation="nearest",
    )
    plt.plot(np.arange(1, TT + 1), x_true[H:], linewidth=1.5)
    plt.colorbar(label="density")
    plt.xlabel("conditioning time t")
    plt.ylabel(f"x (forecast at t+H, H={H})")
    plt.title(f"Observation forecast p(x_(t+H)|x_1:t) | idx={index}, seed={seed}")
    plt.tight_layout()
    if show:
        plt.show()

    # 5) H-step mass difference
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(1, TT + 1), out["mass_diff_H"], linewidth=2)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.ylim([-1.05, 1.05])
    plt.xlabel("conditioning time t")
    plt.ylabel(f"Δ^(H), H={H}")
    plt.title(f"H-step-ahead mass difference | idx={index}, seed={seed}")
    plt.tight_layout()
    if show:
        plt.show()


def plot_first_k_delayed(
    artifact: DoubleWell1DDatasetArtifact,
    *,
    k: int = 3,
    H: int = 30,
):
    idx = torch.where(artifact.delayed_flag)[0].tolist()
    idx = idx[:k]
    if len(idx) == 0:
        print("No delayed examples found.")
        return
    for i in idx:
        plot_dataset_index(artifact, index=i, H=H, show=True)


def plot_first_k_non_delayed(
    artifact: DoubleWell1DDatasetArtifact,
    *,
    k: int = 3,
    H: int = 30,
):
    idx = torch.where(~artifact.delayed_flag)[0].tolist()
    idx = idx[:k]
    if len(idx) == 0:
        print("No non-delayed examples found.")
        return
    for i in idx:
        plot_dataset_index(artifact, index=i, H=H, show=True)


if __name__ == "__main__":
    dataset_root = Path("data/datasets/doublewell_1d/benchmark_v1")
    artifact = DoubleWell1DDatasetArtifact.load(dataset_root)

    print("Loaded dataset from:", dataset_root)
    print("num sequences      :", artifact.x.shape[0])
    print("num delayed        :", int(artifact.delayed_flag.sum().item()))
    print("num non-delayed    :", int((~artifact.delayed_flag).sum().item()))

    # Choose ONE of the following modes:

    # 1) plot a specific dataset index
    plot_dataset_index(artifact, index=0, H=30)

    # 2) plot first few delayed examples
    # plot_first_k_delayed(artifact, k=3, H=30)

    # 3) plot first few non-delayed examples
    # plot_first_k_non_delayed(artifact, k=3, H=30)