# doublewell_predictive_heatmap_quadrature_with_forecast.py
# Extends doublewell_predictive_heatmap_quadrature.py 
# Adds a heatmap for p(x_{t+H} | x_{1:t}) via deterministic quadrature.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0) SETTINGS
# ============================================================
SEEDS = [n for n in range(100)]          # keep small; this plots per-seed
T = 200
H = 30              # <-- forecast horizon: plot p(x_{t+H} | x_{1:t})
threshold = 0.8     # Threshold used to estimate disambiguation time

# ---- Transition params ----
a = 3.0
V = 0.06                        # 0.06
sigma_z = 0.05

# ---- Emission params ----
n = 1
d = 2.0
sigma_x = 0.12

# ---- Prior ----
z1_std = 1.0

# ---- Quadrature grid over z ----
zmin, zmax = -4.0, 4.0
G = 1000

# ---- Grid over x for forecast heatmap ----
Gx = 10000
# ============================================================


# ============================================================
# 1) MODEL FUNCTIONS
# ============================================================

def transition_mean(z):
    # U(z) = V/a^4 (z^2 - a^2)^2 => dU/dz = (4V/a^4) z (z^2 - a^2)
    dU = (4.0 * V / (a ** 4)) * z * (z ** 2 - a ** 2)
    return z - dU

def transition_cov(z):
    return sigma_z ** 2

def emission_mean(z):
    z = np.asarray(z)
    inside = (np.abs(z) <= d)
    mu = np.empty_like(z, dtype=float)
    mu[inside] = z[inside] ** (2 * n)
    mu[~inside] = z[~inside]
    return mu

def emission_cov(z):
    return sigma_x ** 2


# ============================================================
# 2) QUADRATURE SETUP
# ============================================================

# z grid
z_grid = np.linspace(zmin, zmax, G)
dz = z_grid[1] - z_grid[0]

# prior density p(z1) on grid
p_pred0 = (1.0 / (np.sqrt(2.0 * np.pi) * z1_std)) * np.exp(-0.5 * (z_grid / z1_std) ** 2)
p_pred0 = p_pred0 / (np.sum(p_pred0) * dz + 1e-12)

# transition kernel K[i,j] = p(z_next=z_i | z=z_j)
mu_zj = transition_mean(z_grid)
var_z = transition_cov(z_grid)
diff = z_grid[:, None] - mu_zj[None, :]
K = (1.0 / np.sqrt(2.0 * np.pi * var_z)) * np.exp(-0.5 * (diff ** 2) / var_z)
# (We renormalize the propagated density each time to handle finite grid truncation.)

# emission mean on grid (used repeatedly)
mu_x_grid = emission_mean(z_grid)
var_x = emission_cov(z_grid)

# choose x grid automatically from emission range on z grid
xmin = float(np.min(mu_x_grid) - 4.0 * np.sqrt(var_x))
xmax = float(np.max(mu_x_grid) + 4.0 * np.sqrt(var_x))
x_grid = np.linspace(xmin, xmax, Gx)
dx = x_grid[1] - x_grid[0]


# ============================================================
# 3) MAIN SCRIPT
# ============================================================

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    # ---- Generate one true trajectory and observations ----
    z_true = np.zeros(T, dtype=float)
    x_obs = np.zeros(T, dtype=float)

    z_true[0] = rng.normal(0.0, z1_std)
    x_obs[0] = emission_mean(z_true[0]) + rng.normal(0.0, sigma_x)

    for t in range(1, T):
        z_true[t] = transition_mean(z_true[t - 1]) + rng.normal(0.0, sigma_z)
        x_obs[t] = emission_mean(z_true[t]) + rng.normal(0.0, sigma_x)

    # ------------------------------------------------------------
    # 3A) Quadrature recursion:
    #     - p_pred[t]  ≈ p(z_{t+1} | x_<t+1)   predictive-before x_{t+1}
    #     - p_filt[t]  ≈ p(z_{t+1} | x_<=t+1)  filtering-after x_{t+1}
    # ------------------------------------------------------------
    p_pred = np.zeros((T, G), dtype=float)
    p_filt = np.zeros((T, G), dtype=float)

    p_pred[0] = p_pred0.copy()  # p(z1 | x_<1)

    for t in range(T):
        # Update with x_{t+1} = x_obs[t]
        xt = x_obs[t]

        # likelihood on z-grid: L[j] = p(xt | z_grid[j])
        logL = -0.5 * ((xt - mu_x_grid) ** 2) / var_x - 0.5 * np.log(2.0 * np.pi * var_x)
        logL -= np.max(logL)
        L = np.exp(logL)

        # filtering
        pf = L * p_pred[t]
        pf = pf / (np.sum(pf) * dz + 1e-12)
        p_filt[t] = pf

        # predict next (if any): p(z_{t+2} | x_<=t+1) = ∫ p(z'|z) p_filt(z) dz
        if t < T - 1:
            pn = K @ (pf * dz)
            pn = pn / (np.sum(pn) * dz + 1e-12)
            p_pred[t + 1] = pn

    # ============================================================
    # Marginal mass difference: m_t^+ - m_t^-
    # using predictive p(z_t | x_<t) stored in p_pred
    # ============================================================

    mass_diff = np.zeros(T)

    # indices for positive and negative regions
    pos_mask = (z_grid >= 0)
    neg_mask = (z_grid < 0)

    for t in range(T):
        pt = p_pred[t]

        m_pos = np.sum(pt[pos_mask]) * dz
        m_neg = np.sum(pt[neg_mask]) * dz

        mass_diff[t] = m_pos - m_neg

    idx = np.where(np.abs(mass_diff) > threshold)[0]
    if len(idx) > 0:
        disamb_time = idx[0] + 1
        print(f"[seed {seed}] Estimated disambiguation time (|m+ - m-|>{threshold}): t={disamb_time}")
    else:
        print(f"[seed {seed}] No disambiguation detected at threshold {threshold}")

    # ------------------------------------------------------------
    # Plot mass difference
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 3))

    plt.plot(np.arange(1, T + 1), mass_diff, linewidth=2)
    plt.axhline(0.0, linestyle='--', linewidth=1)

    plt.ylim([-1.05, 1.05])
    plt.xlabel("t")
    plt.ylabel("m⁺ - m⁻")
    plt.title("Marginal mass difference: p(z_t | x_<t)")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 3B) Heatmap 1: p(z_t | x_<t)  
    #     Here p_pred[t] corresponds to p(z_{t+1} | x_<t+1).
    #     So plotting p_pred as columns t=1..T is correct.
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.imshow(
        p_pred.T,
        origin="lower",
        aspect="auto",
        extent=[1, T, zmin, zmax],
        interpolation="nearest",
    )
    plt.plot(np.arange(1, T + 1), z_true, linewidth=1.5)
    plt.colorbar(label="density")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.title("Quadrature heatmap of predictive p(z_t | x_<t)")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 3C) Forecast heatmaps aligned at horizon H:
    #   p(z_{t+H} | x_{1:t})  and  p(x_{t+H} | x_{1:t})
    # for conditioning times t = 1..T-H
    # ------------------------------------------------------------
    TT = T - H

    pzH_heat = np.zeros((TT, G), dtype=float)   # over z_grid
    pxH_heat = np.zeros((TT, Gx), dtype=float)  # over x_grid

    # Precompute emission Gaussian pdf values on (x_grid, z_grid)
    # G_emit[i,j] = N(x_grid[i]; mu_x(z_j), var_x)
    X = x_grid[:, None]
    MU = mu_x_grid[None, :]
    G_emit = (1.0 / np.sqrt(2.0 * np.pi * var_x)) * np.exp(-0.5 * ((X - MU) ** 2) / var_x)

    for t in range(1, TT + 1):  # conditioning time t = 1..T-H (math time)
        # start from filtering at time t: p(z_t | x_{1:t})
        pz = p_filt[t - 1].copy()

        # propagate H steps (no updates) -> p(z_{t+H} | x_{1:t})
        for _ in range(H):
            pz = K @ (pz * dz)
            pz = pz / (np.sum(pz) * dz + 1e-12)

        # store latent forecast density
        pzH_heat[t - 1] = pz

        # map to x forecast by integrating emission
        px = G_emit @ (pz * dz)
        px = px / (np.sum(px) * dx + 1e-12)
        pxH_heat[t - 1] = px


    # ---- Plot z-forecast heatmap: p(z_{t+H} | x_{1:t}) ----
    plt.figure(figsize=(10, 4))
    plt.imshow(
        pzH_heat.T,
        origin="lower",
        aspect="auto",
        extent=[1, TT, zmin, zmax],
        interpolation="nearest",
    )
    plt.plot(np.arange(1, TT + 1), z_true[H:], linewidth=1.5)  # overlay true z_{t+H}
    plt.colorbar(label="density")
    plt.xlabel("conditioning time t")
    plt.ylabel(f"z (forecast at t+H, H={H})")
    plt.title(f"Quadrature latent forecast: p(z_(t+H) | x_1:t), H={H}")
    plt.tight_layout()
    plt.show()


    # ---- Plot x-forecast heatmap: p(x_{t+H} | x_{1:t}) ----
    plt.figure(figsize=(10, 4))
    plt.imshow(
        pxH_heat.T,
        origin="lower",
        aspect="auto",
        extent=[1, TT, xmin, xmax],
        interpolation="nearest",
    )
    plt.plot(np.arange(1, TT + 1), x_obs[H:], linewidth=1.5)  # overlay true x_{t+H}
    plt.colorbar(label="density")
    plt.xlabel("conditioning time t")
    plt.ylabel(f"x (forecast at t+H, H={H})")
    plt.title(f"Quadrature observation forecast: p(x_(t+H) | x_1:t), H={H}")
    plt.tight_layout()
    plt.show()

    # ============================================================
    # H-step-ahead marginal mass difference
    # Δ_t^(H) = P(z_{t+H} > 0 | x_{1:t}) - P(z_{t+H} < 0 | x_{1:t})
    # ============================================================

    mass_diff_H = np.zeros(TT)

    pos_mask = (z_grid >= 0)
    neg_mask = (z_grid < 0)

    for t in range(TT):
        pz = pzH_heat[t]

        m_pos = np.sum(pz[pos_mask]) * dz
        m_neg = np.sum(pz[neg_mask]) * dz

        mass_diff_H[t] = m_pos - m_neg


    # ------------------------------------------------------------
    # Plot H-step-ahead mass difference
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 3))

    plt.plot(np.arange(1, TT + 1), mass_diff_H, linewidth=2)
    plt.axhline(0.0, linestyle='--', linewidth=1)

    plt.ylim([-1.05, 1.05])
    plt.xlabel("conditioning time t")
    plt.ylabel(f"Δ^(H) (H={H})")
    plt.title(f"H-step-ahead marginal mass difference: p(z_(t+H)|x_1:t)")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 3C) Heatmap 2: p(x_{t+H} | x_{1:t})
    #
    # For each conditioning time t (1..T-H), start from filtering density:
    #   p(z_t | x_{1:t})  ≈ p_filt[t-1]   (since p_filt is stored for z_{t})
    # Then push forward H steps with the transition kernel (no updates):
    #   p(z_{t+H} | x_{1:t}) = (K^H) applied to p(z_t | x_{1:t})
    # Then integrate emission:
    #   p(x | x_{1:t}) = ∫ N(x; mu_x(z), sigma_x^2) p(z_{t+H}|x_{1:t}) dz
    # ------------------------------------------------------------
    # TT = T - H
    # px_heat = np.zeros((TT, Gx), dtype=float)

    # # precompute Gaussian emission pdf on the x-grid for each z_grid point:
    # # G_emit[i,j] = N(x_grid[i]; mu_x(z_j), var_x)
    # # This is (Gx,G). Memory is fine here: 600*800 ~ 480k floats.
    # X = x_grid[:, None]
    # MU = mu_x_grid[None, :]
    # G_emit = (1.0 / np.sqrt(2.0 * np.pi * var_x)) * np.exp(-0.5 * ((X - MU) ** 2) / var_x)

    # for t in range(1, TT + 1):  # math t = 1..T-H
    #     # start from filtering at time t: p(z_t | x_{1:t})
    #     pz = p_filt[t - 1].copy()

    #     # propagate H steps: p(z_{t+H} | x_{1:t})
    #     for _ in range(H):
    #         pz = K @ (pz * dz)
    #         pz = pz / (np.sum(pz) * dz + 1e-12)

    #     # emission mixture integral:
    #     # p(x) = ∫ N(x; mu_x(z), var_x) pz(z) dz  ≈ sum_j G_emit[:,j] * pz[j] * dz
    #     px = G_emit @ (pz * dz)
    #     px = px / (np.sum(px) * dx + 1e-12)
    #     px_heat[t - 1] = px

    # # plot heatmap over conditioning time t, y-axis is x value
    # plt.figure(figsize=(10, 4))
    # plt.imshow(
    #     px_heat.T,
    #     origin="lower",
    #     aspect="auto",
    #     extent=[1, TT, xmin, xmax],
    #     interpolation="nearest",
    # )
    # # overlay the realized target x_{t+H}
    # x_target = x_obs[H:]  # length T-H; x_obs[t+H-1] for t=1..T-H
    # plt.plot(np.arange(1, TT + 1), x_target, linewidth=1.5)
    # plt.colorbar(label="density")
    # plt.xlabel("conditioning time t")
    # plt.ylabel(f"x (forecast at t+H, H={H})")
    # plt.title(f"Quadrature forecast heatmap: p(x_(t+H) | x_1:t),  H={H}")
    # plt.tight_layout()
    # plt.show()

print("done")