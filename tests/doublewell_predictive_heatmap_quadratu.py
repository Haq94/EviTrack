# doublewell_predictive_heatmap_quadrature.py
# Deterministic quadrature (grid) approximation of predictive p(z_t | x_<t)
# for your double-well + piecewise emission model.
#
# This replaces the PF part of doublewell_predictive_heatmap_script_fixed.py :contentReference[oaicite:1]{index=1}

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0) SETTINGS (match your script)
# ============================================================
SEEDS = [n for n in range(100)]   # lower this if you don't want many plots
T = 120

# ---- Transition params ----
a = 3.0
V = 0.06
sigma_z = 0.05

# ---- Emission params ----
n = 2
d = 2.0
sigma_x = 0.12

# ---- Prior ----
z1_std = 1.0

# ---- Quadrature grid ----
zmin, zmax = -4.0, 4.0
G = 800                      # increase for sharper accuracy; 400–2000 are typical
# ============================================================


# ============================================================
# 1) MODEL FUNCTIONS (same as your code)
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
# 2) QUADRATURE RECURSION SETUP
# ============================================================

# grid and spacing
z_grid = np.linspace(zmin, zmax, G)
dz = z_grid[1] - z_grid[0]

# prior density on grid: p(z1)
# p0[j] ≈ p(z_grid[j])
p_pred0 = (1.0 / (np.sqrt(2.0 * np.pi) * z1_std)) * np.exp(-0.5 * (z_grid / z1_std) ** 2)
p_pred0 = p_pred0 / (np.sum(p_pred0) * dz + 1e-12)

# transition kernel K[i,j] = p(z_{t+1}=z_i | z_t=z_j)
# where p(·|z_j) is Gaussian N(mean=transition_mean(z_j), var=sigma_z^2)
mu_zj = transition_mean(z_grid)          # shape (G,)
var_z = transition_cov(z_grid)           # scalar
# Broadcast to (G,G): rows are z_i, cols are z_j
diff = z_grid[:, None] - mu_zj[None, :]
K = (1.0 / np.sqrt(2.0 * np.pi * var_z)) * np.exp(-0.5 * (diff ** 2) / var_z)
# Note: K columns integrate to ~1 over z_i only if grid covers enough mass; we renormalize after predict.


# ============================================================
# 3) MAIN SCRIPT (per seed): generate data, run quadrature, plot heatmap
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

    # ---- Quadrature recursion to get predictive densities p(z_t | x_<t) ----
    # store p_pred[t] as vector over grid (density)
    p_pred = np.zeros((T, G), dtype=float)

    # t=1 predictive (before seeing x1): p(z1 | x_<1) = p(z1)
    p_pred[0] = p_pred0.copy()

    # iterate t = 1..T-1 (math time)
    # at loop step t (0-index):
    #  - we have p_pred[t] = p(z_{t+1} | x_<t+1)
    #  - update with x_{t+1} to get p_filt(z_{t+1} | x_<=t+1)
    #  - predict to get p_pred[t+1] = p(z_{t+2} | x_<t+2)
    for t in range(T - 1):
        # Update with x_{t+1} = x_obs[t]
        xt = x_obs[t]

        mu_x = emission_mean(z_grid)
        var_x = emission_cov(z_grid)

        # likelihood on grid: L[j] = p(xt | z_grid[j])
        # Use log then exp for stability
        logL = -0.5 * ((xt - mu_x) ** 2) / var_x - 0.5 * np.log(2.0 * np.pi * var_x)
        # stabilize
        logL = logL - np.max(logL)
        L = np.exp(logL)

        # filtering (unnormalized): p_filt ∝ L * p_pred
        p_filt = L * p_pred[t]
        p_filt = p_filt / (np.sum(p_filt) * dz + 1e-12)

        # Predict: p_next[i] = ∫ p(z_{t+2}=z_i | z_{t+1}=z) p_filt(z) dz
        # Discrete: p_next = K @ (p_filt * dz)
        p_next = K @ (p_filt * dz)

        # renormalize (fix truncation from finite grid)
        p_next = p_next / (np.sum(p_next) * dz + 1e-12)

        p_pred[t + 1] = p_next

    # ---- Heatmap plot of p(z_t | x_<t) ----
    # p_pred[t-1] corresponds to time t (1-indexed); we plot columns 1..T with p_pred rows 0..T-1.
    plt.figure(figsize=(10, 4))
    plt.imshow(
        p_pred.T,  # (G,T)
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

print("done")