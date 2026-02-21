# doublewell_predictive_heatmap_script_fixed.py
# Heatmap of predictive p(z_t | x_<t)
# Double-well transition with U(0)=V exactly, minima at ±a
# Piecewise emission: z^(2n) for |z|<=d, else z
#
# in the drift discretization: z_t = z_{t-1} - dU/dz + sigma_z * eps.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0) SETTINGS (edit here)
# ============================================================
SEEDS = [n for n in range(100)]
T = 120
N = 100                 # PF particles
RESAMPLE_EVERY = 1         # 1 = every step, 0 = never

# ---- Transition params ----
a = 3.0                    # attractor locations ±a
V = 0.06                   # barrier height; U(0)=V
sigma_z = 0.05             # transition noise std

# ---- Emission params ----
n = 2
d = 2.0
sigma_x = 0.12             # emission noise std

# ---- Prior ----
z1_std = 1.0               # z1 ~ N(0, z1_std^2)

# ---- Heatmap grid ----
zmin, zmax = -4.0, 4.0
G = 400
# ============================================================


# ============================================================
# 1) MODEL FUNCTIONS
# ============================================================

def transition_mean(z):
    """
    Double-well potential:
      U(z) = V/a^4 * (z^2 - a^2)^2
    so U(0) = V, minima at ±a with U(±a)=0.

    Drift discretization (overdamped Langevin, Euler):
      mean = z - dU/dz
    """
    # dU/dz = (4V/a^4) * z * (z^2 - a^2)
    dU = (4.0 * V / (a ** 4)) * z * (z ** 2 - a ** 2)
    return z - dU


def transition_cov(z):
    # constant variance for now (scalar since dz=1)
    return sigma_z ** 2


def emission_mean(z):
    """
    mu_x(z) = z^(2n)  for |z| <= d
              z       for |z| >  d
    """
    z = np.asarray(z)
    inside = (np.abs(z) <= d)
    mu = np.empty_like(z, dtype=float)
    mu[inside] = z[inside] ** (2 * n)
    mu[~inside] = z[~inside]
    return mu


def emission_cov(z):
    # constant variance for now (scalar since dx=1)
    return sigma_x ** 2


# ============================================================
# 2) PARTICLE FILTER: predictive p(z_t | x_<t)
# ============================================================

def particle_filter_predictive(x_obs, rng, N, resample_every):
    """
    Returns pred_cloud list of length T where:
      pred_cloud[t-1] ~ samples from p(z_t | x_<t)   (t = 1..T)
    i.e.
      pred_cloud[0]  = p(z1 | x_<1) = p(z1)
      pred_cloud[1]  = p(z2 | x_<=1) = p(z2 | x_<2)
      ...
    """
    T = len(x_obs)

    # ---- initialize z1 particles ~ p(z1) ----
    z_particles = rng.normal(0.0, z1_std, size=N)
    w = np.full(N, 1.0 / N)

    pred_cloud = [z_particles.copy()]  # p(z1 | x_<1)

    for t in range(1, T):  # will construct p(z_{t+1} | x_<=t) each loop
        # ---- weight using the newest observed x_t (math time t) ----
        # at loop index t, we incorporate x_obs[t-1] to get p(z_t | x_<=t-1),
        # then propagate to z_{t+1}. BUT we want p(z_{t} | x_<t) samples saved
        # already as pred_cloud[t-1]. So here we:
        #   1) weight by x_obs[t-1] (turn p(z_t | x_<t) -> p(z_t | x_<=t))
        #   2) resample (optional)
        #   3) propagate to get p(z_{t+1} | x_<=t) = p(z_{t+1} | x_<t+1)
        #
        # For t=1: weight by x_obs[0], propagate to get z2|x1, save as pred_cloud[1].

        x_new = x_obs[t - 1]
        mu_x = emission_mean(z_particles)
        var_x = emission_cov(z_particles)

        diff = x_new - mu_x
        logw = -0.5 * (diff ** 2) / var_x - 0.5 * np.log(2.0 * np.pi * var_x)

        # stable normalize
        logw = logw - np.max(logw)
        w = w * np.exp(logw)
        w_sum = np.sum(w) + 1e-12
        w = w / w_sum

        # ---- resample ----
        if resample_every > 0 and (t % resample_every == 0):
            cdf = np.cumsum(w)
            u0 = rng.random() / N
            us = u0 + np.arange(N) / N
            idx = np.searchsorted(cdf, us)
            z_particles = z_particles[idx]
            w.fill(1.0 / N)

        # ---- propagate to next latent ----
        mu_z = transition_mean(z_particles)
        var_z = transition_cov(z_particles)
        z_particles = mu_z + rng.normal(0.0, np.sqrt(var_z), size=N)

        # now z_particles ~ p(z_{t+1} | x_<=t) = p(z_{t+1} | x_<t+1)
        pred_cloud.append(z_particles.copy())

    # pred_cloud has length T; pred_cloud[t-1] = p(z_t | x_<t)
    return pred_cloud


# ============================================================
# 3) MAIN SCRIPT
# ============================================================

z_edges = np.linspace(zmin, zmax, G + 1)
binw = (zmax - zmin) / G

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    # ---- Generate a single true trajectory + observations ----
    z_true = np.zeros(T, dtype=float)
    x_obs = np.zeros(T, dtype=float)

    # z1
    z_true[0] = rng.normal(0.0, z1_std)
    x_obs[0] = emission_mean(z_true[0]) + rng.normal(0.0, sigma_x)

    for t in range(1, T):
        z_true[t] = transition_mean(z_true[t - 1]) + rng.normal(0.0, sigma_z)
        x_obs[t] = emission_mean(z_true[t]) + rng.normal(0.0, sigma_x)

    # ---- PF predictive clouds ----
    pred_cloud = particle_filter_predictive(x_obs, rng, N, RESAMPLE_EVERY)

    # ---- Heatmap for p(z_t | x_<t) ----
    H = np.zeros((T, G), dtype=float)
    for t in range(1, T + 1):  # math time
        cloud = pred_cloud[t - 1]  # p(z_t | x_<t)
        counts, _ = np.histogram(cloud, bins=z_edges, density=False)
        H[t - 1] = counts / (np.sum(counts) * binw + 1e-12)  # density

    # ---- Plot ----
    plt.figure(figsize=(10, 4))
    plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[1, T, zmin, zmax],
        interpolation="nearest",
    )
    plt.plot(np.arange(1, T + 1), z_true, linewidth=1.5)
    plt.colorbar(label="density")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.title("Heatmap of predictive p(z_t | x_<t)")
    plt.tight_layout()
    plt.show()

print("done")