# tests/delayed_disambiguation_synthetic_script.py

import numpy as np
import matplotlib.pyplot as plt

from data.synthetic_generator import (
    GaussianPrior, GaussianTransition, GaussianEmission, generate_sequences
)

# ----------------------------
# 1) Stationary nonlinear Gaussian-conditionals
# ----------------------------
def f(z):
    """
    Nonlinear drift with two "basins" (dz can be 1+).
    z: (B, dz)
    """
    a = 0.92
    b = 0.55
    return a * z + b * np.tanh(3.0 * z)

def g(z):
    """
    Many-to-one-ish emission to induce ambiguity.
    Works for dz=1 and dz>=2.

    Output dx=2 always:
      x1 = z1^2 + c*z1  (even-ish, sign ambiguous early)
      x2 = z1           (small linear leak can help eventual resolution; set c small)
    """
    c = 0.02
    z1 = z[:, :1]                # (B,1), always valid
    x1 = z1**2 + c * z1
    x2 = z1                      # keep 2D observation even for dz=1
    return np.concatenate([x1, x2], axis=1)  # (B,2)

def trans_mean(z_prev, extras):
    return f(z_prev)

def trans_cov(z_prev, extras):
    dz = z_prev.shape[-1]
    q = float(extras.get("process_noise", 0.15))
    return (q**2) * np.eye(dz)

def emit_mean(z, extras):
    return g(z)

def emit_cov(z, extras):
    r = float(extras.get("emit_noise", 0.10))
    # infer dx from mean_fn output (avoids brittle extras["dx"])
    dx = int(emit_mean(z, extras).shape[-1])
    return (r**2) * np.eye(dx)

# ----------------------------
# 2) Oracle PF: approximate p(z_t | x_{1:t}) and predictive p(z_{t+1} | x_{1:t})
# ----------------------------
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

def oracle_pf(x_seq, prior, transition, emission, *, N=20000, seed=0, extras=None, resample_every=1):
    """
    x_seq: (T, dx)
    Returns:
      z_filt[t] ~ p(z_t | x_{1:t}) via particles after weighting (before resample)
      z_pred[t] ~ p(z_{t+1} | x_{1:t}) via particles (for t=0..T-2)
      ess[t]    ESS after weighting with x_t
    """
    rng = np.random.default_rng(seed)
    extras = dict(extras or {})

    T, dx = x_seq.shape
    dz = int(np.asarray(prior.mu0).shape[0])

    # init particles from prior
    z = rng.multivariate_normal(mean=np.asarray(prior.mu0), cov=np.asarray(prior.cov0), size=N)  # (N,dz)
    w = np.ones(N, dtype=float) / N

    z_filt = []
    z_pred = []
    ess = []

    for t in range(T):
        # weight by likelihood p(x_t | z_t)
        mu_x = np.asarray(emission.mean_fn(z, extras), dtype=float)             # (N,dx)
        cov_x = np.asarray(emission.cov_fn(z, extras), dtype=float)             # (dx,dx) or (N,dx,dx)
        xt_rep = np.repeat(x_seq[t:t+1], N, axis=0)                             # (N,dx)

        logw = log_gaussian_full(xt_rep, mu_x, cov_x)
        logw = logw - logw.max()
        w = w * np.exp(logw)
        w = w / (w.sum() + 1e-12)

        ess_t = 1.0 / np.sum(w**2)
        ess.append(ess_t)

        # store weighted particle cloud for p(z_t | x_{1:t})
        z_filt.append(z.copy())

        # resample (optionally less frequent)
        if resample_every > 0 and ((t + 1) % resample_every == 0):
            idx = systematic_resample(w, rng)
            z = z[idx]
            w = np.ones(N, dtype=float) / N

        # predictive for next step: z_{t+1} ~ p(z_{t+1} | z_t)
        if t < T - 1:
            mu_z = np.asarray(transition.mean_fn(z, extras), dtype=float)        # (N,dz)
            cov_z = np.asarray(transition.cov_fn(z, extras), dtype=float)        # (dz,dz) or (N,dz,dz)
            if cov_z.ndim == 2:
                cov_z = np.broadcast_to(cov_z, (N, dz, dz))

            z_next = np.empty((N, dz), dtype=float)
            for i in range(N):
                z_next[i] = rng.multivariate_normal(mean=mu_z[i], cov=cov_z[i])

            z_pred.append(z_next)
            z = z_next

    return z_filt, z_pred, np.asarray(ess, dtype=float)

# ----------------------------
# 3) Simple multimodality diagnostics (dz=1 focus)
# ----------------------------
def bimodality_kurtosis_proxy(z_particles_1d):
    """
    z_particles_1d: (N,) samples
    Returns a crude proxy: -excess_kurtosis (bigger tends to indicate heavier tails / bimodal-ish)
    """
    z = z_particles_1d
    m = z.mean()
    v = ((z - m) ** 2).mean() + 1e-12
    k = ((z - m) ** 4).mean() / (v ** 2)
    excess = k - 3.0
    return -excess

def two_cluster_balance_and_gap(z_particles_1d):
    """
    Very cheap 2-cluster diagnostic: split by sign around median.
    Returns:
      balance in [0,0.5], gap (mean difference between halves)
    """
    z = np.sort(z_particles_1d)
    N = len(z)
    left = z[: N // 2]
    right = z[N // 2 :]
    # balance is always ~0.5 by construction; useful mainly for the gap
    gap = float(right.mean() - left.mean())
    return 0.5, gap

# ----------------------------
# 4) Run
# ----------------------------
if __name__ == "__main__":
    # You can start with dz=1 safely now.
    T = 80
    dz = 1
    seed = 0

    extras = {
        "process_noise": 0.18,
        "emit_noise": 0.10,
    }

    prior = GaussianPrior(mu0=np.zeros(dz), cov0=np.eye(dz))
    transition = GaussianTransition(mean_fn=trans_mean, cov_fn=trans_cov)
    emission = GaussianEmission(mean_fn=emit_mean, cov_fn=emit_cov)

    # generate ONE sequence (B=1)
    batch = generate_sequences(
        T=T, B=1,
        prior=prior, transition=transition, emission=emission,
        seed=seed, return_logp=False,
        extras=extras
    )
    x = batch.x[0]  # (T, dx) where dx=2 from g(z)

    # oracle PF
    z_filt, z_pred, ess = oracle_pf(
        x, prior, transition, emission,
        N=30000, seed=123,
        extras=extras,
        resample_every=1
    )

    # Diagnostics on predictive p(z_t | x_{<t}) using z_pred[t-1] for t>=1
    M_kurt = []
    M_gap = []
    for t in range(1, T):
        zp = z_pred[t - 1][:, 0]  # (N,) since dz=1
        M_kurt.append(bimodality_kurtosis_proxy(zp))
        _, gap = two_cluster_balance_and_gap(zp)
        M_gap.append(gap)

    tgrid = np.arange(1, T)

    plt.figure()
    plt.plot(tgrid, M_kurt)
    plt.xlabel("t")
    plt.ylabel("-excess kurtosis (predictive z)")
    plt.title("Predictive multimodality proxy vs time (p(z_t | x_{<t}))")
    plt.grid(True)

    plt.figure()
    plt.plot(tgrid, M_gap)
    plt.xlabel("t")
    plt.ylabel("2-cluster gap (predictive z)")
    plt.title("Predictive cluster-separation proxy vs time (p(z_t | x_{<t}))")
    plt.grid(True)

    plt.figure()
    plt.plot(np.arange(1, T + 1), ess)
    plt.xlabel("t")
    plt.ylabel("ESS after weighting with x_t")
    plt.title("Evidence shock proxy (ESS) vs time")
    plt.grid(True)

    plt.show()