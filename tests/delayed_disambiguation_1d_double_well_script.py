# tests/delayed_disambiguation_1d_double_well_script.py

import numpy as np
import matplotlib.pyplot as plt

from data.synthetic_generator import (
    GaussianPrior, GaussianTransition, GaussianEmission, generate_sequences
)

# ============================================================
# 1) Stationary synthetic: 1D double-well + sign-reveal emission
# ============================================================

def trans_mean(z_prev, extras):
    """
    Double-well drift in 1D:
        z_t = z_{t-1} + a z_{t-1} - b z_{t-1}^3 + eps
    This creates two stable basins (around +/- sqrt(a/b) if a,b>0),
    yielding two plausible latent hypotheses early on.
    """
    a = float(extras.get("a", 0.10))
    b = float(extras.get("b", 0.10))
    return z_prev + a * z_prev - b * (z_prev ** 3)

def trans_cov(z_prev, extras):
    q = float(extras.get("process_noise", 0.12))
    return (q ** 2) * np.eye(1)

def emit_mean(z, extras):
    """
    2D emission (dx=2), stationary and nonlinear:
      x1 = z^2 + noise  (many-to-one: z and -z map similarly)
      x2 = tanh(k z) + noise  (near 0: almost linear + small SNR; away from 0: saturates to +/-1)
    By tuning k and noise, sign is ambiguous early (two modes),
    then disambiguates later (one mode survives).
    """
    k = float(extras.get("k", 2.5))
    z1 = z[:, :1]  # (B,1)
    x1 = z1 ** 2
    # x2
    eps, n = extras["eps"], extras["n"]
    inside = (np.abs(z1) < eps)         # (N,1) bool
    x2_inside = (z1 / eps) ** (2 * n)   # (N,1)
    x2_outside = z1                     # (N,1)
    x2 = np.where(inside, x2_inside, x2_outside)
    return np.concatenate([x1, x2], axis=1)  # (B,2)

def emit_cov(z, extras):
    r = float(extras.get("emit_noise", 0.25))
    return (r ** 2) * np.eye(2)

# ============================================================
# 2) Oracle/bootstrap particle filter
#    Approximates:
#      - predictive p(z_t | x_{1:t-1}) via z_pred[t-1]
#      - filtering  p(z_t | x_{1:t})   via z_filt[t]
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

def oracle_pf(x_seq, prior, transition, emission, *, N=30000, seed=0, extras=None, resample_every=1):
    """
    x_seq: (T, dx)

    Returns:
      z_filt[t] ~ p(z_t | x_{1:t})      (particle cloud AFTER weighting, BEFORE resample)
      z_pred[t] ~ p(z_{t+1} | x_{1:t})  (predictive cloud), for t=0..T-2
      ess[t]    = 1 / sum_i w_i^2       after weighting with x_t
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
        mu_x = np.asarray(emission.mean_fn(z, extras), dtype=float)  # (N,dx)
        cov_x = np.asarray(emission.cov_fn(z, extras), dtype=float)  # (dx,dx) or (N,dx,dx)
        xt_rep = np.repeat(x_seq[t:t+1], N, axis=0)                  # (N,dx)

        logw = log_gaussian_full(xt_rep, mu_x, cov_x)
        logw = logw - logw.max()
        w = w * np.exp(logw)
        w = w / (w.sum() + 1e-12)

        ess_t = 1.0 / np.sum(w ** 2)
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
            mu_z = np.asarray(transition.mean_fn(z, extras), dtype=float)  # (N,dz)
            cov_z = np.asarray(transition.cov_fn(z, extras), dtype=float)  # (dz,dz) or (N,dz,dz)
            if cov_z.ndim == 2:
                cov_z = np.broadcast_to(cov_z, (N, dz, dz))

            z_next = np.empty((N, dz), dtype=float)
            for i in range(N):
                z_next[i] = rng.multivariate_normal(mean=mu_z[i], cov=cov_z[i])

            z_pred.append(z_next)
            z = z_next

    return z_filt, z_pred, np.asarray(ess, dtype=float)

# ============================================================
# 3) Diagnostics
#    (a) simple proxies: kurtosis, two-halves gap
#    (b) Hartigan dip statistic + bootstrap p-value (approx)
#    (c) KDE mode count + critical bandwidth + bootstrap p-value (Silverman-style)
# ============================================================

def bimodality_kurtosis_proxy(z_1d):
    """
    Proxy: -excess kurtosis.
    Gaussian => ~0.
    Often increases when distribution becomes flatter/heavier-tailed or bimodal-ish.
    Not a proof; use as a cheap curve.
    """
    z = np.asarray(z_1d, dtype=float)
    m = z.mean()
    v = ((z - m) ** 2).mean() + 1e-12
    k = ((z - m) ** 4).mean() / (v ** 2)
    excess = k - 3.0
    return -excess

def two_halves_gap(z_1d):
    """
    Split sorted samples into two halves and compute mean gap.
    For a symmetric bimodal distribution, this tends to be large.
    For unimodal concentrated, this shrinks.
    """
    z = np.sort(np.asarray(z_1d, dtype=float))
    N = len(z)
    left = z[: N // 2]
    right = z[N // 2 :]
    return float(right.mean() - left.mean())

# ----------------------------
# 3b) Hartigan dip statistic (1D) + bootstrap p-value
# ----------------------------
# This is a compact implementation for practical use (not micro-optimized).
# It returns the dip statistic; p-value is computed by bootstrap under a unimodal null
# (we use normal null with matched mean/var for practicality).

def _dip_statistic_1d(x):
    """
    Compute Hartigan's dip statistic for 1D samples.
    Returns dip (float). Works on sorted unique-ish values.
    Implementation based on standard dip algorithm structure.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n < 3:
        return 0.0

    # If all equal, dip=0
    if np.allclose(x[0], x[-1]):
        return 0.0

    # Helper arrays
    mn = np.zeros(n, dtype=int)
    mj = np.zeros(n, dtype=int)

    # "GCM" and "LCM" helpers (greatest convex minorant / least concave majorant)
    def gcm():
        # returns mn (contact points)
        mn[:] = 0
        mn[0] = 0
        for j in range(1, n):
            mn[j] = j - 1
            while mn[j] > 0:
                j1 = mn[j]
                j2 = mn[j1]
                # slope(j2->j1) <= slope(j1->j) ?
                if (x[j1] - x[j2]) * (j - j1) <= (x[j] - x[j1]) * (j1 - j2):
                    mn[j] = j2
                else:
                    break

    def lcm():
        mj[:] = 0
        mj[n - 1] = n - 1
        for k in range(n - 2, -1, -1):
            mj[k] = k + 1
            while mj[k] < n - 1:
                k1 = mj[k]
                k2 = mj[k1]
                # slope(k->k1) >= slope(k1->k2) ?
                if (x[k1] - x[k]) * (k2 - k1) >= (x[k2] - x[k1]) * (k1 - k):
                    mj[k] = k2
                else:
                    break

    # Empirical CDF values
    F = (np.arange(1, n + 1) / n).astype(float)

    gcm()
    lcm()

    # Now compute dip as maximum distance between GCM and LCM on CDF scale.
    # This is a simplified practical computation (works well as a diagnostic).
    dip = 0.0
    # compute piecewise linear minorant/majorant via mn/mj pointers
    for i in range(n):
        # GCM segment from mn[i] to i
        i0 = mn[i]
        if i0 != i:
            slope_g = (F[i] - F[i0]) / (x[i] - x[i0] + 1e-12)
            # value at x[i] is F[i] by definition, so line matches endpoints
        # LCM segment from i to mj[i]
        j = mj[i]
        if j != i:
            slope_l = (F[j] - F[i]) / (x[j] - x[i] + 1e-12)

        # Evaluate local vertical gap at x[i]
        # (This is an approximation; exact dip uses iterative tightening.
        # For our use (time-series diagnostics), this is sufficient and stable.)
        # Here we use nearest convex/concave segment endpoints.
        # Gap approx:
        gap = 0.0
        if i0 != i and j != i:
            # compare the two lines at x[i] (both equal F[i]); use midpoints around i
            mid_left = 0.5 * (x[i0] + x[i])
            mid_right = 0.5 * (x[i] + x[j])

            # compute GCM line at mid_right using segment (i0,i)
            g_at_mid = F[i0] + slope_g * (mid_right - x[i0])
            # compute LCM line at mid_left using segment (i,j)
            l_at_mid = F[i] + slope_l * (mid_left - x[i])
            gap = max(0.0, l_at_mid - g_at_mid)

        dip = max(dip, gap)

    return float(dip)

def dip_test_pvalue(z_1d, *, B=200, seed=0):
    """
    Dip statistic + bootstrap p-value under unimodal normal null with matched mean/var.
    Returns: (dip, p_value)
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(z_1d, dtype=float)
    n = z.size
    dip_obs = _dip_statistic_1d(z)

    m = float(z.mean())
    s = float(z.std() + 1e-12)

    dips = []
    for _ in range(B):
        samp = rng.normal(loc=m, scale=s, size=n)
        dips.append(_dip_statistic_1d(samp))
    dips = np.asarray(dips)

    # p-value = fraction of null dips >= observed dip
    p = float((np.sum(dips >= dip_obs) + 1.0) / (B + 1.0))
    return dip_obs, p

# ----------------------------
# 3c) KDE mode count + "critical bandwidth" (Silverman-style) + bootstrap p-value
# ----------------------------

def gaussian_kde_1d(z, grid, h):
    """
    Simple Gaussian KDE evaluation on a grid.
    z: (N,)
    grid: (G,)
    h: bandwidth
    returns: (G,) density (unnormalized scale ok for mode counting)
    """
    z = z[:, None]           # (N,1)
    g = grid[None, :]        # (1,G)
    u = (g - z) / (h + 1e-12)
    # Gaussian kernel
    K = np.exp(-0.5 * u * u)
    dens = K.mean(axis=0) / (h + 1e-12)
    return dens

def count_modes_from_kde(grid, dens):
    """
    Count local maxima in a 1D density curve on a grid.
    """
    d = dens
    # local maxima: d[i] > d[i-1] and d[i] > d[i+1]
    peaks = np.where((d[1:-1] > d[:-2]) & (d[1:-1] > d[2:]))[0] + 1
    return int(len(peaks))

def critical_bandwidth_unimodal(z, *, grid=None, h_lo=None, h_hi=None, max_iter=30):
    """
    Find smallest bandwidth h such that KDE is unimodal on the grid.
    Returns h_crit.
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    if grid is None:
        lo = float(np.percentile(z, 0.5))
        hi = float(np.percentile(z, 99.5))
        pad = 0.25 * (hi - lo + 1e-9)
        grid = np.linspace(lo - pad, hi + pad, 512)

    # Silverman rule-of-thumb as a scale reference
    s = float(z.std() + 1e-12)
    h0 = 1.06 * s * (n ** (-1 / 5))

    if h_lo is None:
        h_lo = 0.05 * h0
    if h_hi is None:
        h_hi = 10.0 * h0

    # Ensure upper bound is unimodal
    for _ in range(10):
        dens = gaussian_kde_1d(z, grid, h_hi)
        if count_modes_from_kde(grid, dens) <= 1:
            break
        h_hi *= 2.0

    # Binary search for smallest h giving unimodal KDE
    lo, hi = float(h_lo), float(h_hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        dens = gaussian_kde_1d(z, grid, mid)
        if count_modes_from_kde(grid, dens) <= 1:
            hi = mid
        else:
            lo = mid
    return float(hi)

def silverman_style_pvalue(z_1d, *, B=200, seed=0):
    """
    "Silverman-style" test using critical bandwidth:
      - compute h_crit for observed data (small => strongly multimodal; large => more unimodal)
      - bootstrap under unimodal normal null (matched mean/var) to get p-value:
            p = P_null(h_crit_null >= h_crit_obs)
        (If observed needs a large bandwidth to become unimodal, it looks more multimodal.)

    Returns: (h_crit, p_value)

    Note: This is not the exact Silverman test (which bootstraps from a smoothed KDE null).
    This version is practical, stable, and good for your diagnostic curves.
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(z_1d, dtype=float)
    n = z.size

    grid = None  # let critical_bandwidth build grid from z each time for simplicity
    h_obs = critical_bandwidth_unimodal(z, grid=grid)

    m = float(z.mean())
    s = float(z.std() + 1e-12)

    hs = []
    for _ in range(B):
        samp = rng.normal(loc=m, scale=s, size=n)
        hs.append(critical_bandwidth_unimodal(samp, grid=grid))
    hs = np.asarray(hs)

    p = float((np.sum(hs >= h_obs) + 1.0) / (B + 1.0))
    return h_obs, p

# ============================================================
# 4) Heatmap: empirical density of predictive p(z_t | x_{1:t-1})
# ============================================================

def predictive_heatmap(z_pred, *, z_min=None, z_max=None, nbins=160, eps=1e-12):
    """
    z_pred: list length T-1, each element (N,1) particle samples for p(z_{t+1} | x_{1:t})
    We build a (T, nbins) matrix for t=1..T-1 representing p(z_t | x_{1:t-1}).

    Returns:
      H: (T-1, nbins) density estimates (rows sum to 1)
      centers: (nbins,) bin centers
    """
    # Determine histogram range
    allz = np.concatenate([zp[:, 0] for zp in z_pred], axis=0)
    if z_min is None:
        z_min = float(np.percentile(allz, 0.5))
    if z_max is None:
        z_max = float(np.percentile(allz, 99.5))
    pad = 0.20 * (z_max - z_min + 1e-9)
    z_min -= pad
    z_max += pad

    edges = np.linspace(z_min, z_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    H = np.zeros((len(z_pred), nbins), dtype=float)
    for t, zp in enumerate(z_pred):
        hist, _ = np.histogram(zp[:, 0], bins=edges, density=False)
        hist = hist.astype(float)
        hist = hist / (hist.sum() + eps)
        H[t] = hist

    return H, centers

# ============================================================
# 5) Run
# ============================================================

if __name__ == "__main__":
    # ---- settings ----
    T = 100
    dz = 1
    seed_data = 4555
    seed_pf = 4035

    alpha = 1.0
    V = 0.1

    a = 4*V/alpha**2
    b = 4*V/alpha**4

    extras = {
        # transition
        "a": a,
        "b": b,
        "process_noise": 0.2,
        # emission
        "n": 2,
        "eps": 0.4,
        "k": 2.5,
        "emit_noise": 0.1,
    }

    # ---- build generator ----
    prior = GaussianPrior(mu0=np.zeros(dz), cov0=np.eye(dz))
    transition = GaussianTransition(mean_fn=trans_mean, cov_fn=trans_cov)
    emission = GaussianEmission(mean_fn=emit_mean, cov_fn=emit_cov)

    # ---- generate ONE sequence (B=1) ----
    batch = generate_sequences(
        T=T, B=1,
        prior=prior, transition=transition, emission=emission,
        seed=seed_data, return_logp=False,
        extras=extras
    )
    x = batch.x[0]        # (T,2)
    z_true = batch.z[0]   # (T,1)

    # ---- oracle PF ----
    z_filt, z_pred, ess = oracle_pf(
        x, prior, transition, emission,
        N=1000, seed=seed_pf,
        extras=extras,
        resample_every=3
    )

    # ========================================================
    # A) Heatmap of predictive p(z_t | x_{1:t-1})
    #    Note: z_pred[t-1] ~ p(z_t | x_{1:t-1}) for t=1..T-1
    # ========================================================
    H, z_grid = predictive_heatmap(z_pred, nbins=180)

    plt.figure(figsize=(10, 4))
    # imshow expects (rows, cols): rows correspond to time steps 1..T-1
    plt.imshow(
        H,
        aspect="auto",
        origin="lower",
        extent=[z_grid[0], z_grid[-1], 1, T - 1],
        interpolation="nearest"
    )
    plt.colorbar(label="empirical density (histogram)")
    # overlay true z_t (for t=1..T-1)
    plt.plot(z_true[1:, 0], np.arange(1, T), linewidth=2)
    plt.xlabel("z")
    plt.ylabel("t")
    plt.title("Heatmap of predictive p(z_t | x_{1:t-1}) (particles) with true z_t overlay")
    plt.grid(False)

    # ========================================================
    # B) Time-series diagnostics on predictive particles
    # ========================================================
    tgrid = np.arange(1, T)

    kurt_proxy = []
    gap_proxy = []
    dip_stat = []
    dip_p = []
    hcrit = []
    hcrit_p = []

    # to keep runtime reasonable, subsample particles for tests
    # (heatmap uses all; tests can use fewer)
    rng = np.random.default_rng(0)
    test_N = 4000

    for t in range(1, T):
        zp_full = z_pred[t - 1][:, 0]  # predictive samples for z_t | x_{<t}
        if zp_full.size > test_N:
            idx = rng.choice(zp_full.size, size=test_N, replace=False)
            zp = zp_full[idx]
        else:
            zp = zp_full

        kurt_proxy.append(bimodality_kurtosis_proxy(zp))
        gap_proxy.append(two_halves_gap(zp))

        d, pval = dip_test_pvalue(zp, B=200, seed=1000 + t)
        dip_stat.append(d)
        dip_p.append(pval)

        hc, pval2 = silverman_style_pvalue(zp, B=120, seed=2000 + t)
        hcrit.append(hc)
        hcrit_p.append(pval2)

    kurt_proxy = np.asarray(kurt_proxy)
    gap_proxy = np.asarray(gap_proxy)
    dip_stat = np.asarray(dip_stat)
    dip_p = np.asarray(dip_p)
    hcrit = np.asarray(hcrit)
    hcrit_p = np.asarray(hcrit_p)

    plt.figure(figsize=(10, 3))
    plt.plot(tgrid, kurt_proxy)
    plt.xlabel("t")
    plt.ylabel("-excess kurtosis")
    plt.title("Predictive shape proxy (higher often = more bimodal/heavy-tailed)")
    plt.grid(True)

    plt.figure(figsize=(10, 3))
    plt.plot(tgrid, gap_proxy)
    plt.xlabel("t")
    plt.ylabel("two-halves gap")
    plt.title("Predictive two-cluster separation proxy (bigger = more separated mass)")
    plt.grid(True)

    plt.figure(figsize=(10, 3))
    plt.plot(tgrid, dip_stat, label="dip statistic")
    plt.xlabel("t")
    plt.ylabel("dip")
    plt.title("Hartigan dip statistic on predictive particles (bigger = less unimodal)")
    plt.grid(True)

    plt.figure(figsize=(10, 3))
    plt.plot(tgrid, dip_p)
    plt.axhline(0.05, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("p-value")
    plt.title("Dip test bootstrap p-value (below 0.05 => reject unimodality, approx)")
    plt.grid(True)

    plt.figure(figsize=(10, 3))
    plt.plot(tgrid, hcrit)
    plt.xlabel("t")
    plt.ylabel("critical bandwidth")
    plt.title("KDE critical bandwidth for unimodality (larger = more multimodal structure)")
    plt.grid(True)

    plt.figure(figsize=(10, 3))
    plt.plot(tgrid, hcrit_p)
    plt.axhline(0.05, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("p-value")
    plt.title("Critical-bandwidth bootstrap p-value (below 0.05 => evidence of multimodality, approx)")
    plt.grid(True)

    # ========================================================
    # C) ESS curve (evidence shock proxy)
    # ========================================================
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(1, T + 1), ess)
    plt.xlabel("t")
    plt.ylabel("ESS after weighting with x_t")
    plt.title("ESS curve (drops indicate evidence concentrates weight on few hypotheses)")
    plt.grid(True)

    # ========================================================
    # D) Observations (sanity)
    # ========================================================
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(1, T + 1), x[:, 0], label="x1 = z^2 + noise")
    plt.plot(np.arange(1, T + 1), x[:, 1], label="x2 = tanh(k z) + noise")
    plt.xlabel("t")
    plt.title("Observation channels")
    plt.grid(True)
    plt.legend()

    plt.show()