import os
os.makedirs("paper_figures", exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# SETTINGS
# ============================================================
SEED = 46
threshold = 0.8

T = 200
a = 3.0
V = 0.06
sigma_z = 0.05
n = 1
d = 2.0
sigma_x = 0.12
z1_std = 1.0
zmin, zmax = -4.0, 4.0
G = 1000
SHOW_TITLE = False

# ============================================================
# MODEL FUNCTIONS
# ============================================================
def transition_mean(z):
    dU = (4.0 * V / (a ** 4)) * z * (z ** 2 - a ** 2)
    return z - dU

def emission_mean(z):
    z = np.asarray(z)
    inside = (np.abs(z) <= d)
    mu = np.empty_like(z, dtype=float)
    mu[inside] = z[inside] ** (2 * n)
    mu[~inside] = z[~inside]
    return mu

# ============================================================
# QUADRATURE SETUP
# ============================================================
z_grid = np.linspace(zmin, zmax, G)
dz = z_grid[1] - z_grid[0]

p_pred0 = (1.0 / (np.sqrt(2.0 * np.pi) * z1_std)) * np.exp(-0.5 * (z_grid / z1_std) ** 2)
p_pred0 = p_pred0 / (np.sum(p_pred0) * dz + 1e-12)

mu_zj = transition_mean(z_grid)
var_z = sigma_z ** 2
diff = z_grid[:, None] - mu_zj[None, :]
K = (1.0 / np.sqrt(2.0 * np.pi * var_z)) * np.exp(-0.5 * (diff ** 2) / var_z)

mu_x_grid = emission_mean(z_grid)
var_x = sigma_x ** 2

# ============================================================
# GENERATE TRAJECTORY
# ============================================================
rng = np.random.default_rng(SEED)

z_true = np.zeros(T)
x_obs = np.zeros(T)
z_true[0] = rng.normal(0.0, z1_std)
x_obs[0] = emission_mean(z_true[0]) + rng.normal(0.0, sigma_x)
for t in range(1, T):
    z_true[t] = transition_mean(z_true[t - 1]) + rng.normal(0.0, sigma_z)
    x_obs[t] = emission_mean(z_true[t]) + rng.normal(0.0, sigma_x)

# ============================================================
# QUADRATURE RECURSION
# ============================================================
p_pred = np.zeros((T, G))
p_pred[0] = p_pred0.copy()

for t in range(T):
    xt = x_obs[t]
    logL = -0.5 * ((xt - mu_x_grid) ** 2) / var_x - 0.5 * np.log(2.0 * np.pi * var_x)
    logL -= np.max(logL)
    L = np.exp(logL)

    pf = L * p_pred[t]
    pf = pf / (np.sum(pf) * dz + 1e-12)

    if t < T - 1:
        pn = K @ (pf * dz)
        pn = pn / (np.sum(pn) * dz + 1e-12)
        p_pred[t + 1] = pn

# ============================================================
# DISAMBIGUATION TIME
# ============================================================
pos_mask = z_grid >= 0
neg_mask = z_grid < 0
mass_diff = np.zeros(T)
for t in range(T):
    pt = p_pred[t]
    mass_diff[t] = np.sum(pt[pos_mask]) * dz - np.sum(pt[neg_mask]) * dz

idx = np.where(np.abs(mass_diff) > threshold)[0]
disamb_time = int(idx[0]) + 1 if len(idx) > 0 else None

# ============================================================
# PLOT
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

fig, ax = plt.subplots(figsize=(7, 3.2))

im = ax.imshow(
    p_pred.T,
    origin="lower",
    aspect="auto",
    extent=[1, T, zmin, zmax],
    interpolation="bilinear",
    cmap="Blues",
    vmin=0,
    vmax=4,
)

# True trajectory
ax.plot(np.arange(1, T + 1), z_true, color="black", linewidth=1.2,
        alpha=0.9, label=r"$z_t^*$")

# Well minima at ±a — red dashed
for z_well in [a, -a]:
    ax.axhline(z_well, color="#e74c3c", linewidth=1.0, linestyle="--",
               alpha=0.85, label=r"$\pm a$ (well minima)" if z_well == a else None)

# Emission boundary at ±d — dark dotted
ax.axhline( d, color="#444444", linewidth=0.9, linestyle=":",
            label=r"$\pm d$ (emission boundary)")
ax.axhline(-d, color="#444444", linewidth=0.9, linestyle=":")

# Disambiguation time — vertical line + direct annotation
if disamb_time is not None:
    ax.axvline(disamb_time, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.text(disamb_time + 1.5, zmax - 0.3, rf"$t_{{DD}}={disamb_time}$",
            fontsize=9, va="top", ha="left", color="black")

cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("density", fontsize=10)
cbar.ax.tick_params(labelsize=9)
cbar.outline.set_linewidth(0.5)

ax.set_xlim(1, T)
ax.set_ylim(zmin, zmax)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$z$")
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

ax.legend(
    loc="upper right",
    fontsize=9,
    framealpha=0.6,
    edgecolor="gray",
    handlelength=1.5,
    borderpad=0.5,
)

if SHOW_TITLE:
    ax.set_title(rf"$p(z_t \mid x_{{<t}})$ — seed {SEED}")

plt.tight_layout()
plt.savefig(f"paper_figures/predictive_heatmap_seed{SEED}.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"paper_figures/predictive_heatmap_seed{SEED}.png", bbox_inches="tight", dpi=300)
plt.show()