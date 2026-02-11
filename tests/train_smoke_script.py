# tests/train_smoke_script.py
#
# Smoke training script: WM + Proposal + Trainer + plots.
# No argparse, no main(). Edit "Run parameters" and run directly.

import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from world_model import WorldModelConfig, MarkovWorldModel, NonMarkovWorldModel
from proposal import Proposal, ProposalConfig
from training.trainer import Trainer, TrainerConfig

from utils.bundle import ModelBundle
from utils.builders import make_wm_config_dict, make_proposal_config_dict

from data.synthetic_generator import (
    GaussianPrior,
    GaussianTransition,
    GaussianEmission,
    generate_sequences,
)


# -------------------------
# Run parameters (edit here)
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

SAVE_DIR = "results/_smoke_train_run"
SAVE_EVERY = 200  # steps

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# Data
BATCH_SIZE = 64
N_TRAIN = 2048
N_VAL = 512
T = 25
DZ = 4
DX = 3

# Synthetic difficulty knobs
PROCESS_NOISE = 0.10   # latent transition noise std
EMIT_NOISE = 0.15      # emission noise std

# WM kind
WM_KIND = "nonmarkov"  # "markov" | "nonmarkov"

# World model modes (match your configs)
WM_X_MODE = "memory"   # "none" | "markov" | "memory"
WM_X_MEM_DIM = 32
WM_Z_MEM_DIM = 64

# Proposal modes
Q_Z_MODE = "memory"    # "markov" | "memory"
Q_Z_MEM_DIM = 64
Q_X_MODE = "memory"    # "none" | "markov" | "memory"
Q_X_MEM_DIM = 32

# Objective
OBJECTIVE = "beta_elbo"    # "beta_elbo" | "iwae"
BETA = 1.0
K = 16

# Train loop
STEPS = 1500
LR = 3e-4
GRAD_CLIP = 1.0
AMP = False  # set True if you want mixed precision (cuda)

PLOT_EVERY = 50


# -------------------------
# Build WM + Proposal
# -------------------------
def build_wm(kind: str, cfg: WorldModelConfig):
    kind = kind.lower()
    if kind in ("markov", "m"):
        return MarkovWorldModel(cfg)
    if kind in ("nonmarkov", "non-markov", "nm"):
        return NonMarkovWorldModel(cfg)
    raise ValueError(f"Unknown WM_KIND: {kind}")


wm_cfg = WorldModelConfig(
    dz=DZ,
    dx=DX,
    x_mode=WM_X_MODE,
    x_mem_dim=WM_X_MEM_DIM,
    z_mem_dim=WM_Z_MEM_DIM,
)

wm = build_wm(WM_KIND, wm_cfg).to(device=DEVICE, dtype=DTYPE)

q_cfg = ProposalConfig(
    dz=DZ,
    dx=DX,
    z_mode=Q_Z_MODE,
    z_mem_dim=Q_Z_MEM_DIM,
    x_mode=Q_X_MODE,
    x_mem_dim=Q_X_MEM_DIM,
    # keep sharing off for clean behavior (you can flip later)
    share_z_gru_from_wm=False,
    share_x_gru_from_wm=False,
    strict_share=True,
)

proposal = Proposal(q_cfg, wm=wm).to(device=DEVICE, dtype=DTYPE)

trainer_cfg = TrainerConfig(
    objective=OBJECTIVE,
    beta=BETA,
    K=K,
    lr=LR,
    grad_clip_norm=GRAD_CLIP,
    amp=AMP,
    reduce_time="mean",
)

trainer = Trainer(wm=wm, proposal=proposal, cfg=trainer_cfg)


# -------------------------
# Synthetic dataset (linear Gaussian)
#   z1 ~ N(0, I)
#   zt = A z_{t-1} + eps
#   xt = C zt + eta
# -------------------------
rng = np.random.default_rng(SEED)

# Stable A
A = rng.standard_normal((DZ, DZ)) * 0.2
# make it contractive
u, s, vt = np.linalg.svd(A, full_matrices=False)
s = np.clip(s, 0.0, 0.9)
A = (u * s) @ vt

C = rng.standard_normal((DX, DZ)) * 0.8

prior = GaussianPrior(mu0=np.zeros(DZ), cov0=np.eye(DZ))

def trans_mean(z_prev, extras):
    # z_prev: (B, DZ)
    return (z_prev @ A.T)

def trans_cov(z_prev, extras):
    return (PROCESS_NOISE**2) * np.eye(DZ)

transition = GaussianTransition(mean_fn=trans_mean, cov_fn=trans_cov)

def emit_mean(z, extras):
    return (z @ C.T)

def emit_cov(z, extras):
    return (EMIT_NOISE**2) * np.eye(DX)

emission = GaussianEmission(mean_fn=emit_mean, cov_fn=emit_cov)


def make_dataset(N: int, seed0: int):
    # generate sequences in mini-batches so seeds vary
    xs = []
    for i in range(N):
        batch = generate_sequences(
            T=T,
            B=1,
            prior=prior,
            transition=transition,
            emission=emission,
            seed=seed0 + i,
            return_logp=False,
        )
        xs.append(batch.x[0])  # (T, DX)
    x = np.stack(xs, axis=0)  # (N, T, DX)
    return torch.tensor(x, dtype=DTYPE)


x_train = make_dataset(N_TRAIN, seed0=1000)
x_val = make_dataset(N_VAL, seed0=9000)

train_loader = DataLoader(TensorDataset(x_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(x_val), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

train_iter = iter(train_loader)


# -------------------------
# Train loop + logging
# -------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

history = {
    "step": [],
    "loss": [],
    "elbo": [],
    "logp_x": [],
    "logp_z": [],
    "logq_z": [],
    "iwae": [],
}

def maybe_get(d, k):
    return float(d[k]) if k in d else float("nan")


for step in range(1, STEPS + 1):
    try:
        (xb,) = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        (xb,) = next(train_iter)

    stats = trainer.train_step({"x": xb})

    history["step"].append(step)
    history["loss"].append(stats["loss"])
    history["elbo"].append(maybe_get(stats, "elbo"))
    history["logp_x"].append(maybe_get(stats, "logp_x"))
    history["logp_z"].append(maybe_get(stats, "logp_z"))
    history["logq_z"].append(maybe_get(stats, "logq_z"))
    history["iwae"].append(maybe_get(stats, "iwae"))

    if step % PLOT_EVERY == 0 or step == 1:
        print(
            f"step={step:5d}  loss={stats['loss']:.4f}  "
            + (f"elbo={maybe_get(stats,'elbo'):.4f}  logp_x={maybe_get(stats,'logp_x'):.4f}" if OBJECTIVE == "beta_elbo"
               else f"iwae={maybe_get(stats,'iwae'):.4f}")
        )

    if step % SAVE_EVERY == 0 or step == STEPS:
        # save bundle snapshot
        wm_cfg_dict = make_wm_config_dict(wm_cfg, kind=WM_KIND)
        q_cfg_dict = make_proposal_config_dict(q_cfg)

        bundle = ModelBundle(
            wm=wm,
            proposal=proposal,
            wm_config=wm_cfg_dict,
            proposal_config=q_cfg_dict,
            meta={
                "seed": SEED,
                "objective": OBJECTIVE,
                "beta": BETA,
                "K": K,
                "step": step,
                "note": "smoke training checkpoint",
            },
        )
        ckpt_dir = os.path.join(SAVE_DIR, f"ckpt_step_{step:06d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        bundle.save(ckpt_dir)
        print(f"Saved checkpoint: {ckpt_dir}")


# -------------------------
# Quick validation (one pass)
# -------------------------
with torch.no_grad():
    vals = []
    for (xb,) in val_loader:
        out = trainer.eval_step({"x": xb})
        vals.append(out["loss"])
    val_loss = float(np.mean(vals))
print(f"VAL loss: {val_loss:.4f}")


# -------------------------
# Plots
# -------------------------
steps = np.array(history["step"], dtype=int)

plt.figure()
plt.plot(steps, history["loss"])
plt.xlabel("step")
plt.ylabel("loss")
plt.title(f"Training loss ({OBJECTIVE})")
plt.grid(True)

if OBJECTIVE == "beta_elbo":
    plt.figure()
    plt.plot(steps, history["elbo"])
    plt.xlabel("step")
    plt.ylabel("ELBO")
    plt.title("ELBO")
    plt.grid(True)

    plt.figure()
    plt.plot(steps, history["logp_x"], label="logp_x")
    plt.plot(steps, history["logp_z"], label="logp_z")
    plt.plot(steps, history["logq_z"], label="logq_z")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title("ELBO components")
    plt.legend()
    plt.grid(True)
else:
    plt.figure()
    plt.plot(steps, history["iwae"])
    plt.xlabel("step")
    plt.ylabel("IWAE")
    plt.title(f"IWAE (K={K})")
    plt.grid(True)

plt.show()
