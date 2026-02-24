# tests/runner_script.py
#
# Full smoke test for ExperimentRunner:
#   - trains
#   - evaluates
#   - plots metrics
#   - saves plots to run directory
#
# No argparse, no main().

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import glob
import numpy as np
import matplotlib.pyplot as plt

from training.runner import ExperimentRunner, RunConfig, DataConfig
from training.trainer import TrainerConfig
from world_model.base import WorldModelConfig
from proposal import ProposalConfig


# -------------------------
# Run parameters
# -------------------------
EXPERIMENT_NAME = "_runner_smoke"
RUN_ROOT = "results"
SEED = 0

EPOCHS = 20
LOG_EVERY_STEPS = 10
SAVE_EVERY_STEPS = 100

WM_KIND = "nonmarkov"
DZ = 4
DX = 1
T = 25

OBJECTIVE = "beta_elbo"   # "beta_elbo" | "iwae"
BETA = 1.0
K = 8
LR = 3e-4

# Data
BATCH_SIZE = 64
N_TRAIN = 1024
N_VAL = 256

# -------------------------
# Build configs
# -------------------------
wm_cfg = WorldModelConfig(
    dz=DZ,
    dx=DX,
    x_mode="memory",
    x_mem_dim=32,
    z_mem_dim=64,
)

q_cfg = ProposalConfig(
    dz=DZ,
    dx=DX,
    z_mode="memory",
    z_mem_dim=64,
    x_mode="memory",
    x_mem_dim=32,
)

trainer_cfg = TrainerConfig(
    objective=OBJECTIVE,
    beta=BETA,
    K=K,
    lr=LR,
    grad_clip_norm=1.0,
    amp=False,
    reduce_time="mean",
)

data_cfg = DataConfig(
    kind="synthetic",
    builder="data.synthetic_tasks.doublewell_1d:build_loaders",
    builder_kwargs=dict(
        a=3.0,
        V=0.06,
        sigma_z=0.05,
        d=2.0,
        n=1,
        sigma_x=0.12,
    ),
    batch_size=BATCH_SIZE,
    n_train=N_TRAIN,
    n_val=N_VAL,
    T=T,
)

run_cfg = RunConfig(
    experiment_name=EXPERIMENT_NAME,
    run_root=RUN_ROOT,
    seed=SEED,
    epochs=EPOCHS,
    log_every_steps=LOG_EVERY_STEPS,
    save_every_steps=SAVE_EVERY_STEPS,
    wm_kind=WM_KIND,
    wm_cfg=wm_cfg,
    proposal_cfg=q_cfg,
    trainer_cfg=trainer_cfg,
    data_cfg=data_cfg,
    note="runner_script with eval + plots",
)

# -------------------------
# Run training
# -------------------------
runner = ExperimentRunner(run_cfg)
runner.setup()
runner.fit()

# -------------------------
# Final evaluation
# -------------------------
val_stats = runner.evaluate()
print("\nFinal validation stats:", val_stats)

# -------------------------
# Load metrics and plot
# -------------------------
run_dir = os.path.join(RUN_ROOT, EXPERIMENT_NAME, f"seed_{SEED:03d}")
metrics_path = os.path.join(run_dir, "metrics.jsonl")

steps = []
train_loss = []
val_loss = []
elbo = []
iwae = []

with open(metrics_path, "r") as f:
    for line in f:
        row = json.loads(line)
        if "loss" in row:
            steps.append(row["step"])
            train_loss.append(row["loss"])
            if "elbo" in row:
                elbo.append(row["elbo"])
            if "iwae" in row:
                iwae.append(row["iwae"])
        if "val_loss" in row:
            val_loss.append((row["step"], row["val_loss"]))

# Convert val_loss into arrays
val_steps = [v[0] for v in val_loss]
val_vals = [v[1] for v in val_loss]

# -------------------------
# Plot 1: Training loss
# -------------------------
plt.figure()
plt.plot(steps, train_loss, label="train loss")
if val_vals:
    plt.plot(val_steps, val_vals, label="val loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Training / Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(run_dir, "loss_curve.png"))
plt.close()

# -------------------------
# Plot 2: ELBO or IWAE
# -------------------------
if OBJECTIVE == "beta_elbo" and elbo:
    plt.figure()
    plt.plot(steps[:len(elbo)], elbo)
    plt.xlabel("step")
    plt.ylabel("ELBO")
    plt.title("ELBO")
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "elbo_curve.png"))
    plt.close()

if OBJECTIVE == "iwae" and iwae:
    plt.figure()
    plt.plot(steps[:len(iwae)], iwae)
    plt.xlabel("step")
    plt.ylabel("IWAE")
    plt.title("IWAE")
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "iwae_curve.png"))
    plt.close()

print("\nSaved plots to:", run_dir)
print("Available files:", os.listdir(run_dir))
print("✅ Runner script completed with evaluation + plots.")