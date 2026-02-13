# tests/runner_script.py
#
# Minimal smoke test for ExperimentRunner:
#  - builds WM + Proposal
#  - builds synthetic DataLoaders
#  - trains for a few epochs
#  - saves checkpoints + final bundle under results/<experiment>/seed_<seed>/

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import glob

from training.runner import ExperimentRunner, RunConfig, DataConfig
from training.trainer import TrainerConfig
from world_model import WorldModelConfig
from proposal import ProposalConfig


# -------------------------
# Run parameters (edit here)
# -------------------------
EXPERIMENT_NAME = "_runner_smoke"
RUN_ROOT = "results"
SEED = 0

EPOCHS = 2
LOG_EVERY_STEPS = 10
SAVE_EVERY_STEPS = 50

# WM / Proposal
WM_KIND = "nonmarkov"  # "markov" | "nonmarkov"
DZ = 4
DX = 3
T = 25

WM_X_MODE = "memory"   # "none" | "markov" | "memory"
WM_X_MEM_DIM = 32
WM_Z_MEM_DIM = 64

Q_Z_MODE = "memory"    # "markov" | "memory"
Q_Z_MEM_DIM = 64
Q_X_MODE = "memory"    # "none" | "markov" | "memory"
Q_X_MEM_DIM = 32

# Training objective
OBJECTIVE = "beta_elbo"   # "beta_elbo" | "iwae"
BETA = 1.0
K = 8
LR = 3e-4

# Data
BATCH_SIZE = 64
N_TRAIN = 512
N_VAL = 128
PROCESS_NOISE = 0.10
EMIT_NOISE = 0.15


# -------------------------
# Build configs
# -------------------------
wm_cfg = WorldModelConfig(
    dz=DZ,
    dx=DX,
    x_mode=WM_X_MODE,
    x_mem_dim=WM_X_MEM_DIM,
    z_mem_dim=WM_Z_MEM_DIM,
)

q_cfg = ProposalConfig(
    dz=DZ,
    dx=DX,
    z_mode=Q_Z_MODE,
    z_mem_dim=Q_Z_MEM_DIM,
    x_mode=Q_X_MODE,
    x_mem_dim=Q_X_MEM_DIM,
    share_z_gru_from_wm=False,
    share_x_gru_from_wm=False,
    strict_share=True,
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
    batch_size=BATCH_SIZE,
    n_train=N_TRAIN,
    n_val=N_VAL,
    T=T,
    dz=DZ,
    dx=DX,
    process_noise=PROCESS_NOISE,
    emit_noise=EMIT_NOISE,
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
    note="runner_script.py smoke test",
)

runner = ExperimentRunner(run_cfg)
runner.setup()
runner.fit()

# -------------------------
# Post-run checks (lightweight)
# -------------------------
run_dir = os.path.join(RUN_ROOT, EXPERIMENT_NAME, f"seed_{SEED:03d}")
print("\n=== Post-run checks ===")
print("Run dir:", run_dir)

cfg_path = os.path.join(run_dir, "run_config.json")
metrics_path = os.path.join(run_dir, "metrics.jsonl")
final_dir = os.path.join(run_dir, "final")

assert os.path.exists(cfg_path), f"Missing {cfg_path}"
assert os.path.exists(metrics_path), f"Missing {metrics_path}"
assert os.path.isdir(final_dir), f"Missing {final_dir}"

required_final = [
    "wm_state.pt",
    "proposal_state.pt",
    "wm_config.json",
    "proposal_config.json",
    "meta.json",
]
for fn in required_final:
    p = os.path.join(final_dir, fn)
    assert os.path.exists(p), f"Missing final artifact: {p}"

# Print last metrics row
with open(metrics_path, "r", encoding="utf-8") as f:
    rows = f.read().strip().splitlines()
    if rows:
        last = json.loads(rows[-1])
        print("Last metrics row:", last)
    else:
        print("metrics.jsonl empty (unexpected)")

# List checkpoints
ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_step_*")))
print("Checkpoints:", ckpts[:5], ("..." if len(ckpts) > 5 else ""))
print("✅ runner_script completed successfully.")
