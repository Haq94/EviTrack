# tests/proposal_script.py
#
# Smoke test for Proposal + World Model interaction.
# No argparse, no main(). Edit the "Run parameters" section and run directly.
#
# This script:
#   - builds a WM (markov/nonmarkov) and a Proposal
#   - optionally shares WM GRU summarizers inside the Proposal
#   - rolls forward for T steps:
#       * propose z_t ~ q(z_t | z_{<t}, x_{<t})   (forecasting-causal)
#       * generate x_t ~ p(x_t | z_state_curr, x_state_prev) using the WM emission
#       * update WM states with (z_t, x_t)
#       * update Proposal states:
#           - z_state_q updated inside propose()
#           - x_state_q updated via observe_x(x_t) AFTER x_t is "observed"
#
# NOTE: This is a *plumbing* test. It does not do evidence scoring or pruning.
# It just verifies that shapes, state updates, and optional sharing work.

# -------------------------
# Imports (package-friendly)
# -------------------------
import torch

from world_model import WorldModelConfig, MarkovWorldModel, NonMarkovWorldModel
from proposal import Proposal, ProposalConfig  

# -------------------------
# Run parameters (edit here)
# -------------------------
WM_KIND = "nonmarkov"        # "markov" | "nonmarkov"
DEVICE = "cpu"               # "cpu" | "cuda"
DTYPE = torch.float32

B = 3
T = 6

DZ = 4
DX = 3

# WM emission x-conditioning mode:
#   "none"   : ignore x_{<t}
#   "markov" : use x_{t-1}
#   "memory" : GRU(x_{<t})
WM_X_MODE = "memory"
WM_X_MEM_DIM = 16

# WM latent memory (only used by NonMarkov WM)
WM_Z_MEM_DIM = 32

# Proposal modes (forecasting-causal: proposal at time t uses only x_{<t})
Q_Z_MODE = "memory"          # "markov" | "memory"
Q_Z_MEM_DIM = 32

Q_X_MODE = "memory"          # "none" | "markov" | "memory"
Q_X_MEM_DIM = 16

# Optional sharing of GRU summarizers from WM into Proposal
SHARE_Z_GRU_FROM_WM = True
SHARE_X_GRU_FROM_WM = True
STRICT_SHARE = False         # if True, raise if sharing is requested but impossible/mismatched

# Optional: show a quick confirmation that GRUs are tied (same object) when sharing is on
PRINT_SHARE_CHECKS = True


def build_wm(kind: str, cfg: "WorldModelConfig"):
    k = kind.lower()
    if k in ("markov", "m"):
        return MarkovWorldModel(cfg)
    if k in ("nonmarkov", "non-markov", "nm"):
        return NonMarkovWorldModel(cfg)
    raise ValueError(f"Unknown WM_KIND='{kind}'")


# -------------------------
# Build WM + Proposal
# -------------------------
device = torch.device(DEVICE)

wm_cfg = WorldModelConfig(
    dz=DZ,
    dx=DX,
    x_mode=WM_X_MODE,
    x_mem_dim=WM_X_MEM_DIM,
    z_mem_dim=WM_Z_MEM_DIM,  # ignored by Markov WM
)

wm = build_wm(WM_KIND, wm_cfg).to(device=device, dtype=DTYPE)
wm.eval()

q_cfg = ProposalConfig(
    dz=DZ,
    dx=DX,
    z_mode=Q_Z_MODE,
    z_mem_dim=Q_Z_MEM_DIM,
    x_mode=Q_X_MODE,
    x_mem_dim=Q_X_MEM_DIM,
    share_z_gru_from_wm=SHARE_Z_GRU_FROM_WM,
    share_x_gru_from_wm=SHARE_X_GRU_FROM_WM,
    strict_share=STRICT_SHARE,
)

q = Proposal(q_cfg, wm=wm).to(device=device, dtype=DTYPE)
q.eval()

if PRINT_SHARE_CHECKS:
    wm_zm = getattr(wm, "z_memory", None)
    wm_xm = getattr(wm, "x_memory", None)
    q_zm = getattr(q, "z_memory", None)
    q_xm = getattr(q, "x_memory", None)

    print("---- Share checks ----")
    print(f"WM_KIND={WM_KIND}  WM_X_MODE={WM_X_MODE}  Q_Z_MODE={Q_Z_MODE}  Q_X_MODE={Q_X_MODE}")
    print(f"share_z_gru_from_wm={SHARE_Z_GRU_FROM_WM}  share_x_gru_from_wm={SHARE_X_GRU_FROM_WM}")
    print(f"wm.z_memory exists: {wm_zm is not None} | q.z_memory exists: {q_zm is not None} | tied: {wm_zm is q_zm}")
    print(f"wm.x_memory exists: {wm_xm is not None} | q.x_memory exists: {q_xm is not None} | tied: {wm_xm is q_xm}")
    print("----------------------\n")


# -------------------------
# Initialize states
# -------------------------
# World model states (used to generate x_t consistently with WM's own x_mode and z_mode)
wm_z_state = wm.init_z_state(B, device=device, dtype=DTYPE)
wm_x_state = wm.init_x_state(B, device=device, dtype=DTYPE)

# Proposal states (used for q(z_t | z_{<t}, x_{<t}))
q_z_state = q.init_z_state(B, device=device, dtype=DTYPE)
q_x_state = q.init_x_state(B, device=device, dtype=DTYPE)

z_prev = None  # for markov z_mode in proposal (and just for logging)
x_prev = None

print("---- Rollout ----")
for t in range(1, T + 1):
    # 1) Propose z_t using only prefix info (x_{<t} encoded in q_x_state)
    out_q = q.propose(
        B=B,
        z_prev=z_prev,
        z_state_prev=q_z_state,
        x_state_prev=q_x_state,
        device=device,
        dtype=DTYPE,
    )
    z_t = out_q["z_t"]
    logq_t = out_q["logq"]
    q_z_state = out_q["z_state_t"]  # may be None if Q_Z_MODE="markov"

    # 2) Generate x_t from the world model emission conditioned on current z-summary and WM x_state
    #    (This is just for plumbing; in real inference you'd *observe* x_t from data.)
    wm_z_state_curr = wm.z_state_curr(wm_z_state, z_t)  # IMPORTANT: current z-summary for emission

    emit_params = wm.emission_params(z_state_curr=wm_z_state_curr, x_state_prev=wm_x_state)
    x_t = wm.sample_emission(emit_params)
    logp_x = wm.log_prob_emission(x_t, emit_params)

    # 3) Update WM stored states (so WM can continue to generate consistently)
    wm_z_state = wm.update_z_state(wm_z_state, z_t)  # Markov WM may return None
    wm_x_state = wm.update_x_state(wm_x_state, x_t)

    # 4) Update Proposal x-state AFTER observing x_t (so it becomes part of x_{<t+1})
    q_x_state = q.observe_x(x_t, q_x_state)

    # 5) Bookkeeping for next step
    z_prev = z_t
    x_prev = x_t

    print(
        f"t={t:02d} | "
        f"z={tuple(z_t.shape)} "
        f"x={tuple(x_t.shape)} "
        f"logq={tuple(logq_t.shape)} "
        f"logp_x={tuple(logp_x.shape)} | "
        f"wm_z_state={'None' if wm_z_state is None else tuple(wm_z_state.shape)} "
        f"wm_x_state={'None' if wm_x_state is None else tuple(wm_x_state.shape)} | "
        f"q_z_state={'None' if q_z_state is None else tuple(q_z_state.shape)} "
        f"q_x_state={'None' if q_x_state is None else tuple(q_x_state.shape)}"
    )

print("---- Done ----")
