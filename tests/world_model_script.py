import torch

from world_model import WorldModelConfig, MarkovWorldModel, NonMarkovWorldModel


# -------------------------
# Run parameters (edit here)
# -------------------------
WM_KIND = "nonmarkov"          # "markov" | "nonmarkov"
B = 3                       # batch size (# independent streams)
T = 5                       # timesteps to roll forward
DEVICE = "cpu"              # "cpu" | "cuda"
DTYPE = torch.float32       # torch.float32, torch.float64, etc.

# x-conditioning mode in emission:
#   "none"   : ignore x_{<t}
#   "markov" : use x_{t-1}
#   "memory" : use GRU(x_{<t})
X_MODE = "markov"
X_MEM_DIM = 16              # only used if X_MODE == "memory"

# latent sizes
DZ = 4
DX = 3
Z_MEM_DIM = 32              # only used by NonMarkov (if you use it)


def build_wm(kind: str, cfg: WorldModelConfig):
    kind = kind.lower()
    if kind in ("markov", "m"):
        return MarkovWorldModel(cfg)
    if kind in ("nonmarkov", "non-markov", "nm"):
        return NonMarkovWorldModel(cfg)
    raise ValueError(f"Unknown WM_KIND='{kind}'. Use 'markov' or 'nonmarkov'.")


# -------------------------
# Construct world model
# -------------------------
cfg = WorldModelConfig(
    dz=DZ,
    dx=DX,
    x_mode=X_MODE,
    x_mem_dim=X_MEM_DIM,
    z_mem_dim=Z_MEM_DIM,
)

wm = build_wm(WM_KIND, cfg).to(device=DEVICE, dtype=DTYPE)
wm.eval()

device = torch.device(DEVICE)

# -------------------------
# Init states
# -------------------------
z_state = wm.init_z_state(B, device=device, dtype=DTYPE)
x_state = wm.init_x_state(B, device=device, dtype=DTYPE)

# -------------------------
# t = 1  (prior + emission)
# -------------------------
# Sample z1 from prior
z_t = wm.sample_z1(B, device=device, dtype=DTYPE)
logp_z = wm.log_prob_z1(z_t) if hasattr(wm, "log_prob_z1") else None

# Emission uses current z-summary
z_state_curr = wm.z_state_curr(z_state, z_t)

emit_params = wm.emission_params(z_state_curr=z_state_curr, x_state_prev=x_state)
x_t = wm.sample_emission(emit_params)
logp_x = wm.log_prob_emission(x_t, emit_params)

# Update stored states
z_state = wm.update_z_state(z_state, z_t)   # Markov may return None
x_state = wm.update_x_state(x_state, x_t)

print(
    f"[{WM_KIND}] t=1 "
    f"z={tuple(z_t.shape)} "
    f"x={tuple(x_t.shape)} "
    f"logp_x={tuple(logp_x.shape)} "
    f"z_state={'None' if z_state is None else tuple(z_state.shape)} "
    f"x_state={'None' if x_state is None else tuple(x_state.shape)}"
)

# -------------------------
# t = 2..T  (transition + emission via step)
# -------------------------
for t in range(2, T + 1):
    out = wm.step(
        B=B,
        z_prev=z_t,
        z_state_prev=z_state,
        x_state_prev=x_state,
        device=device,
        dtype=DTYPE,
    )

    z_t = out["z_t"]
    x_t = out["x_t"]
    z_state = out["z_state_t"]
    x_state = out["x_state_t"]

    logp_z = out.get("logp_z", None)
    logp_x = out.get("logp_x", None)

    print(
        f"[{WM_KIND}] t={t} "
        f"z={tuple(z_t.shape)} "
        f"x={tuple(x_t.shape)} "
        f"logp_z={None if logp_z is None else tuple(logp_z.shape)} "
        f"logp_x={None if logp_x is None else tuple(logp_x.shape)} "
        f"z_state={'None' if z_state is None else tuple(z_state.shape)} "
        f"x_state={'None' if x_state is None else tuple(x_state.shape)}"
    )
