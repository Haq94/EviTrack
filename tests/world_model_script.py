import torch
from world_model import WorldModelConfig, NonMarkovWorldModel

cfg = WorldModelConfig(
    dz=4,
    dx=3,
    use_x_memory_in_emission=True,
    x_mem_dim=16,
    z_mem_dim=32,
)

wm = NonMarkovWorldModel(cfg)
wm.eval()

B = 3        # set to 1 for sanity/debug
T = 5
device = "cpu"

# ---- init states ----
z_state = wm.init_z_state(B, device=device)
x_state = wm.init_x_state(B, device=device)

# ---- t = 1 (prior) ----
z_t = wm.sample_z1(B, device=device)

# build current z-state for emission
z_state_curr = wm.z_state_curr(z_state, z_t)

emit_params = wm.emission_params(
    z_state_curr=z_state_curr,
    x_state_prev=x_state,
)
x_t = wm.sample_emission(emit_params)
logp_x = wm.log_prob_emission(x_t, emit_params)

# update stored states
z_state = wm.update_z_state(z_state, z_t)
x_state = wm.update_x_state(x_state, x_t)

print(f"t=1  z={z_t.shape}  x={x_t.shape}  logp_x={logp_x.shape}")

# ---- t = 2..T ----
for t in range(2, T + 1):
    out = wm.step(
        B=B,
        z_prev=z_t,
        z_state_prev=z_state,
        x_state_prev=x_state,
        device=device,
    )

    z_t = out["z_t"]
    x_t = out["x_t"]
    z_state = out["z_state_t"]
    x_state = out["x_state_t"]

    print(
        f"t={t}  "
        f"z={z_t.shape}  "
        f"x={x_t.shape}  "
        f"logp_z={out['logp_z'].shape}  "
        f"logp_x={out['logp_x'].shape}"
    )

