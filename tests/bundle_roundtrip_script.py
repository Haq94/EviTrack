# tests/bundle_roundtrip_script.py
#
# Smoke test: build WM + Proposal -> wrap in ModelBundle -> save -> load -> verify
# No argparse, no main(). Edit "Run parameters" and run directly.

import os
import torch

from utils.bundle import ModelBundle
from utils.builders import make_wm_config_dict, make_proposal_config_dict

from world_model.base import WorldModelConfig
from world_model.markov import MarkovWorldModel
from world_model.nonmarkov import NonMarkovWorldModel
from proposal import Proposal, ProposalConfig


# -------------------------
# Run parameters (edit here)
# -------------------------
SAVE_DIR = "results/_bundle_roundtrip_test"  # folder created/overwritten
DEVICE = "cpu"  # "cpu" or "cuda"

WM_KIND = "nonmarkov"  # "markov" | "nonmarkov"
DZ = 4
DX = 3

# WM toggles
WM_X_MODE = "memory"     # "none" | "markov" | "memory"   (whatever your WMConfig expects)
WM_X_MEM_DIM = 16
WM_Z_MEM_DIM = 32

# Proposal toggles
Q_Z_MODE = "memory"      # "markov" | "memory"
Q_Z_MEM_DIM = 32
Q_X_MODE = "memory"      # "none" | "markov" | "memory"
Q_X_MEM_DIM = 16

# Optional sharing from WM (should be False for this test unless you want to check sharing too)
SHARE_Z_GRU_FROM_WM = False
SHARE_X_GRU_FROM_WM = False
STRICT_SHARE = True

META = {"note": "bundle roundtrip test"}


# -------------------------
# Helpers
# -------------------------
def build_wm(kind: str, cfg: WorldModelConfig):
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
    z_mem_dim=WM_Z_MEM_DIM,  # ignored by Markov
)
wm = build_wm(WM_KIND, wm_cfg).to(device)

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
q = Proposal(q_cfg, wm=wm).to(device)

# Put configs into JSONable dicts (and IMPORTANTLY inject wm kind)
wm_cfg_dict = make_wm_config_dict(wm_cfg, kind=WM_KIND)
q_cfg_dict = make_proposal_config_dict(q_cfg)

# -------------------------
# Save
# -------------------------
if os.path.exists(SAVE_DIR):
    # simple "overwrite": remove files inside (keeps it safe-ish)
    for fn in os.listdir(SAVE_DIR):
        try:
            os.remove(os.path.join(SAVE_DIR, fn))
        except Exception:
            pass
else:
    os.makedirs(SAVE_DIR, exist_ok=True)

bundle = ModelBundle(
    wm=wm,
    proposal=q,
    wm_config=wm_cfg_dict,
    proposal_config=q_cfg_dict,
    meta=META,
)
bundle.save(SAVE_DIR)
print(f"Saved bundle to: {SAVE_DIR}")

# -------------------------
# Load
# -------------------------
bundle2 = ModelBundle.load(SAVE_DIR, device=device)
wm2 = bundle2.wm
q2 = bundle2.proposal

print("Loaded bundle.")
print("  wm kind:", bundle2.wm_config.get("kind", None))
print("  meta:", bundle2.meta)

# -------------------------
# Verify state_dict equality
# -------------------------
def assert_state_dict_equal(sd_a, sd_b, name: str):
    if sd_a.keys() != sd_b.keys():
        missing_a = sd_b.keys() - sd_a.keys()
        missing_b = sd_a.keys() - sd_b.keys()
        raise AssertionError(f"{name}: keys mismatch. missing_in_a={missing_a}, missing_in_b={missing_b}")

    max_abs = 0.0
    max_key = None
    for k in sd_a.keys():
        a = sd_a[k].detach().cpu()
        b = sd_b[k].detach().cpu()
        if a.shape != b.shape:
            raise AssertionError(f"{name}: shape mismatch at key='{k}': {a.shape} vs {b.shape}")
        diff = (a - b).abs().max().item() if a.numel() > 0 else 0.0
        if diff > max_abs:
            max_abs = diff
            max_key = k

    print(f"{name}: max_abs_param_diff={max_abs} (key={max_key})")
    if max_abs != 0.0:
        raise AssertionError(f"{name}: state_dict differs (max_abs={max_abs} at {max_key})")

assert_state_dict_equal(wm.state_dict(), wm2.state_dict(), "WM")
assert_state_dict_equal(q.state_dict(), q2.state_dict(), "Proposal")

print("✅ Bundle roundtrip success: WM and Proposal weights match exactly.")
