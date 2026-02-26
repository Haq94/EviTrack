# tests/analytical_world_model_script.py
import numpy as np
import torch

from data.synthetic_generator import generate_sequences
from data.synthetic_tasks.doublewell_1d import make_prior, make_transition, make_emission

from world_model.analytical import AnalyticalWorldModel
from world_model.base import WorldModelConfig
from proposal import Proposal, ProposalConfig
from training.trainer import Trainer, TrainerConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    seed = 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- build analytic SSM components (truth) ----
    prior = make_prior(z0_mean=0.0, z0_std=1.0)
    transition = make_transition(a=3.0, V=0.06, dt=1.0, sigma_z=0.05)
    emission = make_emission(d=2.0, n=1, sigma_x=0.12)

    # ---- build analytical world model ----
    wm_cfg = WorldModelConfig(dz=1, dx=1)  # fill in any required fields in your cfg
    wm = AnalyticalWorldModel(
        cfg=wm_cfg,
        prior_mu0=prior.mu0,
        prior_cov0=prior.cov0,
        trans_mean=transition.mean_fn,
        trans_cov=transition.cov_fn,
        emit_mean=emission.mean_fn,
        emit_cov=emission.cov_fn,
    ).to(device=device, dtype=dtype)

    # ---- build a proposal (can be learned; later you can also make analytic proposal) ----
    q_cfg = ProposalConfig(dz=1, dx=1)  # set modes/memory per your project defaults
    proposal = Proposal(q_cfg).to(device=device, dtype=dtype)

    # ---- trainer ----
    tcfg = TrainerConfig(
        lr=1e-3,
        beta=1.0,
        amp=False,
    )
    trainer = Trainer(wm=wm, proposal=proposal, cfg=tcfg)

    # ---- generate a batch using the SAME analytic SSM ----
    B, T = 8, 50
    batch = generate_sequences(
        T=T,
        B=B,
        prior=prior,
        transition=transition,
        emission=emission,
        seed=seed + 123,
        return_logp=False,
    )
    x = torch.tensor(batch.x, dtype=dtype, device=device)  # (B,T,1)

    # ---- smoke eval ----
    out = trainer.eval_step({"x": x})
    print("OK | eval_step:", out)


if __name__ == "__main__":
    main()