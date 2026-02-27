# inference_test.py
import torch

from world_model.base import WorldModelConfig
from world_model.markov import MarkovWorldModel

from proposal import Proposal, ProposalConfig

from inference.evitrack import EviTrackEngine, EviTrackConfig
from inference.baselines.smc import SMCEngine, SMCConfig
from inference.baselines.bpf import BFPEngine, BPFConfig
from inference.baselines.sis import SISEngine

from inference.utils import normalize_logweights
from inference.resampling import multinomial_resample_indices, resample_particles


def generate_synthetic_x(*, T: int, dx: int, B: int = 1, device="cpu", dtype=torch.float32, seed: int = 0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(B, T, dx, device=device, dtype=dtype, generator=g)


def run_engine(engine, x):
    """
    Runs inference sequentially over x[:, t] and performs basic sanity checks.
    """
    state = engine.init_state(B=x.shape[0], device=x.device, dtype=x.dtype)

    for t in range(x.shape[1]):
        state, stats = engine.step(state, x[:, t])
        assert stats.t == t + 1

    w, support = engine.get_mixture(state)

    # mixture sanity
    assert torch.isfinite(w).all(), "Non-finite mixture weights."
    assert (w >= 0).all(), "Negative mixture weights."
    assert abs(float(w.sum().item()) - 1.0) < 1e-5, "Mixture weights must sum to 1."
    assert len(support) > 0, "Empty support returned."

    # cost sanity (at least emissions)
    assert getattr(state, "cost").emission_evals > 0, "Cost counter didn't record emissions."

    return state


def test_resampling_no_aliasing(particles):
    """
    Catches aliasing bugs in resampling (tree_clone / resample_particles).
    """
    if len(particles) < 4:
        return

    # force heavy duplication with skewed weights
    logw = torch.tensor([0.0, -20.0, -20.0, -20.0], device=particles[0].logw.device, dtype=particles[0].logw.dtype)
    w = normalize_logweights(logw, dim=0)
    idx = multinomial_resample_indices(w, N=len(particles))
    new_particles = resample_particles(particles, w, idx)

    # mutate one particle and ensure others don't change (alias check)
    new_particles[0].logw += 123.0
    for j in range(1, len(new_particles)):
        assert not torch.allclose(new_particles[0].logw, new_particles[j].logw), (
            "Aliasing detected: resampled particles share storage/objects."
        )


if __name__ == "__main__":
    DEVICE = "cpu"
    DTYPE = torch.float32

    # -----------------------------
    # Build WM + Proposal
    # -----------------------------
    wm_cfg = WorldModelConfig(dz=4, dx=3, x_mode="markov")
    wm = MarkovWorldModel(wm_cfg).to(device=DEVICE, dtype=DTYPE)
    wm.eval()

    # Forecasting-causal proposal q(z_t | z_<t, x_<t)
    # Uses GRU summaries by default (z_mode="memory", x_mode="memory"). :contentReference[oaicite:1]{index=1}
    q_cfg = ProposalConfig(
        dz=wm_cfg.dz,
        dx=wm_cfg.dx,
        z_mode="memory",   # or "markov"
        x_mode="memory",   # or "markov" / "none"
        z_mem_dim=32,
        x_mem_dim=32,
        share_z_gru_from_wm=False,
        share_x_gru_from_wm=False,
        strict_share=True,
    )
    proposal = Proposal(q_cfg, wm=None).to(device=DEVICE, dtype=DTYPE)
    proposal.eval()

    # -----------------------------
    # Synthetic observations
    # -----------------------------
    x = generate_synthetic_x(T=12, dx=wm_cfg.dx, B=1, device=DEVICE, dtype=DTYPE, seed=0)

    # -----------------------------
    # Engines
    # -----------------------------
    engines = [
        (
            "EviTrack(proposal-expand)",
            EviTrackEngine(
                wm=wm,
                proposal=proposal,
                cfg=EviTrackConfig(K=4, C=2, tau=1, expand="proposal"),
            ),
        ),
        (
            "SMC(with proposal)",
            SMCEngine(
                wm=wm,
                proposal=proposal,
                cfg=SMCConfig(N=16, resample_every_step=False),
            ),
        ),
        (
            "BPF(transition-only)",
            BFPEngine(
                wm=wm,
                cfg=BPFConfig(N=16, resample_every_step=False),
            ),
        ),
        (
            "SIS(with proposal)",
            SISEngine(
                wm=wm,
                proposal=proposal,
                N=16,
            ),
        ),
    ]

    for name, eng in engines:
        state = run_engine(eng, x)
        print(f"\n{name}: OK")
        print("  transition_evals:", state.cost.transition_evals)
        print("  proposal_evals  :", state.cost.proposal_evals)
        print("  emission_evals  :", state.cost.emission_evals)

        if hasattr(state, "particles"):
            test_resampling_no_aliasing(state.particles)
            print("  resampling alias check: OK")