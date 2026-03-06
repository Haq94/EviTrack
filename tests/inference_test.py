# inference_test.py
import torch

from world_model.base import WorldModelConfig
from world_model.markov import MarkovWorldModel

from proposal import Proposal, ProposalConfig

from inference.evitrack import EviTrackEngine, EviTrackConfig
from inference.baselines.smc import SMCEngine, SMCConfig
from inference.baselines.bpf import BFPEngine, BPFConfig
from inference.baselines.sis import SISEngine
from inference.resampling import resample_particles


def generate_synthetic_x(*, T: int, dx: int, B: int = 1, device="cpu", dtype=torch.float32, seed: int = 0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(B, T, dx, device=device, dtype=dtype, generator=g)


def _first_support_group(support):
    """
    Returns a flat list of hypotheses/particles suitable for alias tests.

    Supports either:
      - flat support: List[obj]
      - batched support: List[List[obj]]
    """
    if len(support) == 0:
        return []
    first = support[0]
    if isinstance(first, list):
        return first
    return support


def _check_mixture_shape_and_normalization(w: torch.Tensor, B_expected: int):
    """
    Accept either legacy unbatched weights [N] or batched weights [B, N].

    For the new batched SMC/EviTrack path we expect [B, N].
    For legacy engines that still only expose a single set of weights, we only
    permit that when B_expected == 1.
    """
    assert torch.isfinite(w).all(), "Non-finite mixture weights."
    assert (w >= 0).all(), "Negative mixture weights."

    if w.ndim == 1:
        assert B_expected == 1, (
            f"Engine returned unbatched weights of shape {tuple(w.shape)} for B={B_expected}. "
            "Either run this engine with B=1 or update get_mixture() to return [B, N]."
        )
        assert abs(float(w.sum().item()) - 1.0) < 1e-5, "Mixture weights must sum to 1."
        return

    assert w.ndim == 2, f"Expected mixture weights to have ndim 1 or 2, got shape {tuple(w.shape)}"
    assert w.shape[0] == B_expected, (
        f"Mixture weights batch mismatch: got {tuple(w.shape)}, expected batch size {B_expected}."
    )
    row_sums = w.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        "Mixture weights must sum to 1 along the support dimension."
    )


def run_engine(engine, x):
    """
    Runs inference sequentially over x[:, t] and performs basic sanity checks.
    """
    state = engine.init_state(B=x.shape[0], device=x.device, dtype=x.dtype)

    for t in range(x.shape[1]):
        state, stats = engine.step(state, x[:, t])
        assert stats.t == t + 1

    w, support = engine.get_mixture(state)

    _check_mixture_shape_and_normalization(w, B_expected=x.shape[0])
    assert len(support) > 0, "Empty support returned."

    # cost sanity (at least emissions)
    assert getattr(state, "cost").emission_evals > 0, "Cost counter didn't record emissions."

    return state, w, support


def test_resampling_no_aliasing(particles):
    """
    Catches aliasing bugs in resampling (tree_clone / resample_particles).

    Expects a flat particle list (one batch element's particle set).
    """
    if len(particles) < 4:
        return

    base_particles = particles[:4]

    # Force duplication of the first particle
    idx = torch.zeros(len(base_particles), dtype=torch.long, device=base_particles[0].logw.device)

    new_particles = resample_particles(base_particles, idx)

    # mutate one particle and ensure others don't change
    new_particles[0].logw += 123.0

    for j in range(1, len(new_particles)):
        assert not torch.allclose(new_particles[0].logw, new_particles[j].logw), (
            "Aliasing detected: resampled particles share storage."
        )


def main():
    DEVICE = "cpu"
    DTYPE = torch.float32

    # -----------------------------
    # Build WM + Proposal
    # -----------------------------
    wm_cfg = WorldModelConfig(dz=4, dx=3, x_mode="markov")
    wm = MarkovWorldModel(wm_cfg).to(device=DEVICE, dtype=DTYPE)
    wm.eval()

    # Forecasting-causal proposal q(z_t | z_<t, x_<t)
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
    # Engines
    # -----------------------------
    engines = [
        {
            "name": "EviTrack(proposal-expand)",
            "engine": EviTrackEngine(
                wm=wm,
                proposal=proposal,
                cfg=EviTrackConfig(K=5, G=3, C=2, tau=1, expand="proposal"),
            ),
            "B": 7,
            "alias_check": False,
        },
        {
            # After the batching refactor, SMC should work with B>1 like EviTrack.
            "name": "SMC(with proposal)",
            "engine": SMCEngine(
                wm=wm,
                proposal=proposal,
                cfg=SMCConfig(N=10, resample_every_step=False),
            ),
            "B": 7,
            "alias_check": True,
        },
        {
            # Keep B=1 unless/until BPF is also refactored to return [B, N] mixtures.
            "name": "BPF(transition-only)",
            "engine": BFPEngine(
                wm=wm,
                cfg=BPFConfig(N=10, resample_every_step=False),
            ),
            "B": 7,
            "alias_check": True,
        },
        {
            # Keep B=1 unless/until SIS is also refactored to return [B, N] mixtures.
            "name": "SIS(with proposal)",
            "engine": SISEngine(
                wm=wm,
                proposal=proposal,
                N=10,
            ),
            "B": 7,
            "alias_check": False,
        },
    ]

    for spec in engines:
        x = generate_synthetic_x(
            T=12,
            dx=wm_cfg.dx,
            B=spec["B"],
            device=DEVICE,
            dtype=DTYPE,
            seed=0,
        )

        state, w, support = run_engine(spec["engine"], x)
        print(f"\n{spec['name']}: OK")
        print("  x.shape          :", tuple(x.shape))
        print("  w.shape          :", tuple(w.shape))
        print("  transition_evals :", state.cost.transition_evals)
        print("  proposal_evals   :", state.cost.proposal_evals)
        print("  emission_evals   :", state.cost.emission_evals)

        if spec["alias_check"] and hasattr(state, "particles"):
            particles = _first_support_group(support)
            test_resampling_no_aliasing(particles)
            print("  resampling alias check: OK")


if __name__ == "__main__":
    main()
