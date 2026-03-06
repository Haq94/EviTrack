# inference/baselines/smc.py
from dataclasses import dataclass
from .pf import ParticleFilterEngine, ParticleFilterConfig

@dataclass
class SMCConfig:
    N: int
    resample_every_step: bool = True
    ess_threshold_frac: float = 0.5

class SMCEngine(ParticleFilterEngine):
    def __init__(self, *, wm, proposal, cfg: SMCConfig):
        super().__init__(
            wm=wm,
            proposal=proposal,
            cfg=ParticleFilterConfig(
                N=cfg.N,
                proposal_mode="proposal",
                resample=True,
                resample_every_step=cfg.resample_every_step,
                ess_threshold_frac=cfg.ess_threshold_frac,
            ),
        )
