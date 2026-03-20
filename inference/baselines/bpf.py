# inference/baselines/bpf.py
from dataclasses import dataclass
from .pf import ParticleFilterEngine, ParticleFilterConfig

@dataclass
class BPFConfig:
    N: int
    resample_every_step: bool = True
    ess_threshold_frac: float = 0.5

class BPFEngine(ParticleFilterEngine):
    def __init__(self, *, wm, cfg: BPFConfig):
        super().__init__(
            wm=wm,
            proposal=None,
            cfg=ParticleFilterConfig(
                N=cfg.N,
                proposal_mode="transition",
                resample=True,
                resample_every_step=cfg.resample_every_step,
                ess_threshold_frac=cfg.ess_threshold_frac,
            ),
        )