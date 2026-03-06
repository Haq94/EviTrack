# inference/baselines/sis.py
from .pf import ParticleFilterEngine, ParticleFilterConfig

class SISEngine(ParticleFilterEngine):
    def __init__(self, *, wm, proposal, N: int):
        super().__init__(
            wm=wm,
            proposal=proposal,
            cfg=ParticleFilterConfig(
                N=N,
                proposal_mode="proposal",
                resample=False,
            ),
        )