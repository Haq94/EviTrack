# inference/baselines/sis.py

class SISEngine(InferenceEngine):
    def __init__(self, *, wm, proposal, N: int):
        super().__init__(wm=wm, proposal=proposal, cfg=None)
        self.N = N

    def init_state(self, B, device, dtype):
        particles = []
        for _ in range(self.N):
            wm_z_state = self.wm.init_z_state(B, device=device, dtype=dtype)
            wm_x_state = self.wm.init_x_state(B, device=device, dtype=dtype)
            q_z_state = self.proposal.init_z_state(B, device=device, dtype=dtype)
            q_x_state = self.proposal.init_x_state(B, device=device, dtype=dtype)

            z0 = torch.zeros(B, self.wm.dz, device=device, dtype=dtype)
            logw = torch.zeros(B, device=device, dtype=dtype)

            particles.append(
                Particle(
                    z_t=z0,
                    wm_z_state=wm_z_state,
                    wm_x_state=wm_x_state,
                    q_z_state=q_z_state,
                    q_x_state=q_x_state,
                    logw=logw,
                )
            )

        return ParticleState(particles=particles, t=0, cost=CostCounter())

    def step(self, state: ParticleState, x_t: Tensor):
        new_particles = []
        t_new = state.t + 1

        for p in state.particles:
            z_prev = p.z_t if t_new > 1 else None

            q_out = self.proposal.propose(
                B=x_t.shape[0],
                z_prev=z_prev,
                z_state_prev=p.q_z_state,
                x_state_prev=p.q_x_state,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            z_t = q_out["z_t"]
            logq_t = q_out["logq"]
            q_z_state = q_out["z_state_t"]

            if t_new == 1:
                logpzt = self.wm.log_prob_z1(z_t)
            else:
                trans_params = self.wm.transition_params(
                    z_prev=z_prev,
                    z_state_prev=p.wm_z_state,
                )
                logpzt = self.wm.log_prob_transition(z_t, trans_params)

            z_state_curr = self.wm.z_state_curr(p.wm_z_state, z_t)
            emit_params = self.wm.emission_params(
                z_state_curr=z_state_curr,
                x_state_prev=p.wm_x_state,
            )
            logpxt = self.wm.log_prob_emission(x_t, emit_params)

            logw_new = p.logw + logpxt + logpzt - logq_t

            wm_z_state_new = self.wm.update_z_state(p.wm_z_state, z_t)
            wm_x_state_new = self.wm.update_x_state(p.wm_x_state, x_t)
            q_x_state_new = self.proposal.update_x_state(
                x_t=x_t,
                x_state_prev=p.q_x_state,
            )

            new_particles.append(
                Particle(
                    z_t=z_t,
                    wm_z_state=wm_z_state_new,
                    wm_x_state=wm_x_state_new,
                    q_z_state=q_z_state,
                    q_x_state=q_x_state_new,
                    logw=logw_new,
                )
            )

        return ParticleState(new_particles, t_new, state.cost), StepStats(
            t=t_new,
            kept=len(new_particles),
            candidates=len(new_particles),
            cost=state.cost,
        )

    def get_mixture(self, state: ParticleState):
        logw = torch.stack([p.logw.mean() for p in state.particles])
        w = normalize_logweights(logw, dim=0)
        return w, state.particles