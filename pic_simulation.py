import jax
import jax.numpy as jnp
import equinox as eqx

class PICSimulation(eqx.Module):
    boxsize: float = eqx.field(static=True)
    N_particles: int = eqx.field(static=True)
    N_mesh: int = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    n0: float = eqx.field(static=True)
    vb: float = eqx.field(static=True)
    vth: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    n_steps: float = eqx.field(static=True)
    m: float = eqx.field(static=True)
    q: float = eqx.field(static=True)
    eps0: float = eqx.field(static=True)

    # Frequency
    k: jax.Array
    nonzero_k: jax.Array
    k_masked: jax.Array
    k_masked_inv2: jax.Array

    # Trajectories
    ts: jax.Array
    positions: jax.Array
    velocities: jax.Array
    accelerations: jax.Array
    E_field: jax.Array
    E_ext: jax.Array
    rho: jax.Array
    higher_moments: bool = eqx.field(static=True)
    momentum: jax.Array
    energy: jax.Array

    def __init__(self, boxsize, N_particles, N_mesh, n0, vb, vth, dt, t1, m=1, q=1, eps0=1, t0=0, higher_moments=False):
        self.boxsize = boxsize
        self.N_particles = N_particles
        self.N_mesh = N_mesh
        self.dx = self.boxsize / self.N_mesh
        self.n0 = n0 # Background number density
        self.vb = vb
        self.vth = vth
        self.dt = dt
        self.t0 = t0
        self.t1 = t1
        self.n_steps = int(jnp.floor((self.t1-self.t0) / dt))
        self.m = m
        self.q = q
        self.eps0 = eps0

        # Frequencies
        self.k = 2 * jnp.pi * jnp.fft.fftfreq(self.N_mesh, d=self.dx)  # Wavenumbers
        self.nonzero_k = self.k != 0
        self.k_masked = jnp.where(self.nonzero_k, self.k, 1.0)
        self.k_masked_inv2 = 1.0/self.k_masked**2

        # Trajectories
        self.ts = self.t0 + dt * jnp.arange(self.n_steps)
        self.positions = None
        self.velocities = None
        self.accelerations = None
        self.E_field = None
        self.E_ext = None
        self.rho = None
        self.higher_moments = higher_moments
        self.momentum = None
        self.energy = None
    
    def create_y0(
            self,
            key,
            pos_sample: bool = False,
            *,
            eps: float = 1e-2,
            m_min: int = 1,
            m_max: int = 4,
            random_phase: bool = True,
            random_amp: bool = True,
            max_ar_iters: int = 50,
        ):
        """
        Create initial conditions (pos, vel) with a tiny density perturbation:

            p(x) ∝ 1 + Σ_{m=m_min}^{m_max} a_m cos(k_m x + φ_m),
            k_m = 2π m / L,
            a_m ~ Uniform(-eps, eps) if random_amp else a_m = eps,
            φ_m ~ Uniform(0, 2π) if random_phase else φ_m = 0.

        eps is the maximum amplitude magnitude for each mode.

        - If pos_sample=True: sample positions via accept-reject from p(x) (approx; exact for given p).
        - If pos_sample=False: "quiet start" positions with a small displacement map that imprints the same
        density modulation to first order in eps (cheap + low noise).
        """
        key_pos, key_vel, key_pert = jax.random.split(key, 3)

        N = self.N_particles
        L = self.boxsize

        # Modes to include
        m_min = int(m_min)
        m_max = int(m_max)
        if m_min < 1:
            raise ValueError("m_min must be >= 1 (exclude DC).")
        if m_max < m_min:
            raise ValueError("m_max must be >= m_min.")
        ms = jnp.arange(m_min, m_max + 1, dtype=jnp.float64)     # (M,)
        ks = 2.0 * jnp.pi * ms / jnp.array(L, dtype=jnp.float64) # (M,)
        M = ms.shape[0]

        # Random amplitudes and phases
        if random_amp:
            key_pert, kamp = jax.random.split(key_pert)
            amps = jax.random.uniform(kamp, (M,), minval=-eps, maxval=eps, dtype=jnp.float64)
        else:
            amps = jnp.full((M,), eps, dtype=jnp.float64)

        if random_phase:
            key_pert, kphi = jax.random.split(key_pert)
            phis = jax.random.uniform(kphi, (M,), minval=0.0, maxval=2.0 * jnp.pi, dtype=jnp.float64)
        else:
            phis = jnp.zeros((M,), dtype=jnp.float64)

        # Safety: ensure p(x) stays positive for accept-reject.
        # A sufficient condition is Σ |a_m| < 1.
        sum_abs = jnp.sum(jnp.abs(amps))
        # If too large, scale down (keeps distribution valid)
        scale = jnp.minimum(1.0, 0.99 / (sum_abs + 1e-12))
        amps = amps * scale

        # Helper: compute modulation S(x) = Σ a_m cos(k_m x + φ_m)
        def S(x):
            # x: (N,) float64
            # returns: (N,) float64
            x = x[:, None]  # (N,1)
            return jnp.sum(amps[None, :] * jnp.cos(ks[None, :] * x + phis[None, :]), axis=1)

        # -------------------------
        # Positions with perturbation
        # -------------------------
        if pos_sample:
            # Accept-reject from p(x) ∝ 1 + S(x)
            # Accept prob = (1 + S(x)) / (1 + max(S)) where max(S) <= Σ|a_m|.
            Smax = jnp.sum(jnp.abs(amps))  # safe upper bound on max |S|
            denom = 1.0 + Smax

            key_ar, key_u = jax.random.split(key_pos, 2)

            def ar_round(carry, _):
                key_ar, key_u, xs, filled = carry
                key_ar, kx = jax.random.split(key_ar)
                key_u, ku = jax.random.split(key_u)

                x_prop = jax.random.uniform(kx, (N,), minval=0.0, maxval=L, dtype=jnp.float64)
                u = jax.random.uniform(ku, (N,), minval=0.0, maxval=1.0, dtype=jnp.float64)

                p = (1.0 + S(x_prop)) / denom
                accept = u < p

                x_acc = x_prop[accept]
                n_acc = x_acc.shape[0]

                space = N - filled
                take = jnp.minimum(space, n_acc)

                x_acc_pad = jnp.pad(x_acc, (0, N - n_acc))
                xs = xs.at[filled:filled + take].set(x_acc_pad[:take])
                filled = filled + take
                return (key_ar, key_u, xs, filled), None

            xs0 = jnp.zeros((N,), dtype=jnp.float64)
            carry0 = (key_ar, key_u, xs0, jnp.array(0, dtype=jnp.int32))

            (key_ar, key_u, xs, filled), _ = jax.lax.scan(
                ar_round, carry0, xs=None, length=max_ar_iters
            )

            # Fallback fill if not fully accepted (rare for tiny eps)
            key_ar, kfill = jax.random.split(key_ar)
            x_fill = jax.random.uniform(kfill, (N,), minval=0.0, maxval=L, dtype=jnp.float64)
            xs = jnp.where(jnp.arange(N) < filled, xs, x_fill)

            pos = xs[:, None].astype(jnp.float64)

        else:
            # Quiet start baseline: permuted grid
            idx = jax.random.permutation(key_pos, N)
            x0 = (idx.astype(jnp.float64) * (jnp.array(L, dtype=jnp.float64) / jnp.array(N, dtype=jnp.float64)))  # (N,)

            # Displacement map:
            # Choose g(x) = Σ (a_m / k_m) sin(k_m x + φ_m)
            # Then dx/dx0 = 1 + Σ a_m cos(k_m x0 + φ_m) = 1 + S(x0)
            # Hence rho(x) ∝ 1/(dx/dx0) ≈ 1 - S(x0).
            # To get +S in rho to first order, use x = x0 - g(x0).
            x0_col = x0[:, None]  # (N,1)
            g = jnp.sum((amps[None, :] / ks[None, :]) * jnp.sin(ks[None, :] * x0_col + phis[None, :]), axis=1)  # (N,)
            x = jnp.mod(x0 - g, jnp.array(L, dtype=jnp.float64))

            pos = x[:, None].astype(jnp.float64)

        # -------------------------
        # Velocities (unchanged)
        # -------------------------
        vel = self.vth * jax.random.normal(key_vel, (N, 1), dtype=jnp.float64) + jnp.array(self.vb, dtype=jnp.float64)
        Nh = N // 2
        vel = vel.at[Nh:, :].set(-vel[Nh:, :])
        vel = vel - jnp.mean(vel)

        return (pos, vel)

    def cic_deposition(self, pos, vel=None):
        pos = jnp.mod(pos, self.boxsize)

        x = pos / self.dx
        j = jnp.floor(x).astype(jnp.int32)
        j = jnp.mod(j, self.N_mesh)
        jp1 = jnp.mod(j + 1, self.N_mesh)
        frac = x - j.astype(x.dtype)
        weight_j = 1.0 - frac
        weight_jp1 = frac
        w0 = self.q * self.n0 * (self.boxsize / self.N_particles) / self.dx

        def deposit(q=None):
            if q is None:
                g = jax.ops.segment_sum(weight_j[:, 0], j[:, 0], num_segments=self.N_mesh)
                g += jax.ops.segment_sum(weight_jp1[:, 0], jp1[:, 0], num_segments=self.N_mesh)
            else:
                g = jax.ops.segment_sum((weight_j * q)[:, 0], j[:, 0], num_segments=self.N_mesh)
                g += jax.ops.segment_sum((weight_jp1 * q)[:, 0], jp1[:, 0], num_segments=self.N_mesh)
            return g * w0
    
        moments = deposit()[:,None]
        if self.higher_moments:
            momentum = deposit(self.m * vel)[:,None]
            energy = deposit(0.5 * self.m * vel**2)[:,None]
            moments = jnp.concatenate((moments,momentum,energy),axis=-1)
        return moments, j, jp1, weight_j, weight_jp1

    def poisson_solver(self, rho):
        rho_k = jnp.fft.fft(rho)
        rho_k = rho_k.at[0].set(0) # Enforce quasineutrality
        phi_k = jnp.where(self.nonzero_k, -rho_k*self.k_masked_inv2, 0.0)
        E_k = -1j * self.k * phi_k / self.eps0  # Electric field in k-space
        E = jnp.fft.ifft(E_k).real  # Electric field in real space
        return E, rho_k

    def cic_gather(self, y, E_grid, j, jp1, weight_j, weight_jp1, E_ext=None):
        pos, vel, acc = y
        # Interpolate grid value onto particle locations
        E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]

        # Add external electric field
        if E_ext is not None:
            E += weight_j * E_ext[j] + weight_jp1 * E_ext[jp1]

        return E

    def step(self, y, n, E_control=None):
        pos, vel, acc, E_field, E_ext, moments = y

        # (1/2) kick
        vel += acc * self.dt / 2.0

        # drift (and apply periodic boundary conditions)
        pos += vel * self.dt
        pos = jnp.mod(pos, self.boxsize)

        moments, j, jp1, weight_j, weight_jp1 = self.cic_deposition(pos, vel)
        E_grid, rho_k = self.poisson_solver(moments[:,0])

        E_ext = 0
        if E_control is None:
            E_ext = None
        else:
            if E_control.closed_loop:
                E_ext = E_control(
                    n,
                    state=(jnp.fft.rfft(moments[:, 0]), jnp.fft.rfft(moments[:, 1])),
                )
            else:
                E_ext = E_control(n)

        E = self.cic_gather((pos, vel, acc), E_grid, j, jp1, weight_j, weight_jp1, E_ext=E_ext)

        # update accelerations
        acc = -self.q*E/self.m

        # (1/2) kick
        vel += acc * self.dt / 2.0

        return pos, vel, acc, E_grid, E_ext, moments

    def run_simulation(self, y0, E_control=None):
        pos, vel = y0

        pos = jnp.mod(pos, self.boxsize)

        moments, j, jp1, weight_j, weight_jp1 = self.cic_deposition(pos, vel)
        E_grid, rho_k = self.poisson_solver(moments[:,0])

        E_ext = 0
        if E_control is None:
            E_ext = None
        else:
            if E_control.closed_loop:
                E_ext = E_control(
                    jnp.asarray(0),
                    state=(jnp.fft.rfft(moments[:, 0]), jnp.fft.rfft(moments[:, 1])),
                )
            else:
                E_ext = E_control(jnp.asarray(0))

        E = self.cic_gather((pos,vel,jnp.zeros_like(pos)), E_grid, j, jp1, weight_j, weight_jp1, E_ext=E_ext)

        acc = -self.q*E/self.m

        y0 = (pos, vel, acc, E_grid, E_ext, moments)

        def step_fn(y, n):
            y_next = self.step(y, n, E_control=E_control)
            return y_next, y_next

        _, outs = jax.lax.scan(step_fn, y0, xs=jnp.arange(len(self.ts)), length=self.n_steps)

        pos_traj, vel_traj, acc_traj, E_traj, Eext_traj, moments_traj = outs

        new_obj = None
        if self.higher_moments:
            new_obj = eqx.tree_at(
                lambda s: (s.positions, s.velocities, s.accelerations, s.E_field, s.E_ext, s.rho, s.momentum, s.energy),
                self,
                (pos_traj.squeeze(), vel_traj.squeeze(), acc_traj.squeeze(), E_traj, Eext_traj, moments_traj[:,:,0], moments_traj[:,:,1], moments_traj[:,:,2]),
                is_leaf=lambda x: x is None,
            )
        else:
            new_obj = eqx.tree_at(
                lambda s: (s.positions, s.velocities, s.accelerations, s.E_field, s.E_ext, s.rho),
                self,
                (pos_traj.squeeze(), vel_traj.squeeze(), acc_traj.squeeze(), E_traj, Eext_traj, moments_traj[:,:,0]),
                is_leaf=lambda x: x is None,
            )
        return new_obj