import json
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import scipy
from typing import Optional, Tuple

def _project_time_rfft_real_constraints(c: jax.Array, Nt: int) -> jax.Array:
    """Enforce the minimal constraints on an rFFT coefficient vector so irfft returns a real signal."""
    c = c.at[0].set(jnp.real(c[0]))
    if Nt % 2 == 0:
        c = c.at[Nt // 2].set(jnp.real(c[Nt // 2]))
    return c


def _rfft_truncated_time_signal(
    coeff_train: jax.Array, Nt: int, n_modes_time: int
) -> jax.Array:
    """
    coeff_train: (n_modes_time,) complex trainable prefix of the time rFFT coefficients.
    Returns: (Nt,) real time signal, with all higher time rFFT modes assumed zero.
    """
    Nt_pos = Nt // 2 + 1
    n_keep = int(min(n_modes_time, Nt_pos))
    full = jnp.zeros((Nt_pos,), dtype=coeff_train.dtype)
    full = full.at[:n_keep].set(coeff_train[:n_keep])
    full = _project_time_rfft_real_constraints(full, Nt)
    return jnp.fft.irfft(full, n=Nt)  # (Nt,), real


class FourierActuator(eqx.Module):
    # Trainable parameters (independent DOFs only)
    # a_hat_train[m, kt] and b_hat_train[m, kt] are *truncated* time-rFFT coefficients for each spatial mode m
    # Shapes:
    #   a_hat_train: (n_modes_space, n_modes_time) complex
    #   b_hat_train: (n_modes_space, n_modes_time) complex, but row m=0 is always zero (no sin at DC space mode)
    a_hat_train: jax.Array
    b_hat_train: jax.Array

    # Hyperparams / configuration
    zero: bool = eqx.field(static=True)
    closed_loop: bool = eqx.field(static=True)

    Nt: int = eqx.field(static=True)
    N_mesh: int = eqx.field(static=True)
    boxsize: float = eqx.field(static=True)

    n_modes_time: int = eqx.field(static=True)   # how many time rFFT bins are trainable (prefix)
    n_modes_space: int = eqx.field(static=True)  # how many spatial Fourier modes (m=0..n_modes_space-1)

    # Optional closed-loop controller params (kept for compatibility with your design)
    K0: Optional[jax.Array] = eqx.field(static=True, default=None)
    u_max: Optional[jax.Array] = eqx.field(static=True, default=None)

    # Optional init metadata (not used in forward; useful to store)
    init_scale: float = eqx.field(static=True, default=0.0)

    def __init__(
        self,
        Nt: int,
        N_mesh: int,
        boxsize: float,
        *,
        n_modes_time: int,
        n_modes_space: int,
        key: Optional[jax.random.PRNGKey] = None,
        init_scale: float = 1e-4,
        zero: bool = False,
        closed_loop: bool = False,
        K0: Optional[jax.Array] = None,
        u_max: Optional[jax.Array] = None,
    ):
        self.Nt = int(Nt)
        self.N_mesh = int(N_mesh)
        self.boxsize = float(boxsize)

        self.n_modes_time = int(n_modes_time)
        self.n_modes_space = int(n_modes_space)

        self.zero = bool(zero)
        self.closed_loop = bool(closed_loop)
        self.K0 = K0
        self.u_max = u_max
        self.init_scale = float(init_scale)

        # Allocate trainable truncated time-rFFT coefficients
        shape = (self.n_modes_space, self.n_modes_time)

        if key is None:
            # deterministic (all zeros) if no key supplied
            a = jnp.zeros(shape, dtype=jnp.complex64)
            b = jnp.zeros(shape, dtype=jnp.complex64)
        else:
            k1, k2, k3, k4 = jax.random.split(key, num=4)
            # small random complex init
            a = init_scale * (
                jax.random.normal(k1, shape, dtype=jnp.float32)
                + 1j * jax.random.normal(k2, shape, dtype=jnp.float32)
            )
            b = init_scale * (
                jax.random.normal(k3, shape, dtype=jnp.float32)
                + 1j * jax.random.normal(k4, shape, dtype=jnp.float32)
            )
            a = a.astype(jnp.complex64)
            b = b.astype(jnp.complex64)

        # Enforce "no sin at m=0" (independent DOFs only)
        b = b.at[0].set(jnp.zeros((self.n_modes_time,), dtype=jnp.complex64))

        # Enforce time rFFT constraints on the *trainable prefix* where applicable:
        # - DC bin (kt=0) must be real
        a = a.at[:, 0].set(jnp.real(a[:, 0]))
        b = b.at[:, 0].set(jnp.real(b[:, 0]))
        # - Nyquist bin is only present if Nt even AND we are training it (kt == Nt//2 is within prefix)
        nyq = self.Nt // 2
        if self.Nt % 2 == 0 and self.n_modes_time > nyq:
            a = a.at[:, nyq].set(jnp.real(a[:, nyq]))
            b = b.at[:, nyq].set(jnp.real(b[:, nyq]))

        self.a_hat_train = a
        self.b_hat_train = b

    def field(self) -> jax.Array:
        """Return E(t, x) with shape (Nt, N_mesh), real."""
        # Spatial grid
        x = jnp.linspace(0.0, self.boxsize, self.N_mesh, endpoint=False)  # (N_mesh,)
        m = jnp.arange(self.n_modes_space)                                # (n_modes_space,)
        k = 2.0 * jnp.pi * m / self.boxsize                               # (n_modes_space,)

        cos_kx = jnp.cos(k[None, :] * x[:, None])                         # (N_mesh, n_modes_space)
        sin_kx = jnp.sin(k[None, :] * x[:, None])                         # (N_mesh, n_modes_space)
        sin_kx = sin_kx.at[:, 0].set(0.0)                                 # enforce no sin at m=0

        # Build time signals a_m(t), b_m(t) from truncated time rFFT coeffs
        # vmap over spatial mode index m
        a_t = jax.vmap(lambda c: _rfft_truncated_time_signal(c, self.Nt, self.n_modes_time))(self.a_hat_train)
        b_t = jax.vmap(lambda c: _rfft_truncated_time_signal(c, self.Nt, self.n_modes_time))(self.b_hat_train)
        # a_t, b_t: (n_modes_space, Nt) -> (Nt, n_modes_space)
        a_t = jnp.swapaxes(a_t, 0, 1)
        b_t = jnp.swapaxes(b_t, 0, 1)

        # Combine time amplitudes with spatial basis
        E = (a_t @ cos_kx.T) + (b_t @ sin_kx.T)                           # (Nt, N_mesh)
        return E  # real

    def __call__(self, n, x=None):
        """Return E[n] (N_mesh,) for open-loop, or closed-loop control if enabled."""
        if self.zero:
            return jnp.zeros(self.N_mesh)

        if self.closed_loop:
            if x is None:
                raise ValueError("closed_loop=True requires state x to be provided.")
            x_rom = x  # replace with ROM mapping
            u = -(self.K0 @ x_rom)
            return jnp.tanh(u)

        # Open-loop: precompute full field and slice
        E_all = self.field()
        return E_all[n.astype(int)]
    
    def get_modes_summary(self):
        """
        Returns paper-style coefficients for the *static* part of the field:
            E(x) = c0 + sum_{m>=1} [ c_m cos(k_m x) + s_m sin(k_m x) ]

        Only meaningful when n_modes_time == 1 (time-DC).
        """
        if self.n_modes_time != 1:
            #raise ValueError("get_modes_summary is only valid for n_modes_time == 1 (static field).")
            return ""

        # Time-DC amplitudes (scalars)
        # a_hat_train[m, 0] and b_hat_train[m, 0] are real by construction
        a0 = jnp.real(self.a_hat_train[:, 0])  # (n_modes_space,)
        b0 = jnp.real(self.b_hat_train[:, 0])  # (n_modes_space,)

        summary = []
        for m in range(self.n_modes_space):
            k_m = 2 * jnp.pi * m / self.boxsize
            if m == 0:
                summary.append({
                    "m": 0,
                    "k": 0.0,
                    "offset": float(a0[0]),
                })
            else:
                summary.append({
                    "m": m,
                    "k": float(k_m),
                    "cos_coeff": float(a0[m]),
                    "sin_coeff": float(b0[m]),
                })
        return summary

    # -----------------------
    # Save / Load
    # -----------------------
    def save_model(self, filename: str):
        with open(filename, "wb") as f:
            hyperparams = {
                "zero": self.zero,
                "closed_loop": self.closed_loop,
                "Nt": self.Nt,
                "N_mesh": self.N_mesh,
                "boxsize": self.boxsize,
                "n_modes_time": self.n_modes_time,
                "n_modes_space": self.n_modes_space,
                "init_scale": self.init_scale,
                # closed-loop extras (may be None)
                "has_K0": self.K0 is not None,
                "has_u_max": self.u_max is not None,
            }
            f.write((json.dumps(hyperparams) + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load_model(cls, filename: str):
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())

            model = cls(
                Nt=hyperparams["Nt"],
                N_mesh=hyperparams["N_mesh"],
                boxsize=hyperparams["boxsize"],
                n_modes_time=hyperparams["n_modes_time"],
                n_modes_space=hyperparams["n_modes_space"],
                key=None,  # we'll overwrite parameters from file
                init_scale=hyperparams.get("init_scale", 0.0),
                zero=hyperparams["zero"],
                closed_loop=hyperparams["closed_loop"],
                K0=None,     # overwritten if present in leaves
                u_max=None,  # overwritten if present in leaves
            )

            return eqx.tree_deserialise_leaves(f, model)

class ModeFeedbackActuator(eqx.Module):
    # -------------------------
    # Static hyperparameters
    # -------------------------
    N_mesh: int = eqx.field(static=True)
    boxsize: float = eqx.field(static=True)

    n_modes_space_in: int = eqx.field(static=True)
    n_modes_space_out: int = eqx.field(static=True)
    init_scale: float = eqx.field(static=True)

    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    use_linear: bool = eqx.field(static=True)
    include_dc: bool = eqx.field(static=True)
    include_density_input: bool = eqx.field(static=True)
    u_max: float | None = eqx.field(static=True)
    e_ext_range: float = eqx.field(static=True)

    zero: bool = eqx.field(static=True)
    closed_loop: bool = eqx.field(static=True)

    # -------------------------
    # Learnable / leaf params
    # -------------------------
    mlp: eqx.Module | None
    K0: jax.Array | None           # (n_out, n_in) complex64 if linear
    dc_value: jax.Array | None     # (1,) float if include_dc else None

    def __init__(self,N_mesh,boxsize,use_linear=False,width=64,depth=2,include_dc=False,include_density_input=False,u_max=None,e_ext_range=0.3,zero=False,closed_loop=True,n_modes_space_in=4,n_modes_space_out=4,init_scale=1.0,*,key):
        self.N_mesh = N_mesh
        self.boxsize = boxsize
        self.include_dc = include_dc
        self.include_density_input = include_density_input
        self.u_max = u_max
        self.e_ext_range = float(e_ext_range)
        if self.e_ext_range <= 0.0:
            raise ValueError("e_ext_range must be positive.")
        self.zero = zero
        self.closed_loop = closed_loop
        self.n_modes_space_in = n_modes_space_in
        self.n_modes_space_out = n_modes_space_out
        self.init_scale = init_scale
        self.use_linear = use_linear
        self.width = width
        self.depth = depth
        n_input_states = 2 if self.include_density_input else 1

        if self.use_linear:
            k1, k2, k3 = jax.random.split(key, num=3)
            shape = (self.n_modes_space_out, n_input_states * self.n_modes_space_in)
            self.K0 = self.init_scale * (
                    jax.random.normal(k1, shape, dtype=jnp.float64)
                    + 1j * jax.random.normal(k2, shape, dtype=jnp.float64)
                )
            self.K0 = self.K0.astype(jnp.complex64)
            if self.include_dc:
                self.dc_value = self.init_scale * jax.random.normal(k3, (1,), dtype=jnp.float64)
            else:
                self.dc_value = None
            self.mlp = None
        else:
            self.K0 = None
            self.dc_value = None
            in_size = n_input_states * (2 * self.n_modes_space_in + (1 if self.include_dc else 0))
            out_size = 2*self.n_modes_space_out
            if self.include_dc:
                out_size += 1
            self.mlp = eqx.nn.MLP(
                in_size=in_size,
                out_size=out_size,
                width_size=width,
                depth=depth,
                activation=jnn.tanh,
                key=key,
            )

    def __call__(self, n: int, *, state=None):
        """Return E_ext(x) on the grid for step index n.

        Parameters
        ----------
        n : int
            time-step index (kept for interface compatibility; not used here).
        state : complex array or tuple(complex array, complex array)
            One-sided rFFT coefficients at current step. When tuple, expects
            (density_spectrum, first_moment_spectrum).
\
        Returns
        -------
        E_ext : float array, shape (N_mesh,)
        """
        #jax.debug.print("Current gain: {gain}", gain=jnp.linalg.norm(self.K0))
        if self.zero:
            return jnp.zeros((self.N_mesh,), dtype=jnp.float32)

        rho_state = None
        mom_state = state
        if isinstance(state, tuple):
            if len(state) != 2:
                raise ValueError("If tuple state is provided, expected (density, first_moment).")
            rho_state, mom_state = state
        if mom_state is None:
            raise ValueError("closed_loop actuator requires `state` input.")
        if self.include_density_input and rho_state is None:
            raise ValueError(
                "include_density_input=True requires tuple state=(density_spectrum, first_moment_spectrum)."
            )

        def _encode_observed_modes(spec):
            s = spec[: self.n_modes_space_in + 1]
            if self.include_dc:
                return jnp.concatenate(
                    [
                        jnp.array([jnp.real(s[0])], dtype=jnp.float64),
                        jnp.real(s[1:]).astype(jnp.float64),
                        jnp.imag(s[1:]).astype(jnp.float64),
                    ],
                    axis=0,
                )
            return jnp.concatenate(
                [
                    jnp.real(s[1:]).astype(jnp.float64),
                    jnp.imag(s[1:]).astype(jnp.float64),
                ],
                axis=0,
            )

        if self.use_linear:
            mom_meas = mom_state[1 : self.n_modes_space_in + 1]
            if self.include_density_input:
                rho_meas = rho_state[1 : self.n_modes_space_in + 1]
                meas = jnp.concatenate([rho_meas, mom_meas], axis=0)
            else:
                meas = mom_meas

            # Feedback in Fourier domain: u_m = -K * meas
            u_m = (-self.K0) @ meas

            # Build one-sided spectrum for E_ext and invert to real space
            spec = jnp.zeros((self.N_mesh // 2 + 1,), dtype=jnp.complex64)

            if self.include_dc:
                spec = spec.at[0].set(jnp.array(self.dc_value[0], dtype=jnp.complex64))

            spec = spec.at[1:self.n_modes_space_out+1].set(u_m.astype(jnp.complex64))
        else:
            if self.mlp is None:
                raise ValueError("use_linear=False but mlp is None.")

            x = _encode_observed_modes(mom_state)
            if self.include_density_input:
                x = jnp.concatenate([_encode_observed_modes(rho_state), x], axis=0)

            u_m = self.mlp(x)  # (2*n_out [+1]) real

            spec = jnp.zeros((self.N_mesh // 2 + 1,), dtype=jnp.complex64)

            if self.include_dc:
                spec = spec.at[0].set(jnp.array(u_m[0], dtype=jnp.complex64))
                modes = u_m[1:self.n_modes_space_out+1] + 1j * u_m[self.n_modes_space_out+1:]
                spec = spec.at[1:self.n_modes_space_out+1].set(modes.astype(jnp.complex64))
            else:
                modes = u_m[:self.n_modes_space_out] + 1j * u_m[self.n_modes_space_out:]
                spec = spec.at[1:self.n_modes_space_out+1].set(modes.astype(jnp.complex64))       

        E_ext = jnp.fft.irfft(spec, n=self.N_mesh).real  # -> real-valued (N_mesh,)
        E_ext = self.e_ext_range * jnp.tanh(E_ext / self.e_ext_range)
        #jax.debug.print("Output: {out}", out=jnp.linalg.norm(E_ext))
        return E_ext

    # -----------------------
    # Save / Load
    # -----------------------
    def save_model(self, filename: str):
        with open(filename, "wb") as f:
            hyperparams = {
                "zero": bool(self.zero),
                "closed_loop": bool(self.closed_loop),
                "N_mesh": int(self.N_mesh),
                "boxsize": float(self.boxsize),
                "width": int(self.width),
                "depth": int(self.depth),
                "n_modes_space_in": int(self.n_modes_space_in),
                "n_modes_space_out": int(self.n_modes_space_out),
                "init_scale": float(self.init_scale),
                "include_dc": bool(self.include_dc),
                "include_density_input": bool(self.include_density_input),
                "use_linear": bool(self.use_linear),
                "u_max": float(self.u_max) if self.u_max is not None else None,
                "e_ext_range": float(self.e_ext_range),
            }
            f.write((json.dumps(hyperparams) + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load_model(cls, filename: str):
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())

            # IMPORTANT: build the same structure as saved:
            # use_linear/include_dc decide which leaves exist (K0/dc_value vs mlp).
            model = cls(
                N_mesh=hyperparams["N_mesh"],
                boxsize=hyperparams["boxsize"],
                use_linear=hyperparams.get("use_linear", False),
                width=hyperparams.get("width", 64),
                depth=hyperparams.get("depth", 2),
                include_dc=hyperparams.get("include_dc", False),
                include_density_input=hyperparams.get("include_density_input", False),
                u_max=hyperparams.get("u_max", None),
                e_ext_range=hyperparams.get("e_ext_range", 0.3),
                zero=hyperparams.get("zero", False),
                closed_loop=hyperparams.get("closed_loop", True),
                n_modes_space_in=hyperparams.get("n_modes_space_in", 4),
                n_modes_space_out=hyperparams.get("n_modes_space_out", 4),
                init_scale=hyperparams.get("init_scale", 1.0),
                key=jax.random.PRNGKey(0),  # overwritten by deserialised leaves
            )

            return eqx.tree_deserialise_leaves(f, model)

class DissipativeModeFeedbackActuator(eqx.Module):
    # -------------------------
    # Static hyperparameters
    # -------------------------
    N_mesh: int = eqx.field(static=True)
    boxsize: float = eqx.field(static=True)

    n_modes_space_in: int = eqx.field(static=True)
    n_modes_space_out: int = eqx.field(static=True)

    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    include_dc: bool = eqx.field(static=True)
    u_max: float | None = eqx.field(static=True)
    e_ext_range: float = eqx.field(static=True)

    zero: bool = eqx.field(static=True)
    closed_loop: bool = eqx.field(static=True)

    # -------------------------
    # Learnable / leaf params
    # -------------------------
    mlp: eqx.Module | None

    def __init__(
        self,
        N_mesh,
        boxsize,
        *,
        width=64,
        depth=2,
        include_dc=False,
        u_max=None,
        e_ext_range=0.3,
        zero=False,
        closed_loop=True,
        n_modes_space_in=4,
        n_modes_space_out=4,
        key,
    ):
        self.N_mesh = int(N_mesh)
        self.boxsize = float(boxsize)
        self.include_dc = bool(include_dc)
        self.u_max = u_max
        self.e_ext_range = float(e_ext_range)
        if self.e_ext_range <= 0.0:
            raise ValueError("e_ext_range must be positive.")
        self.zero = bool(zero)
        self.closed_loop = bool(closed_loop)
        self.n_modes_space_in = int(n_modes_space_in)
        self.n_modes_space_out = int(n_modes_space_out)
        self.width = int(width)
        self.depth = int(depth)

        # INPUT: Re/Im of observed modes (plus optional DC real)
        in_size = 2 * self.n_modes_space_in + (1 if self.include_dc else 0)

        # OUTPUT: one REAL gain per complex output mode (plus optional DC gain)
        out_size = self.n_modes_space_out + (1 if self.include_dc else 0)

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=self.width,
            depth=self.depth,
            activation=jnn.tanh,
            key=key,
        )

    def __call__(self, n: int, *, state=None):
        """Return E_ext(x) on the grid for step index n.

        Parameters
        ----------
        n : int
            timestep index (kept for interface compatibility)
        state : complex array, shape (N_mesh//2+1,)
            One-sided rFFT coefficients of momentum/current J(x) at current step.

        Returns
        -------
        E_ext : float array, shape (N_mesh,)
        """
        if self.zero:
            return jnp.zeros((self.N_mesh,), dtype=jnp.float64)
        if self.mlp is None:
            raise ValueError("MLP is required!")
        if state is None:
            raise ValueError("closed_loop actuator requires `state` (rFFT of momentum/current).")

        # -------------------------
        # Build REAL observation vector x_in from observed modes
        # -------------------------
        # Observe J modes: 1..n_in (and optional DC)
        Jin = state[: self.n_modes_space_in + 1]  # includes DC at index 0

        if self.include_dc:
            # x_in = [J0_real, Re(J1..Jn), Im(J1..Jn)]
            x_in = jnp.concatenate(
                [
                    jnp.array([jnp.real(Jin[0])], dtype=jnp.float64),
                    jnp.real(Jin[1:]).astype(jnp.float64),
                    jnp.imag(Jin[1:]).astype(jnp.float64),
                ],
                axis=0,
            )  # shape: 1 + 2*n_in
        else:
            # x_in = [Re(J1..Jn), Im(J1..Jn)]
            x_in = jnp.concatenate(
                [
                    jnp.real(Jin[1:]).astype(jnp.float64),
                    jnp.imag(Jin[1:]).astype(jnp.float64),
                ],
                axis=0,
            )  # shape: 2*n_in

        # -------------------------
        # Predict nonnegative gains (one per output mode [+ optional DC])
        # -------------------------
        gains = jnn.softplus(self.mlp(x_in)).astype(jnp.float64)  # (n_out [+1]), >= 0

        # -------------------------
        # Build dissipative control: E_m = -alpha_m * J_m  (controlled band)
        # -------------------------
        spec = jnp.zeros((self.N_mesh // 2 + 1,), dtype=jnp.complex64)

        Jout = state[: self.n_modes_space_out + 1]  # includes DC at index 0

        if self.include_dc:
            alpha0 = gains[0]
            J0 = jnp.real(Jout[0]).astype(jnp.float64)   # real
            E0 = (-alpha0 * J0).astype(jnp.float64)      # real
            spec = spec.at[0].set(E0.astype(jnp.complex64))

            alpha = gains[1 : 1 + self.n_modes_space_out]  # (n_out,)
            Jm = Jout[1 : 1 + self.n_modes_space_out]      # (n_out,) complex
            # Because of the sign conventions in PICSimulator, dissipative means + sign here
            Em = (alpha * Jm).astype(jnp.complex64)
            spec = spec.at[1 : 1 + self.n_modes_space_out].set(Em)
        else:
            alpha = gains[: self.n_modes_space_out]        # (n_out,)
            Jm = Jout[1 : 1 + self.n_modes_space_out]      # (n_out,) complex
            # Because of the sign conventions in PICSimulator, dissipative means + sign here
            Em = (alpha * Jm).astype(jnp.complex64)
            spec = spec.at[1 : 1 + self.n_modes_space_out].set(Em)

        # Back to real space field
        e_ext = jnp.fft.irfft(spec, n=self.N_mesh).real.astype(jnp.float64)
        return self.e_ext_range * jnp.tanh(e_ext / self.e_ext_range)

    # -----------------------
    # Save / Load
    # -----------------------
    def save_model(self, filename: str):
        with open(filename, "wb") as f:
            hyperparams = {
                "zero": bool(self.zero),
                "closed_loop": bool(self.closed_loop),
                "N_mesh": int(self.N_mesh),
                "boxsize": float(self.boxsize),
                "width": int(self.width),
                "depth": int(self.depth),
                "n_modes_space_in": int(self.n_modes_space_in),
                "n_modes_space_out": int(self.n_modes_space_out),
                "include_dc": bool(self.include_dc),
                "u_max": float(self.u_max) if self.u_max is not None else None,
                "e_ext_range": float(self.e_ext_range),
            }
            f.write((json.dumps(hyperparams) + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load_model(cls, filename: str):
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())

            # IMPORTANT: build the same structure as saved:
            # use_linear/include_dc decide which leaves exist (K0/dc_value vs mlp).
            model = cls(
                N_mesh=hyperparams["N_mesh"],
                boxsize=hyperparams["boxsize"],
                width=hyperparams.get("width", 64),
                depth=hyperparams.get("depth", 2),
                include_dc=hyperparams.get("include_dc", False),
                u_max=hyperparams.get("u_max", None),
                e_ext_range=hyperparams.get("e_ext_range", 0.3),
                zero=hyperparams.get("zero", False),
                closed_loop=hyperparams.get("closed_loop", True),
                n_modes_space_in=hyperparams.get("n_modes_space_in", 4),
                n_modes_space_out=hyperparams.get("n_modes_space_out", 4),
                key=jax.random.PRNGKey(0),  # overwritten by deserialised leaves
            )

            return eqx.tree_deserialise_leaves(f, model)

def ctrb(A, B):
    n = A.shape[0]
    blocks = []
    AB = B
    for _ in range(n):
        blocks.append(AB)
        AB = A @ AB
    return jnp.concatenate(blocks, axis=1)

def continuous_lqr(A, B, Q=None, R=None):
    """
    Continuous-time LQR for xdot = A x + B u.
    Returns K, P, eigvals(A-BK)
    """
    A_np = jnp.asarray(A)
    B_np = jnp.asarray(B)
    n = A_np.shape[0]
    m = B_np.shape[1]

    if Q is None:
        Q_np = jnp.eye(n)
    else:
        Q_np = jnp.asarray(Q)

    if R is None:
        R_np = jnp.eye(m)
    else:
        R_np = jnp.asarray(R)

    # Solve CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
    P = scipy.linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)

    # K = R^{-1} B^T P
    K = jnp.linalg.solve(R_np, B_np.T @ P)

    eig_cl = jnp.linalg.eigvals(A_np - B_np @ K)
    return jnp.asarray(K), jnp.asarray(P), jnp.asarray(eig_cl)

def discrete_lqr(A, B, Q=None, R=None):
    """
    Discrete-time LQR for x_{k+1} = A x_k + B u_k.
    Minimizes sum_{k=0}^\infty (x_k^T Q x_k + u_k^T R u_k).
    Returns K, P, eigvals(A - B K)
    """
    A_np = jnp.asarray(A)
    B_np = jnp.asarray(B)
    n = A_np.shape[0]
    m = B_np.shape[1]

    if Q is None:
        Q_np = jnp.eye(n)
    else:
        Q_np = jnp.asarray(Q)

    if R is None:
        R_np = jnp.eye(m)
    else:
        R_np = jnp.asarray(R)

    # Solve DARE: P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q
    # scipy returns a NumPy array; we'll wrap back to jnp
    P = scipy.linalg.solve_discrete_are(
        jnp.asarray(A_np), jnp.asarray(B_np), jnp.asarray(Q_np), jnp.asarray(R_np)
    )

    P = jnp.asarray(P)

    # K = (R + B^T P B)^{-1} (B^T P A)
    S = R_np + B_np.T @ P @ B_np
    K = jnp.linalg.solve(S, B_np.T @ P @ A_np)

    eig_cl = jnp.linalg.eigvals(A_np - B_np @ K)
    return jnp.asarray(K), jnp.asarray(P), jnp.asarray(eig_cl)