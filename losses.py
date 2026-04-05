import jax
import jax.numpy as jnp

def loss_metric(pic, gamma=0.98):
    energy_t = jnp.mean(pic.E_field**2, axis=-1)
    T = energy_t.shape[0]
    w = gamma ** (T - 1 - jnp.arange(T))   # largest weight at final time
    w = w / w.sum()
    return jnp.sum(w * energy_t)

def loss_metric_density_modes(pic, *, modes=(1,2), tail_frac=0.25, lam_tail=0.7, eps=1e-12):
    """
    Penalize unstable density modes over time, emphasizing the tail.

    Assumes:
      - pic.rho is real-space density history with shape (T, Nx)
    """
    rho_k = jnp.fft.rfft(pic.rho.astype(jnp.float32), axis=-1)

    # Select modes and compute mode energy time series
    ms = jnp.array(modes, dtype=jnp.int32)
    sel = rho_k[:, ms]                                  # (T, |modes|) complex
    mode_energy_t = jnp.sum(jnp.abs(sel) ** 2, axis=-1) # (T,)

    # (Optional) normalize to reduce sensitivity to overall density scale
    # Divide by DC^2 (avoid division by 0)
    dc = jnp.abs(rho_k[:, 0]) ** 2 + eps                # (T,)
    mode_energy_t = mode_energy_t / dc

    # Tail emphasis (low-variance "terminal" proxy)
    T = mode_energy_t.shape[0]
    t0 = int(max(0, (1.0 - tail_frac) * T))
    tail = mode_energy_t[t0:]

    return (1.0 - lam_tail) * mode_energy_t.mean() + lam_tail * tail.mean()


def loss_metric_cancel_self_field(pic):
    """
    Direct supervised control target:
      E_ext(t, x) ~= -E_self(t, x)

    This is an overfit-style diagnostic for controller capacity. If the controller
    can represent the mapping from observed state -> cancelling field, then
    mean((E_ext + E_self)^2) should approach zero on the training trajectory.
    """
    if getattr(pic, "E_ext", None) is None:
        raise ValueError("loss_metric_cancel_self_field requires controlled simulation with E_ext.")
    residual = pic.E_ext - (-pic.E_field)
    return jnp.mean(residual**2)

def _smoothmax(x, beta=30.0, eps=1e-12):
    # stable log-mean-exp
    x0 = jnp.max(x)
    return x0 + jnp.log(jnp.mean(jnp.exp(beta * (x - x0))) + eps) / beta

def _rfft_band_energy(pic_field_tx, m_max, *, p_weight=0.0, dtype=jnp.float32):
    """
    Returns A(t) = sum_{m=1..m_max} w_m |hat(field)_m(t)|^2, shape (T,).
    p_weight=0 gives uniform weights; p_weight>0 emphasizes high-k.
    """
    fk = jnp.fft.rfft(pic_field_tx.astype(dtype), axis=-1)      # (T, Nx//2+1)
    band = fk[:, 1:m_max+1]                                     # (T, m_max)
    m = jnp.arange(1, m_max+1, dtype=band.real.dtype)
    w = m**p_weight
    return jnp.sum((jnp.abs(band) ** 2) * w[None, :], axis=-1)   # (T,)

def _normalized_growth_penalty(A_t, *, tail_frac=0.25, eps=1e-12):
    """
    Penalize growth from early -> tail, normalized.
    JIT-safe: t0 computed from static shape, not traced values.
    """
    T = A_t.shape[0]                          # static Python int under jit
    t0 = max(1, int((1.0 - tail_frac) * T))   # static Python int
    t0 = min(t0, T)                           # safety

    head = A_t[:t0]
    tail = A_t[t0:]

    head_mean = jnp.mean(head) + eps
    # if tail empty (can happen if t0==T), just reuse last value
    tail_mean = jnp.mean(tail) + eps if (t0 < T) else (A_t[-1] + eps)

    # penalize positive log-ratio
    return jnp.maximum(0.0, jnp.log(tail_mean / head_mean))


def _normalized_spike_penalty(A_t, *, tail_frac=0.25, beta=30.0, eps=1e-12):
    """
    JIT-safe spike penalty.
    """
    T = A_t.shape[0]
    t0 = max(1, int((1.0 - tail_frac) * T))
    t0 = min(t0, T)

    tail = A_t[t0:] if (t0 < T) else A_t[-1:]     # ensure nonempty
    base = jnp.mean(tail) + eps

    z = A_t / base

    # smooth max via log-mean-exp (stable)
    x0 = jnp.max(z)
    return x0 + jnp.log(jnp.mean(jnp.exp(beta * (z - x0))) + eps) / beta

def loss_metric_stable(pic,
                       *,
                       # --- main objective ---
                       modes=(1,2),
                       tail_frac=0.25,
                       lam_tail=0.7,
                       # --- regularization bands ---
                       m_band_rho=40,
                       m_band_P=40,
                       m_band_E=40,
                       # emphasize high-k for moments to prevent filament spikes
                       p_weight_rho=0.0,
                       p_weight_P=1.0,
                       p_weight_E=1.0,
                       # --- weights (density dominates) ---
                       lam_spike_rho=0.10,
                       lam_spike_P=0.05,
                       lam_spike_E=0.05,
                       lam_growth_rho=0.10,
                       lam_growth_P=0.05,
                       lam_growth_E=0.05,
                       # --- control regularization (optional) ---
                       lam_u_spike=0.01,
                       lam_du_spike=0.05,
                       beta=30.0,
                       eps=1e-12):
    """
    Normalized loss:
      L = L_density_modes
          + small normalized spike+growth penalties on rho/momentum/energy bands
          + small normalized spike penalties on control magnitude and slew (if E_ext exists)
    """

    # 1) Main objective
    L = loss_metric_density_modes(
        pic, modes=modes, tail_frac=tail_frac, lam_tail=lam_tail, eps=eps
    )

    # 2) Spectral-band energies (time series)
    A_rho = _rfft_band_energy(pic.rho,      m_band_rho, p_weight=p_weight_rho)  # (T,)
    A_P   = _rfft_band_energy(pic.momentum, m_band_P,   p_weight=p_weight_P)
    A_E   = _rfft_band_energy(pic.energy,   m_band_E,   p_weight=p_weight_E)

    # 3) Normalized spike penalties (dimensionless, ~O(1))
    L += lam_spike_rho * _normalized_spike_penalty(A_rho, tail_frac=tail_frac, beta=beta, eps=eps)
    L += lam_spike_P   * _normalized_spike_penalty(A_P,   tail_frac=tail_frac, beta=beta, eps=eps)
    L += lam_spike_E   * _normalized_spike_penalty(A_E,   tail_frac=tail_frac, beta=beta, eps=eps)

    # 4) Normalized growth penalties (dimensionless)
    L += lam_growth_rho * _normalized_growth_penalty(A_rho, tail_frac=tail_frac, eps=eps)
    L += lam_growth_P   * _normalized_growth_penalty(A_P,   tail_frac=tail_frac, eps=eps)
    L += lam_growth_E   * _normalized_growth_penalty(A_E,   tail_frac=tail_frac, eps=eps)

    # 5) Control spike / slew (also normalized, optional)
    if getattr(pic, "E_ext", None) is not None and pic.E_ext is not None:
        u = pic.E_ext.astype(jnp.float32)          # (T, Nx)
        u2_t = jnp.mean(u**2, axis=-1)             # (T,)
        L += lam_u_spike * _normalized_spike_penalty(u2_t, tail_frac=tail_frac, beta=beta, eps=eps)

        du = (u[1:] - u[:-1]) / jnp.array(pic.dt, dtype=u.dtype)
        du2_t = jnp.mean(du**2, axis=-1)           # (T-1,)
        L += lam_du_spike * _normalized_spike_penalty(du2_t, tail_frac=tail_frac, beta=beta, eps=eps)

    return L