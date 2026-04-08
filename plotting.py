import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML

def _maybe_save_or_show(fig, save_path=None, dpi=200):
    """If save_path is None -> show; else save and close."""
    if save_path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

def scatter_animation(
    ts,
    pos_history,
    vel_history,
    Nh,
    boxsize,
    fps=30,
    k=1,
    vlim=(-6, 6),
    save_path=None,
    title_fmt="t = {t:.1f}",
):
    """
    Display an animation of (x,v) scatter plots in a Jupyter notebook,
    and optionally save it as an .mp4 file.

    Parameters
    ----------
    ts          : (Nt,) array
    pos_history : (Nt, Np) array
    vel_history : (Nt, Np) array
    Nh          : int   # first Nh particles blue, rest red
    boxsize     : float # x-axis limit [0, boxsize]
    fps         : int   # frames per second
    k           : int   # downsample factor for frames
    vlim        : tuple # (vmin, vmax)
    save_path   : str or None, e.g. "simulation.mp4"
    title_fmt   : str
    """
    Nt, Np = pos_history.shape
    assert vel_history.shape == (Nt, Np), "pos/vel shapes must match"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, boxsize)
    ax.set_ylim(*vlim)
    ax.set_xlabel("x")
    ax.set_ylabel("v")

    sc_blue = ax.scatter([], [], s=0.4, color="blue", alpha=0.5)
    sc_red  = ax.scatter([], [], s=0.4, color="red",  alpha=0.5)

    empty_offsets = jnp.empty((0, 2), dtype=float)  # <-- key fix

    def init():
        sc_blue.set_offsets(empty_offsets)
        sc_red.set_offsets(empty_offsets)
        ax.set_title(title_fmt.format(t=ts[0]))
        return sc_blue, sc_red

    def update(t):
        # Build offsets as (N,2) arrays
        blue_offsets = jnp.column_stack([pos_history[t, :Nh], vel_history[t, :Nh]]) if Nh > 0 else empty_offsets
        red_offsets  = jnp.column_stack([pos_history[t, Nh:], vel_history[t, Nh:]]) if Nh < Np else empty_offsets

        sc_blue.set_offsets(blue_offsets)
        sc_red.set_offsets(red_offsets)
        ax.set_title(title_fmt.format(t=ts[t]))
        return sc_blue, sc_red

    # interval expects milliseconds; ensure it's an int
    frames = range(0, Nt, k)
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         blit=True, interval=int(1000 / fps))

    if save_path:
        # Requires ffmpeg installed
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        writer = FFMpegWriter(
                fps=fps,
                codec="libvpx-vp9",   # or "libvpx-vp9"
                bitrate=-1
            )
        anim.save(save_path, writer=writer)
        print(f"Saved animation to {save_path}")

    plt.close(fig)  # avoid duplicate static image
    return HTML(anim.to_jshtml())


def plot_pde_solution(ts, y, boxsize, name="", label="", save_path=None):
    Nt, Nx = y.shape

    # If you know these, use them. Otherwise this is a sensible default.
    L = boxsize
    x = jnp.linspace(0.0, L, Nx, endpoint=False)

    # --- 1) Space–time heatmap (PDE-style) ---
    fig = plt.figure(figsize=(8, 4))
    im = plt.imshow(
        y,
        origin="lower",                 # puts early time at bottom
        aspect="auto",                  # avoid squishing
        extent=[x[0], x[-1] + (x[1]-x[0]), ts[0], ts[-1]],  # x on horizontal, t on vertical
        interpolation="nearest",         # or "bilinear" if you prefer smoother look
    )
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(name)
    plt.colorbar(im, label=label)
    plt.tight_layout()

    _maybe_save_or_show(fig, save_path=save_path)


def plot_modes(ts, y, max_mode_spect, max_mode_time, boxsize, num=4, name="", label="", zero_mean=True, save_path=None):
    """
    y: real signal of shape (Nt, Nx). First axis time, second axis space.
    Plots magnitude/power spectra at a few time indices, and low-mode growth over time.
    """
    Nt, N_mesh = y.shape

    # Real-input FFT along space: shape (Nt, Nx//2 + 1)
    Y = jnp.fft.rfft(y, axis=-1)
    if zero_mean:
        Y = Y.at[:, 0].set(0)

    K = Y.shape[-1]  # = N_mesh//2 + 1

    # max_mode is an index into rFFT bins (m=1..), so clamp to available bins
    max_mode_time = int(min(max_mode_time, K - 1))
    max_mode_spect = int(min(max_mode_spect, K - 1))

    # Correct frequency axis: must use the *original* spatial length N_mesh
    dx = boxsize / N_mesh
    modes = jnp.fft.rfftfreq(N_mesh, d=dx)  # shape (K,)

    # Choose which modes to display in the spectrum plots.
    # (Include 0 for completeness; it'll be 0 if zero_mean=True.)
    show = slice(0, max_mode_spect + 1)

    indices = list(range(0, Nt, max(1, Nt // num)))
    if indices[-1] != Nt - 1:
        indices.append(Nt - 1)
    nplots = len(indices)

    fig, axes = plt.subplots(
        nplots, 2,
        figsize=(10, 3 * nplots),
        sharex=True
    )
    if nplots == 1:
        axes = axes[None, :]

    for row, index in enumerate(indices):
        c = Y[index, :]  # (K,)
        mag = jnp.abs(c)
        power = mag**2
        t_pct = ts[index]

        axes[row, 0].stem(modes[show], mag[show])
        axes[row, 0].set_ylabel(label)
        axes[row, 0].set_title(f"Magnitude at t={t_pct:.1f}")

        axes[row, 1].semilogy(modes[show], power[show])
        axes[row, 1].set_ylabel(label)
        axes[row, 1].set_title(f"Power at t={t_pct:.1f}")

    axes[-1, 0].set_xlabel("Mode")
    axes[-1, 1].set_xlabel("Mode")
    
    plt.title(f"{name} spectrum")
    plt.tight_layout()

    # If we're saving, save this figure and the growth figure as a separate file.
    if save_path is not None:
        root, ext = os.path.splitext(save_path)
        save_path_main = f"{root}_spectrum{ext if ext else '.png'}"
        save_path_growth = f"{root}_evolution{ext if ext else '.png'}"
    else:
        save_path_main = None
        save_path_growth = None

    _maybe_save_or_show(fig, save_path=save_path_main)

    # Growth of low modes over time (m=1..max_mode)
    amps = jnp.abs(Y[:, 1:max_mode_time + 1])  # (Nt, max_mode)

    fig2 = plt.figure(figsize=(7, 4))
    plt.semilogy(ts, amps)
    plt.xlabel("Time")
    plt.ylabel(label)
    plt.legend([f"k={i}" for i in range(1, max_mode_time + 1)])
    plt.title(f"Growth of low modes: {name}")
    plt.tight_layout()

    _maybe_save_or_show(fig2, save_path=save_path_growth)