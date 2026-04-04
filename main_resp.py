import jax
import jax.numpy as jnp
import equinox as eqx
from pic_simulation import PICSimulation
from control import FourierActuator
from plotting import scatter_animation, plot_pde_solution, plot_modes

# Simulation parameters
N_particles = 40000  # Number of particles
N_mesh = 400  # Number of mesh cells
t1 = 20  # time at which simulation ends
dt = 0.1  # timestep
boxsize = 50  # periodic domain [0,boxsize]
n0 = 1  # electron number density
vb = 3  # beam velocity
vth = 1  # beam width
pos_sample = False


key = jax.random.key(0)
key1, key2, key3 = jax.random.split(key, num=3)

if pos_sample:
    pos = jax.random.uniform(key1, (N_particles, 1)) * boxsize
else:
    pos = jnp.zeros(N_particles)
    for i in range(N_particles):
        pos = pos.at[i].set(i*boxsize/N_particles)
    pos = jax.random.choice(key1, pos, shape=pos.shape, replace=False)
    pos = pos[:,None]

vel = vth * jax.random.normal(key2, (N_particles, 1)) + vb
Nh = int(N_particles / 2)
vel = vel.at[Nh:].set(-1*vel[Nh:])

y0 = (pos, vel)

pic = PICSimulation(boxsize, N_particles, N_mesh, n0, vb, vth, dt, t1, t0=0, higher_moments=True)

# Build a deterministic open-loop response field with a single (time,space) mode.
n_mode = 3
m_mode = 5
amp = 1e5
phase = 0.0
E_control = FourierActuator(
    Nt=pic.n_steps,
    N_mesh=pic.N_mesh,
    boxsize=pic.boxsize,
    n_modes_time=n_mode + 1,
    n_modes_space=m_mode + 1,
    key=None,
    init_scale=0.0,
    zero=False,
    closed_loop=False,
)

coeff = (amp / 2.0) * jnp.exp(1j * phase)
a_hat = E_control.a_hat_train.at[m_mode, n_mode].set(jnp.asarray(coeff, dtype=jnp.complex64))
E_control = eqx.tree_at(lambda m: m.a_hat_train, E_control, a_hat)

pic = pic.run_simulation(y0,E_control=E_control)

u = jax.vmap(E_control)(jnp.arange(pic.ts.shape[0]))

print("E_control shape: ", u.shape)

plot_pde_solution(pic.ts, u, boxsize, name=r"External field", label=r"$E_{ext}$", save_path="plots/resp/external_field.png")

scatter_animation(pic.ts, pic.positions, pic.velocities, Nh, boxsize=boxsize, k=1, fps=10, save_path="plots/resp/scatter.mp4")

plot_pde_solution(pic.ts, pic.rho, boxsize, name=r"Density", label=r"$\rho$", save_path="plots/resp/density.png")
plot_pde_solution(pic.ts, pic.momentum, boxsize, name=r"Momentum", label=r"$P$", save_path="plots/resp/momentum.png")
plot_pde_solution(pic.ts, pic.energy, boxsize, name=r"Energy", label=r"$E$", save_path="plots/resp/energy.png")

plot_modes(pic.ts, pic.rho, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, num=4, zero_mean=True, save_path="plots/resp/density_modes.png")
plot_modes(pic.ts, pic.momentum, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, num=4, zero_mean=True, save_path="plots/resp/momentum_modes.png")
plot_modes(pic.ts, pic.energy, max_mode_spect=10, max_mode_time=5, boxsize=boxsize, num=4, zero_mean=True, save_path="plots/resp/energy_modes.png")