import time
import numpy as np

import matplotlib.pyplot as plt

import equinox as eqx
import optax
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from dataloader import DataLoader
from utils import make_dir

def grad_diagnostics(grads):
    # Collect gradient leaves (JAX-safe)
    leaves = jax.tree_util.tree_leaves(grads)

    # Filter out None leaves
    leaves = [g for g in leaves if g is not None]

    # Number of gradient arrays
    num_leaves = len(leaves)

    # Flatten all gradients into one vector
    flat_grads = jnp.concatenate([jnp.ravel(g) for g in leaves])

    # Compute stats
    max_abs = jnp.max(jnp.abs(flat_grads))
    mean_abs = jnp.mean(jnp.abs(flat_grads))
    l2_norm = jnp.linalg.norm(flat_grads)

    # JAX-safe printing
    jax.debug.print(
        "Grad diagnostics | leaves: {n}, max|g|: {mx:.3e}, mean|g|: {mn:.3e}, ||g||₂: {l2:.3e}",
        n=num_leaves,
        mx=max_abs,
        mn=mean_abs,
        l2=l2_norm,
    )

class Optimizer():
    def __init__(self, pic,
                       model,
                       loss_metric,
                       loss_kwargs=None,
                       K=None,
                       tbptt_k=None,
                       tbptt_s=None,
                       batch_size=None,
                       y0=None,
                       lr=1e-4,
                       optim=None,
                       save_dir="model/", 
                       save_name="model_checkpoint",
                       seed=0):
        self.pic = pic
        # Backward compatibility:
        # old `K` represented trajectory-batch size. New API uses `batch_size`.
        if batch_size is None:
            batch_size = K if K is not None else 1
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.tbptt_k = int(tbptt_k) if tbptt_k is not None else int(self.pic.n_steps)
        self.tbptt_s = int(tbptt_s) if tbptt_s is not None else int(self.tbptt_k)
        if self.tbptt_k < 1:
            raise ValueError("tbptt_k must be >= 1.")
        if self.tbptt_s < 1:
            raise ValueError("tbptt_s must be >= 1.")
        if self.tbptt_s > self.tbptt_k:
            raise ValueError("tbptt_s must satisfy tbptt_s <= tbptt_k.")

        self.y0 = y0
        if self.y0 is not None:
            if self.y0[0].ndim != 2:
                raise ValueError(f"Expected unbatched y0 (N,1). Got {self.y0[0].shape}")
        self.model = model
        if loss_kwargs is None: loss_kwargs = {}
        self.loss_metric = loss_metric
        self.loss = lambda model, y0: self.loss_function(model, y0, **loss_kwargs) # Same signature as L2_loss, but different implementation
        self.grad_loss = eqx.filter_value_and_grad(self.loss) # Do NOT mutate loss after this point, it is jitted already
        self.base_lr = float(lr)
        # Fair-play scaling: keep trajectory budget fixed while compensating for fewer
        # optimizer updates under batched trajectories.
        self.lr = self.base_lr * float(self.batch_size)
        if optim is None:
            self.optim = optax.adam(self.lr)
        else:
            self.optim = optim(self.lr) # if you want to modify other parameters, prepass through a lambda (TODO: ugly, fix later)

        self.save_dir = save_dir
        self.save_name = save_name

    def _init_scan_state(self, y0, model):
        pos, vel = y0
        pos = jnp.mod(pos, self.pic.boxsize)

        moments, j, jp1, weight_j, weight_jp1 = self.pic.cic_deposition(pos, vel)
        E_grid, rho_k = self.pic.poisson_solver(moments[:, 0])

        if model is None:
            E_ext = None
        else:
            if model.closed_loop:
                E_ext = model(jnp.asarray(0), state=jnp.fft.rfft(moments[:, 1]))
            else:
                E_ext = model(jnp.asarray(0))

        E = self.pic.cic_gather((pos, vel, jnp.zeros_like(pos)), E_grid, j, jp1, weight_j, weight_jp1, E_ext=E_ext)
        acc = -self.pic.q * E / self.pic.m
        return (pos, vel, acc, E_grid, E_ext, moments)

    def _scan_steps(self, y_init, n_start, length, model):
        def step_fn(y, n):
            y_next = self.pic.step(y, n, E_control=model)
            return y_next, y_next
        xs = jnp.arange(n_start, n_start + length)
        y_end, outs = jax.lax.scan(step_fn, y_init, xs=xs, length=length)
        return y_end, outs

    def _pic_from_scan_outs(self, outs):
        pos_traj, vel_traj, acc_traj, E_traj, Eext_traj, moments_traj = outs
        if self.pic.higher_moments:
            return eqx.tree_at(
                lambda s: (s.positions, s.velocities, s.accelerations, s.E_field, s.E_ext, s.rho, s.momentum, s.energy),
                self.pic,
                (
                    pos_traj.squeeze(),
                    vel_traj.squeeze(),
                    acc_traj.squeeze(),
                    E_traj,
                    Eext_traj,
                    moments_traj[:, :, 0],
                    moments_traj[:, :, 1],
                    moments_traj[:, :, 2],
                ),
                is_leaf=lambda x: x is None,
            )
        return eqx.tree_at(
            lambda s: (s.positions, s.velocities, s.accelerations, s.E_field, s.E_ext, s.rho),
            self.pic,
            (
                pos_traj.squeeze(),
                vel_traj.squeeze(),
                acc_traj.squeeze(),
                E_traj,
                Eext_traj,
                moments_traj[:, :, 0],
            ),
            is_leaf=lambda x: x is None,
        )

    def _tbptt_loss_single(self, model, y0, **kwargs):
        total_steps = int(self.pic.n_steps)
        if self.tbptt_k >= total_steps and self.tbptt_s >= total_steps:
            pic = self.pic.run_simulation(y0, E_control=model)
            return self.loss_metric(pic, **kwargs)

        y_state = self._init_scan_state(y0, model)
        losses = []
        for n0 in range(0, total_steps, self.tbptt_s):
            length = min(self.tbptt_k, total_steps - n0)
            y_end, outs = self._scan_steps(y_state, n0, length, model)
            pic_window = self._pic_from_scan_outs(outs)
            losses.append(self.loss_metric(pic_window, **kwargs))

            advance = min(self.tbptt_s, length)
            if advance == length:
                y_state = y_end
            else:
                y_state = tuple(arr[advance - 1] for arr in outs)
            y_state = jax.tree_util.tree_map(jax.lax.stop_gradient, y_state)

        return jnp.mean(jnp.stack(losses))
    
    def loss_function(self, model, y0, **kwargs):
        pos, vel = y0
        if pos.ndim == 2:   # (N,1) single IC
            return self._tbptt_loss_single(model, (pos, vel), **kwargs)

        # batched: (B,N,1)
        losses = jax.vmap(lambda pos_i, vel_i: self._tbptt_loss_single(model, (pos_i, vel_i), **kwargs))(pos, vel)
        return losses.mean()
    
    def make_step(self, model, opt_state, y0):
        loss, grads = self.grad_loss(model, y0)
        grad_diagnostics(grads)
        #jax.debug.print("{grads}",grads=grads)
        updates, opt_state = self.optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def train(self, n_steps, save_every=100, seed=0, print_status=True):
        make_step = eqx.filter_jit(self.make_step) # Do NOT mutate anything inside self.make_step from this point on, it is jitted already    

        make_dir(self.save_dir)

        opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        train_losses = []
        valid_losses = []

        ic_key = jax.random.PRNGKey(seed)
        total_trajectories = int(n_steps)
        trajectories_done = 0
        step = 0
        while trajectories_done < total_trajectories:
            current_batch = min(self.batch_size, total_trajectories - trajectories_done)
            if print_status:
                print("--------------------")
                print(f"Step: {step}")            
            start = time.time()
            if current_batch > 1:
                ic_key, subkey = jax.random.split(ic_key)
                ic_key_arr = jax.random.split(subkey, current_batch)
                y0 = jax.vmap(self.pic.create_y0)(ic_key_arr)
            else:
                ic_key, subkey = jax.random.split(ic_key)
                y0 = self.pic.create_y0(subkey)
            loss, self.model, opt_state = make_step(self.model, opt_state, y0)
            end = time.time()
            train_losses.append(loss)
            if print_status: print(f"Train loss: {loss}")
            if step % save_every == 0 and step > 0 and trajectories_done < total_trajectories - 1:
                if print_status: print(f"Saving model at step {step}")
                checkpoint_name = self.save_dir+self.save_name+f"_{step}"
                self.model.save_model(checkpoint_name)
            trajectories_done += current_batch
            step += 1
        if print_status: print("Training complete.")
        checkpoint_name = self.save_dir+self.save_name+"_final"
        if print_status: print(f"Saving model at {checkpoint_name}")
        self.model.save_model(checkpoint_name)

        return self.model, train_losses, valid_losses