import time
import numpy as np

import matplotlib.pyplot as plt

import equinox as eqx
import optax
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

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
                       num_ics=10,
                       y0=None,
                       lr=1e-4,
                       lr_final=None,
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

        self.num_ics = int(num_ics)
        if self.num_ics < 1:
            raise ValueError("num_ics must be >= 1.")
        if self.batch_size > 1 and self.num_ics % self.batch_size != 0:
            raise ValueError("For batched training, num_ics must be a multiple of batch_size.")

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
        self.base_lr_final = float(lr_final) if lr_final is not None else None
        # Fair-play scaling: keep trajectory budget fixed while compensating for fewer
        # optimizer updates under batched trajectories.
        self.lr = self.base_lr * float(self.batch_size)
        self.lr_final = (
            self.base_lr_final * float(self.batch_size)
            if self.base_lr_final is not None
            else None
        )
        if self.lr_final is not None and self.lr_final <= 0.0:
            raise ValueError("lr_final must be positive when provided.")
        self.optim_builder = optim
        if self.optim_builder is None:
            self.optim = optax.adam(self.lr)
        else:
            self.optim = self.optim_builder(self.lr) # if you want to modify other parameters, prepass through a lambda (TODO: ugly, fix later)

        self.save_dir = save_dir
        self.save_name = save_name

    def _build_optimizer(self, total_updates=None):
        lr_or_schedule = self.lr
        if self.lr_final is not None and total_updates is not None and total_updates > 1:
            alpha = self.lr_final / self.lr
            lr_or_schedule = optax.cosine_decay_schedule(
                init_value=self.lr,
                decay_steps=max(int(total_updates) - 1, 1),
                alpha=float(alpha),
            )
        if self.optim_builder is None:
            return optax.adam(lr_or_schedule)
        return self.optim_builder(lr_or_schedule)

    def _init_scan_state(self, y0, model):
        pos, vel = y0
        pos = jnp.mod(pos, self.pic.boxsize)

        moments, j, jp1, weight_j, weight_jp1 = self.pic.cic_deposition(pos, vel)
        E_grid, rho_k = self.pic.poisson_solver(moments[:, 0])

        if model is None:
            E_ext = None
        else:
            if model.closed_loop:
                E_ext = model(
                    jnp.asarray(0),
                    state=(jnp.fft.rfft(moments[:, 0]), jnp.fft.rfft(moments[:, 1])),
                )
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

        if self.tbptt_k >= total_steps:
            # Fallback to full-trajectory loss when K exceeds horizon.
            pic = self.pic.run_simulation(y0, E_control=model)
            return self.loss_metric(pic, **kwargs)

        y_state = self._init_scan_state(y0, model)
        losses = []
        max_start = total_steps - self.tbptt_k
        for n0 in range(0, max_start + 1, self.tbptt_s):
            length = self.tbptt_k
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

    def train(self, n_steps, save_every=100, seed=0, ic_seed=0, print_status=True):
        total_trajectories = int(n_steps)
        total_updates = int(np.ceil(total_trajectories / self.batch_size))
        self.optim = self._build_optimizer(total_updates=total_updates)
        make_step = eqx.filter_jit(self.make_step) # Do NOT mutate anything inside self.make_step from this point on, it is jitted already

        make_dir(self.save_dir)

        opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        train_losses = []
        valid_losses = []

        # `ic_seed` controls only the training initial-condition pool.
        # `seed` is reserved for optimizer-side stochasticity.
        _ = seed
        ic_key = jax.random.PRNGKey(ic_seed)
        ic_keys = jax.random.split(ic_key, self.num_ics)
        # Make IC generation seed-consistent:
        # create_y0(PRNGKey(seed)) must match pool index 0 for any num_ics.
        ic_keys = ic_keys.at[0].set(ic_key)
        y0_pool = jax.vmap(self.pic.create_y0)(ic_keys)

        trajectories_done = 0
        step = 0
        while trajectories_done < total_trajectories:
            current_batch = min(self.batch_size, total_trajectories - trajectories_done)
            if print_status:
                print("--------------------")
                print(f"Step: {step}")            
            start = time.time()

            # Cycle deterministically through a fixed pool of ICs:
            # ic#1 -> ic#num_ics -> repeat.
            start_idx = trajectories_done % self.num_ics
            ic_idx = (start_idx + jnp.arange(current_batch)) % self.num_ics
            if current_batch > 1:
                y0 = tuple(arr[ic_idx] for arr in y0_pool)
            else:
                y0 = tuple(arr[ic_idx[0]] for arr in y0_pool)
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


class PPOTrainer:
    def __init__(
        self,
        pic,
        model,
        *,
        lr=3e-4,
        lr_final=None,
        save_dir="model_cl_ppo/",
        save_name="model_checkpoint",
        num_ics=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        ppo_epochs=4,
    ):
        self.pic = pic
        self.model = model
        self.base_lr = float(lr)
        self.base_lr_final = float(lr_final) if lr_final is not None else None
        self.save_dir = save_dir
        self.save_name = save_name
        self.num_ics = int(num_ics)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.ppo_epochs = int(ppo_epochs)
        if self.num_ics < 1:
            raise ValueError("num_ics must be >= 1.")
        if self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be >= 1.")

    def _build_optimizer(self, total_updates):
        if self.base_lr_final is None or total_updates <= 1:
            return optax.adam(self.base_lr)
        alpha = self.base_lr_final / self.base_lr
        schedule = optax.cosine_decay_schedule(
            init_value=self.base_lr,
            decay_steps=max(int(total_updates) - 1, 1),
            alpha=float(alpha),
        )
        return optax.adam(schedule)

    def _reward_from_moments(self, moments):
        rho = moments[:, 0]
        rho_k = jnp.fft.rfft(rho.astype(jnp.float32))
        dc = jnp.abs(rho_k[0]) ** 2 + 1e-12
        mode_energy = jnp.sum(jnp.abs(rho_k[jnp.array([1, 2])]) ** 2)
        return -(mode_energy / dc)

    def _init_state(self, y0, key):
        pos, vel = y0
        pos = jnp.mod(pos, self.pic.boxsize)
        moments, j, jp1, weight_j, weight_jp1 = self.pic.cic_deposition(pos, vel)
        rho_hat = jnp.fft.rfft(moments[:, 0])
        mom_hat = jnp.fft.rfft(moments[:, 1])
        e_ext0, _, _, _ = self.model.act((rho_hat, mom_hat), key)
        E_grid, _ = self.pic.poisson_solver(moments[:, 0])
        E = self.pic.cic_gather((pos, vel, jnp.zeros_like(pos)), E_grid, j, jp1, weight_j, weight_jp1, E_ext=e_ext0)
        acc = -self.pic.q * E / self.pic.m
        return pos, vel, acc

    def _rollout_episode(self, y0, key):
        T = int(self.pic.n_steps)
        pos, vel, acc = self._init_state(y0, key)
        keys = jax.random.split(key, T + 1)

        rho_hist = []
        mom_hist = []
        act_hist = []
        logp_hist = []
        value_hist = []
        reward_hist = []

        for n in range(T):
            vel = vel + acc * self.pic.dt / 2.0
            pos = jnp.mod(pos + vel * self.pic.dt, self.pic.boxsize)

            moments, j, jp1, weight_j, weight_jp1 = self.pic.cic_deposition(pos, vel)
            rho_hat = jnp.fft.rfft(moments[:, 0])
            mom_hat = jnp.fft.rfft(moments[:, 1])

            e_ext, action, logp, value = self.model.act((rho_hat, mom_hat), keys[n])
            E_grid, _ = self.pic.poisson_solver(moments[:, 0])
            E = self.pic.cic_gather((pos, vel, acc), E_grid, j, jp1, weight_j, weight_jp1, E_ext=e_ext)
            acc = -self.pic.q * E / self.pic.m
            vel = vel + acc * self.pic.dt / 2.0

            reward = self._reward_from_moments(moments)
            rho_hist.append(rho_hat)
            mom_hist.append(mom_hat)
            act_hist.append(action)
            logp_hist.append(logp)
            value_hist.append(value)
            reward_hist.append(reward)

        rho_hist = jnp.stack(rho_hist, axis=0)
        mom_hist = jnp.stack(mom_hist, axis=0)
        act_hist = jnp.stack(act_hist, axis=0)
        logp_hist = jnp.stack(logp_hist, axis=0)
        value_hist = jnp.stack(value_hist, axis=0)
        reward_hist = jnp.stack(reward_hist, axis=0)
        return (rho_hist, mom_hist), act_hist, logp_hist, value_hist, reward_hist

    def _compute_advantages(self, rewards, values):
        T = rewards.shape[0]
        advantages = []
        gae = 0.0
        next_value = 0.0
        for t in range(T - 1, -1, -1):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.append(gae)
            next_value = values[t]
        advantages = jnp.array(advantages[::-1], dtype=jnp.float64)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _loss(self, model, obs, actions, old_logp, advantages, returns):
        rho_hist, mom_hist = obs
        logp, entropy, values = jax.vmap(model.evaluate_action, in_axes=((0, 0), 0))(
            (rho_hist, mom_hist), actions
        )
        ratio = jnp.exp(logp - old_logp)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        value_loss = jnp.mean((returns - values) ** 2)
        entropy_bonus = jnp.mean(entropy)
        total = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus
        return total, (policy_loss, value_loss, entropy_bonus)

    def train(self, n_steps, save_every=100, seed=0, ic_seed=0, print_status=True):
        total_updates = int(n_steps)
        make_dir(self.save_dir)
        self.optim = self._build_optimizer(total_updates)
        opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))
        grad_loss = eqx.filter_value_and_grad(self._loss, has_aux=True)

        ic_key = jax.random.PRNGKey(ic_seed)
        ic_keys = jax.random.split(ic_key, self.num_ics)
        ic_keys = ic_keys.at[0].set(ic_key)
        y0_pool = jax.vmap(self.pic.create_y0)(ic_keys)

        train_losses = []
        aux_logs = []
        rng = jax.random.PRNGKey(seed)

        for step in range(total_updates):
            rng, rollout_key = jax.random.split(rng)
            idx = step % self.num_ics
            y0 = tuple(arr[idx] for arr in y0_pool)

            obs, actions, old_logp, values, rewards = self._rollout_episode(y0, rollout_key)
            advantages, returns = self._compute_advantages(rewards, values)

            loss_value = None
            aux_value = None
            for _ in range(self.ppo_epochs):
                (loss_value, aux_value), grads = grad_loss(self.model, obs, actions, old_logp, advantages, returns)
                updates, opt_state = self.optim.update(grads, opt_state)
                self.model = eqx.apply_updates(self.model, updates)

            train_losses.append(float(loss_value))
            aux_logs.append(tuple(float(x) for x in aux_value))

            if print_status:
                pol, val, ent = aux_value
                print("--------------------")
                print(f"Step: {step}")
                print(f"Loss: {loss_value}")
                print(f"Policy: {pol}, Value: {val}, Entropy: {ent}")

            if step % save_every == 0 and step > 0 and step < total_updates - 1:
                checkpoint_name = self.save_dir + self.save_name + f"_{step}"
                self.model.save_model(checkpoint_name)

        checkpoint_name = self.save_dir + self.save_name + "_final"
        self.model.save_model(checkpoint_name)
        return self.model, train_losses, aux_logs