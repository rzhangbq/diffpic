import argparse
import json
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from control import DissipativeModeFeedbackActuator, FourierActuator, ModeFeedbackActuator
from losses import (
    loss_metric,
    loss_metric_cancel_self_field,
    loss_metric_density_modes,
    loss_metric_stable,
)
from optimize import Optimizer
from pic_simulation import PICSimulation
from plotting import plot_modes, plot_pde_solution, scatter_animation


def make_pic(cfg: dict, t1: float) -> PICSimulation:
    return PICSimulation(
        cfg["boxsize"],
        cfg["N_particles"],
        cfg["N_mesh"],
        cfg["n0"],
        cfg["vb"],
        cfg["vth"],
        cfg["dt"],
        t1,
        t0=0,
        higher_moments=True,
    )


def apply_cfg_overrides(cfg: dict, args) -> dict:
    out = dict(cfg)
    if args.n_particles is not None:
        out["N_particles"] = args.n_particles
    if args.n_mesh is not None:
        out["N_mesh"] = args.n_mesh
    if args.t1 is not None:
        out["t1"] = args.t1
    if args.dt is not None:
        out["dt"] = args.dt
    if args.boxsize is not None:
        out["boxsize"] = args.boxsize
    if args.n0 is not None:
        out["n0"] = args.n0
    if args.vb is not None:
        out["vb"] = args.vb
    if args.vth is not None:
        out["vth"] = args.vth
    return out


def pick(value, default):
    return default if value is None else value


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def make_run_id(mode: str, args) -> str:
    if args.run_name:
        return args.run_name
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{mode}"


def prepare_run_dir(base_dir: str, run_id: str) -> Path:
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_config(run_dir: Path, config: dict) -> None:
    cfg_path = run_dir / "run_config.json"
    cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def _latest_subrun_checkpoint(base_dir: str, checkpoint_name: str = "model_checkpoint_final") -> Path | None:
    base = Path(base_dir)
    if not base.exists():
        return None
    candidates = [d for d in base.iterdir() if d.is_dir() and (d / checkpoint_name).exists()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest / checkpoint_name


def resolve_checkpoint(base_dir: str, args, checkpoint_name: str = "model_checkpoint_final") -> str:
    if args.model_run is not None:
        requested = Path(base_dir) / args.model_run / checkpoint_name
        if not requested.exists():
            raise FileNotFoundError(f"Checkpoint not found for --model-run: {requested}")
        return str(requested)

    latest = _latest_subrun_checkpoint(base_dir, checkpoint_name=checkpoint_name)
    if latest is not None:
        return str(latest)

    legacy = Path(base_dir) / checkpoint_name
    if legacy.exists():
        return str(legacy)
    raise FileNotFoundError(
        f"No checkpoint found in {base_dir}. Expected either {base_dir}/{checkpoint_name} "
        f"or {base_dir}/<run_id>/{checkpoint_name}."
    )


def save_training_curve(train_losses, save_path: str, *, use_log: bool) -> None:
    ensure_parent(save_path)
    plt.figure()
    if use_log:
        plt.semilogy(train_losses)
    else:
        plt.plot(train_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_time_series(y, ts, save_path: str, *, ylabel: str, title: str) -> None:
    ensure_parent(save_path)
    plt.figure()
    plt.plot(ts, y)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_common_plots(
    pic: PICSimulation,
    u,
    *,
    boxsize,
    nh: int,
    out_dir: str,
    base_modes: dict,
    with_external_modes: bool = False,
    external_modes: dict | None = None,
) -> None:
    scatter_animation(
        pic.ts,
        pic.positions,
        pic.velocities,
        nh,
        boxsize=boxsize,
        k=1,
        fps=10,
        save_path=f"{out_dir}/scatter.mp4",
    )

    plot_pde_solution(
        pic.ts,
        u,
        boxsize,
        name=r"External field",
        label=r"$E_{ext}$",
        save_path=f"{out_dir}/external_field.png",
    )
    plot_pde_solution(
        pic.ts,
        pic.E_field,
        boxsize,
        name=r"Self-generated field",
        label=r"$E_{self}$",
        save_path=f"{out_dir}/self_field.png",
    )
    plot_pde_solution(
        pic.ts,
        pic.rho,
        boxsize,
        name=r"Density",
        label=r"$\rho$",
        save_path=f"{out_dir}/density.png",
    )
    plot_pde_solution(
        pic.ts,
        pic.momentum,
        boxsize,
        name=r"Momentum",
        label=r"$P$",
        save_path=f"{out_dir}/momentum.png",
    )
    plot_pde_solution(
        pic.ts,
        pic.energy,
        boxsize,
        name=r"Energy",
        label=r"$E$",
        save_path=f"{out_dir}/energy.png",
    )

    if with_external_modes and external_modes is not None:
        plot_modes(
            pic.ts,
            u,
            max_mode_spect=external_modes["max_mode_spect"],
            max_mode_time=external_modes["max_mode_time"],
            boxsize=boxsize,
            name=r"External field",
            label=r"$\hat E_{ext}$",
            num=external_modes["num"],
            zero_mean=True,
            save_path=f"{out_dir}/external_field_modes.png",
        )

    self_field_modes_cfg = base_modes.get("e_field", base_modes.get("rho"))
    if self_field_modes_cfg is not None:
        plot_modes(
            pic.ts,
            pic.E_field,
            max_mode_spect=self_field_modes_cfg["max_mode_spect"],
            max_mode_time=self_field_modes_cfg["max_mode_time"],
            boxsize=boxsize,
            name=r"Self-generated field",
            label=r"$\hat E_{self}$",
            num=self_field_modes_cfg["num"],
            zero_mean=True,
            save_path=f"{out_dir}/self_field_modes.png",
        )

    for field, label, name in [
        ("rho", r"$\hat\rho_k$", r"Density"),
        ("momentum", r"$\hat\mathcal{{p}}_k$", r"Momentum"),
        ("energy", r"$\hat\mathcal{{E}}_k$", r"Energy"),
    ]:
        cfg = base_modes[field]
        state = getattr(pic, field)
        plot_modes(
            pic.ts,
            state,
            max_mode_spect=cfg["max_mode_spect"],
            max_mode_time=cfg["max_mode_time"],
            boxsize=boxsize,
            name=name,
            label=label,
            num=cfg["num"],
            zero_mean=True,
            save_path=f"{out_dir}/{field}_modes.png",
        )


def save_state_only_plots(pic: PICSimulation, *, boxsize, nh: int, out_dir: str, base_modes: dict) -> None:
    scatter_animation(
        pic.ts,
        pic.positions,
        pic.velocities,
        nh,
        boxsize=boxsize,
        k=1,
        fps=10,
        save_path=f"{out_dir}/scatter.mp4",
    )

    plot_pde_solution(
        pic.ts,
        pic.E_field,
        boxsize,
        name=r"Self-generated field",
        label=r"$E_{self}$",
        save_path=f"{out_dir}/self_field.png",
    )
    plot_pde_solution(
        pic.ts,
        pic.rho,
        boxsize,
        name=r"Density",
        label=r"$\rho$",
        save_path=f"{out_dir}/density.png",
    )
    plot_pde_solution(
        pic.ts,
        pic.momentum,
        boxsize,
        name=r"Momentum",
        label=r"$P$",
        save_path=f"{out_dir}/momentum.png",
    )
    plot_pde_solution(
        pic.ts,
        pic.energy,
        boxsize,
        name=r"Energy",
        label=r"$E$",
        save_path=f"{out_dir}/energy.png",
    )

    for field, label, name in [
        ("rho", r"$\hat\rho_k$", r"Density"),
        ("momentum", r"$\hat\mathcal{{p}}_k$", r"Momentum"),
        ("energy", r"$\hat\mathcal{{E}}_k$", r"Energy"),
    ]:
        cfg = base_modes[field]
        state = getattr(pic, field)
        plot_modes(
            pic.ts,
            state,
            max_mode_spect=cfg["max_mode_spect"],
            max_mode_time=cfg["max_mode_time"],
            boxsize=boxsize,
            name=name,
            label=label,
            num=cfg["num"],
            zero_mean=True,
            save_path=f"{out_dir}/{field}_modes.png",
        )

    self_field_modes_cfg = base_modes.get("e_field", base_modes.get("rho"))
    if self_field_modes_cfg is not None:
        plot_modes(
            pic.ts,
            pic.E_field,
            max_mode_spect=self_field_modes_cfg["max_mode_spect"],
            max_mode_time=self_field_modes_cfg["max_mode_time"],
            boxsize=boxsize,
            name=r"Self-generated field",
            label=r"$\hat E_{self}$",
            num=self_field_modes_cfg["num"],
            zero_mean=True,
            save_path=f"{out_dir}/self_field_modes.png",
        )


def create_response_initial_conditions(pic: PICSimulation, *, seed: int, pos_sample: bool):
    key = jax.random.key(seed)
    key_pos, key_vel, _ = jax.random.split(key, num=3)

    if pos_sample:
        pos = jax.random.uniform(key_pos, (pic.N_particles, 1)) * pic.boxsize
    else:
        pos = jnp.arange(pic.N_particles) * pic.boxsize / pic.N_particles
        pos = jax.random.choice(key_pos, pos, shape=pos.shape, replace=False)
        pos = pos[:, None]

    vel = pic.vth * jax.random.normal(key_vel, (pic.N_particles, 1)) + pic.vb
    nh = pic.N_particles // 2
    vel = vel.at[nh:].set(-vel[nh:])
    return pos, vel


def run_resp(args) -> None:
    run_id = make_run_id("resp", args)
    plot_dir = prepare_run_dir("plots/resp", run_id)
    cfg = {
        "N_particles": 40000,
        "N_mesh": 400,
        "t1": 20.0,
        "dt": 0.1,
        "boxsize": 50.0,
        "n0": 1.0,
        "vb": 3.0,
        "vth": 1.0,
    }
    cfg = apply_cfg_overrides(cfg, args)
    nh = cfg["N_particles"] // 2
    pic = make_pic(cfg, cfg["t1"])

    y0 = create_response_initial_conditions(pic, seed=pick(args.seed_ic, 0), pos_sample=False)

    n_mode = 3
    m_mode = 5
    # Keep response forcing on a scale comparable to trained open-loop control.
    amp = pick(args.resp_amp, 1.0)
    phase = 0.0
    e_control = FourierActuator(
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
    a_hat = e_control.a_hat_train.at[m_mode, n_mode].set(jnp.asarray(coeff, dtype=jnp.complex64))
    e_control = eqx.tree_at(lambda m: m.a_hat_train, e_control, a_hat)

    pic = pic.run_simulation(y0, E_control=e_control)
    u = jax.vmap(e_control)(jnp.arange(pic.ts.shape[0]))
    print("E_control shape:", u.shape)

    default_modes = {
        "rho": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
        "momentum": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
        "energy": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
    }
    save_common_plots(pic, u, boxsize=cfg["boxsize"], nh=nh, out_dir=str(plot_dir), base_modes=default_modes)
    write_run_config(
        plot_dir,
        {
            "mode": "resp",
            "run_id": run_id,
            "args": vars(args),
            "cfg": cfg,
            "mode_params": {"n_mode": n_mode, "m_mode": m_mode, "amp": amp, "phase": phase},
            "outputs": {"plot_dir": str(plot_dir)},
        },
    )


def run_opt(args) -> None:
    run_id = make_run_id("opt", args)
    plot_dir = prepare_run_dir("plots/trained", run_id)
    model_dir = prepare_run_dir("model", run_id)
    cfg = {
        "N_particles": 40000,
        "N_mesh": 256,
        "t1": 20.0,
        "dt": 0.1,
        "boxsize": 10 * jnp.pi,
        "n0": 1.0,
        "vb": 2.4,
        "vth": 0.5,
    }
    cfg = apply_cfg_overrides(cfg, args)
    nh = cfg["N_particles"] // 2
    seed_ic = pick(args.seed_ic, 10)
    n_modes_time = pick(args.open_n_modes_time, 2)
    n_modes_space = pick(args.open_n_modes_space, 10)
    open_init_scale = pick(args.open_init_scale, 1e-4)
    train_steps = pick(args.train_steps, 200)
    save_every = pick(args.save_every, 100)
    train_seed = pick(args.train_seed, seed_ic)
    num_ics = args.num_ics
    eval_mult = pick(args.eval_mult, 2.0)
    tbptt_k = args.tbptt_k
    tbptt_s = args.tbptt_s
    tbptt_b = pick(args.tbptt_b, 1)
    lr_start = pick(args.lr_start, 5e-2)
    lr_end = pick(args.lr_end, 1e-3)
    pic = make_pic(cfg, cfg["t1"])

    e_control = FourierActuator(
        Nt=pic.n_steps,
        N_mesh=pic.N_mesh,
        boxsize=pic.boxsize,
        n_modes_time=n_modes_time,
        n_modes_space=n_modes_space,
        key=jax.random.PRNGKey(0),
        init_scale=open_init_scale,
        zero=False,
        closed_loop=False,
    )

    optimizer = Optimizer(
        pic=pic,
        model=e_control,
        K=None,
        loss_metric=loss_metric,
        lr=lr_start / tbptt_b,
        lr_final=lr_end / tbptt_b,
        save_dir=f"{model_dir}/",
        tbptt_k=tbptt_k,
        tbptt_s=tbptt_s,
        batch_size=tbptt_b,
        num_ics=num_ics,
    )
    e_control, train_losses, _ = optimizer.train(
        n_steps=train_steps, save_every=save_every, seed=train_seed, ic_seed=seed_ic, print_status=True
    )

    pic_eval = make_pic(cfg, eval_mult * cfg["t1"])
    checkpoint_path = str(model_dir / "model_checkpoint_final")
    e_control = FourierActuator.load_model(checkpoint_path)
    print("Control modes:\n", e_control.get_modes_summary())

    y0_eval = pic_eval.create_y0(jax.random.key(pick(args.seed_ic_eval, seed_ic)))
    pic_eval = pic_eval.run_simulation(y0_eval, E_control=e_control)
    u = jax.vmap(e_control)(jnp.arange(pic_eval.ts.shape[0]))

    default_modes = {
        "rho": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
        "momentum": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
        "energy": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
    }
    save_common_plots(
        pic_eval,
        u,
        boxsize=cfg["boxsize"],
        nh=nh,
        out_dir=str(plot_dir),
        base_modes=default_modes,
    )
    save_time_series(
        jnp.sum(pic_eval.energy + pic_eval.E_field**2, axis=-1),
        pic_eval.ts,
        str(plot_dir / "energy_evolution.png"),
        ylabel="Energy",
        title="Energy Evolution",
    )
    save_training_curve(train_losses, str(plot_dir / "train_losses.png"), use_log=False)
    run_cfg = {
        "mode": "opt",
        "run_id": run_id,
        "args": vars(args),
        "cfg": cfg,
        "train": {
            "train_steps": train_steps,
            "save_every": save_every,
            "train_seed": train_seed,
            "num_ics": num_ics,
            "lr_start": lr_start,
            "lr_end": lr_end,
        },
        "model": {
            "n_modes_time": n_modes_time,
            "n_modes_space": n_modes_space,
            "init_scale": open_init_scale,
        },
        "eval": {"eval_mult": eval_mult, "seed_ic_eval": pick(args.seed_ic_eval, seed_ic)},
        "tbptt": {"K": tbptt_k, "S": tbptt_s, "B": tbptt_b},
        "outputs": {"plot_dir": str(plot_dir), "model_dir": str(model_dir), "checkpoint": checkpoint_path},
    }
    write_run_config(plot_dir, run_cfg)
    write_run_config(model_dir, run_cfg)


def run_opt_cl(args) -> None:
    run_id = make_run_id("opt_cl", args)
    plot_dir = prepare_run_dir("plots/trained_cl", run_id)
    model_dir = prepare_run_dir("model_cl", run_id)
    cfg = {
        "N_particles": 100000,
        "N_mesh": 400,
        "t1": 30.0,
        "dt": 0.1,
        "boxsize": 10 * jnp.pi,
        "n0": 1.0,
        "vb": 2.4,
        "vth": 0.5,
    }
    cfg = apply_cfg_overrides(cfg, args)
    nh = cfg["N_particles"] // 2
    seed_ic = pick(args.seed_ic, 10)
    train_steps = pick(args.train_steps, 200)
    save_every = pick(args.save_every, 100)
    train_seed = pick(args.train_seed, seed_ic)
    num_ics = args.num_ics
    eval_mult = pick(args.eval_mult, 2.0)
    tbptt_k = args.tbptt_k
    tbptt_s = args.tbptt_s
    tbptt_b = pick(args.tbptt_b, 4)
    lr_start = pick(args.lr_start, 5e-2)
    lr_end = pick(args.lr_end, 1e-3)
    pic = make_pic(cfg, cfg["t1"])

    e_control = ModeFeedbackActuator(
        N_mesh=pic.N_mesh,
        boxsize=pic.boxsize,
        n_modes_space_in=10,
        n_modes_space_out=10,
        use_linear=False, 
        width=32,
        depth=2,
        init_scale=0.0,
        u_max=None,
        include_dc=False,
        include_density_input=True,
        closed_loop=True,
        key=jax.random.PRNGKey(9),
    )
    optimizer = Optimizer(
        pic=pic,
        model=e_control,
        loss_metric=loss_metric_density_modes,
        lr=lr_start / tbptt_b,
        lr_final=lr_end / tbptt_b,
        save_dir=f"{model_dir}/",
        tbptt_k=tbptt_k,
        tbptt_s=tbptt_s,
        batch_size=tbptt_b,
        num_ics=num_ics,
    )
    e_control, train_losses, _ = optimizer.train(
        n_steps=train_steps, save_every=save_every, seed=train_seed, ic_seed=seed_ic, print_status=True
    )

    pic_eval = make_pic(cfg, eval_mult * cfg["t1"])
    checkpoint_path = str(model_dir / "model_checkpoint_final")
    e_control = ModeFeedbackActuator.load_model(checkpoint_path)
    y0_eval = pic_eval.create_y0(jax.random.key(pick(args.seed_ic_eval, 1024)))
    pic_eval = pic_eval.run_simulation(y0_eval, E_control=e_control)

    e_control_vmappable = lambda n, rho_hat, mom_hat: e_control(n, state=(rho_hat, mom_hat))
    u = jax.vmap(e_control_vmappable, in_axes=(0, 0, 0))(
        jnp.arange(pic_eval.ts.shape[0]),
        jnp.fft.rfft(pic_eval.rho),
        jnp.fft.rfft(pic_eval.momentum),
    )

    state_modes = {
        "rho": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
        "momentum": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
        "energy": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
    }
    ext_modes = {"max_mode_spect": 10, "max_mode_time": 10, "num": 6}
    save_common_plots(
        pic_eval,
        u,
        boxsize=cfg["boxsize"],
        nh=nh,
        out_dir=str(plot_dir),
        base_modes=state_modes,
        with_external_modes=True,
        external_modes=ext_modes,
    )
    save_time_series(
        jnp.sum(pic_eval.energy + pic_eval.E_field**2, axis=-1),
        pic_eval.ts,
        str(plot_dir / "energy_evolution.png"),
        ylabel="Energy",
        title="Energy Evolution",
    )
    save_training_curve(train_losses, str(plot_dir / "train_losses.png"), use_log=True)
    run_cfg = {
        "mode": "opt_cl",
        "run_id": run_id,
        "args": vars(args),
        "cfg": cfg,
        "train": {
            "train_steps": train_steps,
            "save_every": save_every,
            "train_seed": train_seed,
            "num_ics": num_ics,
            "lr_start": lr_start,
            "lr_end": lr_end,
        },
        "eval": {"eval_mult": eval_mult, "seed_ic_eval": pick(args.seed_ic_eval, 1024)},
        "tbptt": {"K": tbptt_k, "S": tbptt_s, "B": tbptt_b},
        "outputs": {"plot_dir": str(plot_dir), "model_dir": str(model_dir), "checkpoint": checkpoint_path},
    }
    write_run_config(plot_dir, run_cfg)
    write_run_config(model_dir, run_cfg)


def run_opt_cl_self(args) -> None:
    run_id = make_run_id("opt_cl_self", args)
    plot_dir = prepare_run_dir("plots/trained_cl_self", run_id)
    model_dir = prepare_run_dir("model_cl_self", run_id)
    cfg = {
        "N_particles": 100000,
        "N_mesh": 400,
        "t1": 30.0,
        "dt": 0.1,
        "boxsize": 10 * jnp.pi,
        "n0": 1.0,
        "vb": 2.4,
        "vth": 0.5,
    }
    cfg = apply_cfg_overrides(cfg, args)
    nh = cfg["N_particles"] // 2
    seed_ic = pick(args.seed_ic, 10)
    train_steps = pick(args.train_steps, 200)
    save_every = pick(args.save_every, 100)
    train_seed = pick(args.train_seed, seed_ic)
    num_ics = args.num_ics
    eval_mult = pick(args.eval_mult, 2.0)
    tbptt_k = args.tbptt_k
    tbptt_s = args.tbptt_s
    tbptt_b = pick(args.tbptt_b, 4)
    lr_start = pick(args.lr_start, 5e-2)
    lr_end = pick(args.lr_end, 1e-3)
    pic = make_pic(cfg, cfg["t1"])

    e_control = ModeFeedbackActuator(
        N_mesh=pic.N_mesh,
        boxsize=pic.boxsize,
        n_modes_space_in=10,
        n_modes_space_out=10,
        use_linear=False,
        width=32,
        depth=2,
        init_scale=0.0,
        u_max=None,
        include_dc=False,
        include_density_input=True,
        closed_loop=True,
        key=jax.random.PRNGKey(9),
    )
    optimizer = Optimizer(
        pic=pic,
        model=e_control,
        loss_metric=loss_metric_cancel_self_field,
        lr=lr_start / tbptt_b,
        lr_final=lr_end / tbptt_b,
        save_dir=f"{model_dir}/",
        tbptt_k=tbptt_k,
        tbptt_s=tbptt_s,
        batch_size=tbptt_b,
        num_ics=num_ics,
    )
    e_control, train_losses, _ = optimizer.train(
        n_steps=train_steps, save_every=save_every, seed=train_seed, ic_seed=seed_ic, print_status=True
    )

    pic_eval = make_pic(cfg, eval_mult * cfg["t1"])
    checkpoint_path = str(model_dir / "model_checkpoint_final")
    e_control = ModeFeedbackActuator.load_model(checkpoint_path)
    y0_eval = pic_eval.create_y0(jax.random.key(pick(args.seed_ic_eval, 1024)))
    pic_eval = pic_eval.run_simulation(y0_eval, E_control=e_control)

    e_control_vmappable = lambda n, rho_hat, mom_hat: e_control(n, state=(rho_hat, mom_hat))
    u = jax.vmap(e_control_vmappable, in_axes=(0, 0, 0))(
        jnp.arange(pic_eval.ts.shape[0]),
        jnp.fft.rfft(pic_eval.rho),
        jnp.fft.rfft(pic_eval.momentum),
    )
    cancel_residual = u + pic_eval.E_field
    cancel_mse_t = jnp.mean(cancel_residual**2, axis=-1)
    save_time_series(
        cancel_mse_t,
        pic_eval.ts,
        str(plot_dir / "cancel_residual_mse.png"),
        ylabel=r"$\mathrm{MSE}(E_{ext}+E_{self})$",
        title="Controller cancellation residual",
    )

    state_modes = {
        "rho": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
        "momentum": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
        "energy": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
    }
    ext_modes = {"max_mode_spect": 10, "max_mode_time": 10, "num": 6}
    save_common_plots(
        pic_eval,
        u,
        boxsize=cfg["boxsize"],
        nh=nh,
        out_dir=str(plot_dir),
        base_modes=state_modes,
        with_external_modes=True,
        external_modes=ext_modes,
    )
    save_time_series(
        jnp.sum(pic_eval.energy + pic_eval.E_field**2, axis=-1),
        pic_eval.ts,
        str(plot_dir / "energy_evolution.png"),
        ylabel="Energy",
        title="Energy Evolution",
    )
    save_training_curve(train_losses, str(plot_dir / "train_losses.png"), use_log=True)
    run_cfg = {
        "mode": "opt_cl_self",
        "run_id": run_id,
        "args": vars(args),
        "cfg": cfg,
        "train": {
            "train_steps": train_steps,
            "save_every": save_every,
            "train_seed": train_seed,
            "num_ics": num_ics,
            "lr_start": lr_start,
            "lr_end": lr_end,
            "loss": "mean((E_ext + E_self)^2)",
        },
        "eval": {"eval_mult": eval_mult, "seed_ic_eval": pick(args.seed_ic_eval, 1024)},
        "tbptt": {"K": tbptt_k, "S": tbptt_s, "B": tbptt_b},
        "outputs": {"plot_dir": str(plot_dir), "model_dir": str(model_dir), "checkpoint": checkpoint_path},
    }
    write_run_config(plot_dir, run_cfg)
    write_run_config(model_dir, run_cfg)


def run_load(args) -> None:
    run_id = make_run_id("load", args)
    plot_dir = prepare_run_dir("plots/test", run_id)
    cfg = {
        "N_particles": 100000,
        "N_mesh": 4000,
        "t1": 30.0,
        "dt": 0.1,
        "boxsize": 10 * jnp.pi,
        "n0": 1.0,
        "vb": 2.4,
        "vth": 0.5,
    }
    cfg = apply_cfg_overrides(cfg, args)
    nh = cfg["N_particles"] // 2
    pic = make_pic(cfg, cfg["t1"])
    y0 = pic.create_y0(jax.random.key(pick(args.seed_ic, 42)))

    checkpoint_path = resolve_checkpoint("model", args)
    e_control = FourierActuator.load_model(checkpoint_path)
    print(e_control)
    print("Control modes:\n", e_control.get_modes_summary())

    pic = pic.run_simulation(y0, E_control=e_control)
    u = jax.vmap(e_control)(jnp.arange(pic.ts.shape[0]))

    default_modes = {
        "rho": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
        "momentum": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
        "energy": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
    }
    save_common_plots(pic, u, boxsize=cfg["boxsize"], nh=nh, out_dir=str(plot_dir), base_modes=default_modes)
    write_run_config(
        plot_dir,
        {
            "mode": "load",
            "run_id": run_id,
            "args": vars(args),
            "cfg": cfg,
            "inputs": {"checkpoint": checkpoint_path},
            "outputs": {"plot_dir": str(plot_dir)},
        },
    )


def run_load_cl(args) -> None:
    run_id = make_run_id("load_cl", args)
    plot_dir = prepare_run_dir("plots/test_cl", run_id)
    cfg = {
        "N_particles": 40000,
        "N_mesh": 256,
        "t1": 20.0,
        "dt": 0.1,
        "boxsize": 10 * jnp.pi,
        "n0": 1.0,
        "vb": 2.4,
        "vth": 0.5,
    }
    cfg = apply_cfg_overrides(cfg, args)
    nh = cfg["N_particles"] // 2
    eval_mult = pick(args.eval_mult, 2.0)
    pic = make_pic(cfg, eval_mult * cfg["t1"])
    checkpoint_path = resolve_checkpoint("model_cl", args)
    e_control = ModeFeedbackActuator.load_model(checkpoint_path)
    y0 = pic.create_y0(jax.random.key(pick(args.seed_ic, 234)))
    pic = pic.run_simulation(y0, E_control=e_control)

    e_control_vmappable = lambda n, rho_hat, mom_hat: e_control(n, state=(rho_hat, mom_hat))
    u = jax.vmap(e_control_vmappable, in_axes=(0, 0, 0))(
        jnp.arange(pic.ts.shape[0]),
        jnp.fft.rfft(pic.rho),
        jnp.fft.rfft(pic.momentum),
    )

    state_modes = {
        "rho": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
        "momentum": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
        "energy": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
    }
    ext_modes = {"max_mode_spect": 10, "max_mode_time": 10, "num": 6}
    save_common_plots(
        pic,
        u,
        boxsize=cfg["boxsize"],
        nh=nh,
        out_dir=str(plot_dir),
        base_modes=state_modes,
        with_external_modes=True,
        external_modes=ext_modes,
    )
    write_run_config(
        plot_dir,
        {
            "mode": "load_cl",
            "run_id": run_id,
            "args": vars(args),
            "cfg": cfg,
            "inputs": {"checkpoint": checkpoint_path},
            "outputs": {"plot_dir": str(plot_dir)},
        },
    )


def run_load_cl_dis(args) -> None:
    run_id = make_run_id("load_cl_dis", args)
    plot_dir = prepare_run_dir("plots/test_cl_dis", run_id)
    cfg = {
        "N_particles": 40000,
        "N_mesh": 256,
        "t1": 20.0,
        "dt": 0.1,
        "boxsize": 10 * jnp.pi,
        "n0": 1.0,
        "vb": 2.4,
        "vth": 0.5,
    }
    cfg = apply_cfg_overrides(cfg, args)
    nh = cfg["N_particles"] // 2
    eval_mult = pick(args.eval_mult, 2.0)
    pic = make_pic(cfg, eval_mult * cfg["t1"])
    checkpoint_path = resolve_checkpoint("model_cl_dis", args)
    e_control = DissipativeModeFeedbackActuator.load_model(checkpoint_path)
    y0 = pic.create_y0(jax.random.key(pick(args.seed_ic, 234)))
    pic = pic.run_simulation(y0, E_control=e_control)

    e_control_vmappable = lambda n, x: e_control(n, state=x)
    u = jax.vmap(e_control_vmappable, in_axes=(0, 0))(jnp.arange(pic.ts.shape[0]), jnp.fft.rfft(pic.momentum))

    state_modes = {
        "rho": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
        "momentum": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
        "energy": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
    }
    ext_modes = {"max_mode_spect": 10, "max_mode_time": 10, "num": 6}
    save_common_plots(
        pic,
        u,
        boxsize=cfg["boxsize"],
        nh=nh,
        out_dir=str(plot_dir),
        base_modes=state_modes,
        with_external_modes=True,
        external_modes=ext_modes,
    )
    write_run_config(
        plot_dir,
        {
            "mode": "load_cl_dis",
            "run_id": run_id,
            "args": vars(args),
            "cfg": cfg,
            "inputs": {"checkpoint": checkpoint_path},
            "outputs": {"plot_dir": str(plot_dir)},
        },
    )


def run_opt_cl_dis(args) -> None:
    run_id = make_run_id("opt_cl_dis", args)
    plot_dir = prepare_run_dir("plots/trained_cl_dis", run_id)
    model_dir = prepare_run_dir("model_cl_dis", run_id)
    cfg = {
        "N_particles": 40000,
        "N_mesh": 256,
        "t1": 20.0,
        "dt": 0.1,
        "boxsize": 10 * jnp.pi,
        "n0": 1.0,
        "vb": 2.4,
        "vth": 0.5,
    }
    cfg = apply_cfg_overrides(cfg, args)
    nh = cfg["N_particles"] // 2
    seed_ic = pick(args.seed_ic, 10)
    n_modes_space_out = 10
    train_steps = pick(args.train_steps, 200)
    save_every = pick(args.save_every, 100)
    train_seed = pick(args.train_seed, seed_ic)
    num_ics = args.num_ics
    eval_mult = pick(args.eval_mult, 2.0)
    tbptt_k = args.tbptt_k
    tbptt_s = args.tbptt_s
    tbptt_b = pick(args.tbptt_b, 4)
    lr_start = pick(args.lr_start, 5e-2)
    lr_end = pick(args.lr_end, 1e-3)

    pic = make_pic(cfg, cfg["t1"])
    e_control = DissipativeModeFeedbackActuator(
        N_mesh=pic.N_mesh,
        boxsize=pic.boxsize,
        n_modes_space_in=10,
        n_modes_space_out=n_modes_space_out,
        width=32,
        depth=2,
        u_max=None,
        include_dc=False,
        closed_loop=True,
        key=jax.random.PRNGKey(9),
    )

    optimizer = Optimizer(
        pic=pic,
        model=e_control,
        loss_metric=loss_metric_stable,
        lr=lr_start / tbptt_b,
        lr_final=lr_end / tbptt_b,
        save_dir=f"{model_dir}/",
        tbptt_k=tbptt_k,
        tbptt_s=tbptt_s,
        batch_size=tbptt_b,
        num_ics=num_ics,
    )
    e_control, train_losses, _ = optimizer.train(
        n_steps=train_steps, save_every=save_every, seed=train_seed, ic_seed=seed_ic, print_status=True
    )

    pic_eval = make_pic(cfg, eval_mult * cfg["t1"])
    checkpoint_path = str(model_dir / "model_checkpoint_final")
    e_control = DissipativeModeFeedbackActuator.load_model(checkpoint_path)
    y0_eval = pic_eval.create_y0(jax.random.key(pick(args.seed_ic_eval, 1024)))
    pic_eval = pic_eval.run_simulation(y0_eval, E_control=e_control)

    e_control_vmappable = lambda n, x: e_control(n, state=x)
    u = jax.vmap(e_control_vmappable, in_axes=(0, 0))(
        jnp.arange(pic_eval.ts.shape[0]), jnp.fft.rfft(pic_eval.momentum)
    )

    state_modes = {
        "rho": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
        "momentum": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
        "energy": {"max_mode_spect": 100, "max_mode_time": 5, "num": 6},
    }
    ext_modes = {"max_mode_spect": n_modes_space_out, "max_mode_time": n_modes_space_out, "num": 6}
    save_common_plots(
        pic_eval,
        u,
        boxsize=cfg["boxsize"],
        nh=nh,
        out_dir=str(plot_dir),
        base_modes=state_modes,
        with_external_modes=True,
        external_modes=ext_modes,
    )
    save_time_series(
        jnp.sum(pic_eval.energy, axis=-1),
        pic_eval.ts,
        str(plot_dir / "energy_evolution.png"),
        ylabel="Energy",
        title="Energy Evolution",
    )
    save_training_curve(train_losses, str(plot_dir / "train_losses.png"), use_log=True)
    run_cfg = {
        "mode": "opt_cl_dis",
        "run_id": run_id,
        "args": vars(args),
        "cfg": cfg,
        "train": {
            "train_steps": train_steps,
            "save_every": save_every,
            "train_seed": train_seed,
            "num_ics": num_ics,
            "lr_start": lr_start,
            "lr_end": lr_end,
        },
        "eval": {"eval_mult": eval_mult, "seed_ic_eval": pick(args.seed_ic_eval, 1024)},
        "tbptt": {"K": tbptt_k, "S": tbptt_s, "B": tbptt_b},
        "outputs": {"plot_dir": str(plot_dir), "model_dir": str(model_dir), "checkpoint": checkpoint_path},
    }
    write_run_config(plot_dir, run_cfg)
    write_run_config(model_dir, run_cfg)


def run_zir(args) -> None:
    run_id = make_run_id("zir", args)
    plot_dir = prepare_run_dir("plots/zir", run_id)
    cfg = {
        "N_particles": 100000,
        "N_mesh": 400,
        "t1": 30.0,
        "dt": 0.1,
        "boxsize": 10 * jnp.pi,
        "n0": 1.0,
        "vb": 2.4,
        "vth": 0.5,
    }
    cfg = apply_cfg_overrides(cfg, args)
    nh = cfg["N_particles"] // 2
    pic = make_pic(cfg, cfg["t1"])
    y0 = pic.create_y0(jax.random.key(pick(args.seed_ic, 42)), eps=1e-2)
    pic = pic.run_simulation(y0)

    default_modes = {
        "rho": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
        "momentum": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
        "energy": {"max_mode_spect": 10, "max_mode_time": 5, "num": 4},
    }
    save_state_only_plots(pic, boxsize=cfg["boxsize"], nh=nh, out_dir=str(plot_dir), base_modes=default_modes)
    save_time_series(
        jnp.sum(pic.energy + pic.E_field**2, axis=-1),
        pic.ts,
        str(plot_dir / "energy_evolution.png"),
        ylabel="Energy",
        title="Energy Evolution",
    )
    write_run_config(
        plot_dir,
        {
            "mode": "zir",
            "run_id": run_id,
            "args": vars(args),
            "cfg": cfg,
            "outputs": {"plot_dir": str(plot_dir)},
        },
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run PIC workflows from one entrypoint.")
    parser.add_argument(
        "mode",
        choices=["resp", "opt", "opt_cl", "opt_cl_self", "load", "load_cl", "load_cl_dis", "opt_cl_dis", "zir"],
        help="Workflow to run.",
    )
    parser.add_argument("--n-particles", type=int, default=None, help="Override number of particles.")
    parser.add_argument("--n-mesh", type=int, default=None, help="Override number of mesh cells.")
    parser.add_argument("--t1", type=float, default=None, help="Override base simulation end time.")
    parser.add_argument("--dt", type=float, default=None, help="Override time step.")
    parser.add_argument("--boxsize", type=float, default=None, help="Override periodic domain size.")
    parser.add_argument("--n0", type=float, default=None, help="Override electron number density.")
    parser.add_argument("--vb", type=float, default=None, help="Override beam velocity.")
    parser.add_argument("--vth", type=float, default=None, help="Override beam width.")
    parser.add_argument("--seed-ic", type=int, default=None, help="Override initial-condition seed.")
    parser.add_argument("--seed-ic-eval", type=int, default=None, help="Override evaluation IC seed.")
    parser.add_argument("--train-steps", type=int, default=None, help="Override training iteration count.")
    parser.add_argument("--save-every", type=int, default=None, help="Override checkpoint frequency.")
    parser.add_argument("--train-seed", type=int, default=None, help="Override optimizer train seed.")
    parser.add_argument(
        "--lr-start",
        type=float,
        default=None,
        help="Initial learning rate at start of training (before decay).",
    )
    parser.add_argument(
        "--lr-end",
        type=float,
        default=None,
        help="Final learning rate at end of training.",
    )
    parser.add_argument(
        "--open-n-modes-time",
        type=int,
        default=None,
        help="Open-loop Fourier actuator trainable time rFFT modes (opt mode).",
    )
    parser.add_argument(
        "--open-n-modes-space",
        type=int,
        default=None,
        help="Open-loop Fourier actuator spatial Fourier modes (opt mode).",
    )
    parser.add_argument(
        "--open-init-scale",
        type=float,
        default=None,
        help="Open-loop Fourier actuator initialization scale (opt mode).",
    )
    parser.add_argument("--tbptt-k", type=int, default=None, help="TBPTT truncation length K.")
    parser.add_argument("--tbptt-s", type=int, default=None, help="TBPTT stride S (sliding uses S < K).")
    parser.add_argument("--tbptt-b", type=int, default=None, help="Trajectory batch size B per optimizer step.")
    parser.add_argument("--num-ics", type=int, default=10, help="Number of random training ICs cycled during optimization.")
    parser.add_argument("--eval-mult", type=float, default=None, help="Multiplier for eval horizon vs t1.")
    parser.add_argument(
        "--run-name",
        "--run-dir",
        dest="run_name",
        type=str,
        default=None,
        help="Optional run folder name (alias: --run-dir). Default is timestamped auto name.",
    )
    parser.add_argument(
        "--model-run",
        type=str,
        default=None,
        help="For load/load_cl: load checkpoint from model/<model_run>/model_checkpoint_final.",
    )
    parser.add_argument(
        "--resp-amp",
        type=float,
        default=None,
        help="Override response forcing amplitude (resp mode only).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "resp":
        run_resp(args)
    elif args.mode == "opt":
        run_opt(args)
    elif args.mode == "opt_cl":
        run_opt_cl(args)
    elif args.mode == "opt_cl_self":
        run_opt_cl_self(args)
    elif args.mode == "load":
        run_load(args)
    elif args.mode == "load_cl":
        run_load_cl(args)
    elif args.mode == "load_cl_dis":
        run_load_cl_dis(args)
    elif args.mode == "opt_cl_dis":
        run_opt_cl_dis(args)
    else:
        run_zir(args)


if __name__ == "__main__":
    main()