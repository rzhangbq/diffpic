# diffpic

Single entrypoint for all workflows:

```bash
python main.py <mode> [options]
```

Available modes:

- `resp` - open-loop forced response run
- `zir` - zero-input (no external control) run
- `opt` - train open-loop Fourier actuator, then evaluate
- `opt_cl` - train closed-loop mode-feedback actuator, then evaluate
- `opt_cl_dis` - train closed-loop dissipative actuator, then evaluate
- `load` - load trained open-loop actuator and evaluate
- `load_cl` - load trained closed-loop actuator and evaluate

## Example runs (all cases)

```bash
python main.py resp
python main.py zir
python main.py opt
python main.py opt_cl
python main.py opt_cl_dis
python main.py load
python main.py load_cl
```

## Fair-comparison runs

To compare methods fairly, use the same:

- initial-condition seed (`--seed-ic`)
- training steps (`--train-steps`) for trainable modes
- simulation time horizon (`--t1`) and step (`--dt`)
- particle/mesh/distribution settings (`--n-particles`, `--n-mesh`, `--boxsize`, `--n0`, `--vb`, `--vth`)
- evaluation horizon multiplier (`--eval-mult`)

Example baseline (shared across all modes):

```bash
COMMON="--seed-ic 10 --t1 20 --dt 0.1 --n-particles 40000 --n-mesh 256 --boxsize 31.4159265359 --n0 1 --vb 2.4 --vth 0.5 --eval-mult 2"
TRAIN_COMMON="--train-steps 200 --save-every 100 --train-seed 0"

python main.py resp $COMMON --resp-amp 100000
python main.py zir $COMMON
python main.py opt $COMMON $TRAIN_COMMON --seed-ic-eval 10
python main.py opt_cl $COMMON $TRAIN_COMMON --seed-ic-eval 10
python main.py opt_cl_dis $COMMON $TRAIN_COMMON --seed-ic-eval 10
python main.py load $COMMON
python main.py load_cl $COMMON
```

Notes:

- `--seed-ic-eval` controls the post-training evaluation IC for training modes.
- `load` and `load_cl` use pre-trained checkpoints (`model/` and `model_cl/`).
- Each mode writes outputs to its own plot directory under `plots/`.
