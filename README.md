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
- `opt_cl_self` - train closed-loop controller to cancel self field directly (`E_ext -> -E_self`)
- `opt_cl_dis` - train closed-loop dissipative actuator, then evaluate
- `load` - load trained open-loop actuator and evaluate
- `load_cl` - load trained closed-loop actuator and evaluate
- `load_cl_dis` - load trained closed-loop dissipative actuator and evaluate

## Example runs (all cases)

```bash
python main.py resp
python main.py zir
python main.py opt
python main.py opt_cl
python main.py opt_cl_self
python main.py opt_cl_dis
python main.py load
python main.py load_cl
python main.py load_cl_dis
```

## Fair-comparison runs

To compare methods fairly, use the same:

- initial-condition seed (`--seed-ic`)
- training steps (`--train-steps`) for trainable modes
- simulation time horizon (`--t1`) and step (`--dt`)
- particle/mesh/distribution settings (`--n-particles`, `--n-mesh`, `--boxsize`, `--n0`, `--vb`, `--vth`)
- evaluation horizon multiplier (`--eval-mult`)
- TBPTT params for closed-loop training (`--tbptt-k`, `--tbptt-s`, `--tbptt-b`)

Example baseline (shared across all modes):

```bash
# zsh-safe argument bundles (also works in bash)
COMMON=(--seed-ic 1907 --t1 20 --dt 0.1 --n-particles 40000 --n-mesh 256 --boxsize 31.4159265359 --n0 1 --vb 2.4 --vth 0.5 --eval-mult 2)
TRAIN_COMMON=(--train-steps 300 --save-every 100 --train-seed 0 --num-ics 1)
CL_B1_NAIVE=(--tbptt-k 200 --tbptt-s 200 --tbptt-b 1)    # batch=1 naive BPTT
CL_B1_TBPTT=(--tbptt-k 100 --tbptt-s 100 --tbptt-b 1)      # batch=1 TBPTT
CL_B1_SLIDE=(--tbptt-k 100 --tbptt-s 25 --tbptt-b 1)       # batch=1 sliding-window TBPTT
CL_B4_NAIVE=(--tbptt-k 200 --tbptt-s 200 --tbptt-b 10)   # batched naive BPTT
CL_B4_TBPTT=(--tbptt-k 100 --tbptt-s 100 --tbptt-b 10)     # batched TBPTT
CL_B4_SLIDE=(--tbptt-k 100 --tbptt-s 25 --tbptt-b 10)     # batched sliding-window TBPTT
EXP=fair_cmp

# Open-loop training
python main.py opt "${COMMON[@]}" "${TRAIN_COMMON[@]}" --seed-ic-eval 5212 --run-name "${EXP}_opt"&
# Closed-loop self-field cancellation overfit check (target: E_ext = -E_self)
python main.py opt_cl_self "${COMMON[@]}" "${TRAIN_COMMON[@]}" "${CL_B1_NAIVE[@]}" --num-ics 1 --seed-ic-eval 5212 --run-name "${EXP}_optcl_self_overfit"&
# Closed-loop ablations: _cl under matched train budget
python main.py opt_cl "${COMMON[@]}" "${TRAIN_COMMON[@]}" "${CL_B1_NAIVE[@]}" --seed-ic-eval 5212 --run-name "${EXP}_optcl_b1_naive"&

python main.py opt_cl "${COMMON[@]}" "${TRAIN_COMMON[@]}" "${CL_B1_TBPTT[@]}" --seed-ic-eval 5212 --run-name "${EXP}_optcl_b1_tbptt"&

python main.py opt_cl "${COMMON[@]}" "${TRAIN_COMMON[@]}" "${CL_B1_SLIDE[@]}" --seed-ic-eval 5212 --run-name "${EXP}_optcl_b1_slide"

python main.py opt_cl "${COMMON[@]}" "${TRAIN_COMMON[@]}" "${CL_B4_NAIVE[@]}" --seed-ic-eval 5212 --run-name "${EXP}_optcl_b4_naive"&

python main.py opt_cl "${COMMON[@]}" "${TRAIN_COMMON[@]}" "${CL_B4_TBPTT[@]}" --seed-ic-eval 5212 --run-name "${EXP}_optcl_b4_tbptt"&

python main.py opt_cl "${COMMON[@]}" "${TRAIN_COMMON[@]}" "${CL_B4_SLIDE[@]}" --seed-ic-eval 5212 --run-name "${EXP}_optcl_b4_slide"

# Reproduce legacy open-loop static-field case (n_modes_time=1, n_modes_space=4)
python main.py opt --num-ics 1 --seed-ic 10 --seed-ic-eval 10 --t1 20 --dt 0.1 --n-particles 40000 --n-mesh 256 --boxsize 31.4159265359 --n0 1 --vb 2.4 --vth 0.5 --open-n-modes-time 1 --open-n-modes-space 4 --open-init-scale 1e-4 --lr-start 1e-1 --lr-end 1e-1 --train-steps 200 --save-every 100 --train-seed 0 --eval-mult 2 --tbptt-b 1 --run-name legacy_opt_repro

# testing on one unseen random IC (same IC for all comparisons)
TEST_COMMON=(--seed-ic 4211 --t1 20 --dt 0.1 --n-particles 40000 --n-mesh 256 --boxsize 31.4159265359 --n0 1 --vb 2.4 --vth 0.5 --eval-mult 2)

# Compare open-loop trained model vs resp/zir
python main.py resp "${TEST_COMMON[@]}" --resp-amp 1.0 --run-name "${EXP}_test_opt_resp"
python main.py zir "${TEST_COMMON[@]}" --run-name "${EXP}_test_opt_zir"
python main.py load "${TEST_COMMON[@]}" --model-run "${EXP}_opt" --run-name "${EXP}_test_opt_load"

# Compare each opt_cl variant vs resp/zir (same test IC)
for RUN in optcl_b1_naive optcl_b1_tbptt optcl_b1_slide optcl_b4_naive optcl_b4_tbptt optcl_b4_slide; do
  python main.py resp "${TEST_COMMON[@]}" --resp-amp 1.0 --run-name "${EXP}_test_${RUN}_resp"
  python main.py zir "${TEST_COMMON[@]}" --run-name "${EXP}_test_${RUN}_zir"
  python main.py load_cl "${TEST_COMMON[@]}" --model-run "${EXP}_${RUN}" --run-name "${EXP}_test_${RUN}_loadcl"
done
```

Notes:

- `--seed-ic-eval` controls the post-training evaluation IC for training modes.
- In training modes (`opt`, `opt_cl`, `opt_cl_self`, `opt_cl_dis`), `--seed-ic` controls training IC generation and `--train-seed` is separate (optimizer seed only).
- `--num-ics` controls how many random training ICs are pre-generated and cycled (default: `10`).
- Every execution creates a unique run folder and writes `run_config.json`.
  - plots: `plots/<mode_group>/<run_id>/`
  - training checkpoints: `model/<run_id>/`, `model_cl/<run_id>/`, `model_cl_dis/<run_id>/`
- `load` and `load_cl` resolve checkpoints from the latest run by default; use `--model-run <run_id>` to pin a specific run.
- `opt_cl_self` writes outputs to `plots/trained_cl_self/<run_id>/` and checkpoints to `model_cl_self/<run_id>/`.
- Use `--run-name <name>` (alias: `--run-dir <name>`) to set an explicit run folder name.
- TBPTT semantics:
  - `K = --tbptt-k` is truncate window length.
  - `S = --tbptt-s` is sliding stride (`S=K` is naive TBPTT, `S<K` is sliding-window TBPTT).
  - `B = --tbptt-b` is trajectory batch size per optimizer step.
  - `--train-steps` is treated as the total trajectory budget; changing `B` does not change total trajectories seen.
  - For batched runs, training uses `effective_lr = base_lr * B` to compensate for reduced optimizer update count.