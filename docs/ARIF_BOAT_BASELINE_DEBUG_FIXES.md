# Arif boat-baseline debug: corrected integration

This branch keeps the useful parts of `arif/boat-baseline-debug` (post-training
checkpoint scoring, old-checkpoint compatibility, and a reproducible multi-seed
launcher) on top of `yutong/boat-baseline-reference`.

## Why the branch was rebuilt from the reference

Arif's debug branch was based on the older `main` boat environment. It therefore
still had the old 5 deg/s angular-velocity cap, unbounded policy actions, the old
100 N / 120 N-m actuator defaults, and the old force-frame/damping model. Those
differences are larger than the OOB change and make the scores incomparable with
the current boat reference.

The corrected branch uses the reference versions of:

- `tasks/boat_calm_nav/my_first_task_env.py`
- `tasks/boat_calm_nav/my_first_task_env_cfg.py`
- the boat physics asset/configuration

In particular, actions are clipped to `[-1, 1]`, the yaw safety cap is no longer
5 deg/s, and body/world-frame forces use the corrected reference implementation.

## Corrections to Arif's additions

### OOB behavior

Both the terminal condition and `OOB_PENALTY` trigger at
`max_spawn_distance + 20 m` (50 m for the boat task). The debug branch had a
50 m terminal boundary but still started the reward penalty at 30 m.

### Checkpoint selection

`scripts/train_with_eval.py` now:

- keeps checkpoint selection opt-in (`--eval_mini_steps N`);
- documents that the value is evaluation steps per checkpoint, not a training
  iteration interval;
- forces a real environment reset before every candidate (the skrl Isaac Lab
  wrapper normally resets only once);
- reuses one fixed evaluation seed for every checkpoint;
- restores the same `common_step_counter` and clears action-delay/history state;
- loads old `state_preprocessor` data separately for every old-format checkpoint;
- evaluates the existing reward-selected `best_agent.pt` as a candidate too;
- preserves the original as `best_agent_by_reward.pt` if the eval sweep selects
  a different checkpoint;
- writes `eval_sweep.csv` and `best_agent_selection.json`.

The training run is closed in Weights & Biases before the sweep so evaluation
resets do not contaminate training metrics.

### Benchmark evaluation

`scripts/eval_benchmark.py` now:

- uses the agent's own action path, including its observation preprocessor;
- supports old and new skrl checkpoint preprocessor names;
- counts target events from a per-step mask rather than a periodically reset
  global counter;
- keeps `oob_per_episode` as an event rate that may exceed 1.0;
- separately reports the true `oob_fraction_completed`;
- reports completed episodes, timeouts, and mean completed-episode length;
- records training seed and fixed evaluation seed separately;
- exits normally in headless mode without waiting for keyboard input.

### Reproducible launch

Use:

```powershell
conda activate isaaclab
$env:ISAACLAB_ROOT = "C:\path\to\IsaacLab"
./scripts/train_boat_baseline_multiseed_fixed.ps1
```

Optional example:

```powershell
./scripts/train_boat_baseline_multiseed_fixed.ps1 `
  -IsaacLabRoot C:\Users\arifa\IsaacLab `
  -Seeds 42,456 `
  -MaxIterations 9375 `
  -EvalSweepSteps 1500 `
  -EvalSeed 2026
```

The launcher installs the task from this checkout, invokes the scripts from this
checkout, clears leaked experimental environment variables, records the exact
Git commit, evaluates every training seed on the same evaluation seed, and
writes logs/CSV under `outputs/boat_baseline_fixed_<timestamp>/`.

## ROV checkpoint for P0 Task A

The repository already contains:

`tasks/rov_calm_nav/checkpoints/rov_calm_s42.pt`

The task documentation reports approximately `27.1 +/- 2.4 targets/episode`
across the reference training seeds. The shipped seed-42 checkpoint can be used
for the P0 pipeline check with the corrected benchmark evaluator.
