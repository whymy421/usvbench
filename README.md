# USVBench

> **Status**: Active development (private). Open-source release targeted for 2026.

An open-source reinforcement-learning benchmark for **unmanned surface vehicles (USVs)**,
spanning vessel scales from small ROVs and 5 m boats up to ~100 m ships, in calm and
irregular-wave conditions. Built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

Unlike existing open USV environments (which explicitly use simplified physics), USVBench
provides buoyancy, depth-dependent damping, ocean currents and JONSWAP irregular waves,
with reproducible RL baselines for each vessel × task combination.

## Reference tasks (validated baselines)

| Task | Vessel | Condition | Baseline | Folder |
|------|--------|-----------|----------|--------|
| **A — ROV calm nav** | Doc Ricketts ROV | calm | ~24 targets/ep | [`tasks/rov_calm_nav/`](tasks/rov_calm_nav/STARTER_TASK.md) |
| **B — Boat calm nav** | 5 m monohull | calm | ~4.8 targets/ep | [`tasks/boat_calm_nav/`](tasks/boat_calm_nav/STARTER_TASK.md) |

Both are single-agent PPO point-navigation tasks. **Task A is the recommended Day-1
warmup** (trains in ~30 min with the simplest reward). Task B is the same task on a real
boat hull and needs a tuned reward (see its STARTER doc for why).

More vessel scales and conditions will be added as the benchmark grows.

## Repo layout

```
usvbench/
├── tasks/              # Isaac Lab task implementations (drop into IsaacLab .../direct/)
│   ├── rov_calm_nav/   # Task A  + STARTER_TASK.md
│   └── boat_calm_nav/  # Task B  + STARTER_TASK.md
├── assets/             # vessel USD models (ROV_rigged.usd, boat_physics.usdc)
├── scripts/            # training launchers (.ps1) + eval_benchmark.py
├── docs/
│   ├── ARIF_TASKS.md           # contributor roadmap (week-by-week)
│   ├── EMAIL_TO_ARIF.md        # contributor onboarding note
│   └── GITHUB_COLLABORATION.md # git workflow + wandb for contributors
├── TASKS.md            # benchmark roadmap (physics cleanup, enrichment, Gazebo)
└── README.md
```

## Quick start

1. Clone this repo to your home directory (so `~/usvbench/assets` resolves automatically):
   ```bash
   git clone https://github.com/whymy421/usvbench.git ~/usvbench
   ```
2. Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and activate its conda env.
3. Copy a task folder into Isaac Lab and run it — see
   [`tasks/rov_calm_nav/STARTER_TASK.md`](tasks/rov_calm_nav/STARTER_TASK.md).

## Evaluation

All baselines use one standardized protocol — [`scripts/eval_benchmark.py`](scripts/eval_benchmark.py):
a **deterministic** policy (mean action), a **fixed step budget**, and metrics normalized
per **episode-equivalent**. (These are continuous-retargeting tasks that rarely terminate
inside the window, so we normalize by `total_steps / episode_length`, not by completed
episodes.) Report **mean ± std over 3 seeds (42 / 123 / 456)** — not a single seed.

```bash
python scripts/eval_benchmark.py --task=<gym-id> --num_envs=64 --eval_steps=6000 \
  --headless --checkpoint=<path>/best_agent.pt
```
(Set the same `OBS_DIM`/`OBS_EXTENDED` you trained with — the observation shape must match
the network. Reward env-vars don't affect evaluation.)

| Task | targets/ep (primary) | mean speed | notes |
|------|----------------------|------------|-------|
| ROV calm | **27.1 ± 2.4** (n=3) | 6.6 m/s | robust across seeds |
| boat calm | ~5.1 (seed 42 only) | 4.6 m/s | 3-seed baseline pending; higher seed-variance |

> The **Reference tasks** table above quotes *training-time* throughput (~24 / ~4.8);
> the numbers here are the *standardized deterministic eval* (`eval_benchmark.py`), which
> runs a little higher (~27 / ~5). Compare new methods against these eval numbers.

**Metrics**: `targets_per_episode` (navigation throughput, **primary**) · `mean_speed` ·
`oob_per_episode` (out-of-bounds events per episode — a control-quality diagnostic; in
calm water this is *overshoot*, not a safety failure — safety only becomes meaningful
under waves). Each task's number is its own reference; numbers are comparable *within* a
task (a new method vs the reference), not *across* tasks.

## For contributors

- **Roadmap & your assignment**: [`docs/ARIF_TASKS.md`](docs/ARIF_TASKS.md)
- **Git workflow (how to clone, branch, PR)**: [`docs/GITHUB_COLLABORATION.md`](docs/GITHUB_COLLABORATION.md)

## Authors

- Song Yutong, UCL Mechanical Engineering (lead)
- Arif (contributor, ongoing)

## License

TBD (MIT or Apache-2.0 on public release).

## Citation

TBD (benchmark paper in preparation, target NeurIPS Datasets & Benchmarks 2026).
