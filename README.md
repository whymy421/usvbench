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
| **B — Boat calm nav** | 5 m monohull | calm | ~5.6 targets/ep | [`tasks/boat_calm_nav/`](tasks/boat_calm_nav/STARTER_TASK.md) |

Both are single-agent PPO point-navigation tasks. **Task A is the recommended Day-1
warmup** (trains in ~30 min with the simplest reward). Task B is the same task on a real
boat hull and needs a tuned reward (see its STARTER doc for why).

Wave-aware single-agent (E7) and multi-USV cooperative (E6) tasks are maintained by
Yutong and will be added to this repo as they stabilise.

## Repo layout

```
usvbench/
├── tasks/              # Isaac Lab task implementations (drop into IsaacLab .../direct/)
│   ├── rov_calm_nav/   # Task A  + STARTER_TASK.md
│   └── boat_calm_nav/  # Task B  + STARTER_TASK.md
├── assets/             # vessel USD models (ROV_rigged.usd, boat_physics.usdc)
├── scripts/            # ready-to-run training launchers (.ps1)
├── docs/
│   ├── ARIF_TASKS.md           # contributor roadmap (week-by-week)
│   ├── GITHUB_COLLABORATION.md # git workflow for contributors
│   └── BENCHMARK_DESIGN.md     # overall architecture
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

## For contributors

- **Roadmap & your assignment**: [`docs/ARIF_TASKS.md`](docs/ARIF_TASKS.md)
- **Git workflow (how to clone, branch, PR)**: [`docs/GITHUB_COLLABORATION.md`](docs/GITHUB_COLLABORATION.md)

## Authors

- Yutong Wang, UCL Mechanical Engineering (lead)
- Arif (contributor, ongoing)

## License

TBD (MIT or Apache-2.0 on public release).

## Citation

TBD (benchmark paper in preparation, target NeurIPS Datasets & Benchmarks 2026).
