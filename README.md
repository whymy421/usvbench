# USVBench

> **Status**: Active development (private). Open-source release targeted for 2026.

USVBench is a reinforcement-learning benchmark for unmanned surface vehicles (USVs),
built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab). It covers calm-water
navigation, path following, hazard avoidance, docking, station keeping, and mission
tasks across ROVs, monohulls, BlueBoat, and catamarans.

## Uploaded task modules

The current branch contains the 11 task modules below. `Code` means the task environment,
registration, and configuration are checked in. `Checkpoint` only reports whether a model
weight is checked in; a missing checkpoint does not mean that the task code is missing.

| Task | Vehicle / condition | Gym id(s) | Code | Checkpoint | Documentation |
|------|---------------------|-----------|------|------------|---------------|
| ROV calm navigation | Doc Ricketts ROV / calm | `Isaac-My-First-Task-Calm-Direct-v1` | uploaded | yes | [`STARTER_TASK.md`](tasks/rov_calm_nav/STARTER_TASK.md) |
| Boat calm navigation | 5 m monohull / calm | `Isaac-USVBench-Boat-Calm-Direct-v1` | uploaded | yes | [`STARTER_TASK.md`](tasks/boat_calm_nav/STARTER_TASK.md) |
| BlueBoat calm navigation | BlueBoat / calm | `Isaac-USV-BlueBoat-Calm-Direct-v1` | uploaded | yes | [`STARTER_TASK.md`](tasks/blueboat_calm_nav/STARTER_TASK.md) |
| Static hazard navigation | BlueBoat / calm and waves | `Isaac-USV-HazardNav-Direct-v1` ... `-v4`, `...-Airy-Direct-v3`, `...-Jonswap-Direct-v3` | uploaded | yes | [`STARTER_TASK.md`](tasks/hazard_nav/STARTER_TASK.md) |
| Path following | ROV and BlueBoat / calm | `Isaac-USV-PathFollow-Direct-v1`, `...-BlueBoat-Direct-v1` | uploaded | yes | [`STARTER_TASK.md`](tasks/path_following/STARTER_TASK.md) |
| Path hazard | BlueBoat / calm path and hazards | `Isaac-USV-PathHazard-Direct-v1` | uploaded | yes | [`STARTER_TASK.md`](tasks/path_hazard/STARTER_TASK.md) |
| Docking | Monohull and BlueBoat / calm | `Isaac-USV-Dock-Direct-v1`, `...-BlueBoat-Direct-v1`, `...-BlueBoat-Current-Direct-v1` | uploaded | yes | [`STARTER_TASK.md`](tasks/docking/STARTER_TASK.md) |
| Station keeping | ROV and BlueBoat / calm | `Isaac-USV-StationKeep-Direct-v1`, `...-BlueBoat-Direct-v1`, `...-BlueBoat-Current-Direct-v1` | uploaded | yes | [`STARTER_TASK.md`](tasks/station_keeping/STARTER_TASK.md) |
| Station keeping (boat) | Monohull / calm | `Isaac-USV-StationKeep-Boat-Direct-v1` | uploaded | no | [`STARTER_TASK.md`](tasks/station_keeping_boat/STARTER_TASK.md) |
| Ordered harbor mission | BlueBoat / calm | `Isaac-USV-HarborMission-Direct-v1` | uploaded | no | [`STARTER_TASK.md`](tasks/harbor_mission/STARTER_TASK.md) |
| Catamaran patrol | Catamaran / calm | `Isaac-Catamaran-Patrol-Direct-v1` | uploaded | yes | [`NOTES.md`](tasks/catamaran_patrol/NOTES.md) |

## Not uploaded yet

These entries are intentionally placeholders. No collaborator implementation is claimed
until the corresponding task directory, registration, and documentation are committed.

| Planned task | Expected deliverable | Status |
|--------------|----------------------|--------|
| Wave navigation (ROV + boat) | Wave task, baselines, and robustness metric | placeholder |
| Wave obstacle avoidance | Static/dynamic obstacles, baseline, and collision metric | placeholder |
| Multi-target / coverage variants | Task implementation and standardized evaluation | placeholder |
| Gazebo sim-to-sim validation | Isaac-vs-Gazebo trajectory comparison and transfer-gap metric | placeholder |

## Reference tasks (validated baselines)

| Task | Vessel | Condition | Evaluation reference | Folder |
|------|--------|-----------|----------------------|--------|
| **A - ROV calm navigation** | Doc Ricketts ROV | calm | **4.48** targets/episode (seed 42) | [`tasks/rov_calm_nav/`](tasks/rov_calm_nav/STARTER_TASK.md) |
| **B - Boat calm navigation** | 5 m monohull | calm | **3.75** targets/episode (seed 42) | [`tasks/boat_calm_nav/`](tasks/boat_calm_nav/STARTER_TASK.md) |

These are deterministic `eval_benchmark.py` measurements on the shipped seed-42
checkpoints, using the realistic-dynamics physics. The 3-seed means are still pending;
do not present the single-seed values as a multi-seed result. See [`TASKS.md`](TASKS.md#current-baselines)
for the measurement protocol and the source of truth.

## Wave benchmark

HazardNav provides calm, Airy, and JONSWAP variants with identical policy observation and
action layouts. Wave generation is shared across tasks in
[`tasks/_shared/waves.py`](tasks/_shared/waves.py); the JONSWAP spectrum uses 30 frequency
components and is normalized so the configured `Hs` equals `4 sqrt(m0)`.

HazardNav plant v2 couples the surface to the BlueBoat through six distributed buoyancy
stations and relative-water drag, with no authored wave-force gains. See
[`tasks/hazard_nav/WAVES.md`](tasks/hazard_nav/WAVES.md) for task ids and commands, and
[`docs/WAVE_PARAMETER_ALIGNMENT.md`](docs/WAVE_PARAMETER_ALIGNMENT.md) for the unified
`Hs`/`Tp` definitions and reporting fields. No wave-trained baseline is claimed yet.

## Repository layout

```
usvbench/
├── tasks/              # Isaac Lab task implementations
│   ├── _shared/        # shared vehicle, wave, and utility code
│   ├── rov_calm_nav/
│   ├── boat_calm_nav/
│   ├── blueboat_calm_nav/
│   ├── hazard_nav/
│   ├── path_following/
│   ├── path_hazard/
│   ├── docking/
│   ├── station_keeping/
│   ├── station_keeping_boat/
│   ├── harbor_mission/
│   └── catamaran_patrol/
├── assets/             # vessel USD models
├── scripts/            # training and evaluation launchers
├── docs/               # current collaboration and task docs
├── TASKS.md            # benchmark roadmap and measured baselines
└── README.md
```

## Quick start

1. Clone this repo and install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).
2. Activate the Isaac Lab conda environment.
3. Copy the selected task folder and `tasks/_shared/` into
   `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/`.
4. Follow that task's `STARTER_TASK.md` for the gym id, environment variables, and
   evaluation command.

## Evaluation

Reference evaluations use [`scripts/eval_benchmark.py`](scripts/eval_benchmark.py) with a
deterministic policy (mean action), a fixed step budget, and metrics normalized per
episode-equivalent. Report mean +/- std over seeds 42, 123, and 456 when three seeds are
available; a single-seed result must be labeled as such.

```bash
python scripts/eval_benchmark.py --task=<gym-id> --num_envs=64 --eval_steps=6000 \
  --headless --checkpoint=<path>/best_agent.pt
```

The task-specific numbers above are comparable within the same task, not across different
vehicles or objectives. Training-time wandb metrics must be labeled separately from the
deterministic evaluation score.

## Current documentation

- [Task roadmap and measured baselines](TASKS.md)
- [Git workflow and collaboration guide](docs/GITHUB_COLLABORATION.md)
- [Student task assignments](docs/STUDENTS_TASKS.md)

## License

TBD (MIT or Apache-2.0 on public release).

## Citation

TBD (benchmark paper in preparation, target ICRA Datasets & Benchmarks 2026).
