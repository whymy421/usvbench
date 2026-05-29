# USVBench

> **Status**: Active development (private). Open-source release targeted for 2026.

An open-source reinforcement learning benchmark for unmanned surface vessels (USVs), covering vessel scales from 5m boats to 100m cruise ships, calm and irregular wave conditions, single and multi-agent tasks.

## What's here

- `tasks/` — Isaac Lab task implementations
  - `my_first_task_boat_calm/` — single 5m monohull, calm water, point navigation (reference baseline)
  - `my_first_task_e6/` — 3-USV SAR in JONSWAP waves (multi-agent cooperation research)
- `assets/` — vessel USD models
- `docs/` — design documents and task assignments
- `scripts/` — training launcher scripts

## Quick start

See `tasks/my_first_task_boat_calm/STARTER_TASK.md`.

## Authors

- Yutong Wang, UCL Mechanical Engineering (lead)
- Arif (contributor, ongoing)

## License

TBD (will be MIT or Apache-2.0 on public release).

## Citation

TBD (benchmark paper in preparation, target NeurIPS Datasets & Benchmarks 2026).
