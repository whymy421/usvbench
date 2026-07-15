# USV Station Keeping

> **Vehicle:** realistic ROV (`ROV_rigged.usd`)  
> **Condition:** calm water (no waves or current)  
> **Gym id:** `Isaac-USV-StationKeep-Direct-v1`

## Task

Each 120 s episode asks one ROV to approach and hold the fixed `(x, y)` origin of
its environment. The ROV starts at rest at `z=-0.15`, a uniformly sampled radial
distance of 5-15 m away, with a uniformly random heading.

The policy observation is `(dot, cross, dist_norm)`, where `dot` and `cross`
describe the heading relative to the hold point and `dist_norm = distance / 15`.
The action is `(forward_thrust, yaw_torque)` in `[-1, 1]^2`.

## Reward-free success criterion

Success depends only on state: horizontal distance must remain at or below 2.0 m
for 60 consecutive seconds. Leaving the zone resets the hold timer and does not
end the episode. Success terminates the episode; reaching the 120 s time limit
without success is failure.

The environment exposes per-environment `hold_timer` (seconds) and `path_length`
(cumulative planar metres). Completed episodes log `success`, `time_to_success_s`
(`nan` on failure), `path_length_m`, and `final_hold_timer_s`.

## Reference reward

The included reward is a **reference baseline; methods are free to use any
shaping**:

```text
-distance / 30 + (1.0 if distance <= 2.0 else 0.0)
```

This reward is not part of the success predicate.

## Visualization

`StationKeepingEnvCfg.visual` controls the calm water surface and the orange
2 m hold-zone ring used in rendered demos. These settings are render-only and
are never read by physics, observations, reward, termination, or the success
predicate. Record demo videos with both `enable_water` and
`enable_hold_zone_marker` enabled (the defaults).

## Install

Copy this folder into the Isaac Lab direct-task package:

```text
<IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/station_keeping/
```

Set `USVBENCH_ASSETS` to this repository's `assets` directory if the repository
is not at `~/usvbench`, then activate the Isaac Lab environment.

## Train

The observation size is fixed to 3 in `StationKeepingEnvCfg`; no observation
environment variable is needed.

```bash
python <IsaacLab>/scripts/reinforcement_learning/skrl/train_with_eval.py \
  --task=Isaac-USV-StationKeep-Direct-v1 \
  --num_envs=64 --headless --max_iterations=3000 --seed=42 \
  --video --video_interval 50000 --video_length 200
```

## Smoke test

From the USVBench repository:

```powershell
$env:USVBENCH_ASSETS = (Resolve-Path .\assets).Path
python .\tasks\station_keeping\smoke.py --headless
```

Expected behavior: **approach then hold; drag on the realistic hull makes holding
nearly passive in calm water — the learning content is the approach + stop**.
