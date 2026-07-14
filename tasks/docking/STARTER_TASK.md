# USV Boat Docking

> **Vehicle:** realistic boat hull (`boat_physics.usdc`)
> **Condition:** calm water (no waves or current)
> **Gym id:** `Isaac-USV-Dock-Direct-v1`

## Task

Each 120 s episode asks the boat to dock at the local environment origin with
its bow pointing along world `+X`. The boat starts at rest on the current
curriculum radius, with a uniformly random bearing and uniformly random heading.
The boat asset's bow/forward axis is body `-X`.

The fixed observation is
`(dot, cross, dist/25, dock_dot, dock_cross, planar_speed/2)`. The first two
components describe the dock position relative to the boat's forward direction;
`dock_dot` and `dock_cross` align boat forward with world `+X`. Actions are
forward thrust and yaw torque in `[-1, 1]^2`.

## Reward-free success predicate

All three conditions must hold simultaneously for 5 consecutive seconds:

- horizontal position error `<= 2.5 m`;
- absolute heading error from dock heading `<= 15 degrees`;
- planar speed `<= 0.3 m/s`.

Breaking any condition resets only the 5 s timer. Success terminates the
episode. Reaching 120 s is failure. Success is computed only from state and does
not depend on reward.

## Official curriculum

The success-rate EMA starts at zero and is updated once per finished episode.
The spawn radius never decreases.

| Parameter | Value |
|---|---:|
| Initial spawn radius | 5.0 m |
| Success-rate EMA decay | 0.99 |
| Advancement threshold | 0.6 |
| Radius increment | 2.5 m |
| Maximum spawn radius | 25.0 m |

When the updated EMA is at least 0.6, that episode advances the radius by one
2.5 m stage, capped at 25 m. The EMA is retained across stages.

## Reference reward

The included reward is **reference-only** and is not part of success:

```text
-distance/25 + 0.5 * dock_dot * exp(-distance/5)
    + 1.0 * [instantaneous success predicate is true]
```

## Train

Copy this folder into the Isaac Lab direct-task package, set `USVBENCH_ASSETS`
if the repository is not at `~/usvbench`, activate Isaac Lab, then run:

```bash
python <IsaacLab>/scripts/reinforcement_learning/skrl/train_with_eval.py \
  --task=Isaac-USV-Dock-Direct-v1 \
  --num_envs=64 --headless --max_iterations=3000 --seed=42 \
  --video --video_interval 50000 --video_length 200
```

## Smoke tests

The curriculum tests and curriculum-only smoke do not require Isaac Lab or a
GPU:

```powershell
python .\tasks\docking\test_curriculum.py
python .\tasks\docking\smoke.py --curriculum-only
```

Full Isaac Lab smoke:

```powershell
$env:USVBENCH_ASSETS = (Resolve-Path .\assets).Path
python .\tasks\docking\smoke.py --headless
```
