# HazardNav v11 showcase

This folder packages the strongest visually inspected HazardNav v11 checkpoint,
three deterministic rollout videos, and a fixed-level evaluation result.

## Included files

- Model: `tasks/hazard_nav/checkpoints/hazard_nav_v11_best_s42.pt`
- Task: `Isaac-USV-HazardNav-Direct-v3`
- Observation: 42 values
- Videos:
  - `videos/hazard_nav_v11_seed42.mp4`
  - `videos/hazard_nav_v11_seed123.mp4`
  - `videos/hazard_nav_v11_seed456.mp4`
- Evaluation: `eval/v11_best_level1_seed42.json`
- Comparison with the student baseline: `COMPARISON_WITH_JINSHI_BRADY.md`

All three videos use the same deterministic checkpoint. Only the environment
seed changes, so the clips show different obstacle layouts rather than repeated
recordings of one map.

## Independent level-1 evaluation

Protocol: 64 parallel environments, 64 completed episodes, seed 42,
curriculum frozen at level 1, deterministic mean actions.

| Metric | Result |
|---|---:|
| Success rate | 0.3281 (21/64) |
| Obstacle-aware SPL | 0.2916 |
| Mean time to success | 19.43 s |
| Mean path length | 27.36 m |
| Mean minimum clearance | 0.255 m |
| Reverse-action ratio before goal | 0.8904 |
| Reverse-velocity ratio before goal | 0.8944 |

SPL is computed using the environment's saved obstacle-aware shortest route:

```text
L* = max(route_geodesic_length - goal_radius, 0)
SPL = success * L* / max(actual_path_length, L*)
```

It does not use the obstructed straight-line distance.

## Known limitation

v11 is good at reaching the circle and avoiding many obstacles, but it strongly
prefers stern-first travel. The independent evaluation measured reverse motion
on about 89% of pre-goal control steps. The small v11 reverse-action cost is not
large enough to overcome the goal-entry and time bonuses. This branch publishes
v11 as the current navigation showcase, not as a solved forward-transit policy.

## Visualize the checkpoint

After synchronizing `tasks/hazard_nav` into the Isaac Lab task installation:

```powershell
conda activate isaaclab
$env:USVBENCH_ASSETS="C:\path\to\usvbench\assets"
cd C:\path\to\IsaacLab

python .\scripts\reinforcement_learning\skrl\play.py `
  --task=Isaac-USV-HazardNav-Direct-v3 `
  --num_envs=1 `
  --seed=42 `
  --checkpoint="C:\path\to\usvbench\tasks\hazard_nav\checkpoints\hazard_nav_v11_best_s42.pt"
```

## Re-run the fixed-level evaluation

```powershell
python C:\path\to\usvbench\scripts\eval_hazard_nav.py `
  --task=Isaac-USV-HazardNav-Direct-v3 `
  --num_envs=64 `
  --episodes=64 `
  --eval_level=1 `
  --seed=42 `
  --headless `
  --checkpoint="C:\path\to\usvbench\tasks\hazard_nav\checkpoints\hazard_nav_v11_best_s42.pt"
```
