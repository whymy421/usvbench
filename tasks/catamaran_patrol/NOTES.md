# Catamaran P1 — Branch Notes

> Reviewed against commit `3f2d799`; see `FIXES.md` for what was changed and why.
> The results table below is Arif's original one and has not reproduced yet.

## Task
`Isaac-Catamaran-Patrol-Direct-v1`

## Config (all defaults, no overrides)
| Parameter | Value | Note |
|-----------|-------|------|
| N_WAYPOINTS | 4 | env var default; not overridden |
| patrol_radius | 12.0 m | ±20% jitter per reset (code samples 0.8–1.2×) |
| goal_radius | 3.0 m | same as BlueBoat |
| episode_length_s | 120.0 s | |
| OBS_DIM | 12 | default |
| PROGRESS_COEF | 3.0 | default |
| REACH_BONUS | 150.0 | default |
| SPEED_COUPLE | 1 | default |
| MAX_THRUST | 200 N | hardcoded |
| MAX_TORQUE | 100 N·m | hardcoded |
| mass | 120 kg | catamaran.usd |
| max_linear_damping | 40.0 | manual water drag |
| max_angular_damping | 40.0 | |

## Metric
`targets_per_episode` = cumulative waypoints reached across the full 120 s episode.
The circuit loops, so values above N_WAYPOINTS=4 are possible.
Same counter as boat/ROV tasks (via `self.reached_count`).

## Seed results (P1 bar: ≥2.0 tgt/ep averaged across 3 seeds)
| Seed | Checkpoint | tgt/ep |
|------|-----------|--------|
| 42 | agent_1500000.pt (1.5M steps) | 1.35 |
| 123 | not yet trained | — |
| 456 | not yet trained | — |

Seed 42 is below the P1 bar. Seeds 123/456 still to run.

## Physics notes
- Catamaran USD: `tasks/catamaran_patrol/assets/catamaran.usd (committed to branch). Set USVBENCH_ASSETS=<repo_root>/tasks/catamaran_patrol/assets before running.`
- Ground plane placed at z=-50m to prevent hull clipping
- Buoyancy: volume=0.3 m³, density=1000 kg/m³ → net upward force at surface
- Linear/angular damping applied manually via external force each physics substep
- Forward axis: body +X (FWD_X=1)
