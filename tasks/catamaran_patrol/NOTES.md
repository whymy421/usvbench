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

## Seed results

> The P1 bar is currently **TBD** in `docs/ARIF_TASKS.md`. The old ≥2.0 was calibrated
> against reference baselines roughly 5x higher than they are now. With this task ported
> onto the reference physics and scoring 17.850 on seed 42, the bar can be set by the same
> ~80% proportion the P0 bars use — but that is Yutong's call once 123/456 are in.
| Seed | Checkpoint | tgt/ep | source |
|------|-----------|--------|--------|
| 42 | agent_1500000.pt (1.5M steps), pre-fix | 1.35 reported / **0.000 measured** | reported number's origin unconfirmed; measured with `eval_benchmark.py`, 64×6000, eval seed 2026 |
| 42 | agent_256000.pt (256k steps), post-fix | 21.469 | before the hull rotation / heave damping |
| 42 | agent_256000.pt, ported but with the broken attitude spring | 15.600 | void — the hull was tumbling |
| 42 | agent_256000.pt (256k steps), **ported to the reference physics** | **17.850** @ 2.556 m/s | `eval_benchmark.py`, 64×6000, eval seed 2026 |
| 123 | not yet trained | — | |
| 456 | not yet trained | — | |

Seeds 123/456 still to run before P1 can be signed off.

## Physics notes
- **Ported onto the reference physics on 2026-07-31** — per-DOF linear+quadratic damping
  Froude-scaled from the VRX WAM-V, actuator limits in the cfg, attitude spring,
  non-binding velocity caps. See `FIXES.md` for the derivation and the measured terminal
  values (2.500 m/s surge, 1.000 rad/s yaw, both exactly on their design targets).
- The attitude spring must stay yaw-invariant, `k*(hull_up x world_up)`. The Euler-angle
  form capsizes the hull as soon as it turns, and measures clean at yaw = 0.
  `check_catamaran_physics.py` phase 3 is the guard.
- Catamaran USD: `tasks/catamaran_patrol/assets/catamaran.usd (committed to branch). Set USVBENCH_ASSETS=<repo_root>/tasks/catamaran_patrol/assets before running.`
- Ground plane placed at z=-50m to prevent hull clipping
- Buoyancy: volume=0.3 m³, density=1000 kg/m³ → net upward force at surface
- Linear/angular damping applied manually via external force each physics substep
- Forward axis: body +X (FWD_X=1)
