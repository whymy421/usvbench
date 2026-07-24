# v11 compared with `jinshi-brady/benchmark-v2`

Baseline reference:

- Branch: `jinshi-brady/benchmark-v2`
- Commit: `d49d0c195564c1c8b63f9f9559afbf504fc7b943`

The v11 showcase branch is based directly on that commit. The comparison below
therefore describes the actual code delta, not a comparison inferred from folder
names or old videos.

## Main differences

| Area | Student benchmark-v2 | HazardNav v11 |
|---|---|---|
| Registered task | v1 and v2 | Adds `Isaac-USV-HazardNav-Direct-v3` |
| Native observation | v1: 39D; v2: 41D | 42D |
| Goal state | dot, cross, normalized distance | Same |
| Motion state | v2 adds reached latch and scalar speed magnitude | Body-frame surge, sway and yaw rate |
| Range sensor | 36 planar rays, 10-degree spacing | Same |
| Goal entry reward | None | 50-point clean entry bonus |
| Time-dependent goal reward | None | Up to 50 additional points from remaining time |
| Reverse handling | No explicit reverse penalty | Small negative-thrust action cost, scale 0.05 |
| Idle/swiftness cost | Scale 0.05, v2 only | Scale 0.25 when velocity is observable |
| PPO discount / GAE lambda | 0.99 / 0.95 | 0.999 / 0.99 |
| Checkpoint interval | 3200 trainer steps | 256 trainer steps |
| Boat rendering | Physics asset's original visual | Official blue BlueBoat render mesh attached; physics unchanged |
| SPL evaluator | Uses straight distance `D0 - goal_radius` | Uses `route_geodesic_length - goal_radius` |

## Observation layouts

Student v1:

```text
[goal_dot, goal_cross, distance_norm, ray_0, ..., ray_35] = 39D
```

Student v2:

```text
[goal_dot, goal_cross, distance_norm, reached, speed_norm,
 ray_0, ..., ray_35] = 41D
```

v11/v3:

```text
[goal_dot, goal_cross, distance_norm,
 surge_body, sway_body, yaw_rate_body,
 ray_0, ..., ray_35] = 42D
```

The extra kinematic components make inertia observable: the policy can tell
whether the boat is moving forward, sliding sideways, or still rotating. The
student v2 scalar speed cannot distinguish those cases.

## Reward differences

Both versions retain:

- normalized potential progress, scale 20;
- analytic ray-proximity cost;
- 25-point contact-entry penalty;
- 1-point contact-dwell penalty;
- the same success predicate and fixed 120-second episode.

v11 adds a one-time clean goal-entry reward:

```text
goal_bonus = 50 + 50 * remaining_episode_fraction
```

This addresses the earlier tendency to approach or orbit the goal-circle edge
without decisively entering. It also creates a strong incentive to arrive early.

v11 additionally charges negative thrust:

```text
reverse_cost = 0.05 * dt * relu(-thrust_action)^2
```

This term is too weak in practice. A roughly 20-second reverse approach costs
about one point while a fast goal entry can pay close to 100 points. The policy
therefore still finds stern-first navigation profitable.

## Physics and termination semantics

These are intentionally unchanged:

- BlueBoat bow axis: body `+X`;
- forward/reverse thrust limits: 80 N / 48 N;
- maximum yaw torque: 23 N m;
- the same hydrodynamic damping, buoyancy and restoring coefficients;
- reaching the goal does not terminate the simulator episode;
- collision does not immediately terminate the episode;
- a collision before first goal entry makes the completed episode unsuccessful;
- timeout at 120 seconds is the normal episode boundary.

The blue official model in v11 is render-only. Collision, mass, inertia,
buoyancy and reward calculations still use `blueboat_physics.usd`.

## Evaluation correction

The student branch's `scripts/eval_cross_champion.py` computes HazardNav SPL
with:

```text
shortest_path = D0 - goal_radius
```

However, the generator deliberately places a cylinder on the direct route and
the environment already stores the obstacle-aware shortest path as
`route_geodesic_length`. The straight-line expression systematically
underestimates the required route length.

The included `scripts/eval_hazard_nav.py` uses:

```text
shortest_path = route_geodesic_length - goal_radius
```

It also freezes curriculum difficulty for the whole evaluation and reports the
pre-goal reverse-action and reverse-velocity ratios.

## Practical conclusion

v11 improves goal entry, exposes the boat's planar motion state, saves dense
checkpoints, fixes SPL evaluation, and uses the requested blue visual model. It
does not solve forward-only navigation: its best fixed-level checkpoint matches
the strongest observed success rate but still travels mostly in reverse.
