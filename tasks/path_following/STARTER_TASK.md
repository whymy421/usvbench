# USV T1 Path Following

> **Vehicle:** realistic ROV (`ROV_rigged.usd`)  
> **Condition:** calm water (no waves or current)  
> **Gym id:** `Isaac-USV-PathFollow-Direct-v1`

## Task

Each 120 s episode generates a four-segment planar path from the environment
origin. Segment lengths are sampled uniformly from 12-20 m. The first segment
heading is uniform over `[0, 360)` degrees; each later heading changes by a value
sampled uniformly from `[-60, +60]` degrees. Waypoints are the cumulative,
environment-origin-relative segment endpoints.

The ROV starts at the environment origin, at rest at `z=-0.15`, with a uniformly
random heading. It must visit all four waypoint gates in order. There is no
corridor failure and no curriculum in v1.

The policy observation is `(dot, cross, dist_norm)` relative to the current
waypoint, with `dist_norm = distance / 20`. The action is
`(forward_thrust, yaw_torque)` in `[-1, 1]^2`. Entering the current 2.0 m gate
advances the observation target immediately to the next waypoint.

Use a bounded `Box([-1, 1])` action space with `clip_actions: True` and `initial_log_std: -1.0`, because an unbounded Gaussian mean can escape the +/-1 rail under environment-side clipping.

## Reward-free success predicate

Success depends only on ordered gate entry: the horizontal distance to each
current waypoint must become at most 2.0 m, in sequence, and success fires when
the fourth gate is reached. Success terminates the episode. Reaching the 120 s
time limit before the fourth gate is failure. Reward values never enter this
predicate.

## Reference reward

The included reward is a **reference baseline only; methods may use other
shaping**. V3 uses potential-based progress toward the waypoint that was active
at the start of the control step:

```text
20.0 * (previous_distance - current_distance)
    + 10.0 * (waypoint gate passed this step)
    + 100.0 * (terminal success this step)
```

Both distances are measured against the same step-start waypoint. After the
reward delta is computed, the previous-distance potential is refreshed against
the possibly new active waypoint. Thus passing a gate does not mix the old
target's previous distance with the new target's current distance. The dense
progress return is capped by route progress: orbiting or moving away and back
nets zero rather than creating an indefinitely farmable per-step stream.

## Reference reward history

V3 replaces the V1/V2 flow reward after the post-mortem showed the
`forward_speed * exp(alignment)` stream paid approximately 14k for orbiting near
a gate for the full 120 s, versus approximately 8.6k for finishing in 68 s.
Both PPO runs reached 100% success and four gates at roughly 3-4k steps, then
rationally abandoned completion while reward rose to approximately 14k because
success termination cut off the stream. The V3 potential-based term makes
progress farming-immune and adds a one-time 100-point terminal success bonus
while retaining the 10-point per-gate bonus.

This follows the docking V4/V6 design rule: success-terminated tasks need dense
terms whose total obtainable return is farming-capped, plus a dominant terminal
bonus that makes completing the task preferable to prolonging the episode.

## Metrics

Primary reporting is success rate (SR), success-weighted path length (SPL), and
time to success. For SPL, use the generated route length
`L = sum(segment_lengths)`, travelled planar `P = episode_path_length`, and
`SPL = mean(S * L / max(L, P))`. `PathFollowingEnvCfg.goal_radius = 2.0` exposes
the gate tolerance to evaluation code.

The environment exposes live per-environment `path_length`, `gates_passed`
(`torch.long`), and `xte_rms`, plus completed-episode snapshots
`episode_success`, `time_to_success` (`nan` on failure), `episode_path_length`,
`episode_gates_passed`, `episode_xte_rms`, and `episode_route_length`.
`gates_passed` and `xte_rms` are diagnostics, not success predicates. XTE is the
per-step perpendicular distance to the active segment centerline, RMS-aggregated
over the episode.

## Visualization

`PathFollowingEnvCfg.visual` controls the station-keeping-style flat water mesh
and orange 2.0 m annulus at each waypoint. Only env 0 is marked. Its four rings
are recreated when env 0 receives a new path, guarded by `try/except`; disabling
the marker removes all per-step marker cost. Visual settings do not affect task
physics, observations, reward, termination, or success.

## Install and train

Copy this folder into the Isaac Lab direct-task package:

```text
<IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/path_following/
```

Set `USVBENCH_ASSETS` to this repository's `assets` directory if the repository
is not at `~/usvbench`, then activate the Isaac Lab environment. Observation size
is fixed to 3; no `OBS_DIM` environment variable is needed.

```bash
python <IsaacLab>/scripts/reinforcement_learning/skrl/train_with_eval.py \
  --task=Isaac-USV-PathFollow-Direct-v1 \
  --num_envs=64 --headless --max_iterations=3000 --seed=42 \
  --video --video_interval 50000 --video_length 200
```

## Smoke test

From the USVBench repository:

```powershell
$env:USVBENCH_ASSETS = (Resolve-Path .\assets).Path
python .\tasks\path_following\smoke.py --headless
```

The smoke test creates 16 environments, checks the `(16, 3)` observation and
300 random-step reward range, validates path sampling bounds, then teleports env
0 to all four waypoints in order and requires terminal success only on gate four.
