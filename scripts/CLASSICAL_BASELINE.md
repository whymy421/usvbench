# Classical baseline

`classical_baseline.py` is USVBench's zero-learning reference controller. It
uses the same deterministic evaluation protocols as learned policies, but its
control stack is conventional marine line-of-sight guidance plus heading PID:

- bearing error comes from `atan2(cross, dot)`;
- PID maps wrapped heading error to yaw torque;
- thrust is reduced near the goal or hold zone, is zero inside it, and uses a
  small creep command while the vehicle turns;
- path following instead treats each gate as a pass-through target: it steers
  toward the environment's current waypoint, keeps full thrust when aligned,
  and smoothly throttles down as heading error grows;
- docking uses a per-environment three-phase state machine: LOS approach,
  dock-heading alignment plus braking, then a zero-thrust hold with yaw PID.

The point-navigation, station-keeping, and path-following controller reads only
`obs[0:3]` (`dot`, `cross`, and normalized distance). Docking additionally reads
the environment's `dock_point`, `dock_heading`, boat pose, and planar velocity
to control alignment and braking. The boat bow is body `-X`, opposite to the
ROV convention; docking uses the environment helper when available and
otherwise computes `R(quat) @ (-1, 0, 0)` explicitly.

## Why every task publishes this number

Every benchmark task should publish the classical baseline beside learned-policy
results. It pins a practical floor/ceiling and exposes tasks whose nominal
learning challenge is already solved by standard feedback control: a gate a PID
can pass is not a learning gate. Use `--protocol throughput` for the fixed-step
targets-per-episode evaluation and `--protocol mission` for completed-episode
SR/SPL evaluation.

```powershell
python scripts/classical_baseline.py --task Isaac-My-First-Task-Calm-Direct-v1 `
  --protocol throughput --num_envs 64 --eval_steps 6000 --headless

python scripts/classical_baseline.py --task Isaac-USV-PathFollow-Direct-v1 `
  --protocol mission --num_envs 64 --episodes 128 --headless

python scripts/classical_baseline.py --task Isaac-USV-StationKeep-Direct-v1 `
  --protocol mission --num_envs 64 --episodes 128 --headless

python scripts/classical_baseline.py --task Isaac-USV-Dock-Direct-v1 `
  --protocol mission --num_envs 64 --episodes 128 --headless
```

Point-navigation distance observations default to a 30 m scale. Station-keeping
defaults to 15 m. Path following defaults to its 20 m maximum segment length,
and docking records the environment's 25 m observation scale for
reproducibility. The task/mode dispatch is inferred from the environment;
override a nonstandard observation normalization with `--dist-scale`.

Path-following mission output reports SR, time to success, route-length SPL,
and the completed-episode gate-count distribution (`0` through `4`). The
environment advances the active waypoint automatically when the ROV enters each
ordered 2.0 m gate, so the controller does not brake or hold at intermediate
gates.

## Docking mode

`Isaac-USV-Dock-Direct-v1` uses the realistic boat actuator convention:
`action[0]` is clipped to `[-1, 1]` and mapped environment-side to 500 N
forward or 200 N reverse, while `action[1]` commands yaw torque. The controller
uses these phases independently for every vectorized environment:

1. **APPROACH:** steer the bow toward `dock_point`. Thrust reaches 1.0 outside
   the 6 m braking radius and tapers linearly to zero at 2.0 m.
2. **ALIGN+BRAKE:** after entering 2.2 m, steer the bow toward `dock_heading`,
   target zero surge speed, and use a small radial correction to remain inside
   the docking zone. Reverse thrust is limited to a gentle corrective command.
3. **HOLD:** once dock-heading error is within 10 degrees and planar speed is
   below 0.2 m/s, command zero thrust and yaw PID only. If that condition
   breaks, return to ALIGN+BRAKE so the environment's continuous 5 s
   `hold_timer` can restart.

The environment defines success as horizontal distance at most 2.5 m, bow
alignment within 15 degrees, and planar speed at most 0.3 m/s, continuously for
5 s. Mission output uses the completed-episode `episode_success`,
`time_to_success`, and `episode_path_length` values and reports SR, mean and
median time to success, SPL, timeout failures, and other failures in the
standard summary and CSV formats.

## Gain tuning

The defaults (`kp=2.0`, `ki=0.0`, `kd=0.5`, slow radius 5 m for the shared LOS
controller and 6 m for docking) were hand-set. Gain tuning is method-side
freedom: classical methods may tune these values just as learning methods tune
their algorithms. Always publish the gains, slow radius, distance scale, task
seed, and tuning procedure with the result. Do not tune on the reported
evaluation episodes.
