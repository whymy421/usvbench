# USV Station Keeping - Boat Replication

> **Vehicle:** realistic boat hull (`boat_physics.usdc`)  
> **Condition:** calm water (no waves or current)  
> **Gym id:** `Isaac-USV-StationKeep-Boat-Direct-v1`

This is the vehicle-replication of `tasks/station_keeping`: the capability test,
spawn distribution, reward-free hold predicate, and evaluation tensors are held
constant while the ROV is replaced by the realistic boat. The hold radius is
3.0 m rather than 2.0 m, following the repository's hull-scale convention for
the 5.5 m boat.

Each 120 s episode starts the boat at rest, 5-15 m from its environment origin,
with a random heading. Success requires horizontal distance <= 3.0 m for 60
consecutive seconds. Leaving the zone resets the timer without failing; success
terminates the episode and timeout is failure.

The fixed 3D observation is `(dot, cross, dist_norm)` toward the origin, using
body `-X` as forward and `dist_norm = distance / 15`. Actions are forward thrust
and yaw torque in `[-1, 1]^2`.

The included reward is reference-only and is not part of success:

```text
-distance / 30 + (1.0 if distance <= 3.0 else 0.0)
```

Smoke test from the repository root:

```powershell
$env:USVBENCH_ASSETS = (Resolve-Path .\assets).Path
python .\tasks\station_keeping_boat\smoke.py --headless
```
