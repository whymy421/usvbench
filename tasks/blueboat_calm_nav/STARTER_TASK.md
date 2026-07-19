# BlueBoat Calm-Water Point Navigation

> **Vessel:** real Blue Robotics BlueBoat, two-battery condition  
> **Asset:** `assets/blueboat_physics.usd` (`/World/BlueBoat`)  
> **Condition:** calm water (no waves, no current)  
> **Gym id:** `Isaac-USV-BlueBoat-Calm-Direct-v1`  
> **Environment:** `BlueBoatCalmNavEnv`

This is the same continuous point-navigation benchmark as `boat_calm_nav`, with
the real BlueBoat asset and BlueBoat-calibrated dynamics. It coexists with the
WAM-V-style boat task; it does not replace it. Keeping both makes the pair a
controlled hull-swap ablation: task logic and target refresh stay fixed while
the hull, mass properties, actuator envelope, and hydrodynamics change.

## Task contract

| Item | Value | Source / decision |
|---|---:|---|
| Observation | source task; `OBS_DIM` selectable (3 by default) | copied from `boat_calm_nav` |
| Action | 2D `(surge thrust, yaw torque)`, hard-clipped to `[-1, 1]` | copied action contract; clipping is in `_pre_physics_step` |
| Episode | 120 s | copied from `boat_calm_nav` |
| Goal radius | 3.0 m | copied from `boat_calm_nav` for the hull-swap ablation |
| Target distance | 10--30 m, refreshed continuously on reach | copied exactly from `boat_calm_nav`; BlueBoat 3.0 m/s vs source calibrated 2.0 m/s does not require expanding an already suitable range |
| PPO experiment | `blueboat_calm_v1` | task-specific run namespace |

The PPO critic retains `value.clip_actions: false`; actions are an actor/env
concept and clipping critic values would change value learning.

## Parameter derivation

| Parameter | Value | Source / derivation |
|---|---:|---|
| Vehicle mass | 17.3 kg | official Blue Robotics BlueBoat Technical Details, two-battery condition; authored in the USD via MassAPI |
| COM and inertia | USD-authored | official BlueBoat CAD-derived asset; intentionally not overridden by task configuration |
| Hull form | displacement catamaran | official BlueBoat product/CAD geometry |
| Hull spacing | 0.7214 m | official BlueBoat CAD mapped bounds |
| Bow/body axis | `+X` | committed USD mapped bounds show length along X and bow on +X; both `forward_vec` and surge-force sign use this convention |
| Total forward thrust | 80.0 N | two official M200 thrusters, approximately 40 N each |
| Total reverse thrust | 48.0 N | two M200 reverse limits at approximately 60% of forward due to propeller asymmetry |
| Maximum yaw torque | 23.0 N·m | CAD half-spacing lever arm: `0.3607 * (40.2 + 24) = 23.16 N·m`, rounded |
| Surge drag | `-(9.1 + 5.9|v|)v` N | two-anchor calibration below |
| Sway drag | `-(27 + 18|v|)v` N | approximately 3x surge for catamaran lateral bluffness; estimate pending system identification |
| Heave damping | 200 N·s/m | CAD-scale settling estimate; pending heave system identification |
| Yaw drag | `-(4 + 6|r|)r` N·m | plausible-order estimate balancing about 23 N·m near 1.6 rad/s; pending yaw system identification |
| Roll stiffness | 280 N·m/rad | CAD hydrostatics: `rho*g*V_disp*GM_T`, with `V_disp=0.0173 m^3`, `GM_T=1.64 m` |
| Pitch stiffness | 141 N·m/rad | CAD hydrostatics: `rho*g*V_disp*GM_L`, with `V_disp=0.0173 m^3`, `GM_L=0.83 m` |
| Roll/pitch rate damping | 470 N·m·s/rad | source WAM-V task used 2000 at stiffness 5000; `2000*sqrt(280/5000)=473`, rounded to preserve approximately constant damping ratio; pending system identification |
| Buoyancy model volume | 0.0346 m³ | source submersion model equilibrates at 50%: `2*17.3/1000` |
| Buoyancy model height | 0.376 m | official BlueBoat CAD mapped height |
| Water surface | `z=0` m | benchmark convention |
| Buoyancy-center offset | -0.05 m | below-COM distance estimated from hull geometry; pending immersion/system identification |

The restoring terms use the shared implementation in
`tasks/docking/restoring.py`, including separate roll and pitch stiffnesses.

## Two-anchor surge-drag calibration

Assume `F_d = -(a + b|v|)v`.

1. The official BlueBoat performance data gives about 3.0 m/s terminal speed
   under 80.4 N total forward thrust, so `3a + 9b = 80.4`.
2. At 1.0 m/s cruise, 30 W electrical power with an assumed 50% propulsive
   efficiency gives about 15 W useful power. Since `P = Fv`, this implies
   about 15 N drag, so `a + b = 15`.

Solving gives `a = 9.1 N·s/m` and `b = 5.9 N·s²/m²`. The 50% propulsive
efficiency is an explicit uncertainty, not a measured BlueBoat efficiency;
replace this anchor when tow-test or electrical/shaft-power system-identification
data becomes available.

## Run

Use the same environment-variable reward/observation settings as the source
task when making a hull-swap comparison. For example:

```bash
USVBENCH_ASSETS=<repo>/assets OBS_DIM=9 OBS_EXTENDED=1 \
REWARD_VARIANT=V23 SPEED_COUPLE=1 REACH_BONUS=50.0 \
python <IsaacLab>/scripts/reinforcement_learning/skrl/train.py \
  --task=Isaac-USV-BlueBoat-Calm-Direct-v1 --num_envs=64 --headless
```

The asset is the authority for mass, center of mass, and inertia. Do not add a
spawn-time mass-properties block to the task configuration.
