# Waves on hazard_nav

Three task ids share one environment, one observation layout, and one action
space. They differ only in the sea state, so a checkpoint trained on any of
them runs on all three without adaptation.

| task id | sea state |
|---|---|
| `Isaac-USV-HazardNav-Direct-v3` | flat water |
| `Isaac-USV-HazardNav-Airy-Direct-v3` | regular wave, H = 0.12 m, T = 3.0 s |
| `Isaac-USV-HazardNav-Jonswap-Direct-v3` | irregular sea, Hs 0.06-0.18 m, Tp 2-4 s |

Train exactly as for v3, only the `--task` changes:

```bash
python scripts/reinforcement_learning/skrl/train.py --task=Isaac-USV-HazardNav-Jonswap-Direct-v3 --num_envs=64 --headless --max_iterations=3000 --seed=42
```

Evaluate the same way:

```bash
python scripts/eval_hazard_nav.py --task Isaac-USV-HazardNav-Jonswap-Direct-v3 --checkpoint <path>.pt --headless --seed 42 --eval_level 1 --episodes 64 --num_envs 64 --output_json result.json
```

## How waves reach the boat — plant v2

There are no wave-specific gains. Every wave effect comes out of hydrodynamic
coefficients that already existed for calm water:

**Lift, roll and pitch** come from buoyancy sampled at six stations spread
across the hull instead of one point at the origin. Each station carries a
sixth of the displaced volume and is submerged according to the water surface
directly beneath it, so a crest under one side lifts that side and the couple
follows. A beam sea rolls the boat and a head sea pitches it, both as a
consequence of hydrostatics rather than an authored torque.

**Horizontal forcing** comes from taking drag against the water rather than
the ground: `v_boat - u_orbital` instead of `v_boat`, with `u_orbital` the
Airy surface particle velocity. Heave damping likewise uses `v_z - eta_dot`.
In still water both wave terms are zero and the expressions collapse to the
calm ones exactly.

The policy cannot see the waves, it only feels them. Giving it wave
observations means appending channels to `tasks/_shared/obs_superset.py`
(append-only — read the header there first) and retraining; that is separate
work and deliberately not done here.

### Station geometry is hydrodynamically equivalent, not plan-form

The six offsets are solved so they reproduce the vehicle's calibrated
`restoring_stiffness_roll` and `restoring_stiffness_pitch`. For BlueBoat that
puts them at ±0.5572 m across and ±0.4843 m along, which does **not** match
the CAD hull spacing of 0.7214 m.

That difference is deliberate. Placing stations on the geometric hull
centrelines lumps each hull's buoyancy onto a line and discards the waterplane
inertia the hull's own width contributes — for BlueBoat that yields a roll
stiffness of 117 N·m/rad against the calibrated 280, and the boat becomes so
tender it capsizes into obstacles within 10 m. The equivalent radii keep the
vessel's real attitude dynamics; they are the same kind of fitted quantity as
`displaced_volume_m3` and `hull_height_m`, neither of which is a CAD integral
either. Recorded in `vehicles.py` alongside the measured `hull_length_m` and
`hull_spacing_m`, which are kept separate precisely because they are measured.

To go back to single-point sampling:

```bash
--task Isaac-USV-HazardNav-Jonswap-Direct-v3 env.wave.buoyancy_stations=1
```

Be aware what that means: buoyancy at one point cannot produce a couple no
matter how the surface tilts, so the boat heaves but never rolls or pitches.
It is the honest way back to plant v1 behaviour, not a cheaper approximation.

## Startup line

Every wave run prints what it is actually simulating:

```
[WAVE] plant v2 mode=jonswap eta_std=0.030 m peak=0.119 m hull=0.376 m peak/hull=0.32 | 6 buoyancy stations, no wave gains
```

Watch `peak/hull`. Past about 0.5 the hull leaves the water or submerges
entirely, the submerged fraction clamps, and the sea state stops being
something the boat can respond to. Do not copy sea states from `rov_calm_nav`
or `boat_calm_nav`: the ROV displaces 0.5 m³ against BlueBoat's 0.0346 m³, and
its settings drive `peak/hull` past 2.

## Sea state parameters

Every field lives on `wave` in `hazard_nav_env_cfg.py` and overrides from the
command line:

```bash
--task Isaac-USV-HazardNav-Jonswap-Direct-v3 env.wave.hs_max_m=0.25 env.wave.direction_deg=90
```

`direction_deg` defaults to `None`, randomising the heading per environment.
Pin it to a number for a controlled beam-sea or following-sea sweep. `mode`
accepts `calm`, `airy`, `jonswap`; an unknown value raises rather than quietly
falling back to calm.

The certification protocol defaults to `Hs=U[0.06, 0.18] m`, `Tp=U[2, 4] s`,
fixed `gamma=3.3`, `f_min_hz=0.10`, `f_max_hz=1.60`, `spread_deg=30` (uniform
`±15°` around the mean direction), and `n_components=30`. `gamma=3.3` is the
agreed protocol default; exploratory runs may explicitly choose another range.
Certification should use `sampling_mode="levels"` with an explicitly recorded
frozen Hs/Tp grid and `gamma_levels=(3.3,)`. The
training/interpolation/extrapolation `Hs×Tp` levels remain a joint decision and
are not silently frozen here.

For paired evaluation, wave randomness is stateless:
`f(eval_seed, environment_index, episode_index, stream_index)`. The first reset
of each environment uses episode index 0 and every later reset increments it.
The same protocol also seeds the hazard layout, so changing the order of a
partial reset cannot change another environment's sea state or obstacle route.
The process-wide Torch RNG is not used for wave phases, directions, or sea-state
draws. Each reset also anchors that environment's wave clock at local `t=0`, so
the same episode does not inherit a different phase merely because the previous
episode ended at a different global simulation time.

`Hs=0` is a strict no-op. A calm mode, an Airy height of zero, or a JONSWAP
configuration with `hs_max_m=0` is constructed as `CalmWater`, and the resulting
forces, torques, observations, and reset path are bit-identical to flat water.

The JONSWAP spectrum is normalised numerically over its own discrete band, so
`4√m₀ = Hs` holds exactly for any Tp, gamma, or frequency band. The usual
`(1 − 0.287 ln γ)` closed form is derived for a continuous unbounded spectrum
and drifts about 8% low at Tp = 4 s with gamma = 5; that value is only relevant
to an explicitly requested exploratory range, not the certification default.

The 30 frequencies use a fixed, small nonuniform jitter around the nominal
`df=(f_max-f_min)/30=0.05 Hz` grid. A uniform 30-point sum would repeat every
20 s, shorter than the 120 s episode; the jitter removes that exact repeat.
Each component uses its own bin width `Delta_f_n` in both the spectrum moment and
amplitude, preserving the Hs normalisation.

The valid-domain guard enforces `Hs/lambda_p <= 0.05`, with
`lambda_p=g Tp^2/(2 pi)`. This remains a deep-water linear model; finite-depth
corrections are not included and require a separate validation before use.

## Recording video

`play.py` initialises wandb, and if stale `wandb-core` processes hold the
service port it hangs silently at ~0% CPU until a 30-second timeout kills it.
Disable wandb when recording:

```powershell
$env:WANDB_MODE = "disabled"
python scripts\reinforcement_learning\skrl\play.py --task=Isaac-USV-HazardNav-Jonswap-Direct-v3 --num_envs=1 --seed=42 --headless --video --video_length=1800 --checkpoint=<path>.pt
```

`eval_hazard_nav.py` disables wandb itself, so evaluation is unaffected.

Wave runs of four environments or fewer get a deforming water surface coloured
by height (turbo, matching the E7 line's rendering), so the sea is visible and
not just felt. Larger runs keep the flat plane — rebuilding thousands of
vertices per frame is wasted during training.

## Known simplifications

Written down so nobody has to rediscover them:

- No added mass. A hull's heave added mass is of the order of its displaced
  mass, so natural periods are shifted relative to a real vessel.
- No second-order drift force. First-order orbital forcing only; drift scales
  with Hs² and needs a coefficient that would be another authored number.
- No diffraction or radiation memory. Froude-Krylov forcing only.
- Six stations resolve waves down to roughly their spacing. Much shorter waves
  are averaged rather than properly integrated over the hull.
