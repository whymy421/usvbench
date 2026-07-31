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

## How waves reach the boat

Waves change exactly one thing: the water surface stops being the plane
`water_surface_z` and becomes `water_surface_z + eta(t, x, y)`. Buoyancy —
and therefore heave — then follows from the hydrostatics that were already
there, with no added force terms. The surface slope separately tilts the
restoring equilibrium toward the wave normal, which is what produces roll and
pitch. Set `wave.slope_torque_scale = 0` to keep heave and remove that tilt.

Because the observation layout is untouched, the policy cannot see the waves.
It feels them. Giving the policy wave observations means appending channels to
`tasks/_shared/obs_superset.py` (append-only — read the header there first) and
retraining; that is a separate piece of work.

## Recording video

`play.py` initialises wandb, and if stale `wandb-core` processes are still
holding the service port it hangs — no error, no output, just a process
sitting at ~0% CPU until it eventually dies on a 30-second token timeout.
Disable wandb when recording:

```powershell
$env:WANDB_MODE = "disabled"
python scripts\reinforcement_learning\skrl\play.py --task=Isaac-USV-HazardNav-Jonswap-Direct-v3 --num_envs=1 --seed=42 --headless --video --video_length=900 --checkpoint=<path>.pt
```

`eval_hazard_nav.py` turns wandb off itself, so evaluation is unaffected; this
only bites on `play.py`. Sample clips of all three sea states are in
`artifacts/hazard_nav_wave_sweep/videos/`.

## Sea states are per-vehicle

BlueBoat's hull is 0.376 m tall, so the surface only has to move +-0.188 m for
the boat to leave the water or submerge completely. Past that point the
submerged fraction clamps, the boat alternates between free-fall and being
launched by buoyancy, and every wave model starts scoring the same — it looks
like a result but it is a broken setup.

The environment prints a check at startup and warns when this happens:

```
[WAVE] mode=jonswap eta_std=0.030 m peak=0.119 m hull_half=0.188 m saturated=0.0%
```

Keep `saturated` at 0%. If you raise the sea state or move to a different
vessel, watch this line. Do not copy the sea states from `rov_calm_nav` or
`boat_calm_nav` — those vessels are far larger (the ROV displaces 20 m^3
against BlueBoat's 0.0346 m^3) and their settings saturate BlueBoat 25-46% of
the time.

## Before you tune anything, read this

`artifacts/hazard_nav_wave_sweep/FINDINGS.md` records a zero-shot sweep of the
calm-trained v11 champion across all three ids. Its success rate drops from
21/64 to 1/64 under either wave model, and the drop is a switch rather than a
dose response: scaling the slope torque down to 5% scores identically to full
strength. Whether that is the policy being fragile or the coupling having a
fault was not settled.

This matters to you because it may also show up in training: if a run refuses
to learn under waves, that document tells you what has already been ruled out
(wave clock, hull physics, buoyancy saturation, the restoring plumbing) so you
do not repeat the search. The single experiment that would separate the two
explanations is written down at the end of it.

## Changing the sea state

Every field is on `wave` in `hazard_nav_env_cfg.py` and can be overridden from
the command line:

```bash
--task Isaac-USV-HazardNav-Jonswap-Direct-v3 env.wave.hs_max_m=0.25 env.wave.direction_deg=90
```

`direction_deg` is `None` by default, which randomizes the heading per
environment. Pin it to a number for a controlled beam-sea or following-sea
sweep. `mode` accepts `calm`, `airy`, `jonswap`; an unknown value raises rather
than silently falling back to calm.
