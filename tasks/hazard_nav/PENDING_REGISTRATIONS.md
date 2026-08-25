# PENDING REGISTRATIONS (Task C, 2026-08-23)

Coordination note: Task C was instructed NOT to touch the drift-test pin
(one agent per night owns `tasks/_shared/test_registry_drift.py`). The
NATIVE_LAYOUTS entry for the id below is ALREADY ADDED to
`tasks/_shared/obs_superset.py`; the gym registration itself is deferred to
this file. Measured side effects while the registration is pending:

- `test_registry_drift.py`: all three checks still PASS at pin (61, 2).
  The orphan check does NOT flag the pending entry -- docking's f-string
  registration (`id=f"Isaac-USV-{_suffix}-Direct-v1"`) compiles to the
  pattern `Isaac-USV-.+-Direct-v1`, which shadows any orphan of this id
  shape. That is a pre-existing blind spot in the orphan test, worth its
  own fix by whoever owns the drift file.
- `tasks/_shared/test_obs_superset.py`: already failing BEFORE Task C (its
  expected-id set lacks the five SuiteS ids and
  Isaac-USV-HazardBandFortWayTax-Direct-v1); the pending entry adds a 7th
  id to the same red set-equality assertion. When applying the
  registration, also add to that file's EXPECTED_V2_DIMS:
  `"Isaac-USV-StationKeep-BlueBoat-RampCurrent-Direct-v1": 3,  # station_keeping_env_cfg.py:288`

## 1. Register: Isaac-USV-StationKeep-BlueBoat-RampCurrent-Direct-v1

Append verbatim to `tasks/station_keeping/__init__.py`:

```python
# C2 x ramping current: speed climbs linearly within each episode
# (current_ramp_start_mps -> current_ramp_end_mps, direction fixed per
# episode). Append-only: certified Current ids untouched.
gym.register(
    id="Isaac-USV-StationKeep-BlueBoat-RampCurrent-Direct-v1",
    entry_point=f"{__name__}.station_keeping_env:StationKeepingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.station_keeping_env_cfg:"
            "StationKeepingBlueBoatRampCurrentEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

Everything the id needs is already in the tree:

- `tasks/station_keeping/station_keeping_env_cfg.py` --
  `StationKeepingBlueBoatRampCurrentEnvCfg` (subclass of the certified
  Current cfg; `current_ramp_start_mps=0.5`, `current_ramp_end_mps=3.0`).
- `tasks/station_keeping/current_ramp.py` -- Isaac-free modulation law,
  CPU-tested by `tasks/station_keeping/test_current_ramp.py`.
- `tasks/station_keeping/station_keeping_env.py` -- guarded per-step
  injection in `_compute_current_forces` (inert unless the cfg declares
  BOTH ramp fields).
- `tasks/_shared/obs_superset.py` -- NATIVE_LAYOUTS entry (`list(_NAV)`,
  same 3-D nav observation as the other non-Kin station ids).

### Pin arithmetic (`tasks/_shared/test_registry_drift.py`)

One new literal id, no new f-string pattern: bump the literal count by +1.
At the time of writing the pin reads `(61, 2)` (Task A already moved it
from `(56, 2)` tonight), so applying this block alone makes it `(62, 2)`.
If other registrations land first, apply `(+1, +0)` to whatever the pin
then reads.

## 2. Item 1 (line-avoid tier 0): NO new gym id, one wiring decision flagged

The tier-0 extension is geometry-only
(`tasks/path_hazard/path_hazard_geometry.py`: `sample_layout` now accepts
`blockers_override=0`) and needs no registration and no pin change.

Owner decision needed before a PathHazard tier-0 gym cfg or probe run can
exist: the cfg field `on_line_blockers_override` uses `0` as its "no
override" sentinel, and `path_hazard_env.py` maps it with
`getattr(cfg, "on_line_blockers_override", 0) or None` -- so a cfg value of
0 resolves to the frozen three blockers, NOT to tier 0. Wiring tier 0
through the env needs either a new cfg field (append-only safe) or a new
sentinel for "no override" on a NEW cfg only; existing cfg semantics must
not move.
