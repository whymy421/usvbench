"""Measurement groups consumed by each observation layout (D2 review item 7, E3).

Keyed by (family, observation contract). A perception dose may only target a
measurement the layout actually consumes; anything else is rejected at
construction (D2 REVIEW.md F7: a velocity dose on native HazardNav v1 was
accepted and certified although the v1 observation has no velocity channel).

Groups (physical measurements, degraded BEFORE feature construction):
  pos  the task-relative position vector (goal / hold point / waypoint minus
       own position), world axes, metres. A position-estimate error e enters
       as -e (the estimated own position is truth + e).
  vel  body-frame planar kinematics (surge m/s, sway m/s, yaw rate rad/s).
       ``vel_channels`` says which of the three the layout consumes: a
       speed-only layout (HazardNav v2 native, every 51-D superset) consumes
       surge and sway only, so a yaw-rate dose there is rejected too.
  ray  range returns (m), clamped to [0, max range] (also when pooled).
Temporal axes (latency, outage) act on every group the layout consumes.

The observation contract of an env cfg is ``cfg.observation_contract`` when
the task contract declares it (round F / E5), otherwise it is derived from the
family's existing layout switches (``derive_contract``). Evidence (J-base
file:line) is in DESIGN_E3.md section 5. ``hooks_wired`` says whether the
family's _get_observations already routes the groups through the condition
layer; families wired in round F must flip it together with their hook
edits. A perception dose on an unwired family is rejected (fail closed).

F0 (CONTEXT_R4 D5, D14): journal contracts (``journal=True`` rows, named
``journal_*``). Every journal task carries the 36-ray block, so every journal
contract consumes ``ray`` and range noise is admitted on all of them
(station keeping included). Journal contracts degrade the OWN-STATE
ESTIMATE, not a task-relative vector: the env calls
``estimate_position/estimate_velocity/estimate_rays`` exactly once per
control step for every group in ``estimate_groups`` (pos, vel, ray; the
same estimates feed PPO, MPPI and the classical controllers). The temporal
axes (latency, outage) act on all three estimate groups, so every controller
sees one consistent delayed state; the per-group noise axes are admitted
only on the groups (and velocity channels) the PPO layout consumes (the D2
item 7 rule is kept). A journal cfg without an explicit
``observation_contract`` gets ``JOURNAL_TASK_CONTRACTS[cfg.journal_task]``.

Hooks flag for journal rows: the table ships ``hooks_wired=False``; a family
flips it in ITS OWN patch by setting ``journal_observation_hooks_wired = True``
on its journal cfgs once its _get_observations calls the estimate API
(``effective_contract``). The flag can only unlock ``journal=True`` rows; the
legacy rows keep their table value.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

FULL = ("surge", "sway", "yaw_rate")
PLANAR = ("surge", "sway")


@dataclass(frozen=True)
class ObservationContract:
    family: str
    contract: str
    groups: tuple
    vel_channels: tuple
    sea_state_observed: bool
    hooks_wired: bool
    note: str
    journal: bool = False                 # F0: own-state estimate contract (D5)
    estimate_groups: tuple = ()           # F0: groups estimated once per control step


ESTIMATE_GROUPS = ("pos", "vel", "ray")
JOURNAL_RAY_BLOCK = 36      # journal_cfg_v2.RAY_SENSOR_RAY_COUNT (D14, E5)
HOOKS_FLAG = "journal_observation_hooks_wired"


def _c(family, contract, groups, note, *, vel=(), sea=False, wired=True, journal=False):
    if ("vel" in groups) != bool(vel):
        raise ValueError("vel group and vel channels must be declared together")
    estimates = ESTIMATE_GROUPS if journal else ()
    return (family, contract), ObservationContract(family, contract, tuple(groups), tuple(vel), sea, wired,
                                                   note, journal, estimates)


def _hazard_rows():
    rows = []
    for base in ("v1", "v2", "v3"):
        for pooled in (False, True):
            for superset in (None, 1, 2):
                if superset == 2:
                    vel = FULL          # superset v2 kinematics slice from the vel group
                elif superset == 1:
                    vel = PLANAR        # 51-D superset: speed_norm slot only
                else:
                    vel = {"v1": (), "v2": PLANAR, "v3": FULL}[base]
                name = base + ("_pooled" if pooled else "") + ("" if superset is None else f"_superset{superset}")
                groups = ("pos", "vel", "ray") if vel else ("pos", "ray")
                rows.append(_c("hazard_nav", name, groups,
                               f"HazardNav {base} observation{' (pooled ranges)' if pooled else ''}"
                               f"{'' if superset is None else f', superset v{superset}'}", vel=vel))
    return rows


OBSERVATION_CONTRACTS = dict([
    # station keeping (station_keeping_env.py:_get_observations)
    _c("station_keeping", "native3", ("pos",), "dot, cross, distance to hold point (3 obs)"),
    _c("station_keeping", "kinematic6", ("pos", "vel"), "native3 + body surge/sway/yaw rate", vel=FULL),
    _c("station_keeping", "sea_state9", ("pos", "vel"), "kinematic + observed sea state", vel=FULL, sea=True),
    _c("station_keeping", "sea_state6", ("pos",), "native3 + observed sea state", sea=True),
    *_hazard_rows(),
    # families wired in round F (declared from obs_superset.py native layouts)
    _c("blueboat_calm_nav", "native3", ("pos",), "reach: nav only", wired=False),
    _c("path_following", "native7", ("pos",), "path: nav + lookahead (position-derived)", wired=False),
    _c("docking", "native6", ("pos", "vel"), "nav + dock (speed channel)", vel=PLANAR, wired=False),
    _c("docking", "kinematic9", ("pos", "vel"), "docking + body kinematics", vel=FULL, wired=False),
    _c("docking", "wall12", ("pos", "vel", "ray"), "docking + wall rays", vel=PLANAR, wired=False),
    _c("path_hazard", "native43", ("pos", "ray"), "path + 36 ranges", wired=False),
    _c("path_hazard", "kinematic46", ("pos", "vel", "ray"), "path + kinematics + ranges", vel=FULL, wired=False),
    _c("harbor_mission", "native45", ("pos", "ray"), "nav + ranges + phase + dock", wired=False),
    _c("harbor_mission", "kinematic48", ("pos", "vel", "ray"), "harbour + kinematics", vel=FULL, wired=False),
    # F0 journal contracts (CONTEXT_R4 D5/D14): own-state estimates + the 36-ray block on EVERY task.
    _c("hazard_nav", "journal_v3_ray36", ("pos", "vel", "ray"),
       "journal reach/scatter/ring/fortress/crossing/iceberg/basin: HazardNav v3 layout (goal vector from "
       "the position estimate, body kinematics, 36 ranges)", vel=FULL, wired=False, journal=True),
    _c("path_following", "journal_nav7_ray36", ("pos", "ray"),
       "journal path following: native7 (goal/cross-track/lookahead from the position estimate) + 36 "
       "ranges; no velocity channel", wired=False, journal=True),
    _c("station_keeping", "journal_kin6_ray36", ("pos", "vel", "ray"),
       "journal station keeping: kinematic6 (hold vector from the position estimate, body rates) + 36 "
       "ranges", vel=FULL, wired=False, journal=True),
    _c("docking", "journal_dock6_ray36", ("pos", "vel", "ray"),
       "journal docking: native6 (berth pose from the position estimate, speed channel) + 36 ranges",
       vel=PLANAR, wired=False, journal=True),
    _c("path_hazard", "journal_kin46_ray36", ("pos", "vel", "ray"),
       "journal path with hazards: kinematic46 (path from the position estimate + kinematics + 36 "
       "ranges)", vel=FULL, wired=False, journal=True),
    _c("harbor_mission", "journal_kin48_ray36", ("pos", "vel", "ray"),
       "journal harbour mission: kinematic48 (phase targets from the position estimate + kinematics + "
       "36 ranges)", vel=FULL, wired=False, journal=True),
])

# journal task (journal_cfg_v2.TASKS) -> (carrier family, journal contract key)
JOURNAL_TASK_CONTRACTS = {
    "reach": ("hazard_nav", "journal_v3_ray36"),
    "path_following": ("path_following", "journal_nav7_ray36"),
    "station_keeping": ("station_keeping", "journal_kin6_ray36"),
    "docking": ("docking", "journal_dock6_ray36"),
    "path_hazard": ("path_hazard", "journal_kin46_ray36"),
    "scatter": ("hazard_nav", "journal_v3_ray36"),
    "ring": ("hazard_nav", "journal_v3_ray36"),
    "fortress": ("hazard_nav", "journal_v3_ray36"),
    "forced_crossing": ("hazard_nav", "journal_v3_ray36"),
    "iceberg": ("hazard_nav", "journal_v3_ray36"),
    "moving_basin": ("hazard_nav", "journal_v3_ray36"),
    "harbour_mission": ("harbor_mission", "journal_kin48_ray36"),
}


def derive_contract(family: str, env_cfg) -> str:
    """Observation contract key of an env cfg (explicit declaration wins).

    F0: a journal cfg (``journal_contract_version == 2`` with a
    ``journal_task``) without a declaration gets its journal contract.
    """
    declared = getattr(env_cfg, "observation_contract", None)
    if isinstance(declared, str) and declared:
        return declared
    task = getattr(env_cfg, "journal_task", None)
    if task is not None and getattr(env_cfg, "journal_contract_version", None) == 2:
        if task not in JOURNAL_TASK_CONTRACTS:
            raise ValueError(f"unknown journal task {task!r}")
        carrier, contract = JOURNAL_TASK_CONTRACTS[task]
        if carrier != family:
            raise ValueError(f"journal task {task!r} is carried by family {carrier!r}, not {family!r}")
        return contract
    if family == "station_keeping":
        kinematic = bool(getattr(env_cfg, "obs_kinematic", False))
        if getattr(env_cfg, "obs_sea_state", False):
            return "sea_state9" if kinematic else "sea_state6"
        return "kinematic6" if kinematic else "native3"
    if family == "hazard_nav":
        if getattr(env_cfg, "obs_v3", False):
            base = "v3"
        elif getattr(env_cfg, "obs_v2", False):
            base = "v2"
        else:
            base = "v1"
        name = base + ("_pooled" if getattr(env_cfg, "obs_feasibility", False) else "")
        if getattr(env_cfg, "emit_superset_obs", False):
            name += "_superset%d" % int(getattr(env_cfg, "superset_version", 1))
        return name
    raise ValueError(f"family {family!r} must declare cfg.observation_contract")


def lookup(family: str, contract: str) -> ObservationContract:
    try:
        return OBSERVATION_CONTRACTS[(family, contract)]
    except KeyError:
        raise ValueError(f"observation contract {contract!r} of family {family!r} has no declared "
                         "measurement groups; add it to observation_groups.py") from None


def effective_contract(contract: ObservationContract, env_cfg=None) -> ObservationContract:
    """The contract with the family's hooks flag applied (journal rows only)."""
    if contract.journal and not contract.hooks_wired and env_cfg is not None:
        flag = getattr(env_cfg, HOOKS_FLAG, False)
        if flag is True:
            return replace(contract, hooks_wired=True)
        if flag not in (False, None):
            raise ValueError(HOOKS_FLAG + " must be a bool")
    return contract


def journal_contract_for(task: str) -> ObservationContract:
    """Table entry of a journal task's default contract."""
    family, key = JOURNAL_TASK_CONTRACTS[task]
    return lookup(family, key)


def perception_group_doses(p) -> dict:
    """Group -> active axis names of a schema-3 perception config."""
    doses = {"pos": [], "vel": [], "ray": [], "*": []}
    if p.pos_noise_sigma_m > 0:
        doses["pos"].append("pos_noise_sigma_m")
    if p.pos_bias_m > 0:
        doses["pos"].append("pos_bias_m")
    if p.vel_noise_sigma_mps > 0:
        doses["vel"].append("vel_noise_sigma_mps")
    if p.yaw_rate_noise_sigma_rad_s > 0:
        doses["vel"].append("yaw_rate_noise_sigma_rad_s")
    if p.ray_noise_sigma_m > 0:
        doses["ray"].append("ray_noise_sigma_m")
    if p.latency_s > 0:
        doses["*"].append("latency_s")
    if p.outage_s > 0:
        doses["*"].append("outage_s")
    return doses


def admit_perception(contract: ObservationContract, perception) -> None:
    """Reject a nonzero perception dose that the layout cannot receive."""
    if not perception.enabled:
        return
    doses = perception_group_doses(perception)
    active = [axis for axes in doses.values() for axis in axes]
    if not active:
        return
    if not contract.hooks_wired:
        raise ValueError(f"family {contract.family!r} observation hooks are not wired to the "
                         f"condition layer yet (round F); perception doses {active} rejected")
    for group in ("pos", "vel", "ray"):
        if doses[group] and group not in contract.groups:
            raise ValueError(f"perception dose {doses[group]} targets group {group!r}, which the "
                             f"{contract.family}/{contract.contract} observation does not contain "
                             f"(groups {contract.groups})")
    if perception.vel_noise_sigma_mps > 0 and not set(PLANAR) <= set(contract.vel_channels):
        raise ValueError("velocity noise needs a layout that consumes surge/sway")
    if perception.yaw_rate_noise_sigma_rad_s > 0 and "yaw_rate" not in contract.vel_channels:
        raise ValueError(f"yaw-rate noise targets a channel the {contract.family}/{contract.contract} "
                         f"observation does not consume (vel channels {contract.vel_channels})")


__all__ = ["ObservationContract", "OBSERVATION_CONTRACTS", "derive_contract", "lookup",
           "perception_group_doses", "admit_perception", "FULL", "PLANAR", "ESTIMATE_GROUPS",
           "JOURNAL_TASK_CONTRACTS", "JOURNAL_RAY_BLOCK", "HOOKS_FLAG", "effective_contract",
           "journal_contract_for"]
