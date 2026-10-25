"""Isaac Lab journal cfg classes, contract v2 -- INTEGRATION TEMPLATE.

NEVER imported by the CPU tests (it imports isaaclab). It is UNVERIFIED until
the env families are wired (part F) together with every edit in
INTEGRATION_SPEC.md / ANCHORS_E5.md; until then the J-base parents reject
layout_mode="journal" (J-base hazard_nav_env_cfg.py:683-699) and the v2
vessel profiles (tasks/_shared/vehicles_v2.py) are not reachable through the
legacy get_vehicle registry (tasks/_shared/vehicles.py) that the parents use.

One class per task per vessel (24): BlueBoat ``blueboat_v2`` and the default
WAM-V profile ``wamv_vrx_v2_opt1`` (journal_cfg_v2.DEFAULT_WAMV_PROFILE,
PENDING owner decision). The vessel is a CLASS DEFAULT, so it is selected
before the parent __post_init__ builds the asset, hydrodynamics and thrust
limits (C1 finding 46); journal_validate() rejects any later change. Another
registered profile (e.g. ``wamv_vrx_v2_opt2``) is a subclass overriding
``vehicle``; its plan comes from journal_cfg_v2.plan_class. ``journal_mode``
marks the cfg for J-base's validated condition layer
(tasks/_shared/conditions/layer.py:208-250). No gym id is registered here;
the proposed ids are journal_cfg_v2.JOURNAL_ENV_IDS.
"""

import os

from isaaclab.utils import configclass

from . import journal_cfg_v2 as plans
from .journal_plan_slot import HIDDEN_CFG_MEMBERS, PLAN_KEY, JournalPlanSlot  # noqa: F401  (G6)
from ..docking.docking_env_cfg import DockingBlueBoatWallEnvCfg
from ..harbor_mission.harbor_mission_env_cfg import HarborMissionEnvCfg
from ..hazard_nav.hazard_nav_env_cfg import HazardNavV3EnvCfg
from ..path_following.path_following_env_cfg import PathFollowingEnvCfg
from ..path_hazard.path_hazard_env_cfg import PathHazardV2EnvCfg
from ..station_keeping.station_keeping_env_cfg import StationKeepingEnvCfg


def _journal_post_init(cfg, parent_post_init):
    """Shared body of every journal __post_init__ (call order matters)."""
    plan_cls = plans.plan_class(cfg.journal_task, cfg.vehicle)
    plan = plan_cls(int(cfg.journal_geometry_tier))
    cfg.journal_plan = plan
    # Physics v2 must be selected BEFORE the parent rebuilds underwater physics;
    # the C2 rebuild preserves an explicitly selected version
    # (e.g. J-base hazard_nav_env_cfg.py:573-575).
    cfg.underwater_physics_cfg.physics_version = plans.PHYSICS_VERSION
    parent_post_init()
    cfg.underwater_physics_cfg.physics_version = plans.PHYSICS_VERSION
    cfg.episode_length_s = plan.max_deadline_s
    if hasattr(cfg, "appear_count"):
        cfg.appear_count = 0
    if hasattr(cfg, "max_obstacles"):
        # hazard_nav_env_cfg.py:189 and harbor_mission_env_cfg.py:181 (J-base);
        # path_hazard/docking allocate internally (INTEGRATION_SPEC.md 3.4/3.5).
        cfg.max_obstacles = max(1, plans.STATIC_CAPACITY[cfg.journal_task]
                                + plans.D7_RESERVED_SLOTS[cfg.journal_task])
    for flag in plans.FORBIDDEN_CFG_FLAGS_TRUE:
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    journal_validate(cfg)


def journal_validate(cfg):
    """Pure re-check after every override batch (fail closed)."""
    plan = cfg.journal_plan
    plan.assert_vehicle_override(cfg.vehicle)
    if getattr(cfg, "journal_mode", False) is not True:
        raise ValueError("journal cfgs run with journal_mode=True (validated condition layer)")
    if int(cfg.journal_contract_version) != plans.CONTRACT_VERSION:
        raise ValueError("journal_contract_version is frozen at 2")
    if int(cfg.underwater_physics_cfg.physics_version) != plans.PHYSICS_VERSION:
        raise ValueError("journal physics_version is frozen at 2")
    if int(cfg.journal_geometry_tier) != plan.geometry_tier:
        raise ValueError("reconstruct the cfg to change the geometry tier")
    if abs(float(cfg.episode_length_s) - plan.max_deadline_s) > 1.0e-9:
        raise ValueError("the episode cap is derived from the frozen route budget")
    for flag in plans.FORBIDDEN_CFG_FLAGS_TRUE:
        if bool(getattr(cfg, flag, False)):
            raise ValueError("journal controllers do not observe sea/current/wind: " + flag)
    if getattr(cfg, "appear_count", 0) != 0:
        raise ValueError("legacy appearance is disabled in journal cfgs; D7 owns dynamic hazards")
    # C1 finding 61: inherited environment variables silently redefined
    # legacy cells (goal radius, observation width, action delay, wave mode).
    present = [name for name in plans.REQUIRED_ENV_OVERRIDES_ABSENT if name in os.environ]
    if present:
        raise ValueError("journal runs refuse ambient overrides: " + ", ".join(sorted(present)))


# ---- reach (carrier: hazard_nav, open-water journal layout) ---------------

@configclass
class ReachBlueBoatJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "reach"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class ReachWAMVJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "reach"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


# ---- path following -------------------------------------------------------

@configclass
class PathFollowingBlueBoatJournalEnvCfg(PathFollowingEnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "path_following"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class PathFollowingWAMVJournalEnvCfg(PathFollowingEnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "path_following"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


# ---- station keeping ------------------------------------------------------

@configclass
class StationKeepingBlueBoatJournalEnvCfg(StationKeepingEnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "station_keeping"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    obs_sea_state: bool = False

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class StationKeepingWAMVJournalEnvCfg(StationKeepingEnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "station_keeping"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    obs_sea_state: bool = False

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


# ---- docking --------------------------------------------------------------

@configclass
class DockingBlueBoatJournalEnvCfg(DockingBlueBoatWallEnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "docking"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class DockingWAMVJournalEnvCfg(DockingBlueBoatWallEnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "docking"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


# ---- path with hazards ----------------------------------------------------

@configclass
class PathHazardBlueBoatJournalEnvCfg(PathHazardV2EnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "path_hazard"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class PathHazardWAMVJournalEnvCfg(PathHazardV2EnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "path_hazard"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


# ---- hazard-nav family: scatter, ring, fortress, crossing, iceberg, basin --

@configclass
class ScatterBlueBoatJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "scatter"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class ScatterWAMVJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "scatter"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class RingBlueBoatJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "ring"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class RingWAMVJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "ring"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class FortressBlueBoatJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "fortress"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class FortressWAMVJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "fortress"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class ForcedCrossingBlueBoatJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "forced_crossing"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class ForcedCrossingWAMVJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "forced_crossing"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class IcebergBlueBoatJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "iceberg"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class IcebergWAMVJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "iceberg"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class MovingBasinBlueBoatJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "moving_basin"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class MovingBasinWAMVJournalEnvCfg(HazardNavV3EnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "moving_basin"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    layout_mode: str = "journal"

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


# ---- harbour mission ------------------------------------------------------

@configclass
class HarbourMissionBlueBoatJournalEnvCfg(HarborMissionEnvCfg, JournalPlanSlot):
    vehicle: str = "blueboat_v2"
    journal_task: str = "harbour_mission"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    mission_depth: int = 3
    spawn_phase: int = 0

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)


@configclass
class HarbourMissionWAMVJournalEnvCfg(HarborMissionEnvCfg, JournalPlanSlot):
    vehicle: str = "wamv_vrx_v2_opt1"
    journal_task: str = "harbour_mission"
    journal_contract_version: int = 2
    journal_mode: bool = True
    journal_geometry_tier: int = 2
    mission_depth: int = 3
    spawn_phase: int = 0

    def __post_init__(self):
        _journal_post_init(self, super().__post_init__)
