# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .._shared.vehicles import PhysicsVersionCfg, physics_version

import os as _os

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .._shared.sea_state import SeaStateCfg
from .._shared.vehicles import VehicleSpec, get_vehicle
from . import level_table


# USVBench asset directory. This locates the USD; it is not a physics knob.
_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)


_DEFAULT_VEHICLE = get_vehicle("wamv")


def _build_robot_cfg(spec: VehicleSpec) -> RigidObjectCfg:
    """Build a docking robot spawn config without overriding authored mass."""
    spawn_kwargs = {
        "usd_path": _os.path.join(_ASSET_DIR, spec.usd_relpath),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # Non-binding safety bounds, not vehicle-physics parameters.
            max_linear_velocity=8.0,
            max_angular_velocity=573.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        "activate_contact_sensors": False,
    }
    if spec.mass_kg is not None:
        spawn_kwargs["mass_props"] = sim_utils.MassPropertiesCfg(mass=spec.mass_kg)

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        collision_group=0,
    )


# Kept as a public compatibility alias; its values now come from the registry.
BOAT_CONFIG = _build_robot_cfg(_DEFAULT_VEHICLE)


@configclass
class UnderwaterPhysicsCfg(PhysicsVersionCfg):
    """Calm-water hull dynamics populated from a :class:`VehicleSpec`."""

    water_density: float = 1000.0
    gravity: float = 9.8
    water_surface_z: float = 0.0

    rov_volume: float = _DEFAULT_VEHICLE.displaced_volume_m3
    rov_height: float = _DEFAULT_VEHICLE.hull_height_m
    buoyancy_center_offset: float = _DEFAULT_VEHICLE.buoyancy_center_offset_m

    surge_lin_damping: float = _DEFAULT_VEHICLE.surge_lin
    surge_quad_damping: float = _DEFAULT_VEHICLE.surge_quad
    sway_lin_damping: float | None = _DEFAULT_VEHICLE.sway_lin
    sway_quad_damping: float | None = _DEFAULT_VEHICLE.sway_quad
    heave_damping: float = _DEFAULT_VEHICLE.heave_damping
    yaw_lin_damping: float = _DEFAULT_VEHICLE.yaw_lin
    yaw_quad_damping: float = _DEFAULT_VEHICLE.yaw_quad
    restoring_stiffness_roll: float = _DEFAULT_VEHICLE.restoring_stiffness_roll
    restoring_stiffness_pitch: float = _DEFAULT_VEHICLE.restoring_stiffness_pitch
    rollpitch_rate_damping: float = _DEFAULT_VEHICLE.rollpitch_rate_damping

    enable_current: bool = False
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0


def _build_underwater_physics_cfg(spec: VehicleSpec) -> UnderwaterPhysicsCfg:
    """Translate registry names to the existing docking physics interface."""
    return UnderwaterPhysicsCfg(
        rov_volume=spec.displaced_volume_m3,
        rov_height=spec.hull_height_m,
        buoyancy_center_offset=spec.buoyancy_center_offset_m,
        surge_lin_damping=spec.surge_lin,
        surge_quad_damping=spec.surge_quad,
        sway_lin_damping=spec.sway_lin,
        sway_quad_damping=spec.sway_quad,
        heave_damping=spec.heave_damping,
        yaw_lin_damping=spec.yaw_lin,
        yaw_quad_damping=spec.yaw_quad,
        restoring_stiffness_roll=spec.restoring_stiffness_roll,
        restoring_stiffness_pitch=spec.restoring_stiffness_pitch,
        rollpitch_rate_damping=spec.rollpitch_rate_damping,
    )


@configclass
class WavePhysicsCfg:
    """Dormant calm-water settings copied from the realistic boat baseline."""

    enable_wave: bool = False
    wave_height: float = 0.5
    wave_period: float = 5.0
    wave_dir_x: float = 1.0
    wave_dir_y: float = 0.0


@configclass
class VisualCfg:
    """Render-only settings; never read by physics, observations, or reward."""

    enable_water: bool = True
    water_size_m: float = 300.0
    water_res: int = 60
    water_color: tuple = (0.1, 0.3, 0.8)
    enable_tolerance_ring: bool = True
    tolerance_ring_segments: int = 64
    tolerance_ring_color: tuple = (1.0, 0.15, 0.0)
    tolerance_ring_line_width_m: float = 0.35
    enable_berth_marker: bool = True
    berth_arrow_length_m: float = 3.0
    berth_arrow_width_m: float = 0.5
    berth_arrow_color: tuple = (1.0, 0.85, 0.0)
    enable_berth_box: bool = True
    berth_box_length_m: float = 6.5
    berth_box_width_m: float = 3.2
    berth_box_line_width_m: float = 0.25
    berth_box_color: tuple = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Journal (contract v2) support. Used ONLY when a cfg carries
# journal_contract_version == 2 (E5 templates DockingBlueBoatJournalEnvCfg /
# DockingWAMVJournalEnvCfg, tasks/_shared/journal_runtime_cfg_v2.py); legacy
# ids never reach this code.
# ---------------------------------------------------------------------------

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Default folder of the D7 pools; the pool files ship with the release here (CONTEXT_R4 D19).
# cfg.d7_pool_dir / cfg.d7_pool_file are RUNTIME path variables (a different folder may hold the
# same pinned files); the file content is always checked against the pins below.
JOURNAL_D7_POOL_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "journal_pools")
# Pinned SHA-256 of every pool file the env may load (tools_f5/admit_d7_pool.py; D16 pools: admitted
# ladders + explicit not_admitted records) and of journal_pools/POOL_MANIFEST.json (stamped in the
# certificate header as dynamic_hazards_header.pool_manifest_sha256). Rewritten by tools_f5/pin_pools.py.
# G49 (v2.6.5, J29): + the G48 pools of seeds 11002 41041 41047 41051 41057 (names carry the seed).
# G53 (v2.6.7): + the pools of the competence-gate seed 11001 (selection-gate-01).
JOURNAL_D7_POOLS = {
    "docking_blueboat_v2_g2_s42_64x2.json.xz":
        "a28e85bd2426fa675015642b1ad057921cb37f3137ba7929f5a6959f894e4f4d",
    "docking_wamv_vrx_v2_opt1_g2_s42_64x2.json.xz":
        "334ececea3c23ca808a7bd96ff3cb64601fb2ae42cd2a64ba58d22cf2497bc49",
    "docking_blueboat_v2_g2_s11001_64x2.json.xz":
        "50b1fb643eb074e1080e67bb65bcddb9d2f3ce306b2b8489f258099db7ee2213",
    "docking_blueboat_v2_g2_s11002_64x2.json.xz":
        "7f9ba53c55fb3e9dc512389864f0aeec0e515e63fe8ab3e36a5113b10185e521",
    "docking_blueboat_v2_g2_s41041_64x2.json.xz":
        "d17151091dfd339429f1875e14bf182adf8406177faa4ef9de2bdb114086dc2a",
    "docking_blueboat_v2_g2_s41047_64x2.json.xz":
        "ea269f3c89fc2e9e0d74d1aab06faf3ed80cce315408b467636e58720acd867b",
    "docking_blueboat_v2_g2_s41051_64x2.json.xz":
        "37b15365ee54d19013061d03744d1a61d4b04c17e0bff729e5d92b2d8f4bbc26",
    "docking_blueboat_v2_g2_s41057_64x2.json.xz":
        "7ea3a09256498685df540c097cd8bb9e095b522df05fac438b15d0abf597438c",
    "docking_wamv_vrx_v2_opt1_g2_s11001_64x2.json.xz":
        "e7e9c2ca9df9dbda32e7ca4130e8f0d584fb528904a1d9afa5243875bd6a898e",
    "docking_wamv_vrx_v2_opt1_g2_s11002_64x2.json.xz":
        "c8ee0204136ce80c666ea528c0f037de3b269f4f7905bf723cd3012ef80c1a28",
    "docking_wamv_vrx_v2_opt1_g2_s41041_64x2.json.xz":
        "c34e43bdeba3b5bc8f9d7441fb8659f990c0a72f85c99013a4c44cd40b4c79f1",
    "docking_wamv_vrx_v2_opt1_g2_s41047_64x2.json.xz":
        "b0b96db4563e576f5df24a1468a62763515de722fb4fa10d593c6a972f04678e",
    "docking_wamv_vrx_v2_opt1_g2_s41051_64x2.json.xz":
        "eb11ea36680f8d46ee6826cd50724a6b504f7a63fd92e9ce1fef902d23687075",
    "docking_wamv_vrx_v2_opt1_g2_s41057_64x2.json.xz":
        "54a94171c934eb71400459dad407a6b11818fd74bfc6b4f4580cbb16684f8e4b",
}
JOURNAL_D7_POOL_MANIFEST_SHA256 = "632b072493ada0163109f234ca9d97598e17c2ae82772c7929f19301a1d940d3"


@configclass
class JournalUnderwaterPhysicsCfg(UnderwaterPhysicsCfg):
    """Physics selector of journal docking cfgs: hydro_v2 (CONTEXT_R3 decision 13).

    Under hydro_v2 the hull laws come from the v2 vessel profile
    (tasks/_shared/hydro_profiles.py via journal_vehicle); the inherited legacy
    damping/buoyancy numbers are NOT used by the journal branch of DockingEnv.
    water_surface_z is the env-local mean water level.
    """

    hydrodynamics_model: str = "hydro_v2"
    physics_version: int = 2


def _build_journal_robot_cfg(usd_path: str) -> RigidObjectCfg:
    """Spawn the hash-verified v2 asset (origin = COM, bow +x); authored mass kept."""
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                # Non-binding safety bounds, not vehicle-physics parameters.
                max_linear_velocity=8.0,
                max_angular_velocity=573.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        collision_group=0,
    )


def journal_d7_pool_name(vehicle: str, tier: int, seed: int, num_envs: int, episodes: int,
                         berth_walls: bool = True) -> str:
    """File name written by tools_f5/admit_d7_pool.py for one vessel/tier/seed/grid."""
    return "docking_%s%s_g%d_s%d_%dx%d.json.xz" % (vehicle, "" if berth_walls else "_nowalls", int(tier), int(seed),
                                                   int(num_envs), int(episodes))


def _journal_docking_post_init(cfg) -> None:
    """Journal branch of DockingEnvCfg.__post_init__ (called by the E5 template's parent post-init).

    ANCHORS_E5 C6: the v2 registry name is resolved through
    journal_vehicle.resolve_journal_vehicle (no legacy get_vehicle), the asset
    comes from assets_v2 with its registry SHA-256 checked, hydro_v2 is
    selected, vessel_profile / vessel_profile_status are declared for the E1
    certificate, and the ray sensor comes from the frozen plan (36 rays,
    30 m x s; CONTEXT_R4 D14). The journal observation is the docking native
    six + 36 ranges (observation_groups journal_dock6_ray36; hooks wired).
    """
    from .._shared.journal_vehicle import ASSETS_DIRNAME, resolve_journal_vehicle

    plan = getattr(cfg, "journal_plan", None)
    if plan is None or getattr(cfg, "journal_task", None) != "docking":
        raise ValueError("journal docking cfgs are built from the E5 templates (journal_task='docking', "
                         "journal_plan set before the parent post-init)")
    options = dict(getattr(cfg, "journal_vehicle_options", None) or {})
    vehicle = resolve_journal_vehicle(cfg.vehicle, **options)
    usd = vehicle.verify_asset(_os.path.join(_REPO_ROOT, ASSETS_DIRNAME))
    cfg.robot_cfg = _build_journal_robot_cfg(str(usd))
    physics = JournalUnderwaterPhysicsCfg()
    physics.physics_version = 2
    physics.hydrodynamics_model = "hydro_v2"
    cfg.underwater_physics_cfg = physics
    cfg.vessel_profile = vehicle.hydro_profile.name
    cfg.vessel_profile_status = vehicle.profile_status
    params = plan.params
    scale = float(params["length_scale"])
    cfg.ray_count = int(params["ray_sensor_ray_count"])
    cfg.ray_max_range_m = float(params["ray_sensor_max_range_m"])
    # berth_walls: the template parent (DockingBlueBoatWallEnvCfg) sets True; False keeps a virtual
    # berth with the same 36-ray block (D7 slot only; R4 D14) and needs its own admitted pool.
    cfg.berth_walls = bool(cfg.berth_walls)
    cfg.obs_kinematic = False
    cfg.observation_space = 6 + cfg.ray_count
    cfg.observation_contract = "journal_dock6_ray36"
    cfg.journal_observation_hooks_wired = True
    # Task lengths scale with the vessel (CONTEXT_R4 D3); speeds are 1:1 (cruise 1.0 m/s both).
    cfg.observation_distance_scale_m = 25.0 * scale
    cfg.reference_reward_distance_scale_m = 25.0 * scale
    cfg.reference_reward_alignment_decay_m = 2.5 * scale
    cfg.reference_reward_braking_decay_m = 2.5 * scale
    cfg.success_position_tolerance_m = float(params["position_tolerance_m"])
    cfg.goal_radius = float(params["position_tolerance_m"])
    cfg.success_heading_tolerance_deg = float(params["heading_tolerance_rad"]) * 180.0 / 3.141592653589793
    cfg.success_speed_tolerance_mps = float(params["speed_tolerance_m_s"])
    cfg.required_hold_time_s = float(params["hold_s"])
    # The journal geometry is frozen by the plan (no curriculum); evaluators pin curriculum_frozen=True.
    cfg.curriculum_frozen = True
    cfg.eval_level = 0
    # Render-only markers assume the legacy fixed +X berth at the env origin.
    cfg.visual.enable_tolerance_ring = False
    cfg.visual.enable_berth_marker = False
    cfg.visual.enable_berth_box = False
    extent = (float(params["approach_distance_m"]) - float(params["slip_opening_x_m"])
              + float(params["region_padding_m"]))
    cfg.scene.env_spacing = max(float(cfg.scene.env_spacing), 2.0 * extent)
    if not getattr(cfg, "d7_pool_dir", None):
        cfg.d7_pool_dir = JOURNAL_D7_POOL_DIR
    if not hasattr(cfg, "d7_pool_file"):
        cfg.d7_pool_file = ""


@configclass
class DockingEnvCfg(DirectRLEnvCfg):
    vehicle: str = "wamv"

    decimation = 2
    episode_length_s = 120.0

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    observation_space = 6
    state_space = 0

    success_position_tolerance_m: float = 2.5
    # Mission-evaluation alias of the position tolerance, used to compute SPL.
    goal_radius: float = 2.5
    success_heading_tolerance_deg: float = 15.0
    success_speed_tolerance_mps: float = 0.3
    required_hold_time_s: float = 5.0

    # V3 begins inside the position tolerance so align-and-stop is learnable
    # before the approach distance grows.
    curriculum_start_distance_m: float = 2.0
    curriculum_distance_increment_m: float = 1.0
    curriculum_max_distance_m: float = 25.0
    curriculum_ema_decay: float = 0.99
    curriculum_success_threshold: float = 0.6

    # Evaluation contract. Without these, a fresh env starts the curriculum at
    # `curriculum_start_distance_m` (2 m) and then ADVANCES during the eval run
    # itself, so a 128-episode score mixes difficulties and is not comparable
    # across checkpoints. hazard_nav has had this since the v6 campaign; the
    # docking family did not, which is why its certified numbers must be read
    # as "starting from the easiest curriculum level".
    curriculum_frozen: bool = False
    eval_level: int = 0
    # Spawn distance used at each frozen level, in metres.
    eval_level_distances: tuple = (25.0, 15.0, 8.0, 2.0)
    # Physics v2 freezes the same distances in easiest-to-hardest order.
    # Keep a separate table so existing ids and legacy overrides are unchanged.
    eval_level_distances_v2: tuple = (2.0, 8.0, 15.0, 25.0)
    # Tier-regrade probe: pins the frozen-eval spawn distance directly, in
    # metres (0.0 = use the level table above, byte-identical). The level
    # sweep of 2026-08-13 measured 100/23/12/6 percent from 2 m out to 25 m,
    # so a T1-T5 regrade needs intermediate distances to be measurable.
    spawn_distance_override_m: float = 0.0

    # V5 spawns on the approach lane behind the berth, with positions and bow
    # headings sampled relative to each environment's dock_heading rather than
    # in fixed world coordinates. Bow-ward drift and forward thrust therefore
    # carry the boat toward the berth while alignment supports the approach.
    spawn_bearing_limit_deg: float = 60.0
    spawn_heading_offset_limit_deg: float = 45.0

    observation_distance_scale_m: float = 25.0
    observation_speed_scale_mps: float = 2.0
    reference_reward_distance_scale_m: float = 25.0
    reference_reward_alignment_scale: float = 0.5
    reference_reward_alignment_decay_m: float = 2.5
    reference_reward_braking_scale: float = 0.4
    reference_reward_braking_decay_m: float = 2.5
    reference_reward_braking_speed_scale_mps: float = 1.0
    reference_reward_success_bonus: float = 3.0

    # C3 x C5 (solid berth walls). Off by default so the certified open-water
    # docking ids stay byte-identical; the wall variant gets its own gym id.
    berth_walls: bool = False
    # v11 kinematic channels; append-only, enabled by dedicated gym ids.
    obs_kinematic: bool = False
    yaw_rate_obs_scale_rad_s: float = 1.0
    slip_width_beams: float = 4.0  # tier ladder, as in hazard (5/4/3/2.5)
    slip_length_m: float = 7.0
    half_beam_m: float = 0.45
    ray_count: int = 36
    ray_max_range_m: float = 30.0
    # Contact economics carried over verbatim from the certified hazard ledger.
    reward_contact_entry_penalty: float = 25.0
    reward_contact_dwell_penalty: float = 1.0

    thrust_max_fwd: float = _DEFAULT_VEHICLE.thrust_fwd_n
    thrust_max_rev: float = _DEFAULT_VEHICLE.thrust_rev_n
    yaw_torque_max: float = _DEFAULT_VEHICLE.yaw_torque_nm

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    robot_cfg: RigidObjectCfg = BOAT_CONFIG
    underwater_physics_cfg: UnderwaterPhysicsCfg = _build_underwater_physics_cfg(
        _DEFAULT_VEHICLE
    )
    wave_cfg: WavePhysicsCfg = WavePhysicsCfg()
    # Rendering only: this block must not affect physics, observations, reward,
    # termination, or success semantics.
    visual: VisualCfg = VisualCfg()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=56.0,
        replicate_physics=True,
    )

    dof_names = []

    def __post_init__(self) -> None:
        """Resolve all hull-dependent config from the selected registry entry."""
        base_post_init = getattr(super(), "__post_init__", None)
        if base_post_init is not None:
            base_post_init()

        if getattr(self, "journal_contract_version", None) == 2:
            # Journal (contract v2) cfgs: v2 vessel through journal_vehicle, hydro_v2,
            # journal ray sensor, pinned D7 pool (ANCHORS_E5 C6/C8; F_FAMILY_COMMON 6-7).
            _journal_docking_post_init(self)
            self._docking_level_snapshot = level_table.level_snapshot(self)
            return

        spec = get_vehicle(self.vehicle)
        self.robot_cfg = _build_robot_cfg(spec)
        version = physics_version(self.underwater_physics_cfg)
        self.underwater_physics_cfg = _build_underwater_physics_cfg(spec)
        self.underwater_physics_cfg.physics_version = version
        self.thrust_max_fwd = spec.thrust_fwd_n
        self.thrust_max_rev = spec.thrust_rev_n
        self.yaw_torque_max = spec.yaw_torque_nm
        self._docking_level_snapshot = level_table.level_snapshot(self)

    def docking_level_table_in_force(self) -> str:
        """D2 item 14: the level table this cfg's env reads (or ``journal_plan``)."""
        return level_table.level_table_in_force(self)

    def validate_after_overrides(self) -> None:
        """Refuse overrides of a level table that is not in force (D2 item 14)."""
        snapshot = getattr(self, "_docking_level_snapshot", None)
        if snapshot is None:
            snapshot = level_table.level_snapshot(self)
            self._docking_level_snapshot = snapshot
        level_table.check_level_overrides(self, snapshot)


@configclass
class DockingBlueBoatEnvCfg(DockingEnvCfg):
    vehicle: str = "blueboat"


@configclass
class DockingBlueBoatCurrentEnvCfg(DockingBlueBoatEnvCfg):
    """BlueBoat docking with randomized crossing current."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.underwater_physics_cfg.enable_current = True
        self.underwater_physics_cfg.current_speed_min = 1.0
        self.underwater_physics_cfg.current_speed_max = 1.5
        self.underwater_physics_cfg.current_drag_coeff = 8.0


@configclass
class DockingBlueBoatWallEnvCfg(DockingBlueBoatEnvCfg):
    """C3 x C5: docking into a slip with SOLID walls (the fourth crossing).

    The berth stops being a virtual tolerance region and becomes a U of
    collision geometry: the hull must thread the opening and hold the pose
    without ever touching a wall (prefix semantics). Observation gains the
    frozen 36-ray contract, so the policy can actually see the slip.
    """

    berth_walls: bool = True
    observation_space = 6 + 36
    slip_width_beams: float = 4.0
    slip_length_m: float = 7.0


@configclass
class DockingBlueBoatKinEnvCfg(DockingBlueBoatEnvCfg):
    """Open-water docking with the v11 kinematic observation block."""

    obs_kinematic: bool = True
    observation_space = 9


@configclass
class DockingBlueBoatCurrentKinEnvCfg(DockingBlueBoatCurrentEnvCfg):
    """Docking x current with the v11 kinematic observation block."""

    obs_kinematic: bool = True
    observation_space = 9


@configclass
class DockingBlueBoatWallKinEnvCfg(DockingBlueBoatWallEnvCfg):
    """Solid-berth docking with the v11 kinematic observation block."""

    obs_kinematic: bool = True
    observation_space = 6 + 3 + 36


@configclass
class DockingBlueBoatWaveEnvCfg(DockingBlueBoatEnvCfg):
    """Docking x waves: the certified berthing exam in an irregular sea.

    BACKGROUND DISTURBANCE ONLY. Unlike the station-keeping Wave id (which
    appends 3 sea-state observation channels), this variant changes NOTHING
    about the observation, reward, termination, or curriculum of its parent
    (Isaac-USV-Dock-BlueBoat-Direct-v1): the JONSWAP field enters through the
    same world-frame force path as the certified current, and the policy is
    never told the sea exists. A certified checkpoint therefore loads
    zero-shot and the comparison is "same policy, same observation, world now
    has waves".

    Band: the frozen evaluation box (2026-08-26): H_s 0.30-0.60 m (the
    station-keeping wave band for this hull), T_p 2.0-2.5 s. Every corner
    passes the DNV-RP-C205 steepness admission
    (sea_state.validate_steepness_admission; worst corner Hs=0.60 m at
    Tp=2.0 s gives Sp ~ 1/10.4 < 1/7) and captures >= 95% of the analytic
    JONSWAP variance on the default 104-component 0.04-1.60 Hz band (worst
    corner 98.8% at gamma=1). The rung ladder pins Hs per-rung later via
    --set; hs_range stays the full certified band here. The certified
    station-keeping id's 1.5 s floor stays excluded: it is grandfathered
    history, not a template.

    DUAL CALM REFERENCE: enable=True with Hs -> 0 is NOT calm water -- the
    quadratic hull drag in sea_state.forces() still opposes the hull with
    orbital_drag_coeff * speed^2 (0.08 N at 0.1 m/s, 8 N at 1 m/s, 72 N at
    3 m/s; see sea_state.zero_amplitude_drag_force). Any dose ladder over
    this variant must carry BOTH a true-calm reference (enable=False) and an
    intercept rung (enable=True, Hs -> 0).
    """

    sea_state: SeaStateCfg = SeaStateCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sea_state.enable = True
        self.sea_state.hs_range = (0.30, 0.60)
        self.sea_state.tp_range = (2.0, 2.5)
