# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .._shared.vehicles import PhysicsVersionCfg, physics_version

import os as _os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .._shared.sea_state import SeaStateCfg
from .._shared.vehicles import VehicleSpec, get_vehicle


# USVBench asset directory. This locates the USD; it is not a physics knob.
_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)

_DEFAULT_VEHICLE = get_vehicle("rov")


def _build_robot_cfg(spec: VehicleSpec) -> ArticulationCfg | RigidObjectCfg:
    """Build the selected asset while preserving the legacy ROV spawn config."""
    spawn_kwargs = {
        "usd_path": _os.path.join(_ASSET_DIR, spec.usd_relpath),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # Non-binding safety bounds, not vehicle-physics parameters.
            max_linear_velocity=10.0 if spec.asset_kind == "articulation" else 8.0,
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

    if spec.asset_kind == "articulation":
        spawn_kwargs["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        )
        return ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, -0.15),
                rot=(1.0, 0.0, 0.0, 0.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
                joint_pos={},
                joint_vel={},
            ),
            actuators={},
            collision_group=0,
        )

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


# Public compatibility alias; values now resolve through the registry.
ROV_CONFIG = _build_robot_cfg(_DEFAULT_VEHICLE)


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
    """Dormant calm-water settings copied from the realistic ROV baseline."""

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
    enable_hold_zone_marker: bool = True
    hold_zone_segments: int = 64
    hold_zone_color: tuple = (1.0, 0.15, 0.0)
    hold_zone_line_width_m: float = 0.35


@configclass
class StationKeepingEnvCfg(DirectRLEnvCfg):
    vehicle: str = "rov"

    decimation = 2
    episode_length_s = 120.0

    action_space = 2
    observation_space = 3
    state_space = 0
    # v11-style kinematic channels (body-frame surge / sway / yaw rate). Off
    # here so the certified 3-D ids stay byte-identical; kinematic variants
    # register their own gym ids (append-only discipline).
    obs_kinematic: bool = False
    yaw_rate_obs_scale_rad_s: float = 1.0

    # Sea state (JONSWAP). Enters through the SAME world-frame force path as
    # the certified current, so a task may carry both. Off by default.
    sea_state: SeaStateCfg = SeaStateCfg()
    obs_sea_state: bool = False

    # --- Observation-channel degradation (eval-time OOD hooks) ---------------
    # Physical-unit corruption of the IDEAL measurements (hold-point vector,
    # body-frame velocities) before feature construction, inside
    # _get_observations only; reward/termination/success keep reading clean
    # state. All zero => no degrader object is built and every certified id
    # runs its frozen observation path byte-identical. Eval-time knobs (set
    # via eval_v6_frozen --set); no gym id may ship nonzero defaults without
    # its own registration. Yaw-rate sigma rides the vel knob scaled by
    # yaw_rate_obs_scale_rad_s / SPEED_SCALE_MPS. No ray group here.
    obs_noise_sigma_pos: float = 0.0  # m, white noise on the hold vector
    obs_noise_sigma_vel: float = 0.0  # m/s, white noise on surge/sway (+ yaw)
    obs_bias_sigma_pos: float = 0.0   # m, per-episode constant offset draw
    obs_bias_sigma_vel: float = 0.0   # m/s, per-episode constant offset draw
    obs_delay_steps: int = 0          # control steps of measurement latency
    obs_dropout_p: float = 0.0        # per-step frame loss (sample-and-hold)

    hold_radius: float = 2.0
    required_hold_time_s: float = 60.0
    min_spawn_distance: float = 5.0
    max_spawn_distance: float = 15.0
    reference_reward_distance_scale: float = 30.0

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

    robot_cfg: ArticulationCfg | RigidObjectCfg = ROV_CONFIG
    underwater_physics_cfg: UnderwaterPhysicsCfg = _build_underwater_physics_cfg(
        _DEFAULT_VEHICLE
    )
    # Tier-regrade probe: pins the current to one speed (min = max = override)
    # and force-enables it. 0.0 = untouched physics, byte-identical. Consumed
    # by the env AFTER __post_init__ rebuilds underwater_physics_cfg, so the
    # rebuild cannot silently erase an eval-time override.
    current_speed_override_mps: float = 0.0
    wave_cfg: WavePhysicsCfg = WavePhysicsCfg()
    # Rendering only: this block must not affect physics, observations, reward,
    # termination, or success semantics.
    visual: VisualCfg = VisualCfg()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=36.0,
        replicate_physics=True,
    )

    dof_names = []

    # Opt-in condition layer; inherited legacy registrations remain disabled.
    conditions: dict | None = None
    condition_presets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        base_post_init = getattr(super(), "__post_init__", None)
        if base_post_init is not None:
            base_post_init()

        if getattr(self, "journal_contract_version", None) == 2:
            # Journal (task contract v2) branch, F4: the v2 vessel is resolved
            # through journal_vehicle BEFORE the env builds the asset; legacy
            # registry lookups below never run for journal cfgs (CONTEXT_R4 D15).
            _journal_station_post_init(self)
            return

        spec = get_vehicle(self.vehicle)
        self.robot_cfg = _build_robot_cfg(spec)
        version = physics_version(self.underwater_physics_cfg)
        self.underwater_physics_cfg = _build_underwater_physics_cfg(spec)
        self.underwater_physics_cfg.physics_version = version
        self.thrust_max_fwd = spec.thrust_fwd_n
        self.thrust_max_rev = spec.thrust_rev_n
        self.yaw_torque_max = spec.yaw_torque_nm


@configclass
class StationKeepingBlueBoatEnvCfg(StationKeepingEnvCfg):
    vehicle: str = "blueboat"


@configclass
class StationKeepingBlueBoatCurrentEnvCfg(StationKeepingBlueBoatEnvCfg):
    """BlueBoat station keeping with randomized crossing current."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.underwater_physics_cfg.enable_current = True
        self.underwater_physics_cfg.current_speed_min = 2.0
        self.underwater_physics_cfg.current_speed_max = 2.5
        self.underwater_physics_cfg.current_drag_coeff = 8.0


@configclass
class StationKeepingBlueBoatKinEnvCfg(StationKeepingBlueBoatEnvCfg):
    """Calm-water station keeping with the v11 kinematic observation block."""

    obs_kinematic: bool = True
    observation_space = 6


@configclass
class StationKeepingBlueBoatCurrentKinEnvCfg(StationKeepingBlueBoatCurrentEnvCfg):
    """Station keeping x current with the v11 kinematic observation block."""

    obs_kinematic: bool = True
    observation_space = 6


@configclass
class StationKeepingBlueBoatWaveEnvCfg(StationKeepingBlueBoatKinEnvCfg):
    """C2 x waves: hold station in an irregular sea.

    Peak periods are pinned SHORT on purpose. For a 1.2 m hull, a 6 s wave has
    a 56 m wavelength -- 47 boat lengths, which the boat simply rides. The
    band that actually disturbs this vessel is 1-3 s (1.6-14 m wavelength), so
    copying a competition's "hard" tier verbatim would manufacture a fake
    difficulty of exactly the kind the admission protocol exists to catch.
    """

    obs_sea_state: bool = True
    observation_space = 6 + 3

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sea_state.enable = True
        self.sea_state.hs_range = (0.30, 0.60)
        self.sea_state.tp_range = (1.5, 3.0)


@configclass
class StationKeepingBlueBoatRampCurrentEnvCfg(StationKeepingBlueBoatCurrentEnvCfg):
    """C2 x ramping current: the speed climbs linearly within each episode.

    Owner-proposed variant (meeting-approved design). The episode's current
    DIRECTION is sampled exactly as in the Current variant; the SPEED is
    overridden every control step with a linear ramp from
    ``current_ramp_start_mps`` at the first step to ``current_ramp_end_mps``
    at the final step (see current_ramp.py). The inherited
    current_speed_min/max only seed the direction-carrying base vector.
    Gym id registration is PENDING -- see
    tasks/hazard_nav/PENDING_REGISTRATIONS.md -- so the certified Current ids
    stay byte-identical (cfgs without both ramp fields never enter the ramp
    path).
    """

    current_ramp_start_mps: float = 0.5
    current_ramp_end_mps: float = 3.0


@configclass
class StationKeepingBlueBoatJBaseCalmEnvCfg(StationKeepingBlueBoatKinEnvCfg):
    """J-base calm helper (renamed CONTEXT_R4 D10; not a journal id): station keeping, BlueBoat,
    calm water, physics v2.

    NOT registered (the proposed id block is in the D1 MERGE_NOTES.md).
    Parent: the calm Kin contract (hold-point vector + body rates, 6 obs
    channels), so velocity perception doses are supported. The condition
    layer adds NO sea, current or wind observation channel. Every suite
    defaults off (calm v2); conditions are selected as data through
    ``conditions`` / ``condition_presets`` and validated by
    ``build_conditions`` in journal mode, which rejects the legacy current,
    ramp, sea state, actuator, perception and appearance controls.
    """

    journal_mode: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        # After the registry rebuild, which preserves an explicit version.
        self.underwater_physics_cfg.physics_version = 2


# --------------------------------------------------------------------------
# Journal (task contract v2) cfg branch, round F / F4. Reached only from
# StationKeepingEnvCfg.__post_init__ when journal_contract_version == 2, i.e.
# from E5's journal_runtime_cfg_v2 templates (whose _journal_post_init has
# already set journal_plan and physics_version = 2). Legacy cfgs never enter.
# --------------------------------------------------------------------------
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# G23e: approach-design pools (sk_approach_crossing_v1); the retired hold-design pools stay in d7_pools/ (unused)
JOURNAL_D7_POOL_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "d7_pools_approach")
JOURNAL_D7_POOL_INDEX = _os.path.join(JOURNAL_D7_POOL_DIR, "INDEX.json")
# Non-binding PhysX safety bounds (not vessel physics): both v2 top speeds
# (BlueBoat 3.0 m/s, WAM-V 2.26 m/s) lie well inside them.
JOURNAL_MAX_LINEAR_VELOCITY_M_S = 8.0
JOURNAL_MAX_ANGULAR_VELOCITY_DEG_S = 573.0


def _build_journal_robot_cfg(usd_path: str, com_z_m: float) -> RigidObjectCfg:
    """v2 asset (origin = COM, bow +x); mass, inertia and COM are the USD-authored
    registry values (no mass override); the COM spawns at water + calm datum."""
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=JOURNAL_MAX_LINEAR_VELOCITY_M_S,
                max_angular_velocity=JOURNAL_MAX_ANGULAR_VELOCITY_DEG_S,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(com_z_m)),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        collision_group=0,
    )


def _journal_d7_pool_index_entries(vehicle: str, tier: int) -> dict:
    """{INDEX.json key: entry} of every pinned pool of (vessel, tier), any evaluation seed (G49)."""
    import json as _json

    if not _os.path.isfile(JOURNAL_D7_POOL_INDEX):
        return {}
    with open(JOURNAL_D7_POOL_INDEX, "r", encoding="ascii") as handle:
        index = _json.load(handle)
    base = "%s/tier%d" % (vehicle, int(tier))
    return {key: entry for key, entry in index.get("pools", {}).items()
            if key == base or key.startswith(base + "/seed")}


def journal_d7_pool_entry(vehicle: str, tier: int, seed=None):
    """(absolute path, sha256) of the admitted D7 pool manifest for (vessel, tier[, seed]), or (None, None).

    G49 (J29, v2.6.5): INDEX.json pins the pools of several evaluation seeds. "<vessel>/tier<k>" is the v2.6.4
    entry (seed 42); every other seed's pool is "<vessel>/tier<k>/seed<seed>". seed None -> the
    "<vessel>/tier<k>" entry (v2.6.4 behaviour); otherwise the entry pinned for that seed, or (None, None).
    """
    entries = _journal_d7_pool_index_entries(vehicle, tier)
    base = "%s/tier%d" % (vehicle, int(tier))
    if seed is None:
        entry = entries.get(base)
    else:
        entry = entries.get("%s/seed%d" % (base, int(seed)))
        if entry is None and base in entries and int(entries[base].get("seed", -1)) == int(seed):
            entry = entries[base]
    if entry is None:
        return None, None
    return _os.path.join(JOURNAL_D7_POOL_DIR, entry["file"]), str(entry["sha256"])


def journal_d7_pool_for_seed(cfg, tier: int):
    """G49: (pool, sha256) the station-keeping env loads for cfg.seed.

    __post_init__ fills cfg.journal_d7_pool / journal_d7_pool_sha256 with the v2.6.4 INDEX.json entry (the
    template seed); the evaluator sets cfg.seed afterwards (eval_protocol.resolve_protocol). When the cfg still
    holds an INDEX.json entry of this vessel/tier (any seed) and INDEX.json pins a pool for cfg.seed, that pool's
    (path, sha256) is returned; anything else (an explicit path, an in-memory pool, a seed without pins) is
    returned unchanged, and StationD7Pool.check_env refuses a seed mismatch exactly as in v2.6.4.
    """
    pool = getattr(cfg, "journal_d7_pool", None)
    pinned = getattr(cfg, "journal_d7_pool_sha256", None)
    seed = getattr(cfg, "seed", None)
    if not isinstance(pool, str) or seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        return pool, pinned
    held = _os.path.normcase(_os.path.abspath(pool))
    known = {(_os.path.normcase(_os.path.abspath(_os.path.join(JOURNAL_D7_POOL_DIR, e["file"]))), str(e["sha256"]))
             for e in _journal_d7_pool_index_entries(cfg.vehicle, tier).values()}
    if (held, pinned) not in known:
        return pool, pinned
    path, sha = journal_d7_pool_entry(cfg.vehicle, tier, seed)
    return (path, sha) if path is not None else (pool, pinned)


def journal_pinned_pool_fields(pool, vehicle: str, tier: int, seed=None) -> dict:
    """G6b (CONTEXT_R4 D19/D23): D7 header fields that pin the pool to the release.

    The station-keeping pools ship in the release under d7_pools/ and INDEX.json is their
    pinned manifest (file name, SHA-256, pool digest per vessel/tier). Every other family
    stamps the SHA-256 of its pinned manifest as dynamic_hazards_header.pool_manifest_sha256
    (F1 hazard_nav MANIFEST.json, F2 path_hazard d7_pool_manifest.json, F3 path_following
    POOL_MANIFEST.json, F5 docking POOL_MANIFEST.json, F6 harbour POOL_MANIFEST.json);
    cert_v3.journal_pool_manifest_sha256 refuses a journal certificate without it. Station
    keeping never published it (absent from T-base v1 on; the P5 v3 run failed in cert_v3).
    The digest is stamped only when the loaded pool IS the index entry (same file SHA-256);
    for any other pool (e.g. an in-memory test pool) nothing is stamped and cert_v3 refuses
    the certificate (fail closed, D19).
    G49: ``seed`` (cfg.seed) selects the INDEX.json entry of that evaluation seed (None = the v2.6.4 entry).
    """
    import hashlib as _hashlib

    _path, pinned_sha256 = journal_d7_pool_entry(vehicle, tier, seed)
    source = getattr(pool, "source", None) or {}
    if pinned_sha256 is None or source.get("sha256") != pinned_sha256:
        return {}
    with open(JOURNAL_D7_POOL_INDEX, "rb") as handle:
        return {"pool_manifest_sha256": _hashlib.sha256(handle.read()).hexdigest()}


def _journal_station_post_init(cfg) -> None:
    """Station-keeping journal cfg (F4; ANCHORS_E5 C6, CONTEXT_R4 D1-D3, D5, D13, D14).

    * vessel: resolve_journal_vehicle(cfg.vehicle) (v2 registry, legacy names
      rejected), asset from assets_v2 with the registry SHA-256 (fail closed);
    * physics: physics_version 2, hydrodynamics_model "hydro_v2", no legacy
      current / sea state; vessel_profile(+status) for the E1 certificate;
    * observation: kinematic 6 + the plan's 36-ray block (controller decision:
      KINEMATIC), journal_observation_hooks_wired = True;
    * task: hold radius / approach / hold time from the frozen plan;
    * D7: the admitted pool manifest by path (cfg.journal_d7_pool) + pinned SHA-256.
    """
    from .._shared.journal_vehicle import ASSETS_DIRNAME, resolve_journal_vehicle

    plan = getattr(cfg, "journal_plan", None)
    if plan is None or getattr(cfg, "journal_task", None) != "station_keeping":
        raise ValueError("journal station-keeping cfgs are built from journal_runtime_cfg_v2 "
                         "(journal_task='station_keeping' with a journal_plan)")
    if plan.task != "station_keeping" or plan.vessel != cfg.vehicle:
        raise ValueError("journal plan does not match the cfg task/vessel")
    for name in ("current_ramp_start_mps", "current_ramp_end_mps"):
        if getattr(cfg, name, None) is not None:
            raise ValueError("journal station keeping never declares the legacy current ramp "
                             "(CONTEXT_R4 D6, ANCHORS_E5 C3): " + name)
    if float(getattr(cfg, "current_speed_override_mps", 0.0) or 0.0) != 0.0:
        raise ValueError("journal current comes from the condition layer, never current_speed_override_mps")
    options = dict(getattr(cfg, "journal_vehicle_options", None) or {})
    vehicle = resolve_journal_vehicle(cfg.vehicle, **options)
    asset = vehicle.verify_asset(_os.path.join(_REPO_ROOT, ASSETS_DIRNAME))
    params = plan.params
    water_z = 0.0
    com_z = water_z + float(vehicle.hydro_profile.hydro.equilibrium_com_z_m)
    cfg.robot_cfg = _build_journal_robot_cfg(str(asset), com_z)
    physics = UnderwaterPhysicsCfg()
    physics.physics_version = 2
    physics.water_surface_z = water_z
    physics.enable_current = False
    physics.hydrodynamics_model = "hydro_v2"
    cfg.underwater_physics_cfg = physics
    thrusters = vehicle.hydro_profile.thrusters
    cfg.thrust_max_fwd = float(sum(thrusters.forward_limit_n))
    cfg.thrust_max_rev = float(sum(thrusters.reverse_limit_n))
    cfg.yaw_torque_max = float(thrusters.yaw_command_limit_nm)
    cfg.vessel_profile = vehicle.name
    cfg.vessel_profile_status = vehicle.profile_status
    cfg.sea_state.enable = False
    cfg.obs_sea_state = False
    cfg.obs_kinematic = True
    cfg.ray_count = int(params["ray_sensor_ray_count"])
    cfg.ray_max_range_m = float(params["ray_sensor_max_range_m"])
    cfg.action_space = 2
    cfg.observation_space = 6 + cfg.ray_count
    cfg.journal_observation_hooks_wired = True
    cfg.hold_radius = float(params["hold_radius_m"])
    cfg.required_hold_time_s = float(params["hold_s"])
    cfg.min_spawn_distance = float(params["approach_distance_m"])
    cfg.max_spawn_distance = float(params["approach_distance_m"])
    cfg.reference_reward_distance_scale = 30.0 * float(params["length_scale"])
    # Render/scene spacing only: one layout box per env (bounds width).
    width = float(params["approach_distance_m"]) + 2.0 * float(params["region_padding_m"])
    cfg.scene.env_spacing = max(36.0, width)
    cfg.journal_d7_pool, cfg.journal_d7_pool_sha256 = journal_d7_pool_entry(
        cfg.vehicle, int(cfg.journal_geometry_tier))
