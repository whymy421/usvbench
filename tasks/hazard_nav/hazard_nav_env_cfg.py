# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the static Hazard Navigation direct task."""

from __future__ import annotations

import os as _os

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .._shared.obs_superset import SUPERSET_DIM, SUPERSET_DIM_V2
from .._shared.vehicles import VehicleSpec, get_vehicle


_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)
_DEFAULT_VEHICLE = get_vehicle("blueboat")


def _build_robot_cfg(spec: VehicleSpec) -> ArticulationCfg | RigidObjectCfg:
    """Build a registry-selected vehicle without overriding authored mass."""
    spawn_kwargs = {
        "usd_path": _os.path.join(_ASSET_DIR, spec.usd_relpath),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
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


@configclass
class UnderwaterPhysicsCfg:
    """Calm-water hull dynamics populated from the vehicle registry."""

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
class VisualCfg:
    """Render-only layer; no field is read by task physics or scoring."""

    enable_water: bool = True
    water_size_m: float = 1000.0
    water_res: int = 80
    water_color: tuple = (0.1, 0.3, 0.8)
    enable_goal_ring: bool = True
    goal_ring_segments: int = 64
    goal_ring_line_width_m: float = 0.30
    goal_ring_color: tuple = (1.0, 0.45, 0.0)
    enable_obstacles: bool = True
    obstacle_color: tuple = (0.35, 0.37, 0.40)
    obstacle_height_m: float = 2.0


@configclass
class HazardNavEnvCfg(DirectRLEnvCfg):
    """Adversarially-converged Task A: Static Hazard Navigation."""

    vehicle: str = "blueboat"

    decimation = 2
    episode_length_s = 120.0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    observation_space = 39
    state_space = 0
    emit_superset_obs: bool = False
    superset_version: int = 1
    # B1 (v2 id): native obs gains the reached latch + speed_norm (41-D), the
    # superset stage slot carries the latch, and the swiftness term activates.
    obs_v2: bool = False
    # v3 exposes the minimal planar dynamic state: body-frame surge, sway, and
    # yaw rate. It is mutually exclusive with the append-only v2 layout.
    obs_v3: bool = False
    yaw_rate_obs_scale_rad_s: float = 1.0

    # Render-only official Blue Robotics model. The gray physics mesh remains
    # authoritative for mass, inertia, collision, buoyancy, and drag.
    use_official_blueboat_visual: bool = True
    official_blueboat_visual_usd_path: str = _os.path.join(
        _ASSET_DIR, "BB120_official_visual_only.usd"
    )
    official_blueboat_visual_translation: tuple = (-0.60396, -0.59322, -0.22357)
    official_blueboat_visual_orientation: tuple = (0.5, 0.5, 0.5, 0.5)

    goal_radius: float = 2.0
    min_goal_distance_m: float = 20.0
    max_goal_distance_m: float = 40.0
    half_beam_m: float = 0.45
    collision_margin_m: float = 0.20
    safe_clearance_m: float = 0.90

    # "scatter" = the certified corridor layout; ring modes put the spawn at
    # the center, while fortress modes put the goal there and spawn outside.
    layout_mode: str = "scatter"

    ray_count: int = 36  # v3: 10 deg spacing
    ray_max_range_m: float = 30.0
    # Level 3 spawns K=14; 12-slot buffers crashed the first level-3 reset,
    # so no pre-v6 run ever contained level-3 experience.
    max_obstacles: int = 14
    layout_max_attempts: int = 20
    layout_seed: int = 0

    curriculum_start_level: float = 0.0
    curriculum_level_increment: float = 1.0
    curriculum_max_level: float = 3.0  # v4: ultimate tier = 2x-beam gaps
    curriculum_ema_decay: float = 0.99
    curriculum_success_threshold: float = 0.60
    # Eval-only: the global curriculum EMA updates on every completed episode,
    # so a good policy can cross the 0.60 threshold MID-EVAL and silently
    # shift the tier under a harvest run. Freeze pins the level for the whole
    # process; every reported number must state its tier.
    curriculum_frozen: bool = False
    eval_level: int = 0

    reward_progress_scale: float = 20.0
    # Optional fortress-only substitution: progress follows the certified
    # half-beam geodesic field instead of straight-line distance. Off keeps the
    # historical reward path byte-for-byte unchanged for every existing id.
    reward_progress_geodesic: bool = False
    # v6a: quadratic graze tax retired (0.0) but KEPT for the shape-ablation
    # arm -- reward_clearance_scale=1.0 + reward_prox_scale=0.0 restores the
    # exact v5 reward path.
    reward_clearance_scale: float = 0.0
    # v6a: per-ray log proximity field over the 36 rays, band
    # [prox_ray_floor_m, safe_clearance_m + half_beam_m] = [0.45, 1.35] m.
    # Zero beyond ~2x hull-beam clearance; w=4.1 matches v5's at-touch cost
    # averaged over ray phase at the r=1.4 m reference cylinder.
    reward_prox_scale: float = 4.1
    prox_ray_floor_m: float = 0.45
    reward_contact_entry_penalty: float = 25.0
    reward_contact_dwell_penalty: float = 1.0
    reward_contact_dwell_quadratic: bool = False
    contact_dwell_tau_s: float = 0.5
    # Defaults preserve the historical contact terminal and clean-goal reward
    # contract for every existing environment id. New variants must opt out.
    contact_terminates: bool = True
    clean_goal_gate: bool = True
    # Paid once on the first collision-free goal entry. The bounded 50..100
    # range rewards decisive entry without letting lucky exploratory successes
    # dominate PPO's value targets.
    reward_goal_entry_bonus: float = 50.0
    reward_goal_time_bonus: float = 50.0
    # Integrated squared negative surge command. A full 120 s reverse episode
    # costs 6 points; a one-second escape manoeuvre costs at most 0.05.
    reward_reverse_action_scale: float = 0.05
    # Swiftness applies only when velocity is observable (v2/v3).
    # Full-idle episode cost = scale * 120 s = 30.0, making a timeout worse
    # than active progress while remaining comparable to one contact penalty.
    reward_swift_scale: float = 0.25

    # --- Potential-shaping correctness (OFF by default; new gym ids opt in) ---
    # Our progress term is a RAW difference Phi(s') - Phi(s). The policy-
    # invariance guarantee (Ng, Harada & Russell 1999, Thm 1) requires
    # gamma*Phi(s') - Phi(s); the missing factor is worth up to ~20 discounted
    # reward units over a 7200-step episode, and its sign is a standing "hurry
    # up and cut the straight-line distance" pressure. Grzes (AAMAS 2017)
    # further shows that with MULTIPLE terminal states -- ours are success,
    # contact and timeout -- Phi must be forced to zero at termination, or the
    # residual gamma^N Phi(s_N) term is action-dependent and flips the optimum.
    pbrs_correct: bool = False
    pbrs_gamma: float = 0.999
    # Zeroing Phi at TRUE terminals (goal reached / contact) is what Grzes
    # requires. It must NOT be applied at a time-limit truncation: that state
    # is not absorbing -- the critic bootstraps through it -- so forcing Phi=0
    # there injects a reward of 20*(d/D0) for "time ran out far from the goal".
    # Measured cost of getting this wrong: certified SR 75.8% -> 0.0%.
    pbrs_zero_at_terminal: bool = False
    # Shift the potential to be NON-NEGATIVE (route fraction covered) instead
    # of non-positive (distance remaining). With Phi <= 0 the discounted form
    # pays 20*(1-gamma)*|Phi| every step for merely existing far from the goal
    # -- 144 reward units per 7200-step episode against a progress budget of
    # 20, which is why both discounted variants collapsed (1.6% and 4.7%
    # against an 84.4% baseline). With Phi >= 0 the same drift term is a cost,
    # so standing still can never be profitable.
    pbrs_shift_potential: bool = False

    # Ring packing floor. None keeps the shipped value, whose 1.00 m surface
    # gaps the 0.899 m hull can slip through; RING_SEALED_OVERLAP_M closes
    # them to 0.67 m so the designed gap is the only way out.
    ring_neighbor_overlap_m: float | None = None

    # --- Feasibility pooling (observation layer) -----------------------------
    # Raw ranges encode a passable gap as two nearby threats and leave the
    # policy to infer whether its own beam fits. Pooling replaces them with
    # "how far can THIS hull travel in this direction", so a gap wider than the
    # beam reads as open water (Meyer et al., IEEE Access 2020).
    obs_feasibility: bool = False
    feasibility_sectors: int = 9
    feasibility_width_multiplier: float = 1.0

    # --- Open-water tax (the advisor's proposal, priced in arc_breakeven.py) --
    # "If the surrounding water has no obstacles, the pillar field is one big
    # obstacle and the boat will go around it." The fix is to make the water
    # AROUND the field cost something, rather than to pay a bonus for the gap.
    #
    # Why a penalty and not the half-sine arc: the deterministic ledger already
    # prefers threading by +5.8, so detouring is bought by RISK, not by reward
    # ranking. A bonus on the risky option is only collected on the runs that
    # succeed, so it must be ~1/q times the gap it closes -- 58 at q=0.5, 140
    # at q=0.3, against a goal bonus of 50-100. A tax on the safe option lands
    # with certainty and closes the same gap at face value.
    #
    # Charged per second while the nearest obstacle is farther than
    # open_water_radius_m, the hull has not reached the goal, and it is not on
    # final approach (or the goal's own 5 m clear disk would be taxed).
    # 0.0 keeps every existing id numerically identical.
    reward_open_water_scale: float = 0.0
    open_water_radius_m: float = 6.0
    open_water_goal_exempt_m: float = 6.0

    # --- One-shot threading bonus (owner's half-sine arc) --------------------
    # Paid once per episode on a completed clean passage; maximum on the gap
    # centreline, zero where the hull would touch, self-normalising by gap
    # width so one formula covers every tier.
    reward_threading_amplitude: float = 0.0

    # --- Suite D axis: left/right thrust imbalance ---------------------------
    # Fraction by which the starboard thruster out-pulls the port one:
    # port gain = 1 - x, starboard gain = 1 + x. 0.0 is a perfectly matched
    # pair and is an algebraic identity with the pre-existing lumped model, so
    # every certified id is untouched.
    #
    # This is the cleanest of the five Suite D axes because it is the only one
    # that breaks a SYMMETRY rather than shifting a scalar: a heavier boat or a
    # weaker motor makes the task uniformly harder, while an imbalance makes
    # straight-line travel require continuous asymmetric correction. Batista's
    # group is the closest prior work and varies it one parameter at a time
    # with the hydrodynamics frozen during training; the held-out packs below
    # are what they do not have.
    thrust_imbalance: float = 0.0
    # Non-empty values enable uniform per-episode sampling. Empty keeps the
    # scalar path above, including its exact zero-imbalance identity.
    thrust_imbalance_choices: tuple = ()

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
    robot_cfg: ArticulationCfg | RigidObjectCfg = _build_robot_cfg(_DEFAULT_VEHICLE)
    underwater_physics_cfg: UnderwaterPhysicsCfg = _build_underwater_physics_cfg(
        _DEFAULT_VEHICLE
    )
    # 96 m spacing contains a 40 m route and its sampled lateral corridor.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=96.0,
        replicate_physics=True,
    )
    visual: VisualCfg = VisualCfg()
    dof_names = []

    def __post_init__(self) -> None:
        base_post_init = getattr(super(), "__post_init__", None)
        if base_post_init is not None:
            base_post_init()

        spec = get_vehicle(self.vehicle)
        self.robot_cfg = _build_robot_cfg(spec)
        self.underwater_physics_cfg = _build_underwater_physics_cfg(spec)
        self.thrust_max_fwd = spec.thrust_fwd_n
        self.thrust_max_rev = spec.thrust_rev_n
        self.yaw_torque_max = spec.yaw_torque_nm

        if self.obs_v2 and self.obs_v3:
            raise ValueError("obs_v2 and obs_v3 are mutually exclusive")
        if self.superset_version not in (1, 2):
            raise ValueError("superset_version must be 1 or 2")
        # Feasibility pooling collapses the ray block into one number per
        # sector, so the range part of the observation shrinks accordingly.
        range_dim = (
            self.feasibility_sectors if self.obs_feasibility else self.ray_count
        )
        if self.obs_v3:
            native_dim = 6 + range_dim
        elif self.obs_v2:
            native_dim = 5 + range_dim
        else:
            native_dim = 3 + range_dim
        if self.emit_superset_obs:
            expected_observation_space = (
                SUPERSET_DIM_V2 if self.superset_version == 2 else SUPERSET_DIM
            )
        else:
            expected_observation_space = native_dim
        if self.ray_count != 36 or self.observation_space != expected_observation_space:
            raise ValueError(
                "HazardNav requires 36 rays and observation_space="
                f"{expected_observation_space} when "
                f"emit_superset_obs={self.emit_superset_obs}, "
                f"superset_version={self.superset_version}, "
                f"obs_v2={self.obs_v2}, obs_v3={self.obs_v3}"
            )
        if self.yaw_rate_obs_scale_rad_s <= 0.0:
            raise ValueError("yaw_rate_obs_scale_rad_s must be positive")
        if self.layout_mode not in (
            "scatter",
            "ring",
            "ring2",
            "fortress",
            "fortress2",
            "bandfort",
            "forced",
            "basin",
        ):
            raise ValueError(
                "layout_mode must be 'scatter', 'ring', 'ring2', 'fortress', "
                "'fortress2', 'bandfort', 'forced' or 'basin'"
            )
        if self.reward_progress_geodesic and self.layout_mode not in (
            "bandfort",
            "fortress",
            "fortress2",
        ):
            raise ValueError(
                "reward_progress_geodesic requires layout_mode 'bandfort', "
                "'fortress' or 'fortress2'"
            )
        # Undersizing this crashes the first reset of the affected tier, which
        # is how no pre-v6 run ever contained level-3 experience. The forced
        # floor is the 10k-layout audit's worst case (79) plus headroom.
        # Fortress floors are the samplers' hard returned-array worst cases
        # plus headroom: 23 + 9 and (32 + 56) + 8.
        min_obstacles = {
            "ring": 18,
            "ring2": 80,
            "fortress": 32,
            "fortress2": 96,
            "bandfort": 96,
            "forced": 80,
            "basin": 66,
        }.get(self.layout_mode, 14)
        if self.max_obstacles < min_obstacles:
            raise ValueError(
                f"max_obstacles must be >= {min_obstacles} for "
                f"layout_mode={self.layout_mode}"
            )
        if self.min_goal_distance_m != 20.0 or self.max_goal_distance_m != 40.0:
            raise ValueError("HazardNav v1 fixes D0 sampling to U[20, 40] m")
        if self.reward_goal_entry_bonus < 0.0 or self.reward_goal_time_bonus < 0.0:
            raise ValueError("goal-entry reward scales must be non-negative")
        if self.reward_reverse_action_scale < 0.0:
            raise ValueError("reward_reverse_action_scale must be non-negative")
        if self.contact_dwell_tau_s <= 0.0:
            raise ValueError("contact_dwell_tau_s must be positive")


@configclass
class HazardNavV2EnvCfg(HazardNavEnvCfg):
    """B1 wave: velocity observability + reached latch + swiftness term.

    Registered under the append-only gym id Isaac-USV-HazardNav-Direct-v2;
    the v1 id, layout, champions, and certificates stay untouched.
    """

    obs_v2: bool = True
    observation_space = 41


@configclass
class HazardNavV3EnvCfg(HazardNavEnvCfg):
    """Minimal Markov planar observation for inertia-aware navigation."""

    obs_v3: bool = True
    observation_space = 42
    # Fixed-angle chase view for v11 playback and video capture. Isaac Lab's
    # viewport controller translates this camera with env 0's boat at every
    # render update, but does not rotate it with the hull.
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, -6.0, 11.0),
        lookat=(0.0, 0.0, 0.30),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
        resolution=(1280, 720),
    )


@configclass
class HazardNavC64EnvCfg(HazardNavV3EnvCfg):
    """V3 scatter task emitting the complete observation contract v2."""

    emit_superset_obs: bool = True
    superset_version: int = 2
    observation_space = 64


@configclass
class HazardRingEnvCfg(HazardNavV3EnvCfg):
    """Ring-siege variant on the v11 recipe: encircled spawn, one gap.

    Inherits v3's kinematic observation, goal-entry bonus, and terminate-on
    -outcome behaviour; only the layout sampler changes, so a v11 champion
    can be evaluated here directly as a zero-shot structural-generalisation
    probe (same observation contract, same reward, different geometry).
    """

    layout_mode: str = "ring"
    max_obstacles: int = 18


@configclass
class HazardNavV3PbrsEnvCfg(HazardNavV3EnvCfg):
    """v11 recipe with the potential term in its policy-invariant form.

    Identical to v3 except the progress shaping becomes gamma*Phi(s') - Phi(s)
    with Phi forced to zero at every terminal state. This is the single change
    whose omission adds a standing "cut the straight-line distance now"
    pressure worth up to ~20 discounted units per episode.
    """

    pbrs_correct: bool = True


@configclass
class HazardNavV3FeasEnvCfg(HazardNavV3EnvCfg):
    """v11 recipe with feasibility-pooled ranges instead of raw rays.

    Observation width drops from 3 + 3 + 36 to 3 + 3 + 9, because each sector
    now carries one number -- the distance THIS hull can actually travel that
    way -- rather than nine raw ranges the policy must interpret.
    """

    obs_feasibility: bool = True
    feasibility_sectors: int = 9
    observation_space = 3 + 3 + 9


@configclass
class HazardNavV3ThreadEnvCfg(HazardNavV3EnvCfg):
    """v11 recipe plus the one-shot half-sine threading bonus."""

    reward_threading_amplitude: float = 5.0


@configclass
class HazardNavV3PbrsTermEnvCfg(HazardNavV3EnvCfg):
    """Discounted potential difference AND Phi = 0 at true terminals only.

    Separated from the plain discounted-difference variant so the two changes
    can be attributed independently; the first attempt bundled them and also
    (wrongly) zeroed at timeouts, which cost 75.8 points of success.
    """

    pbrs_correct: bool = True
    pbrs_zero_at_terminal: bool = True


@configclass
class HazardNavV3PbrsShiftEnvCfg(HazardNavV3EnvCfg):
    """Discounted potential difference over a NON-NEGATIVE potential."""

    pbrs_correct: bool = True
    pbrs_shift_potential: bool = True


@configclass
class HazardRingSealedEnvCfg(HazardRingEnvCfg):
    """Ring whose non-gap openings are narrower than the hull.

    The shipped ring builds neighbours to overlap by 0.30 m measured on the
    INFLATED disks, leaving 2*0.65 - 0.30 = 1.00 m between physical surfaces --
    wider than the 0.899 m beam, so the boat could leave anywhere. Verified in
    test_ring_sealed.py: 12/30 shipped layouts leak, 0/30 sealed ones do.
    """

    ring_neighbor_overlap_m: float = 0.55


@configclass
class HazardDoubleRingEnvCfg(HazardNavV3EnvCfg):
    """Two sealed siege rings with deliberately misaligned tier-width gaps."""

    layout_mode: str = "ring2"
    # The admission audit's worst shipped layout uses fewer than 70 cylinders;
    # 80 leaves reset-time headroom without inheriting the wall-heavy basin cap.
    max_obstacles: int = 80
    layout_max_attempts: int = 60


@configclass
class HazardFortressEnvCfg(HazardNavV3EnvCfg):
    """Sealed ring around the goal; the outside spawn must enter its one gap."""

    layout_mode: str = "fortress"
    # Sampler hard maximum 23 plus nine reset-buffer slots of headroom.
    max_obstacles: int = 32
    layout_max_attempts: int = 60


@configclass
class HazardFortress2EnvCfg(HazardNavV3EnvCfg):
    """Two sealed, misaligned rings around the goal; spawn outside both."""

    layout_mode: str = "fortress2"
    # Sampler hard maximum 32 inner + 56 outer, plus eight slots headroom.
    max_obstacles: int = 96
    layout_max_attempts: int = 80


@configclass
class HazardBandFortEnvCfg(HazardNavV3EnvCfg):
    """Two constructive brick-wall bands; geometry supplies the constraint."""

    layout_mode: str = "bandfort"
    # The 150-layout-per-tier audit observed 72 cylinders at worst; 96 leaves
    # 24 reset-buffer slots of headroom without changing the canonical ledger.
    max_obstacles: int = 96
    layout_max_attempts: int = 80


@configclass
class HazardBandFortGeoEnvCfg(HazardBandFortEnvCfg):
    """Band fortress whose unchanged progress ledger uses route distance."""

    reward_progress_geodesic: bool = True


@configclass
class HazardBandFortSoftEnvCfg(HazardBandFortEnvCfg):
    """Band fortress with the v12 soft ledger and quadratic contact dwell."""

    contact_terminates: bool = False
    clean_goal_gate: bool = False
    reward_contact_dwell_penalty: float = 2.0
    reward_contact_dwell_quadratic: bool = True


@configclass
class HazardForcedCrossingEnvCfg(HazardNavV3EnvCfg):
    """Closed basin split by a bulkhead with exactly one gate.

    The scatter arena has no boundary, so every certified Task A number
    measures willingness to detour: circumventing the field costs 1.63-1.82x
    at every tier and the 128/128 champion takes that route. Here admission
    proves that sealing the gate makes the goal unreachable AT THE TRUE
    HALF-BEAM (test_forced_crossing.py), so a success is a transit.

    Same observation contract and reward as v3, so a v11 champion can be
    evaluated here zero-shot. Tier difficulty is gate width only -- 5/4/3/2
    hull beams -- since the route length barely moves across tiers.
    """

    layout_mode: str = "forced"
    # Walls are tiled cylinders; the audit's worst case over 10k layouts sets
    # this. Undersizing it crashes the first reset of the affected tier, which
    # is how no pre-v6 run ever contained level-3 experience.
    max_obstacles: int = 96
    layout_max_attempts: int = 60
    # Obstacle marker prims are authored even when headless, and the reset path
    # walks max_obstacles of them per resetting env in Python. At 96 that is 7x
    # the scatter task's per-reset USD traffic, on a task whose walls make early
    # terminations frequent. Off for training; HazardCrossDemo turns them back
    # on for recording, where env counts are small.
    visual: VisualCfg = VisualCfg(enable_obstacles=False)


@configclass
class HazardCrossImbalanceEnvCfg(HazardForcedCrossingEnvCfg):
    """Forced crossing trained on the frozen Suite D imbalance pack."""

    thrust_imbalance_choices: tuple = (0.0, 0.02, 0.04)


@configclass
class HazardCrossDemoEnvCfg(HazardForcedCrossingEnvCfg):
    """Forced crossing with the walls drawn. For recording only, few envs."""

    visual: VisualCfg = VisualCfg(enable_obstacles=True)


@configclass
class HazardOpenWaterTaxEnvCfg(HazardNavV3EnvCfg):
    """v11 recipe plus a tax on the empty water around the obstacle field.

    The advisor's mechanism, with both parameters measured rather than guessed
    (scripts/measure_open_water.py, paired runs on identical layouts):

      per-step clearance   threading v11: p10 0.31  median 3.56  p90 9.30
                           detouring v8 : p10 3.24  median 6.77  p90 12.09

    Radius 12 m taxes 12.3% of the detour's pre-goal steps and only 2.9% of the
    threading champion's -- the knee of the selectivity curve. A 6 m radius
    would have taxed threading 28.4% of the time, i.e. punished the behaviour
    we are trying to buy; that was the default before it was measured.

    Scale is deliberately NOT the 5.12 that would close the modelled q = 0.5
    risk gap in one step: q is an assumption, not a measurement, and tuning a
    large reward term to an assumed q is how the shifted potential ended up
    teaching a bigger detour. 2.0/s costs the detour 29.5 per episode against
    the threading policy's 7.0 -- enough to move the balance, small enough to
    read the result. Raise it only if training says it is not enough.
    """

    reward_open_water_scale: float = 2.0
    open_water_radius_m: float = 12.0


@configclass
class HazardSoftLedgerEnvCfg(HazardOpenWaterTaxEnvCfg):
    """v9 open-water tax with the owner-approved softened contact ledger."""

    contact_terminates: bool = False
    clean_goal_gate: bool = False
    reward_contact_dwell_penalty: float = 0.1


@configclass
class HazardTaxPoolEnvCfg(HazardOpenWaterTaxEnvCfg):
    """v10 = v9's open-water tax + v5's feasibility-pooled observation.

    Owner's call (2026-08-02): combine the two interventions that each worked
    on one side. The tax (v9) fixed the INCENTIVE -- both seeds converge to
    35-40 m routes instead of one threading and one detouring -- but paid for
    it in collisions (8.4% -> 14.3%), because the policy still has to infer
    "do I fit?" from 36 raw rays while being pushed toward the gaps. Pooling
    (v5) fixed the PERCEPTION -- beam-aware "how far can THIS hull travel per
    sector" made one seed thread at 100% with a 33 m path -- but left the
    incentive alone, so its other seed still sat at 71-78%.

    Hypothesis: tax supplies the reason, pooling supplies the eyes; the
    combination should keep v9's route consistency while recovering v5's
    collision-free threading. Falsifier: if v10 collisions stay at v9 levels,
    perception was not the binding constraint and the contact-ledger redesign
    moves back up the queue.
    """

    obs_feasibility: bool = True
    feasibility_sectors: int = 9
    observation_space = 3 + 3 + 9


@configclass
class HazardOpenBasinEnvCfg(HazardForcedCrossingEnvCfg):
    """The crossing basin with the bulkhead removed -- walls and nothing else.

    Control for the zero-shot result. The v11 champion scores 0/128 with 128/128
    collisions on the crossing task at BOTH tier 0 (4.5 m gate) and tier 3
    (1.8 m gate), identically -- gate width changes nothing, and the median path
    of 4.95 m is short of the >= 6 m to the bulkhead. If it also fails here, the
    walls alone account for it and the gate contributed nothing.

    Identical perimeter distribution to the crossing task, so the bulkhead is
    the only difference between the two.
    """

    layout_mode: str = "basin"
    max_obstacles: int = 80
