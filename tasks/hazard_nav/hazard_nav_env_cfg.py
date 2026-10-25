# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the static Hazard Navigation direct task."""

from __future__ import annotations

import math as _math
from .._shared.vehicles import PhysicsVersionCfg, physics_version

import os as _os

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .._shared.obs_superset import SUPERSET_DIM, SUPERSET_DIM_V2
from .._shared.sea_state import SeaStateCfg
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


# Journal (task contract v2) vessels: D5 v2 assets under the repository's assets_v2/
# folder (origin at the COM, bow +x), SHA-256 checked against the registry
# (journal_vehicle.verify_asset). USVBENCH_ASSETS_V2 may point at a deployed copy.
_JOURNAL_ASSET_DIR_V2 = _os.environ.get(
    "USVBENCH_ASSETS_V2",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "assets_v2"),
)


def _build_journal_robot_cfg(usd_path: str, com_height_m: float) -> RigidObjectCfg:
    """Rigid v2 hull (USD-authored mass/inertia; origin = COM, so the spawn pose IS the COM)."""
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
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
            pos=(0.0, 0.0, float(com_height_m)),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        collision_group=0,
    )


@configclass
class UnderwaterPhysicsCfg(PhysicsVersionCfg):
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

    # Tier-regrade probe knobs (evaluation only; 0.0 = the frozen tier tables,
    # byte-identical). Set via eval_imbalance --axis so intermediate fortress
    # apertures and gate widths are measurable without minting tier semantics.
    fortress_aperture_override_m: float = 0.0
    gate_width_override_m: float = 0.0

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
    # Option-b perception coupling: point nav channels at a short-horizon
    # carrot on the solved route. Off preserves every existing observation.
    nav_targets_waypoint: bool = False
    waypoint_lookahead_m: float = 6.0
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

    # --- Suite D axes: scalar identity + optional per-episode pack ----------
    # BlueBoat's mass and inertia are authored on the USD rigid body. The
    # environment therefore models a common mass/inertia multiplier by scaling
    # the assembled external wrench, never by mutating the shared asset.
    mass_scale: float = 1.0
    mass_scale_choices: tuple = ()
    # Multiplier on planar surge/sway and yaw linear/quadratic drag loads.
    drag_scale: float = 1.0
    drag_scale_choices: tuple = ()
    # Multiplier on forward/reverse surge caps (yaw remains its own channel).
    thrust_cap_scale: float = 1.0
    thrust_cap_scale_choices: tuple = ()
    # First-order actuator lag. Zero bypasses filtering exactly and preserves
    # the legacy instantaneous-thrust trajectory bit for bit.
    motor_tau_s: float = 0.0
    motor_tau_s_choices: tuple = ()

    # --- Suite D axis: deck payload (point mass strapped on deck) ------------
    # The single easiest real-boat perturbation -- bolt a known weight to the
    # deck -- which makes it the anchor axis for embodied validation. Modelled
    # honestly against the 3-DOF planar wrench _apply_action integrates: a
    # POINT mass ``payload_mass_kg`` at body position (``payload_offset_x_m``
    # forward, ``payload_offset_y_m`` to port) relative to the hull's
    # USD-authored centre of mass. Exact planar consequences, all three
    # carried (hazard_nav_env.py, payload block in _apply_action):
    #   (1) total mass   m -> m + mp: every linear acceleration falls by
    #       m/(m+mp) -- thrust, drag, and wave forces alike;
    #   (2) yaw inertia  Iz -> Iz + mu*|d|^2 with mu = m*mp/(m+mp), the
    #       parallel-axis/reduced-mass term about the SHIFTED centre of mass
    #       (a centred payload leaves yaw response exactly untouched);
    #   (3) COM shift    delta = mp*d/(m+mp): every planar force, still
    #       physically applied at the hull reference point, acquires the
    #       moment N_arm = delta_y*Fx_body - delta_x*Fy_body about the new
    #       COM -- a thrust-proportional trim plus velocity-dependent drag
    #       moments (a port-side weight pulls the bow to port under thrust).
    # NOT modelled, deliberately: heel/trim attitude, added-mass change, and
    # the ~|delta| kinematic offset of the tracked reference point (5.9 cm for
    # 3 kg at 0.4 m) -- effects the planar model cannot carry honestly. The
    # roll/pitch torque channels are left alone (mass_scale scales them, this
    # axis does not: a planar point mass is not a uniform density change).
    # Heavier floats deeper, exactly as mass_scale already floats deeper.
    #
    # Why this axis is not redundant given the other five: ``mass_scale``
    # constrains the linear and angular acceleration scale factors to be
    # EQUAL, while a centred payload slows surge/sway and leaves yaw agility
    # unchanged; ``thrust_imbalance``'s yaw bias x*L*T comes chained to the
    # reciprocal surge coupling x*N/L and never touches drag, while the
    # payload trim delta_y*T carries no surge coupling and adds drag-borne
    # yaw moments (delta_y*Dx - delta_x*Dy) no existing knob can produce.
    # Numeric independence proof: test_payload_axis.py.
    #
    # Zero payload_mass_kg is INERT: the scalar path never executes the block
    # and the choices path routes drawn-zero envs through torch.where to the
    # untouched tensors, so every certified id stays byte-identical. Offsets
    # with zero mass are physically no payload and stay inert. Primary mode
    # is constant-per-run via --set (the dose axis); the choices tuple
    # samples per episode off the scenario protocol's own
    # "actuator_payload_mass_kg" stream.
    payload_mass_kg: float = 0.0
    payload_mass_kg_choices: tuple = ()
    payload_offset_x_m: float = 0.0
    payload_offset_y_m: float = 0.0
    # The wrench-division trick works in RATIOS to the USD-authored rigid
    # body, so the authored values must be named here (mass_scale never
    # needed them; independent mass/inertia ratios do). Read from
    # assets/blueboat_physics.usd /World/BlueBoat MassAPI: physics:mass
    # 17.26 kg, physics:diagonalInertia z 2.376 kg m^2. Reference values for
    # the ratio only -- the USD stays authoritative and untouched.
    payload_ref_mass_kg: float = 17.26
    payload_ref_izz_kgm2: float = 2.376

    # --- Observation-channel degradation (eval-time OOD hooks) ---------------
    # Physical-unit corruption of the IDEAL measurements (goal vector, body-
    # frame velocities, ray returns) before feature construction, inside
    # _get_observations only; reward/termination/success keep reading clean
    # state. All zero => no degrader object is built and every certified id
    # runs its frozen observation path byte-identical. Eval-time knobs (set
    # via eval_v6_frozen --set); no gym id may ship nonzero defaults without
    # its own registration. Yaw-rate sigma rides the vel knob scaled by
    # yaw_rate_obs_scale_rad_s / SPEED_SCALE_MPS (same fraction of full
    # scale as the linear channels).
    obs_noise_sigma_pos: float = 0.0  # m, white noise on the goal vector
    obs_noise_sigma_vel: float = 0.0  # m/s, white noise on surge/sway (+ yaw)
    obs_noise_sigma_ray: float = 0.0  # m, white noise per ray return
    obs_bias_sigma_pos: float = 0.0   # m, per-episode constant offset draw
    obs_bias_sigma_vel: float = 0.0   # m/s, per-episode constant offset draw
    obs_bias_sigma_ray: float = 0.0   # m, per-episode per-ray offset draw
    obs_delay_steps: int = 0          # control steps of measurement latency
    obs_dropout_p: float = 0.0        # per-step frame loss (sample-and-hold)

    # --- Mid-episode obstacle appearance (sudden terrain change) -------------
    # At a per-episode time t0 -- drawn from the scenario protocol's own
    # "appearance" group, so eval seeds control it -- appear_count EXTRA
    # cylinders flip from inactive to active in obstacle_active. The buffers
    # are already read every step, so nothing structural changes: the extra
    # slots are simply reserved (left inactive) at reset and flipped at t0.
    #
    # FAIRNESS RANGE (corrected 2026-08-31, derivation at hazard_geometry.py
    # plan_obstacle_appearance): the appearance azimuth is aligned to one of
    # the 36 sensor rays at the moment of appearance, and the centre distance
    # from the CURRENT boat position must lie in
    # [appear_min_distance_m(radius), 25] m -- delivered EXACTLY, raising
    # outside. The floor is BRAKING, not ray visibility: collision geometry
    # (radius + half beam) plus the 0.868 m a full-reverse stop carries the
    # hull, i.e. 2.118 m at the minimum radius. The old note here claimed a
    # 9.2 m visibility floor with braking non-binding; that had the direction
    # backwards (9.18 m is an UPPER bound, and ray alignment already
    # guarantees the hit), and the clamp under it silently capped this axis at
    # its lightest usable rung. Admission reruns the layout
    # sampler's own check (BFS feasibility at planning inflation) on the
    # post-appearance field; an inadmissible azimuth is resampled (bounded by
    # appear_max_attempts distinct rays), and exhaustion SKIPS the appearance
    # for that env and records the skip.
    #
    # appear_count = 0 keeps every certified id bit-identical: no buffer is
    # written, no stream is consumed, no step-path branch is taken.
    # appear_distance_m is the DOSE axis (far = light, near = heavy), pinned
    # per run via eval --set on the appearance gym id, never by minting
    # per-rung ids. The 2026-08-28 ladder 20 / 15 / 12 / 9.5 m proved inert;
    # the redesigned ladder is stated as stopping margin
    # R = (d - r - half_beam) / 0.868, R = 9.50 / 6.00 / 3.50 / 2.20 / 1.50 /
    # 1.15, i.e. d = 9.5 / 6.46 / 4.29 / 3.16 / 2.55 / 2.25 m at r = 0.8.
    # R = 1 is the floor, where the hull stops exactly on the obstacle.
    # NOTE: the near rungs are only meaningful once the appearance azimuth is
    # drawn along the instantaneous velocity direction rather than uniformly
    # over all 36 rays; see hazard_nav_env._maybe_appear_obstacles.
    appear_count: int = 0
    # Physics v2 reserves this many EXTRA slots in obstacle_slot_count.
    # max_obstacles remains the static-layout budget (14 at scatter level 3).
    # Legacy physics continues sharing its existing max_obstacles slots.
    appear_time_min_s: float = 5.0
    appear_time_max_s: float = 30.0
    appear_distance_m: float = 20.0
    # 0.8 m is MIN_OBSTACLE_RADIUS_M; 2.0 is MAX_OBSTACLE_RADIUS_M. The
    # distance floor scales with this radius via appear_min_distance_m.
    appear_radius_m: float = 0.8
    # Admission BFS resolution. 0.25 m is FORCED_SEAL_CELL_M -- the value the
    # crossing sampler itself audits with -- and resolves every tier gate.
    appear_bfs_cell_m: float = 0.25
    appear_max_attempts: int = 36
    # Added 2026-08-31 (master TODO section 8.3 items 3 and 4).  BOTH default
    # to the shipped behaviour, so every certified id and every certificate
    # already on the books is untouched; the redesigned near ladder turns them
    # on with --set.
    #
    # appear_velocity_aligned: order the candidate bearings by angular
    # distance from the instantaneous direction of travel instead of drawing a
    # uniform permutation of all rays.  Mandatory below the 9.18 m ray-
    # visibility figure -- plan_obstacle_appearance RAISES without it there,
    # because a cylinder that close in a random bearing can sit astern and the
    # rung would look easy for a reason that has nothing to do with the dose.
    #
    # appear_strict_ray: try ONLY the aligned bearing.  If it is inadmissible
    # the episode records a skip instead of quietly moving to another azimuth.
    # This is what makes a ladder paired: nothing before t0 depends on the
    # appearance distance, so the aligned ray is the same at every rung, and a
    # rung either uses that exact ray or declares that it did not run.
    appear_velocity_aligned: bool = False
    appear_strict_ray: bool = False
    # Added 2026-08-31. Third alignment mode, kept ALONGSIDE the velocity one
    # rather than replacing it, because the comparison is the point: same
    # distance, same archive, only the bearing rule changes, which prices the
    # bearing choice on its own.
    #
    # appear_route_aligned puts the cylinder on the ray closest to the
    # direction the TASK still requires -- the geodesic route to the goal at
    # planning inflation -- instead of the direction the boat happens to be
    # travelling. On a gated wall whose opening is offset from the goal
    # bearing the two differ by about 31.6 degrees, i.e. roughly three rays.
    # An obstacle on the velocity vector can be stepped around by changing
    # course; one on the route has to be dealt with.
    #
    # Takes precedence over appear_velocity_aligned when both are set. The
    # route is computed from the geometry the planner already holds, not from
    # the env's descent map, because that map only exists when
    # nav_targets_waypoint is set and the crossing family does not set it.
    appear_route_aligned: bool = False
    appear_route_lookahead_m: float = 8.0
    # Added 2026-08-31 after an audit found that the env never passed a
    # stopping distance at all, so the fairness floor silently used the
    # NOMINAL hull figure even on ids that perturb braking.  That is the
    # silent-clamp defect re-entering through the argument list.
    #
    # The floor is radius + half beam + this.  0.868 m is the nominal
    # full-reverse figure; with the motor-lag knob on it becomes 1.163 m at
    # tau = 0.10 s and 1.561 m at tau = 0.25 s, which lift the r = 0.8 floor
    # from 2.118 m to 2.413 m and 2.811 m.  There is no fitted formula here
    # on purpose: only those three points were integrated against the repo's
    # own dynamics, so anything between them would be invented.  Set it
    # explicitly when a braking-relevant knob is on; the env refuses a near
    # rung rather than guess (see _reset_appearance).
    appear_stop_distance_m: float = 0.868

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

    # Opt-in condition layer; inherited legacy registrations remain disabled.
    conditions: dict | None = None
    condition_presets: tuple[str, ...] = ()

    @property
    def obstacle_slot_count(self) -> int:
        """Allocate appearance capacity without changing the static layout."""
        static_slots = int(self.max_obstacles)
        if getattr(self.underwater_physics_cfg, "physics_version", 1) == 2:
            # G3: a journal cfg whose resolved condition stack names the appearing
            # suite gets the appearing slot (D7 CapacityPlan.appearing_slots); the
            # legacy timed appearance (appear_count) stays 0 on journal cfgs.
            appearing = int(
                getattr(self, "journal_contract_version", 1) == 2
                and bool(getattr(getattr(self, "journal_conditions", None), "appearing_enabled", False))
            )
            return static_slots + int(self.appear_count) + appearing
        return static_slots

    @obstacle_slot_count.setter
    def obstacle_slot_count(self, value) -> None:
        # G6 (Isaac Lab 2.3.2 configclass.py:392-402): the configclass post-init re-assigns
        # every non-callable member listed by dir(obj) and recognises a property only in the
        # instance's OWN class dict (configclass.py:399-401), so on any subclass it assigns
        # this derived value back to itself. Accept exactly that re-assignment; anything
        # else stays an error, as for the read-only property before.
        if value != self.obstacle_slot_count:
            raise AttributeError(
                "obstacle_slot_count is derived from max_obstacles, appear_count and the "
                f"appearing suite ({self.obstacle_slot_count}); it cannot be set to {value!r}"
            )

    def __post_init__(self) -> None:
        base_post_init = getattr(super(), "__post_init__", None)
        if base_post_init is not None:
            base_post_init()

        if getattr(self, "journal_contract_version", 1) == 2:
            # Journal task contract v2 (E5 templates): v2 registry vessel, v2
            # asset, hydro_v2, the plan's ray sensor (ANCHORS_E5 C6/C8, F1).
            self._journal_post_init_family()
            return

        spec = get_vehicle(self.vehicle)
        self.robot_cfg = _build_robot_cfg(spec)
        version = physics_version(self.underwater_physics_cfg)
        self.underwater_physics_cfg = _build_underwater_physics_cfg(spec)
        self.underwater_physics_cfg.physics_version = version
        self.thrust_max_fwd = spec.thrust_fwd_n
        self.thrust_max_rev = spec.thrust_rev_n
        self.yaw_torque_max = spec.yaw_torque_nm

        self.validate_after_overrides()

    def validate_after_overrides(self) -> None:
        """Raise when the fully assembled cfg violates a task invariant.

        This method is deliberately pure: override callers can invoke it after
        installing a complete batch without rebuilding or normalizing the cfg.
        """

        def finite(field, value):
            if isinstance(value, bool):
                raise TypeError(f"{field}={value!r} must be a finite number")
            try:
                is_finite = _math.isfinite(value)
            except TypeError:
                raise TypeError(
                    f"{field}={value!r} must be a finite number"
                ) from None
            if not is_finite:
                raise ValueError(f"{field}={value!r} must be finite")

        def integer(field, value):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field}={value!r} must be an integer")

        def actuator_axis(
            field, value, choices_field, choices, *, positive=False, max_abs=None
        ):
            finite(field, value)
            if max_abs is not None:
                if abs(value) > max_abs:
                    raise ValueError(
                        f"{field}={value!r} must lie in [-{max_abs}, {max_abs}]"
                    )
            elif positive:
                if value <= 0.0:
                    raise ValueError(f"{field}={value!r} must be positive")
            elif value < 0.0:
                raise ValueError(f"{field}={value!r} must be non-negative")
            if not isinstance(choices, tuple):
                raise TypeError(f"{choices_field}={choices!r} must be a tuple")
            for index, choice in enumerate(choices):
                choice_field = f"{choices_field}[{index}]"
                finite(choice_field, choice)
                if max_abs is not None:
                    if abs(choice) > max_abs:
                        raise ValueError(
                            f"{choice_field}={choice!r} must lie in "
                            f"[-{max_abs}, {max_abs}]"
                        )
                elif positive:
                    if choice <= 0.0:
                        raise ValueError(
                            f"{choice_field}={choice!r} must be positive"
                        )
                elif choice < 0.0:
                    raise ValueError(
                        f"{choice_field}={choice!r} must be non-negative"
                    )

        if self.obs_v2 and self.obs_v3:
            raise ValueError(
                f"obs_v2={self.obs_v2!r} and obs_v3={self.obs_v3!r} "
                "are mutually exclusive"
            )
        integer("superset_version", self.superset_version)
        if self.superset_version not in (1, 2):
            raise ValueError(
                f"superset_version={self.superset_version!r} must be 1 or 2"
            )
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
                f"ray_count={self.ray_count!r} and "
                f"observation_space={self.observation_space!r} are invalid; "
                "HazardNav requires ray_count=36 and observation_space="
                f"{expected_observation_space} when "
                f"emit_superset_obs={self.emit_superset_obs}, "
                f"superset_version={self.superset_version}, "
                f"obs_v2={self.obs_v2}, obs_v3={self.obs_v3}"
            )
        finite("yaw_rate_obs_scale_rad_s", self.yaw_rate_obs_scale_rad_s)
        if self.yaw_rate_obs_scale_rad_s <= 0.0:
            raise ValueError(
                "yaw_rate_obs_scale_rad_s="
                f"{self.yaw_rate_obs_scale_rad_s!r} must be positive"
            )
        journal = getattr(self, "journal_contract_version", 1) == 2
        if journal:
            self._journal_validate_family()
        if not journal and self.layout_mode not in (
            "scatter",
            "ring",
            "ring2",
            "fortress",
            "fortress2",
            "bandfort",
            "forced",
            "basin",
            "iceberg",
            "moving_basin",
        ):
            raise ValueError(
                f"layout_mode={self.layout_mode!r} must be 'scatter', 'ring', "
                "'ring2', 'fortress', "
                "'fortress2', 'bandfort', 'forced', 'basin', 'iceberg' or "
                "'moving_basin'"
            )
        if self.reward_progress_geodesic and not journal and self.layout_mode not in (
            "bandfort",
            "fortress",
            "fortress2",
        ):
            raise ValueError(
                f"reward_progress_geodesic={self.reward_progress_geodesic!r} "
                f"is invalid with layout_mode={self.layout_mode!r}; geodesic "
                "progress requires 'bandfort', 'fortress' or 'fortress2'"
            )
        if self.nav_targets_waypoint and not self.reward_progress_geodesic:
            raise ValueError(
                f"nav_targets_waypoint={self.nav_targets_waypoint!r} requires "
                "reward_progress_geodesic=True, got "
                f"{self.reward_progress_geodesic!r}"
            )
        finite("waypoint_lookahead_m", self.waypoint_lookahead_m)
        if self.waypoint_lookahead_m <= 0.0:
            raise ValueError(
                f"waypoint_lookahead_m={self.waypoint_lookahead_m!r} "
                "must be positive"
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
            "iceberg": 3,
            # D7 N1 / C1 finding 42: the moving basin sampler's hard maximum
            # (D4/E5 resolved maximum, journal_cfg_v2.STATIC_CAPACITY).
            "moving_basin": 128,
        }.get(self.layout_mode, 14)
        if journal:
            # E5 static capacity + the D7 reserved slot (D7 B04/N1).
            min_obstacles = self._journal_min_obstacles()
        integer("max_obstacles", self.max_obstacles)
        if self.max_obstacles < min_obstacles:
            raise ValueError(
                f"max_obstacles={self.max_obstacles!r} must be >= "
                f"{min_obstacles} for "
                f"layout_mode={self.layout_mode}"
            )
        finite("min_goal_distance_m", self.min_goal_distance_m)
        finite("max_goal_distance_m", self.max_goal_distance_m)
        if self.min_goal_distance_m != 20.0 or self.max_goal_distance_m != 40.0:
            raise ValueError(
                f"min_goal_distance_m={self.min_goal_distance_m!r} and "
                f"max_goal_distance_m={self.max_goal_distance_m!r} are "
                "invalid; HazardNav v1 fixes D0 sampling to U[20, 40] m"
            )
        finite("reward_goal_entry_bonus", self.reward_goal_entry_bonus)
        finite("reward_goal_time_bonus", self.reward_goal_time_bonus)
        if self.reward_goal_entry_bonus < 0.0 or self.reward_goal_time_bonus < 0.0:
            raise ValueError(
                f"reward_goal_entry_bonus={self.reward_goal_entry_bonus!r} and "
                f"reward_goal_time_bonus={self.reward_goal_time_bonus!r} "
                "must be non-negative"
            )
        finite("reward_reverse_action_scale", self.reward_reverse_action_scale)
        if self.reward_reverse_action_scale < 0.0:
            raise ValueError(
                "reward_reverse_action_scale="
                f"{self.reward_reverse_action_scale!r} must be non-negative"
            )
        finite("contact_dwell_tau_s", self.contact_dwell_tau_s)
        if self.contact_dwell_tau_s <= 0.0:
            raise ValueError(
                f"contact_dwell_tau_s={self.contact_dwell_tau_s!r} "
                "must be positive"
            )
        integer("appear_count", self.appear_count)
        if self.appear_count < 0:
            raise ValueError(
                f"appear_count={self.appear_count!r} must be non-negative"
            )
        if self.appear_count > 0:
            finite("appear_time_min_s", self.appear_time_min_s)
            finite("appear_time_max_s", self.appear_time_max_s)
            if (
                self.appear_time_min_s < 0.0
                or self.appear_time_max_s < self.appear_time_min_s
            ):
                raise ValueError(
                    f"appear_time_min_s={self.appear_time_min_s!r} and "
                    f"appear_time_max_s={self.appear_time_max_s!r} must "
                    "satisfy 0 <= min <= max"
                )
            # Radius bounds are the sampler's own admitted range
            # (hazard_geometry.MIN/MAX_OBSTACLE_RADIUS_M). The distance floor
            # is computed from this radius, so a larger cylinder simply gets a
            # larger floor rather than a separate rule.
            finite("appear_distance_m", self.appear_distance_m)
            finite("appear_radius_m", self.appear_radius_m)
            if not 0.8 <= self.appear_radius_m <= 2.0:
                raise ValueError(
                    f"appear_radius_m={self.appear_radius_m!r} must lie in "
                    "[0.8, 2.0] m"
                )
            finite("appear_bfs_cell_m", self.appear_bfs_cell_m)
            if self.appear_bfs_cell_m <= 0.0:
                raise ValueError(
                    f"appear_bfs_cell_m={self.appear_bfs_cell_m!r} "
                    "must be positive"
                )
            integer("appear_max_attempts", self.appear_max_attempts)
            if self.appear_max_attempts < 1:
                raise ValueError(
                    f"appear_max_attempts={self.appear_max_attempts!r} "
                    "must be at least one"
                )
            finite("appear_stop_distance_m", self.appear_stop_distance_m)
            if self.appear_stop_distance_m < 0.0:
                raise ValueError(
                    f"appear_stop_distance_m={self.appear_stop_distance_m!r} "
                    "must be non-negative"
                )

        actuator_axis(
            "mass_scale", self.mass_scale,
            "mass_scale_choices", self.mass_scale_choices,
            positive=True,
        )
        actuator_axis(
            "drag_scale", self.drag_scale,
            "drag_scale_choices", self.drag_scale_choices,
        )
        actuator_axis(
            "thrust_cap_scale", self.thrust_cap_scale,
            "thrust_cap_scale_choices", self.thrust_cap_scale_choices,
        )
        actuator_axis(
            "motor_tau_s", self.motor_tau_s,
            "motor_tau_s_choices", self.motor_tau_s_choices,
        )
        actuator_axis(
            "payload_mass_kg", self.payload_mass_kg,
            "payload_mass_kg_choices", self.payload_mass_kg_choices,
        )
        actuator_axis(
            "thrust_imbalance", self.thrust_imbalance,
            "thrust_imbalance_choices", self.thrust_imbalance_choices,
            max_abs=1.0,
        )

    # ---- journal task contract v2 (F1; E5 templates; ANCHORS_E5 C6/C8) ------
    def _journal_plan_for_cfg(self):
        from .._shared import journal_cfg_v2

        plan = getattr(self, "journal_plan", None)
        if plan is None:
            plan = journal_cfg_v2.plan_class(self.journal_task, self.vehicle)(
                int(getattr(self, "journal_geometry_tier", 2))
            )
            self.journal_plan = plan
        return plan

    def _journal_post_init_family(self) -> None:
        """Parent post-init of a journal cfg (vessel chosen as a class default).

        The v2 registry vessel is resolved here, BEFORE the env builds its
        asset: the legacy get_vehicle registry is never consulted (legacy
        names are rejected with their reason). Physics = hydro_v2 on
        physics_version 2; the legacy UnderwaterPhysicsCfg hull numbers are
        unused (the HydroV2Runtime profile is authoritative). The legacy
        lumped thrust limits are NaN so no journal path can read them.
        Reward-shaping lengths scale with the vessel length (s = L / L_ref).
        """
        from .._shared.journal_cfg_v2 import vessel_record
        from .._shared.journal_vehicle import resolve_journal_vehicle, verify_asset

        plan = self._journal_plan_for_cfg()
        plan.assert_vehicle_override(self.vehicle)
        options = dict(getattr(self, "journal_vehicle_options", None) or {})
        vehicle = resolve_journal_vehicle(self.vehicle, **options)
        record = vessel_record(self.vehicle)
        asset = verify_asset(vehicle, _JOURNAL_ASSET_DIR_V2)
        self.robot_cfg = _build_journal_robot_cfg(
            str(asset), vehicle.calm_com_height_above_water_m
        )
        version = physics_version(self.underwater_physics_cfg)
        self.underwater_physics_cfg = UnderwaterPhysicsCfg()
        self.underwater_physics_cfg.physics_version = version
        self.underwater_physics_cfg.hydrodynamics_model = "hydro_v2"
        self.thrust_max_fwd = float("nan")
        self.thrust_max_rev = float("nan")
        self.yaw_torque_max = float("nan")
        params = plan.params
        scale = float(params["length_scale"])
        self.ray_count = int(params["ray_sensor_ray_count"])
        self.ray_max_range_m = float(params["ray_sensor_max_range_m"])
        self.goal_radius = float(params["goal_radius_m"])
        self.half_beam_m = float(record.geometry.lateral_half_extent_m)
        self.prox_ray_floor_m = self.half_beam_m
        self.safe_clearance_m = 0.90 * scale
        self.collision_margin_m = 0.20 * scale
        self.use_official_blueboat_visual = False
        self.vessel_profile = vehicle.name
        self.vessel_profile_status = vehicle.profile_status
        self.observation_contract = "journal_v3_ray36"
        self.journal_observation_hooks_wired = True
        self.appear_count = 0
        from . import journal_hazard_nav

        self.max_obstacles = journal_hazard_nav.storage_width(self.journal_task)
        self.validate_after_overrides()

    def _journal_min_obstacles(self) -> int:
        from . import journal_hazard_nav

        return journal_hazard_nav.storage_width(self.journal_task)

    def _journal_validate_family(self) -> None:
        """Journal invariants of the HazardNav parent (pure; re-run after overrides)."""
        from . import journal_hazard_nav

        task = getattr(self, "journal_task", None)
        if task not in journal_hazard_nav.CARRIED_TASKS:
            raise ValueError(
                f"journal_task={task!r} is not carried by HazardNav "
                f"({journal_hazard_nav.CARRIED_TASKS})"
            )
        if self.layout_mode not in ("journal", "scatter"):
            # "journal" is the only runtime value (HazardNavEnv._journal_init
            # checks it); "scatter" is tolerated here so F0's stacking harness
            # (scripts/tests_journal_glue, legacy checks run as scatter) stays valid.
            raise ValueError(f"journal cfgs use layout_mode='journal', got {self.layout_mode!r}")
        if not self.obs_v3 or self.obs_v2 or self.emit_superset_obs or self.obs_feasibility:
            raise ValueError(
                "journal HazardNav cfgs use the fixed journal_v3_ray36 observation "
                "(obs_v3, no superset, no pooled ranges)"
            )
        if self.nav_targets_waypoint:
            raise ValueError("journal observation contract has no waypoint carrot")
        plan = self._journal_plan_for_cfg()
        params = plan.params
        if int(self.ray_count) != int(params["ray_sensor_ray_count"]) or not _math.isclose(
            float(self.ray_max_range_m), float(params["ray_sensor_max_range_m"]), rel_tol=1e-12
        ):
            raise ValueError(
                "journal ray sensor must be the task plan's 36 rays x 30 m x length scale (D14)"
            )
        if int(getattr(self, "appear_count", 0)) != 0:
            raise ValueError("legacy appearance is off in journal cfgs; D7 owns dynamic hazards")


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
class HazardIcebergEnvCfg(HazardNavV3EnvCfg):
    """Few huge cylinders in open water; success requires an early detour."""

    layout_mode: str = "iceberg"
    # ICEBERG_TIERS tops out at three cylinders; three spare slots are reset
    # buffer headroom. The validator's hard floor is the tier maximum, three.
    max_obstacles: int = 6
    layout_max_attempts: int = 60
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
class HazardBandFortWayEnvCfg(HazardBandFortGeoEnvCfg):
    """Geodesic band fortress exposing a moving route waypoint."""

    nav_targets_waypoint: bool = True
    waypoint_lookahead_m: float = 6.0


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
class HazardCrossAppearEnvCfg(HazardForcedCrossingEnvCfg):
    """Forced crossing + one mid-episode obstacle appearance (sudden terrain
    change).

    Parent: the certified crossing cfg -- observation, reward, termination,
    layout distribution and tier semantics are inherited byte for byte, so a
    certified crossing checkpoint loads here zero-shot and the ONLY new event
    in an episode is the appearance. One extra cylinder of the minimum admitted
    radius (0.8 m) flips active at a per-episode time t0 ~ U[appear_time_min_s,
    appear_time_max_s] drawn from the scenario protocol's own "appearance"
    group; the fairness floor (ray-aligned azimuth, [9.2, 25] m distance clamp)
    and the sampler-grade BFS admission are documented at the appear_* fields
    on the base cfg and in hazard_geometry.plan_obstacle_appearance.

    DOSE AXIS = appearance distance (far = light, near = heavy). The frozen
    rung ladder is 20 / 15 / 12 / 9.5 m, pinned per run via
    ``eval --set appear_distance_m=<rung>`` on this id -- the same discipline
    as the observation-degradation doses, so intermediate rungs never mint
    ids. The default (20 m) is the lightest rung.

    The time window default [5, 30] s brackets the certified crossing
    champions' median time-to-success (18-21 s), so the appearance lands
    mid-transit for a competent policy rather than after the episode is
    decided; both endpoints stay --set-able for probes.

    Slot budget: the parent reserves max_obstacles = 96 against the 10k-layout
    audit's worst case of 79 wall cylinders, so the one appearance slot rides
    the existing 17-slot headroom; the env still hard-fails if a slot is ever
    unavailable rather than silently skipping.
    """

    appear_count: int = 1


@configclass
class HazardMovingBasinEnvCfg(HazardNavV3EnvCfg):
    """P5 layer three: a closed basin whose obstacles MOVE.

    Three segments along the track, in order: two staggered rows of cylinders
    sliding sideways, ten metres of empty water so the two mechanics can be
    attributed apart, then one cylinder orbiting a fixed centre, then a clear
    run-out to the goal.

    Parent: ``HazardNavV3EnvCfg``, so observation_space stays 42 and every
    certified 42-dimensional archive loads here zero-shot.  That is a hard
    constraint, not a convenience: the moment the observation width changes,
    this task loses its connection to every existing result.  The moving
    cylinders are seen through the existing 36-ray sensor and add no channel.

    DOSE AXIS.  Both segments express their dose as a fraction of the boat's
    own top speed, so a rung means the same physical demand in either:

        moving_mu = u / v_max            row translation speed
        moving_nu = omega * R / v_max    orbit tangential speed

    The frozen ladder is 0 / 0.12 / 0.24 / 0.36 / 0.55 / 0.80, pinned per run
    via ``eval --set moving_mu=<rung> --set moving_nu=<rung>``, never per-rung
    ids.  Rung 0 is the parent baseline and the sampler marks every cylinder
    static there, so nothing in the step path moves and the cell is a genuine
    undosed control.

    WHY 0.36 IS IN THE LADDER.  Below mu* = 0.360 a boat entering a gap dead
    centre clears the row without any lateral tracking; above it the gap
    outruns that margin and the boat must chase it sideways while still making
    headway.  The ladder deliberately straddles that transition, because the
    appearance axis spent two days measuring a single regime and reporting it
    as saturation.  The derivation and its executable form live in
    hazard_geometry.moving_transition_mu and test_moving_basin.py.

    WHY 0.80 IS THE TOP.  At mu = 1 the gap moves at the boat's top speed and
    no lateral tracking can exist, so the rung would be unsolvable rather than
    hard.  At 0.80 a constant-heading pass still exists but its window is about
    four tenths of a second; the sampler refuses anything it cannot construct a
    pass for.

    ZERO-SHOT CAVEAT, to be written into any report.  No existing archive has
    ever seen a moving obstacle, so running a certified champion here is an
    out-of-distribution generalisation test, not a capability test.  A low
    score is not evidence that the task is unfair.

    Slot budget: the sampler builds 94 to 100 cylinders (basin walls plus two
    combs plus the orbiter) and refuses above 128; the cap here leaves headroom
    on top of that so the appearance axis could be layered on later without
    touching this number.  The design deliberately does NOT layer them yet, so
    that a result can be attributed to movement alone.
    """

    layout_mode: str = "moving_basin"
    max_obstacles: int = 136

    # Dose. Both default to zero, which is the undosed parent.
    moving_mu: float = 0.0
    moving_nu: float = 0.0

    # Rows translate rigidly with a fixed half-period stagger by default, which
    # is what forces a re-aim between them. See MOVING_ANTI_PHASE_ROWS in
    # hazard_geometry.py for why the literal reading of the design note does
    # not produce that property.
    moving_anti_phase: bool = False

    # Probe hook, same discipline as gate_width_override_m on the crossing cfg.
    moving_gap_width_override_m: float = 0.0


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


# Append-only Suite D v3 training carriers. Each changes exactly one dynamics
# axis from HazardForcedCrossingEnvCfg and samples the frozen train pack.
@configclass
class HazardCrossMassEnvCfg(HazardForcedCrossingEnvCfg):
    """Forced crossing trained on the frozen Suite D mass pack."""

    mass_scale_choices: tuple = (0.80, 1.00, 1.30)


@configclass
class HazardCrossDragEnvCfg(HazardForcedCrossingEnvCfg):
    """Forced crossing trained on the frozen Suite D drag pack."""

    drag_scale_choices: tuple = (0.75, 1.00, 1.50)


@configclass
class HazardCrossThrustEnvCfg(HazardForcedCrossingEnvCfg):
    """Forced crossing trained on the frozen Suite D thrust-cap pack."""

    thrust_cap_scale_choices: tuple = (0.75, 0.875, 1.00)


@configclass
class HazardCrossTauEnvCfg(HazardForcedCrossingEnvCfg):
    """Forced crossing trained on the frozen Suite D motor-lag pack."""

    motor_tau_s_choices: tuple = (0.00, 0.10, 0.25)


# Clean discount-coupling control. These inherit the same carrier and differ
# only in the one flag named by the experiment.
@configclass
class HazardPbrsNoGammaEnvCfg(HazardForcedCrossingEnvCfg):
    """Forced crossing with the legacy missing-gamma potential difference."""

    pbrs_correct: bool = False


@configclass
class HazardPbrsGammaEnvCfg(HazardPbrsNoGammaEnvCfg):
    """Forced crossing with gamma*Phi(s') - Phi(s)."""

    pbrs_correct: bool = True


@configclass
class HazardBandFortWayTaxEnvCfg(HazardBandFortWayEnvCfg):
    """BandFortWay + the open-water loiter tax.

    The 08-21 forensics identified the fortress collapse mechanism: the water
    OUTSIDE the fortress is a zero-gradient, zero-cost absorbing region, so
    retreating and loitering dodges the -25 contact penalty for free and every
    from-scratch seed converges to it. This variant prices the loiter with the
    settings the healthy open-water-tax family already measured; everything
    else is inherited unchanged, so bfway-vs-bfwaytax isolates the tax.
    """

    reward_open_water_scale: float = 2.0
    open_water_radius_m: float = 12.0


# --- Suite S: frozen structural-generalization layouts (append-only) --------
@configclass
class HazardSuiteSEnvCfg(HazardNavV3EnvCfg):
    """Frozen Suite S structural layouts under the certified scatter contract.

    Same hull, sensors, reward, and tier semantics as the HazardNavV3EnvCfg
    family the certified crossing / v9 / v12 champions trained on (42-D
    native obs: nav 3 + kinematics 3 + 36 rays), so any of those checkpoints
    loads here unchanged. Only the reset-time layout source differs: instead
    of an i.i.d. scatter draw, each episode places one of the ten frozen,
    checksum-verified public layouts of ``suite_s_class`` from
    ``assets/suite_s`` (admission-audited at generation; the JSON is ground
    truth). ``route_geodesic_length`` is latched from the asset's audited
    ``geodesic_length_m``, so USV-10K route scoring runs on basis=geo.

    ``suite_s_index_rotation`` staggers layout indices by env and advances by
    one per completed episode -- ``(env + episode) % 10`` -- so ten
    consecutive episodes of one env cover all ten layouts and a 128-episode
    certification at 64 envs spreads 12-14 episodes onto every layout; off
    pins each env to ``env % 10``. The layout index is a pure function of
    ``(env, per-env episode count)``, giving exact episode-for-episode layout
    pairing across checkpoints certified at the same env count.
    """

    # Which frozen class this id serves; the four subclasses below override
    # it. Validated against SUITE_S_CLASSES when the layouts are loaded.
    suite_s_class: str = "single_row"
    suite_s_index_rotation: bool = True
    # Suite S assets are frozen at the level-2 gap width (3.0 hull beams);
    # other levels have no committed assets and fail loudly at load time.
    suite_s_level: int = 2
    # Empty resolves to the committed in-repo assets/suite_s directory.
    suite_s_asset_dir: str = ""
    # gap_wall's worst frozen instance uses 17 cylinders (measured over the
    # 50 assets; scatter's base cap of 14 would truncate it). 24 leaves seven
    # reset-buffer slots of headroom without touching any base cfg.
    max_obstacles: int = 24


@configclass
class HazardSuiteSStaggeredRowsEnvCfg(HazardSuiteSEnvCfg):
    """Suite S: two brick-staggered rows; every visible gap is backed."""

    suite_s_class: str = "staggered_rows"


@configclass
class HazardSuiteSDiagonalRowEnvCfg(HazardSuiteSEnvCfg):
    """Suite S: picket line tilted 25-35 degrees off route-perpendicular."""

    suite_s_class: str = "diagonal_row"


@configclass
class HazardSuiteSClustersEnvCfg(HazardSuiteSEnvCfg):
    """Suite S: three lumpy merged super-obstacles with channels between."""

    suite_s_class: str = "clusters"


@configclass
class HazardSuiteSGapWallEnvCfg(HazardSuiteSEnvCfg):
    """Suite S: overlap-sealed wall with one off-axis tier-width gate."""

    suite_s_class: str = "gap_wall"


@configclass
class HazardForcedCrossingWaveEnvCfg(HazardForcedCrossingEnvCfg):
    """Forced crossing x waves: the certified exam, sailed in an irregular sea.

    BACKGROUND DISTURBANCE ONLY. Unlike the station-keeping Wave id (which
    appends 3 sea-state observation channels), this variant changes NOTHING
    about the observation, reward, termination, or layout of its parent
    (Isaac-USV-HazardCross-Direct-v1): the JONSWAP field enters through the
    same world-frame force path as the certified current, and the policy is
    never told the sea exists. A certified crossing checkpoint therefore
    loads zero-shot and the comparison is "same policy, same observation,
    world now has waves".

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


@configclass
class HazardIcebergWaveEnvCfg(HazardIcebergEnvCfg):
    """Iceberg x waves: the detour exam in an irregular sea.

    BACKGROUND DISTURBANCE ONLY -- same discipline as
    HazardForcedCrossingWaveEnvCfg above: identical 64-D contract-mode
    observation, reward, and layout as Isaac-USV-Iceberg-Direct-v1; the sea
    enters only as a world-frame force, so the certified iceberg checkpoint
    loads zero-shot. The frozen evaluation box (H_s 0.30-0.60 m, T_p
    2.0-2.5 s), its steepness/capture admission, and the DUAL CALM REFERENCE
    note (enable=True with Hs -> 0 is NOT calm water) are documented on the
    crossing variant above.
    """

    sea_state: SeaStateCfg = SeaStateCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sea_state.enable = True
        self.sea_state.hs_range = (0.30, 0.60)
        self.sea_state.tp_range = (2.0, 2.5)


@configclass
class HazardNavV3JBaseCalmEnvCfg(HazardNavV3EnvCfg):
    """J-base calm helper (renamed D10; not a journal id): scatter task, BlueBoat, calm water, physics v2.

    NOT registered (proposed ids in the D1 MERGE_NOTES.md). Same v3
    observation contract (42 channels) as HazardNav-Direct-v3; the condition
    layer adds NO sea, current or wind observation channel. Every suite
    defaults off; conditions are selected as data and validated by
    ``build_conditions`` in journal mode (legacy wave/current/actuator/
    perception/appearance controls must stay off).
    """

    journal_mode: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.underwater_physics_cfg.physics_version = 2
        # physics_version changes obstacle_slot_count; re-run the pure check.
        self.validate_after_overrides()


@configclass
class HazardForcedCrossingJBaseCalmEnvCfg(HazardForcedCrossingEnvCfg):
    """J-base calm helper (renamed D10; not a journal id): forced crossing, BlueBoat, calm water, physics v2.

    NOT registered (proposed ids in the D1 MERGE_NOTES.md). Same observation
    contract and reward as HazardForcedCrossing; no sea, current or wind
    observation channel; every suite defaults off.
    """

    journal_mode: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.underwater_physics_cfg.physics_version = 2
        self.validate_after_overrides()
