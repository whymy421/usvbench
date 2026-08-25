"""USVBench MPPI baseline evaluator (eval_v6_frozen-compatible records).

INFORMATION-DIET DISCLOSURE (same as classical_planner_pid.py, published with
every number): the controller reads PRIVILEGED ground truth from the env --
boat pose/velocities, the goal / gate chain / hold point, and the per-episode
obstacle centers+radii list. It pins the "model-based control with a perfect
map and a NOMINAL dynamics model" floor next to learned observation-only
policies; it must never be presented as observation-only.

The internal model (scripts/mppi_controller.py) is frozen nominal BlueBoat:
it does NOT track Suite D parameter shifts (mass/drag/thrust scale, motor
tau, thrust imbalance). Runs where the env cfg carries a non-inert shift are
disclosed on stdout but the model stays nominal -- that mismatch is the
experiment.

The model is also CALM WATER: it has no current term and no wave term. The
current and the sea state do not live on the top-level cfg, so they are
disclosed by a separate pass (_water_motion_notes) on the same stdout
channel -- Dock-BlueBoat-Current runs the controller through a 1.0-1.5 m/s
crossing current it was never told about, and that is now said out loud
rather than silently allowed or silently refused.

Supported task families (auto-detected from privileged attributes):
  gates   PathFollow family    (env.waypoints + env.gates_passed; the cost
                                follows the env's own ordered gate sequence)
  hold    StationKeep family   (env.hold_point; goal-hold cost)
  goal    HazardNav family     (env.target_pos; obstacle cost active when the
                                env publishes obstacle_centers/radii/active)
  dock    Docking family       (env.dock_point + env.dock_heading; TERMINAL
                                POSE cost -- position AND heading AND
                                approach speed, because the env's success is
                                that conjunction held for five seconds)
  mission HarborMission family (env.gate_midpoints + env.gate_normals +
                                env.berth_point + env.dock_heading; the
                                ordered gate chain scored by the env's own
                                signed crossing rule, then the dock leg.
                                HarborStage1/2/3 are the SAME env and the
                                SAME route with a shallower scoring depth --
                                cfg.mission_depth picks which milestone ends
                                the episode -- so one mission controller
                                drives all four ids)

Detection is deliberately attribute-based and still REFUSES loudly on any
family whose target it cannot read; it never falls back to a guessed cost.

Records: the eval_v6_frozen.py per-episode schema, unchanged, plus a
"controller": "mppi" tag per record and an "mppi" parameter block in the JSON
header -- scripts/usv10k_score.py consumes the --out JSON as-is.

Protocol mirrors eval_v6_frozen.py: env layout stream seeded by --eval-seed,
frozen curriculum level where the cfg supports it, fixed-horizon resets, one
record per finished episode. Same action space ([-1,1]^2) and control rate
(60 Hz) as every other controller: one controller action per env control step.

GPU sanity commands (16 envs / 16 episodes smoke; certification uses 64/128):

  python scripts/eval_mppi.py --task Isaac-USV-PathFollow-BlueBoat-Direct-v1 ^
      --num_envs 16 --episodes 16 --eval-seed 42 --headless ^
      --out C:/usvb/wf_mppi_pf16.json
  python scripts/eval_mppi.py --task Isaac-USV-StationKeep-BlueBoat-Direct-v1 ^
      --num_envs 16 --episodes 16 --eval-seed 42 --headless ^
      --out C:/usvb/wf_mppi_sk16.json
  python scripts/eval_mppi.py --task Isaac-USV-HazardCross-Direct-v1 ^
      --num_envs 16 --episodes 16 --level 0 --eval-seed 42 --headless ^
      --out C:/usvb/wf_mppi_cross16.json
  python scripts/eval_mppi.py --task Isaac-USV-Dock-BlueBoat-Direct-v1 ^
      --num_envs 16 --episodes 16 --level 0 --eval-seed 42 --headless ^
      --out C:/usvb/wf_mppi_dock16.json
  python scripts/eval_mppi.py --task Isaac-USV-HarborMission-Direct-v1 ^
      --num_envs 16 --episodes 16 --eval-seed 42 --headless ^
      --out C:/usvb/wf_mppi_harbor16.json

CPU validation without Isaac: python scripts/test_mppi_cpu.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)

SUPPORTED_TASK_PREFIX = "Isaac-USV-"

# Suite D shift fields the nominal model DELIBERATELY ignores; non-inert
# values are disclosed on stdout so shifted-pack runs are self-documenting.
SUITE_D_FIELDS = (
    ("mass_scale", 1.0),
    ("drag_scale", 1.0),
    ("thrust_cap_scale", 1.0),
    ("motor_tau_s", 0.0),
    ("thrust_imbalance", 0.0),
    ("mass_scale_choices", ()),
    ("drag_scale_choices", ()),
    ("thrust_cap_scale_choices", ()),
    ("motor_tau_s_choices", ()),
    ("thrust_imbalance_choices", ()),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="USVBench privileged MPPI baseline (nominal BlueBoat model)."
    )
    parser.add_argument("--task", default="Isaac-USV-PathFollow-BlueBoat-Direct-v1")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="JSON per-episode records")
    parser.add_argument("--K", type=int, default=256, help="MPPI samples")
    parser.add_argument("--H", type=int, default=60, help="Horizon (control steps)")
    parser.add_argument("--sigma-thrust", type=float, default=0.50)
    parser.add_argument("--sigma-yaw", type=float, default=0.80)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.3)
    parser.add_argument("--w-goal", type=float, default=1.0)
    parser.add_argument("--w-gate", type=float, default=50.0)
    parser.add_argument("--w-obstacle", type=float, default=40.0)
    parser.add_argument("--clearance-margin", type=float, default=0.60)
    parser.add_argument("--collision-cost", type=float, default=2000.0)
    parser.add_argument("--w-effort", type=float, default=0.02)
    parser.add_argument("--w-terminal", type=float, default=20.0)
    parser.add_argument("--w-speed-hold", type=float, default=0.6)
    # Terminal-pose weights, used by the dock and mission families only.
    # Not tuned: both are anchored at the 0.3 m/s success speed tolerance and
    # the shipped horizon. Heading is UNGATED and worth v_tol*dt*(H-1)/2 =
    # 0.15 (turning fully abeam to chase a lateral berth offset still pays at
    # the slowest speed the boat ever needs); the speed term applies inside
    # the 2.5 m tolerance and is worth dt*(H-1)/(2*v_tol) = 1.64 (0.300 m/s
    # is then the break-even approach speed inside the berth). Full
    # derivation, and the two rejected weightings that produced it, in
    # scripts/mppi_controller.py's module docstring.
    parser.add_argument("--w-dock-heading", type=float, default=0.15)
    parser.add_argument("--w-dock-speed", type=float, default=1.64)
    parser.add_argument("--mppi-seed", type=int, default=0,
                        help="Sampling seed (deterministic per device).")
    parser.add_argument("--set", dest="extra_sets", action="append", default=[],
                        metavar="FIELD=VALUE",
                        help="Extra top-level cfg overrides (repeatable), "
                             "e.g. Suite D probes: --set thrust_cap_scale=0.7")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Pre-flight guards; parser.error exits 2. CPU-testable without Isaac."""
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.num_envs < 1:
        parser.error("--num_envs must be >= 1")
    if args.K < 2:
        parser.error("--K must be >= 2")
    if args.H < 1:
        parser.error("--H must be >= 1")
    if args.sigma_thrust <= 0.0 or args.sigma_yaw <= 0.0:
        parser.error("--sigma-thrust and --sigma-yaw must be positive")
    if args.lambda_ <= 0.0:
        parser.error("--lambda must be positive")
    if args.w_dock_heading < 0.0 or args.w_dock_speed < 0.0:
        parser.error("--w-dock-heading and --w-dock-speed must be >= 0")
    if not args.task.startswith(SUPPORTED_TASK_PREFIX):
        parser.error(f"--task must start with {SUPPORTED_TASK_PREFIX!r}")
    for assignment in args.extra_sets:
        if "=" not in assignment:
            parser.error(f"--set expects FIELD=VALUE, got {assignment!r}")


def _quat_rotate(torch, quat, vec):
    """Rotate (N, 3) vectors by (N, 4) wxyz quaternions (pure torch)."""
    w = quat[:, 0:1]
    q_vec = quat[:, 1:4]
    doubled = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + w * doubled + torch.cross(q_vec, doubled, dim=-1)


def _detect_mode(base) -> str:
    """Pick a cost mode from the env's privileged attributes, or REFUSE.

    Widened, never loosened: each family is still recognized by the specific
    attributes its cost needs, and an env that exposes none of them still
    exits rather than being driven by a guessed cost. The three certified
    branches keep their exact order and predicates; the two new ones are
    appended after "gates" (they cannot collide with it -- neither docking
    env has waypoints/gates_passed) and before "hold"/"goal" (neither of
    which exists on the docking families either).
    """
    if hasattr(base, "waypoints") and hasattr(base, "gates_passed"):
        return "gates"
    if (
        hasattr(base, "gate_midpoints")
        and hasattr(base, "gate_normals")
        and hasattr(base, "berth_point")
        and hasattr(base, "dock_heading")
    ):
        return "mission"
    if hasattr(base, "dock_point") and hasattr(base, "dock_heading"):
        return "dock"
    if hasattr(base, "hold_point"):
        return "hold"
    if hasattr(base, "target_pos"):
        return "goal"
    raise SystemExit(
        "eval_mppi: env exposes none of waypoints/hold_point/target_pos/"
        "dock_point/gate_midpoints+berth_point; not a supported task family"
    )


def _water_motion_notes(base) -> list[str]:
    """Disclosure lines for water the nominal model does not know is moving.

    SUITE_D_FIELDS above only reaches TOP-LEVEL cfg fields. The current and
    the wave field live one level down, on cfg.underwater_physics_cfg and
    cfg.sea_state, so an env that turns either on used to sail past the
    disclosure loop in silence. That is exactly the case this closes:
    Isaac-USV-Dock-BlueBoat-Current-Direct-v1 enables a randomized 1.0-1.5 m/s
    crossing current (docking_env_cfg.py:272-280) and is accepted by the
    widened family detection, while scripts/mppi_controller.py's internal
    model is CALM WATER -- it has no current term and no wave term at all
    (module docstring: "Deliberately NOT modeled"). The same mismatch reaches
    StationKeep-BlueBoat-Current (2.0-2.5 m/s, station_keeping_env_cfg.py:
    255-263), its RampCurrent twin, and any sea-state variant.

    Treated exactly like a Suite D shift, and for the same reason: the
    mismatch IS the experiment, so this neither refuses the run nor hides it.
    Read from base.physics_cfg first -- StationKeepingEnv rewrites that object
    in __init__ for the speed-override and ramp variants
    (station_keeping_env.py:38-61), so the cfg class attributes are not the
    live values.
    """
    notes = []
    cfg = getattr(base, "cfg", None)
    physics = getattr(base, "physics_cfg", None)
    if physics is None and cfg is not None:
        physics = getattr(cfg, "underwater_physics_cfg", None)
    if physics is not None and bool(getattr(physics, "enable_current", False)):
        speed_min = float(getattr(physics, "current_speed_min", 0.0))
        speed_max = float(getattr(physics, "current_speed_max", 0.0))
        drag = float(getattr(physics, "current_drag_coeff", 0.0))
        notes.append(
            "[NOMINAL-MODEL NOTE] underwater_physics_cfg.enable_current=True "
            f"(current {speed_min:.2f}-{speed_max:.2f} m/s, "
            f"drag_coeff={drag:g}); "
            "the MPPI internal model is CALM WATER and carries no current "
            "term, so the controller is never told the water is moving"
        )
        ramp_start = getattr(cfg, "current_ramp_start_mps", None)
        ramp_end = getattr(cfg, "current_ramp_end_mps", None)
        if ramp_start is not None and ramp_end is not None:
            notes.append(
                "[NOMINAL-MODEL NOTE] the current also RAMPS within each "
                f"episode ({float(ramp_start):.2f} -> {float(ramp_end):.2f} "
                "m/s); the nominal model tracks neither the current nor its "
                "ramp"
            )
    sea_state = getattr(cfg, "sea_state", None) if cfg is not None else None
    if sea_state is not None and bool(getattr(sea_state, "enable", False)):
        notes.append(
            "[NOMINAL-MODEL NOTE] cfg.sea_state.enable=True (irregular wave "
            f"field, Hs {tuple(sea_state.hs_range)} m, Tp "
            f"{tuple(sea_state.tp_range)} s); the MPPI internal model is calm "
            "water and carries no wave term"
        )
    return notes


def _nominal_guard(base, nominal: dict) -> None:
    """Refuse to run when the env is not the nominal BlueBoat contract."""
    spec_name = getattr(getattr(base, "vehicle_spec", None), "name", None)
    if spec_name != "blueboat":
        raise SystemExit(
            f"eval_mppi: task vehicle is {spec_name!r}; the nominal model is "
            "BlueBoat-only (pick the BlueBoat variant of the task)"
        )
    cfg = base.cfg
    checks = (
        ("sim.dt", float(cfg.sim.dt), nominal["physics_dt_s"]),
        ("decimation", float(cfg.decimation), float(nominal["decimation"])),
        ("thrust_max_fwd", float(cfg.thrust_max_fwd), nominal["thrust_fwd_n"]),
        ("thrust_max_rev", float(cfg.thrust_max_rev), nominal["thrust_rev_n"]),
        ("yaw_torque_max", float(cfg.yaw_torque_max), nominal["yaw_torque_nm"]),
    )
    physics = getattr(base, "physics_cfg", None) or getattr(
        cfg, "underwater_physics_cfg", None
    )
    if physics is not None:
        checks = checks + (
            ("surge_lin_damping", float(physics.surge_lin_damping), nominal["surge_lin"]),
            ("surge_quad_damping", float(physics.surge_quad_damping), nominal["surge_quad"]),
            ("sway_lin_damping", float(physics.sway_lin_damping), nominal["sway_lin"]),
            ("sway_quad_damping", float(physics.sway_quad_damping), nominal["sway_quad"]),
            ("yaw_lin_damping", float(physics.yaw_lin_damping), nominal["yaw_lin"]),
            ("yaw_quad_damping", float(physics.yaw_quad_damping), nominal["yaw_quad"]),
        )
    for name, actual, expected in checks:
        if abs(actual - expected) > 1.0e-9:
            raise SystemExit(
                f"eval_mppi: env {name}={actual} != nominal {expected}; the "
                "frozen nominal model no longer matches the deployed default "
                "physics -- update scripts/mppi_controller.py consciously"
            )
    half_beam = getattr(cfg, "half_beam_m", None)
    if half_beam is not None and abs(float(half_beam) - nominal["half_beam_m"]) > 1.0e-9:
        raise SystemExit(
            f"eval_mppi: cfg half_beam_m={half_beam} != nominal "
            f"{nominal['half_beam_m']}"
        )
    for field_name, inert in SUITE_D_FIELDS:
        value = getattr(cfg, field_name, None)
        if value is None:
            continue
        non_inert = (tuple(value) if isinstance(value, (list, tuple)) else value) != inert
        if non_inert:
            print(
                f"[NOMINAL-MODEL NOTE] cfg.{field_name}={value!r} shifts the "
                "env; the MPPI internal model stays nominal by design",
                flush=True,
            )
    # Water motion lives BELOW the top-level cfg, so the loop above cannot
    # see it; disclosed here on the same channel, in the same format.
    for note in _water_motion_notes(base):
        print(note, flush=True)
    data = getattr(base.robot, "data", None)
    default_mass = getattr(data, "default_mass", None)
    if default_mass is not None:
        try:
            sim_mass = float(default_mass.reshape(base.num_envs, -1)[0, 0])
            if abs(sim_mass - nominal["mass_kg"]) > 0.05:
                print(
                    f"[WARN] sim authored mass {sim_mass:.3f} kg differs from "
                    f"nominal {nominal['mass_kg']} kg",
                    flush=True,
                )
        except Exception:
            pass


class EnvBridge:
    """Privileged state/target/obstacle reads for one supported env family."""

    def __init__(self, torch, base, mode: str):
        self.t = torch
        self.base = base
        self.mode = mode
        if mode == "gates":
            self.goal_radius = float(base.cfg.goal_radius)
        elif mode == "hold":
            self.goal_radius = float(base.cfg.hold_radius)
        elif mode in ("dock", "mission"):
            # The BERTHING tolerance, i.e. the radius the success predicate
            # actually tests (docking_env.py:653-657,
            # harbor_mission_env.py:789-793). cfg.goal_radius aliases it in
            # both families -- harbor_mission_env_cfg.py:284-285 enforces the
            # alias -- but the predicate's own field is the honest source.
            self.goal_radius = float(
                getattr(base.cfg, "success_position_tolerance_m", None)
                or base.cfg.goal_radius
            )
        else:
            self.goal_radius = float(
                getattr(base, "goal_radius", None) or base.cfg.goal_radius
            )
        self.gate_half_width_m = float(getattr(base.cfg, "gate_half_width_m", 0.0))
        self.gate_min_normal_speed_mps = float(
            getattr(base.cfg, "gate_min_normal_speed_mps", 0.0)
        )

    def position_xy(self):
        """The reference point the ACTIVE family's OWN success predicate scores.

        This is the only place the evaluator chooses between the two points a
        deployed env can measure success at, and choosing wrong is SILENT: the
        controller still runs, still converges, and still reports a plausible
        number -- against the wrong point.

          family          publishes _com_xy   scored at        branch below
          PathFollow      no                  root_pos_w       3 (fallback)
          StationKeep     no                  root_pos_w       3 (fallback)
          HazardNav       YES                 root_com_pos_w   1 (helper)
          PathHazard      YES                 root_com_pos_w   1 (helper)
          Docking         no                  root_com_pos_w   2 (dock/mission)
          HarborMission   YES                 root_com_pos_w   1 (helper)

        Sources for the "scored at" column, each inside the block that ends up
        latching episode_success: path_following_env.py:588 + :611-619;
        station_keeping_env.py:557-558 (used at :658); hazard_nav_env.py:
        856-860 + :1428-1430; path_hazard_env.py:589-590 + :840, :883-892;
        docking_env.py:630-637 + :649-657; harbor_mission_env.py:749-750 +
        :788, :975, :1013-1022. scripts/test_mppi_cpu.py pins every row, and
        test_bridge_reference_points_match_the_env_sources re-derives that
        column from tasks/ so the table cannot rot into a comment-only claim.

        1. _com_xy when the env publishes it. Three families do, and all three
           score at the center of mass.
        2. root_com_pos_w for dock/mission. Docking does NOT publish _com_xy,
           yet still scores at the COM: the boat asset's USD origin sits
           ~1.0 m from it -- 40% of the 2.5 m berthing tolerance, and a yaw
           reorientation sweeps the origin around that 1 m arc
           (docking_env.py:630-637, "probed 2026-07-16; broke both PID and
           RL"). Reading the origin here buys wrong docking numbers that look
           right, not a visible failure.
        3. root_pos_w, the USD body origin. ONLY PathFollow and StationKeep
           reach it, and those two do score at the body origin.

        NOTE, because an earlier version of this comment said otherwise: it is
        NOT true that "the certified families keep root_pos_w because that is
        what their own predicates score". HazardNav and PathHazard are
        certified families, they publish _com_xy, they leave through branch 1,
        and they score at the center of mass. Only two of the four certified
        families reach branch 3.
        """
        helper = getattr(self.base, "_com_xy", None)
        if callable(helper):
            return helper()
        if self.mode in ("dock", "mission"):
            return self.base.robot.data.root_com_pos_w[:, :2]
        return self.base.robot.data.root_pos_w[:, :2]

    def state(self):
        t = self.t
        base = self.base
        pos = self.position_xy()
        vel = base.robot.data.root_com_vel_w
        if vel.shape[-1] == 6:
            lin = vel[:, :2]
            wz = vel[:, 5]
        else:
            lin = vel[:, :2]
            wz = base.robot.data.root_ang_vel_w[:, 2]
        fwd_helper = getattr(base, "_forward_2d", None)
        if callable(fwd_helper):
            fwd = fwd_helper()
        else:
            quat_helper = getattr(base, "_root_quat", None)
            quat = quat_helper() if callable(quat_helper) else (
                base.robot.data.root_link_quat_w
            )
            fwd = _quat_rotate(t, quat, base.forward_vec)[:, :2]
            fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        yaw = t.atan2(fwd[:, 1], fwd[:, 0])
        cos_y = t.cos(yaw)
        sin_y = t.sin(yaw)
        u = cos_y * lin[:, 0] + sin_y * lin[:, 1]
        v = -sin_y * lin[:, 0] + cos_y * lin[:, 1]
        return t.stack((pos[:, 0], pos[:, 1], yaw, u, v, wz), dim=-1)

    def targets(self):
        t = self.t
        base = self.base
        if self.mode == "gates":
            world = base.waypoints + base.scene.env_origins[:, None, :2]
            num_gates = world.shape[1]
            idx = base.gates_passed.clamp(max=num_gates - 1).to(t.long)
            return world, idx
        if self.mode == "hold":
            return base.hold_point[:, None, :], None
        if self.mode == "dock":
            return base.dock_point[:, None, :], None
        if self.mode == "mission":
            # gate_midpoints and berth_point are already WORLD frame -- the
            # env adds scene.env_origins when it samples the route
            # (harbor_mission_env.py:1264-1271), unlike PathFollow's
            # env-local waypoints. Do NOT add origins again here.
            route = t.cat((base.gate_midpoints, base.berth_point[:, None, :]), dim=1)
            return route, self._mission_leg()
        return base.target_pos[:, None, :], None

    def _mission_leg(self):
        """Active leg index: 0/1 = exit gates, 2 = field-exit gate, 3 = berth.

        Read from the env's OWN automaton, not re-derived: phase advances at
        harbor_mission_env.py:1040-1046 and :1088-1090, and the sub-state that
        separates the two exit gates while phase is still 0 is
        _exit_gate_progress (:1002-1005, :1030-1032). Getting this wrong is
        not cosmetic -- an index that lags behind the boat charges distance to
        a gate already astern and drives it BACKWARD -- so the geometric
        fallback below exists only for a build that renames that counter, and
        says so on stdout the first time it is used.
        """
        t = self.t
        base = self.base
        phase = base.phase.to(t.long)
        progress = getattr(base, "_exit_gate_progress", None)
        if progress is None:
            if not getattr(self, "_warned_exit_progress", False):
                self._warned_exit_progress = True
                print(
                    "[WARN] env has no _exit_gate_progress; falling back to a "
                    "geometric test of the first exit gate plane",
                    flush=True,
                )
            first_mid = base.gate_midpoints[:, 0, :]
            first_normal = base.gate_normals[:, 0, :]
            signed = ((self.position_xy() - first_mid) * first_normal).sum(dim=-1)
            progress = (signed > 0.0).to(t.long)
        else:
            progress = progress.to(t.long)
        leg = t.where(phase >= 1, phase + 1, progress.clamp(max=1))
        return leg.clamp(max=3)

    def dock_heading(self):
        return self.base.dock_heading

    def gate_normals(self):
        return self.base.gate_normals

    def obstacles(self):
        base = self.base
        if not hasattr(base, "obstacle_centers"):
            return None, None, None
        return (
            base.obstacle_centers,
            base.obstacle_radii,
            getattr(base, "obstacle_active", None),
        )


def _run_isaac_eval() -> None:
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    _validate_args(parser, args_cli)
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args_cli).app

    import torch
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers the deployed task ids)
    from isaaclab_tasks.utils import parse_env_cfg

    # The controller module sits beside this script locally; on the boxes the
    # pair may be deployed as C:/usvb/wf_eval_mppi.py + wf_mppi_controller.py.
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    try:
        from mppi_controller import (
            MPPIController,
            MPPIParams,
            NOMINAL_BLUEBOAT,
            NominalBlueBoatDynamics,
        )
    except ImportError:
        from wf_mppi_controller import (
            MPPIController,
            MPPIParams,
            NOMINAL_BLUEBOAT,
            NominalBlueBoatDynamics,
        )
    # Scenario-protocol stamping, on the same dual import scripts/
    # eval_v6_frozen.py:74-80 uses so the script keeps working from the repo
    # and from the deployed task tree.
    if _REPO_ROOT not in sys.path:
        sys.path.append(_REPO_ROOT)
    try:
        from tasks._shared.scenario_draws import (
            episode_scenario_hashes_for,
            scenario_protocol_notice,
            scenario_protocol_stamp,
        )
    except ImportError:
        from isaaclab_tasks.direct._shared.scenario_draws import (
            episode_scenario_hashes_for,
            scenario_protocol_notice,
            scenario_protocol_stamp,
        )

    task = args_cli.task
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.eval_seed
    for assignment in args_cli.extra_sets:
        field_name, _, raw = assignment.partition("=")
        if not hasattr(env_cfg, field_name):
            raise SystemExit(f"--set target {field_name!r} is not a cfg field")
        current = getattr(env_cfg, field_name)
        caster = type(current) if isinstance(current, (int, float, bool)) else str
        setattr(env_cfg, field_name, caster(raw) if caster is not bool
                else raw.lower() in ("1", "true", "yes"))
        print(f"set {field_name}={getattr(env_cfg, field_name)}", flush=True)
    if hasattr(env_cfg, "curriculum_frozen"):
        env_cfg.curriculum_frozen = True
        env_cfg.eval_level = args_cli.level

    env = gym.make(task, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    env.reset()

    mode = _detect_mode(base)
    if mode == "dock" and bool(getattr(base.cfg, "berth_walls", False)):
        # DockWall publishes its U-shaped slip only privately (_wall_centers /
        # _wall_radii, docking_env.py:118-124, :139-150), so the obstacle
        # bridge below sees nothing and the cost has no wall term. Contact
        # before the hold completes voids the episode (:775-789), so any
        # DockWall number from this controller is a FLOOR that has not been
        # told the walls exist -- not a tuned result. Disclosed, not silent.
        print(
            "[NOMINAL-MODEL NOTE] cfg.berth_walls=True but the env exposes no "
            "public wall geometry; the MPPI cost carries NO berth-wall term",
            flush=True,
        )
    _nominal_guard(base, NOMINAL_BLUEBOAT)
    expected_dt = NOMINAL_BLUEBOAT["physics_dt_s"] * NOMINAL_BLUEBOAT["decimation"]
    if abs(float(base.control_step_s) - expected_dt) > 1.0e-9:
        raise SystemExit(
            f"eval_mppi: control_step_s={base.control_step_s} != {expected_dt}"
        )

    params = MPPIParams(
        K=args_cli.K,
        H=args_cli.H,
        sigma=(args_cli.sigma_thrust, args_cli.sigma_yaw),
        lambda_=args_cli.lambda_,
        w_goal=args_cli.w_goal,
        w_gate=args_cli.w_gate,
        w_obstacle=args_cli.w_obstacle,
        clearance_margin_m=args_cli.clearance_margin,
        collision_cost=args_cli.collision_cost,
        w_effort=args_cli.w_effort,
        w_terminal=args_cli.w_terminal,
        w_speed_hold=args_cli.w_speed_hold,
        w_dock_heading=args_cli.w_dock_heading,
        w_dock_speed=args_cli.w_dock_speed,
        seed=args_cli.mppi_seed,
    )
    dynamics = NominalBlueBoatDynamics(device=base.device)
    controller = MPPIController(
        num_envs=base.num_envs,
        device=base.device,
        params=params,
        dynamics=dynamics,
    )
    bridge = EnvBridge(torch, base, mode)
    mppi_repro = {
        "controller": "mppi",
        "mode": mode,
        "privileged": True,
        "nominal_model": "blueboat_default_calm_water",
        "tracks_suite_d_shifts": False,
        **params.as_dict(),
    }
    print("[REPRO] " + "  ".join(f"{k}={v}" for k, v in mppi_repro.items()),
          flush=True)

    # Does THIS env carry the scenario protocol?  Same guard, same marker and
    # same warning as scripts/eval_v6_frozen.py:148-162, so an MPPI
    # certificate and a policy certificate at the same --eval-seed can be
    # scenario-paired instead of merely assumed comparable.
    scenario_protocol = scenario_protocol_stamp(base)
    notice = scenario_protocol_notice(task, scenario_protocol)
    if notice:
        print(notice, flush=True)

    records = []
    ep_counter = torch.zeros(base.num_envs, dtype=torch.long)
    max_steps = (
        args_cli.episodes // base.num_envs + 3
    ) * base.max_episode_length
    step = 0
    ess_sum = 0.0
    ess_steps = 0
    while len(records) < args_cli.episodes and step < max_steps:
        with torch.inference_mode():
            state = bridge.state()
            targets, active_index = bridge.targets()
            centers, radii, active = bridge.obstacles()
            extra = {}
            if mode in ("dock", "mission"):
                extra["dock_heading"] = bridge.dock_heading()
            if mode == "mission":
                extra["gate_normals"] = bridge.gate_normals()
                extra["gate_half_width_m"] = bridge.gate_half_width_m
                extra["gate_min_normal_speed_mps"] = (
                    bridge.gate_min_normal_speed_mps
                )
            actions, info = controller.rollout(
                state,
                targets=targets,
                active_index=active_index,
                goal_radius=bridge.goal_radius,
                mode=mode,
                obstacle_centers=centers,
                obstacle_radii=radii,
                obstacle_active=active,
                **extra,
            )
        ess_sum += float(info["ess"].mean())
        ess_steps += 1
        # d0_per_env is rewritten inside _reset_idx, which runs during step():
        # snapshot it while it still belongs to the episode about to end
        # (same protocol note as eval_v6_frozen.py).
        d0_prev = (
            base.d0_per_env.clone() if hasattr(base, "d0_per_env") else None
        )
        _obs, _rew, term, trunc, _info = env.step(actions)
        step += 1
        done = term | trunc
        done = done.squeeze(-1) if done.dim() > 1 else done
        with torch.inference_mode():
            controller.reset(done)
        ids = torch.nonzero(done).flatten()
        for i in ids.tolist():
            episode_contact_steps = getattr(base, "episode_contact_steps", None)
            episode_contact_longest_steps = getattr(
                base, "episode_contact_longest_steps", None
            )
            episode_contact_depth_sum = getattr(
                base, "episode_contact_depth_sum", None
            )
            episode_max_phase = getattr(base, "episode_max_phase", None)
            rec = {
                "env": i,
                "ep": int(ep_counter[i]),
                "controller": "mppi",
                "success": bool(base.episode_success[i]),
                "tts_s": (
                    None
                    if math.isnan(float(base.time_to_success[i]))
                    else float(base.time_to_success[i])
                ),
                "path_length_m": float(base.episode_path_length[i]),
            }
            if hasattr(base, "episode_min_clearance"):
                rec["min_clearance_m"] = float(base.episode_min_clearance[i])
            # Cross-track error, mirrored from scripts/eval_v6_frozen.py so every
            # controller writes the same record schema (the parity tests enforce it).
            if hasattr(base, "episode_xte_rms"):
                rec["xte_rms_m"] = float(base.episode_xte_rms[i])
            # Per-primitive digests of the scenario the FINISHED episode ran,
            # latched at reset exactly like episode_min_clearance.  Absent on
            # families not yet on the scenario protocol, and empty until an env
            # has completed its first episode; both cases are guarded inside
            # the helper, which returns None for "write no field".
            scenario_hashes = episode_scenario_hashes_for(base, i)
            if scenario_hashes is not None:
                rec["scenario_hashes"] = scenario_hashes
            if (
                episode_contact_steps is not None
                and episode_contact_longest_steps is not None
                and episode_contact_depth_sum is not None
            ):
                contact_steps = float(episode_contact_steps[i])
                rec["contact_steps"] = int(contact_steps)
                rec["contact_seconds"] = contact_steps * base.control_step_s
                rec["contact_longest_seconds"] = (
                    float(episode_contact_longest_steps[i]) * base.control_step_s
                )
                rec["contact_depth_mean_m"] = (
                    float(episode_contact_depth_sum[i]) / contact_steps
                    if contact_steps > 0.0
                    else 0.0
                )
            if episode_max_phase is not None:
                rec["max_phase"] = int(episode_max_phase[i])
            if hasattr(base, "episode_gates_passed"):
                rec["gates"] = int(base.episode_gates_passed[i])
            if d0_prev is not None:
                rec["d0_m"] = float(d0_prev[i])
            if hasattr(base, "route_geodesic_length"):
                geodesic = float(base.route_geodesic_length[i])
                if geodesic > 0.0:
                    rec["route_geodesic_m"] = geodesic
            records.append(rec)
            ep_counter[i] += 1

    records = records[: args_cli.episodes]
    n = len(records)
    successes = [r for r in records if r["success"]]
    tts = sorted(r["tts_s"] for r in successes if r["tts_s"] is not None)
    clearances = sorted(
        r["min_clearance_m"] for r in records if "min_clearance_m" in r
    )
    collided = sum(1 for c in clearances if c < 0.0)

    def pct(sorted_values, q):
        if not sorted_values:
            return float("nan")
        k = min(len(sorted_values) - 1, max(0, int(q * (len(sorted_values) - 1))))
        return sorted_values[k]

    sr = len(successes) / max(n, 1)
    print(f"EVAL task={task} level={args_cli.level} seed={args_cli.eval_seed} "
          f"controller=mppi")
    print(f"  episodes={n} SR={sr:.4f} ({len(successes)}/{n})")
    print(f"  tts median={pct(tts, 0.5):.1f}s p90={pct(tts, 0.9):.1f}s" if tts
          else "  tts: no successes")
    print(f"  collision_episodes={collided}/{n} ({collided / max(n, 1):.3f}) "
          f"min_clearance p10={pct(clearances, 0.10):.2f}m")
    path = sorted(r["path_length_m"] for r in successes) or sorted(
        r["path_length_m"] for r in records
    )
    print(f"  path_len median={pct(path, 0.5):.1f}m p90={pct(path, 0.9):.1f}m",
          end="")
    ratios = sorted(
        r["path_length_m"] / r["d0_m"]
        for r in successes
        if r.get("d0_m", 0.0) > 0.0
    )
    print(f"  detour median={pct(ratios, 0.5):.2f}x straight" if ratios else "")
    if ess_steps:
        print(f"  mppi: mean ESS {ess_sum / ess_steps:.1f} of K={args_cli.K} "
              f"over {ess_steps} control steps")

    if args_cli.out:
        with open(args_cli.out, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "task": task,
                    "level": args_cli.level,
                    "seed": args_cli.eval_seed,
                    # The header block when the env carries the protocol
                    # object, the off-protocol literal when it does not --
                    # never unconditionally the header.
                    "scenario_protocol": scenario_protocol,
                    "controller": "mppi",
                    "mppi": mppi_repro,
                    "records": records,
                },
                stream,
                indent=1,
            )
        print(f"  records -> {args_cli.out}")
    sys.stdout.flush()
    env.close()
    app.close()


if __name__ == "__main__":
    _run_isaac_eval()
