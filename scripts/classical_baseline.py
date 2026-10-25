"""Zero-learning LOS + PID baseline for USVBench.

The controller consumes only the policy observation: target dot/cross products
and normalized distance in observation columns 0:3. It supports the fixed-step
throughput protocol from eval_benchmark.py and the completed-episode mission
protocol from eval_mission.py.

The default gains are hand-set method-side choices, not tuned benchmark
parameters. Authors may tune their classical controller, but must report the
values used.

Examples (current BlueBoat registry ids):
  python scripts/classical_baseline.py \
    --task Isaac-USV-BlueBoat-Calm-Direct-v1 --protocol throughput \
    --num_envs 64 --eval_steps 6000 --headless

  python scripts/classical_baseline.py \
    --task Isaac-USV-PathFollow-BlueBoat-Direct-v1 --protocol mission \
    --num_envs 64 --episodes 128 --headless --out records.json

  python scripts/classical_baseline.py \
    --task Isaac-USV-StationKeep-BlueBoat-Direct-v1 --protocol mission \
    --num_envs 64 --episodes 128 --headless --out records.json

  python scripts/classical_baseline.py \
    --task Isaac-USV-StationKeep-BlueBoat-Current-Direct-v1 --protocol mission \
    --num_envs 64 --episodes 128 --headless --current-speed 2.0

  python scripts/classical_baseline.py \
    --task Isaac-USV-Dock-BlueBoat-Direct-v1 --protocol mission \
    --num_envs 64 --episodes 128 --level 3 --headless --out records.json

Frozen-curriculum tasks (docking) are always evaluated with the curriculum
frozen, mirroring scripts/eval_v6_frozen.py; --level selects the pinned
difficulty (docking eval_level_distances: 0=25m, 1=15m, 2=8m, 3=2m). The
July-era certified dock 100% ran the live curriculum from its 2 m start, so
--level 3 is the comparable setting. The point-nav family (blueboat_calm_nav)
respawns targets in place and only supports --protocol throughput.

--out writes eval_v6_frozen-compatible per-episode JSON records (mission
protocol only) so usv10k_score.py and the census tooling ingest the classical
column unchanged.

Journal ids (F7, CONTEXT_R4 D5/D12): --task takes a journal id or
journal:<task> with --vessel (eval_protocol.select_task_id picks the
vessel-specific id before the cfg is parsed). The controller then reads the D5
estimate (journal_controllers.JournalBaselineController: env.journal_estimate +
heading + task targets, never sea/current/wind/D7), uses the frozen per-vessel
gains of scripts/classical_gains_v2.json (--kp/--ki/--kd are legacy-only and
refused on journal ids), scales its lengths by the vessel length ratio, and the
certificate takes the journal mixin header/records.

G17 (owner decision B10 supplement, option B): on the journal path following,
station keeping and docking ids that controller is wrapped by the obstacle-
avoidance layer of scripts/journal_avoidance.py (frozen constants
scripts/classical_avoidance_v2.json); the layer is inactive unless the rays show
an obstacle near the vessel, and while inactive the action is the base law's own.
G24b certificate names (G17's los_pid_avoid_v1 / docking_3phase_pid_avoid_v1 retired):
los_pid_avoid_v2 (path following; layer memory without expiry), los_pid_cont_avoid_v2
(station keeping; hold mode continuous_v2, journal_avoidance.journal_hold_mode) and
docking_runup_pid_avoid_v2 (docking; journal_controllers.RunupDockingLaw).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Mapping

# Shared --set FIELD=VALUE parsing. Dotted paths reach nested cfgs, which is
# what the wave ladder needs: its rungs live at cfg.sea_state.hs_range, one
# level below anything a top-level-only override could touch. Imported before
# Isaac so the no-Isaac argparse layer of test_classical_baseline_math.py
# still reaches the parser.
try:
    from cfg_override import apply_overrides, validate_assignments
except ImportError:  # invoked from a cwd that is not scripts/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cfg_override import apply_overrides, validate_assignments

from cert_v3 import CertificateRun
# G6: fail-closed exit codes and certificate publication (scripts/evaluator_exit.py).
import evaluator_exit  # noqa: E402

_EXIT_GUARD = evaluator_exit.Guard()
from eval_protocol import add_protocol_arguments, preflight_protocol, resolve_protocol
from isaaclab.app import AppLauncher

ORIGINAL_ARGV = list(sys.argv)

parser = argparse.ArgumentParser(description="USVBench classical LOS + PID baseline.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument(
    "--protocol",
    type=str,
    required=True,
    choices=["throughput", "mission"],
    help="Fixed-step targets/episode or completed-episode SR/SPL evaluation.",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128, help="Completed episodes for mission protocol.")
parser.add_argument("--eval_steps", type=int, default=6000, help="Steps for throughput protocol.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--level",
    type=int,
    default=0,
    help=(
        "Frozen-curriculum eval level for tasks with curriculum_frozen/"
        "eval_level (docking: 0=25m, 1=15m, 2=8m, 3=2m spawn distance). "
        "Tasks without a curriculum ignore it."
    ),
)
parser.add_argument(
    "--spawn-distance",
    type=float,
    default=None,
    help=(
        "Spawn distance override in metres. Sets spawn_distance_override_m on "
        "frozen-eval tasks (docking), else curriculum_start_distance_m; tasks "
        "without either field ignore it."
    ),
)
parser.add_argument(
    "--current-speed",
    type=float,
    default=None,
    help=(
        "Pin the disturbance current to one speed (m/s) via "
        "current_speed_override_mps (station keeping); tasks without the "
        "field ignore it."
    ),
)
parser.add_argument("--csv", type=str, default=None, help="If set, append one result row to this CSV.")
parser.add_argument(
    "--out",
    type=str,
    default=None,
    help=(
        "Mission protocol only: write eval_v6_frozen-compatible per-episode "
        "JSON records (plus controller name and gains) to this path."
    ),
)
parser.add_argument(
    "--dist-scale",
    type=float,
    default=None,
    help=(
        "Observation distance scale in metres "
        "(default: 30 point-nav, 20 path-follow, 15 station-keep, 25 docking)."
    ),
)
parser.add_argument(
    "--slow-radius",
    type=float,
    default=None,
    help="Distance in metres at which thrust ramps down (default: 5, or 6 for docking).",
)
parser.add_argument("--hold-mode", choices=["deadband", "continuous", "continuous_v2"], default=None,
                    help="Station hold behavior: frozen deadband (calm-water heritage) or "
                         "continuous regulation (textbook DP; required against sustained current). "
                         "Default: deadband on legacy ids; journal ids resolve it with "
                         "journal_avoidance.journal_hold_mode (G24b: station keeping continuous_v2, "
                         "any other request refused; path following / docking deadband = unused).")
parser.add_argument("--kp", type=float, default=2.0, help="Heading PID proportional gain (hand-set default).")
parser.add_argument("--ki", type=float, default=0.0, help="Heading PID integral gain (hand-set default).")
parser.add_argument("--kd", type=float, default=0.5, help="Heading PID derivative gain (hand-set default).")
parser.add_argument("--set", dest="extra_sets", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="Extra cfg overrides (repeatable). Dotted paths reach "
                         "nested cfgs, e.g. one pinned wave rung: "
                         "--set sea_state.hs_range=0.6,0.6 "
                         "--set sea_state.tp_range=2.25,2.25")
add_protocol_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
preflight_protocol(args_cli, ORIGINAL_ARGV, parser)
validate_assignments(args_cli.extra_sets, parser.error)
# G24b: legacy ids keep the deadband default (and have no continuous_v2 law); journal ids resolve the hold
# mode once the task mode is known (journal_avoidance.journal_hold_mode, below).
if getattr(args_cli, "journal_task_info", None) is None:
    if args_cli.hold_mode == "continuous_v2":
        parser.error("--hold-mode continuous_v2 is the journal station-keeping law (journal ids only)")
    if args_cli.hold_mode is None:
        args_cli.hold_mode = "deadband"

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive")
if args_cli.eval_steps <= 0:
    parser.error("--eval_steps must be positive")
if args_cli.dist_scale is not None and args_cli.dist_scale <= 0.0:
    parser.error("--dist-scale must be positive")
if args_cli.level < 0:
    parser.error("--level must be >= 0")
if args_cli.spawn_distance is not None and args_cli.spawn_distance <= 0.0:
    parser.error("--spawn-distance must be positive")
if args_cli.current_speed is not None and args_cli.current_speed <= 0.0:
    parser.error("--current-speed must be positive")
if args_cli.out and args_cli.protocol != "mission":
    parser.error("--out records are per completed episode; use --protocol mission")
if args_cli.slow_radius is not None and args_cli.slow_radius <= 0.0:
    parser.error("--slow-radius must be positive")
if not all(math.isfinite(value) for value in (args_cli.kp, args_cli.ki, args_cli.kd)):
    parser.error("PID gains must be finite")

# G6: Hydra's run directory and logs go to the output folder, never the working directory
# (scripts/hydra_outputs.py; Isaac Lab hydra.py:83 leaves Hydra's cwd-relative defaults).
# scripts/ is already on sys.path here (cfg_override / cert_v3 imports above).
import hydra_outputs  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args + hydra_outputs.run_dir_overrides(args_cli.out or args_cli.csv, hydra_args)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import (  # noqa: F401
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# F7: journal ids are registered by tasks/journal (D10) before hydra resolves
# the task's cfg entry point; --task was already replaced by the
# vessel-specific journal id in preflight_protocol.
JOURNAL = getattr(args_cli, "journal_task_info", None)
if JOURNAL is not None:
    from eval_protocol import register_journal_tasks

    register_journal_tasks()
    if (args_cli.kp, args_cli.ki, args_cli.kd) != (2.0, 0.0, 0.5):
        parser.error("--kp/--ki/--kd are legacy options; journal ids use the frozen per-vessel gains of "
                     "scripts/classical_gains_v2.json (F7 TUNING_PROCEDURE.md)")

# Scenario-protocol stamping, on the same dual import scripts/
# eval_v6_frozen.py:74-80 uses so the script keeps working from the repo and
# from the deployed task tree.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

if JOURNAL is not None:
    # F7: journal records take the D11 primitive map from the mixin latch
    # (cert_v3.finish_episode); the legacy scenario-protocol helper is inert.
    def episode_scenario_hashes_for(_base, _env_id):
        return None


def _base_env(env):
    """Dig through evaluation wrappers to the underlying Isaac environment."""
    e = env
    for _ in range(10):
        if any(hasattr(e, name) for name in ("reached_count", "episode_success", "path_length")):
            return e
        if hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        elif hasattr(e, "env"):
            e = e.env
        else:
            break
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _git_commit() -> str:
    """Resolve HEAD against the repository containing this script."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cfg_fingerprint(env_cfg) -> tuple[str, dict[str, object]]:
    """Hash protocol-relevant scalar config fields in a stable order."""
    names = {
        "curriculum_frozen",
        "curriculum_start_distance_m",
        "current_speed_override_mps",
        "episode_length_s",
        "eval_level",
        "goal_radius",
        "heading_change_max_deg",
        "hold_radius",
        "num_waypoints",
        "required_hold_time_s",
        "segment_length_max",
        "segment_length_min",
        "success_heading_tolerance_deg",
        "success_position_tolerance_m",
        "success_speed_tolerance_mps",
    }
    names.update(name for name in dir(env_cfg) if "spawn" in name.lower() and "distance" in name.lower())

    scalars = {}
    for name in sorted(names):
        if name.startswith("_") or not hasattr(env_cfg, name):
            continue
        value = getattr(env_cfg, name)
        if isinstance(value, (bool, int, float, str)):
            scalars[name] = value

    encoded = json.dumps(scalars, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:10], scalars


def _radius_from_env(base, env_cfg) -> float:
    """Read the fixed task-zone radius, preferring an instance override."""
    for owner in (base, getattr(base, "cfg", None), env_cfg):
        if owner is None:
            continue
        for name in ("hold_radius", "goal_radius"):
            value = getattr(owner, name, None)
            if value is not None:
                return float(value)
    raise RuntimeError("Task has neither hold_radius nor goal_radius; cannot define the speed controller zone")


# G6: journal cfgs name their task; the three classical_baseline journal tasks map directly
# (the journal docking env has no legacy hold_timer, so the attribute probe below would fall
# through to point_nav). Legacy cfgs keep the attribute probe unchanged.
JOURNAL_TASK_MODES = {"path_following": "path_follow", "station_keeping": "station_keep", "docking": "docking"}
# G17: journal modes whose controller carries the obstacle-avoidance layer (scripts/journal_avoidance.py).
JOURNAL_AVOIDANCE_MODES = ("path_follow", "station_keep", "docking")


def _task_mode(base, env_cfg) -> str:
    """Dispatch controller/metrics behavior from the selected task."""
    journal_task = getattr(env_cfg, "journal_task", None)
    if getattr(env_cfg, "journal_contract_version", None) == 2 and journal_task in JOURNAL_TASK_MODES:
        return JOURNAL_TASK_MODES[journal_task]
    if all(
        hasattr(base, name)
        for name in ("dock_point", "dock_heading", "hold_timer", "episode_path_length")
    ):
        return "docking"
    if (
        callable(getattr(base, "_current_waypoint_world", None))
        and hasattr(base, "gates_passed")
        and hasattr(base, "xte_rms")
    ):
        return "path_follow"
    if hasattr(env_cfg, "hold_radius") or hasattr(getattr(base, "cfg", None), "hold_radius"):
        return "station_keep"
    return "point_nav"


def _control_step_s(env_cfg) -> float:
    return float(env_cfg.sim.dt) * int(getattr(env_cfg, "decimation", 1))


def _policy_obs(obs) -> torch.Tensor:
    if isinstance(obs, Mapping):
        obs = obs.get("policy", obs)
    if isinstance(obs, Mapping):
        if len(obs) != 1:
            raise RuntimeError("Cannot select a single policy observation from the environment mapping")
        obs = next(iter(obs.values()))
    tensor = torch.as_tensor(obs)
    if tensor.ndim != 2 or tensor.shape[1] < 3:
        raise RuntimeError(f"Expected policy observations shaped (num_envs, >=3), got {tuple(tensor.shape)}")
    return tensor


class LOSPIDController:
    """Vectorized line-of-sight guidance and heading PID controller."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        dt: float,
        dist_scale: float,
        zone_radius: float,
        slow_radius: float,
        kp: float,
        ki: float,
        kd: float,
        pass_through: bool = False,
        hold_mode: str = "deadband",
    ) -> None:
        if dt <= 0.0:
            raise ValueError("Control timestep must be positive")
        if slow_radius <= zone_radius:
            raise ValueError(
                f"--slow-radius ({slow_radius:g}) must exceed the task zone radius ({zone_radius:g})"
            )
        self.dt = dt
        self.dist_scale = dist_scale
        self.zone_radius = zone_radius
        self.hold_mode = hold_mode
        self.slow_radius = slow_radius
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.pass_through = pass_through
        self.integral = torch.zeros(num_envs, device=device, dtype=dtype)
        self.previous_error = torch.zeros_like(self.integral)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def act(self, obs) -> torch.Tensor:
        policy_obs = _policy_obs(obs)
        dot = policy_obs[:, 0]
        cross = policy_obs[:, 1]
        distance = policy_obs[:, 2].clamp_min(0.0) * self.dist_scale

        # atan2 returns the LOS heading error already wrapped to [-pi, pi].
        error = torch.atan2(cross, dot)
        inside_zone = (
            torch.zeros_like(distance, dtype=torch.bool)
            if self.pass_through
            else distance <= self.zone_radius
        )
        active = ~inside_zone
        self.integral.add_(torch.where(active, error * self.dt, torch.zeros_like(error)))
        derivative = torch.where(
            self.initialized & active,
            (error - self.previous_error) / self.dt,
            torch.zeros_like(error),
        )
        yaw = (self.kp * error + self.ki * self.integral + self.kd * derivative).clamp(-1.0, 1.0)
        self.previous_error.copy_(error)
        self.initialized.fill_(True)

        # Two hold philosophies. "deadband" (frozen default, calm-water
        # heritage): zero thrust and WIPE PID memory inside the zone -- which
        # under any sustained current deletes the integral exactly where it is
        # needed and cycles the hull in/out of the zone forever (measured:
        # 0/32 at every Ki, even at 1.0 m/s). "continuous" (textbook dynamic
        # positioning): keep regulating inside the zone, keep the integral, so
        # the controller can hold a steady counter-thrust against the stream.
        if self.hold_mode == "deadband":
            yaw = torch.where(inside_zone, torch.zeros_like(yaw), yaw)
            self.integral[inside_zone] = 0.0
            self.previous_error[inside_zone] = 0.0
            self.initialized[inside_zone] = False
        else:
            # Anti-windup for the retained integral: clamp its yaw authority.
            self.integral.clamp_(-1.0 / max(self.ki, 1e-6), 1.0 / max(self.ki, 1e-6))

        if self.pass_through:
            # Gates are pass-through targets. Keep full thrust on the LOS and
            # smoothly throttle toward the existing 0.2 turn-in-place creep.
            thrust = 0.2 + 0.8 * torch.cos(error).clamp(0.0, 1.0)
        else:
            speed_profile = (
                (distance - self.zone_radius) / (self.slow_radius - self.zone_radius)
            ).clamp(0.0, 1.0)
            aligned = error.abs() < (math.pi / 4.0)
            thrust = torch.where(aligned, speed_profile, torch.full_like(speed_profile, 0.2))
            if self.hold_mode == "deadband":
                thrust = torch.where(inside_zone, torch.zeros_like(thrust), thrust)
            else:
                # Inside the zone: thrust proportional to remaining offset so a
                # stream-displaced hull leans on the water instead of coasting.
                inzone_thrust = (distance / max(self.zone_radius, 1e-6)).clamp(0.0, 0.6)
                thrust = torch.where(inside_zone, inzone_thrust, thrust)
        return torch.stack((thrust, yaw), dim=-1)

    def reset(self, done) -> None:
        mask = torch.as_tensor(done, device=self.integral.device).reshape(-1).bool()
        if mask.any():
            self.integral[mask] = 0.0
            self.previous_error[mask] = 0.0
            self.initialized[mask] = False


def _boat_bow_2d(base) -> torch.Tensor:
    """Return the world-frame boat bow direction.

    Prefers the environment's own ``_forward_2d`` helper -- every current
    docking env defines it and resolves the vehicle registry's per-hull
    ``bow_body_axis`` (BlueBoat is +X, the July-era WAM-V was -X). The first
    fallback rotates the env's published ``forward_vec``; the last-resort
    branch keeps the legacy hard-coded minus-X convention for old snapshots.
    """
    helper = getattr(base, "_forward_2d", None)
    if callable(helper):
        return helper()

    # Isaac Lab stores quaternions as (w, x, y, z).
    quat = base.robot.data.root_quat_w
    w, x, y, z = quat.unbind(dim=-1)
    forward_vec = getattr(base, "forward_vec", None)
    if forward_vec is not None:
        # Planar part of R(root_quat_w) @ forward_vec (body z-component is 0
        # for every surface hull in the registry).
        fx, fy = forward_vec[:, 0], forward_vec[:, 1]
        bow = torch.stack(
            (
                fx * (1.0 - 2.0 * (y * y + z * z)) + fy * (2.0 * (x * y - w * z)),
                fx * (2.0 * (x * y + w * z)) + fy * (1.0 - 2.0 * (x * x + z * z)),
            ),
            dim=-1,
        )
    else:
        # Legacy: R(root_quat_w) @ (-1, 0, 0), the July 2026 WAM-V convention.
        bow = torch.stack(
            (
                -(1.0 - 2.0 * (y * y + z * z)),
                -2.0 * (x * y + w * z),
            ),
            dim=-1,
        )
    return bow / torch.norm(bow, dim=-1, keepdim=True).clamp_min(1.0e-6)


def _linear_velocity_xy(base) -> torch.Tensor:
    velocity = base.robot.data.root_com_vel_w
    return velocity[:, :2]


class DockingController:
    """Three-phase boat docking controller with per-environment state."""

    APPROACH = 0
    ALIGN_BRAKE = 1
    HOLD = 2

    def __init__(
        self,
        base,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        dt: float,
        brake_radius: float,
        kp: float,
        ki: float,
        kd: float,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("Control timestep must be positive")
        if brake_radius <= 2.2:
            raise ValueError(
                f"--slow-radius ({brake_radius:g}) must exceed docking's 2.2 m alignment radius"
            )
        self.base = base
        self.dt = dt
        self.brake_radius = brake_radius
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.phase = torch.full(
            (num_envs,), self.APPROACH, device=device, dtype=torch.long
        )
        self.integral = torch.zeros(num_envs, device=device, dtype=dtype)
        self.previous_error = torch.zeros_like(self.integral)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def _pid(self, error: torch.Tensor, reset_mask: torch.Tensor) -> torch.Tensor:
        if reset_mask.any():
            self.integral[reset_mask] = 0.0
            self.previous_error[reset_mask] = 0.0
            self.initialized[reset_mask] = False

        self.integral.add_(error * self.dt)
        wrapped_delta = torch.remainder(
            error - self.previous_error + math.pi, 2.0 * math.pi
        ) - math.pi
        derivative = torch.where(
            self.initialized,
            wrapped_delta / self.dt,
            torch.zeros_like(error),
        )
        yaw = (self.kp * error + self.ki * self.integral + self.kd * derivative).clamp(
            -1.0, 1.0
        )
        self.previous_error.copy_(error)
        self.initialized.fill_(True)
        return yaw

    def act(self, obs) -> torch.Tensor:
        del obs
        position = self.base.robot.data.root_pos_w[:, :2]
        dock_point = self.base.dock_point
        dock_heading = self.base.dock_heading
        dock_heading = dock_heading / torch.norm(
            dock_heading, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        bow = _boat_bow_2d(self.base)
        velocity = _linear_velocity_xy(self.base)

        to_dock = dock_point - position
        distance = torch.norm(to_dock, dim=-1)
        direction = to_dock / distance.unsqueeze(-1).clamp_min(1.0e-6)
        planar_speed = torch.norm(velocity, dim=-1)
        forward_speed = torch.sum(velocity * bow, dim=-1)

        los_dot = torch.sum(bow * direction, dim=-1).clamp(-1.0, 1.0)
        los_cross = bow[:, 0] * direction[:, 1] - bow[:, 1] * direction[:, 0]
        los_error = torch.atan2(los_cross, los_dot)

        dock_dot = torch.sum(bow * dock_heading, dim=-1).clamp(-1.0, 1.0)
        dock_cross = (
            bow[:, 0] * dock_heading[:, 1] - bow[:, 1] * dock_heading[:, 0]
        )
        dock_error = torch.atan2(dock_cross, dock_dot)

        previous_phase = self.phase.clone()
        self.phase[
            (previous_phase == self.APPROACH) & (distance <= 2.2)
        ] = self.ALIGN_BRAKE

        hold_ready = (
            (distance <= 2.2)
            & (dock_error.abs() <= math.radians(10.0))
            & (planar_speed < 0.2)
        )
        self.phase[
            (previous_phase == self.ALIGN_BRAKE) & hold_ready
        ] = self.HOLD
        self.phase[
            (previous_phase == self.HOLD) & ~hold_ready
        ] = self.ALIGN_BRAKE
        self.phase[
            (previous_phase != self.APPROACH) & (distance > 3.0)
        ] = self.APPROACH

        yaw_error = torch.where(
            self.phase == self.APPROACH, los_error, dock_error
        )
        # Docking uses PD with MEASURED-rate damping instead of the shared
        # numerical-derivative PID: differentiating the error at 60 Hz
        # amplifies per-step noise 60x, saturates the +/-1 command, and pumps
        # a yaw limit cycle (probed: err swung +/-180 deg, 6.25% SR). The
        # physical yaw rate is clean. Gains are docking-specific (kp 1.0,
        # rate damping 1.0), validated by per-second probe traces.
        yaw_rate = self.base.robot.data.root_com_vel_w[:, 5]
        yaw = (1.0 * yaw_error - 1.0 * yaw_rate).clamp(-1.0, 1.0)
        # HOLD deadband: once parked within 8 deg, stop stirring -- calm water
        # keeps a parked boat parked; active yaw only reintroduces rate.
        in_deadband = (self.phase == self.HOLD) & (
            dock_error.abs() < math.radians(8.0)
        )
        yaw = torch.where(in_deadband, torch.zeros_like(yaw), yaw)

        # APPROACH: LOS steering with a linear braking taper from full thrust at
        # brake_radius to zero at 2.0 m. Thrust is reduced while turning.
        approach_thrust = (
            (distance - 2.0) / (self.brake_radius - 2.0)
        ).clamp(0.0, 1.0)
        approach_thrust *= los_dot.clamp(0.0, 1.0)

        # ALIGN+BRAKE: align-only. The boat reaches this phase at rest inside
        # the position tolerance; yaw does not translate, so zero thrust
        # preserves the parked state. Probing showed every active thrust
        # correction (speed brake / drift brake / edge nudge) reintroduces
        # velocity that breaks the 0.3 m/s hold condition.
        align_thrust = torch.zeros_like(forward_speed)

        thrust = torch.where(
            self.phase == self.APPROACH,
            approach_thrust,
            align_thrust,
        )
        thrust = torch.where(
            self.phase == self.HOLD, torch.zeros_like(thrust), thrust
        )
        return torch.stack((thrust, yaw), dim=-1)

    def reset(self, done) -> None:
        mask = torch.as_tensor(done, device=self.phase.device).reshape(-1).bool()
        if mask.any():
            self.phase[mask] = self.APPROACH
            self.integral[mask] = 0.0
            self.previous_error[mask] = 0.0
            self.initialized[mask] = False


def _initial_distances(base, num_envs: int, device) -> torch.Tensor:
    helper = getattr(base, "_horizontal_distance", None)
    if helper is None:
        return torch.full((num_envs,), torch.nan, device=device)
    try:
        distances = torch.as_tensor(helper(), device=device, dtype=torch.float32).reshape(-1)
        if distances.numel() != num_envs:
            raise ValueError(f"expected {num_envs} distances, got {distances.numel()}")
        return distances.clone()
    except Exception as exc:
        print(f"[WARN] Could not read initial horizontal distance ({exc}); SPL will be nan.")
        return torch.full((num_envs,), torch.nan, device=device)


def _values_at(base, name: str, env_ids: torch.Tensor, default) -> list:
    value = getattr(base, name, None)
    if value is None:
        return [default] * int(env_ids.numel())
    try:
        tensor = torch.as_tensor(value, device=env_ids.device).reshape(-1)
        return tensor[env_ids].detach().cpu().tolist()
    except Exception:
        return [default] * int(env_ids.numel())


def _completed_path_lengths(base, env_ids: torch.Tensor) -> list[float]:
    name = "episode_path_length" if hasattr(base, "episode_path_length") else "path_length"
    return [float(value) for value in _values_at(base, name, env_ids, math.nan)]


def _append_csv(path: str, repro: dict[str, object], metrics: dict[str, object]) -> None:
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    common_fields = [
        "task",
        "mode",
        "controller",
        "seed",
        "num_envs",
        "dist_scale",
        "slow_radius",
        "kp",
        "ki",
        "kd",
    ]
    if repro["protocol"] == "throughput":
        metric_fields = [
            "eval_steps",
            "targets_per_episode",
            "mean_speed",
            "oob_per_episode",
            "total_targets",
            "episode_equivalents",
        ]
    else:
        metric_fields = [
            "episodes",
            "sr",
            "spl",
            "mean_time_to_success_s",
            "median_time_to_success_s",
            "successes",
            "failures_timeout",
            "failures_other",
        ]
        if repro["mode"] == "path_follow":
            metric_fields.append("gates_passed_distribution")
    fields = common_fields + metric_fields + ["cfg_sha1", "git_commit"]

    row = {**repro, **metrics}
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [f"# {key}={value}" if index == 0 else f"{key}={value}" for index, (key, value) in enumerate(repro.items())]
        )
        if new_file:
            writer.writerow(fields)
        writer.writerow([row[field] for field in fields])


def _run_throughput(env, base, controller, obs) -> dict[str, object]:
    max_ep = float(getattr(base, "max_episode_length", 0)) or float(args_cli.eval_steps)
    reached_total = 0
    reached_prev = int(getattr(base, "reached_count", 0))
    n_oob = 0
    speed_sum, speed_n = 0.0, 0

    print(f"[INFO] Evaluating {args_cli.eval_steps} steps x {args_cli.num_envs} envs (classical deterministic)...")
    for _ in range(args_cli.eval_steps):
        with torch.inference_mode():
            actions = controller.act(obs)
            obs, _, terminated, truncated, _ = env.step(actions)

        cur = int(getattr(base, "reached_count", 0))
        delta = cur - reached_prev
        if delta > 0:
            reached_total += delta
        reached_prev = cur

        try:
            vel = base.robot.data.root_com_vel_w
            speed_sum += torch.norm(vel[:, :2], dim=-1).mean().item()
            speed_n += 1
        except Exception:
            pass

        terminated = torch.as_tensor(terminated).reshape(-1)
        truncated = torch.as_tensor(truncated).reshape(-1)
        n_oob += int(terminated.sum().item())
        controller.reset(terminated.bool() | truncated.bool())

    ep_equiv = (args_cli.num_envs * args_cli.eval_steps) / max(max_ep, 1.0)
    return {
        "eval_steps": args_cli.eval_steps,
        "targets_per_episode": reached_total / max(ep_equiv, 1.0e-9),
        "mean_speed": speed_sum / max(speed_n, 1),
        "oob_per_episode": n_oob / max(ep_equiv, 1.0e-9),
        "total_targets": reached_total,
        "episode_equivalents": ep_equiv,
        "max_episode_length": max_ep,
    }


def _run_mission(
    env,
    base,
    controller,
    mode: str,
    zone_radius: float,
    device,
    obs,
    cert,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    num_envs = int(base.num_envs)
    is_path_follow = mode == "path_follow"
    if not is_path_follow and not hasattr(base, "_horizontal_distance"):
        print("[WARN] Mission env has no _horizontal_distance(); SPL will be nan.")

    initial_distance = None if is_path_follow else _initial_distances(base, num_envs, device)
    completed = 0
    successes = 0
    failures_timeout = 0
    failures_other = 0
    spl_values = []
    success_times = []
    gates_passed = []
    # Per-episode records in the scripts/eval_v6_frozen.py schema, so
    # usv10k_score.py and the census tooling ingest the classical column
    # unchanged.
    records: list[dict[str, object]] = []
    ep_counter = torch.zeros(num_envs, dtype=torch.long)
    step = 0
    quota = cert.quota
    max_steps = (quota + 3) * int(
        getattr(base, "max_episode_length", 0) or args_cli.episodes * 10_000
    )

    print(
        f"[INFO] Evaluating {quota} episodes per env ({args_cli.episodes} requested) x "
        f"{num_envs} envs (classical deterministic)..."
    )
    while bool((ep_counter < quota).any()) and step < max_steps:
        # d0_per_env is rewritten inside _reset_idx, which runs during step(),
        # so reading it alongside the other episode stats would report the NEXT
        # episode's straight-line distance. Snapshot it while it still belongs
        # to the episode that is about to end. (No current classical-column
        # task publishes d0_per_env; the guard keeps the record schema aligned
        # with eval_v6_frozen wherever the field exists.)
        d0_prev = base.d0_per_env.clone() if hasattr(base, "d0_per_env") else None
        with torch.inference_mode():
            with cert.time_controller():
                actions = controller.act(obs)
            with cert.time_environment():
                obs, _, terminated, truncated, _ = env.step(actions)
        cert.after_step()
        step += 1

        terminated = torch.as_tensor(terminated, device=device).reshape(-1).bool()
        truncated = torch.as_tensor(truncated, device=device).reshape(-1).bool()
        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
        controller.reset(terminated | truncated)
        if done_ids.numel() == 0:
            continue

        # Each env contributes the same quota regardless of completion order.
        counted_ids = torch.as_tensor(
            [i for i in done_ids.tolist() if int(ep_counter[i]) < quota],
            dtype=torch.long, device=device,
        )
        for env_id in done_ids.tolist():
            if int(ep_counter[env_id]) >= quota:
                cert.discard_episode(env_id)
        counted_terminated = terminated[counted_ids].detach().cpu().tolist()
        counted_truncated = truncated[counted_ids].detach().cpu().tolist()
        episode_success = [bool(value) for value in _values_at(base, "episode_success", counted_ids, False)]
        episode_paths = _completed_path_lengths(base, counted_ids)
        episode_times = [float(value) for value in _values_at(base, "time_to_success", counted_ids, math.nan)]
        if is_path_follow:
            episode_shortest_paths = [
                float(value) for value in _values_at(base, "episode_route_length", counted_ids, math.nan)
            ]
            gates_passed.extend(
                int(value) for value in _values_at(base, "episode_gates_passed", counted_ids, 0)
            )
        else:
            episode_shortest_paths = initial_distance[counted_ids].detach().cpu().tolist()

        counted_env_ids = counted_ids.detach().cpu().tolist()
        for env_id, success, timed_out, terminated_flag, shortest_path_source, path_length, time_s in zip(
            counted_env_ids,
            episode_success,
            counted_truncated,
            counted_terminated,
            episode_shortest_paths,
            episode_paths,
            episode_times,
        ):
            if success:
                successes += 1
                if math.isfinite(time_s):
                    success_times.append(time_s)
            elif timed_out:
                failures_timeout += 1
            elif terminated_flag:
                failures_other += 1
            else:
                failures_other += 1

            if math.isfinite(shortest_path_source) and math.isfinite(path_length):
                shortest_path = (
                    float(shortest_path_source)
                    if is_path_follow
                    else max(float(shortest_path_source) - zone_radius, 0.0)
                )
                denominator = max(float(path_length), shortest_path)
                spl = shortest_path / denominator if success and denominator > 0.0 else 0.0
                spl_values.append(spl)
            else:
                spl_values.append(math.nan)

            # eval_v6_frozen record schema, field for field.
            rec = {
                "env": int(env_id),
                "ep": int(ep_counter[env_id]),
                "success": bool(success),
                "terminated": bool(terminated_flag),
                "truncated": bool(timed_out),
                "tts_s": (None if math.isnan(float(time_s)) else float(time_s)),
                "path_length_m": float(path_length),
            }
            # Station keeping and path following have no obstacle field, hence
            # no clearance buffer; every hazard/docking family records it.
            # Longest continuous hold, the graded quantity the binary success
            # criterion thresholds. Guarded by hasattr so families without a
            # hold phase are unchanged.
            if hasattr(base, "episode_max_hold_s"):
                rec["max_hold_s"] = float(base.episode_max_hold_s[env_id])
            if hasattr(base, "episode_station_error_rms_m"):
                rec["station_error_rms_m"] = float(
                    base.episode_station_error_rms_m[env_id]
                )
                rec["station_error_p95_m"] = float(
                    base.episode_station_error_p95_m[env_id]
                )
                rec["station_error_final_60s_rms_m"] = float(
                    base.episode_station_error_final_60s_rms_m[env_id]
                )
                rec["station_error_final_60s_p95_m"] = float(
                    base.episode_station_error_final_60s_p95_m[env_id]
                )
            if hasattr(base, "episode_min_clearance"):
                rec["min_clearance_m"] = float(base.episode_min_clearance[env_id])
            # Cross-track error, mirrored from scripts/eval_v6_frozen.py so every
            # controller writes the same record schema (the parity tests enforce it).
            if hasattr(base, "episode_xte_rms"):
                rec["xte_rms_m"] = float(base.episode_xte_rms[env_id])
            # Per-primitive digests of the scenario the FINISHED episode ran,
            # latched at reset exactly like episode_min_clearance.  Absent on
            # families not yet on the scenario protocol, and empty until an env
            # has completed its first episode; both cases are guarded inside
            # the helper, which returns None for "write no field".
            scenario_hashes = episode_scenario_hashes_for(base, env_id)
            if scenario_hashes is not None:
                rec["scenario_hashes"] = scenario_hashes
            episode_contact_steps = getattr(base, "episode_contact_steps", None)
            episode_contact_longest_steps = getattr(
                base, "episode_contact_longest_steps", None
            )
            episode_contact_depth_sum = getattr(
                base, "episode_contact_depth_sum", None
            )
            if (
                episode_contact_steps is not None
                and episode_contact_longest_steps is not None
                and episode_contact_depth_sum is not None
            ):
                contact_steps = float(episode_contact_steps[env_id])
                rec["contact_steps"] = int(contact_steps)
                rec["contact_seconds"] = contact_steps * base.control_step_s
                rec["contact_longest_seconds"] = (
                    float(episode_contact_longest_steps[env_id])
                    * base.control_step_s
                )
                rec["contact_depth_mean_m"] = (
                    float(episode_contact_depth_sum[env_id]) / contact_steps
                    if contact_steps > 0.0
                    else 0.0
                )
            episode_max_phase = getattr(base, "episode_max_phase", None)
            if episode_max_phase is not None:
                rec["max_phase"] = int(episode_max_phase[env_id])
            if hasattr(base, "episode_gates_passed"):
                rec["gates"] = int(base.episode_gates_passed[env_id])
            if d0_prev is not None:
                rec["d0_m"] = float(d0_prev[env_id])
            elif not is_path_follow:
                rec["d0_m"] = float(initial_distance[env_id])
            if is_path_follow and math.isfinite(shortest_path_source):
                rec["route_geodesic_m"] = float(shortest_path_source)
            # The shortest feasible route for THIS layout, wherever the env
            # has latched it (hazard families); absent on the classical tasks.
            if hasattr(base, "route_geodesic_length"):
                geodesic = float(base.route_geodesic_length[env_id])
                if geodesic > 0.0:
                    rec["route_geodesic_m"] = geodesic
            records.append(cert.finish_episode(env_id, rec))
            ep_counter[env_id] += 1

        completed += int(counted_ids.numel())
        if not is_path_follow:
            next_distance = _initial_distances(base, num_envs, device)
            initial_distance[done_ids] = next_distance[done_ids]

    if not bool((ep_counter >= quota).all()):
        print(f"[INCOMPLETE] step budget exhausted: {completed}/{num_envs * quota} episodes")
    metrics = {
        "episodes": completed,
        "episodes_requested": int(args_cli.episodes),
        "episodes_per_env": int(quota),
        "complete": bool((ep_counter >= quota).all()),
        "sr": successes / completed if completed else math.nan,
        "spl": sum(spl_values) / completed if completed and all(math.isfinite(value) for value in spl_values) else math.nan,
        "mean_time_to_success_s": sum(success_times) / len(success_times) if success_times else math.nan,
        "median_time_to_success_s": statistics.median(success_times) if success_times else math.nan,
        "successes": successes,
        "failures_timeout": failures_timeout,
        "failures_other": failures_other,
    }
    if is_path_follow:
        num_waypoints = int(getattr(getattr(base, "cfg", None), "num_waypoints", 4))
        counts = {gate: 0 for gate in range(num_waypoints + 1)}
        for gate in gates_passed:
            counts[gate] = counts.get(gate, 0) + 1
        metrics["gates_passed_distribution"] = ",".join(
            f"{gate}:{count}" for gate, count in sorted(counts.items())
        )
    return metrics, records


def _print_summary(metrics: dict[str, object], controller_name: str) -> None:
    print("\n" + "=" * 64)
    print(f"  USVBench classical {args_cli.protocol} eval - {args_cli.task}")
    print(f"  controller: {controller_name}  seed={args_cli.seed}")
    print("-" * 64)
    if args_cli.protocol == "throughput":
        print(f"  targets_per_episode : {metrics['targets_per_episode']:7.2f}   (primary score)")
        print(f"  mean_speed (m/s)    : {metrics['mean_speed']:7.2f}")
        print(f"  oob_per_episode     : {metrics['oob_per_episode']:7.2f}   (out-of-bounds events / episode)")
        print(f"  total_targets       : {metrics['total_targets']}")
        print(
            f"  episode_equivalents : {metrics['episode_equivalents']:7.1f}"
            f"   (ep_len={metrics['max_episode_length']:.0f} steps)"
        )
    else:
        completed = metrics["episodes"]
        print(f"  SR                         : {metrics['sr']:8.4f}  ({metrics['successes']}/{completed})")
        print(f"  SPL                        : {metrics['spl']:8.4f}")
        print(f"  mean time to success (s)   : {metrics['mean_time_to_success_s']:8.3f}")
        print(f"  median time to success (s) : {metrics['median_time_to_success_s']:8.3f}")
        print(f"  failures - timeout         : {metrics['failures_timeout']:8d}")
        print(f"  failures - other           : {metrics['failures_other']:8d}")
        if "gates_passed_distribution" in metrics:
            print(f"  gates passed distribution  : {metrics['gates_passed_distribution']}")
    print("=" * 64 + "\n")
    # Isaac/Kit shutdown can swallow buffered output on some machines.
    sys.stdout.flush()


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg):
    applied_overrides = resolve_protocol(env_cfg, args_cli)
    experiment_cfg["seed"] = env_cfg.seed

    # Fingerprint AFTER the overrides so cfg_sha1 describes the run as
    # configured (the resurrected script hashed before applying overrides).
    cfg_sha1, cfg_scalars = _cfg_fingerprint(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    base = _base_env(env)
    device = torch.device(getattr(base, "device", env_cfg.sim.device))
    zone_radius = _radius_from_env(base, env_cfg)
    mode = _task_mode(base, env_cfg)
    if args_cli.protocol == "mission" and not hasattr(base, "episode_success"):
        raise RuntimeError(
            f"Task {args_cli.task} does not publish completed-episode stats "
            "(episode_success/time_to_success/episode_path_length); use "
            "--protocol throughput. The point-nav family (blueboat_calm_nav) "
            "respawns targets in place and is a throughput task."
        )
    default_dist_scale = {
        "point_nav": 30.0,
        "path_follow": 20.0,
        "station_keep": 15.0,
        "docking": 25.0,
    }[mode]
    dist_scale = args_cli.dist_scale if args_cli.dist_scale is not None else default_dist_scale
    slow_radius = (
        args_cli.slow_radius
        if args_cli.slow_radius is not None
        else (6.0 if mode == "docking" else 5.0)
    )
    controller_name = "docking_3phase_pid" if mode == "docking" else "los_pid"
    journal_block = None
    if JOURNAL is not None:
        import journal_avoidance as ja
        import journal_controllers as jc

        # G24b: the journal hold mode (station keeping: continuous_v2; other modes keep the request / deadband).
        try:
            args_cli.hold_mode = ja.journal_hold_mode(mode, args_cli.hold_mode)
        except ValueError as exc:
            parser.error(str(exc))
        vessel = str(env_cfg.vehicle)
        journal_gains = jc.load_gains(vessel)
        journal_scale = jc.length_scale(vessel)
        if args_cli.slow_radius is None:
            slow_radius = slow_radius * journal_scale
        journal_tolerance = float(getattr(env_cfg, "success_position_tolerance_m", 2.5 * journal_scale))
        journal_block = jc.controller_stamp(env_cfg, journal_gains, {
            "mode": mode, "hold_mode": args_cli.hold_mode, "zone_radius_m": zone_radius,
            "slow_radius_m": slow_radius, "docking_tolerance_m": journal_tolerance,
            "controller_class": "journal_controllers.JournalBaselineController"})
        if mode in JOURNAL_AVOIDANCE_MODES:
            # G17 (owner decision B10 supplement, option B): one controller per journal task for all
            # its cells, the unchanged JournalBaselineController wrapped by the obstacle-avoidance
            # layer (inactive unless the rays show an obstacle near the vessel). The layer's frozen
            # constants (scripts/classical_avoidance_v2.json) go into the controller block.
            journal_avoidance, journal_block["avoidance"] = ja.prepare(
                env_cfg, mode, journal_gains, dt=_control_step_s(env_cfg), zone_radius_m=zone_radius,
                slow_radius_m=slow_radius, tolerance_m=journal_tolerance, scale=journal_scale)
            journal_block["controller_class"] = "journal_avoidance.AvoidanceLayerController"
            controller_name = ja.CONTROLLER_NAMES[mode]
    cert = None
    if args_cli.protocol == "mission" and journal_block is not None:
        cert = CertificateRun(
            base=base, env_cfg=env_cfg, evaluator=__file__, task=args_cli.task,
            controller=controller_name,
            controller_config={**journal_block, "protocol": args_cli.protocol,
                               "num_envs": int(base.num_envs), "device": str(device),
                               "dt": _control_step_s(env_cfg), "action_limits": [-1.0, 1.0]},
            checkpoint=None, eval_seed=args_cli.seed, level=args_cli.level,
            episodes_requested=args_cli.episodes, overrides=applied_overrides,
            argv=ORIGINAL_ARGV, cli_args=vars(args_cli),
        )
    elif args_cli.protocol == "mission":
        cert = CertificateRun(
            base=base, env_cfg=env_cfg, evaluator=__file__, task=args_cli.task,
            controller=controller_name,
            controller_config={"mode": mode, "protocol": args_cli.protocol,
                               "num_envs": int(base.num_envs), "device": str(device),
                               "dt": _control_step_s(env_cfg),
                               "kp": args_cli.kp, "ki": args_cli.ki, "kd": args_cli.kd,
                               "dist_scale": dist_scale, "zone_radius": zone_radius,
                               "slow_radius": slow_radius, "brake_radius": slow_radius,
                               "hold_mode": args_cli.hold_mode,
                               "pass_through": mode == "path_follow",
                               "action_limits": [-1.0, 1.0],
                               "fixed_control_parameters": (
                                   {"heading_control": "measured_rate_pd",
                                    "yaw_error_gain": 1.0, "yaw_rate_damping": 1.0,
                                    "cli_pid_gains_used_for_act": False,
                                    "alignment_radius_m": 2.2,
                                    "hold_heading_tolerance_deg": 10.0,
                                    "hold_speed_threshold_mps": 0.2,
                                    "resume_approach_radius_m": 3.0,
                                    "yaw_hold_deadband_deg": 8.0,
                                    "approach_stop_radius_m": 2.0,
                                    "align_and_hold_thrust": 0.0}
                                   if mode == "docking" else
                                   {"heading_control": "numerical_derivative_pid",
                                    "turn_creep_thrust": 0.2,
                                    "pass_through_aligned_thrust_gain": 0.8,
                                    "aligned_heading_limit_rad": math.pi / 4.0,
                                    "continuous_hold_thrust_cap": 0.6,
                                    "integral_yaw_authority_limit": 1.0})},
            checkpoint=None, eval_seed=args_cli.seed, level=args_cli.level,
            episodes_requested=args_cli.episodes, overrides=applied_overrides,
            argv=ORIGINAL_ARGV, cli_args=vars(args_cli),
        )
    obs, _ = env.reset()
    if cert is not None:
        cert.after_reset()
    obs_tensor = _policy_obs(obs)
    if cert is not None:
        cert.header["controller"]["configuration"]["dtype"] = str(obs_tensor.dtype)

    if journal_block is not None and "avoidance" in journal_block:
        controller = ja.build_controller(
            base, env_cfg, mode, journal_gains, journal_avoidance, dt=_control_step_s(env_cfg),
            zone_radius_m=zone_radius, slow_radius_m=slow_radius, tolerance_m=journal_tolerance,
            scale=journal_scale, hold_mode=args_cli.hold_mode, dtype=obs_tensor.dtype)
        if list(controller.hold_zone) != journal_block["avoidance"]["hold_zone_m"]:
            raise RuntimeError("avoidance hold zone differs from the certificate block")
    elif journal_block is not None:
        controller = jc.JournalBaselineController(
            base, mode, journal_gains, dt=_control_step_s(env_cfg), zone_radius_m=zone_radius,
            slow_radius_m=slow_radius, tolerance_m=journal_tolerance, scale=journal_scale,
            hold_mode=args_cli.hold_mode, dtype=obs_tensor.dtype)
    elif mode == "docking":
        controller = DockingController(
            base=base,
            num_envs=int(base.num_envs),
            device=obs_tensor.device,
            dtype=obs_tensor.dtype,
            dt=_control_step_s(env_cfg),
            brake_radius=slow_radius,
            kp=args_cli.kp,
            ki=args_cli.ki,
            kd=args_cli.kd,
        )
        controller_name = "docking_3phase_pid"
    else:
        controller = LOSPIDController(
            num_envs=int(base.num_envs),
            device=obs_tensor.device,
            dtype=obs_tensor.dtype,
            dt=_control_step_s(env_cfg),
            dist_scale=dist_scale,
            zone_radius=zone_radius,
            hold_mode=args_cli.hold_mode,
            slow_radius=slow_radius,
            kp=args_cli.kp,
            ki=args_cli.ki,
            kd=args_cli.kd,
            pass_through=mode == "path_follow",
        )
        controller_name = "los_pid"

    # Does THIS env carry the scenario protocol?  Same guard, same marker and
    # same warning as scripts/eval_v6_frozen.py:148-162, so a classical
    # certificate and a policy certificate at the same evaluation seed can be
    # scenario-paired instead of merely assumed comparable.
    scenario_protocol = scenario_protocol_stamp(base)
    notice = scenario_protocol_notice(args_cli.task, scenario_protocol)
    if notice:
        print(notice, flush=True)

    repro = {
        "git_commit": _git_commit(),
        "task": args_cli.task,
        "mode": mode,
        "controller": controller_name,
        "protocol": args_cli.protocol,
        "num_envs": args_cli.num_envs,
        "episodes" if args_cli.protocol == "mission" else "eval_steps": (
            args_cli.episodes if args_cli.protocol == "mission" else args_cli.eval_steps
        ),
        "seed": args_cli.seed,
        "level": args_cli.level,
        "dist_scale": dist_scale,
        "slow_radius": slow_radius,
        "kp": args_cli.kp,
        "ki": args_cli.ki,
        "kd": args_cli.kd,
        "cfg_sha1": cfg_sha1,
    }
    print("[REPRO] " + "  ".join(f"{key}={value}" for key, value in repro.items()))
    print("[REPRO] cfg_scalars=" + json.dumps(cfg_scalars, sort_keys=True, separators=(",", ":")))

    if args_cli.protocol == "throughput":
        metrics = _run_throughput(env, base, controller, obs)
        records = []
    else:
        metrics, records = _run_mission(
            env, base, controller, mode, zone_radius, device, obs, cert
        )
    if journal_block is not None and "avoidance" in journal_block:
        # G17: the layer's run telemetry (activations, active steps, plans; episodes with an
        # activation) inside the controller block, written before the certificate is built.
        telemetry = controller.telemetry()
        if cert is not None:
            cert.header["controller"]["configuration"]["avoidance"]["telemetry"] = telemetry
        print("[AVOID] " + "  ".join(f"{key}={telemetry[key]}" for key in (
            "activations", "handbacks", "active_steps", "plans", "plan_failures", "fallback_steps",
            "episodes_with_activation", "episodes_finished")), flush=True)

    if args_cli.csv:
        _append_csv(args_cli.csv, repro, metrics)
        print(f"[INFO] appended result to {args_cli.csv}")

    if args_cli.out:
        # Same top-level layout as scripts/eval_v6_frozen.py, with the
        # controller identity and gains standing where a learned policy's
        # checkpoint path would.
        with _EXIT_GUARD.output(args_cli.out).open() as stream:
            json.dump(
                cert.build({
                    "task": args_cli.task,
                    "num_envs": int(base.num_envs),
                    "episodes_requested": int(args_cli.episodes),
                    "episodes_per_env": metrics["episodes_per_env"],
                    "complete": metrics["complete"],
                    "level": args_cli.level,
                    "seed": args_cli.seed,
                    # The header block when the env carries the protocol
                    # object, the off-protocol literal when it does not --
                    # never unconditionally the header.
                    "scenario_protocol": scenario_protocol,
                    # Every --set actually written to the cfg, under the same
                    # key scripts/eval_v6_frozen.py:340 uses so one schema
                    # serves both. A wave-ladder cell is identified by this
                    # field, not by whatever the caller named the file.
                    "overrides": applied_overrides,
                    "controller": controller_name,
                    "gains": {
                        "kp": args_cli.kp,
                        "ki": args_cli.ki,
                        "kd": args_cli.kd,
                        "dist_scale": dist_scale,
                        "slow_radius": slow_radius,
                    },
                    "cfg_sha1": cfg_sha1,
                    "git_commit": repro["git_commit"],
                    "records": records,
                }),
                stream,
                indent=1,
            )
        print(f"  records -> {args_cli.out}")

    _print_summary(metrics, controller_name)
    env.close()


if __name__ == "__main__":
    # G6b: an exception inside the Hydra-decorated main must reach the guard as itself.
    # hydra's run_and_report (hydra/_internal/utils.py) re-raises it when HYDRA_FULL_ERROR=1;
    # otherwise it prints it and calls sys.exit(1), which Kit could intercept. The guard
    # then takes the failure path (no close(), os._exit(1)); success publishes, then closes.
    os.environ["HYDRA_FULL_ERROR"] = "1"
    evaluator_exit.run(main, _EXIT_GUARD, app=simulation_app)
