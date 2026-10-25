"""Staged classical controller for the journal harbour mission (F7, CONTEXT_R4).

Journal ids only (Isaac-USV-HarborMission-<BlueBoat|WAMVVRX>-Journal-Direct-v1,
carrier harbor_mission, contract journal_contracts_v2.HarborMissionEpisode:
ordered departure gates, the transit exit gate, then berth entry + dwell).

  departure / transit  ray-map planner + LOS/PID (classical_planner_pid core,
                       journal_ladder inflation) through every remaining
                       virtual gate, each as a pre-point and a post-point on
                       its normal (the contract arms a gate only after the COM
                       was at least gate_approach_m before its plane and
                       credits a crossing only in the +normal direction)
  docking              line up on the berth mouth (pre-point on the dock
                       axis, LOS/PID), then the three-phase docking law
                       (journal_controllers.DockingLaw) on the berth pose
  complete             zero action

INFORMATION DIET: journal_controllers.INFORMATION_DIET (the D5 estimate +
heading + task targets; obstacles only from the 36-ray block) plus F6's phase
targets (task geometry and contract progress latches). It never reads D7
objects, the static obstacle list (not even the phase-target dict's
``static_obstacles`` chart: obstacles come from the rays, as for every journal
controller), sea, current or wind. Gains: the frozen per-vessel values of
scripts/classical_gains_v2.json.

PHASE-TARGET INTERFACE: F6 ``env.journal_phase_targets()``, schema
``usvbench.harbour_phase_targets/1`` (20260930_f6_harbor_mission/work/
PHASE_TARGETS_F6.md; tasks/harbor_mission/harbor_journal.py phase_targets).
Keys used: stage [N] (0 departure, 1 transit, 2 docking, 3 complete),
active_gate_index [N] (0..3; 4 = berth entry latched), active_gate_midpoint /
active_gate_normal [N,2], active_gate_half_width_m [N], gate_midpoints /
gate_normals [N,4,2], remaining_gates bool [N,4], gate_approach_m [N,3],
berth_pose [N,3] (x, y, dock heading), berth_interior [N,3]; frame
``env_local_com_xy`` (the frame of est.position). ``phase_targets_world``
fails closed on another schema or missing keys. Coded against F6's document;
F6's env code itself was not run here (UNVERIFIED end to end).

CPU checks: python -B scripts/classical_harbour.py --selftest ; tests in
scripts/test_classical_harbour.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import NamedTuple

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from cfg_override import validate_assignments  # noqa: E402

# G6: fail-closed exit codes and certificate publication (scripts/evaluator_exit.py).
import evaluator_exit  # noqa: E402

_EXIT_GUARD = evaluator_exit.Guard()

STAGE_DEPARTURE, STAGE_TRANSIT, STAGE_DOCKING, STAGE_COMPLETE = 0, 1, 2, 3
BERTH_MOUTH, NO_GATE = 3, 4
PHASE_TARGETS_SCHEMA = "usvbench.harbour_phase_targets/1"
PHASE_TARGET_KEYS = ("stage", "active_gate_index", "active_gate_midpoint", "active_gate_normal",
                     "active_gate_half_width_m", "gate_midpoints", "gate_normals", "remaining_gates",
                     "gate_approach_m", "berth_pose", "berth_interior")
PHASE_TARGETS_STATUS = ("F6 PHASE_TARGETS_F6.md schema usvbench.harbour_phase_targets/1; F6 env code not "
                        "executed by F7 (UNVERIFIED end to end)")
CONTROLLER_NAME = "staged_harbour"


class PhaseTargets(NamedTuple):
    """World-frame phase targets (F6 schema, env origin added)."""

    stage: object               # [N] long
    active_index: object        # [N] long, 0..3, 4 = berth entry latched
    gate_midpoint: object       # [N,2] active gate midpoint (berth centre for index 4)
    gate_normal: object         # [N,2] active gate normal (dock heading for index 4)
    gate_half_width: object     # [N]
    gate_midpoints: object      # [N,4,2]
    gate_normals: object        # [N,4,2]
    remaining: object           # [N,4] bool
    gate_approach: object       # [N,3]
    berth_point: object         # [N,2]
    berth_heading: object       # [N,2] unit
    berth_half_width: object    # [N] slip inner half width

    STAGE_DOCKING = STAGE_DOCKING


def phase_targets_world(env):
    """env.journal_phase_targets() (F6 schema) as world-frame tensors (fail closed)."""
    import torch

    getter = getattr(env, "journal_phase_targets", None)
    if not callable(getter):
        raise RuntimeError("the journal harbour env publishes no journal_phase_targets() (" + PHASE_TARGETS_STATUS + ")")
    raw = getter()
    if not isinstance(raw, dict) or raw.get("schema") != PHASE_TARGETS_SCHEMA:
        raise RuntimeError("journal_phase_targets() schema %r is not %s"
                           % (raw.get("schema") if isinstance(raw, dict) else type(raw), PHASE_TARGETS_SCHEMA))
    missing = [key for key in PHASE_TARGET_KEYS if raw.get(key) is None]
    if missing:
        raise RuntimeError("journal_phase_targets() lacks %s (%s)" % (missing, PHASE_TARGETS_STATUS))
    if raw.get("frame") != "env_local_com_xy":
        raise RuntimeError("journal_phase_targets() frame must be 'env_local_com_xy', got %r" % raw.get("frame"))
    origin = env.scene.env_origins[:, :2].to(torch.float64)
    n = origin.shape[0]

    def tensor(key, dtype=torch.float64):
        return torch.as_tensor(raw[key], device=origin.device).to(dtype)

    def unit(value):
        return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1.0e-9)

    berth = tensor("berth_pose")
    mids = tensor("gate_midpoints")
    if mids.shape != (n, 4, 2) or berth.shape != (n, 3):
        raise RuntimeError("gate_midpoints must be [N,4,2] and berth_pose [N,3]")
    heading = torch.stack((torch.cos(berth[:, 2]), torch.sin(berth[:, 2])), -1)
    return PhaseTargets(
        tensor("stage", torch.long).reshape(n), tensor("active_gate_index", torch.long).reshape(n),
        tensor("active_gate_midpoint") + origin, unit(tensor("active_gate_normal")),
        tensor("active_gate_half_width_m").reshape(n), mids + origin[:, None, :], unit(tensor("gate_normals")),
        tensor("remaining_gates", torch.bool), tensor("gate_approach_m"), berth[:, :2] + origin, heading,
        tensor("berth_interior")[:, 2])


class StagedHarbourController:
    """Per-env staged controller (see the module docstring); pure torch/numpy."""

    def __init__(self, base, *, gains, max_range_m, scale, replan_every=60, waypoint_spacing_m=2.0,
                 accept_radius_m=1.0, plan_cell_m=0.25, gate_offset_m=3.0, ladder=None, tolerance_m=None):
        import torch

        import journal_controllers as jc
        from classical_planner_pid import journal_ladder

        if replan_every < 1:
            raise ValueError("replan_every must be >= 1")
        self.base = base
        self.num_envs = int(base.num_envs)
        self.device = base.device
        self.dt = float(base.control_step_s)
        self.scale = float(scale)
        self.gains = gains
        self.max_range_m = float(max_range_m)
        self.replan_every = int(replan_every)
        self.spacing = float(waypoint_spacing_m) * self.scale
        self.accept = float(accept_radius_m) * self.scale
        self.cell = float(plan_cell_m) * self.scale
        self.gate_offset = float(gate_offset_m) * self.scale
        vessel = str(base.cfg.vehicle)
        self.ladder = tuple(ladder) if ladder is not None else journal_ladder(vessel)
        if tolerance_m is None:
            tolerance_m = float(getattr(base.cfg, "success_position_tolerance_m", 2.5 * self.scale))
        dtype = torch.float32
        self.los = jc.LOSController(self.num_envs, self.device, dtype, self.dt, gains)
        self.dock = jc.DockingLaw(self.num_envs, self.device, dtype, self.dt, gains,
                                  tolerance_m=tolerance_m, scale=self.scale)
        self.tracker = jc.WaypointTracker(self.num_envs, self.accept)
        self.replan_in = np.zeros(self.num_envs, dtype=np.int64)
        self.planned_key = [None] * self.num_envs
        self.lined_up = [False] * self.num_envs
        self.plan_calls = 0
        self.straight_fallbacks = 0

    def _offset(self, phase, i, k):
        """Pre/post offset of gate k: the declared offset, at least 1.5 x its arming distance."""
        approach = float(phase.gate_approach[i, k]) if k < phase.gate_approach.shape[1] else 0.0
        return max(self.gate_offset, 1.5 * approach)

    def route_points(self, phase, i, position=None):
        """World points of env i for its current stage.

        Departure / transit: a pre-point and a post-point on the normal of
        every remaining virtual gate (0..2), in order; the ACTIVE gate's
        pre-point is dropped once the hull is less than half the offset behind
        its plane. Docking before the line-up: the berth-mouth pre-point.
        """
        remaining = phase.remaining[i].detach().cpu().tolist()
        mids = phase.gate_midpoints[i].detach().cpu().double().numpy()
        normals = phase.gate_normals[i].detach().cpu().double().numpy()
        stage = int(phase.stage[i])
        points = []
        if stage in (STAGE_DEPARTURE, STAGE_TRANSIT):
            first = True
            for k in range(BERTH_MOUTH):
                if not remaining[k]:
                    continue
                offset = self._offset(phase, i, k)
                signed = None if position is None else float(np.dot(np.asarray(position) - mids[k], normals[k]))
                if not (first and signed is not None and signed >= -0.5 * offset):
                    points.append(mids[k] - normals[k] * offset)
                points.append(mids[k] + normals[k] * offset)
                first = False
        elif stage == STAGE_DOCKING:
            points.append(mids[BERTH_MOUTH] - normals[BERTH_MOUTH] * 2.0 * self.gate_offset)
        return points

    def _plan(self, i, position, phase, centers, radii):
        from classical_planner_pid import plan_with_ladder, resample_waypoints

        cursor = np.asarray(position, dtype=np.float64)
        pieces = []
        for target in self.route_points(phase, i, cursor):
            leg, _ = plan_with_ladder(cursor, target, centers, radii, cell_m=self.cell, ladder=self.ladder)
            self.plan_calls += 1
            if leg is None:
                leg = np.vstack((cursor, target))
                self.straight_fallbacks += 1
            pieces.append(resample_waypoints(leg, spacing_m=self.spacing))
            cursor = np.asarray(target, dtype=np.float64)
        if not pieces:
            pieces = [np.vstack((cursor, cursor))]
        waypoints = pieces[0]
        for piece in pieces[1:]:
            waypoints = np.vstack((waypoints, piece[1:] if len(piece) > 1 else piece))
        self.tracker.set_plan(i, waypoints)
        self.replan_in[i] = self.replan_every

    def _line_up(self, phase, i, position):
        """Docking stage: on the dock axis just before the mouth, or at the mouth pre-point."""
        if self.lined_up[i] or int(phase.active_index[i]) >= NO_GATE:
            self.lined_up[i] = True
            return True
        mouth = phase.gate_midpoints[i, BERTH_MOUTH].detach().cpu().double().numpy()
        normal = phase.gate_normals[i, BERTH_MOUTH].detach().cpu().double().numpy()
        rel = np.asarray(position) - mouth
        signed = float(np.dot(rel, normal))
        lateral = abs(float(-rel[0] * normal[1] + rel[1] * normal[0]))
        half = float(phase.berth_half_width[i])
        pre = mouth - normal * 2.0 * self.gate_offset
        on_axis = -3.0 * self.gate_offset <= signed <= 0.0 and lateral <= 0.5 * half
        if on_axis or float(np.linalg.norm(np.asarray(position) - pre)) <= self.accept:
            self.lined_up[i] = True
        return self.lined_up[i]

    def act(self):
        import torch

        import journal_controllers as jc

        inputs = jc.read_journal_inputs(self.base)
        phase = phase_targets_world(self.base)
        centers_t, radii_t, active_t = jc.ray_obstacles(inputs.position_world, inputs.yaw, inputs.rays,
                                                        self.max_range_m)
        centers = centers_t.detach().cpu().double().numpy()
        radii = radii_t.detach().cpu().double().numpy()
        active = active_t.detach().cpu().numpy()
        position = inputs.position_world.detach().cpu().double().numpy()
        stage = phase.stage.detach().cpu().tolist()
        tracking, docking = [], []
        for i in range(self.num_envs):
            dock_now = stage[i] == STAGE_DOCKING and self._line_up(phase, i, position[i])
            track_now = stage[i] in (STAGE_DEPARTURE, STAGE_TRANSIT) or (stage[i] == STAGE_DOCKING and not dock_now)
            tracking.append(track_now)
            docking.append(dock_now)
            if not track_now:
                continue
            key = (stage[i], int(phase.active_index[i]))
            if self.tracker.plans[i] is None or key != self.planned_key[i] or self.replan_in[i] <= 0:
                self._plan(i, position[i], phase, centers[i][active[i]], radii[i][active[i]])
                self.planned_key[i] = key
            else:
                self.replan_in[i] -= 1
        device = inputs.yaw.device
        track_mask = torch.tensor(tracking, device=device)
        dock_mask = torch.tensor(docking, device=device)
        if bool(track_mask.any()):
            for i in range(self.num_envs):
                if not tracking[i] and self.tracker.plans[i] is None:
                    self.tracker.set_plan(i, position[i:i + 1])
            transit_action = self.los.act(inputs, self.tracker.targets(inputs.position_world))
        else:
            transit_action = torch.zeros((self.num_envs, 2), device=device, dtype=inputs.yaw.dtype)
        dock_action = self.dock.act(inputs, phase.berth_point.to(inputs.yaw.dtype),
                                    phase.berth_heading.to(inputs.yaw.dtype))
        zero = torch.zeros_like(dock_action)
        action = torch.where(track_mask.unsqueeze(-1), transit_action.to(dock_action.dtype),
                             torch.where(dock_mask.unsqueeze(-1), dock_action, zero))
        return action.clamp(-1.0, 1.0)

    def reset(self, done):
        import torch

        mask = torch.as_tensor(done).reshape(-1).bool()
        self.los.reset(mask.to(self.los.pid.integral.device))
        self.dock.reset(mask.to(self.dock.phase.device))
        self.tracker.reset(mask)
        for i, flag in enumerate(mask.tolist()):
            if flag:
                self.planned_key[i] = None
                self.replan_in[i] = 0
                self.lined_up[i] = False

    def stamp(self):
        return {"controller": CONTROLLER_NAME, "stages": ["departure", "transit", "docking", "complete"],
                "replan_every": self.replan_every, "waypoint_spacing_m": self.spacing,
                "accept_radius_m": self.accept, "plan_cell_m": self.cell, "gate_offset_m": self.gate_offset,
                "inflation_ladder_m": list(self.ladder), "docking_radii_m": {
                    "align": self.dock.align_radius, "exit": self.dock.exit_radius,
                    "stop": self.dock.stop_radius, "brake": self.dock.brake_radius},
                "phase_targets": PHASE_TARGETS_STATUS, "phase_targets_schema": PHASE_TARGETS_SCHEMA,
                "static_chart_used": False}


# ------------------------------------------------------------------ selftest --
def _selftest() -> int:
    """CPU: a fake harbour env on the registry model (gates then berth), no Isaac."""
    import test_classical_harbour as t

    failures = t.run_all()
    print("SELFTEST " + ("PASS" if not failures else "FAIL: " + "; ".join(failures)))
    return 1 if failures else 0


# ----------------------------------------------------------------- evaluator --
def _build_parser():
    parser = argparse.ArgumentParser(description="USVBench staged classical harbour controller (journal ids).")
    parser.add_argument("--task", default="journal:harbour_mission")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument("--replan-every", type=int, default=60)
    parser.add_argument("--set", dest="extra_sets", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--selftest", action="store_true")
    return parser


def _run_isaac_eval() -> None:
    from isaaclab.app import AppLauncher
    from cert_v3 import CertificateRun
    from eval_protocol import add_protocol_arguments, preflight_protocol, register_journal_tasks, resolve_protocol

    original_argv = list(sys.argv)
    parser = _build_parser()
    add_protocol_arguments(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    preflight_protocol(args_cli, original_argv, parser)
    validate_assignments(args_cli.extra_sets, parser.error)
    journal = getattr(args_cli, "journal_task_info", None)
    if journal is None or journal["journal_task"] != "harbour_mission":
        parser.error("classical_harbour.py runs the journal harbour ids only (journal:harbour_mission or "
                     "Isaac-USV-HarborMission-<BlueBoat|WAMVVRX>-Journal-Direct-v1)")
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args_cli).app
    _EXIT_GUARD.attach_app(app)

    import torch
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import journal_controllers as jc

    register_journal_tasks()
    task = args_cli.task
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args_cli.num_envs)
    applied_overrides = resolve_protocol(env_cfg, args_cli)
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    vessel = str(env_cfg.vehicle)
    gains = jc.load_gains(vessel)
    controller = StagedHarbourController(base, gains=gains, max_range_m=jc.ray_max_range(env_cfg),
                                         scale=jc.length_scale(vessel), replan_every=args_cli.replan_every)
    cert = CertificateRun(
        base=base, env_cfg=env_cfg, evaluator=__file__, task=task, controller=CONTROLLER_NAME,
        controller_config={**controller.stamp(), **jc.controller_stamp(env_cfg, gains),
                           "num_envs": int(base.num_envs), "device": str(base.device),
                           "control_step_s": controller.dt},
        checkpoint=None, eval_seed=args_cli.eval_seed, level=args_cli.level,
        episodes_requested=args_cli.episodes, overrides=applied_overrides,
        argv=original_argv, cli_args=vars(args_cli),
    )
    env.reset()
    cert.after_reset()
    records = []
    ep_counter = torch.zeros(base.num_envs, dtype=torch.long)
    quota = cert.quota
    max_steps = (quota + 3) * base.max_episode_length
    step = 0
    while bool((ep_counter < quota).any()) and step < max_steps:
        with cert.time_controller(), torch.inference_mode():
            actions = controller.act()
        with cert.time_environment():
            _obs, _rew, term, trunc, _info = env.step(actions)
        cert.after_step()
        step += 1
        done = term | trunc
        done = done.squeeze(-1) if done.dim() > 1 else done
        controller.reset(done)
        for i in torch.nonzero(done).flatten().tolist():
            if int(ep_counter[i]) >= quota:
                cert.discard_episode(i)
                ep_counter[i] += 1
                continue
            tts = float(base.time_to_success[i]) if hasattr(base, "time_to_success") else float("nan")
            rec = {"env": i, "ep": int(ep_counter[i]), "controller": CONTROLLER_NAME,
                   "success": bool(base.episode_success[i]),
                   "terminated": bool(term.reshape(-1)[i]), "truncated": bool(trunc.reshape(-1)[i]),
                   "tts_s": None if math.isnan(tts) else tts,
                   "path_length_m": float(base.episode_path_length[i])}
            if hasattr(base, "episode_min_clearance"):
                rec["min_clearance_m"] = float(base.episode_min_clearance[i])
            if hasattr(base, "episode_max_phase"):
                rec["max_phase"] = int(base.episode_max_phase[i])
            records.append(cert.finish_episode(i, rec))
            ep_counter[i] += 1
    grid_complete = bool((ep_counter >= quota).all())
    successes = sum(1 for r in records if r["success"])
    print(f"EVAL task={task} controller={CONTROLLER_NAME} episodes={len(records)} SR={successes}/{len(records)} "
          f"plan_calls={controller.plan_calls} straight_fallbacks={controller.straight_fallbacks}")
    if args_cli.out:
        with _EXIT_GUARD.output(args_cli.out).open() as stream:
            json.dump(cert.build({"task": task, "num_envs": int(base.num_envs),
                                  "episodes_requested": int(args_cli.episodes), "episodes_per_env": int(quota),
                                  "complete": grid_complete, "level": args_cli.level, "seed": args_cli.eval_seed,
                                  "overrides": applied_overrides, "controller": CONTROLLER_NAME,
                                  "records": records}), stream, indent=1)
    sys.stdout.flush()
    env.close()
    # G6b: evaluator_exit.run publishes the certificate (verify + rename), THEN closes the app
    # under a watchdog; a failure never calls close() (evaluator_exit.py docstring).


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(_selftest())
    evaluator_exit.run(_run_isaac_eval, _EXIT_GUARD)
