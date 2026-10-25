"""G17: obstacle-avoidance layer for the journal classical controllers of path following,
station keeping and docking (owner decision "B10 supplement, option B", 2026-10-02).

The layer wraps the UNCHANGED base controller (journal_controllers.JournalBaselineController:
LOS/PID for path following and station keeping, the three-phase docking law for docking) and is
inactive unless the ray block shows an obstacle near the vessel. While it is inactive the action
is the base controller's own tensor (the same object), no random number is drawn and nothing in
the base controller is touched except by its own act() / reset(), which are called exactly as the
evaluator called them before (act() every control step, reset() on every done mask).

Perception (G17 rule 1): the SAME inputs the journal planner uses (journal_controllers
read_journal_inputs: the D5 own-position estimate, the root-link heading and the 36 ray ranges;
classical_planner_pid.JournalPlannerLOSPIDController). No obstacle state is read: never
obstacle_centers / radii, D7 objects or the appearing slot (journal_controllers
FORBIDDEN_ENV_MEMBERS). A return is the nearest surface point of an obstacle of assumed radius
obstacle_radius_m (the appearing radius rule 0.8 m x s), known in bearing to half a ray spacing.

Static task structure (rule 2): on docking the berth walls are part of the task. Their geometry
is rebuilt from the task's own definitions handed to the controller (the published berth pose
dock_point / dock_heading and the frozen journal plan parameters, through the same
journal_layouts_v2._berth_structure E5 builds the walls with); a ray return lying within
static_mask_tolerance_m (one planning cell) of a known wall surface is explained and removed
before the gate looks at the returns. When perception error of the own-pose estimate displaces the
wall returns (noise / bias shift them, latency / dropout with the current heading turn them), a
consensus rigid registration of the scan onto the walls (register_to_walls) runs first; it never
runs on a scan the plain mask already explains (every calm docking step). Path following and
station keeping have no static structure in the ray block (the point-return proxy sees only D7 /
appearing objects), so nothing is removed there.

Gate with hysteresis (rule 3): ACTIVATE when an unexplained return lies within trigger_gap_m of
the D7 footprint rectangle (the footprint the appearing suite measures its d_k from);
HAND BACK when no unexplained return lies within clear_gap_m AND the vessel is back near its base
reference (path following: cross-track distance to the active leg <= the gate radius; station
keeping: within the hold radius of the station; docking: lateral offset from the berth axis <=
the spawn lateral offset bound).

Behaviour while active (rule 4): plans with classical_planner_pid.plan_with_ladder (the journal
planner's planning core, its vessel ladder journal_ladder and its cell / spacing / acceptance
defaults scaled by the vessel length ratio) on discs of the unexplained returns (docking: plus the
known walls, so no plan enters the quay): path following to a rejoin point on the active leg beyond
the obstacles; station keeping and the berth zone of docking to the nearest SAFE evasion point
(hold there while an obstacle is within the clear gap), then back to the station / berth centre;
the docking approach to the berth centre around the obstacles (in the berth zone the vessel leaves the
course for the berth centre only when the centre itself is unsafe). A plan is made on activation and
kept while it stays clear at the thinnest ladder rung; it is checked every 61 control steps (the
journal planner's counter rule, replan_every = 60) and replaced when blocked, and replaced at once
for a new goal kind / leg, a moved hold target or a pass-through plan run to its end. Tracking: the
journal planner's LOS/PID law (frozen per-vessel heading PID gains, pass-through throttle
0.2 + 0.8 cos e) or, for a hold target, the base deadband hold law (journal_controllers.
LOSController), with the surge speed capped at u_ref (the speed the dose ladder and t_avoid are
defined at) while the nearest return does not close on the vessel; a return that closes (its own speed
toward the vessel, from the range rate minus the own velocity along the bearing over 30 steps, above
0.25 u_ref) is passed at the base law's speed and, at a hold target, evaded off its line of sight. The
last seen discs stay in the map for one re-plan period (a small object can fall between two rays) and
the hand-back waits as long. The layer does NOT predict trajectories (like the journal planner): every
return is a static disc in the map, rebuilt at every check. See work/DESIGN_AVOIDANCE.md.

G24b (owner decisions 2026-10-04, NC 09; 20261006_g24b_classical_fixes): (1) the station-keeping base
law holds with journal_controllers.LOSController hold mode "continuous_v2" instead of the deadband
(base name los_pid_cont, certificate los_pid_cont_avoid_v2); (2) the docking base law is
journal_controllers.RunupDockingLaw: first the berth-axis run-up point (slip opening x - registry
approach_runup_m; acceptance = the layer's waypoint acceptance radius 1.0 s), then the berth (base name
docking_runup_pid, certificate docking_runup_pid_avoid_v2); (3) the layer memory has NO time expiry
(rule constant obstacle_memory_rule "passed"): the last seen discs stay until the current rays show
water beyond their surface points on both sides (planner_memory rule S1 with the planner's constants,
in units of the layer's plan cell), at most one scan of entries (<= the ray count); the hand-back waits
until no current return and no remembered surface point lies within the clear gap. Path following keeps
its base law (certificate los_pid_avoid_v2 for the memory change). Layer version g24b_avoidance_v2.

Imports torch at module scope (like journal_controllers): import it only after AppLauncher on the
Isaac boxes; CPU tests import it directly.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import journal_controllers as jc

SCRIPTS = Path(__file__).resolve().parent
AVOID_PATH = SCRIPTS / "classical_avoidance_v2.json"
AVOID_FORMAT = "usvbench-classical-avoidance-v1"
LAYER_VERSION = "g24b_avoidance_v2"
# Certificate names (one controller per task for all its cells, calm included). G24b: every name changed
# (G17 had los_pid_avoid_v1 for path following and station keeping, docking_3phase_pid_avoid_v1 for docking).
CONTROLLER_NAMES = {"path_follow": "los_pid_avoid_v2", "station_keep": "los_pid_cont_avoid_v2",
                    "docking": "docking_runup_pid_avoid_v2"}
BASE_CONTROLLER_NAMES = {"path_follow": "los_pid", "station_keep": "los_pid_cont", "docking": "docking_runup_pid"}
TASK_OF_MODE = {"path_follow": "path_following", "station_keep": "station_keeping", "docking": "docking"}
MODES = tuple(CONTROLLER_NAMES)
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
JOURNAL_TIER = 2                      # E5 journal geometry tier of every journal cfg (journal_cfg_v2)

# --------------------------------------------------------------- rule constants --
# Every number below is frozen in scripts/classical_avoidance_v2.json (rule_constants) together
# with the per-vessel / per-task values derived from it; load_avoidance re-derives them and
# refuses to run on any difference. Reasons: work/DESIGN_AVOIDANCE.md section 3.
RULE_CONSTANTS = {
    # gate: gaps measured from the D7 footprint rectangle, in units of t_avoid * u_ref
    "trigger_ttc_ratio": 6.0,           # = D1 ratio 4 x (u_ref + 0.5 m/s D7 intruder) / u_ref
    "clear_ttc_ratio": 8.0,             # = 2 x the D1 ratio; hysteresis band 2 t_avoid u_ref
    # perception
    "obstacle_radius_length_factor": 0.8,   # appearing radius rule RADIUS_LENGTH_FACTOR (x s)
    "bearing_uncertainty_ray_spacings": 0.5,  # a return's bearing is known to +- half a ray spacing
    "static_mask_cells": 1.0,           # x journal plan grid_cell_m (0.25 s): one planning cell
    # docking: rigid registration of the scan onto the known walls before the mask, run only when the unregistered
    # mask leaves a return unexplained within the clear gap (perception error of the own-pose estimate, absolute units
    # like the perception ladder): shift <= 4.0 m (noise D4 sigma 1.0 m: 3.44 sigma radial = 99.7 %; bias D4 1.0 m),
    # turn <= 0.5 rad (latency D4 0.5 s x the design yaw-rate bound u_ref / R_turn = 0.83 rad/s BlueBoat = 0.42 rad)
    "wall_registration_max_shift_m": 4.0,
    "wall_registration_max_turn_rad": 0.5,
    "wall_registration_iterations": 8,
    "wall_registration_turn_grid": 41,
    # a registration counts only if it newly explains at least this many returns: a D7 docking intruder (diameter 0.5
    # beam) spans at most 3 rays of the 10-degree fan even at the closest D4 detection range (BlueBoat 0.465 m at
    # 1.59 m: 16.6 deg; WAM-V 1.27 m at 3.97 m: 18.1 deg), so with an exact pose (no perception error) the unexplained
    # returns are the intruder's (<= 3) and no registration can be admitted: an obstacle is never registered away
    "wall_registration_min_new_returns": 4,
    "ray_no_return_fraction": 1.0e-6,   # a range >= max * (1 - this) is "no return" (ray_obstacles)
    # planning and tracking (journal planner defaults, classical_planner_pid.py:1063-1067, x s)
    "replan_every_steps": 60,
    "plan_cell_ref_m": 0.25,
    "waypoint_spacing_ref_m": 2.0,
    "accept_radius_ref_m": 1.0,
    "rejoin_lead_spacings": 2.0,        # rejoin point at least 2 waypoint spacings ahead
    "fallback_distance_spacings": 2.0,  # unplannable map: steer 2 spacings away from the nearest return
    # speed while active: the reference speed the dose ladder and t_avoid are defined at (CONTEXT_R4 D1)
    "avoidance_speed_ratio": 1.0,       # x u_ref
    "speed_gain_per_ref_speed": 1.0,    # thrust command per u_ref of surge-speed excess (P law, clamp [-1, 1])
    # station keeping / docking evasion targets
    "evade_padding_fraction": 0.5,      # evasion disk radius = 0.5 x the task's region padding
    "evade_bearings": 16,
    "evade_ring_fractions": [0.25, 0.5, 0.75, 1.0],
    # an evasion point is SAFE when its clearance from every obstacle disc is at least the fattest ladder
    # rung + the footprint reach + the distance covered at u_ref in one re-plan period (no motion model)
    "evade_margin_replan_periods": 1.0,
    # closing test (no trajectory prediction): the nearest return CLOSES when its own speed toward the vessel,
    # -(range rate + own velocity . bearing) over the last closing_window_steps, exceeds closing_speed_ratio x u_ref
    # (half the D7 default intruder speed 0.5 m/s at u_ref 1 m/s). The u_ref speed cap is for obstacles that do not
    # close (static ones: slow down, more time to go round); a closing (moving) one is passed at the base law's speed
    # (slowing to u_ref would re-create the encounter the D7 dose built on the nominal u_ref plan) and, at a hold
    # target (station keeping, docking berth zone), evaded off its line of sight
    "closing_window_steps": 30,
    "closing_speed_ratio": 0.25,
    # memory: a small object can fall between two rays of the 10-degree fan (a D7 intruder of 0.5 beam subtends less
    # than one spacing beyond 2.7 m BlueBoat / 7.3 m WAM-V); the last seen obstacle discs stay in the map, and the
    # hand-back waits, while the memory holds them. G24b (owner decision: no timeout expiry): rule "passed" = the
    # discs stay until the current rays see through their surface points on both sides (planner_memory S1 with the
    # planner's constants: lateral 0.5, margin 1.0 plan cells); G17's rule "steps" (61 control steps since the last
    # return inside the clear gap) is kept only as the comparison arm of the G24b evidence runs
    "obstacle_memory_rule": "passed",
    "obstacle_memory_steps": 61,
    "memory_carve_lateral_cells": 0.5,
    "memory_carve_margin_cells": 1.0,
    # G24b base laws: station keeping holds with LOSController "continuous_v2" (owner decision: continuous, not
    # deadband); docking approaches via the berth-axis run-up point (RunupDockingLaw, acceptance accept_radius_m)
    "station_hold_mode": "continuous_v2",
    "berth_axis_points": 5,             # docking: evasion candidates inside the slip on its axis
}


# ------------------------------------------------------------------ derivation --
def _lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def journal_plan(task, vessel, tier=JOURNAL_TIER):
    plans = jc.shared_module("journal_cfg_v2")
    return plans.plan_class(task, vessel)(geometry_tier=int(tier))


def d7_envelope(vessel):
    """D7 VesselEnvelope exactly as the journal families build it (journal_path_following.d7_envelope)."""
    plans = jc.shared_module("journal_cfg_v2")
    registry = jc.shared_module("vehicles_v2")
    dynamic = jc.shared_module("conditions.dynamic")
    record = plans.vessel_record(vessel)
    envelope, _ = dynamic.envelope_from_spec_v2(registry.get_vehicle_v2(vessel),
                                                reference_speed_mps=record.cruise_speed_m_s,
                                                turning_radius_m=record.turning_radius_design_m)
    return envelope


def derive_vessel_constants(vessel, rules=None):
    """Per-vessel layer constants from the registry, the D7 envelope and the rule constants."""
    from classical_planner_pid import journal_ladder

    r = dict(RULE_CONSTANTS if rules is None else rules)
    envelope = d7_envelope(vessel)
    s = jc.length_scale(vessel)
    t_avoid = float(envelope.t_avoid_s)
    u_ref = float(envelope.reference_speed_mps)
    fp = envelope.footprint
    spacing = r["waypoint_spacing_ref_m"] * s
    params = journal_plan("path_following", vessel).params
    grid_cell = float(params["grid_cell_m"])
    ladder = [float(x) for x in journal_ladder(vessel)]
    control_dt = float(jc.shared_module("journal_cfg_v2").CONTROL_DT_S)
    return {
        "vessel": vessel, "length_scale": s, "t_avoid_s": t_avoid, "reference_speed_mps": u_ref,
        "footprint_m": [float(fp.x_min_m), float(fp.x_max_m), float(fp.y_min_m), float(fp.y_max_m)],
        "footprint_reach_m": float(fp.reach_m),
        "trigger_gap_m": r["trigger_ttc_ratio"] * t_avoid * u_ref,
        "clear_gap_m": r["clear_ttc_ratio"] * t_avoid * u_ref,
        "obstacle_radius_m": r["obstacle_radius_length_factor"] * s,
        "static_mask_tolerance_m": r["static_mask_cells"] * grid_cell,
        "replan_every_steps": int(r["replan_every_steps"]),
        "plan_cell_m": r["plan_cell_ref_m"] * s,
        "waypoint_spacing_m": spacing,
        "accept_radius_m": r["accept_radius_ref_m"] * s,
        "rejoin_lead_m": r["rejoin_lead_spacings"] * spacing,
        "fallback_distance_m": r["fallback_distance_spacings"] * spacing,
        "inflation_ladder_m": ladder,
        # derived from the values above (recorded so that every number the layer uses is in the file)
        "control_dt_s": control_dt,
        "keep_inflation_m": min(ladder),
        "bearing_pad_tan": math.tan(r["bearing_uncertainty_ray_spacings"] * 2.0 * math.pi
                                    / int(params["ray_sensor_ray_count"])),
        "avoidance_speed_mps": r["avoidance_speed_ratio"] * u_ref,
        "closing_speed_mps": r["closing_speed_ratio"] * u_ref,
        "evade_clearance_m": (ladder[0] + float(fp.reach_m)
                              + r["evade_margin_replan_periods"] * int(r["replan_every_steps"]) * control_dt * u_ref),
    }


def derive_task_constants(task, vessel, rules=None, tier=JOURNAL_TIER):
    """Per-task constants from the task's own frozen plan (tier ``tier``)."""
    r = dict(RULE_CONSTANTS if rules is None else rules)
    p = journal_plan(task, vessel, tier).params
    out = {"task": task, "tier": int(tier)}
    if task == "path_following":
        out["rejoin_tolerance_m"] = float(p["gate_radius_m"])
    elif task == "station_keeping":
        out["return_tolerance_m"] = float(p["hold_radius_m"])
        out["evade_radius_m"] = r["evade_padding_fraction"] * float(p["region_padding_m"])
    elif task == "docking":
        out["approach_line_tolerance_m"] = float(p["lateral_offset_max_m"])
        out["evade_radius_m"] = r["evade_padding_fraction"] * float(p["region_padding_m"])
        for name in ("slip_opening_x_m", "slip_back_face_x_m", "slip_free_width_m", "wall_radius_m",
                     "position_tolerance_m"):
            out[name] = float(p[name])
        # G24b run-up point of RunupDockingLaw: the E5 layout's via (journal_layouts_v2._cand_docking)
        runup = float(jc.shared_module("journal_cfg_v2").vessel_record(vessel).geometry.approach_runup_m)
        out["approach_runup_m"] = runup
        out["runup_x_m"] = float(p["slip_opening_x_m"]) - runup
    else:
        raise ValueError("the avoidance layer covers path_following, station_keeping and docking only")
    return out


def derive_document(rules=None):
    """The whole frozen document (classical_avoidance_v2.json) as the code derives it now."""
    r = dict(RULE_CONSTANTS if rules is None else rules)
    vessels = {}
    for vessel in VESSELS:
        block = derive_vessel_constants(vessel, r)
        block["tasks"] = {task: derive_task_constants(task, vessel, r)
                          for task in ("path_following", "station_keeping", "docking")}
        vessels[vessel] = block
    return {"format": AVOID_FORMAT, "layer_version": LAYER_VERSION, "controller_names": dict(CONTROLLER_NAMES),
            "base_controller_names": dict(BASE_CONTROLLER_NAMES), "rule_constants": r,
            "derivation": "scripts/journal_avoidance.py derive_document (G17); reasons in "
                          "work/DESIGN_AVOIDANCE.md section 3",
            "vessels": vessels}


def document_text(document):
    return json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n"


def load_avoidance(vessel, task, path=None):
    """Frozen constants of (vessel, task) + provenance; fail closed on any drift.

    The file must equal what derive_document yields from the rule constants it records (the
    registry, the D7 envelope or a task plan changed -> refuse), and its rule constants must
    equal this module's RULE_CONSTANTS.
    """
    path = Path(path or AVOID_PATH)
    if not path.is_file():
        raise RuntimeError("avoidance constants file missing: %s (python -B scripts/journal_avoidance.py --freeze)"
                           % path)
    data = json.loads(path.read_text(encoding="ascii"))
    if data.get("format") != AVOID_FORMAT or data.get("layer_version") != LAYER_VERSION:
        raise RuntimeError("%s: not a %s / %s file" % (path, AVOID_FORMAT, LAYER_VERSION))
    if data.get("rule_constants") != RULE_CONSTANTS:
        raise RuntimeError("%s: rule constants differ from journal_avoidance.RULE_CONSTANTS" % path)
    fresh = derive_document(data["rule_constants"])
    if json.loads(document_text(fresh)) != data:
        raise RuntimeError("%s: frozen avoidance constants differ from the current derivation (registry, D7 "
                           "envelope or task plan changed); re-freeze deliberately" % path)
    block = data["vessels"].get(vessel)
    if block is None or task not in block["tasks"]:
        raise RuntimeError("no frozen avoidance constants for %s / %s" % (vessel, task))
    constants = {k: v for k, v in block.items() if k != "tasks"}
    return {"vessel": vessel, "task": task, "vessel_constants": constants, "task_constants": dict(block["tasks"][task]),
            "rule_constants": dict(data["rule_constants"]), "file": str(path), "file_sha256": _lf_sha256(path)}


# ------------------------------------------------------------------ geometry --
def berth_wall_template(plan_params):
    """Berth-frame wall cylinders (centres [W, 2], radii [W]) from the frozen plan parameters.

    The SAME builder E5 uses for the docking layout (journal_layouts_v2._berth_structure with the
    berth centre at the origin and heading 0); the env rotates this template by the berth heading
    about the env origin (journal_docking.build_scenario), which act() repeats from dock_heading.
    """
    layouts = jc.shared_module("journal_layouts_v2")
    centers, radii, _jambs, _berth = layouts._berth_structure(dict(plan_params), (0.0, 0.0), 0.0)
    return np.asarray(centers, dtype=np.float64), np.asarray(radii, dtype=np.float64)


def footprint_gap(points_body, footprint):
    """Distance (m) from body-frame points [..., 2] to the footprint rectangle (0 inside)."""
    x_min, x_max, y_min, y_max = footprint
    dx = torch.clamp(torch.maximum(x_min - points_body[..., 0], points_body[..., 0] - x_max), min=0.0)
    dy = torch.clamp(torch.maximum(y_min - points_body[..., 1], points_body[..., 1] - y_max), min=0.0)
    return torch.sqrt(dx * dx + dy * dy)


def ray_points(position_local, yaw, rays, max_range_m, no_return_fraction):
    """(points [N, R, 2] env-local, unit directions [N, R, 2], valid [N, R]) of the ray returns.

    Ray i leaves the COM at yaw + i * 2 pi / R (journal_controllers.ray_obstacles convention); a
    range below max * (1 - no_return_fraction) is a return at range * direction.
    """
    count = rays.shape[-1]
    angles = yaw.unsqueeze(-1) + torch.arange(count, device=rays.device, dtype=rays.dtype) * (2.0 * math.pi / count)
    direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    ranges = rays.clamp(0.0, float(max_range_m))
    points = position_local.unsqueeze(1) + ranges.unsqueeze(-1) * direction
    valid = ranges < float(max_range_m) * (1.0 - float(no_return_fraction))
    return points, direction, valid


def explained_by_walls(points, wall_centers, wall_radii, tolerance_m):
    """[N, R] True where a point lies within tolerance of a wall cylinder surface."""
    if wall_centers.shape[-2] == 0:
        return torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device)
    delta = points.unsqueeze(-2) - wall_centers.unsqueeze(-3)            # [N, R, W, 2]
    surface = torch.abs(torch.linalg.vector_norm(delta, dim=-1) - wall_radii.unsqueeze(-2))
    return (surface <= float(tolerance_m)).any(-1)


def _register(position, relative, valid, walls, radii, *, iterations, max_shift_m, max_turn_rad, gate_m, huber_m,
              dof, min_points, turn0=None):
    """Rigid 2-D registration (shift, turn about the estimated position) of a scan onto the known wall cylinders.

    position [N, 2], relative [N, R, 2] (range x direction around the estimate), valid [N, R], walls [N, W, 2],
    radii [N, W]. Gauss-Newton on the point-to-surface residuals of the returns within gate_m of a wall (nearest
    cylinder re-associated each iteration, Huber weights at huber_m), only the degrees of freedom in ``dof`` (dx, dy,
    turn); an env with fewer than min_points associated returns does not move, a solution beyond max_shift_m /
    max_turn_rad is reset to none. Returns (registered points, |residual| to the nearest wall surface).
    """
    n = position.shape[0]
    dtype, device = position.dtype, position.device
    shift = torch.zeros((n, 2), dtype=dtype, device=device)
    turn = torch.zeros(n, dtype=dtype, device=device) if turn0 is None else turn0.clone()

    def transform(shift, turn):
        c, s = torch.cos(turn).unsqueeze(-1), torch.sin(turn).unsqueeze(-1)
        rotated = torch.stack((c * relative[..., 0] - s * relative[..., 1], s * relative[..., 0] + c * relative[..., 1]), -1)
        return position.unsqueeze(1) + shift.unsqueeze(1) + rotated, rotated

    def nearest(points):
        delta = points.unsqueeze(-2) - walls.unsqueeze(-3)
        dist = torch.linalg.vector_norm(delta, dim=-1)
        signed = dist - radii.unsqueeze(-2)
        index = signed.abs().argmin(-1, keepdim=True)
        d = delta.gather(-2, index.unsqueeze(-1).expand(*index.shape, 2)).squeeze(-2)
        return signed.gather(-1, index).squeeze(-1), d / dist.gather(-1, index).clamp_min(1e-12)

    mask = torch.tensor(dof, dtype=dtype, device=device)
    for _ in range(int(iterations)):
        points, rotated = transform(shift, turn)
        f, normal = nearest(points)
        use = valid & (f.abs() <= gate_m)
        w = torch.where(use, torch.clamp(huber_m / f.abs().clamp_min(1e-12), max=1.0), torch.zeros_like(f))
        perp = torch.stack((-rotated[..., 1], rotated[..., 0]), -1)
        jac = torch.stack((normal[..., 0], normal[..., 1], (normal * perp).sum(-1)), -1)
        jtw = jac * w.unsqueeze(-1)
        hess = jtw.transpose(-1, -2) @ jac + 1e-9 * torch.eye(3, dtype=dtype, device=device)
        hess = hess * mask.unsqueeze(0) * mask.unsqueeze(1) + torch.diag(1.0 - mask)
        step = torch.linalg.solve(hess, -((jtw * f.unsqueeze(-1)).sum(-2) * mask).unsqueeze(-1)).squeeze(-1) * mask
        step = torch.where((use.sum(-1) >= min_points).unsqueeze(-1), step, torch.zeros_like(step))
        shift = shift + step[:, :2]
        turn = turn + step[:, 2]
    ok = (torch.linalg.vector_norm(shift, dim=-1) <= max_shift_m) & (turn.abs() <= max_turn_rad)
    shift = torch.where(ok.unsqueeze(-1), shift, torch.zeros_like(shift))
    turn = torch.where(ok, turn, torch.zeros_like(turn))
    points, _ = transform(shift, turn)
    f, _ = nearest(points)
    return points, f.abs()


def register_to_walls(position, relative, valid, walls, radii, *, tolerance_m, max_shift_m, max_turn_rad, iterations,
                      turn_grid, min_new):
    """Consensus registration of a docking scan onto the known walls (the static mask under own-pose error).

    Candidates, in this order: no registration; the best turn of a grid over [-max_turn, max_turn] (no shift);
    turn-only, shift-only and full Gauss-Newton registrations (the turn models start from the grid's best turn). Per env
    the candidate that explains the most returns (|residual| <= tolerance_m) wins, ties keep the earlier one, so a scan
    the unregistered mask already explains completely is never moved. A candidate is admitted only if it newly explains
    at least min_new returns: with no pose error only the obstacle's returns are unexplained and an obstacle spans fewer
    rays than min_new, so no candidate can be admitted and an obstacle is never registered away; a pose error moves
    every wall return in view. Returns |residual| [N, R] of the winner.
    """
    if walls.shape[-2] == 0:
        return torch.full(valid.shape, math.inf, dtype=position.dtype, device=position.device)
    n, dtype, device = position.shape[0], position.dtype, position.device
    common = dict(max_shift_m=max_shift_m, max_turn_rad=max_turn_rad, gate_m=max_shift_m + tolerance_m,
                  huber_m=tolerance_m)
    _, best = _register(position, relative, valid, walls, radii, iterations=0, dof=(0.0, 0.0, 0.0), min_points=1,
                        **common)
    plain = valid & (best <= tolerance_m)
    best_count = plain.sum(-1)
    grid_turn, grid_count = torch.zeros(n, dtype=dtype, device=device), best_count.clone()

    def take(res):
        nonlocal best, best_count
        explained = valid & (res <= tolerance_m)
        count = explained.sum(-1)
        admitted = (explained & ~plain).sum(-1) >= int(min_new)
        better = admitted & (count > best_count)
        best = torch.where(better.unsqueeze(-1), res, best)
        best_count = torch.where(better, count, best_count)
        return count

    # The turn grid in one batch [G, N, R] (the same arithmetic as G turn-only evaluations in grid order): per env the
    # first admitted grid turn with the most explained returns competes like the other candidates, and the first grid
    # turn with the most explained returns (admitted or not) starts the turn models.
    turns = torch.linspace(-max_turn_rad, max_turn_rad, int(turn_grid), dtype=dtype).to(device)
    c, s = torch.cos(turns).view(-1, 1, 1), torch.sin(turns).view(-1, 1, 1)
    rotated = torch.stack((c * relative[..., 0] - s * relative[..., 1], s * relative[..., 0] + c * relative[..., 1]), -1)
    points = position.view(1, n, 1, 2) + rotated
    signed = torch.linalg.vector_norm(points.unsqueeze(-2) - walls.unsqueeze(0).unsqueeze(-3), dim=-1) \
        - radii.view(1, n, 1, -1)
    residual = signed.abs().min(-1).values                                          # [G, N, R]
    explained = valid.unsqueeze(0) & (residual <= tolerance_m)
    counts = explained.sum(-1)                                                      # [G, N]
    admitted = (explained & ~plain.unsqueeze(0)).sum(-1) >= int(min_new)
    scored = torch.where(admitted, counts, torch.full_like(counts, -1))
    first = scored.argmax(0)                                                         # first maximum (grid order)
    index = torch.arange(n, device=device)
    better = (scored[first, index] > best_count) & admitted[first, index]
    best = torch.where(better.unsqueeze(-1), residual[first, index], best)
    best_count = torch.where(better, counts[first, index], best_count)
    top = counts.argmax(0)
    move = counts[top, index] > grid_count
    grid_turn = torch.where(move, turns[top], grid_turn)
    take(_register(position, relative, valid, walls, radii, iterations=iterations, dof=(0.0, 0.0, 1.0), min_points=2,
                   turn0=grid_turn, **common)[1])
    take(_register(position, relative, valid, walls, radii, iterations=iterations, dof=(1.0, 1.0, 0.0), min_points=3,
                   **common)[1])
    take(_register(position, relative, valid, walls, radii, iterations=iterations, dof=(1.0, 1.0, 1.0), min_points=6,
                   turn0=grid_turn, **common)[1])
    return best


def to_body(points, position, yaw):
    delta = points - position.unsqueeze(1)
    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    return torch.stack((c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]), dim=-1)


# -------------------------------------------------------------- planning helpers --
def _project(point, a, b):
    seg = b - a
    length = float(np.linalg.norm(seg))
    if length < 1.0e-9:
        return 0.0, 0.0, float(np.linalg.norm(point - a)), np.array((1.0, 0.0)), length
    t = seg / length
    rel = point - a
    along = float(rel @ t)
    lateral = float(rel[0] * t[1] - rel[1] * t[0])
    clamped = min(max(along, 0.0), length)
    distance = float(np.linalg.norm(point - (a + clamped * t)))
    return along, lateral, distance, t, length


def _clearance(point, centers, radii):
    """Surface clearance of a point from obstacle discs (inf with no disc)."""
    if len(centers) == 0:
        return math.inf
    return float(np.min(np.linalg.norm(centers - point, axis=1) - radii))


class AvoidanceLayerController:
    """los_pid_avoid_v2 / los_pid_cont_avoid_v2 / docking_runup_pid_avoid_v2: the base journal law plus the layer.

    ``base_controller`` is the unchanged journal_controllers.JournalBaselineController of the
    same mode. ``act(obs)`` calls ``base_controller.act(obs)`` first, every step; ``reset(done)``
    calls ``base_controller.reset(done)`` first, every time.
    """

    def __init__(self, base, base_controller, mode, avoidance, gains, *, dt, max_range_m, plan_params,
                 berth_walls=True):
        if mode not in MODES:
            raise ValueError("mode must be one of %s" % (MODES,))
        if base_controller.mode != mode:
            raise ValueError("base controller mode %r differs from %r" % (base_controller.mode, mode))
        self.base, self.base_controller, self.mode = base, base_controller, mode
        self.avoidance = avoidance
        vc, tc = avoidance["vessel_constants"], avoidance["task_constants"]
        self.vc, self.tc, self.rules = vc, tc, avoidance["rule_constants"]
        self.num_envs = int(base.num_envs)
        self.device = base.device
        self.dt = float(dt)
        self.max_range_m = float(max_range_m)
        self.footprint = tuple(float(x) for x in vc["footprint_m"])
        self.trigger = float(vc["trigger_gap_m"])
        self.clear = float(vc["clear_gap_m"])
        self.r_obs = float(vc["obstacle_radius_m"])
        self.reach = float(vc["footprint_reach_m"])
        self.tol = float(vc["static_mask_tolerance_m"])
        self.cell = float(vc["plan_cell_m"])
        self.spacing = float(vc["waypoint_spacing_m"])
        self.accept = float(vc["accept_radius_m"])
        self.lead = float(vc["rejoin_lead_m"])
        self.fallback_distance = float(vc["fallback_distance_m"])
        self.ladder = tuple(float(x) for x in vc["inflation_ladder_m"])
        self.replan_every = int(vc["replan_every_steps"])
        self.no_return = float(self.rules["ray_no_return_fraction"])
        if abs(self.dt - float(vc["control_dt_s"])) > 1.0e-12:
            raise ValueError("control step %r differs from the frozen %r" % (self.dt, vc["control_dt_s"]))
        self.bearing_pad = float(vc["bearing_pad_tan"])
        self.speed_cap = float(vc["avoidance_speed_mps"])
        self.speed_gain = float(self.rules["speed_gain_per_ref_speed"]) / float(vc["reference_speed_mps"])
        self.keep_inflation = float(vc["keep_inflation_m"])
        self.reg_shift = float(self.rules["wall_registration_max_shift_m"])
        self.reg_turn = float(self.rules["wall_registration_max_turn_rad"])
        self.reg_iterations = int(self.rules["wall_registration_iterations"])
        self.reg_grid = int(self.rules["wall_registration_turn_grid"])
        self.reg_min_new = int(self.rules["wall_registration_min_new_returns"])
        self.evade_clearance = float(vc["evade_clearance_m"])
        self.closing_window = int(self.rules["closing_window_steps"])
        self.memory_steps = int(self.rules["obstacle_memory_steps"])
        # G24b: memory rule "passed" (no expiry; S1 see-through on both sides) or G17's "steps" (comparison arm)
        self.memory_rule = self.rules.get("obstacle_memory_rule", "steps")
        if self.memory_rule not in ("steps", "passed"):
            raise ValueError("obstacle_memory_rule must be 'steps' or 'passed'")
        self.carver = None
        if self.memory_rule == "passed":
            import planner_memory as pm

            carve_rules = dict(pm.RULE_CONSTANTS, carve_lateral_cells=float(self.rules["memory_carve_lateral_cells"]),
                               carve_margin_cells=float(self.rules["memory_carve_margin_cells"]))
            self.carver = pm.RayReturnMemory(1, plan_cell_m=self.cell, replan_every=self.replan_every,
                                             length_scale=float(vc["length_scale"]), ray_max_range_m=self.max_range_m,
                                             reference_speed_mps=float(vc["reference_speed_mps"]), control_dt_s=self.dt,
                                             rules=carve_rules)
        self.closing_speed = float(vc["closing_speed_mps"])
        if self.trigger <= 0.0 or self.clear <= self.trigger:
            raise ValueError("clear gap must exceed the trigger gap (hysteresis)")
        # Base hold-law radii reused for hold targets (station keeping: the base zone / slow radii;
        # docking: the DockingLaw stop / brake radii).
        if mode == "docking":
            dock = base_controller.dock
            self.hold_zone = (float(dock.stop_radius), float(dock.brake_radius))
        else:
            self.hold_zone = (float(base_controller.zone), float(base_controller.slow))
        self.walls_enabled = mode == "docking" and bool(berth_walls)
        if mode == "docking":
            centers, radii = berth_wall_template(plan_params)
            self.wall_template = torch.as_tensor(centers, dtype=torch.float64, device=self.device)
            self.wall_template_radii = torch.as_tensor(radii, dtype=torch.float64, device=self.device)
            self.wall_template_np = (centers, radii)
            if not self.walls_enabled:
                self.wall_template = self.wall_template[:0]
                self.wall_template_radii = self.wall_template_radii[:0]
        los = gains["los_heading_pid"]
        self.pid = jc.HeadingPID(self.num_envs, self.device, torch.float64, self.dt, los["kp"], los["ki"], los["kd"])
        n = self.num_envs
        self.active = [False] * n
        self.active_t = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.plans = [None] * n
        self.plan_index = [0] * n
        self.plan_hold = [False] * n
        self.plan_goal = [None] * n
        self.plan_key = [None] * n
        self.replan_in = [0] * n
        self.evade_target = [None] * n
        self.fallback = [False] * n
        self.history = [[] for _ in range(n)]       # (nearest range, bearing unit, own velocity world) per active step
        self.step = 0
        self.last_seen = [-(10 ** 9)] * n            # control step of the last return inside the clear gap
        self.memory = [None] * n                     # (centers, radii, surface points, S1 side) of that step
        self.memory_in_use = [False] * n             # G24b "passed": a remembered point lies within the clear gap
        self.closing = [False] * n
        self.evade_side = [0] * n
        # telemetry (G17: into header.controller.configuration.avoidance.telemetry at the end)
        self.episode_index = [0] * n
        self.episode_active_steps = [0] * n
        self.episode_activations = [0] * n
        self.counters = {"activations": 0, "handbacks": 0, "active_steps": 0, "plans": 0, "plan_failures": 0,
                         "fallback_steps": 0, "max_simultaneous_active": 0, "episodes_finished": 0,
                         "episodes_with_activation": 0, "wall_registrations": 0, "closing_steps": 0,
                         "memory_steps": 0, "memory_dropped": 0}
        self.episode_rows = []

    # ------------------------------------------------------------- inputs --
    def _walls_local(self, n):
        """[N, W, 2] env-local wall centres from the published berth pose (docking only)."""
        heading = self.base.dock_heading.to(self.device, torch.float64)
        heading = heading / torch.linalg.vector_norm(heading, dim=-1, keepdim=True).clamp_min(1.0e-12)
        center = (self.base.dock_point.to(self.device, torch.float64)
                  - self.base.scene.env_origins[:, :2].to(self.device, torch.float64))
        c, s = heading[:, 0:1], heading[:, 1:2]
        tx, ty = self.wall_template[:, 0].unsqueeze(0), self.wall_template[:, 1].unsqueeze(0)
        walls = torch.stack((c * tx - s * ty, s * tx + c * ty), dim=-1) + center.unsqueeze(1)
        return walls, self.wall_template_radii.unsqueeze(0).expand(n, -1), center, heading

    def observe(self, inputs):
        """Gate quantities of one control step (docking: at most one extra host flag, for the wall registration)."""
        pos = inputs.position_local.to(torch.float64)
        yaw = inputs.yaw.to(torch.float64)
        rays = inputs.rays.to(torch.float64)
        points, direction, valid = ray_points(pos, yaw, rays, self.max_range_m, self.no_return)
        n = pos.shape[0]
        out = {"pos": pos, "yaw": yaw, "points": points, "direction": direction, "valid": valid,
               "ranges": rays.clamp(0.0, self.max_range_m), "surge": inputs.velocity[:, 0].to(torch.float64),
               "velocity": inputs.velocity[:, :2].to(torch.float64)}
        gap = footprint_gap(to_body(points, pos, yaw), self.footprint)
        if self.mode == "docking":
            walls, radii, center, heading = self._walls_local(n)
            explained = explained_by_walls(points, walls, radii, self.tol)
            out.update(walls=walls, wall_radii=radii, dock_center=center, dock_heading=heading)
            # Own-pose error (perception) displaces every wall return together: when the plain mask leaves a return
            # unexplained within the clear gap, and at least wall_registration_min_new_returns unexplained in all (no
            # registration could be admitted otherwise: same result, no cost), register the scan onto the walls and
            # mask with the registered residuals (one extra host flag; never in a scan the plain mask explains, e.g.
            # every calm docking step, nor with only an obstacle's few returns unexplained).
            loose = valid & ~explained
            if self.walls_enabled and bool(((loose & (gap <= self.clear)).any(-1)
                                            & (loose.sum(-1) >= self.reg_min_new)).any()):
                residual = register_to_walls(pos, points - pos.unsqueeze(1), valid, walls, radii, tolerance_m=self.tol,
                                             max_shift_m=self.reg_shift, max_turn_rad=self.reg_turn,
                                             iterations=self.reg_iterations, turn_grid=self.reg_grid,
                                             min_new=self.reg_min_new)
                explained = explained | (residual <= self.tol)
                self.counters["wall_registrations"] += 1
        else:
            explained = torch.zeros_like(valid)
        unexplained = valid & ~explained
        out.update(explained=explained, unexplained=unexplained, gap=gap,
                   near_trigger=(unexplained & (gap <= self.trigger)).any(-1),
                   within_clear=(unexplained & (gap <= self.clear)).any(-1))
        return out

    # ----------------------------------------------------------- reference --
    def _near_reference(self, i, pos, host):
        if self.mode == "path_follow":
            waypoints = host["waypoints"][i]
            k = min(max(int(host["gates"][i]), 0), waypoints.shape[0] - 1)
            if int(host["gates"][i]) >= waypoints.shape[0]:
                return True
            a = np.zeros(2) if k == 0 else waypoints[k - 1]
            _, _, distance, _, _ = _project(pos, a, waypoints[k])
            return distance <= float(self.tc["rejoin_tolerance_m"])
        if self.mode == "station_keep":
            return float(np.linalg.norm(pos - host["station"][i])) <= float(self.tc["return_tolerance_m"])
        xb, yb = self._berth_frame(i, pos, host)
        return abs(yb) <= float(self.tc["approach_line_tolerance_m"])

    def _berth_frame(self, i, point, host):
        h = host["dock_heading"][i]
        rel = point - host["dock_center"][i]
        return float(rel @ h), float(h[0] * rel[1] - h[1] * rel[0])

    def _from_berth(self, i, xb, yb, host):
        h = host["dock_heading"][i]
        return host["dock_center"][i] + xb * h + yb * np.array((-h[1], h[0]))

    # ------------------------------------------------------------- goals --
    def _goal(self, i, pos, centers, radii, within_clear, host):
        """(goal [2], hold flag, key) of one active env; key changes force a re-plan."""
        if self.mode == "path_follow":
            waypoints = host["waypoints"][i]
            gates = int(host["gates"][i])
            k = min(max(gates, 0), waypoints.shape[0] - 1)
            a = np.zeros(2) if k == 0 else waypoints[k - 1]
            b = waypoints[k]
            along, _, _, t, length = _project(pos, a, b)
            need = min(max(along, 0.0), length) + self.lead
            corridor = self.r_obs + self.ladder[0] + self.reach
            for c in centers:
                c_along, c_lateral, _, _, _ = _project(c, a, b)
                if abs(c_lateral) <= corridor and c_along >= -corridor:
                    need = max(need, c_along + corridor)
            s_rej = min(need, length)
            return a + s_rej * t, False, ("leg", k)
        if self.mode == "station_keep":
            station = host["station"][i]
            if within_clear and len(centers):
                return self._evade_target(i, self._sk_candidates(station), station, centers, radii, pos, host), True, "evade"
            self.evade_target[i] = None
            return station, True, "return"
        # docking
        center = host["dock_center"][i]
        xb, _ = self._berth_frame(i, pos, host)
        if xb >= float(self.tc["slip_opening_x_m"]):
            # berth zone: hold course for the berth centre unless the centre itself is unsafe (an obstacle outside
            # the slip that is not at the berth is no reason to leave the slip; turning in it risks the walls)
            if within_clear and len(centers) and _clearance(center, centers, radii) < self.evade_clearance:
                return (self._evade_target(i, self._berth_candidates(i, host), center, centers, radii, pos, host), True,
                        "evade")
            self.evade_target[i] = None
            return center, True, "return"
        self.evade_target[i] = None
        return center, True, "approach"

    def _update_closing(self, i, host):
        """Closing test of env i (rule constants closing_*): appends this step's nearest unexplained return within the
        clear gap and decides whether that return itself moves toward the vessel."""
        keep = host["clear_mask"][i]
        if not keep.any():
            self.history[i] = []
            self.closing[i] = False
            return
        ranges = np.where(keep, host["ranges"][i], np.inf)
        j = int(np.argmin(ranges))
        yaw = float(host["yaw"][i])
        c, s = math.cos(yaw), math.sin(yaw)
        vel = host["velocity"][i]
        own = np.array((c * vel[0] - s * vel[1], s * vel[0] + c * vel[1]))
        self.history[i].append((float(ranges[j]), host["direction"][i][j].copy(), own))
        if len(self.history[i]) > self.closing_window + 1:
            self.history[i].pop(0)
        if len(self.history[i]) <= self.closing_window:
            self.closing[i] = False
            return
        rate = (self.history[i][-1][0] - self.history[i][0][0]) / (self.closing_window * self.dt)
        own_along = float(np.mean([float(o @ u) for _, u, o in self.history[i][1:]]))
        self.closing[i] = -(rate + own_along) > self.closing_speed

    def _corridor_ok(self, i, point, pos, host):
        """Off the closing return's line of sight: the obstacle is assumed to run along its bearing toward and past
        the vessel (a collision course keeps a constant bearing; the D7 station-keeping encounters aim at the
        station), so a point must lie at least the disc radius + evade_clearance off that line, on the committed
        side."""
        keep = host["clear_mask"][i]
        ranges = np.where(keep, host["ranges"][i], np.inf)
        j = int(np.argmin(ranges))
        u = host["direction"][i][j]
        center = host["points"][i][j] + self.r_obs * u
        radius = self.r_obs + (float(ranges[j]) + self.r_obs) * self.bearing_pad
        rel = point - center
        lateral = float(u[0] * rel[1] - u[1] * rel[0])
        side = 1 if lateral > 0 else -1
        if self.evade_side[i] and side != self.evade_side[i]:
            return False
        return abs(lateral) >= radius + self.evade_clearance

    def _sk_candidates(self, station):
        radius = float(self.tc["evade_radius_m"])
        bearings = int(self.rules["evade_bearings"])
        out = [station]
        for fraction in self.rules["evade_ring_fractions"]:
            for k in range(bearings):
                angle = 2.0 * math.pi * k / bearings
                out.append(station + fraction * radius * np.array((math.cos(angle), math.sin(angle))))
        return out

    def _berth_candidates(self, i, host):
        """Evasion targets bounded by the berth geometry: on the slip axis inside the slip and in the
        approach area outside the mouth, each clear of the walls by the thinnest ladder rung."""
        x_open, x_back = float(self.tc["slip_opening_x_m"]), float(self.tc["slip_back_face_x_m"])
        radius = float(self.tc["evade_radius_m"])
        points = []
        inner = np.linspace(x_open + self.reach, x_back - self.reach, int(self.rules["berth_axis_points"]))
        for xb in inner:
            points.append(self._from_berth(i, float(xb), 0.0, host))
        bearings = int(self.rules["evade_bearings"])
        mouth = self._from_berth(i, x_open, 0.0, host)
        h = host["dock_heading"][i]
        for fraction in self.rules["evade_ring_fractions"]:
            for k in range(bearings):
                angle = 2.0 * math.pi * k / bearings
                p = mouth + fraction * radius * np.array((math.cos(angle), math.sin(angle)))
                if float((p - host["dock_center"][i]) @ h) < x_open:      # outside the mouth line
                    points.append(p)
        if self.walls_enabled:
            walls, wall_radii = host["walls"][i], host["wall_radii"][i]
            keep = [p for p in points if _clearance(p, walls, wall_radii) > self.ladder[-1]]
            points = keep or points
        return points

    def _evade_target(self, i, candidates, home, centers, radii, pos, host):
        """Minimal-deviation evasion: the SAFE candidate nearest the home point (station / berth centre);
        with no safe candidate, the one with the largest clearance. While the nearest return closes, SAFE also
        requires the point to lie off its line of sight on one committed side (_corridor_ok). The previous
        target is kept while it stays safe (or, with none safe, while its clearance is within one plan cell of
        the best)."""
        scores = [_clearance(p, centers, radii) for p in candidates]
        distance = [float(np.linalg.norm(p - home)) for p in candidates]
        safe = [j for j in range(len(candidates)) if scores[j] >= self.evade_clearance]
        previous = self.evade_target[i]
        if self.closing[i] and host["clear_mask"][i].any():
            lateral = [j for j in safe if self._corridor_ok(i, candidates[j], pos, host)]
            if lateral:
                safe = lateral
                if not self.evade_side[i]:
                    keep = host["clear_mask"][i]
                    j0 = int(np.argmin(np.where(keep, host["ranges"][i], np.inf)))
                    u = host["direction"][i][j0]
                    rel = candidates[min(safe, key=lambda j: (distance[j], -scores[j]))] - host["points"][i][j0]
                    self.evade_side[i] = 1 if float(u[0] * rel[1] - u[1] * rel[0]) > 0 else -1
                if previous is not None and not self._corridor_ok(i, previous, pos, host):
                    previous = None
        else:
            self.evade_side[i] = 0
        if safe:
            best = candidates[min(safe, key=lambda j: (distance[j], -scores[j]))]
            if previous is not None and _clearance(previous, centers, radii) >= self.evade_clearance:
                best = previous
        else:
            top = max(range(len(candidates)), key=lambda j: (scores[j], -distance[j]))
            best = candidates[top]
            if previous is not None and _clearance(previous, centers, radii) >= scores[top] - self.cell:
                best = previous
        self.evade_target[i] = np.asarray(best, dtype=np.float64)
        return self.evade_target[i]

    # ---------------------------------------------------------- planning --
    def _plan(self, i, pos, goal, centers, radii, host):
        from classical_planner_pid import plan_with_ladder, resample_waypoints

        all_c, all_r = self._map(i, centers, radii, host)
        self.counters["plans"] += 1
        path, _inflation = plan_with_ladder(pos, goal, all_c, all_r, cell_m=self.cell, ladder=self.ladder)
        if path is None:
            self.counters["plan_failures"] += 1
            return None
        return resample_waypoints(path, spacing_m=self.spacing)

    def _fallback_target(self, pos, centers, goal):
        if len(centers) == 0:
            return np.asarray(goal, dtype=np.float64).copy()
        nearest = centers[int(np.argmin(np.linalg.norm(centers - pos, axis=1)))]
        away = pos - nearest
        norm = float(np.linalg.norm(away))
        away = away / norm if norm > 1.0e-9 else np.array((1.0, 0.0))
        return pos + self.fallback_distance * away

    def _map(self, i, centers, radii, host):
        if not self.walls_enabled:
            return centers.reshape(-1, 2), radii.reshape(-1)
        if len(centers) == 0:
            return host["walls"][i], host["wall_radii"][i]
        return np.vstack((host["walls"][i], centers)), np.concatenate((host["wall_radii"][i], radii))

    def _plan_blocked(self, i, pos, centers, radii, host):
        """True when the rest of the current plan (from the vessel on) is blocked at the thinnest
        ladder rung on the CURRENT map (the plan is kept otherwise: no side flip between re-plans)."""
        from classical_planner_pid import hazard_geometry

        all_c, all_r = self._map(i, centers, radii, host)
        if len(all_c) == 0:
            return False
        hg = hazard_geometry()
        points = [pos] + [p for p in self.plans[i][self.plan_index[i]:]]
        for a, b in zip(points[:-1], points[1:]):
            if float(np.linalg.norm(b - a)) > 1.0e-9 and hg.direct_segment_blocked(
                    a, b, all_c, all_r, inflation_m=self.keep_inflation):
                return True
        return False

    def _target(self, i, pos, centers, radii, within_clear, host):
        goal, hold, key = self._goal(i, pos, centers, radii, within_clear, host)
        # Re-plan now: no plan, a new goal kind / leg, a pass-through plan run to its end, or a hold
        # target (evasion point / station / berth) that moved by more than one planning cell. At the
        # journal planner's cadence (every replan_every steps): re-plan when the rest of the plan is
        # blocked on the current map, or retry after a failed plan (fallback).
        replan = (self.plans[i] is None or key != self.plan_key[i]
                  or (not hold and self.plan_index[i] >= len(self.plans[i]) - 1
                      and float(np.linalg.norm(self.plans[i][-1] - pos)) < self.accept)
                  or (hold and float(np.linalg.norm(self.plan_goal[i] - goal)) > self.cell))
        if not replan and self.replan_in[i] <= 0:
            self.replan_in[i] = self.replan_every
            replan = self.fallback[i] or self._plan_blocked(i, pos, centers, radii, host)
        if replan:
            waypoints = self._plan(i, pos, goal, centers, radii, host)
            self.fallback[i] = waypoints is None
            if waypoints is None:
                waypoints = np.vstack((pos, self._fallback_target(pos, centers, goal)))
            self.plans[i], self.plan_index[i], self.plan_key[i] = waypoints, 0, key
            self.plan_hold[i], self.plan_goal[i] = hold and not self.fallback[i], goal
            self.replan_in[i] = self.replan_every
        else:
            self.replan_in[i] -= 1
        plan = self.plans[i]
        index = self.plan_index[i]
        while index < len(plan) - 1 and float(np.linalg.norm(plan[index] - pos)) < self.accept:
            index += 1
        self.plan_index[i] = index
        final = index >= len(plan) - 1
        if self.fallback[i]:
            self.counters["fallback_steps"] += 1
        return plan[index], bool(final and self.plan_hold[i])

    # ---------------------------------------------------------- tracking --
    def _track(self, pos, yaw, surge, speeds, targets, hold_rows, zones, active_rows):
        """Journal planner LOS/PID (pass-through) or the base deadband hold law, float64 env-local.

        Same arithmetic as journal_controllers.LOSController.act: pass-through rows yaw = PID(e),
        thrust = 0.2 + 0.8 cos e; hold rows the "deadband" branch (zero thrust and yaw, PID memory
        wiped inside the zone; speed profile outside, 0.2 creep while |e| >= pi / 4).
        """
        error = jc.wrap(torch.atan2(targets[:, 1] - pos[:, 1], targets[:, 0] - pos[:, 0]) - yaw)
        zone, slow = zones, self.hold_zone[1]
        distance = torch.linalg.vector_norm(targets - pos, dim=-1)
        inside = hold_rows & (distance <= zone)
        yaw_cmd = self.pid(error, active=active_rows & ~inside)
        profile = ((distance - zone) / (slow - zone)).clamp(0.0, 1.0)
        hold_thrust = torch.where(error.abs() < math.pi / 4.0, profile, torch.full_like(profile, 0.2))
        thrust = torch.where(hold_rows, hold_thrust, jc.pass_through_thrust(error))
        thrust = torch.where(inside, torch.zeros_like(thrust), thrust)
        yaw_cmd = torch.where(inside, torch.zeros_like(yaw_cmd), yaw_cmd)
        self.pid.reset(inside)
        # Speed cap: never faster than u_ref while avoiding, zero while giving way (P law on the estimated
        # surge, astern thrust allowed to brake): the time a dose grants (k t_avoid at u_ref) is the time
        # the layer actually gets.
        cap = (self.speed_gain * (speeds - surge)).clamp(-1.0, 1.0)
        thrust = torch.minimum(thrust, cap)
        return torch.stack((thrust, yaw_cmd), dim=-1)

    # ------------------------------------------------------------- act --
    def act(self, obs=None):
        base_action = self.base_controller.act(obs)
        self.step += 1
        inputs = jc.read_journal_inputs(self.base)
        seen = self.observe(inputs)
        need = seen["near_trigger"] | self.active_t
        if not bool(need.any()):
            return base_action
        host = self._host(seen)
        n = self.num_envs
        if self.memory_rule == "passed":
            for i in range(n):
                if self.active[i] and not host["within_clear"][i]:
                    self._carve_memory(i, host)
        for i in range(n):
            if host["within_clear"][i]:
                self.last_seen[i] = self.step
            if not self.active[i] and host["near_trigger"][i]:
                self._activate(i)
            elif (self.active[i] and self._memory_released(i, host)
                  and self._near_reference(i, host["pos"][i], host)):
                self._deactivate(i)
        rows = [i for i in range(n) if self.active[i]]
        if not rows:
            return base_action
        targets = host["pos"].copy()
        holds = np.zeros(n, dtype=bool)
        speeds = np.full(n, self.speed_cap, dtype=np.float64)
        # Hold-law zone per row: the base law's own zone for the station / berth target; one planning cell
        # for an evasion point (the base zone, e.g. the 1.5 s m hold radius, would make an evasion shorter
        # than the zone a no-op).
        zones = np.full(n, self.hold_zone[0], dtype=np.float64)
        for i in rows:
            within = bool(host["within_clear"][i])
            if within:
                keep = host["clear_mask"][i]
                centers = host["points"][i][keep] + self.r_obs * host["direction"][i][keep]
                radii = self.r_obs + (host["ranges"][i][keep] + self.r_obs) * self.bearing_pad
                surface = host["points"][i][keep].copy()
                self.memory[i] = (centers, radii, surface, np.zeros_like(surface))
                self._update_closing(i, host)
            elif self._memory_holds(i):
                centers, radii = self.memory[i][:2]        # last seen discs (between two rays, behind the fan)
                within = True
                self.counters["memory_steps"] += 1
            else:
                centers, radii = np.zeros((0, 2)), np.zeros(0)
                self.history[i], self.closing[i] = [], False
            targets[i], holds[i] = self._target(i, host["pos"][i], centers, radii, within, host)
            if self.plan_key[i] == "evade":
                zones[i] = self.cell
            if self.closing[i]:
                speeds[i] = math.inf                       # a closing (moving) return: no u_ref cap
                self.counters["closing_steps"] += 1
            self.episode_active_steps[i] += 1
        self.counters["active_steps"] += len(rows)
        self.counters["max_simultaneous_active"] = max(self.counters["max_simultaneous_active"], len(rows))
        active_rows = torch.zeros(n, dtype=torch.bool, device=self.device)
        active_rows[rows] = True
        avoid = self._track(seen["pos"], seen["yaw"], seen["surge"],
                            torch.as_tensor(speeds, dtype=torch.float64, device=self.device),
                            torch.as_tensor(targets, dtype=torch.float64, device=self.device),
                            torch.as_tensor(holds, device=self.device),
                            torch.as_tensor(zones, dtype=torch.float64, device=self.device), active_rows)
        out = base_action.clone()
        out[active_rows] = avoid[active_rows].to(out.dtype)
        return out

    # ------------------------------------------------------------ memory --
    def _memory_holds(self, i):
        """The remembered discs are in force this step (rule "steps": at most obstacle_memory_steps steps since the last
        return inside the clear gap; rule "passed": an unpassed remembered surface point lies within the clear gap)."""
        if self.memory[i] is None:
            return False
        if self.memory_rule == "steps":
            return self.step - self.last_seen[i] <= self.memory_steps
        return self.memory_in_use[i]

    def _memory_released(self, i, host):
        """Hand-back allowed by the memory: nothing inside the clear gap now and the memory no longer holds."""
        if self.memory_rule == "steps":
            return self.step - self.last_seen[i] > self.memory_steps
        return not host["within_clear"][i] and not self._memory_holds(i)

    def _carve_memory(self, i, host):
        """G24b rule "passed": drop the remembered entries the current rays see through on both sides (planner_memory
        S1: the planner's test and side bookkeeping), then flag whether a kept surface point lies within the clear gap
        of the footprint. Entries outside the clear gap are kept (no expiry by distance), only not in force."""
        self.memory_in_use[i] = False
        if self.memory[i] is None:
            return
        centers, radii, surface, side = self.memory[i]
        if len(surface):
            carved, side = self.carver._carve(surface, side, host["pos"][i], host["direction"][i], host["ranges"][i],
                                              float(host["yaw"][i]))
            if carved.any():
                self.counters["memory_dropped"] += int(carved.sum())
                keep = ~carved
                centers, radii, surface, side = centers[keep], radii[keep], surface[keep], side[keep]
            self.memory[i] = (centers, radii, surface, side)
        if len(surface):
            yaw = float(host["yaw"][i])
            c, s = math.cos(yaw), math.sin(yaw)
            rel = surface - host["pos"][i]
            bx, by = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]
            x_min, x_max, y_min, y_max = self.footprint
            dx = np.maximum(np.maximum(x_min - bx, bx - x_max), 0.0)
            dy = np.maximum(np.maximum(y_min - by, by - y_max), 0.0)
            self.memory_in_use[i] = bool((np.hypot(dx, dy) <= self.clear).any())

    def _host(self, seen):
        host = {k: seen[k].detach().cpu().numpy() for k in ("pos", "yaw", "points", "direction", "near_trigger",
                                                             "within_clear", "ranges", "velocity")}
        host["clear_mask"] = (seen["unexplained"] & (seen["gap"] <= self.clear)).detach().cpu().numpy()
        if self.mode == "path_follow":
            host["waypoints"] = self.base.waypoints.detach().cpu().numpy().astype(np.float64)
            host["gates"] = self.base.gates_passed.detach().cpu().numpy()
        elif self.mode == "station_keep":
            host["station"] = (self.base.hold_point.to(torch.float64)
                               - self.base.scene.env_origins[:, :2].to(torch.float64)).detach().cpu().numpy()
        else:
            for name in ("walls", "wall_radii", "dock_center", "dock_heading"):
                host[name] = seen[name].detach().cpu().numpy()
        return host

    def _activate(self, i):
        self.active[i] = True
        self.active_t[i] = True
        self.plans[i], self.plan_key[i], self.evade_target[i], self.fallback[i] = None, None, None, False
        self.history[i], self.closing[i], self.evade_side[i] = [], False, 0
        self.last_seen[i], self.memory[i], self.memory_in_use[i] = self.step, None, False
        self.replan_in[i] = 0
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[i] = True
        self.pid.reset(mask)
        self.counters["activations"] += 1
        self.episode_activations[i] += 1

    def _deactivate(self, i):
        self.active[i] = False
        self.active_t[i] = False
        self.plans[i], self.plan_key[i], self.evade_target[i], self.fallback[i] = None, None, None, False
        self.counters["handbacks"] += 1

    # ------------------------------------------------------------ reset --
    def reset(self, done):
        self.base_controller.reset(done)
        mask = torch.as_tensor(done, device=self.device).reshape(-1).bool()
        if not bool(mask.any()):
            return
        for i in torch.nonzero(mask).flatten().detach().cpu().tolist():
            if self.episode_activations[i]:
                self.episode_rows.append([int(i), int(self.episode_index[i]), int(self.episode_active_steps[i]),
                                          int(self.episode_activations[i])])
                self.counters["episodes_with_activation"] += 1
            self.counters["episodes_finished"] += 1
            self.episode_index[i] += 1
            self.episode_active_steps[i] = 0
            self.episode_activations[i] = 0
            self.active[i] = False
            self.active_t[i] = False
            self.plans[i], self.plan_key[i], self.evade_target[i], self.fallback[i] = None, None, None, False
            self.history[i], self.closing[i], self.evade_side[i] = [], False, 0
            self.last_seen[i], self.memory[i], self.memory_in_use[i] = -(10 ** 9), None, False
            self.replan_in[i] = 0
        self.pid.reset(mask)

    # -------------------------------------------------------- telemetry --
    def telemetry(self):
        """Run telemetry for the certificate (controller configuration block, written at the end)."""
        open_rows = [[i, int(self.episode_index[i]), int(self.episode_active_steps[i]),
                      int(self.episode_activations[i])] for i in range(self.num_envs) if self.episode_activations[i]]
        return {**{k: int(v) for k, v in self.counters.items()},
                "episode_rows_fields": ["env", "episode", "active_steps", "activations"],
                "episode_rows": [list(r) for r in self.episode_rows], "unfinished_rows": open_rows,
                "control_step_s": self.dt}


# ------------------------------------------------------------ certificate --
INFORMATION = ("G17 avoidance layer: the journal planner's inputs only (D5 own-position estimate, root-link "
               "heading, 36 rays; journal_controllers.read_journal_inputs); docking berth walls rebuilt from the "
               "published berth pose and the frozen plan; never D7 objects, the static obstacle list, the "
               "appearing slot, sea, current or wind")


def controller_block(mode, avoidance, *, hold_zone, berth_walls):
    """header.controller.configuration['avoidance'] (constants + provenance; telemetry added at the end)."""
    return {"format": AVOID_FORMAT, "layer_version": LAYER_VERSION, "controller_name": CONTROLLER_NAMES[mode],
            "base_controller_name": BASE_CONTROLLER_NAMES[mode], "base_controller_class":
            "journal_controllers.JournalBaselineController", "task": TASK_OF_MODE[mode], "mode": mode,
            "information": INFORMATION, "motion_prediction": False,
            "static_geometry": ("docking berth walls (journal_layouts_v2._berth_structure from the plan, rotated by "
                                "dock_heading about dock_point)" if mode == "docking" and berth_walls else "none"),
            "file": "scripts/" + Path(avoidance["file"]).name, "file_sha256": avoidance["file_sha256"],
            "rule_constants": avoidance["rule_constants"], "vessel_constants": avoidance["vessel_constants"],
            "task_constants": avoidance["task_constants"], "hold_zone_m": [float(x) for x in hold_zone],
            "telemetry": None}


def hold_zone_for(mode, gains, *, dt, zone_radius_m, slow_radius_m, tolerance_m, scale):
    """(zone, slow) of the hold law for hold targets: the base controller's own radii.

    station keeping: JournalBaselineController zone / slow radius; docking: DockingLaw stop /
    brake radius (computed by DockingLaw itself; RunupDockingLaw inherits them); path following has no hold
    target.
    """
    if mode == "docking":
        law = jc.DockingLaw(1, "cpu", torch.float32, dt, gains, tolerance_m=tolerance_m, scale=scale)
        return (float(law.stop_radius), float(law.brake_radius))
    return (float(zone_radius_m), float(slow_radius_m))


def prepare(env_cfg, mode, gains, *, dt, zone_radius_m, slow_radius_m, tolerance_m, scale, path=None):
    """(frozen constants, certificate block) of the journal controller of ``mode`` (before the run)."""
    vessel = str(env_cfg.vehicle)
    plan = getattr(env_cfg, "journal_plan", None)
    if plan is None:
        raise RuntimeError("the avoidance layer needs a journal cfg (journal_plan)")
    avoidance = load_avoidance(vessel, TASK_OF_MODE[mode], path)
    tier = int(getattr(plan, "geometry_tier", JOURNAL_TIER))
    if tier != int(avoidance["task_constants"]["tier"]):
        raise RuntimeError("journal cfg tier %d differs from the frozen avoidance tier %d"
                           % (tier, avoidance["task_constants"]["tier"]))
    berth_walls = bool(getattr(env_cfg, "berth_walls", True)) if mode == "docking" else False
    hold_zone = hold_zone_for(mode, gains, dt=dt, zone_radius_m=zone_radius_m, slow_radius_m=slow_radius_m,
                              tolerance_m=tolerance_m, scale=scale)
    return avoidance, controller_block(mode, avoidance, hold_zone=hold_zone, berth_walls=berth_walls)


def journal_hold_mode(mode, requested=None):
    """G24b: hold mode of the journal base law of ``mode`` (classical_baseline resolves --hold-mode with it).

    Station keeping: the frozen rule constant station_hold_mode ("continuous_v2"); any other request is refused
    (the certificate name los_pid_cont_avoid_v2 means that law). Path following and docking have no hold zone in
    the base law; their value stays the request (default "deadband"), as before G24b.
    """
    if mode == "station_keep":
        rule = RULE_CONSTANTS["station_hold_mode"]
        if requested not in (None, rule):
            raise ValueError("journal station keeping holds with %r (G24b owner decision); --hold-mode %r refused"
                             % (rule, requested))
        return rule
    return "deadband" if requested is None else requested


def runup_spec(mode, avoidance):
    """G24b: RunupDockingLaw parameters of the journal docking base law from the frozen constants (None otherwise):
    run-up point x_b = task constant runup_x_m, acceptance = the layer's waypoint acceptance radius accept_radius_m."""
    if mode != "docking" or "runup_x_m" not in avoidance["task_constants"]:
        return None
    return {"runup_x_m": float(avoidance["task_constants"]["runup_x_m"]),
            "accept_m": float(avoidance["vessel_constants"]["accept_radius_m"])}


def build_controller(base, env_cfg, mode, gains, avoidance, *, dt, zone_radius_m, slow_radius_m, tolerance_m, scale,
                     hold_mode=None, dtype=torch.float32):
    """The journal controller exactly as classical_baseline builds it: JournalBaselineController (same arguments
    as before G17; G24b: the docking law RunupDockingLaw from runup_spec) wrapped by the layer. ``hold_mode``
    None = journal_hold_mode(mode) (G24b; classical_baseline passes the value it resolved the same way)."""
    if hold_mode is None:
        hold_mode = journal_hold_mode(mode)
    base_controller = jc.JournalBaselineController(base, mode, gains, dt=dt, zone_radius_m=zone_radius_m,
                                                   slow_radius_m=slow_radius_m, tolerance_m=tolerance_m,
                                                   scale=scale, hold_mode=hold_mode, dtype=dtype,
                                                   runup=runup_spec(mode, avoidance))
    berth_walls = bool(getattr(env_cfg, "berth_walls", True)) if mode == "docking" else False
    return AvoidanceLayerController(base, base_controller, mode, avoidance, gains, dt=dt,
                                    max_range_m=jc.ray_max_range(env_cfg), plan_params=env_cfg.journal_plan.params,
                                    berth_walls=berth_walls)


def _main(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Freeze or check scripts/classical_avoidance_v2.json (G17).")
    parser.add_argument("--freeze", action="store_true", help="write the file from the current derivation")
    parser.add_argument("--check", action="store_true", help="compare the file with the current derivation")
    args = parser.parse_args(argv)
    text = document_text(derive_document())
    if args.freeze:
        AVOID_PATH.write_bytes(text.encode("ascii"))
        print("wrote %s sha256 %s" % (AVOID_PATH, _lf_sha256(AVOID_PATH)))
        return 0
    current = AVOID_PATH.read_bytes().replace(b"\r\n", b"\n").decode("ascii") if AVOID_PATH.is_file() else None
    same = current == text
    print("classical_avoidance_v2.json %s the current derivation" % ("EQUALS" if same else "DIFFERS FROM"))
    return 0 if same else 1


__all__ = ["AVOID_PATH", "AVOID_FORMAT", "LAYER_VERSION", "CONTROLLER_NAMES", "BASE_CONTROLLER_NAMES", "TASK_OF_MODE",
           "RULE_CONSTANTS", "derive_vessel_constants", "derive_task_constants", "derive_document", "load_avoidance",
           "berth_wall_template", "footprint_gap", "ray_points", "explained_by_walls", "AvoidanceLayerController",
           "controller_block", "hold_zone_for", "prepare", "build_controller", "journal_hold_mode", "runup_spec",
           "INFORMATION"]

if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
