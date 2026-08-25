"""Privileged planner + LOS/PID classical baseline for the hazard family.

INFORMATION-DIET DISCLOSURE (accepted, published beside every number): unlike
classical_baseline.py, this controller reads PRIVILEGED ground truth from the
environment -- per-episode obstacle centers/radii, the goal (or gate chain),
and the boat pose. It exists to pin the "planning with a perfect map" floor
next to the learned observation-only policies; it is NOT an observation-only
baseline and must never be presented as one.

Per episode it:
  1. reads the layout the env just sampled (world frame);
  2. plans the shortest feasible route with the SAME occupancy machinery the
     scoring oracle uses (hazard_geometry.bfs_geodesic_length raster + FIFO
     BFS; a path-returning mirror lives in this file because the oracle only
     returns the LENGTH and hazard_geometry must not be edited);
  3. shortcuts the staircase with analytic segment checks
     (direct_segment_blocked), downsamples to ~2 m waypoints;
  4. tracks the waypoints with line-of-sight guidance + heading PID
     (classical_baseline's hand-set gains: kp 2.0, ki 0.0, kd 0.5);
  5. replans every --replan-every control steps (default 60), so drift and
     the iceberg family's obstacle fields are re-solved from the live pose.

PathHazard is gate-ordered: the plan is the gate sequence in order, each leg
planned around the on-line blockers, and it replans whenever the env's
gates_passed advances.

Records: the eval_v6_frozen per-episode schema, unchanged, plus a
"controller": "planner_los_pid" tag -- scripts/usv10k_score.py consumes the
--out JSON as-is (save it as gcerts/gcert_<name>_e<seed>.json to score it).

The PID class is duplicated (not imported) from classical_baseline.py on
purpose: that module parses argv and launches the Isaac app at import time,
so importing it from any other script would launch a second AppLauncher.

GPU examples (16 envs / 16 episodes smoke; certification uses 64/128):

  python scripts/classical_planner_pid.py --task Isaac-USV-HazardCross-Direct-v1 \
      --num_envs 16 --episodes 16 --level 0 --eval-seed 42 --headless \
      --out wf_pid_cross.json
  python scripts/classical_planner_pid.py --task Isaac-USV-PathHazard-Direct-v1 \
      --num_envs 16 --episodes 16 --eval-seed 42 --headless --out wf_pid_ph.json

Tier-regrade probes ride the same knobs as eval_imbalance.py:
  --gate-width-override 1.35        (HazardCross, physical metres)
  --fortress-aperture-override 5.0  (HazardBandFortWay)
  --on-line-blockers-override 4     (PathHazard)
  --set FIELD=VALUE                 (any other cfg scalar, repeatable)

CPU validation without Isaac (fabricated layouts, admission-audit style):

  python scripts/classical_planner_pid.py --selftest
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from collections import deque

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# torch is imported lazily: the house pattern (eval_v6_frozen, classical
# baseline) never loads torch before AppLauncher on the Isaac boxes, and the
# CPU selftest wants to run on a plain python without launching Kit.
torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch

        torch = _torch
    return torch


# ---------------------------------------------------------------------------
# Geometry access: on the GPU boxes the tasks live in the deployed
# isaaclab_tasks.direct tree (the copy the running env actually samples
# layouts from); in the repo they live under tasks/<family>/ with no package
# __init__, so the fallback mirrors the test files' direct-import style.
# ---------------------------------------------------------------------------
_GEOMETRY_CACHE: dict[str, object] = {}


def _geometry(family: str):
    module_name = {
        "hazard_nav": "hazard_geometry",
        "path_hazard": "path_hazard_geometry",
    }[family]
    if family in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[family]
    try:
        module = importlib.import_module(
            f"isaaclab_tasks.direct.{family}.{module_name}"
        )
    except ImportError:
        directory = os.path.join(_REPO_ROOT, "tasks", family)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        module = importlib.import_module(module_name)
    _GEOMETRY_CACHE[family] = module
    return module


def hazard_geometry():
    return _geometry("hazard_nav")


# ---------------------------------------------------------------------------
# Planning core (numpy only; unit-tested by --selftest without Isaac)
# ---------------------------------------------------------------------------


def plan_grid_path(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    cell_m: float,
    inflation_m: float,
    padding_m: float = 6.0,
    snap_start_m: float = 0.0,
) -> np.ndarray | None:
    """Path-returning mirror of hazard_geometry.bfs_geodesic_length.

    The oracle exposes only the route LENGTH and hazard_geometry may not be
    edited, so the extent snapping, the _occupancy_grid raster, the FIFO BFS
    order, and the sub-cell endpoint attachment are duplicated here verbatim
    -- the polyline this returns has exactly the length the oracle reports
    for the same arguments. Returns (M, 2) float64 [exact start, free cell
    centres..., exact goal] or None when no route exists at this inflation.

    snap_start_m > 0 additionally allows the start (the live boat, which may
    sit inside the INFLATED ring of an obstacle it grazed) to attach to the
    nearest free cell within that many metres; the oracle itself would return
    None there, so snapping is opt-in and used only for the boat-side start.
    """
    hg = hazard_geometry()
    if cell_m <= 0.0 or padding_m <= 0.0:
        raise ValueError("cell_m and padding_m must be positive")
    start = np.asarray(start_xy, dtype=np.float64).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    radii_i = hg.inflated_radii(radii, inflation_m)

    if float(np.linalg.norm(goal - start)) <= cell_m:
        return np.vstack((start, goal))

    if len(centers):
        x_min = min(float(start[0]), float(goal[0]), float(np.min(centers[:, 0] - radii_i)))
        x_max = max(float(start[0]), float(goal[0]), float(np.max(centers[:, 0] + radii_i)))
        y_min = min(float(start[1]), float(goal[1]), float(np.min(centers[:, 1] - radii_i)))
        y_max = max(float(start[1]), float(goal[1]), float(np.max(centers[:, 1] + radii_i)))
    else:
        x_min = min(float(start[0]), float(goal[0]))
        x_max = max(float(start[0]), float(goal[0]))
        y_min = min(float(start[1]), float(goal[1]))
        y_max = max(float(start[1]), float(goal[1]))

    x_min = math.floor((x_min - padding_m) / cell_m) * cell_m
    x_max = math.ceil((x_max + padding_m) / cell_m) * cell_m
    y_min = math.floor((y_min - padding_m) / cell_m) * cell_m
    y_max = math.ceil((y_max + padding_m) / cell_m) * cell_m
    xs = np.arange(x_min, x_max + 0.5 * cell_m, cell_m)
    ys = np.arange(y_min, y_max + 0.5 * cell_m, cell_m)

    occupied = hg._occupancy_grid(centers, radii, xs, ys, inflation_m)

    def nearest_index(point: np.ndarray) -> tuple[int, int]:
        col = int(np.clip(round((float(point[0]) - x_min) / cell_m), 0, len(xs) - 1))
        row = int(np.clip(round((float(point[1]) - y_min) / cell_m), 0, len(ys) - 1))
        return row, col

    start_idx = nearest_index(start)
    goal_idx = nearest_index(goal)
    if occupied[start_idx] and snap_start_m > 0.0:
        snapped = _nearest_free_cell(occupied, start_idx, int(math.ceil(snap_start_m / cell_m)))
        if snapped is not None:
            start_idx = snapped
    if occupied[start_idx] or occupied[goal_idx]:
        return None

    parent_row = np.full(occupied.shape, -1, dtype=np.int32)
    parent_col = np.full(occupied.shape, -1, dtype=np.int32)
    visited = np.zeros(occupied.shape, dtype=np.bool_)
    visited[start_idx] = True
    queue: deque[tuple[int, int]] = deque([start_idx])
    neighbours = (
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )
    while queue:
        row, col = queue.popleft()
        if (row, col) == goal_idx:
            break
        for d_row, d_col in neighbours:
            next_row, next_col = row + d_row, col + d_col
            if not (0 <= next_row < len(ys) and 0 <= next_col < len(xs)):
                continue
            if occupied[next_row, next_col] or visited[next_row, next_col]:
                continue
            visited[next_row, next_col] = True
            parent_row[next_row, next_col] = row
            parent_col[next_row, next_col] = col
            queue.append((next_row, next_col))

    if not visited[goal_idx]:
        return None

    cells: list[tuple[int, int]] = []
    row, col = goal_idx
    while (row, col) != start_idx:
        cells.append((row, col))
        row, col = int(parent_row[row, col]), int(parent_col[row, col])
    cells.append(start_idx)
    cells.reverse()
    points = np.array([(xs[c], ys[r]) for r, c in cells], dtype=np.float64)
    return np.vstack((start, points, goal))


def _nearest_free_cell(
    occupied: np.ndarray, index: tuple[int, int], max_ring: int
) -> tuple[int, int] | None:
    """Closest free cell to ``index`` within a square ring search."""
    height, width = occupied.shape
    best: tuple[int, int] | None = None
    best_d2 = math.inf
    for ring in range(1, max_ring + 1):
        for d_row in range(-ring, ring + 1):
            for d_col in range(-ring, ring + 1):
                if max(abs(d_row), abs(d_col)) != ring:
                    continue
                row, col = index[0] + d_row, index[1] + d_col
                if not (0 <= row < height and 0 <= col < width):
                    continue
                if occupied[row, col]:
                    continue
                d2 = d_row * d_row + d_col * d_col
                if d2 < best_d2:
                    best_d2 = d2
                    best = (row, col)
        if best is not None:
            return best
    return None


def shortcut_path(
    path: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    inflation_m: float,
    slack_m: float = 0.05,
) -> np.ndarray:
    """Greedy furthest-visible shortcutting with analytic segment checks.

    Visibility uses direct_segment_blocked at (inflation_m - slack_m): the
    small slack exists because a raw grid polyline's diagonal hops can dip a
    few centimetres inside the exact inflated circle even though both cell
    centres are free. Grid-adjacent hops are always accepted as a fallback,
    so this never fails; it only straightens.
    """
    hg = hazard_geometry()
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if len(path) <= 2 or len(centers) == 0:
        return np.asarray(path, dtype=np.float64)
    check_inflation = max(inflation_m - slack_m, 0.0)
    out = [np.asarray(path[0], dtype=np.float64)]
    i = 0
    last = len(path) - 1
    while i < last:
        j = last
        while j > i + 1:
            a, b = path[i], path[j]
            if float(np.linalg.norm(b - a)) < 1.0e-9:
                break
            if not hg.direct_segment_blocked(
                a, b, centers, radii, inflation_m=check_inflation
            ):
                break
            j -= 1
        out.append(np.asarray(path[j], dtype=np.float64))
        i = j
    return np.asarray(out, dtype=np.float64)


def resample_waypoints(path: np.ndarray, *, spacing_m: float) -> np.ndarray:
    """Downsample a polyline to ~spacing_m waypoints, KEEPING every vertex.

    Vertices are corners of analytically checked shortcut segments; dropping
    one would cut a corner into an obstacle, so intermediate points are only
    ever inserted ON the polyline, never substituted for its vertices.
    """
    path = np.asarray(path, dtype=np.float64)
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    if len(path) < 2:
        return path.copy()
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        segment = b - a
        length = float(np.linalg.norm(segment))
        if length < 1.0e-9:
            continue
        steps = int(length // spacing_m)
        for k in range(1, steps + 1):
            point = a + segment * (k * spacing_m / length)
            if float(np.linalg.norm(point - b)) > 0.25:
                out.append(point)
        out.append(b)
    deduped = [out[0]]
    for point in out[1:]:
        if float(np.linalg.norm(point - deduped[-1])) > 1.0e-6:
            deduped.append(point)
    return np.asarray(deduped, dtype=np.float64)


def plan_with_ladder(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    cell_m: float,
    ladder: tuple[float, ...],
    snap_start_m: float = 1.5,
) -> tuple[np.ndarray | None, float | None]:
    """Try inflations fattest-first; return (shortcut polyline, inflation).

    The ladder exists because the admission oracles differ per family: the
    scatter/forced/iceberg oracles admit at OBSTACLE_INFLATION_M (0.65) and
    the band fortress admits at HALF_BEAM_M (0.45), so a fixed 0.65 planner
    would declare admitted band-fortress tiers unreachable. Wide corridors
    get the fattest margin; narrow tiers get whatever the oracle proved fits.
    """
    for inflation in ladder:
        path = plan_grid_path(
            start_xy,
            goal_xy,
            centers,
            radii,
            cell_m=cell_m,
            inflation_m=inflation,
            snap_start_m=snap_start_m,
        )
        if path is not None:
            return (
                shortcut_path(path, centers, radii, inflation_m=inflation),
                inflation,
            )
    return None, None


def polyline_length(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def default_ladder(plan_margin_m: float) -> tuple[float, ...]:
    hg = hazard_geometry()
    rungs = [
        hg.OBSTACLE_INFLATION_M + max(plan_margin_m, 0.0),
        hg.OBSTACLE_INFLATION_M,
        hg.HALF_BEAM_M + 0.05,
    ]
    ladder: list[float] = []
    for rung in rungs:
        if rung > 0.0 and all(abs(rung - kept) > 1.0e-9 for kept in ladder):
            ladder.append(rung)
    return tuple(ladder)


# ---------------------------------------------------------------------------
# Controller (torch; constructed only after AppLauncher inside the eval)
# ---------------------------------------------------------------------------


def _quat_rotate(quat, vec):
    """Rotate (N, 3) vectors by (N, 4) wxyz quaternions (pure torch)."""
    t = _ensure_torch()
    w = quat[:, 0:1]
    q_vec = quat[:, 1:4]
    doubled = 2.0 * t.cross(q_vec, vec, dim=-1)
    return vec + w * doubled + t.cross(q_vec, doubled, dim=-1)


class PlannerLOSPIDController:
    """Per-episode privileged planner + vectorized LOS/PID waypoint tracker."""

    def __init__(
        self,
        base,
        *,
        replan_every: int,
        waypoint_spacing_m: float,
        accept_radius_m: float,
        plan_cell_m: float,
        ladder: tuple[float, ...],
        kp: float,
        ki: float,
        kd: float,
    ) -> None:
        t = _ensure_torch()
        if replan_every < 1:
            raise ValueError("replan_every must be >= 1")
        self.base = base
        self.mode = (
            "gates"
            if hasattr(base, "waypoints") and hasattr(base, "gates_passed")
            else "goal"
        )
        self.num_envs = int(base.num_envs)
        self.device = base.device
        self.dt = float(base.control_step_s)
        self.replan_every = int(replan_every)
        self.spacing = float(waypoint_spacing_m)
        self.accept = float(accept_radius_m)
        self.cell = float(plan_cell_m)
        self.ladder = tuple(ladder)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)

        self.plans: list[np.ndarray | None] = [None] * self.num_envs
        self.wp_index = np.zeros(self.num_envs, dtype=np.int64)
        self.replan_in = np.zeros(self.num_envs, dtype=np.int64)
        self.need_plan = np.ones(self.num_envs, dtype=bool)
        self.planned_gate = np.zeros(self.num_envs, dtype=np.int64)
        self.straight_fallbacks = 0
        self.plan_calls = 0

        self.integral = t.zeros(self.num_envs, device=self.device)
        self.prev_error = t.zeros(self.num_envs, device=self.device)
        self.initialized = t.zeros(
            self.num_envs, dtype=t.bool, device=self.device
        )
        self._origins = (
            base.scene.env_origins[:, :2].detach().cpu().numpy().astype(np.float64)
        )

    # -- privileged reads ---------------------------------------------------

    def _bow_2d(self):
        helper = getattr(self.base, "_forward_2d", None)
        if callable(helper):
            return helper()
        bow = _quat_rotate(self.base._root_quat(), self.base.forward_vec)[:, :2]
        return bow / bow.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def _obstacles_np(self, env_index: int) -> tuple[np.ndarray, np.ndarray]:
        active = self.base.obstacle_active[env_index].detach().cpu().numpy()
        centers = (
            self.base.obstacle_centers[env_index].detach().cpu().numpy()[active]
        )
        radii = self.base.obstacle_radii[env_index].detach().cpu().numpy()[active]
        return centers.astype(np.float64), radii.astype(np.float64)

    def _targets_np(self, env_index: int, gates_passed: int) -> np.ndarray:
        """Ordered remaining target points for one env (world frame)."""
        if self.mode == "gates":
            gates_local = self.base.waypoints[env_index].detach().cpu().numpy()
            gates_world = gates_local.astype(np.float64) + self._origins[env_index]
            first = min(gates_passed, len(gates_world) - 1)
            return gates_world[first:]
        goal = self.base.target_pos[env_index].detach().cpu().numpy()
        return goal.astype(np.float64).reshape(1, 2)

    # -- planning -----------------------------------------------------------

    def _plan_env(self, env_index: int, position: np.ndarray, gates_passed: int):
        centers, radii = self._obstacles_np(env_index)
        targets = self._targets_np(env_index, gates_passed)
        pieces: list[np.ndarray] = []
        cursor = np.asarray(position, dtype=np.float64)
        for target in targets:
            leg, _inflation = plan_with_ladder(
                cursor,
                target,
                centers,
                radii,
                cell_m=self.cell,
                ladder=self.ladder,
            )
            self.plan_calls += 1
            if leg is None:
                # Never strand the episode: an unplannable leg (should not
                # happen on admitted layouts) degrades to the straight line
                # and is counted so the summary discloses it.
                leg = np.vstack((cursor, target))
                self.straight_fallbacks += 1
            pieces.append(resample_waypoints(leg, spacing_m=self.spacing))
            cursor = np.asarray(target, dtype=np.float64)
        waypoints = pieces[0]
        for piece in pieces[1:]:
            waypoints = np.vstack((waypoints, piece[1:] if len(piece) > 1 else piece))
        self.plans[env_index] = waypoints
        self.wp_index[env_index] = 0
        self.planned_gate[env_index] = gates_passed
        self.replan_in[env_index] = self.replan_every
        self.need_plan[env_index] = False

    # -- control ------------------------------------------------------------

    def act(self):
        t = _ensure_torch()
        base = self.base
        position_t = base._com_xy()
        bow_t = self._bow_2d()
        position = position_t.detach().cpu().numpy().astype(np.float64)
        if self.mode == "gates":
            gates_passed = base.gates_passed.detach().cpu().numpy()
            num_gates = int(base.waypoints.shape[1])
        else:
            gates_passed = np.zeros(self.num_envs, dtype=np.int64)
            num_gates = 0

        target_np = np.zeros((self.num_envs, 2), dtype=np.float64)
        for i in range(self.num_envs):
            gate = int(gates_passed[i])
            if (
                self.need_plan[i]
                or self.replan_in[i] <= 0
                or (self.mode == "gates" and gate != int(self.planned_gate[i]))
            ):
                self._plan_env(i, position[i], gate)
            else:
                self.replan_in[i] -= 1
            plan = self.plans[i]
            index = int(self.wp_index[i])
            while (
                index < len(plan) - 1
                and float(np.linalg.norm(plan[index] - position[i])) < self.accept
            ):
                index += 1
            self.wp_index[i] = index
            target_np[i] = plan[index]

        targets = t.as_tensor(target_np, device=self.device, dtype=position_t.dtype)
        to_target = targets - position_t
        distance = to_target.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        direction = to_target / distance
        dot = (bow_t * direction).sum(dim=-1).clamp(-1.0, 1.0)
        cross = bow_t[:, 0] * direction[:, 1] - bow_t[:, 1] * direction[:, 0]
        error = t.atan2(cross, dot)

        self.integral.add_(error * self.dt)
        wrapped_delta = (
            t.remainder(error - self.prev_error + math.pi, 2.0 * math.pi)
            - math.pi
        )
        derivative = t.where(
            self.initialized, wrapped_delta / self.dt, t.zeros_like(error)
        )
        yaw = (
            self.kp * error + self.ki * self.integral + self.kd * derivative
        ).clamp(-1.0, 1.0)
        self.prev_error.copy_(error)
        self.initialized.fill_(True)

        # Gates and single goals are pass-through targets (entry terminates or
        # advances), so keep classical_baseline's path-follow throttle: full
        # thrust on the LOS, a 0.2 creep while turning in place.
        thrust = 0.2 + 0.8 * t.cos(error).clamp(0.0, 1.0)

        if self.mode == "gates" and num_gates > 0:
            parked = t.as_tensor(
                gates_passed >= num_gates, device=self.device
            )
            thrust = t.where(parked, t.zeros_like(thrust), thrust)
            yaw = t.where(parked, t.zeros_like(yaw), yaw)
        return t.stack((thrust, yaw), dim=-1)

    def reset(self, done) -> None:
        t = _ensure_torch()
        mask = t.as_tensor(done, device=self.device).reshape(-1).bool()
        if not bool(mask.any()):
            return
        indices = t.nonzero(mask).flatten().detach().cpu().tolist()
        for i in indices:
            self.need_plan[i] = True
            self.plans[i] = None
            self.wp_index[i] = 0
            self.planned_gate[i] = 0
        self.integral[mask] = 0.0
        self.prev_error[mask] = 0.0
        self.initialized[mask] = False


# ---------------------------------------------------------------------------
# CPU selftest: fabricated layouts, admission-audit style, no Isaac
# ---------------------------------------------------------------------------


def _selftest_wall_with_gate() -> list[str]:
    hg = hazard_geometry()
    failures: list[str] = []
    radius = 0.8
    gate_half = 1.5  # 3.0 m free opening, comfortably above the 0.899 m beam
    top = hg.wall_segment_cylinders(
        np.array((10.0, gate_half + radius)),
        np.array((10.0, 14.0)),
        radius_m=radius,
    )
    bottom = hg.wall_segment_cylinders(
        np.array((10.0, -14.0)),
        np.array((10.0, -(gate_half + radius))),
        radius_m=radius,
    )
    centers = np.vstack((top, bottom))
    radii = np.full(len(centers), radius)
    start = np.array((0.0, 0.0))
    goal = np.array((20.0, 0.0))

    ladder = default_ladder(0.15)
    path, inflation = plan_with_ladder(
        start, goal, centers, radii, cell_m=0.25, ladder=ladder
    )
    if path is None:
        return ["wall_gate: no path found"]

    oracle = hg.bfs_geodesic_length(
        start, goal, centers, radii, cell_m=0.25, inflation_m=inflation
    )
    length = polyline_length(path)
    straight = float(np.linalg.norm(goal - start))
    if oracle is None:
        failures.append("wall_gate: oracle infeasible at the chosen inflation")
    elif not (straight - 1.0e-6 <= length <= oracle * 1.001):
        failures.append(
            f"wall_gate: length {length:.2f} outside [straight {straight:.2f}, "
            f"oracle {oracle:.2f}]"
        )

    crossed_in_gate = False
    for a, b in zip(path[:-1], path[1:]):
        if (a[0] - 10.0) * (b[0] - 10.0) <= 0.0 and abs(b[0] - a[0]) > 1.0e-9:
            fraction = (10.0 - a[0]) / (b[0] - a[0])
            y_cross = a[1] + fraction * (b[1] - a[1])
            if abs(y_cross) < gate_half:
                crossed_in_gate = True
    if not crossed_in_gate:
        failures.append("wall_gate: path does not cross the wall inside the gate")

    waypoints = resample_waypoints(path, spacing_m=2.0)
    gaps = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    if len(waypoints) < 2 or np.any(gaps <= 1.0e-6) or np.any(gaps > 2.0 * 1.5):
        failures.append(
            f"wall_gate: waypoint spacing violated (min {gaps.min():.3f}, "
            f"max {gaps.max():.3f})"
        )
    arc = np.concatenate(([0.0], np.cumsum(gaps)))
    if not np.all(np.diff(arc) > 0.0):
        failures.append("wall_gate: waypoints are not strictly ordered along the path")

    # Shortcut segments are admitted at (inflation - 0.05) slack, so a point
    # ON such a segment may sit up to 5 cm inside the full-inflation circle;
    # check against the slack radius, still far above the 0.45 m true hull.
    radii_slack = hg.inflated_radii(radii, inflation - 0.06)
    for point in waypoints:
        clearance = np.linalg.norm(centers - point, axis=1) - radii_slack
        if float(clearance.min()) <= 0.0:
            failures.append(
                f"wall_gate: waypoint {point} inside the slack-inflated map "
                f"(min clearance {clearance.min():.3f})"
            )
            break
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        if hg.direct_segment_blocked(
            a, b, centers, radii, inflation_m=max(inflation - 0.10, 0.0)
        ):
            failures.append(
                "wall_gate: a waypoint segment clips the inflated map "
                "(grid-slack check at inflation - 0.10)"
            )
            break
    return failures


def _selftest_forced_crossing() -> list[str]:
    hg = hazard_geometry()
    rng = np.random.default_rng(11)
    layout, gate = hg.sample_forced_crossing_layout(0, rng=rng)
    ladder = default_ladder(0.15)
    path, inflation = plan_with_ladder(
        layout.start,
        layout.goal,
        layout.centers,
        layout.radii,
        cell_m=hg.FORCED_SEAL_CELL_M,
        ladder=ladder,
    )
    if path is None:
        return ["forced: no path found on an admitted layout"]
    failures: list[str] = []
    crossed = False
    for a, b in zip(path[:-1], path[1:]):
        if (a[0] - gate.wall_x) * (b[0] - gate.wall_x) <= 0.0 and abs(
            b[0] - a[0]
        ) > 1.0e-9:
            fraction = (gate.wall_x - a[0]) / (b[0] - a[0])
            y_cross = a[1] + fraction * (b[1] - a[1])
            if abs(y_cross - gate.center[1]) < 0.5 * gate.free_width_m:
                crossed = True
    if not crossed:
        failures.append("forced: path does not transit the measured gate opening")
    length = polyline_length(path)
    if not (length <= layout.geodesic_length * 1.05 + 1.0):
        failures.append(
            f"forced: length {length:.2f} far above admission geodesic "
            f"{layout.geodesic_length:.2f} (inflation {inflation:.2f})"
        )
    return failures


def _selftest_band_fortress() -> list[str]:
    hg = hazard_geometry()
    rng = np.random.default_rng(7)
    layout = hg.sample_band_fortress_layout(0, rng=rng)
    ladder = default_ladder(0.15)
    path, inflation = plan_with_ladder(
        layout.start,
        layout.goal,
        layout.centers,
        layout.radii,
        cell_m=0.25,
        ladder=ladder,
    )
    if path is None:
        return ["bandfort: no path found on an admitted two-ring layout"]
    failures: list[str] = []
    length = polyline_length(path)
    # Admission certified geodesic at HALF_BEAM inflation; ours may be fatter.
    if not (length <= layout.geodesic_length * 1.25 + 2.0):
        failures.append(
            f"bandfort: length {length:.2f} vs admission geodesic "
            f"{layout.geodesic_length:.2f} (inflation {inflation:.2f})"
        )
    end_error = float(np.linalg.norm(path[-1] - layout.goal))
    if end_error > 1.0e-6:
        failures.append(f"bandfort: path does not end at the goal ({end_error:.3f} m)")
    # Same 0.06 m slack rationale as the wall-gate check: shortcut segments
    # are admitted at (inflation - 0.05).
    radii_slack = hg.inflated_radii(layout.radii, inflation - 0.06)
    for point in resample_waypoints(path, spacing_m=2.0):
        clearance = (
            np.linalg.norm(layout.centers - point, axis=1) - radii_slack
        )
        if float(clearance.min()) <= 0.0:
            failures.append(
                "bandfort: a waypoint lies inside the slack-inflated map"
            )
            break
    return failures


def _selftest_iceberg() -> list[str]:
    hg = hazard_geometry()
    rng = np.random.default_rng(3)
    layout = hg.sample_iceberg_layout(2, rng=rng)
    ladder = default_ladder(0.15)
    path, _inflation = plan_with_ladder(
        layout.start,
        layout.goal,
        layout.centers,
        layout.radii,
        cell_m=0.25,
        ladder=ladder,
    )
    if path is None:
        return ["iceberg: no path found on an admitted layout"]
    failures: list[str] = []
    length = polyline_length(path)
    straight = float(np.linalg.norm(layout.goal - layout.start))
    if layout.direct_blocked and length <= straight + 0.25:
        failures.append(
            f"iceberg: direct segment is blocked but the plan ({length:.2f} m) "
            f"is not a detour over {straight:.2f} m"
        )
    if not (length <= layout.geodesic_length * 1.05 + 1.0):
        failures.append(
            f"iceberg: length {length:.2f} far above admission geodesic "
            f"{layout.geodesic_length:.2f}"
        )
    return failures


def _selftest_path_hazard_chain() -> list[str]:
    pg = _geometry("path_hazard")
    rng = np.random.default_rng(5)
    layout = pg.sample_layout(rng=rng)
    ladder = default_ladder(0.15)
    failures: list[str] = []
    waypoints_all: list[np.ndarray] = []
    cursor = layout.route_points[0]
    for leg_index, gate in enumerate(layout.route_points[1:]):
        leg, _inflation = plan_with_ladder(
            cursor, gate, layout.centers, layout.radii, cell_m=0.25, ladder=ladder
        )
        if leg is None:
            failures.append(f"path_hazard: leg {leg_index} unplannable")
            return failures
        oracle = layout.leg_geodesic_lengths[leg_index]
        length = polyline_length(leg)
        if not (length <= oracle * 1.15 + 1.0):
            failures.append(
                f"path_hazard: leg {leg_index} length {length:.2f} far above "
                f"admission geodesic {oracle:.2f}"
            )
        waypoints_all.append(resample_waypoints(leg, spacing_m=2.0))
        cursor = gate
    chain = waypoints_all[0]
    for piece in waypoints_all[1:]:
        chain = np.vstack((chain, piece[1:]))
    visit_indices = []
    for gate in layout.route_points[1:]:
        distances = np.linalg.norm(chain - gate, axis=1)
        index = int(np.argmin(distances))
        if float(distances[index]) > 1.0:
            failures.append(
                f"path_hazard: gate {gate} missed by {float(distances[index]):.2f} m"
            )
        visit_indices.append(index)
    if visit_indices != sorted(visit_indices):
        failures.append(
            f"path_hazard: gates visited out of order (indices {visit_indices})"
        )
    return failures


def _selftest_oracle_mirror() -> list[str]:
    """The RAW grid path must have EXACTLY the length the scoring oracle
    reports for identical arguments -- the docstring's mirror claim."""
    hg = hazard_geometry()
    failures: list[str] = []
    for seed in (1, 2, 3):
        layout = hg.sample_layout(1, rng=np.random.default_rng(seed))
        raw = plan_grid_path(
            layout.start,
            layout.goal,
            layout.centers,
            layout.radii,
            cell_m=0.5,
            inflation_m=hg.OBSTACLE_INFLATION_M,
        )
        oracle = hg.bfs_geodesic_length(
            layout.start,
            layout.goal,
            layout.centers,
            layout.radii,
            cell_m=0.5,
            inflation_m=hg.OBSTACLE_INFLATION_M,
        )
        if raw is None or oracle is None:
            failures.append(f"mirror seed {seed}: raw or oracle infeasible")
            continue
        length = polyline_length(raw)
        if abs(length - oracle) > 1.0e-6:
            failures.append(
                f"mirror seed {seed}: raw {length:.6f} != oracle {oracle:.6f}"
            )
    return failures


def _selftest_sealed_infeasible() -> list[str]:
    """Sealing the forced-crossing gate must make the whole ladder fail,
    exactly as the admission oracle's own seal check does."""
    hg = hazard_geometry()
    rng = np.random.default_rng(21)
    layout, gate = hg.sample_forced_crossing_layout(0, rng=rng)
    seal = hg.wall_segment_cylinders(
        np.array((gate.wall_x, gate.center[1] - 0.5 * gate.free_width_m - 0.5)),
        np.array((gate.wall_x, gate.center[1] + 0.5 * gate.free_width_m + 0.5)),
        radius_m=0.8,
    )
    centers = np.vstack((layout.centers, seal))
    radii = np.concatenate((layout.radii, np.full(len(seal), 0.8)))
    path, _inflation = plan_with_ladder(
        layout.start,
        layout.goal,
        centers,
        radii,
        cell_m=hg.FORCED_SEAL_CELL_M,
        ladder=default_ladder(0.15),
    )
    if path is not None:
        return [
            f"sealed: ladder found a {polyline_length(path):.2f} m path "
            "through a sealed basin"
        ]
    return []


def _selftest_quat_bow() -> list[str]:
    t = _ensure_torch()
    failures: list[str] = []
    for yaw_deg, forward, expected in (
        (0.0, (-1.0, 0.0, 0.0), (-1.0, 0.0)),
        (90.0, (-1.0, 0.0, 0.0), (0.0, -1.0)),
        (180.0, (1.0, 0.0, 0.0), (-1.0, 0.0)),
        (45.0, (1.0, 0.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5))),
    ):
        half = math.radians(yaw_deg) / 2.0
        quat = t.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
        vec = t.tensor([list(forward)])
        rotated = _quat_rotate(quat, vec)[0, :2]
        if not np.allclose(rotated.numpy(), expected, atol=1.0e-6):
            failures.append(
                f"quat: yaw {yaw_deg} forward {forward} -> "
                f"{rotated.numpy().tolist()} expected {expected}"
            )
    return failures


def _run_selftest() -> int:
    checks = (
        ("quat_bow", _selftest_quat_bow),
        ("oracle_mirror", _selftest_oracle_mirror),
        ("wall_with_gate", _selftest_wall_with_gate),
        ("forced_crossing", _selftest_forced_crossing),
        ("sealed_infeasible", _selftest_sealed_infeasible),
        ("band_fortress", _selftest_band_fortress),
        ("iceberg", _selftest_iceberg),
        ("path_hazard_chain", _selftest_path_hazard_chain),
    )
    total_failures = 0
    for name, check in checks:
        try:
            failures = check()
        except Exception as exc:  # a crashed check is a failed check
            failures = [f"{name}: raised {type(exc).__name__}: {exc}"]
        if failures:
            total_failures += len(failures)
            for failure in failures:
                print(f"FAIL {name}: {failure}")
        else:
            print(f"PASS {name}")
    print(f"selftest: {len(checks)} checks, {total_failures} failure(s)")
    return 1 if total_failures else 0


# ---------------------------------------------------------------------------
# Isaac evaluation (mirrors eval_v6_frozen.py's protocol and record schema)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="USVBench privileged planner + LOS/PID hazard baseline."
    )
    parser.add_argument("--task", default="Isaac-USV-HazardCross-Direct-v1")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="JSON per-episode records")
    parser.add_argument("--replan-every", type=int, default=60,
                        help="Control steps between replans (default 60).")
    parser.add_argument("--waypoint-spacing", type=float, default=2.0)
    parser.add_argument("--accept-radius", type=float, default=1.0)
    parser.add_argument("--plan-cell", type=float, default=0.25,
                        help="Planner raster cell (m); 0.25 matches the "
                             "forced-crossing seal audit resolution.")
    parser.add_argument("--plan-margin", type=float, default=0.15,
                        help="Extra inflation above OBSTACLE_INFLATION_M for "
                             "the fattest ladder rung.")
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--gate-width-override", type=float, default=None,
                        help="Sets cfg gate_width_override_m (HazardCross).")
    parser.add_argument("--fortress-aperture-override", type=float, default=None,
                        help="Sets cfg fortress_aperture_override_m (BandFort*).")
    parser.add_argument("--on-line-blockers-override", type=int, default=None,
                        help="Sets cfg on_line_blockers_override (PathHazard).")
    parser.add_argument("--set", dest="extra_sets", action="append", default=[],
                        metavar="FIELD=VALUE",
                        help="Extra top-level cfg overrides (repeatable).")
    parser.add_argument("--selftest", action="store_true",
                        help="Run the CPU planning unit tests and exit "
                             "(no Isaac, no GPU).")
    return parser


def _run_isaac_eval() -> None:
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args_cli).app

    t = _ensure_torch()
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers the deployed task ids)
    from isaaclab_tasks.utils import parse_env_cfg

    # Scenario-protocol stamping, on the same dual import scripts/
    # eval_v6_frozen.py:74-80 uses so the script keeps working from the repo
    # and from the deployed task tree.  It lives here rather than at module
    # scope because scenario_draws imports torch, and the house pattern never
    # loads torch before AppLauncher on the Isaac boxes.
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
    for flag, field in (
        (args_cli.gate_width_override, "gate_width_override_m"),
        (args_cli.fortress_aperture_override, "fortress_aperture_override_m"),
        (args_cli.on_line_blockers_override, "on_line_blockers_override"),
    ):
        if flag is None:
            continue
        if not hasattr(env_cfg, field):
            raise SystemExit(f"cfg for {task} has no field {field!r}")
        setattr(env_cfg, field, type(getattr(env_cfg, field))(flag))
        print(f"set {field}={getattr(env_cfg, field)}", flush=True)
    for assignment in args_cli.extra_sets:
        field, separator, raw = assignment.partition("=")
        if not separator or not hasattr(env_cfg, field):
            raise SystemExit(f"--set target {field!r} is not a cfg field")
        current = getattr(env_cfg, field)
        caster = type(current) if isinstance(current, (int, float, bool)) else str
        setattr(env_cfg, field, caster(raw) if caster is not bool
                else raw.lower() in ("1", "true", "yes"))
        print(f"set {field}={getattr(env_cfg, field)}", flush=True)
    if hasattr(env_cfg, "curriculum_frozen"):
        env_cfg.curriculum_frozen = True
        env_cfg.eval_level = args_cli.level

    env = gym.make(task, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    env.reset()

    ladder = default_ladder(args_cli.plan_margin)
    controller = PlannerLOSPIDController(
        base,
        replan_every=args_cli.replan_every,
        waypoint_spacing_m=args_cli.waypoint_spacing,
        accept_radius_m=args_cli.accept_radius,
        plan_cell_m=args_cli.plan_cell,
        ladder=ladder,
        kp=args_cli.kp,
        ki=args_cli.ki,
        kd=args_cli.kd,
    )
    planner_repro = {
        "controller": "planner_los_pid",
        "mode": controller.mode,
        "replan_every": args_cli.replan_every,
        "waypoint_spacing_m": args_cli.waypoint_spacing,
        "accept_radius_m": args_cli.accept_radius,
        "plan_cell_m": args_cli.plan_cell,
        "inflation_ladder_m": [round(r, 4) for r in ladder],
        "kp": args_cli.kp,
        "ki": args_cli.ki,
        "kd": args_cli.kd,
        "privileged": True,
    }
    print("[REPRO] " + "  ".join(f"{k}={v}" for k, v in planner_repro.items()),
          flush=True)

    # Does THIS env carry the scenario protocol?  Same guard, same marker and
    # same warning as scripts/eval_v6_frozen.py:148-162, so a planner
    # certificate and a policy certificate at the same --eval-seed can be
    # scenario-paired instead of merely assumed comparable.
    scenario_protocol = scenario_protocol_stamp(base)
    notice = scenario_protocol_notice(task, scenario_protocol)
    if notice:
        print(notice, flush=True)

    records = []
    ep_counter = t.zeros(base.num_envs, dtype=t.long)
    max_steps = (
        args_cli.episodes // base.num_envs + 3
    ) * base.max_episode_length
    step = 0
    while len(records) < args_cli.episodes and step < max_steps:
        with t.inference_mode():
            actions = controller.act()
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
        controller.reset(done)
        ids = t.nonzero(done).flatten()
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
                "controller": "planner_los_pid",
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
          f"controller=planner_los_pid")
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
    print(f"  planner: {controller.plan_calls} legs planned, "
          f"{controller.straight_fallbacks} straight-line fallbacks")

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
                    "controller": "planner_los_pid",
                    "planner": {
                        **planner_repro,
                        "straight_fallbacks": controller.straight_fallbacks,
                        "legs_planned": controller.plan_calls,
                    },
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
    if "--selftest" in sys.argv:
        raise SystemExit(_run_selftest())
    _run_isaac_eval()
