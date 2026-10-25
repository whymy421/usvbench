"""G25 / G25b / G24b: persistent ray-return obstacle memory of the journal planner (planner_los_pid_mem_v2).

G24b (owner decision 2026-10-04, NC 09 "planner memory: no timeout clearing; moving obstacles go by
'passed = deleted' plus a capacity cap"): the G25b expiry rule S2 is REMOVED. An entry leaves the memory
only by S1 (the object that returned it has been passed: see-through rays on both sides, defined below),
by B (capacity) or by R (episode reset). A static surface that no ray sees through therefore stays
remembered for the whole episode. Certificate name planner_los_pid_mem_v2, memory_version g24b_ray_memory_v3
(the G25b planner_los_pid_mem_v1 / g25b_ray_memory_v2 had the expiry).

Owner decision 2026-10-03 (NC 09, G22 options A2 + A1): the journal planner + LOS/PID
(classical_planner_pid.JournalPlannerLOSPIDController, hazard_nav and path_hazard journal ids) may
remember past ray returns. Information still comes ONLY from the 36-ray block of the D5 estimate
(journal_controllers.read_journal_inputs; CONTEXT_R4 D5/D14 as interpreted by the owner): the
memory stores the surface points the rays returned earlier in the SAME episode, in the world frame
of the estimated own position, and forgets them by the rules below. It never reads obstacle state,
D7 objects, the appearing slot or truth.

Why (G22, work/REPORT_controller_copy.md): the planner's map of discs of radius range * tan(pi/36)
around the CURRENT returns (journal_controllers.ray_obstacles) has holes where the rays meet a
wall obliquely (hit points far apart along the wall); the planner routes through a hole, the hole
closes at a later replan, the waypoint flips and the hull pivots into the wall. Returns remembered
from earlier poses close those holes (G22 replica: fortress WAM-V 0/32 -> 32/32).

Rules (every constant is in RULE_CONSTANTS with its reason; lengths are fractions of the planner's
raster cell, 0.25 m x the E5 length scale s):

  M1 store   every return (range below the maximum) of every control step is stored as a surface
             point in world frame, binned on a square grid of quantum_m; one entry per bin (the latest
             point and the control step it was last returned), so a face seen many times costs one entry.
  M2 map     (G25b, REV-G25 MAJOR-1) the planner's obstacle set at a replan = one disc of radius
             disc_radius_m per CURRENT return and per remembered entry. A return is an exact surface
             point (a line cast), so current and remembered returns are the same kind of evidence and get
             the same disc. The v2.5 map gave current returns radius range * tan(pi/36) to bridge the gap
             between neighbouring rays of ONE scan; with the memory the gaps are filled by returns of
             earlier scans, and the fat discs (up to 11.7 m at the WAM-V range) sealed admitted gates
             (REV-G25: forced crossing WAM-V plannable inflation 0.61-1.03 m on the union against 4.45-
             5.39 m on either map alone; probes/diag_fallback_fc_wamv.json).
  S1 carve   (stale-return rule, moving objects) a current ray SEES THROUGH an entry when the entry lies
             ahead of the origin, within carve_lateral_m of the ray's centre line, and the ray's range
             (the maximum for "no return") exceeds the entry's distance along the ray by more than
             carve_margin_m. An entry is deleted once see-through rays have passed on OPPOSITE sides of
             it (side = the direction from the ray line to the entry; reference = the side of the first
             see-through ray since the entry was last returned) or a ray passes exactly through it.
             Water on both sides of a remembered point, along lines that continue beyond it, means the
             object that returned it has moved (D7 intruder, moving-basin mover): its trail is erased as
             the rays sweep across it.
             Static obstacles, NOISE-FREE pose and ranges only: a static disc is never carved, because
             every line that misses a disc of radius R within l of one of its surface points P lies on
             the outer side of P, with its side direction within acos(1 - l / R) of the inward normal at
             P, so two such sides make an angle below 90 degrees whenever l < (1 - cos 45 deg) R =
             0.293 R. With l = 0.5 cell = 0.125 s this holds for every static disc of radius >= 0.43 s;
             the smallest static radius of the journal hazard_nav / path_hazard layouts is 0.8 s, the
             appearing cylinder is 0.8 s (0.822 s smallest in the scatter appearing pool, REV-G25).
             With an erroneous own-position estimate or noisy ranges (D5 perception suites) the returns
             of one surface are placed inconsistently and static-surface entries ARE carved (REV-G25:
             from 0.02 m position noise or 0.01 m range noise on); S1 then acts as a consistency filter
             (G25: fortress at 0.4 m noise 2/8 and 1/8 with S1, 0/8 and 0/8 without).
             D7 intruders (radius 0.14 s WAM-V, 0.23 s BlueBoat) are BELOW the 0.43 s bound: entries of
             an intruder that is still present may be carved; that is intended (it moves, and the
             current scan still returns it).
             A one-sided rule (any see-through ray) is NOT safe: rays grazing a cylinder wall through its
             sub-beam gaps erase the wall faces (G25 replica, fortress WAM-V 0/4 with it, 4/4 without).
  S2 expire  REMOVED in G24b (owner decision: no timeout expiry). "Passed" (the owner's "moving obstacles
             are dropped when passed") is rule S1 exactly: a remembered return is dropped once the current
             rays show water beyond it on BOTH sides (or a ray runs exactly through it), i.e. the object
             that returned it is no longer at that point. Nothing is dropped for age, distance or range.
             The memory has no object identity or velocity (REV-G25b M5): "passed" is a property of a
             remembered surface POINT seen through, not of an object track. A rule "the boat has passed the
             object" (entry behind the boat) is deliberately absent: it would also drop static walls behind
             the boat, which fortress routes revisit (path / nominal up to 3.6, G25B_TABLES).
             G25b's S2 (ttl = ray range / u_ref) is gone with its constant ttl_sensor_crossings and the
             derived ttl_steps; the "expired" telemetry counter is gone too.
  B  bound   at most capacity entries per env; beyond it the least recently returned are evicted
             (ties: the lower bin key), so memory and planning cost are bounded. With S2 gone this is the
             only size bound; it acts only when an env holds more than capacity_entries entries after the
             step's carve and insert (G24b runs: work/evidence of 20261006_g24b_classical_fixes report the
             peaks and every eviction).
  R  reset   an env's memory and step counter are cleared on its reset (per env, partial resets).
  E  escape  (G24b, REV-G25b CRITICAL-1; the controller's rule, constant here so the certificate pins it) a holding
             env whose replans stay truncated replans on the CURRENT scan only after hold_escape_failed_replans
             consecutive failures (classical_planner_pid.JournalMemoryPlannerController); the memory is untouched.

Ladder (A1, journal_memory_ladder): rungs (C + 0.2 s, C, B / 2 + 0.05 s), C = registry circumscribed
radius about the COM; ladder_terms / memory_ladder below are the single, torch-free source (the
certificate validator recomputes them from the registry).

Deterministic: integer control-step counters per env, no wall clock, no random draw, a total order
on every tie. Pure numpy; per env arrays sorted by bin key.
"""
from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
import sys
import types

import numpy as np

MEMORY_FORMAT = "usvbench-planner-memory-v1"
MEMORY_VERSION = "g24b_ray_memory_v3"
# Certificate names (journal hazard_nav / path_hazard cells; run_cell.JOURNAL_MEMORY_PLANNER declares the same).
CONTROLLER_NAME = "planner_los_pid_mem_v2"
BASE_CONTROLLER_NAME = "planner_los_pid"
CARRIERS = ("hazard_nav", "path_hazard")          # journal carriers that run the memory planner
# Planner settings the certificate pins (classical_planner_pid.py CLI defaults, journal ids scale lengths by s).
PLAN_CELL_REF_M = 0.25
REPLAN_EVERY_STEPS = 60
RAY_RANGE_REF_M = 30.0                           # journal 36-ray fan range, x s (journal_cfg_v2 plans)
SNAP_START_REF_M = 1.5                           # G24b (REV-G25b m5): plan_with_ladder snap start, x s

RULE_CONSTANTS = {
    # M1/M2: bin size and disc radius = half a planner raster cell (G22's prototype, work/tools/
    # replay_classical.py map_variant 'memory'): a return is an exact surface point, so the disc only has
    # to cover the raster (a disc of half a cell marks the cell it lies in); the planner's inflation
    # ladder then adds the hull radius. Bigger discs would narrow admitted gates. G25b: current returns get
    # the same disc (rule M2).
    "quantum_cells": 0.5,
    "disc_radius_cells": 0.5,
    # S1: a see-through ray crosses the remembered disc (centre line within its radius, half a cell) and
    # continues more than one cell beyond the point (it leaves the remembered disc with a disc radius of
    # water to spare). Half a cell keeps the static-safety bound l < 0.293 R for R >= 0.43 s (docstring).
    "carve_lateral_cells": 0.5,
    "carve_margin_cells": 1.0,
    # (S2 removed in G24b: no expiry horizon, owner decision 2026-10-04.)
    # E (G24b, REV-G25b CRITICAL-1; read by classical_planner_pid.JournalMemoryPlannerController): hold escape after
    # this many consecutive truncated replans of a holding env (the first held replan can still see the braking
    # transient at the truncated plan's end; the second confirms the blockage persists with the hull stopped).
    "hold_escape_failed_replans": 2,
    # B: per-env capacity (entries). Returns lie on obstacle SURFACES; observed peaks 1955 calm (REV-G25,
    # moving basin WAM-V) and 4533 under perception noise (REV-G25, ray-noise D4 fortress BlueBoat), no
    # eviction in any run; 8192 bounds the per-replan disc count (8192 x 48 B per env). G24b (no expiry, REPLICA): peaks
    # 4045 calm (fortress WAM-V), 7487 / 7846 under perception noise (ray noise 0.30 fortress WAM-V / BlueBoat = 96 % of
    # the cap), no eviction in any of the 50 G24b replica jobs (20261006_g24b_classical_fixes evidence/planner_compare.*);
    # an eviction drops the least recently returned entry (a staleness order, reported per run as "evicted").
    "capacity_entries": 8192,
}
# A1 ladder margins in units of the E5 length scale s (journal_ladder's margins, unchanged).
LADDER_MARGIN_LENGTHS = 0.2
LADDER_NARROW_LENGTHS = 0.05
LADDER_RULE = "(C + 0.2 s, C, B / 2 + 0.05 s), fattest first, duplicates dropped"


def shared_module(name):
    """tasks/_shared/<name> without torch: the package, the deployed tree, or by path."""
    for package in ("tasks._shared", "isaaclab_tasks.direct._shared"):
        try:
            return importlib.import_module(package + "." + name)
        except ImportError:
            continue
    private = "_g25b_planner_memory_shared"
    if private not in sys.modules:
        module = types.ModuleType(private)
        module.__path__ = [str(Path(__file__).resolve().parents[1] / "tasks" / "_shared")]
        sys.modules[private] = module
    return importlib.import_module(private + "." + name)


def vessel_terms(vessel):
    """Registry numbers the memory planner uses: length scale s, reference (cruise) speed, ray range."""
    record = shared_module("journal_cfg_v2").vessel_record(vessel)
    scale = float(record.length_scale)
    return {"length_scale": scale, "reference_speed_mps": float(record.cruise_speed_m_s),
            "ray_max_range_m": RAY_RANGE_REF_M * scale,
            "control_dt_s": float(shared_module("journal_cfg_v2").CONTROL_DT_S)}


def ladder_terms(vessel):
    """Registry numbers and rule of the A1 ladder (stamped in the certificate, recomputed by the validator)."""
    record = shared_module("journal_cfg_v2").vessel_record(vessel)
    spec = shared_module("vehicles_v2").get_vehicle_v2(vessel)
    return {"rule": LADDER_RULE, "circumscribed_radius_m": float(record.circumscribed_radius_m),
            "half_beam_m": 0.5 * float(spec.beam_overall_m), "length_scale": float(record.length_scale),
            "margin_lengths": LADDER_MARGIN_LENGTHS, "narrow_margin_lengths": LADDER_NARROW_LENGTHS}


def memory_ladder(terms):
    """The A1 ladder of ladder_terms(vessel): rungs fattest first, duplicates dropped."""
    hull, scale = terms["circumscribed_radius_m"], terms["length_scale"]
    ladder = []
    for rung in (hull + terms["margin_lengths"] * scale, hull,
                 terms["half_beam_m"] + terms["narrow_margin_lengths"] * scale):
        if rung > 0.0 and all(abs(rung - kept) > 1.0e-9 for kept in ladder):
            ladder.append(rung)
    return ladder


def derive_constants(rules, plan_cell_m, replan_every, length_scale, ray_max_range_m, reference_speed_mps,
                     control_dt_s):
    """Constants of one controller (metres / control steps) from the rule constants."""
    cell = float(plan_cell_m)
    if not cell > 0.0:
        raise ValueError("plan_cell_m must be positive")
    if int(replan_every) < 1:
        raise ValueError("replan_every must be >= 1")
    for name, value in (("length_scale", length_scale), ("ray_max_range_m", ray_max_range_m),
                        ("reference_speed_mps", reference_speed_mps), ("control_dt_s", control_dt_s)):
        if not float(value) > 0.0:
            raise ValueError(name + " must be positive")
    out = {"plan_cell_m": cell, "replan_every_steps": int(replan_every), "length_scale": float(length_scale),
           "ray_max_range_m": float(ray_max_range_m), "reference_speed_mps": float(reference_speed_mps),
           "control_dt_s": float(control_dt_s),
           "quantum_m": float(rules["quantum_cells"]) * cell,
           "disc_radius_m": float(rules["disc_radius_cells"]) * cell,
           "carve_lateral_m": float(rules["carve_lateral_cells"]) * cell,
           "carve_margin_m": float(rules["carve_margin_cells"]) * cell,
           "capacity_entries": int(rules["capacity_entries"])}
    if not (out["quantum_m"] > 0.0 and out["disc_radius_m"] > 0.0 and out["carve_lateral_m"] >= 0.0
            and out["carve_margin_m"] >= 0.0 and out["capacity_entries"] >= 1):
        raise ValueError("planner memory constants must be positive")
    return out


def module_sha256(path=None):
    """LF-normalised SHA-256 of this file or of ``path`` (provenance stamp, journal_controllers._lf_sha256 style)."""
    return hashlib.sha256(Path(path or __file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


_KEY_OFFSET = 1 << 30


def _bin_keys(points, quantum):
    """Nonnegative int64 bin key of each point: (ix + 2^30) * 2^31 + (iy + 2^30) < 2^62, ix = round(x / q).

    |ix|, |iy| < 2^30 bins (1.3e5 km at the smallest bin, 0.125 m) keeps every key below 2^62: no overflow,
    and key order = (ix, iy) lexicographic order.
    """
    idx = np.rint(points / quantum).astype(np.int64)
    if idx.size and (np.abs(idx).max() >= _KEY_OFFSET):
        raise ValueError("planner memory point outside the bin key range")
    return (idx[:, 0] + _KEY_OFFSET) * (1 << 31) + (idx[:, 1] + _KEY_OFFSET)


class RayReturnMemory:
    """Per-env remembered ray returns (rules M1, M2, S1, B, R of the module docstring; no S2 since G24b)."""

    COUNTERS = ("inserted", "refreshed", "carved", "evicted", "resets", "updates")

    def __init__(self, num_envs, *, plan_cell_m, replan_every, length_scale, ray_max_range_m, reference_speed_mps,
                 control_dt_s, rules=None):
        self.rules = dict(RULE_CONSTANTS if rules is None else rules)
        self.constants = derive_constants(self.rules, plan_cell_m, replan_every, length_scale, ray_max_range_m,
                                          reference_speed_mps, control_dt_s)
        c = self.constants
        self.quantum = c["quantum_m"]
        self.disc_radius = c["disc_radius_m"]
        self.carve_lateral = c["carve_lateral_m"]
        self.carve_margin = c["carve_margin_m"]
        self.capacity = c["capacity_entries"]
        self.num_envs = int(num_envs)
        self.step = np.zeros(self.num_envs, dtype=np.int64)
        self._keys = [np.zeros(0, dtype=np.int64) for _ in range(self.num_envs)]
        self._points = [np.zeros((0, 2), dtype=np.float64) for _ in range(self.num_envs)]
        self._seen = [np.zeros(0, dtype=np.int64) for _ in range(self.num_envs)]
        self._side = [np.zeros((0, 2), dtype=np.float64) for _ in range(self.num_envs)]
        self.counts = {name: 0 for name in self.COUNTERS}
        self.peak_entries = 0

    # ------------------------------------------------------------------ R --
    def reset(self, env_ids):
        for i in env_ids:
            i = int(i)
            self.step[i] = 0
            self._keys[i] = np.zeros(0, dtype=np.int64)
            self._points[i] = np.zeros((0, 2), dtype=np.float64)
            self._seen[i] = np.zeros(0, dtype=np.int64)
            self._side[i] = np.zeros((0, 2), dtype=np.float64)
            self.counts["resets"] += 1

    def size(self, env_index):
        return int(self._keys[int(env_index)].size)

    # --------------------------------------------------------- M1, S1, B --
    def update(self, env_index, origin_xy, yaw, ranges, max_range_m):
        """One control step of one env: carve (S1), store / refresh the current returns (M1), bound (B); no expiry.

        origin_xy: estimated COM xy (world); yaw: heading; ranges: [R] ray ranges (m), ray k at
        yaw + k 2 pi / R (journal_controllers.ray_obstacles); a range >= max (1 - 1e-6) is no return.
        """
        i = int(env_index)
        self.step[i] += 1
        now = int(self.step[i])
        self.counts["updates"] += 1
        origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
        ranges = np.clip(np.asarray(ranges, dtype=np.float64).reshape(-1), 0.0, float(max_range_m))
        count = ranges.size
        angles = float(yaw) + np.arange(count, dtype=np.float64) * (2.0 * math.pi / count)
        directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        hit = ranges < float(max_range_m) * (1.0 - 1.0e-6)

        keys, points, seen, side = self._keys[i], self._points[i], self._seen[i], self._side[i]
        if keys.size:
            carved, side = self._carve(points, side, origin, directions, ranges, yaw)
            self.counts["carved"] += int(carved.sum())
            if carved.any():
                keys, points, seen, side = keys[~carved], points[~carved], seen[~carved], side[~carved]

        if hit.any():
            new_points = origin + ranges[hit, None] * directions[hit]
            new_keys = _bin_keys(new_points, self.quantum)
            # one point per bin per scan: the lowest ray index (np.unique returns first occurrences)
            new_keys, first = np.unique(new_keys, return_index=True)
            new_points = new_points[first]
            pos = np.searchsorted(keys, new_keys)
            found = np.zeros(new_keys.size, dtype=bool)
            inside = pos < keys.size
            found[inside] = keys[pos[inside]] == new_keys[inside]
            if found.any():
                rows = pos[found]
                points, seen, side = points.copy(), seen.copy(), side.copy()
                points[rows] = new_points[found]
                seen[rows] = now
                side[rows] = 0.0                       # a return in the bin: the object is there, evidence reset
                self.counts["refreshed"] += int(found.sum())
            fresh = ~found
            if fresh.any():
                insert_at = pos[fresh]
                n_fresh = int(fresh.sum())
                keys = np.insert(keys, insert_at, new_keys[fresh])
                points = np.insert(points, insert_at, new_points[fresh], axis=0)
                seen = np.insert(seen, insert_at, np.full(n_fresh, now, dtype=np.int64))
                side = np.insert(side, insert_at, np.zeros((n_fresh, 2)), axis=0)
                self.counts["inserted"] += n_fresh
        if keys.size > self.capacity:
            # least recently returned first, ties by the lower key; keep the rest in key order
            order = np.lexsort((keys, seen))
            drop = order[: keys.size - self.capacity]
            keep = np.ones(keys.size, dtype=bool)
            keep[drop] = False
            self.counts["evicted"] += int(drop.size)
            keys, points, seen, side = keys[keep], points[keep], seen[keep], side[keep]
        self._keys[i], self._points[i], self._seen[i], self._side[i] = keys, points, seen, side
        self.peak_entries = max(self.peak_entries, int(keys.size))

    # S1 evaluates, for each remembered point, only the rays that can pass within carve_lateral_m of it: a
    # point at distance d is within l of a ray line only if the ray's bearing differs from the point's by at
    # most asin(l / d); for d >= l / sin(15 deg) that is at most 15 deg, i.e. within CANDIDATE_OFFSETS of the
    # nearest ray (10 deg spacing). Nearer points test every ray. Same decisions as testing every ray.
    CANDIDATE_OFFSETS = (-2, -1, 0, 1, 2)
    CANDIDATE_ANGLE = math.radians(15.0)

    def _carve(self, points, side, origin, directions, ranges, yaw):
        """S1: (carved [M] bool, updated side [M, 2]) of the remembered points against the current rays."""
        rel = points - origin
        count = ranges.size
        carved = np.zeros(points.shape[0], dtype=bool)
        side = side.copy()
        dist = np.hypot(rel[:, 0], rel[:, 1])
        far = dist * math.sin(self.CANDIDATE_ANGLE) >= self.carve_lateral
        if count < 2 * len(self.CANDIDATE_OFFSETS):
            far[:] = False
        for group in (far, ~far):
            rows = np.nonzero(group)[0]
            if rows.size == 0:
                continue
            if group is far:
                bearing = np.arctan2(rel[rows, 1], rel[rows, 0]) - float(yaw)
                nearest = np.rint(bearing / (2.0 * math.pi / count)).astype(np.int64)
                rays = np.sort((nearest[:, None] + np.asarray(self.CANDIDATE_OFFSETS)[None, :]) % count, axis=1)
            else:
                rays = np.broadcast_to(np.arange(count), (rows.size, count))
            c, sd = self._carve_rows(rel[rows], side[rows], directions[rays], ranges[rays])
            carved[rows] = c
            side[rows] = sd
        return carved, side

    def _carve_rows(self, rel, side, dirs, rng):
        """S1 on candidate rays: rel [M, 2], side [M, 2], dirs [M, K, 2], rng [M, K] (rays in index order)."""
        along = rel[:, :1] * dirs[:, :, 0] + rel[:, 1:] * dirs[:, :, 1]               # [M, K]
        wx = rel[:, :1] - along * dirs[:, :, 0]                                       # point - closest line point
        wy = rel[:, 1:] - along * dirs[:, :, 1]
        lateral = np.hypot(wx, wy)
        through = (along > 0.0) & (lateral <= self.carve_lateral) & (rng > along + self.carve_margin)
        any_through = through.any(axis=1)
        if not any_through.any():
            return any_through, side
        exact = (through & (lateral <= 1.0e-9)).any(axis=1)
        safe = np.where(lateral > 1.0e-9, lateral, 1.0)
        ux, uy = wx / safe, wy / safe
        has_ref = (side != 0.0).any(axis=1)
        first = np.argmax(through, axis=1)
        rows = np.arange(rel.shape[0])
        ref_x = np.where(has_ref, side[:, 0], ux[rows, first])
        ref_y = np.where(has_ref, side[:, 1], uy[rows, first])
        opposite = (through & ((ux * ref_x[:, None] + uy * ref_y[:, None]) < 0.0)).any(axis=1)
        carved = exact | opposite
        set_ref = any_through & ~carved & ~has_ref
        if set_ref.any():
            side = side.copy()
            side[set_ref, 0] = ref_x[set_ref]
            side[set_ref, 1] = ref_y[set_ref]
        return carved, side

    # ------------------------------------------------------------------ M2 --
    def discs(self, env_index):
        """(centers [M, 2], radii [M]) of the remembered returns of one env (world frame)."""
        points = self._points[int(env_index)]
        return points.copy(), np.full(points.shape[0], self.disc_radius, dtype=np.float64)

    def telemetry(self):
        return {**{name: int(value) for name, value in self.counts.items()}, "peak_entries": int(self.peak_entries)}

    def stamp(self):
        """Certificate block content (constants; the evaluator adds names, ladder and telemetry)."""
        return {"format": MEMORY_FORMAT, "memory_version": MEMORY_VERSION,
                "rule_constants": dict(self.rules), "constants": dict(self.constants),
                "module_sha256": module_sha256()}


__all__ = ["MEMORY_FORMAT", "MEMORY_VERSION", "CONTROLLER_NAME", "BASE_CONTROLLER_NAME", "CARRIERS", "RULE_CONSTANTS",
           "PLAN_CELL_REF_M", "REPLAN_EVERY_STEPS", "RAY_RANGE_REF_M", "SNAP_START_REF_M", "LADDER_MARGIN_LENGTHS", "LADDER_NARROW_LENGTHS",
           "LADDER_RULE", "RayReturnMemory", "derive_constants", "module_sha256", "shared_module",
           "vessel_terms", "ladder_terms", "memory_ladder"]
