"""G39: two generic fixes of the journal MPPI stalls (release candidates on top of v2.6.1).

Diagnosis (G39 FINAL_REPORT): the journal MPPI cost charges the STRAIGHT-LINE distance to the target
(mppi_controller.MPPIController.rollout, route potential with G = 1) and sees obstacles only through a
memoryless ray map with a clearance margin. Two stationary points follow, both on the CPU replica and in
the v2.6 / v2.6.1 certificates:

  S1 "yaw to shorten" (BlueBoat docking). The dock point is closer to the back wall than the bow reach +
     clearance margin allows, so the position term keeps pressing forward against the margin barrier;
     turning the hull shortens its projection on the berth axis, and the per-rollout position gain of that
     ((H w_goal + w_terminal) x capsule half segment = 26.4 per unit (1 - cos psi)) exceeds the ungated
     heading cost (H w_dock_heading = 9.0). The hull settles at the yaw where its side reach meets the
     side-wall margin (closed form 0.50 rad, certificates 0.48 rad median), above the 15 degree tolerance.
  S2 "wall across the line" (forced crossing both vessels, WAM-V docking beside the slip, the scatter
     appearing cylinder that merges with a field cylinder into one barrier). Every rollout of the 1 s
     (BlueBoat) / 4.5 s (WAM-V) horizon that goes round the barrier ends farther from the target in
     straight-line distance than staying put, so the controller parks at the clearance margin until the
     deadline.

Fix F1 (S1): DOCK POSITION DEAD ZONE. On the berth leg (dock mode, mission berth leg) the position potential
     is relu(potential - f x tol) with tol = the success position tolerance (the radius the success
     predicate tests; eval_mppi passes it as goal_radius) and f = DOCK_DEADZONE_FRACTION = 0.5: position
     error the task does not score is not charged, keeping half the tolerance as a buffer. The heading term
     stays ungated (the module docstring's rim argument is about gating the HEADING term; this gates
     nothing, it only flattens the position term inside a disc that lies wholly inside the scored disc),
     the speed term is unchanged. Stationary-point check (closed form, no seed): the aligned stall points
     (BlueBoat x = -0.79 m, WAM-V x = -3.10 m from the dock point) lie inside the dead zones (1.25 m,
     5.60 m), so no position pressure reaches the back-wall margin and the yaw incentive disappears.

Fix F2 (S2): GEODESIC POTENTIAL. In goal and dock modes (G = 1) the straight-line distance in the stage and
     terminal potential is replaced by a geodesic cost-to-go on a local raster: the obstacles are the SAME
     ray evidence the journal memory planner uses (scripts/planner_memory.RayReturnMemory, rules M1/M2/S1/B/R
     unchanged: current + remembered returns, one disc of half a raster cell each), rasterised with
     classical_planner_pid.occupancy_grid_windowed at the variant's rungs (c2: the memory planner ladder;
     c3: collision capsule radius, B / 2 + 0.05 s) - the fattest that connects boat and target - and the
     cost-to-go is an 8-neighbour chamfer wavefront from the target (unit / sqrt 2 steps, metres).
     Information diet unchanged (the D5 estimate's position, rays, heading; no map, no truth). The obstacle
     COST of the rollout is unchanged (current-ray discs, margin, collision); only the potential changes.
     Cells inside the inflated map get a continuation of the field (second wavefront), so rollouts that
     graze the inflation are still guided; when no rung connects boat and target the env keeps the
     straight-line potential (the v2.6.1 behaviour) and the refresh is counted as a fallback.
     The field is rebuilt every planner_memory.REPLAN_EVERY_STEPS control steps (1 s, the planner's
     replan period), when the target moves by more than a cell, and after a reset; it reaches the
     rollout through MPPIController.set_potential (in-place copies into buffers the CUDA graph reads by
     address; GRAPH_TENSOR_INPUTS unchanged).

Variants (release candidates): "c1" = F1; "c2" = F1 + F2 with the raster inflated by the memory planner ladder
(C + 0.2 s, C, B / 2 + 0.05 s); "c3" = F1 + F2 with the raster inflated by (collision capsule radius,
B / 2 + 0.05 s); "base" = v2.6.1. F1 is inert outside the berth leg, so c2 / c3 differ from c1 in goal tasks only by
F2. The two raster rules are kept as separate candidates because neither dominated on the CPU replica. Nothing here was fitted
to an evaluation seed: f = 0.5 and the raster constants are declared rules; the CPU replica used only the
development keys (eval seed 42) of the v2.6 smoke and the v2.6.1 sweep.
"""
from __future__ import annotations

import math

import numpy as np

FIX_FORMAT = "usvbench-mppi-g39-fix-v1"
VARIANTS = ("base", "c1", "c2", "c3")
# The switch this release candidate ships (journal_mppi_params applies it to every journal MPPI run).
RELEASE_VARIANT = "c3"
DOCK_DEADZONE_FRACTION = 0.5
GRID_CELLS = 160                     # raster side (cells); BlueBoat 0.25 m -> 40 m, WAM-V 1.12 m -> 179 m
WAVEFRONT_CHECK_EVERY = 8            # convergence test period of the wavefront (iterations)
# Step-cost multiplier inside the inflated map for the continuation wavefront: a path THROUGH an obstacle
# must never undercut the way round it (G39 replica, scatter: with multiplier 1 the near face of a cylinder
# took its value from the far side and the field pointed into the cylinder; 17 parked keys). With 10 the
# continuation inside the inflation is ~ (nearest free value + 10 x depth): it slopes out of the obstacle.
CONTINUATION_STEP_FACTOR = 10.0
RULE_TEXT = {
    "dock_deadzone": "berth-leg potential relu(P - f * tol), f = %.2f, tol = success position tolerance" % DOCK_DEADZONE_FRACTION,
    "geodesic": ("goal/dock potential = 8-neighbour chamfer cost-to-go from the target on a %d x %d raster of "
                 "planner_memory cells (0.25 m x s) centred between boat and target; obstacles = current + "
                 "remembered ray returns (planner_memory M1/M2/S1/B/R), inflation = the variant field rule "
                 "(fattest connecting rung); continuation inside the inflation at 10x step cost; straight line when no rung "
                 "connects; refresh every planner_memory.REPLAN_EVERY_STEPS steps, on target move > 1 cell, "
                 "after reset" % (GRID_CELLS, GRID_CELLS)),
}


def variant_flags(variant):
    if variant not in VARIANTS:
        raise ValueError("unknown G39 variant %r (one of %s)" % (variant, ", ".join(VARIANTS)))
    # c1 = F1; c2 = F1 + F2 with the planner-ladder raster; c3 = F1 + F2 with the capsule raster. F1 acts on the
    # berth leg only (bit-identical goal-mode cost), so pairing F2 with F1 loses no information in goal tasks.
    return {"dock_deadzone": variant in ("c1", "c2", "c3"), "geodesic": variant in ("c2", "c3"),
            "field_rule": {"c2": "planner_ladder", "c3": "capsule"}.get(variant)}


FIELD_RULES = ("planner_ladder", "capsule")
FIELD_RULE_TEXT = {
    "planner_ladder": "planner_memory.memory_ladder (C + 0.2 s, C, B / 2 + 0.05 s), fattest connecting rung",
    "capsule": "(collision capsule radius, B / 2 + 0.05 s), fattest connecting rung",
}


def variant_uses_geodesic(variant):
    return variant_flags(variant)["geodesic"]


def params_for_variant(params, variant):
    """MPPIParams with the variant's two switches set (everything else unchanged)."""
    from dataclasses import replace
    flags = variant_flags(variant)
    out = replace(params, dock_deadzone_fraction=DOCK_DEADZONE_FRACTION if flags["dock_deadzone"] else 0.0,
                  geodesic_potential=bool(flags["geodesic"]))
    out.validate()
    return out


# ------------------------------------------------------------------ wavefront --
_NEIGHBOURS = ((0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
               (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)))


def wavefront(passable, seeds, *, max_iterations=None, step_factor=None):
    """Chamfer cost-to-go (cells) of [B, n, n] grids: passable bool, seeds float (inf except seed values).

    Value iteration d <- min(d, d[neighbour] + w * f) on passable cells (seed cells keep their seed if lower);
    f = step_factor (a [B, n, n] multiplier of the cell being entered, default 1).
    stops when an iteration changes nothing (tested every WAVEFRONT_CHECK_EVERY iterations).
    """
    import torch
    import torch.nn.functional as F
    d = seeds.clone()
    b, n, m = d.shape
    inf = torch.tensor(float("inf"), dtype=d.dtype, device=d.device)
    limit = int(max_iterations if max_iterations is not None else 4 * (n + m))
    for it in range(limit):
        pad = F.pad(d, (1, 1, 1, 1), value=float("inf"))
        best = d
        for dy, dx, w in _NEIGHBOURS:
            step = w if step_factor is None else w * step_factor
            best = torch.minimum(best, pad[:, 1 + dy:1 + dy + n, 1 + dx:1 + dx + m] + step)
        best = torch.where(passable, best, torch.where(torch.isfinite(seeds), seeds, inf))
        best = torch.minimum(best, seeds)
        if (it + 1) % WAVEFRONT_CHECK_EVERY == 0 and bool(torch.equal(best, d)):
            d = best
            break
        d = best
    return d


def _nearest_free(occupied, index, max_ring):
    """Closest free cell to index (row, col) within a square ring search (classical_planner_pid rule)."""
    if not occupied[index]:
        return index
    height, width = occupied.shape
    best, best_d2 = None, math.inf
    for ring in range(1, max_ring + 1):
        for dr in range(-ring, ring + 1):
            for dc in range(-ring, ring + 1):
                if max(abs(dr), abs(dc)) != ring:
                    continue
                r, c = index[0] + dr, index[1] + dc
                if 0 <= r < height and 0 <= c < width and not occupied[r, c]:
                    d2 = dr * dr + dc * dc
                    if d2 < best_d2:
                        best, best_d2 = (r, c), d2
        if best is not None:
            return best
    return None


class GeodesicGuide:
    """Per-env ray memory + geodesic cost-to-go raster for MPPIController.set_potential (fix F2)."""

    COUNTERS = ("refreshes", "fallbacks", "snapped_targets", "snapped_starts")

    def __init__(self, num_envs, *, vessel, ray_max_range_m, device="cpu", grid_cells=GRID_CELLS,
                 field_rule="capsule"):
        import planner_memory as pm
        terms = pm.vessel_terms(vessel)
        self.num_envs = int(num_envs)
        self.vessel = str(vessel)
        self.scale = float(terms["length_scale"])
        self.cell = pm.PLAN_CELL_REF_M * self.scale
        self.refresh_every = int(pm.REPLAN_EVERY_STEPS)
        self.ray_max_range_m = float(ray_max_range_m)
        self.memory = pm.RayReturnMemory(self.num_envs, plan_cell_m=self.cell, replan_every=self.refresh_every,
                                         length_scale=self.scale, ray_max_range_m=self.ray_max_range_m,
                                         reference_speed_mps=terms["reference_speed_mps"],
                                         control_dt_s=terms["control_dt_s"])
        # Field inflation (G39 iteration 3): the vessel's collision CAPSULE radius - the footprint MPPI itself
        # scores (VesselDynamicsV2.hull_clearance) - then the planner ladder's narrow rung (B / 2 + 0.05 s).
        # The field must not be narrower than what MPPI may legally traverse: with the memory planner's fattest
        # rung (circumscribed radius + 0.2 s) the BlueBoat slip left a 0.3 m wide free band and one replica
        # key parked at the mouth. Clearance itself stays the job of MPPI's margin / collision terms.
        # G39 keeps both rules as separate release candidates (c2 planner ladder, c3 capsule): on the CPU replica
        # the planner ladder was better in scatter, the capsule better in the slips (FINAL_REPORT section 4).
        if field_rule not in FIELD_RULES:
            raise ValueError("unknown field rule %r" % (field_rule,))
        self.field_rule = field_rule
        if field_rule == "planner_ladder":
            self.ladder = tuple(pm.memory_ladder(pm.ladder_terms(vessel)))
        else:
            from mppi_controller import vessel_params_v2
            terms_l = pm.ladder_terms(vessel)
            rungs = (float(vessel_params_v2(vessel)["capsule_radius_m"]),
                     terms_l["half_beam_m"] + terms_l["narrow_margin_lengths"] * terms_l["length_scale"])
            self.ladder = tuple(sorted({round(r, 12) for r in rungs if r > 0.0}, reverse=True))
        self.snap_m = pm.SNAP_START_REF_M * self.scale
        self.n = int(grid_cells)
        self.device = device
        self.field = np.zeros((self.num_envs, self.n, self.n), dtype=np.float32)
        self.origin = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.valid = np.zeros(self.num_envs, dtype=bool)
        self.age = np.full(self.num_envs, 1 << 30, dtype=np.int64)
        self.target = np.full((self.num_envs, 2), np.nan)
        self.counts = {k: 0 for k in self.COUNTERS}
        self.rung_hist = [0] * len(self.ladder)
        self._dirty = True

    # ------------------------------------------------------------ episode --
    def reset(self, done):
        ids = [int(i) for i in np.nonzero(np.asarray(done, dtype=bool).reshape(-1))[0]]
        if ids:
            self.memory.reset(ids)
            self.valid[ids] = False
            self.age[ids] = 1 << 30
            self.target[ids] = np.nan
            self._dirty = True

    # ------------------------------------------------------------- update --
    def observe(self, position_world, yaw, rays, target_world):
        """One control step: memory update for every env, raster refresh for the envs that are due."""
        pos = _np(position_world).astype(np.float64).reshape(self.num_envs, 2)
        yw = _np(yaw).astype(np.float64).reshape(self.num_envs)
        rg = _np(rays).astype(np.float64).reshape(self.num_envs, -1)
        tg = _np(target_world).astype(np.float64).reshape(self.num_envs, 2)
        for i in range(self.num_envs):
            self.memory.update(i, pos[i], yw[i], rg[i], self.ray_max_range_m)
        self.age += 1
        moved = ~np.isfinite(self.target).all(axis=1) | (np.linalg.norm(np.nan_to_num(self.target) - tg, axis=1) > self.cell)
        due = np.nonzero((self.age >= self.refresh_every) | moved)[0]
        if len(due):
            self._refresh([int(i) for i in due], pos, yw, rg, tg)

    def _obstacles(self, i, pos, yw, rg):
        count = rg.shape[1]
        ang = yw[i] + np.arange(count) * (2.0 * math.pi / count)
        r = np.clip(rg[i], 0.0, self.ray_max_range_m)
        hit = r < self.ray_max_range_m * (1.0 - 1.0e-6)
        current = pos[i][None, :] + (r[:, None] * np.stack((np.cos(ang), np.sin(ang)), -1))
        current = current[hit]
        remembered, remembered_r = self.memory.discs(i)
        centers = np.vstack((current.reshape(-1, 2), np.asarray(remembered, dtype=np.float64).reshape(-1, 2)))
        radii = np.concatenate((np.full(len(current), self.memory.disc_radius), np.asarray(remembered_r, dtype=np.float64)))
        return centers, radii

    def _refresh(self, ids, pos, yw, rg, tg):
        import torch
        from classical_planner_pid import occupancy_grid_windowed
        n, cell = self.n, self.cell
        jobs = []                          # (env, rung index, passable, start idx, target idx)
        geometry = {}
        for i in ids:
            mid = 0.5 * (pos[i] + tg[i])
            ox = math.floor(mid[0] / cell) * cell - 0.5 * (n - 1) * cell
            oy = math.floor(mid[1] / cell) * cell - 0.5 * (n - 1) * cell
            xs = ox + cell * np.arange(n)
            ys = oy + cell * np.arange(n)
            centers, radii = self._obstacles(i, pos, yw, rg)

            def index(p):
                return (int(np.clip(round((p[1] - oy) / cell), 0, n - 1)), int(np.clip(round((p[0] - ox) / cell), 0, n - 1)))
            geometry[i] = (ox, oy)
            ring = int(math.ceil(self.snap_m / cell))
            for k, rung in enumerate(self.ladder):
                occ = occupancy_grid_windowed(centers, radii, xs, ys, rung) if len(centers) else np.zeros((n, n), bool)
                t_idx = _nearest_free(occ, index(tg[i]), ring)
                s_idx = _nearest_free(occ, index(pos[i]), ring)
                if t_idx is None or s_idx is None:
                    continue
                jobs.append((i, k, ~occ, s_idx, t_idx, t_idx != index(tg[i]), s_idx != index(pos[i])))
        chosen = {}
        if jobs:
            passable = torch.as_tensor(np.stack([j[2] for j in jobs]), device=self.device)
            seeds = torch.full(passable.shape, float("inf"), dtype=torch.float32, device=self.device)
            for b, j in enumerate(jobs):
                seeds[b, j[4][0], j[4][1]] = 0.0
            d1 = wavefront(passable, seeds)
            reach = [bool(torch.isfinite(d1[b, j[3][0], j[3][1]])) for b, j in enumerate(jobs)]
            for b, j in enumerate(jobs):
                if reach[b] and j[0] not in chosen:
                    chosen[j[0]] = b
            if chosen:
                sel = sorted(chosen.items())
                # continuation: reached cells keep their phase-1 value (they are not passable for the
                # second wavefront, so nothing can shortcut through the inflation); the cells phase 1
                # did not reach (inflated obstacles, enclosed pockets) are filled from their neighbours.
                seeds2 = torch.stack([d1[b] for _, b in sel])
                unreached = ~torch.isfinite(seeds2)
                d2 = wavefront(unreached, seeds2, step_factor=torch.where(
                    unreached, torch.full_like(seeds2, CONTINUATION_STEP_FACTOR), torch.ones_like(seeds2)),
                    max_iterations=int(4 * n * CONTINUATION_STEP_FACTOR))
                for (i, b), field in zip(sel, d2):
                    self.field[i] = (field * cell).cpu().numpy().astype(np.float32)
                    self.origin[i] = geometry[i]
                    self.valid[i] = True
                    self.rung_hist[jobs[b][1]] += 1
                    self.counts["snapped_targets"] += int(jobs[b][5])
                    self.counts["snapped_starts"] += int(jobs[b][6])
        for i in ids:
            self.counts["refreshes"] += 1
            if i not in chosen:
                self.valid[i] = False
                self.counts["fallbacks"] += 1
            self.age[i] = 0
            self.target[i] = tg[i]
        self._dirty = True

    # -------------------------------------------------------------- output --
    def apply(self, controller):
        """Copy the rasters into the controller's potential buffers (in place; graph-safe)."""
        if not self._dirty:
            return
        import torch
        c = getattr(controller, "controller", controller)       # CudaGraphMPPI wraps the controller
        c.set_potential(torch.as_tensor(self.field.reshape(self.num_envs, -1)),
                        torch.as_tensor(self.origin), torch.full((self.num_envs,), float(self.cell)),
                        torch.as_tensor(self.valid))
        self._dirty = False

    def stamp(self):
        import planner_memory as pm
        return {"format": FIX_FORMAT, "rule": RULE_TEXT["geodesic"], "field_rule": self.field_rule,
                "field_rule_text": FIELD_RULE_TEXT[self.field_rule], "grid_cells": self.n, "cell_m": self.cell,
                "ladder_m": list(self.ladder), "refresh_every_steps": self.refresh_every, "snap_m": self.snap_m,
                "memory_version": pm.MEMORY_VERSION, "memory_constants": dict(self.memory.constants)}

    def telemetry(self):
        return {"counts": dict(self.counts), "rung_histogram": list(self.rung_hist),
                "memory": {"counts": dict(self.memory.counts), "peak_entries": int(self.memory.peak_entries)}}


def _np(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


__all__ = ["FIX_FORMAT", "VARIANTS", "DOCK_DEADZONE_FRACTION", "GRID_CELLS", "RULE_TEXT", "variant_flags",
           "variant_uses_geodesic", "params_for_variant", "wavefront", "GeodesicGuide"]
