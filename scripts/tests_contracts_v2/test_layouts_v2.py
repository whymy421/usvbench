"""Journal layout generators: all 12 tasks x 2 vessels x 4 tiers (CPU only).

Every admitted layout is re-checked here with INDEPENDENT calls (not by
trusting the admission record): counts, pair gaps, endpoint clearance,
compulsory-gate certificates, witness segments, deadlines and D7 hooks.
"""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L = load("d4_test_layouts", ROOT / "tasks/_shared/journal_layouts_v2.py")
J = load("d4_test_contracts_for_layouts", ROOT / "tasks/_shared/journal_contracts_v2.py")
legacy = load("d4_test_legacy_for_layouts", ROOT / "tasks/hazard_nav/hazard_geometry.py")
cfg, geo = L.cfg, L.geo
_CACHE = {}


def layout(task, vessel, tier, seed=0):
    key = (task, vessel, tier, seed)
    if key not in _CACHE:
        _CACHE[key] = L.build_layout(task, tier, vessel, np.random.default_rng(1000 * tier + seed))
    return _CACHE[key]


class AllLayoutsTests(unittest.TestCase):
    def test_every_task_vessel_tier_admits_with_verified_invariants(self):
        summary = []
        for task in cfg.TASKS:
            for vessel in cfg.VESSELS:
                rec = cfg.VESSELS[vessel]
                v = rec.geometry
                for tier in range(4):
                    lay = layout(task, vessel, tier)
                    p = lay.params
                    t = geo.physical_tier(tier, v)
                    self.assertTrue(np.all(np.isfinite(lay.centers)) and np.all(lay.radii > 0))
                    self.assertLessEqual(lay.obstacle_count, cfg.STATIC_CAPACITY[task], (task, vessel, tier))
                    if lay.field_indices:
                        self.assertEqual(len(lay.field_indices), p["requested_count"])
                        fc, fr = lay.centers[list(lay.field_indices)], lay.radii[list(lay.field_indices)]
                        gap = np.linalg.norm(fc[:, None] - fc[None], axis=-1) - fr[:, None] - fr[None]
                        gap[np.diag_indices(len(fr))] = np.inf
                        need = p.get("min_pair_surface_gap_m", p.get("min_edge_gap_m"))
                        self.assertGreaterEqual(gap.min() + 1e-9, need, (task, vessel, tier))
                    if lay.obstacle_count:
                        self.assertGreater(geo.pose_clearance(v.footprint, (*lay.start, lay.start_heading_rad),
                                                              lay.centers, lay.radii), t.planning_margin_m)
                    for gate in lay.gates:
                        if gate.kind == "compulsory":
                            cert = geo.compulsory_gate_certificate(lay.start, lay.goal, lay.centers, lay.radii,
                                                                   gate.jamb_indices, v.inscribed_radius_m)
                            self.assertTrue(cert.valid, (task, vessel, tier, cert.reasons))
                            self.assertTrue(gate.certified)
                    w = lay.witness
                    self.assertTrue(w.feasible)
                    self.assertGreaterEqual(w.min_clearance_m + 1e-9, t.planning_margin_m)
                    vv = np.asarray(w.vertices_m)
                    for a, b in zip(vv, vv[1:]):
                        c = geo.translation_sweep_clearance(v.footprint, a, b, math.atan2(*(b - a)[::-1]),
                                                            lay.centers, lay.radii) if lay.obstacle_count else math.inf
                        self.assertGreater(c, t.planning_margin_m - 1e-9)
                    self.assertLessEqual(lay.nominal_route_length_m, p["route_budget_m"] + 1e-9)
                    steps = cfg.deadline_steps(lay.nominal_route_length_m, rec.cruise_speed_m_s, p["dwell_s"])
                    self.assertEqual(lay.deadline_steps, steps)
                    self.assertAlmostEqual(lay.deadline_s, steps / 60.0, places=12)
                    x0, x1, y0, y1 = lay.bounds_m
                    for q in (lay.start, lay.goal):
                        self.assertTrue(x0 <= q[0] <= x1 and y0 <= q[1] <= y1)
                    hook = lay.d7_hook
                    self.assertEqual(hook["deadline_s"], lay.deadline_s)
                    self.assertEqual(hook["task"], "harbour" if task == "harbour_mission" else task)
                    self.assertEqual(hook["envelope"]["x_max_m"], v.bounds[1])
                    self.assertEqual(hook["envelope"]["reference_speed_mps"], rec.cruise_speed_m_s)
                    self.assertEqual(hook["reserved_dynamic_slots"], cfg.D7_RESERVED_SLOTS[task])
                    self.assertEqual(len(hook["route"]), len(hook["route_mandatory"]))
                    self.assertTrue(all(r[2] > 0 for r in hook["route"]) and hook["goal"][2] > 0)
                    self.assertEqual(hook["initial_pose"][2], lay.start_heading_rad)
                    self.assertEqual(hook["static_obstacles"][:1], tuple((float(c[0]), float(c[1]), float(r))
                                                                         for c, r in zip(lay.centers[:1], lay.radii[:1])))
                    self.assertLessEqual(hook["base_count"], hook["base_max"])
                    self.assertEqual(hook["geometry_id"], lay.geometry_hash)
                    self.assertEqual(len(lay.geometry_hash), 64)
                    summary.append("%s/%s/G%d n=%d L=%.1f T=%.1f" % (task, vessel, tier, lay.obstacle_count,
                                                                     lay.nominal_route_length_m, lay.deadline_s))
        print("admitted %d layouts; e.g. %s" % (len(summary), "; ".join(summary[::12])))

    def test_determinism_and_seed_sensitivity(self):
        for task in ("scatter", "fortress", "harbour_mission", "docking"):
            a = L.build_layout(task, 2, "blueboat_v2", np.random.default_rng(99))
            b = L.build_layout(task, 2, "blueboat_v2", np.random.default_rng(99))
            c = L.build_layout(task, 2, "blueboat_v2", np.random.default_rng(100))
            self.assertEqual(a.geometry_hash, b.geometry_hash)
            self.assertNotEqual(a.geometry_hash, c.geometry_hash)

    def test_rng_type_and_task_checked(self):
        with self.assertRaises(TypeError):
            L.build_layout("scatter", 0, "blueboat_v2", 5)
        with self.assertRaises(ValueError):
            L.build_layout("slalom", 0, "blueboat_v2", np.random.default_rng(0))


class TaskSpecificTests(unittest.TestCase):
    def test_path_following_geometry(self):
        for vessel in cfg.VESSELS:
            for tier in range(4):
                lay = layout("path_following", vessel, tier)
                pts = np.vstack(([lay.start], np.asarray(lay.waypoints)))
                seg = np.diff(pts, axis=0)
                self.assertTrue(np.allclose(np.linalg.norm(seg, axis=1), lay.params["segment_length_m"]))
                h = np.arctan2(seg[:, 1], seg[:, 0])
                turns = np.degrees(np.abs((np.diff(h) + math.pi) % (2 * math.pi) - math.pi))
                self.assertTrue(np.allclose(turns, lay.params["heading_change_deg"]))

    def test_path_hazard_blockers_and_route_band(self):
        for vessel in cfg.VESSELS:
            for tier in range(4):
                lay = layout("path_hazard", vessel, tier)
                ex = lay.extras
                self.assertEqual(len(ex["designated_blocker_segments"]), cfg.BLOCKER_COUNTS[tier])
                self.assertTrue(set(ex["designated_blocker_segments"]) <= set(ex["realized_blocked_segments"]))
                pts = np.vstack(([lay.start], np.asarray(lay.waypoints)))
                for c in lay.centers:
                    d = min(float(geo._distances_to_segment(c[None], a, b)[0]) for a, b in zip(pts, pts[1:]))
                    self.assertLessEqual(d, lay.params["route_band_m"] + 1e-9)

    def test_scatter_direct_line_blocked_and_iceberg_gap(self):
        for vessel in cfg.VESSELS:
            for tier in range(4):
                s = layout("scatter", vessel, tier)
                self.assertTrue(s.direct_blocked)
                self.assertAlmostEqual(s.centers[0][1], 0.0)
                ice = layout("iceberg", vessel, tier)
                self.assertEqual(ice.obstacle_count, cfg.ICEBERG_COUNTS[tier])
                self.assertTrue(ice.direct_blocked)

    def test_ring_single_opening_at_tier_gap_and_goal_outside(self):
        for vessel in cfg.VESSELS:
            v = cfg.VESSELS[vessel].geometry
            for tier in range(4):
                lay = layout("ring", vessel, tier)
                i, j = lay.gates[0].jamb_indices
                gap = np.linalg.norm(lay.centers[i] - lay.centers[j]) - lay.radii[i] - lay.radii[j]
                self.assertAlmostEqual(gap, geo.physical_tier(tier, v).physical_surface_gap_m, places=9)
                self.assertGreater(np.linalg.norm(lay.goal), lay.params["ring_radius_m"] + lay.params["obstacle_radius_m"])
                self.assertGreaterEqual(lay.extras["ring_neighbour_overlap_m"], lay.params["seal_overlap_m"] - 1e-12)

    def test_fortress_two_certified_staggered_gates_no_shortcut(self):
        for vessel in cfg.VESSELS:
            for tier in range(4):
                lay = layout("fortress", vessel, tier)
                self.assertEqual([g.name for g in lay.gates], ["outer_gate", "inner_gate"])
                self.assertTrue(all(g.certified for g in lay.gates))
                self.assertGreaterEqual(lay.extras["gate_stagger_rad"], math.pi / 2 - 1e-12)
                self.assertLessEqual(lay.admission.direct_physical_clearance_m, 0.0)
                self.assertEqual(lay.count_mode, "recorded")

    def test_forced_crossing_gate_width_and_leak_free(self):
        for vessel in cfg.VESSELS:
            v = cfg.VESSELS[vessel].geometry
            for tier in range(4):
                lay = layout("forced_crossing", vessel, tier)
                gate = lay.gates[0]
                self.assertAlmostEqual(gate.physical_gap_m, geo.physical_tier(tier, v).physical_surface_gap_m, places=6)
                self.assertTrue(gate.certified)
                self.assertGreater(lay.extras["wall_x_m"], v.approach_runup_m)

    def test_docking_berth_containment_and_c1_27_style_exterior_point(self):
        for vessel in cfg.VESSELS:
            rec = cfg.VESSELS[vessel]
            for tier in range(4):
                lay = layout("docking", vessel, tier)
                berth = J.Berth.from_layout(lay.extras["berth"])
                hull = rec.footprint.vertices_m
                self.assertTrue(berth.contains_polygon((0, 0), 0.0, hull))
                behind = (lay.extras["berth"]["back_x_m"] + lay.params["wall_radius_m"] * 2
                          + rec.geometry.aft_extent_m + 0.1, 0.0)
                self.assertFalse(berth.contains_polygon(behind, 0.0, hull))
                self.assertTrue(lay.gates[0].certified)      # the mouth is the only way in

    def test_docking_kinematic_trajectory_through_generated_berth_succeeds(self):
        for vessel in cfg.VESSELS:
            rec = cfg.VESSELS[vessel]
            lay = layout("docking", vessel, 2)
            b = lay.extras["berth"]
            plan = cfg.JOURNAL_PLAN_CLASSES[("docking", vessel)](2)
            ep = J.DockingEpisode(J.Berth.from_layout(b), rec.footprint.vertices_m, lay.deadline_s,
                                  hold_s=b["hold_s"], position_tolerance_m=b["position_tolerance_m"],
                                  heading_tolerance_rad=b["heading_tolerance_rad"],
                                  speed_tolerance_mps=b["speed_tolerance_m_s"],
                                  validity_limits=J.ValidityLimits(**plan.validity_limits))
            x_open = b["opening_x_m"]
            ep.reset((x_open - 3.0 * rec.geometry.length_m, 0.0))
            t, dt, u = 0.0, 1.0 / 60.0, 0.5 * rec.cruise_speed_m_s
            x = x_open - 3.0 * rec.geometry.length_m
            decision = None
            while x < 0.0:
                t += dt
                x = min(0.0, x + u * dt)
                vx = u if x < 0.0 else 0.0
                p = J.RigidState((x, 0.0, 0.0), (1, 0, 0, 0), (vx, 0, 0), (0, 0, 0))
                c = geo.pose_clearance(rec.footprint, (x, 0.0, 0.0), lay.centers, lay.radii)
                decision = ep.step(round(t * 60) / 60.0, p, clearance_m=c)
            for _ in range(int(6.0 / dt)):
                t += dt
                p = J.RigidState((0.0, 0.0, 0.0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0))
                decision = ep.step(round(t * 60) / 60.0, p,
                                   clearance_m=geo.pose_clearance(rec.footprint, (0, 0, 0), lay.centers, lay.radii))
                if decision.done:
                    break
            self.assertEqual(decision.reason, J.TerminationReason.SUCCESS, vessel)
            self.assertGreater(ep.last_completed.metrics["min_clearance_m"], 0.0)

    def test_moving_basin_dose_independent_geometry_and_legacy_motion_law(self):
        for vessel in cfg.VESSELS:
            hashes = set()
            for mu in (0.0, 0.08, 0.30):
                lay = L.build_layout("moving_basin", 2, vessel, np.random.default_rng(7), mu=mu)
                hashes.add(lay.geometry_hash)
                m = lay.extras["moving"]
                k, pr = np.asarray(m["kind"]), np.asarray(m["params"])
                for t in (0.0, 1.3, 7.7, 31.0):
                    self.assertTrue(np.array_equal(L.moving_positions(lay.centers, k, pr, t),
                                                   legacy.moving_basin_positions(lay.centers, k, pr, t)))
                self.assertEqual(len(m["base_slots"]), len(lay.d7_hook["base_slots"]))
                self.assertEqual(lay.d7_hook["reserved_dynamic_slots"], 0)
            self.assertEqual(len(hashes), 1, vessel)

    def test_harbour_stage_geometry(self):
        for vessel in cfg.VESSELS:
            for tier in range(4):
                lay = layout("harbour_mission", vessel, tier)
                names = [g.name for g in lay.gates]
                self.assertEqual(names, ["departure_1", "departure_2", "transit_exit", "berth_mouth"])
                xs = [g.midpoint_m[0] for g in lay.gates]
                self.assertEqual(xs, sorted(xs))
                self.assertTrue(lay.gates[-1].certified)
                hook = lay.d7_hook
                self.assertEqual([s[0] for s in hook["stages"]], ["departure", "transit", "docking"])
                self.assertEqual(hook["stages"][hook["site_index"]][0], hook["declared_stage"])
                self.assertEqual([s[2] for s in hook["stages"]], [1, 2, 3])   # route indices of d2, exit, mouth
                self.assertEqual(len(lay.field_indices), cfg.FIELD_COUNTS[tier])

    def test_station_and_reach_open_water(self):
        for vessel in cfg.VESSELS:
            rec = cfg.VESSELS[vessel]
            st = layout("station_keeping", vessel, 1)
            lo, hi = st.extras["late_interval_s"]
            self.assertAlmostEqual(hi, st.deadline_s)
            self.assertAlmostEqual(hi - lo, 60.0)
            self.assertAlmostEqual(st.d7_hook["settled_phase_start_s"], lo)
            r = layout("reach", vessel, 3)
            self.assertEqual(r.obstacle_count, 0)
            self.assertAlmostEqual(r.witness.length_m, r.params["straight_distance_m"])
            align = abs((0.0 - r.start_heading_rad + math.pi) % (2 * math.pi) - math.pi)
            self.assertAlmostEqual(r.nominal_route_length_m,
                                   r.params["straight_distance_m"] + rec.turning_radius_m * align)
            self.assertAlmostEqual(r.deadline_s, math.ceil(2 * r.nominal_route_length_m / rec.cruise_speed_m_s * 60 - 1e-9) / 60)


def drive(episode, lay, rec, *, speed_fraction=0.5, hold_s=0.0, step_kwargs=None):
    """Kinematic COM trajectory along the verified witness polyline at a
    fraction of cruise speed (heading = segment heading; the witness already
    certified translation sweeps and every vertex fillet/pivot), then an
    optional stationary hold. Returns the final Decision."""
    v = np.asarray(lay.witness.vertices_m, dtype=np.float64)
    seg = np.diff(v, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(lens)))
    u = speed_fraction * rec.cruise_speed_m_s
    episode.reset(tuple(v[0]))
    fp = rec.footprint
    k, decision = 0, None
    while True:
        k += 1
        t = k / 60.0
        s = min(u * t, cum[-1])
        i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(lens) - 1)
        pos = v[i] + (s - cum[i]) / lens[i] * seg[i]
        h = math.atan2(seg[i][1], seg[i][0])
        moving = s < cum[-1]
        vel = (u * math.cos(h), u * math.sin(h), 0.0) if moving else (0.0, 0.0, 0.0)
        if not moving:
            hold_s -= 1.0 / 60.0
        qz = (math.cos(h / 2), 0.0, 0.0, math.sin(h / 2))
        st = J.RigidState((pos[0], pos[1], 0.0), qz, vel, (0.0, 0.0, 0.0))
        c = geo.pose_clearance(fp, (pos[0], pos[1], h), lay.centers, lay.radii) if lay.obstacle_count else None
        decision = episode.step(t, st, clearance_m=c, **(step_kwargs or {}))
        if decision.done or (not moving and hold_s <= 0.0) or t >= episode.deadline_s:
            return decision


class EndToEndTests(unittest.TestCase):
    def test_goal_disk_tasks_succeed_along_witness(self):
        for task in ("reach", "scatter", "ring", "fortress", "forced_crossing", "iceberg", "moving_basin"):
            for vessel in cfg.VESSELS:
                rec = cfg.VESSELS[vessel]
                lay = layout(task, vessel, 2)
                plan = cfg.JOURNAL_PLAN_CLASSES[(task, vessel)](2)
                ep = J.ReachEpisode(tuple(lay.goal), lay.params["goal_radius_m"], lay.deadline_s,
                                    J.ValidityLimits(**plan.validity_limits))
                d = drive(ep, lay, rec)
                self.assertEqual(d.reason, J.TerminationReason.SUCCESS, (task, vessel))
                self.assertGreater(ep.last_completed.metrics["min_clearance_m"], 0.0)

    def test_ordered_path_tasks_succeed_along_witness(self):
        for task in ("path_following", "path_hazard"):
            for vessel in cfg.VESSELS:
                rec = cfg.VESSELS[vessel]
                lay = layout(task, vessel, 3)
                route = (tuple(lay.start),) + tuple(lay.waypoints)
                ep = J.OrderedPathEpisode(route, lay.params["gate_radius_m"], lay.deadline_s)
                d = drive(ep, lay, rec)
                self.assertEqual(d.reason, J.TerminationReason.SUCCESS, (task, vessel))
                self.assertEqual(ep.last_completed.metrics["ordered_gates_completed"], 4)

    def test_full_harbour_mission_succeeds_along_witness(self):
        for vessel in cfg.VESSELS:
            rec = cfg.VESSELS[vessel]
            lay = layout("harbour_mission", vessel, 2)
            ex = lay.extras
            gates = [J.VirtualGate(gr.name, gr.midpoint_m, gr.through_m, ex["virtual_gate_half_width_m"],
                                   ex["gate_approach_m"], ex["gate_min_normal_speed_m_s"])
                     for gr in lay.gates if gr.kind == "virtual"]
            b = ex["berth"]
            ep = J.HarborMissionEpisode(J.Berth.from_layout(b), rec.footprint.vertices_m, lay.deadline_s,
                                        departure_gates=gates[:2], transit_gate=gates[2], hold_s=b["hold_s"],
                                        position_tolerance_m=b["position_tolerance_m"],
                                        heading_tolerance_rad=b["heading_tolerance_rad"],
                                        speed_tolerance_mps=b["speed_tolerance_m_s"])
            d = drive(ep, lay, rec, hold_s=6.0)
            self.assertEqual(d.reason, J.TerminationReason.SUCCESS, vessel)
            m = ep.last_completed.metrics
            self.assertTrue(m["stage_departure_complete"] and m["stage_transit_complete"])
            self.assertEqual(m["contract_scope"], "harbour_full_mission")


if __name__ == "__main__":
    unittest.main(verbosity=2)
