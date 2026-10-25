"""Geometry core regressions, including C1 counterexamples against the ACTUAL
copied legacy samplers (tasks/hazard_nav/hazard_geometry.py in this repo,
byte-identical to C2's copy). CPU only; modules loaded by path."""

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


g = load("d4_test_geometry", ROOT / "tasks/_shared/journal_geometry_v2.py")
cfg = load("d4_test_cfg_for_geometry", ROOT / "tasks/_shared/journal_cfg_v2.py")
legacy = load("d4_test_legacy_hazard_geometry", ROOT / "tasks/hazard_nav/hazard_geometry.py")

BB = cfg.BLUEBOAT.geometry
WV = cfg.WAMV.geometry
LEGACY_DISK = g.VesselGeometry.from_dimensions("legacy_disk_proxy", 1.2, 0.9, 1.2)


class TierDefinitionTests(unittest.TestCase):
    def test_blueboat_reference_values_and_wamv_scaling(self):
        self.assertAlmostEqual(BB.length_scale, 1.0, places=12)
        for k, (gap, w) in enumerate(((4.93, 4.0), (3.93, 3.0), (2.93, 2.0), (2.13, 1.2))):
            t = g.physical_tier(k, BB)
            self.assertAlmostEqual(t.physical_surface_gap_m, gap, places=9)
            self.assertAlmostEqual(t.hull_centre_width_m, w, places=9)
            self.assertAlmostEqual(t.planning_margin_m, 0.2, places=12)
            self.assertAlmostEqual(t.planned_centre_width_m, w - 0.4, places=9)
            tw = g.physical_tier(k, WV)
            self.assertAlmostEqual(tw.hull_centre_width_m, w * WV.length_scale, places=9)
            self.assertAlmostEqual(tw.physical_surface_gap_m, WV.beam_m + w * WV.length_scale, places=9)
        self.assertAlmostEqual(WV.length_scale, 5.37719612 / 1.200825846195221, places=6)

    def test_scale_uses_length_beam_enters_gap_turning_radius_does_not_rescale(self):
        base = g.VesselGeometry.from_dimensions("a", 1.2, 0.9, 1.2)
        longer = g.VesselGeometry.from_dimensions("b", 2.4, 0.9, 1.2)
        wider = g.VesselGeometry.from_dimensions("c", 1.2, 1.1, 1.2)
        slow_turn = g.VesselGeometry.from_dimensions("d", 1.2, 0.9, 9.0)
        t0, t1, t2, t3 = (g.physical_tier(3, v) for v in (base, longer, wider, slow_turn))
        self.assertAlmostEqual(t1.hull_centre_width_m, 2.0 * t0.hull_centre_width_m, places=6)
        self.assertAlmostEqual(t2.hull_centre_width_m, t0.hull_centre_width_m, places=9)
        self.assertAlmostEqual(t2.physical_surface_gap_m - t0.physical_surface_gap_m, 0.2, places=9)
        self.assertEqual(t3, t0)
        with self.assertRaises(ValueError):
            g.physical_tier(1.0, BB)
        with self.assertRaises(ValueError):
            g.physical_tier(True, BB)

    def test_centred_heading_tolerance(self):
        self.assertEqual(g.centred_heading_tolerance(1.2, 0.9, 0.8), 0.0)
        self.assertEqual(g.centred_heading_tolerance(1.2, 0.9, 1.6), math.pi / 2)
        a = g.centred_heading_tolerance(4.0, 1.0, 2.0)
        self.assertAlmostEqual(1.0 * math.cos(a) + 4.0 * math.sin(a), 2.0, places=12)


class FootprintAndSweepTests(unittest.TestCase):
    def test_com_relative_footprints(self):
        self.assertAlmostEqual(BB.inscribed_radius_m, 0.418000002, places=6)
        self.assertAlmostEqual(BB.circumscribed_radius_m, math.hypot(0.606825846, 0.512), places=6)
        # E5: WAM-V expectations from the registry's default profile polygon
        # (was the pinned 258 kg platform: 1.267135 / 2.841372301 / 2.535823819).
        poly = cfg.registry.get_vehicle_v2(WV.name).collision.polygon_com_xy_m
        x0, x1 = min(p[0] for p in poly), max(p[0] for p in poly)
        y0, y1 = min(p[1] for p in poly), max(p[1] for p in poly)
        self.assertAlmostEqual(WV.inscribed_radius_m, min(-x0, x1, -y0, y1), places=12)
        self.assertAlmostEqual(WV.circumscribed_radius_m, max(math.hypot(*p) for p in poly), places=12)
        self.assertAlmostEqual(WV.forward_extent_m, x1, places=12)
        self.assertAlmostEqual(WV.aft_extent_m, -x0, places=12)

    def test_capsule_encloses_polygon_and_is_conservative(self):
        for v in (BB, WV):
            cap = v.capsule
            self.assertTrue(cap.contains_polygon(v.footprint))
            rng = np.random.default_rng(3)
            for _ in range(200):
                pose = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-math.pi, math.pi))
                c = rng.uniform(-4, 4, size=(3, 2)) * v.length_scale
                r = rng.uniform(0.1, 1.0, size=3) * v.length_scale
                self.assertLessEqual(g.pose_clearance(cap, pose, c, r),
                                     g.pose_clearance(v.footprint, pose, c, r) + 1e-12)

    def test_orientation_and_length_matter(self):
        hull = g.PolygonFootprint.from_dimensions(1.2, 0.9)
        obstacle = ((0.64, 0.0),)
        self.assertLess(g.pose_clearance(hull, (0, 0, 0), obstacle, (0.08,)), 0)
        self.assertGreater(g.pose_clearance(hull, (0, 0, math.pi / 2), obstacle, (0.08,)), 0)
        # the legacy 0.45 m beam disk misses this bow contact entirely
        self.assertGreater(0.64 - 0.08 - 0.45, 0.0)

    def test_translation_sweep_exact_when_clear_conservative_when_colliding(self):
        rng = np.random.default_rng(11)
        n_clear = n_hit = 0
        for v in (BB, WV):
            for _ in range(60):
                a = rng.uniform(-5, 5, 2) * v.length_scale
                b = rng.uniform(-5, 5, 2) * v.length_scale
                h = rng.uniform(-math.pi, math.pi)
                c = rng.uniform(-6, 6, (4, 2)) * v.length_scale
                r = rng.uniform(0.2, 1.5, 4) * v.length_scale
                sweep = g.translation_sweep_clearance(v.footprint, a, b, h, c, r)
                dense = min(g.pose_clearance(v.footprint, (*(a + f * (b - a)), h), c, r)
                            for f in np.linspace(0, 1, 4001))
                self.assertLessEqual(sweep, dense + 1e-9)          # never optimistic
                if dense > 1e-3 * v.length_scale:
                    n_clear += 1
                    self.assertAlmostEqual(sweep, dense, delta=2e-3 * v.length_scale)   # exact when clear
                if dense < -1e-3 * v.length_scale:
                    n_hit += 1
                    self.assertLess(sweep, 0.0)
        self.assertGreater(n_clear, 10)
        self.assertGreater(n_hit, 10)

    def test_swept_translation_rotation_and_moving_obstacles_cannot_tunnel(self):
        hull = g.PolygonFootprint.from_dimensions(1.2, 0.9)
        self.assertGreater(g.pose_clearance(hull, (-2, 0, 0), ((0, 0),), (0.1,)), 0)
        self.assertGreater(g.pose_clearance(hull, (2, 0, 0), ((0, 0),), (0.1,)), 0)
        self.assertLessEqual(g.swept_clearance(hull, (-2, 0, 0), (2, 0, 0), ((0, 0),), (0.1,)), 0)
        long_hull = g.PolygonFootprint.from_dimensions(4, 0.5)
        self.assertGreater(g.pose_clearance(long_hull, (0, 0, 0), ((1.2, 1.2),), (0.1,)), 0)
        self.assertGreater(g.pose_clearance(long_hull, (0, 0, math.pi / 2), ((1.2, 1.2),), (0.1,)), 0)
        self.assertLessEqual(g.swept_clearance(long_hull, (0, 0, 0), (0, 0, math.pi / 2),
                                               ((1.2, 1.2),), (0.1,)), 0)
        self.assertLessEqual(g.swept_clearance(hull, (0, 0, 0), (0, 0, 0), ((0, -2),), (0.1,),
                                               end_centers_m=((0, 2),)), 0)
        self.assertGreater(g.swept_clearance(hull, (0, 0, 0), (0, 0, 0), ((3, -2),), (0.1,),
                                             end_centers_m=((3, 2),)), 0)
        self.assertGreater(g.swept_clearance(hull, (-1, 2, 0), (1, 2, 0), ((0, 0),), (0.1,)), 0)
        self.assertEqual(g.swept_clearance(hull, (0, 0, float("nan")), (1, 0, 0), ((5, 5),), (1,)),
                         -math.inf)


class GridRouteTests(unittest.TestCase):
    def test_c1_37_actual_scatter_seed5_layout6_bfs_vs_weighted(self):
        rng = np.random.default_rng(5)
        for _ in range(7):
            layout = legacy.sample_layout(3, rng=rng)
        self.assertAlmostEqual(layout.geodesic_length, 40.978913, places=5)
        a = g.weighted_geodesic(layout.start, layout.goal, layout.centers, layout.radii, inflation_m=0.65)
        d = g.weighted_geodesic(layout.start, layout.goal, layout.centers, layout.radii, inflation_m=0.65,
                                algorithm="dijkstra")
        self.assertAlmostEqual(a.length_m, 39.736272, places=5)
        self.assertAlmostEqual(a.length_m, d.length_m, places=9)
        for p, q in zip(a.points_m, a.points_m[1:]):
            self.assertGreater(g.segment_clearance(p, q, layout.centers, layout.radii, 0.65), 0.0)
        print("C1#37 legacy BFS=%.9f m, weighted A*=%.9f m, Dijkstra=%.9f m"
              % (layout.geodesic_length, a.length_m, d.length_m))

    def test_c1_38_colliding_diagonal_edge(self):
        start, goal = (0.0, 0.0), (0.5, 0.5)
        centers, radii = ((-0.612670273, 1.112670273),), (0.8,)
        old = legacy.bfs_geodesic_length(np.array(start), np.array(goal), np.array(centers),
                                         np.array(radii), inflation_m=0.45)
        self.assertAlmostEqual(old, math.sqrt(0.5), places=7)
        self.assertGreater(g.segment_clearance(start, start, centers, radii, 0.45), 0.0)
        self.assertGreater(g.segment_clearance(goal, goal, centers, radii, 0.45), 0.0)
        self.assertAlmostEqual(g.segment_clearance(start, goal, centers, radii, 0.45), -0.03, places=7)
        route = g.weighted_geodesic(start, goal, centers, radii, inflation_m=0.45, bounds_m=(-2, 2, -2, 2))
        self.assertTrue(route.feasible)
        self.assertAlmostEqual(route.length_m, 1.0, places=12)
        field = g.weighted_distance_field(goal, centers, radii, inflation_m=0.45, bounds_m=(-2, 2, -2, 2))
        self.assertAlmostEqual(field.distances_m[4, 4], 1.0, places=12)
        print("C1#38 legacy BFS=%.6f m through an obstacle; validated weighted route=%.6f m"
              % (old, route.length_m))

    def test_connector_and_domain_predicate_checked(self):
        self.assertFalse(g.weighted_geodesic((0.24, 0.24), (1, 1), ((0.12, 0.12),), (0.05,),
                                             inflation_m=0.0, bounds_m=(0, 2, 0, 2)).feasible)
        blocked = g.weighted_geodesic((0, 0), (2, 0), (), (), inflation_m=0, bounds_m=(-1, 3, -1, 2),
                                      segment_allowed=lambda a, b: not (min(a[0], b[0]) < 1 <= max(a[0], b[0])))
        self.assertFalse(blocked.feasible)

    def test_fast_accept_matches_exact_edges(self):
        rng = np.random.default_rng(2)
        for _ in range(8):
            c = rng.uniform(0, 10, (6, 2))
            r = rng.uniform(0.3, 1.2, 6)
            fast = g.weighted_geodesic((0, 0), (10, 10), c, r, inflation_m=0.3, bounds_m=(-1, 11, -1, 11))
            exact = g.weighted_geodesic((0, 0), (10, 10), c, r, inflation_m=0.3, bounds_m=(-1, 11, -1, 11),
                                        segment_allowed=lambda a, b: True)
            self.assertEqual(fast.feasible, exact.feasible)
            if fast.feasible:
                self.assertAlmostEqual(fast.length_m, exact.length_m, places=9)

    def test_nonfinite_fails(self):
        with self.assertRaises(ValueError):
            g.weighted_geodesic((0, 0), (1, 0), ((float("nan"), 0),), (1,), inflation_m=0)


class CertificateAndRingTests(unittest.TestCase):
    def test_c1_33_actual_original_ring_has_runtime_gaps_and_no_certifiable_gate(self):
        for level in (0, 3):
            rng = np.random.default_rng(11)
            for _ in range(3):
                layout = legacy.sample_ring_layout(level, rng=rng)
                order = np.argsort(np.arctan2(layout.centers[:, 1], layout.centers[:, 0]))
                c, r = layout.centers[order], layout.radii[order]
                physical = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1) - r - np.roll(r, -1)
                self.assertGreater(int(np.sum(physical > 0.93)), 1)
                self.assertGreater(float(np.min(physical)), 1.0)
                for r_in in (BB.inscribed_radius_m, 0.45):
                    barrier = g.Barrier(tuple(map(int, order)), int(np.argmax(physical)))
                    self.assertIn("unintended_runtime_gap",
                                  g.validate_barrier(barrier, layout.start, layout.goal, layout.centers,
                                                     layout.radii, r_in))
                    for k in range(len(order)):
                        gate = (int(order[k]), int(order[(k + 1) % len(order)]))
                        self.assertFalse(g.compulsory_gate_certificate(
                            layout.start, layout.goal, layout.centers, layout.radii, gate, r_in).valid)
        print("C1#33 legacy ring seed11 levels 0/3: runtime gaps; no gate certifiable")

    def test_constructed_rings_all_tiers_both_vessels(self):
        for v in (BB, WV):
            s = v.length_scale
            for k in range(4):
                tier = g.physical_tier(k, v)
                ring = g.construct_sealed_ring(v, tier, ring_radius_m=10.5 * s, obstacle_radius_m=1.75 * s,
                                               gate_bearing_rad=0.4)
                self.assertGreaterEqual(ring.neighbour_overlap_m, 0.05 * s - 1e-12)
                i, j = ring.gate
                gap = np.linalg.norm(ring.centers_m[i] - ring.centers_m[j]) - ring.radii_m[i] - ring.radii_m[j]
                self.assertAlmostEqual(gap, tier.physical_surface_gap_m, places=9)
                goal = 30 * s * np.array((math.cos(0.9), math.sin(0.9)))
                cert = g.compulsory_gate_certificate((0, 0), goal, ring.centers_m, ring.radii_m, ring.gate,
                                                     v.inscribed_radius_m)
                self.assertTrue(cert.valid, cert.reasons)
                w = g.route_witness([(0, 0), goal], ring.centers_m, ring.radii_m, v,
                                    margin_m=tier.planning_margin_m, bounds_m=(-35 * s, 35 * s, -35 * s, 35 * s),
                                    start_heading_rad=-2.0)
                self.assertTrue(w.feasible, w.reasons)
                self.assertGreaterEqual(w.min_clearance_m, tier.planning_margin_m - 1e-9)
                with self.assertRaises(ValueError):
                    ring.centers_m[0] = (0, 0)

    def test_wamv_g3_needs_oriented_witness_not_circumscribed_disk(self):
        tier = g.physical_tier(3, WV)
        s = WV.length_scale
        ring = g.construct_sealed_ring(WV, tier, ring_radius_m=10.5 * s, obstacle_radius_m=1.75 * s)
        goal = (30 * s, 0.0)
        bounds = (-35 * s, 35 * s, -35 * s, 35 * s)
        disk = g.weighted_geodesic((0, 0), goal, ring.centers_m, ring.radii_m,
                                   inflation_m=WV.circumscribed_radius_m + tier.planning_margin_m,
                                   cell_m=0.25 * s, bounds_m=bounds)
        self.assertFalse(disk.feasible)
        w = g.route_witness([(0, 0), goal], ring.centers_m, ring.radii_m, WV, margin_m=tier.planning_margin_m,
                            bounds_m=bounds)
        self.assertTrue(w.feasible, w.reasons)
        print("WAM-V G3 ring: orientation-free disk route infeasible; oriented witness %.2f m" % w.length_m)

    def test_forced_basin_certificate_detects_leak_and_closed_gate(self):
        L = load("d4_test_layouts_for_geo", ROOT / "tasks/_shared/journal_layouts_v2.py")
        lay = L.build_layout("forced_crossing", 2, "blueboat_v2", np.random.default_rng(4))
        gate = lay.gates[0].jamb_indices
        cert = g.compulsory_gate_certificate(lay.start, lay.goal, lay.centers, lay.radii, gate,
                                             BB.inscribed_radius_m)
        self.assertTrue(cert.valid)
        # A leak in ONE chamber's outer wall does not make the gate optional
        # (the other chamber stays sealed), so the proof must still hold ...
        wall_x = lay.extras["wall_x_m"]
        rb = lay.params["bulkhead_radius_m"]
        perimeter = [k for k in cert.walk if k not in gate and abs(lay.radii[k] - rb) > 1e-9]
        victim = perimeter[len(perimeter) // 2]
        keep = [k for k in range(len(lay.radii)) if k != victim]
        remap = {old: new for new, old in enumerate(keep)}
        one_side = g.compulsory_gate_certificate(lay.start, lay.goal, lay.centers[keep], lay.radii[keep],
                                                 (remap[gate[0]], remap[gate[1]]), BB.inscribed_radius_m)
        self.assertTrue(one_side.valid)
        # ... but removing a BULKHEAD cylinder (not a jamb) opens a second
        # opening shared by both chambers: no certificate may exist.
        bulk = [k for k in range(len(lay.radii)) if abs(lay.radii[k] - rb) < 1e-9
                and abs(lay.centers[k][0] - wall_x) < 1e-9 and k not in gate]
        victim = bulk[len(bulk) // 2]
        keep = [k for k in range(len(lay.radii)) if k != victim]
        remap = {old: new for new, old in enumerate(keep)}
        leaky = g.compulsory_gate_certificate(lay.start, lay.goal, lay.centers[keep], lay.radii[keep],
                                              (remap[gate[0]], remap[gate[1]]), BB.inscribed_radius_m)
        self.assertFalse(leaky.valid)
        self.assertIn("no_separating_closed_barrier", leaky.reasons)
        # a gate narrower than the runtime footprint is reported as closed
        tier = g.physical_tier(2, BB)
        ring = g.construct_sealed_ring(BB, tier, ring_radius_m=10.5, obstacle_radius_m=1.75)
        closed = g.compulsory_gate_certificate((0, 0), (30, 0), ring.centers_m, ring.radii_m, ring.gate,
                                               tier.physical_surface_gap_m / 2.0 + 0.01)
        self.assertFalse(closed.valid)
        self.assertIn("gate_closed_at_runtime_footprint", closed.reasons)


class FortressValidationTests(unittest.TestCase):
    def test_c1_34_actual_band_fortress_seed42_layout29_rejected(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            layout = legacy.sample_band_fortress_layout(0, rng=rng)
        self.assertFalse(layout.direct_blocked)
        self.assertEqual(layout.obstacle_count, 42)
        self.assertAlmostEqual(float(np.linalg.norm(layout.goal - layout.start)), 38.878868, places=5)
        disk = g.segment_clearance(layout.start, layout.goal, layout.centers, layout.radii, 0.45)
        self.assertAlmostEqual(disk, 0.383028, places=5)
        d = layout.goal - layout.start
        oriented = g.translation_sweep_clearance(BB.footprint, layout.start, layout.goal,
                                                 math.atan2(d[1], d[0]), layout.centers, layout.radii)
        self.assertGreater(oriented, 0.0)
        for skipped in (0, None, 3):
            res = g.validate_band_fortress(layout.start, layout.goal, layout.centers, layout.radii, vessel=BB,
                                           skipped_plugs=skipped, requested_count=42)
            self.assertFalse(res.admitted)
            self.assertIn("direct_shortcut", res.reasons)
            if skipped != 0:
                self.assertTrue(any("plug" in r for r in res.reasons))
        print("C1#34 seed42 index29: disk clearance %.6f m, full-footprint sweep %.6f m -> rejected"
              % (disk, oriented))

    def test_plug_failure_rejected_even_without_shortcut(self):
        wall = np.array([(-5.0, y) for y in np.linspace(-20, 20, 60)])
        radii = np.full(len(wall), 0.6)
        for skipped in (None, 1, -1, True):
            res = g.validate_band_fortress((-10, 0), (0, 0), wall, radii, vessel=BB, skipped_plugs=skipped)
            self.assertFalse(res.admitted)
            self.assertNotIn("direct_shortcut", res.reasons)
        ok = g.validate_band_fortress((-10, 0), (0, 0), wall, radii, vessel=BB, skipped_plugs=0,
                                      requested_count=len(radii))
        self.assertTrue(ok.admitted, ok.reasons)
        rec = g.validate_band_fortress((-10, 0), (0, 0), wall, radii, vessel=BB, skipped_plugs=0,
                                       recorded_realized_count=len(radii))
        self.assertTrue(rec.admitted, rec.reasons)
        self.assertEqual(rec.dose_count, len(radii))
        missing = g.validate_band_fortress((-10, 0), (0, 0), wall, radii, vessel=BB, skipped_plugs=0)
        self.assertIn("obstacle_count_mismatch", missing.reasons)   # recorded count must be stated


class WitnessTests(unittest.TestCase):
    def test_fillet_when_it_fits_else_verified_pivot(self):
        pts = [(0, 0), (5, 0), (5, 5)]
        R = BB.turning_radius_m
        ok = g.route_witness(pts, (), (), BB, margin_m=0.2, bounds_m=(-5, 10, -5, 10))
        self.assertTrue(ok.feasible)
        self.assertEqual((ok.n_fillets, ok.n_pivots), (1, 0))
        self.assertAlmostEqual(ok.length_m, 10 - 2 * R + R * math.pi / 2, places=9)
        self.assertAlmostEqual(ok.time_equivalent_length_m, ok.length_m, places=12)
        big = g.route_witness(pts, (), (), BB, margin_m=0.2, bounds_m=(-5, 10, -5, 10), turn_radius_m=6.0)
        self.assertTrue(big.feasible)                         # fillet cannot fit -> pivot
        self.assertEqual((big.n_fillets, big.n_pivots, big.pivot_vertex_indices), (0, 1, (1,)))
        self.assertAlmostEqual(big.length_m, 10.0, places=9)
        self.assertAlmostEqual(big.time_equivalent_length_m, 10.0 + 6.0 * math.pi / 2, places=9)
        # A bollard clear of both straight sweeps but inside the arc swept by
        # the stern-starboard corner (-0.594, -0.418) mid-pivot is rejected.
        bollard = ((4.875, -0.80),), (0.1,)
        v = np.asarray(pts, dtype=float)
        self.assertGreater(g.translation_sweep_clearance(BB.footprint, v[0], v[1], 0.0, *bollard), 0)
        self.assertGreater(g.translation_sweep_clearance(BB.footprint, v[1], v[2], math.pi / 2, *bollard), 0)
        blocked = g.route_witness(pts, *bollard, BB, margin_m=0.0, bounds_m=(-5, 10, -5, 10), turn_radius_m=6.0)
        self.assertFalse(blocked.feasible)
        self.assertTrue(blocked.reasons[0].startswith("no_clear_fillet_or_pivot"))
        # a bollard right beside the start blocks the start pivot but not the straight leg
        pivot = g.route_witness([(0, 0), (8, 0)], ((0.0, 1.0),), (0.25,), BB, margin_m=0.0,
                                bounds_m=(-5, 10, -5, 10), start_heading_rad=math.pi / 2)
        self.assertFalse(pivot.feasible)
        self.assertIn("start_alignment_collision", pivot.reasons)

    def test_counts(self):
        self.assertFalse(g.admit_counts(14, 13).admitted)
        self.assertFalse(g.admit_counts(14, 13, mode="recorded").admitted)
        rec = g.admit_counts(14, 13, mode="recorded", recorded_realized_count=13)
        self.assertTrue(rec.admitted)
        self.assertEqual(rec.dose_count, 13)
        with self.assertRaises(ValueError):
            g.admit_counts(True, 1)

    def test_digest_is_deterministic_and_sensitive(self):
        # G7 item A: sensitive to every change of at least one declared quantum (1e-6 SI units),
        # blind to sub-quantum noise (last-ulp platform differences, -0.0 vs 0.0).
        a = g.geometry_digest(c=np.zeros((2, 2)), n=1)
        self.assertEqual(a, g.geometry_digest(n=1, c=np.zeros((2, 2))))
        self.assertEqual(g.IDENTITY_QUANTUM, 1.0e-6)
        self.assertNotEqual(a, g.geometry_digest(c=np.array(((0.0, 0.0), (0.0, 1e-6))), n=1))
        self.assertNotEqual(a, g.geometry_digest(c=np.zeros((2, 2)), n=2))
        self.assertEqual(a, g.geometry_digest(c=np.array(((0.0, 0.0), (0.0, 1e-12))), n=1))
        self.assertEqual(a, g.geometry_digest(c=np.array(((-0.0, 0.0), (0.0, -0.0))), n=1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
