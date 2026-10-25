"""G10 item 2: simulated green-water detection (CPU; unit level).

* declared deck-edge point sets (coordinates from the registry, calm freeboards);
* point world positions include heave, roll, pitch and yaw (checked against an
  independent rotation);
* the local elevation is bit-identical to SpectralWaves.sample(footprint=False)
  of the SAME realisation and clock (one evaluation path);
* the consecutive-substep counter and the N rule (5 substeps: no green water,
  6: green water, onset time and point), single-substep flicker and 5-on/1-off
  patterns never trigger, partial (roll) immersion names the right point;
* reset-row isolation; per-episode record fields; the runtime tick cross-check;
* no host synchronisation in the per-substep path (source scan).

Run: python -B scripts/tests_g10/test_g10_detection.py   (or pytest)
"""
from __future__ import annotations

import ast
import inspect
import math
import os
import sys
import textwrap
import unittest

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "tests_journal_glue"))

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

GW = G.shared("conditions.green_water")
RULE = G.shared("conditions.green_water_rule")
V2 = G.shared("vehicles_v2")
DATA = G.shared("vehicles_v2_data")
DT = 1.0 / 120.0
FORMULA = {"version": "usvbench.green_water.formula/1", "probability": 0.0, "limit": 0.05}


def monitor(vessel="blueboat_v2", n=2):
    return GW.GreenWaterMonitor(vessel, n, "cpu", torch.float32, DT, dict(FORMULA))


def quat(roll=0.0, pitch=0.0, yaw=0.0):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(yaw / 2), math.sin(yaw / 2))
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def poses(rows):
    """rows: [(x, y, z, roll, pitch, yaw)] -> [N,7] float32 root-link poses."""
    return torch.tensor([[x, y, z] + quat(r, p, w) for x, y, z, r, p, w in rows], dtype=torch.float32)


ORIGINS = torch.zeros((2, 3), dtype=torch.float32)


class PointSets(unittest.TestCase):
    def test_blueboat_deck_edge_points(self):
        points = GW.deck_edge_points("blueboat_v2")
        self.assertEqual(len(points), 12)
        spec = V2.get_vehicle_v2("blueboat_v2")
        d = DATA.BLUEBOAT
        lo, hi = d["bounds_com_m"]
        com_h = d["com_hull_frame_m"]
        xs = sorted({p.x_m for p in points})
        self.assertEqual(xs, sorted([lo[0], 0.5 * (lo[0] + hi[0]), hi[0]]))
        ys = sorted({round(p.y_m, 12) for p in points})
        outer, inner = hi[1] + com_h[1], 2 * V2.BB_HALF_SPACING_M - (hi[1] + com_h[1])
        expected = sorted(round(s * e - com_h[1], 12) for s in (1, -1) for e in (outer, inner))
        self.assertEqual(ys, expected)
        self.assertAlmostEqual(outer, 0.44936, places=5)
        for p in points:
            self.assertAlmostEqual(p.z_m, spec.hydrostatics.freeboard_m - spec.hydrostatics.com_height_above_water_m,
                                   places=15)
            self.assertAlmostEqual(p.calm_freeboard_m, spec.hydrostatics.freeboard_m, places=15)
        self.assertEqual(len({p.label for p in points}), 12)

    def test_wamv_pontoon_top_points(self):
        # G28: the pontoon top line is the WAM-V report-only set (pontoon_top_points); the binding WAM-V set is the
        # deck edge (tests_g28/test_g28_green_water_probe.py)
        for vessel, config in (("wamv_vrx_v2_opt1", "competition"), ("wamv_vrx_v2_opt1_bare", "bare")):
            points = GW.pontoon_top_points(vessel)
            self.assertEqual(len(points), 10)
            com_b = DATA.WAMV[config]["com_base_link_m"]
            spec = V2.get_vehicle_v2(vessel)
            xs = sorted({round(p.x_m + com_b[0], 12) for p in points})
            self.assertEqual(xs, [round(-0.4 + d, 12) for d in (-2.45, -1.225, 0.0, 1.225, 2.45)])
            self.assertEqual(sorted({round(p.y_m + com_b[1], 12) for p in points}), [-1.03, 1.03])
            for p in points:
                self.assertAlmostEqual(p.z_m, 0.4 - com_b[2], places=15)
                self.assertAlmostEqual(p.calm_freeboard_m, 0.4 - com_b[2] + spec.hydrostatics.com_height_above_water_m,
                                       places=15)
        competition = GW.pontoon_top_points("wamv_vrx_v2_opt1")[0].calm_freeboard_m
        self.assertAlmostEqual(competition, 0.28937, places=5)    # registry freeboard_m 0.29083 at VRX trim

    def test_points_are_port_starboard_symmetric_about_the_waterplane(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            geometry = GW.wetness_geometry(vessel)
            yc = sum(s.cy for s in geometry.waterplane) / len(geometry.waterplane)
            rel = sorted(round(p[2] - yc, 9) for p in geometry.points)
            self.assertEqual(rel, sorted(-r for r in rel), vessel)

    def test_unknown_vessel_is_refused(self):
        with self.assertRaises(ValueError):
            GW.deck_edge_points("rov")


class Geometry(unittest.TestCase):
    def test_point_positions_include_heave_roll_pitch_yaw(self):
        m = monitor("wamv_vrx_v2_opt1")
        rows = [(1.0, -2.0, 0.3, 0.2, -0.15, 1.1), (0.0, 0.0, -0.1, -0.3, 0.05, -2.5)]
        positions = m.point_positions(poses(rows)).to(torch.float64)
        for n, (x, y, z, r, p, w) in enumerate(rows):
            cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(w), math.sin(w)
            rot = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                   [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                   [-sp, cp * sr, cp * cr]]
            for j, pt in enumerate(m.points):
                body = (pt.x_m, pt.y_m, pt.z_m)
                expected = [c + sum(rot[a][b] * body[b] for b in range(3)) for a, c in enumerate((x, y, z))]
                for a in range(3):
                    self.assertAlmostEqual(float(positions[n, j, a]), expected[a], places=5)

    def test_elevation_is_bit_identical_to_spectral_waves_sample(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            cfg = G.journal_cfg(vessel, tokens=["waves=D4"])
            env = G.FakeJournalEnv(cfg, 2)
            env.reset()
            for _ in range(3):
                env.step(torch.tensor([[0.5, 0.2], [0.1, -0.3]]))
            waves = env._conditions.wave_source()
            gw = env._green_water
            data = env.robot.data
            data.root_link_pose_w[:, 3:7] = poses([(0, 0, 0, 0.1, -0.05, 0.7), (0, 0, 0, -0.2, 0.1, -1.9)])[:, 3:7]
            pose = data.root_link_pose_w
            origins = env.scene.env_origins
            time_s = env._hydro_v2.time()
            positions = gw.point_positions(pose)
            spec = env._hydro_v2.profile.hydro
            yaw = torch.zeros(2)
            mine = gw.surface_elevations(waves, positions, origins, time_s)          # one [N,P,M] pass
            gw.vector_limit = 0
            looped = gw.surface_elevations(waves, positions, origins, time_s)        # one [N,M] pass per point
            for j in range(positions.shape[1]):
                world = positions[:, j, :2].contiguous()
                ref = waves.sample(world, origins[:, :2].to(world.dtype), time_s, yaw, spec, footprint=False)
                self.assertTrue(torch.equal(mine[:, j], ref.elevation), (vessel, j))
                self.assertTrue(torch.equal(looped[:, j], ref.elevation), (vessel, j))
            self.assertGreater(float(mine.abs().max()), 0.01)


def submerge(m, rows_seq, origins=ORIGINS, level=0.0):
    out = []
    for rows in rows_seq:
        out.append(m.observe(poses(rows), origins, level, torch.zeros(len(rows)), None))
    return out


DRY = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)          # COM 1 m above water: every point dry
WET = (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)         # COM 1 m below water: every point submerged


class CounterAndNRule(unittest.TestCase):
    def test_five_substeps_do_not_trigger_six_do(self):
        self.assertEqual(GW.N_CONSECUTIVE_SUBSTEPS, 6)
        m = monitor()
        # env 0: 5 consecutive wet substeps; env 1: 6 (ticks 3..8)
        seq = [[DRY, DRY]] * 3 + [[WET, WET]] * 5 + [[DRY, WET]] + [[DRY, DRY]] * 2
        submerge(m, seq)
        self.assertEqual(m.green.tolist(), [False, True])
        self.assertEqual(m.longest.tolist(), [5, 6])
        record0, record1 = m.episode_record(0), m.episode_record(1)
        self.assertEqual(record0["green_water"], False)
        self.assertIsNone(record0["first_time_s"])
        self.assertEqual(record0["submerged_point_substeps"], 5 * 12)
        self.assertEqual(record1["first_time_s"], 3 * DT)
        self.assertEqual(record1["first_point"], m.labels[0])
        for record in (record0, record1):
            self.assertEqual(RULE.record_errors(record, 6, DT, m.labels), [])

    def test_onset_time_and_point(self):
        m = monitor()
        seq = [[DRY, DRY]] * 4 + [[WET, DRY]] * 6 + [[DRY, DRY]] * 3
        submerge(m, seq)
        record = m.episode_record(0, runtime_ticks=13)
        self.assertEqual(record, {"green_water": True, "first_time_s": 4 * DT, "first_point": m.labels[0],
                                  "submerged_point_substeps": 6 * 12, "longest_run_substeps": 6, "substeps": 13})
        self.assertEqual(RULE.record_errors(record, 6, DT, m.labels), [])
        self.assertEqual(m.episode_record(1)["green_water"], False)

    def test_single_substep_flicker_never_triggers(self):
        m = monitor()
        seq = [[WET, DRY] if k % 2 == 0 else [DRY, DRY] for k in range(240)]
        submerge(m, seq)
        self.assertEqual(m.green.tolist(), [False, False])
        self.assertEqual(m.longest.tolist(), [1, 0])
        self.assertEqual(m.total.tolist(), [120 * 12, 0])

    def test_five_on_one_off_never_triggers(self):
        m = monitor()
        seq = [[WET, WET] if k % 6 != 5 else [DRY, DRY] for k in range(240)]
        submerge(m, seq)
        self.assertEqual(m.green.tolist(), [False, False])
        self.assertEqual(m.longest.tolist(), [5, 5])

    def test_threshold_noise_around_the_surface_never_triggers(self):
        # a point hovering at the waterline with sub-millimetre jitter that alternates sign every substep
        m = monitor("wamv_vrx_v2_opt1")
        z0 = -GW.deck_edge_points("wamv_vrx_v2_opt1")[0].z_m
        seq = [[(0.0, 0.0, z0 + (5e-4 if k % 2 else -5e-4), 0.0, 0.0, 0.0), DRY] for k in range(120)]
        submerge(m, seq)
        self.assertFalse(bool(m.green.any()))
        self.assertEqual(int(m.longest[0]), 1)

    def test_partial_roll_immersion_names_a_port_point(self):
        m = monitor()
        com_z = 0.007006873572833999
        # heel to port (negative roll lowers +y): 20 deg puts only the port OUTER deck edge under water
        seq = [[(0.0, 0.0, com_z, -math.radians(20.0), 0.0, 0.0), DRY]] * 8
        submerge(m, seq)
        record = m.episode_record(0)
        self.assertTrue(record["green_water"])
        self.assertTrue(record["first_point"].startswith("port_outer"), record)
        self.assertEqual(record["first_time_s"], 0.0)
        wet = m.run[0].tolist()
        for label, count in zip(m.labels, wet):
            self.assertEqual(count > 0, label.startswith("port_outer"), (label, count))

    def test_water_level_and_env_origin_are_respected(self):
        m = monitor()
        origins = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
        seq = [[(0.0, 0.0, 5.5, 0.0, 0.0, 0.0), (0.0, 0.0, 0.5, 0.0, 0.0, 0.0)]] * 7
        submerge(m, seq, origins=origins, level=0.0)
        self.assertEqual(m.green.tolist(), [False, False])
        submerge(m, seq, origins=origins, level=torch.tensor([1.0, 0.0]))
        self.assertEqual(m.green.tolist(), [True, False])


class ResetAndRecords(unittest.TestCase):
    def test_reset_row_isolation(self):
        m = monitor()
        submerge(m, [[WET, WET]] * 7 + [[DRY, WET]] * 2)
        before = {k: getattr(m, k)[1].clone() for k in ("run", "longest", "total", "green", "first_tick",
                                                         "first_point", "ticks")}
        m.reset(torch.tensor([0]))
        self.assertEqual(m.episode_record(0), {"green_water": False, "first_time_s": None, "first_point": None,
                                               "submerged_point_substeps": 0, "longest_run_substeps": 0,
                                               "substeps": 0})
        for key, value in before.items():
            self.assertTrue(torch.equal(getattr(m, key)[1], value), key)
        m.reset(torch.tensor([], dtype=torch.long))
        self.assertTrue(torch.equal(m.ticks, torch.tensor([0, 9])))

    def test_runtime_tick_cross_check(self):
        m = monitor()
        submerge(m, [[DRY, DRY]] * 3)
        with self.assertRaises(RuntimeError):
            m.episode_record(0, runtime_ticks=4)
        self.assertEqual(m.episode_record(0, runtime_ticks=3)["substeps"], 3)

    def test_header_block(self):
        header = monitor().header()
        self.assertEqual(header["schema"], "usvbench.green_water_header/1")
        self.assertEqual(header["rule"]["formula_limit"], 0.05)
        self.assertEqual(header["rule"]["share_limit"], 0.05)
        self.assertEqual(header["detection"]["n_consecutive_substeps"], 6)
        self.assertAlmostEqual(header["detection"]["min_duration_s"], 0.05)
        self.assertEqual(len(header["detection"]["points"]), 12)
        self.assertEqual(header["detection"]["version"], "usvbench.green_water.detect/1")


class NoHostSync(unittest.TestCase):
    def test_per_substep_path_has_no_host_reads(self):
        forbidden = {"item", "tolist", "cpu", "numpy"}
        builtins = {"bool", "int", "float"}
        for name in ("observe", "point_positions", "point_elevation", "surface_elevations", "wave_terms"):
            source = textwrap.dedent(inspect.getsource(getattr(GW.GreenWaterMonitor, name)))
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Attribute):
                        self.assertNotIn(func.attr, forbidden, (name, ast.unparse(node)))
                    if isinstance(func, ast.Name):
                        self.assertNotIn(func.id, builtins, (name, ast.unparse(node)))
                if isinstance(node, (ast.If, ast.While)):
                    # Python branching only on Python values (waves is None, tensor SHAPES), never on a tensor
                    self.assertIn(ast.unparse(node.test),
                                  ("waves is None", "n * p * waves.amplitude.shape[-1] <= self.vector_limit"),
                                  (name, ast.unparse(node.test)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
