"""CPU tests for vehicles_v2 (run: python -B test_vehicles_v2.py from this folder).

No Isaac import. The cross-task contract tests import the IN-TREE hydro_v2 /
thruster_mixer (tasks/_shared, formerly D3) and wind (tasks/_shared, formerly
C3) modules by path. Independent derivations below re-type the VRX link data
with file:line citations and recompute with vessel_physics_ref_v2, so the
xacro expander in assets_v2/tools/build_vessels_v2.py is checked against a
second code path.

E2 integration changes (D5 original: D5 work/repo/test_vehicles_v2.py):
  * module paths: the registry lives in tasks/_shared, this file in
    tasks/_shared/tests_vessels_v2, the assets in <repo>/assets_v2;
  * D3/C3 contract tests import the in-tree modules instead of absolute
    task-folder paths;
  * the legacy-registry byte comparison uses the J-base vehicles.py (the
    in-tree baseline, C2-modified) instead of the original tree's;
  * the HydroSpec contract now also leaves the two decision-7 Froude-Krylov
    fields to their defaults (see test_d3_hydrospec_and_thrusterspec_accept_profiles).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SHARED = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(SHARED))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v = _load("vehicles_v2_under_test", SHARED / "vehicles_v2.py")
ref = _load("vessel_physics_ref_v2_under_test", SHARED / "vessel_physics_ref_v2.py")
D3_SHARED = SHARED
C3_WORK = SHARED
ASSETS = REPO / "assets_v2"
# Baseline of the legacy registry this tree must not modify (J-base copy);
# override with USVBENCH_JBASE_ROOT=<J-base repo root>.
JBASE_VEHICLES = Path(os.environ.get(
    "USVBENCH_JBASE_ROOT",
    r"D:\claude code\codex_tasks\20260930_d1_integrate_jbase\work\repo")) / "tasks" / "_shared" / "vehicles.py"
NAMES = {"blueboat_v2", "wamv_vrx_v2_opt1", "wamv_vrx_v2_opt2", "wamv_vrx_v2_opt1_bare", "wamv_vrx_v2_opt2_bare"}


def diag(a, b, c):
    return ((a, 0.0, 0.0), (0.0, b, 0.0), (0.0, 0.0, c))


def rotated(R, I):
    return ref.rotate_inertia(R, I)


def rpy(r, p, y):
    return ref.rpy_matrix(r, p, y)


def wamv_parts(with_sensors: bool):
    """Independent re-typing of the VRX 7609d1bd links (mass, COM in base_link, tensor)."""
    parts = [(180.0, (0.0, 0.0, 0.0), diag(120.0, 393.0, 446.0))]                 # wamv_base.urdf.xacro:119-123
    for y in (1.0, -1.0):                                                            # wamv_gazebo.urdf.xacro:44-45
        parts.append((23.5, (0.0, y, 0.45 + 0.15),                                   # battery.xacro:17-27
                      diag(0.5015291666666667, 0.8931958333333333, 1.0183333333333333)))
    for y in (1.027135, -1.027135):                                                  # wamv_aft_thrusters.xacro:4-5
        parts.append((15.0, (-2.373776, y, 0.318237), diag(0.889245, 0.911125, 0.078125)))       # engine.xacro:23-26
        parts.append((0.5, (-2.373776 - 0.278156, y, 0.318237 - 0.509371),                          # engine.xacro:59
                      diag(0.008545, 0.008545, 0.0144)))                                           # engine.xacro:41-44
    if not with_sensors:
        return parts
    parts.append((20.0, (-0.15, 0.0, 1.53), diag(1.51633333333333, 1.93633333333333, 2.71633333333333)))  # cpu_cases.xacro:23-32; top:160
    arm_rpy = (math.radians(-60) - math.pi / 20, 0.0, -math.pi / 2)                  # wamv_camera.xacro:96
    arm_I = diag(0.00003680833, 0.00003680833, 0.00000320000)

    def post(mass_per_m, length, r):
        m = mass_per_m * length
        return m, diag(m / 12 * (3 * r * r + length * length), m / 12 * (3 * r * r + length * length), m / 2 * r * r)

    def sensor(x, y, z, P, Y, post_Y, sensor_mass, sensor_I, to_arm_x, arm_to_sensor_x, arm_to_sensor_z):
        platform = 1.2965
        length = z - platform - arm_to_sensor_z
        pm, pI = post(0.9, length, 0.0076)
        post_z = platform + length / 2
        Rz = rpy(0.0, 0.0, post_Y)
        c, s = math.cos(post_Y), math.sin(post_Y)
        post_p = (x - (to_arm_x + arm_to_sensor_x) * c, y - (to_arm_x + arm_to_sensor_x) * s, post_z)
        arm_p = ref.add(post_p, ref.mat_vec(Rz, (to_arm_x, 0.0, z - post_z - arm_to_sensor_z)))
        sen_p = ref.add(arm_p, ref.mat_vec(Rz, (arm_to_sensor_x, 0.0, arm_to_sensor_z)))
        R_sen = ref.mat_mul(Rz, rpy(0.0, P, Y - post_Y))
        R_arm_i = ref.mat_mul(Rz, rpy(*arm_rpy))
        return [(pm, post_p, rotated(Rz, pI)), (0.1, arm_p, rotated(R_arm_i, arm_I)),
                (sensor_mass, sen_p, rotated(R_sen, sensor_I))]
    cam_I = diag(0.0000450, 0.0001746, 0.0001746)                                    # wamv_camera.xacro:31-40
    P15 = math.radians(15)
    parts += sensor(0.75, 0.1, 1.5, P15, 0.0, 0.0, 0.3, cam_I, 0.03, 0.02, 0.027)    # top:163
    parts += sensor(0.75, -0.1, 1.5, P15, 0.0, 0.0, 0.3, cam_I, 0.03, 0.02, 0.027)   # top:164
    parts += sensor(0.5, -0.45, 1.5, P15, math.radians(-90), math.radians(-90), 0.3, cam_I, 0.03, 0.02, 0.027)  # top:167
    parts.append((1.0, (-0.85, 0.0, 1.3), diag(0.006458, 0.006458, 0.01125)))        # wamv_gps.xacro:3,23-27; top:170
    parts.append((0.1, (0.3, -0.2, 1.3), diag(0.00000542, 0.00002104, 0.00002604)))  # wamv_imu.xacro:3,18-21; top:173
    parts += sensor(0.7, 0.0, 1.8, 0.0, 0.0, 0.0, 1.0, diag(0.00109375, 0.00109375, 0.00125),
                    0.03, 0.04, 0.05)                                                # lidar.xacro:4-15; wamv_3d_lidar.xacro:33-60
    parts.append((0.1, (0.0, 0.0, 0.0), diag(0.00000542, 0.00002104, 0.00002604)))   # wamv_pinger.xacro:11-22
    base_p = (0.54, 0.30, 1.296)                                                     # top:182
    parts.append((0.5, ref.add(base_p, (-0.02, 0.0, 0.05)), diag(0.00083, 0.00052083, 0.00052083)))  # ball_shooter.xacro:12-16
    R_launch = rpy(0.0, -1.04, 0.0)                                                  # ball_shooter.xacro:57-63
    parts.append((0.5, ref.add(base_p, (-0.0204, 0.0, 0.107877)),
                   rotated(ref.mat_mul(R_launch, rpy(0.0, -1.5708, 0.0)), diag(0.001779167, 0.001779167, 0.000225))))  # :43-47
    return parts


class RegistryTests(unittest.TestCase):
    def test_exact_profile_set_and_lookup(self):
        self.assertEqual(set(v.VEHICLES_V2), NAMES)
        self.assertIn(v.DEFAULT_WAMV_PROFILE, v.VEHICLES_V2)
        for name in NAMES:
            self.assertIs(v.get_vehicle_v2(name), v.VEHICLES_V2[name])
        with self.assertRaisesRegex(ValueError, "wamv_vrx_v2_opt1"):
            v.get_vehicle_v2("wamv")

    def test_all_profiles_validate_with_physical_conventions(self):
        for spec in v.VEHICLES_V2.values():
            v.validate_vehicle(spec)
            self.assertEqual((spec.scale, spec.bow_body_axis, spec.com_actor_m), (1.0, "+x", (0.0, 0.0, 0.0)))
            self.assertAlmostEqual(spec.hydrostatics.displaced_volume_m3 * 1000.0, spec.mass_kg, places=9)

    def test_immutability(self):
        spec = v.get_vehicle_v2("blueboat_v2")
        with self.assertRaises(TypeError):
            v.VEHICLES_V2["x"] = spec
        with self.assertRaises(FrozenInstanceError):
            spec.mass_kg = 1.0
        with self.assertRaises(FrozenInstanceError):
            spec.propulsion.motors[0].forward_limit_n = 1.0
        with self.assertRaises(TypeError):
            spec.inertia_com_kg_m2[0][0] = 1.0

    def test_options_share_geometry_and_differ_only_in_dynamics(self):
        for config in ("", "_bare"):
            a, b = v.get_vehicle_v2("wamv_vrx_v2_opt1" + config), v.get_vehicle_v2("wamv_vrx_v2_opt2" + config)
            for f in ("mass_kg", "inertia_com_kg_m2", "hydrostatics", "windage", "collision", "asset_sha256"):
                self.assertEqual(getattr(a, f), getattr(b, f), f)
            self.assertNotEqual(a.damping, b.damping)
            self.assertNotEqual(a.propulsion.motors[0].forward_limit_n, b.propulsion.motors[0].forward_limit_n)

    def test_legacy_registry_untouched(self):
        legacy = _load("vehicles_legacy_copy", SHARED / "vehicles.py")
        self.assertEqual(set(legacy.VEHICLES), {"rov", "wamv", "blueboat"})
        if JBASE_VEHICLES.is_file():
            self.assertEqual((SHARED / "vehicles.py").read_bytes(), JBASE_VEHICLES.read_bytes())

    def test_sources_are_ascii(self):
        for name in ("vehicles_v2.py", "vehicles_v2_data.py", "vessel_physics_ref_v2.py"):
            (SHARED / name).read_bytes().decode("ascii")
        (HERE / "test_vehicles_v2.py").read_bytes().decode("ascii")


class WamvDerivationTests(unittest.TestCase):
    def check(self, config, with_sensors, spec_name):
        mass, com, I = ref.aggregate_inertia(wamv_parts(with_sensors))
        d = v._data.WAMV[config]
        self.assertAlmostEqual(mass, d["mass_kg"], places=9)
        for a, b in zip(com, d["com_base_link_m"]):
            self.assertAlmostEqual(a, b, places=9)
        spec = v.get_vehicle_v2(spec_name)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(I[i][j], spec.inertia_com_kg_m2[i][j], places=7)
        return mass, com, I

    def test_bare_platform_258kg(self):
        mass, com, I = self.check("bare", False, "wamv_vrx_v2_opt1_bare")
        self.assertEqual(mass, 258.0)
        # report 4.2 row C: (-0.2863, 0, 0.1456), 217.03/566.08/682.86, Ixz 11.40
        self.assertAlmostEqual(com[0], -0.2863, places=4)
        self.assertAlmostEqual(com[2], 0.1456, places=4)
        for got, want in ((I[0][0], 217.03), (I[1][1], 566.08), (I[2][2], 682.86), (I[0][2], 11.40)):
            self.assertAlmostEqual(got, want, delta=0.01)

    def test_competition_283kg(self):
        mass, com, I = self.check("competition", True, "wamv_vrx_v2_opt1")
        # report 4.2 row D approximated sensor positions: 283.385 kg, 262.44/615.88/689.98, Ixz 3.60
        self.assertAlmostEqual(mass, 283.385, delta=0.001)
        for got, want in ((I[0][0], 262.44), (I[1][1], 615.88), (I[2][2], 689.98), (I[0][2], 3.60)):
            self.assertLess(abs(got - want) / want, 3e-3)

    def test_old_211kg_assembly_matches_0912_injection(self):
        parts = [p for p in wamv_parts(False) if p[0] != 23.5]
        mass, com, I = ref.aggregate_inertia(parts)
        self.assertEqual(mass, 211.0)
        # vehicle_profile.json of 2026-09-12: COM (-0.35007, 0, 0.04434); I scaled by 0.8^2*100/211
        self.assertAlmostEqual(com[0], -0.350072095, places=8)
        s = 0.8 ** 2 * 100 / 211
        self.assertAlmostEqual(I[0][0] * s, 47.6695952882, places=6)
        self.assertAlmostEqual(I[0][2] * s, 5.72680495905, places=6)


class HydrostaticsTests(unittest.TestCase):
    POINTS = ((0.6, 1.03, 0.0), (-1.4, 1.03, 0.0), (0.6, -1.03, 0.0), (-1.4, -1.03, 0.0))  # dynamics xacro:15-16,32-33

    def solve(self, config, g):
        d = v._data.WAMV[config]
        return ref.vrx_surface_hydrostatics(mass=d["mass_kg"], com_body=d["com_base_link_m"],
                                            points_body=self.POINTS, hull_length=4.9, hull_radius=0.213,
                                            rho=1000.0, g=g)

    def test_resolve_matches_registry(self):
        for config, name in (("competition", "wamv_vrx_v2_opt1"), ("bare", "wamv_vrx_v2_opt2_bare")):
            h = self.solve(config, 9.81)
            s = v.get_vehicle_v2(name).hydrostatics
            self.assertAlmostEqual(h["stiffness_heave_n_m"], s.stiffness_heave_n_m, delta=1e-3)
            self.assertAlmostEqual(h["stiffness_roll_nm_rad"], s.stiffness_roll_nm_rad, delta=1e-3)
            self.assertAlmostEqual(h["stiffness_pitch_nm_rad"], s.stiffness_pitch_nm_rad, delta=1e-3)
            self.assertAlmostEqual(h["com_world_z_m"], s.com_height_above_water_m, places=9)
            self.assertLess(abs(h["residual_force_n"]), 1e-6)

    def test_report_table_5_3_at_vrx_gravity(self):
        # report 5.3, J layout, rho 1000, g 9.8: C 258 kg and D 283.4 kg
        for config, kz, kr, kp, h0 in (("bare", 34891, 36648, 34755, 0.1055),
                                       ("competition", 35648, 37074, 35250, 0.1134)):
            h = self.solve(config, 9.8)
            self.assertLess(abs(h["stiffness_heave_n_m"] / kz - 1), 1e-3)
            self.assertLess(abs(h["stiffness_roll_nm_rad"] / kr - 1), 1e-3)
            self.assertLess(abs(h["stiffness_pitch_nm_rad"] / kp - 1), 1e-3)
            self.assertAlmostEqual(-h["origin_world_z_m"], h0, delta=6e-4)

    def test_closed_form_tangent_stiffness(self):
        h = self.solve("competition", 9.81)
        d = v._data.WAMV["competition"]
        k = [1000 * 9.81 * 2.45 * ref.circle_segment_chord(0.213, depth) for depth in h["point_depths_m"]]
        self.assertAlmostEqual(sum(k), h["stiffness_heave_n_m"], delta=0.05)
        # roll: sum k y^2 - m g z_G (z_G = COM above the point plane); exact solve includes trim
        kroll = sum(ki * p[1] ** 2 for ki, p in zip(k, self.POINTS)) - d["mass_kg"] * 9.81 * d["com_base_link_m"][2]
        self.assertLess(abs(kroll / h["stiffness_roll_nm_rad"] - 1), 2e-3)
        self.assertAlmostEqual(h["maximum_volume_m3"] * 1000, 698.4, delta=0.1)     # report 5.3 buoyancy cap

    def test_blueboat_hydrostatics_consistency(self):
        s = v.get_vehicle_v2("blueboat_v2")
        h, d = s.hydrostatics, v._data.BLUEBOAT
        self.assertAlmostEqual(h.stiffness_roll_nm_rad, 278.16, delta=0.01)   # legacy 280 came from the same CAD
        self.assertAlmostEqual(h.stiffness_pitch_nm_rad, 139.77, delta=0.01)  # legacy 141
        self.assertAlmostEqual(d["gm_roll_m"], d["kb_m"] + 0.0295466 / 0.01726 - d["kg_m"], delta=2e-6)
        # published freeboard 160 mm at 14.5 kg + 2.4 kg batteries (store page)
        self.assertLess(abs(d["freeboard_at_16p9_kg_m"] - 0.160), 0.010)
        self.assertAlmostEqual(h.static_roll_rad, s.mass_kg * 9.81 * 0.047 / h.stiffness_roll_nm_rad, delta=1e-8)


class PropulsionTests(unittest.TestCase):
    def test_classic_glf_matches_reference_and_report(self):
        for c in [i / 200 - 1 for i in range(401)]:
            self.assertEqual(v.vrx_classic_thrust(c), ref.classic_glf_thrust(c))
        for c, t in ((1.0, 242.67), (0.75, 181.9), (0.5, 80.6), (0.25, 14.3), (-0.25, -1.9),
                     (-0.5, -15.3), (-0.75, -56.4), (-1.0, -100.0)):
            self.assertAlmostEqual(v.vrx_classic_thrust(c), t, delta=0.06)

    def test_classic_knot_table_error(self):
        motor = v.get_vehicle_v2("wamv_vrx_v2_opt1").propulsion.motors[0]
        worst = 0.0
        for i in range(2001):
            c = -1 + i / 1000
            if 0.0 <= c <= 0.0125:
                continue
            worst = max(worst, abs(v.command_to_thrust(motor, c) - v.vrx_classic_thrust(c)))
        self.assertLess(worst, 2.0)          # N, piecewise-linear sampling at 0.025

    def test_limits(self):
        j = (51.3 + 72.4 * 7.71667) * 7.71667 / 2
        self.assertAlmostEqual(j, 2353.5338, places=3)
        o1, o2 = v.get_vehicle_v2("wamv_vrx_v2_opt1"), v.get_vehicle_v2("wamv_vrx_v2_opt2")
        self.assertEqual(o2.propulsion.motors[0].forward_limit_n, j)
        self.assertEqual(o2.propulsion.motors[0].reverse_limit_n, 1000.0)
        self.assertAlmostEqual(o1.propulsion.motors[0].forward_limit_n, 242.6659, places=3)
        self.assertEqual(o1.propulsion.motors[0].reverse_limit_n, 100.0)
        self.assertAlmostEqual(o1.propulsion.yaw_command_limit_nm, 352.0, delta=0.1)   # report 9.2
        self.assertAlmostEqual(o2.propulsion.yaw_command_limit_nm, 3444.5, delta=0.1)
        b = v.get_vehicle_v2("blueboat_v2").propulsion
        self.assertAlmostEqual(b.motors[0].forward_limit_n, 8.2 * 9.80665 / 2)
        self.assertAlmostEqual(b.motors[0].reverse_limit_n / b.motors[0].forward_limit_n, 2.81 / 5.63)

    def test_report_performance_table_6_5(self):
        cases = (("wamv_vrx_v2_opt1", 2.26, 0.88), ("wamv_vrx_v2_opt2", 5.28, 1.63))
        for name, u, r in cases:
            s = v.get_vehicle_v2(name)
            self.assertAlmostEqual(s.speeds.top_model_m_s, u, delta=0.01)
            n = s.propulsion.yaw_command_limit_nm
            rate = v.terminal_speed(s.damping.linear[5], s.damping.quadratic[5], n)
            self.assertAlmostEqual(rate, r, delta=0.01)

    def test_command_roundtrip(self):
        for spec in v.VEHICLES_V2.values():
            m = spec.propulsion.motors[0]
            for frac in (-1.0, -0.6, -0.2, -0.01, 0.0, 0.01, 0.3, 0.7, 1.0):
                t = frac * (m.forward_limit_n if frac > 0 else m.reverse_limit_n)
                self.assertAlmostEqual(v.command_to_thrust(m, v.thrust_to_command(m, t)), t, delta=1e-9 + 1e-9 * abs(t))
            self.assertEqual(v.command_to_thrust(m, 1e9), m.forward_limit_n)
            self.assertEqual(v.command_to_thrust(m, -1e9), -m.reverse_limit_n)

    def test_allocation_envelope(self):
        for spec in v.VEHICLES_V2.values():
            p = spec.propulsion
            fwd, rev = p.motors[0].forward_limit_n, p.motors[0].reverse_limit_n
            full = v.allocate_planar(spec, 2 * fwd, p.yaw_command_limit_nm)
            self.assertTrue(full.saturated)
            for t in full.thrusts_n:
                self.assertLessEqual(t, fwd + 1e-9)
                self.assertGreaterEqual(t, -rev - 1e-9)
            # direction preserved by radial saturation
            self.assertAlmostEqual(full.wrench[5] / full.wrench[0], p.yaw_command_limit_nm / (2 * fwd), places=9)
            pure = v.allocate_planar(spec, 0.0, 10 * p.yaw_command_limit_nm)
            self.assertAlmostEqual(pure.wrench[5], p.pure_yaw_limit_nm, places=6)
            self.assertAlmostEqual(pure.wrench[0], 0.0, places=9)
            best = v.allocate_planar(spec, fwd - rev, p.yaw_command_limit_nm)
            self.assertFalse(best.saturated)
            self.assertAlmostEqual(best.wrench[5], p.yaw_command_limit_nm, places=6)

    def test_offcentre_com_yaw_bias_is_excluded(self):
        s = v.get_vehicle_v2("blueboat_v2")
        f = s.propulsion.motors[0].forward_limit_n
        centre = v.motor_wrench(s, (f, f))
        literal = v.motor_wrench(s, (f, f), reference="com_geometric")
        self.assertEqual(centre[5], 0.0)
        self.assertAlmostEqual(literal[5], -2 * f * 0.047, delta=1e-6)   # 3.78 N m spurious yaw
        self.assertLess(centre[4], 1e-12)
        self.assertAlmostEqual(v.motor_wrench(s, (f, f), pitch_coupling=True)[4],
                               2 * f * s.propulsion.thrust_line_height_com_m)

    def test_motor_lag(self):
        m = v.get_vehicle_v2("blueboat_v2").propulsion.motors[0]
        self.assertEqual(v.motor_lag_step(m, 0.0, 10.0, 1 / 120), 10.0)
        lag = replace(m, time_constant_s=0.1)
        self.assertAlmostEqual(v.motor_lag_step(lag, 0.0, 10.0, 0.1), 10.0 * (1 - math.exp(-1)))


class WindTests(unittest.TestCase):
    def test_log_profile(self):
        self.assertAlmostEqual(v.wind_speed_at_height(8.0, 10.0), 8.0)
        self.assertLess(v.wind_speed_at_height(8.0, 0.5), v.wind_speed_at_height(8.0, 2.0))
        with self.assertRaises(ValueError):
            v.wind_speed_at_height(8.0, 1e-5)
        for spec in v.VEHICLES_V2.values():
            ratio = v.vessel_wind_speed_from_u10(spec, 1.0)
            self.assertTrue(0.5 < ratio < 0.8, (spec.name, ratio))
            kw = v.to_c3_windage_kwargs(spec)
            self.assertAlmostEqual(kw["centroid_height_m"], spec.windage.lateral_centroid_height_above_water_m
                                   - spec.hydrostatics.com_height_above_water_m, places=12)

    def test_windage_order_and_bounds(self):
        comp, bare = v.get_vehicle_v2("wamv_vrx_v2_opt1").windage, v.get_vehicle_v2("wamv_vrx_v2_opt1_bare").windage
        self.assertGreater(comp.frontal_area_m2, bare.frontal_area_m2)
        self.assertGreater(comp.lateral_area_m2, bare.lateral_area_m2)
        b = v._data.BLUEBOAT
        self.assertTrue(b["hull_frontal_area_m2"] < b["frontal_area_m2"] < b["upper_bound_frontal_area_m2"])
        self.assertTrue(b["hull_lateral_area_m2"] < b["lateral_area_m2"] < b["upper_bound_lateral_area_m2"])


class ValidationRejectsTests(unittest.TestCase):
    def setUp(self):
        self.spec = v.get_vehicle_v2("wamv_vrx_v2_opt1")

    def bad(self, **changes):
        with self.assertRaises(ValueError):
            v.validate_vehicle(replace(self.spec, **changes))

    def test_rejections(self):
        s = self.spec
        self.bad(scale=0.8)
        self.bad(com_actor_m=(0.1, 0.0, 0.0))
        self.bad(inertia_com_kg_m2=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 3.0)))       # triangle violation
        self.bad(inertia_com_kg_m2=((1.0, 0.1, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))       # asymmetric
        self.bad(damping=replace(s.damping, linear=(0.0,) + s.damping.linear[1:], quadratic=(0.0,) + s.damping.quadratic[1:]))
        self.bad(hydrostatics=replace(s.hydrostatics, stiffness_roll_nm_rad=1.0))
        self.bad(hydrostatics=replace(s.hydrostatics, maximum_volume_m3=0.1))
        self.bad(added_mass=replace(s.added_mass, active_in_v2=True))
        m = s.propulsion.motors[0]
        bad_motor = replace(m, command_to_thrust=((-1.0, -100.0), (0.0, 0.0), (0.5, 300.0), (1.0, 242.6658667518538)))
        self.bad(propulsion=replace(s.propulsion, motors=(bad_motor, s.propulsion.motors[1])))
        self.bad(propulsion=replace(s.propulsion, yaw_command_limit_nm=1.0))
        self.bad(windage=replace(s.windage, roll_lever_com_m=s.windage.lateral_centroid_height_above_water_m))
        self.bad(collision=replace(s.collision, capsule_radius_m=0.5))
        self.bad(speeds=replace(s.speeds, cruise_m_s=10.0))
        self.bad(provenance=s.provenance[:-1])


class ContractTests(unittest.TestCase):
    @unittest.skipUnless((D3_SHARED / "hydro_v2.py").is_file(), "in-tree hydro_v2.py not present")
    def test_d3_hydrospec_and_thrusterspec_accept_profiles(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        sys.path.insert(0, str(D3_SHARED))
        try:
            hydro = _load("d3_hydro_v2", D3_SHARED / "hydro_v2.py")
            mixer = _load("d3_thruster_mixer", D3_SHARED / "thruster_mixer.py")
        finally:
            sys.path.remove(str(D3_SHARED))
        for spec in v.VEHICLES_V2.values():
            h = hydro.HydroSpec(**v.to_d3_hydro_kwargs(spec))
            self.assertAlmostEqual(h.heave_stiffness_n_per_m, spec.hydrostatics.stiffness_heave_n_m, places=6)
            t = mixer.ThrusterSpec(**v.to_d3_thruster_kwargs(spec))
            self.assertEqual(t.arm_m, spec.propulsion.planar_arm_m)
            mixer.ThrusterSpec(**v.to_d3_thruster_kwargs(spec, pitch_coupling=True))
        # E2: decision 7 adds froude_krylov_horizontal (default True) and
        # added_mass_coefficient_horizontal (default 0, nonzero rejected);
        # like rotational_relative_motion they are model switches, not vessel
        # numbers, so the registry adapter leaves them to the defaults.
        self.assertEqual({f.name for f in fields(hydro.HydroSpec)} - set(v.to_d3_hydro_kwargs(spec)),
                         {"rotational_relative_motion", "froude_krylov_horizontal",
                          "added_mass_coefficient_horizontal"})

    @unittest.skipUnless((C3_WORK / "wind.py").is_file(), "in-tree wind.py not present")
    def test_c3_vessel_windage_accepts_profiles(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        wind = _load("c3_wind", C3_WORK / "wind.py")
        for spec in v.VEHICLES_V2.values():
            w = wind.VesselWindage(**v.to_c3_windage_kwargs(spec))
            self.assertEqual(w.lateral_area_m2, spec.windage.lateral_area_m2)

    @unittest.skipUnless(ASSETS.is_dir(), "assets folder not present")
    def test_asset_hashes(self):
        for spec in v.VEHICLES_V2.values():
            path = ASSETS / spec.asset_relpath
            self.assertTrue(path.is_file(), path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), spec.asset_sha256)


class ReferenceModuleTests(unittest.TestCase):
    def test_aggregate_parallel_axis(self):
        m, c, I = ref.aggregate_inertia([(1.0, (1.0, 0.0, 0.0), diag(0, 0, 0)), (1.0, (-1.0, 0.0, 0.0), diag(0, 0, 0))])
        self.assertEqual((m, c), (2.0, (0.0, 0.0, 0.0)))
        self.assertEqual((I[0][0], I[1][1], I[2][2]), (0.0, 2.0, 2.0))

    def test_capsule_contains_rectangle(self):
        cap = ref.enclosing_capsule_for_rectangle(-2.5, 2.8, 1.27)
        for x in (-2.5, 2.8):
            for y in (-1.27, 1.27):
                dx = max(0.0, abs(x - cap["center_xy_m"][0]) - cap["segment_half_length_m"])
                self.assertLessEqual(math.hypot(dx, y), cap["radius_m"] + 1e-9)

    def test_explicit_numbers(self):
        n = ref.explicit_damping_numbers(470.0, 0.0, 0.0, 1.094, 1 / 120)
        self.assertAlmostEqual(n["A"], 3.580134, places=5)      # C1 AUDIT_physics.md pitch
        self.assertFalse(n["stable_explicit"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
