"""CPU tests for vessel profiles and the per-env hydro_v2 runtime.

Execute through run_cpu_tests.py (no Isaac import, bytecode disabled).
E2 (decision 1): profiles are the thin registry adapter
hydro_profiles.profile_for_vehicle over the D5 v2 registry
(tasks/_shared/vehicles_v2.py + vehicles_v2_data.py); D3's provisional
builders and their report numbers are superseded. ProfileValues therefore
checks the adapter against the registry and against the independent
sources D5 cites:
  BlueBoat  registry blueboat_v2 (D5 PARAMS.md 7-9: CAD hydrostatics,
            8.2 kgf static thrust, M200 reverse ratio, legacy damping).
  WAM-V     VRX research report (NC-journal materials, 2026-09-30), sections
            4.2 (mass/COM/inertia), 5.1 (damping), 5.3 (hydrostatics),
            6.1/6.3/6.5 (thrusters and derived speeds), 9.5 (acceptance).
"""
import importlib.util
import math
from pathlib import Path
import sys
import tokenize
import unittest
from dataclasses import replace
import torch

sys.dont_write_bytecode = True
SHARED = Path(__file__).resolve().parents[2] / "tasks" / "_shared"
sys.path.insert(0, str(SHARED))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SHARED / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


physics = load("physics_v2")
load("restoring_v2")
hydro = load("hydro_v2")
mixer = load("thruster_mixer")
reference = load("reference_point")
registry = load("vehicles_v2")
journal = load("journal_vehicle")
profiles = load("hydro_profiles")
runtime = load("hydro_v2_runtime")
torch.set_default_dtype(torch.float64)
BLUEBOAT = profiles.profile_for_vehicle("blueboat_v2")
WAMV = profiles.profile_for_vehicle("wamv_vrx_v2_opt1")
WAMV_CURRENT = profiles.profile_for_vehicle("wamv_vrx_v2_opt2")
# G4-A5: VRX-classic sway/yaw comparison profile (the report 5.1 / 6.5 option-1 values).
WAMV_CLASSIC = profiles.profile_for_vehicle("wamv_vrx_v2_opt1", swayyaw_damping="vrx_classic")
DT = 1/120


def terminal_speed(lin, quad, force):
    return lin/(2*quad)*(math.sqrt(1+4*quad*force/lin**2)-1) if quad > 0 else force/lin


class ProfileValues(unittest.TestCase):
    def test_blueboat_profile_matches_sources(self):
        h, t = BLUEBOAT.hydro, BLUEBOAT.thrusters
        spec = registry.get_vehicle_v2("blueboat_v2")
        kwargs = registry.to_d3_hydro_kwargs(spec)
        # Every HydroSpec field except the decision-2 roll/pitch damping and the
        # model switches equals the registry adapter output.
        for key, value in kwargs.items():
            if key != "angular_linear_damping":
                self.assertEqual(getattr(h, key), value, key)
        self.assertAlmostEqual(h.mass_kg, 17.26000023)
        self.assertEqual(tuple(h.inertia_kgm2[i][i] for i in range(3)), (1.7890000343322754,
                         1.093999981880188, 2.375999927520752))  # float32 USD read-back of 1.789/1.094/2.376
        self.assertAlmostEqual(h.roll_stiffness_nm, 278.16, delta=0.01)    # D5 PARAMS.md:178
        self.assertAlmostEqual(h.pitch_stiffness_nm, 139.77, delta=0.01)
        self.assertAlmostEqual(h.heave_stiffness_n_per_m, 1000*9.81*h.waterplane_m2, places=9)
        self.assertEqual(h.linear_damping, (9.1, 27.0, 200.0))
        self.assertEqual(h.quadratic_damping, (5.9, 18.0, 0.0))
        self.assertEqual(h.angular_quadratic_damping, (0.0, 0.0, 6.0))
        self.assertEqual((h.length_m, h.beam_m), (spec.hydrostatics.wave_footprint_length_m,
                                                  spec.hydrostatics.wave_footprint_beam_m))
        self.assertEqual(h.orbital_depth_m, spec.hydrostatics.draft_m/2)
        self.assertEqual(h.equilibrium_com_z_m, spec.hydrostatics.com_height_above_water_m)
        # Decision 2: journal default zeta = 0.1 per axis with the registry K and I.
        for axis, (k, inertia) in enumerate(((h.roll_stiffness_nm, h.inertia_kgm2[0][0]),
                                             (h.pitch_stiffness_nm, h.inertia_kgm2[1][1]))):
            self.assertAlmostEqual(h.angular_linear_damping[axis], 0.2*math.sqrt(k*inertia), places=12)
            self.assertAlmostEqual(h.angular_linear_damping[axis]/(2*math.sqrt(k*inertia)), 0.1, places=12)
        self.assertEqual(h.angular_linear_damping[2], 4.0)
        self.assertEqual((t.arm_m, t.mount_z_m, t.nominal_lag_s), (0.3607, 0.0, (0.0, 0.0)))
        self.assertEqual(BLUEBOAT.com_offset_actor_m, (0.0, 0.0, 0.0))
        self.assertEqual(BLUEBOAT.actor_to_hull, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        self.assertTrue(h.froude_krylov_horizontal)
        self.assertEqual(h.added_mass_coefficient_horizontal, 0.0)
        # Other decision-2 forms: legacy 470, numeric override, explicit zeta.
        self.assertEqual(profiles.profile_for_vehicle("blueboat_v2", rollpitch_damping="legacy_470")
                         .hydro.angular_linear_damping, (470.0, 470.0, 4.0))
        self.assertEqual(profiles.profile_for_vehicle("blueboat_v2", rollpitch_damping=200)
                         .hydro.angular_linear_damping[:2], (200.0, 200.0))
        z = profiles.profile_for_vehicle("blueboat_v2", rollpitch_damping="zeta:0.5").hydro
        self.assertAlmostEqual(z.angular_linear_damping[0], 5*h.angular_linear_damping[0], places=10)
        for bad in ("zeta:0", "zeta:-1", "zeta:x", "legacy", 0, -3.0, float("nan"), True):
            with self.assertRaises(ValueError):
                profiles.profile_for_vehicle("blueboat_v2", rollpitch_damping=bad)

    def test_wamv_profiles_match_report(self):
        h, t = WAMV.hydro, WAMV.thrusters
        self.assertAlmostEqual(h.mass_kg, 283.385, delta=0.001)              # report 4.2 row D
        for i, value in enumerate((262.44, 615.88, 689.98)):                  # report 4.2 (approx. sensor poses)
            self.assertLess(abs(h.inertia_kgm2[i][i]/value-1), 3e-3)
        self.assertAlmostEqual(h.inertia_kgm2[0][2], 3.594, delta=0.01)       # full tensor kept (Ixz)
        # Registry solves the VRX layout at g = 9.81; report 5.3 used 9.8 (35648).
        self.assertLess(abs(h.waterplane_m2*1000*9.8/35648.0-1), 1e-3)
        self.assertAlmostEqual(h.maximum_volume_m3, math.pi*0.213**2*4.9, places=6)
        self.assertLess(h.mass_kg/1000, h.maximum_volume_m3)
        self.assertLess(abs(h.roll_stiffness_nm/37074.0*9.8/9.81-1), 2e-3)
        self.assertLess(abs(h.pitch_stiffness_nm/35250.0*9.8/9.81-1), 2e-3)
        self.assertEqual(h.linear_damping, (51.3, 40.0, 500.0))
        # G4-A5: journal default adds the Sarda 2016 cross-flow sway and strip-theory yaw quadratic
        # terms to the VRX-classic linear ones (values: scripts/tests_hydro_v2/test_swayyaw_damping.py);
        # VRX classic (report 5.1 option 1: sway 40/0, yaw 400/0) is the comparison profile.
        self.assertEqual((h.quadratic_damping[0], h.quadratic_damping[2]), (72.4, 0.0))
        self.assertAlmostEqual(h.quadratic_damping[1], 485.4, delta=0.05)
        self.assertEqual(h.angular_quadratic_damping[:2], (0.0, 0.0))
        self.assertAlmostEqual(h.angular_quadratic_damping[2], 1784.6, delta=0.05)
        # T25 switch (b), owner decision T1.35b: the VRX roll/pitch values stay selectable ("registry")
        classic = profiles.profile_for_vehicle("wamv_vrx_v2_opt1", swayyaw_damping="vrx_classic",
                                               rollpitch_damping="registry").hydro
        self.assertEqual(classic.quadratic_damping, (72.4, 0.0, 0.0))
        self.assertEqual(classic.angular_quadratic_damping, (0.0, 0.0, 0.0))
        self.assertEqual(classic.angular_linear_damping, (50.0, 50.0, 400.0))
        # T1.35b: journal default roll/pitch = zeta 0.1 (2 x 0.1 x sqrt(K I)); yaw unchanged
        self.assertEqual(h.angular_linear_damping[2], 400.0)
        self.assertAlmostEqual(h.angular_linear_damping[0], 0.2 * math.sqrt(h.roll_stiffness_nm * h.inertia_kgm2[0][0]),
                               places=9)
        self.assertAlmostEqual(h.angular_linear_damping[1],
                               0.2 * math.sqrt(h.pitch_stiffness_nm * h.inertia_kgm2[1][1]), places=9)
        self.assertAlmostEqual(h.equilibrium_com_z_m, 0.155, delta=0.005)
        self.assertAlmostEqual(t.forward_limit_n[0], 242.67, delta=0.005)
        self.assertEqual(t.reverse_limit_n, (100.0, 100.0))
        self.assertEqual(t.arm_m, 1.027135)
        self.assertEqual(t.mount_z_m, 0.0)                                   # decision 6
        c = WAMV_CURRENT.hydro
        self.assertEqual(c.linear_damping, (100.0, 100.0, 500.0))
        self.assertEqual(c.quadratic_damping, (150.0, 100.0, 0.0))
        # T1.35b: roll/pitch = zeta 0.1 for every WAM-V option; the option-2 registry values (300, 900) stay
        # selectable as "registry"; yaw 800 unchanged
        self.assertEqual(c.angular_linear_damping[2], 800.0)
        self.assertAlmostEqual(c.angular_linear_damping[1], 0.2 * math.sqrt(c.pitch_stiffness_nm * c.inertia_kgm2[1][1]),
                               places=9)
        self.assertEqual(profiles.profile_for_vehicle(WAMV_CURRENT.name, rollpitch_damping="registry")
                         .hydro.angular_linear_damping, (300.0, 900.0, 800.0))
        self.assertEqual(c.angular_quadratic_damping, (600.0, 900.0, 800.0))
        platform = profiles.profile_for_vehicle("wamv_vrx_v2_opt1_bare").hydro
        self.assertEqual(platform.mass_kg, 258.0)
        self.assertAlmostEqual(platform.equilibrium_com_z_m, 0.040, delta=0.005)
        with self.assertRaises(ValueError):
            profiles.profile_for_vehicle("wamv_vrx_v2_unknown")
        with self.assertRaises(ValueError):                                   # legacy 470 is BlueBoat only
            profiles.profile_for_vehicle("wamv_vrx_v2_opt1", rollpitch_damping="legacy_470")
        zeta = profiles.profile_for_vehicle("wamv_vrx_v2_opt1", rollpitch_damping="zeta:0.1").hydro
        self.assertAlmostEqual(zeta.angular_linear_damping[0],
                               0.2*math.sqrt(zeta.roll_stiffness_nm*zeta.inertia_kgm2[0][0]), places=9)
        for legacy in ("wamv", "blueboat", "rov"):
            with self.assertRaisesRegex(ValueError, "not a journal"):
                profiles.profile_for_vehicle(legacy)
        with self.assertRaises(TypeError):                                    # names only, v2 only
            profiles.profile_for_vehicle(registry.get_vehicle_v2("blueboat_v2"))

    def test_no_vessel_numbers_in_hydro_profiles(self):
        # Decision 1: every number comes from the registry. Only 0, 1, 2 (and
        # their float forms) may appear as numeric literals in the adapter.
        with open(SHARED / "hydro_profiles.py", "rb") as handle:
            numbers = [token.string for token in tokenize.tokenize(handle.readline)
                       if token.type == tokenize.NUMBER]
        self.assertTrue(set(numbers) <= {"0", "1", "2", "0.0", "1.0", "2.0"}, numbers)

    def test_steady_speeds_match_report_section_6_5(self):
        # Surge: quad*u^2+lin*u = total forward thrust. BlueBoat v2: registry
        # top_model_m_s (8.2 kgf, legacy surge drag fitted to the published
        # 3 m/s, so the agreement is circular, D5 PARAMS); WAM-V classic
        # 2.26 m/s, current (J) 5.28 m/s (report 6.5); reverse 1.35 m/s.
        blueboat_top = registry.get_vehicle_v2("blueboat_v2").speeds.top_model_m_s
        cases = ((BLUEBOAT, 8.2*9.80665, blueboat_top), (WAMV, 485.33, 2.26), (WAMV_CURRENT, 4707.07, 5.28))
        for profile, nominal_thrust, expected in cases:
            h = profile.hydro
            thrust = sum(profile.thrusters.forward_limit_n)
            self.assertAlmostEqual(thrust, nominal_thrust, delta=0.01)
            u = terminal_speed(h.linear_damping[0], h.quadratic_damping[0], thrust)
            self.assertAlmostEqual(u, expected, delta=0.006)
            self.assertLess(self.net_surge_force(profile, 1.0, u), 1e-6)
        u_rev = terminal_speed(51.3, 72.4, 200.0)
        self.assertAlmostEqual(u_rev, 1.35, delta=0.006)
        self.assertLess(self.net_surge_force(WAMV, -1.0, -u_rev), 1e-6)

    def net_surge_force(self, profile, action, speed):
        rt = runtime.HydroV2Runtime(profile, 1, "cpu", DT, dtype=torch.float64)
        pos = torch.tensor([[0., 0., profile.hydro.equilibrium_com_z_m]])
        vel = torch.tensor([[speed, 0., 0., 0., 0., 0.]])
        step = rt.step(torch.tensor([[action, 0.]]), pos, vel, torch.tensor([[1., 0, 0, 0]]), torch.zeros(1, 3))
        return abs(float(step.force_hull[0, 0]))

    def test_pure_yaw_steady_rate_uses_reverse_limited_torque(self):
        # Zero-surge spin: torque 2*F_rev*arm; report 6.5 WAM-V classic 0.51 rad/s (VRX-classic
        # comparison profile); G4-A5 journal default (400 + 1784.6 |r|) r = 205.43 N m -> 0.2452 rad/s.
        for profile, expected in ((WAMV_CLASSIC, 0.51), (WAMV, 0.2452), (BLUEBOAT, None)):
            h, t = profile.hydro, profile.thrusters
            torque = 2*t.reverse_limit_n[0]*t.arm_m
            rate = terminal_speed(h.angular_linear_damping[2], h.angular_quadratic_damping[2], torque)
            if expected is not None:
                self.assertAlmostEqual(rate, expected, delta=0.006)
            rt = runtime.HydroV2Runtime(profile, 1, "cpu", DT, dtype=torch.float64)
            pos = torch.tensor([[0., 0., h.equilibrium_com_z_m]])
            vel = torch.tensor([[0., 0., 0., 0., 0., rate]])
            step = rt.step(torch.tensor([[0., 1.]]), pos, vel, torch.tensor([[1., 0, 0, 0]]), torch.zeros(1, 3))
            self.assertLess(abs(float(step.torque_hull[0, 2])), 1e-6)
            self.assertAlmostEqual(float(step.motors.body_wrench[0, 5]), torque, places=9)


class RuntimeProperties(unittest.TestCase):
    def setUp(self):
        self.n = 6
        self.gen = torch.Generator().manual_seed(4242)

    def random(self, *shape):
        return torch.randn(*shape, generator=self.gen)

    def state(self, profile):
        pos = self.random(self.n, 3)*.02
        pos[:, 2] += profile.hydro.equilibrium_com_z_m
        vel = self.random(self.n, 6)*.3
        quat = torch.cat((torch.ones(self.n, 1), self.random(self.n, 3)*.05), -1)
        quat = quat/quat.norm(dim=-1, keepdim=True)
        actions = (torch.rand(self.n, 2, generator=self.gen)*2-1)
        return actions, pos, vel, quat

    def waves(self, m=24, scale=.05):
        amp = torch.rand(self.n, m, generator=self.gen)*scale
        omega = 2*math.pi*(0.3+torch.rand(self.n, m, generator=self.gen))
        return hydro.SpectralWaves(amp, omega, omega.square()/9.81,
                                   torch.rand(self.n, m, generator=self.gen)*2*math.pi,
                                   torch.rand(self.n, m, generator=self.gen)*2*math.pi)

    def make(self, profile=BLUEBOAT, **kwargs):
        return runtime.HydroV2Runtime(profile, self.n, "cpu", DT, dtype=torch.float64, **kwargs)

    def test_calm_none_equals_zero_amplitude_waves_bit_exact(self):
        actions, pos, vel, quat = self.state(BLUEBOAT)
        zero = self.waves()
        zero = replace(zero, amplitude=torch.zeros_like(zero.amplitude))
        a = self.make().step(actions, pos, vel, quat, torch.zeros(self.n, 3))
        b = self.make().step(actions, pos, vel, quat, torch.zeros(self.n, 3), waves=zero)
        self.assertTrue(torch.equal(a.force_actor, b.force_actor))
        self.assertTrue(torch.equal(a.torque_actor, b.torque_actor))

    def test_clock_advances_once_and_reset_is_row_selective(self):
        rt = self.make(replace(BLUEBOAT, thrusters=replace(BLUEBOAT.thrusters, nominal_lag_s=(.2, .2))))
        actions, pos, vel, quat = self.state(BLUEBOAT)
        for _ in range(5):
            step = rt.step(actions, pos, vel, quat, torch.zeros(self.n, 3))
        self.assertTrue(torch.equal(rt.ticks, torch.full((self.n,), 5, dtype=torch.int64)))
        self.assertAlmostEqual(float(step.time_s[0]), 4*DT)
        before = rt.thruster_state.nominal.clone()
        rt.reset(torch.tensor([1, 4]))
        self.assertEqual(rt.ticks.tolist(), [5, 0, 5, 5, 0, 5])
        self.assertTrue(torch.equal(rt.thruster_state.nominal[[0, 2, 3, 5]], before[[0, 2, 3, 5]]))
        self.assertTrue(torch.equal(rt.thruster_state.nominal[[1, 4]], torch.zeros(2, 2)))
        rt.reset(None)
        self.assertTrue(torch.equal(rt.ticks, torch.zeros(self.n, dtype=torch.int64)))

    def test_unrelated_reset_does_not_change_other_env_forcing(self):
        # C1 finding 8: an env's forcing must not depend on another env's reset.
        waves = self.waves()
        actions, pos, vel, quat = self.state(BLUEBOAT)
        a, b = self.make(), self.make()
        for tick in range(20):
            if tick == 7:
                b.reset(torch.tensor([0]))
            sa = a.step(actions, pos, vel, quat, torch.zeros(self.n, 3), waves=waves)
            sb = b.step(actions, pos, vel, quat, torch.zeros(self.n, 3), waves=waves)
            self.assertTrue(torch.equal(sa.force_actor[1:], sb.force_actor[1:]))
        self.assertFalse(torch.equal(sa.force_actor[0], sb.force_actor[0]))

    def test_clone_translation_invariance_including_vertical_origin(self):
        # C1 finding 9: identical env-local states in shifted clones are identical.
        waves = self.waves()
        actions, pos, vel, quat = self.state(BLUEBOAT)
        shift = self.random(self.n, 3)*50
        a = self.make().step(actions, pos, vel, quat, torch.zeros(self.n, 3), waves=waves)
        b = self.make().step(actions, pos+shift, vel, quat, shift, waves=waves)
        torch.testing.assert_close(a.force_actor, b.force_actor, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(a.torque_actor, b.torque_actor, rtol=1e-9, atol=1e-9)

    def test_actor_basis_gives_identical_world_wrench(self):
        # An asset whose bow is actor -x: hull x = -actor x, hull y = -actor y.
        basis = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
        flipped = replace(BLUEBOAT, actor_to_hull=basis)
        waves = self.waves()
        actions, pos, vel, quat = self.state(BLUEBOAT)
        r_hull = physics.rotation_matrix(quat)
        b = torch.tensor(basis)
        r_actor = r_hull @ b.T
        # Actor quaternion for r_actor = r_hull Rz(pi): q_hull (x) (0,0,0,1).
        w, x, y, z = quat.unbind(-1)
        quat_actor = torch.stack((-z, y, -x, w), -1)
        torch.testing.assert_close(physics.rotation_matrix(quat_actor), r_actor)
        a = self.make().step(actions, pos, vel, quat, torch.zeros(self.n, 3), waves=waves)
        c = self.make(flipped).step(actions, pos, vel, quat_actor, torch.zeros(self.n, 3), waves=waves)
        world_a = (r_hull @ a.force_actor.unsqueeze(-1)).squeeze(-1)
        world_c = (r_actor @ c.force_actor.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(world_a, world_c)
        torch.testing.assert_close((r_hull @ a.torque_actor.unsqueeze(-1)).squeeze(-1),
                                   (r_actor @ c.torque_actor.unsqueeze(-1)).squeeze(-1))
        with self.assertRaises(ValueError):
            self.make(replace(BLUEBOAT, actor_to_hull=((1., 0, 0), (0, 1., 0), (0, 0, -1.))))

    def test_wind_enters_applied_wrench_once_and_the_predictor(self):
        actions, pos, vel, quat = self.state(BLUEBOAT)
        wind = self.random(self.n, 3)*5
        wind[:, 2] = 0
        a = self.make().step(actions, pos, vel, quat, torch.zeros(self.n, 3))
        b = self.make().step(actions, pos, vel, quat, torch.zeros(self.n, 3), wind_force_w=wind)
        r = physics.rotation_matrix(quat)
        wind_hull = (r.transpose(-1, -2) @ wind.unsqueeze(-1)).squeeze(-1)
        # Applied = motors + wind + hydro(buoyancy + implicit damping of the
        # predicted state): the only other change is the damping response.
        torch.testing.assert_close(b.force_hull - b.hydro.damping_force_body,
                                   a.force_hull - a.hydro.damping_force_body + wind_hull)
        # Implicit residual: damping = -D (v_next - u) with v_next from all loads.
        h = BLUEBOAT.hydro
        body_v = (r.transpose(-1, -2) @ vel[:, :3].unsqueeze(-1)).squeeze(-1)
        drag, _ = hydro.damping_matrices(h, b.hydro.relative_velocity_body, torch.zeros(self.n, 3))
        gravity_body = (r.transpose(-1, -2) @ torch.tensor([0., 0., -h.gravity]).expand(self.n, 3)
                        .unsqueeze(-1)).squeeze(-1)
        total = b.force_hull + h.mass_kg*gravity_body
        next_v = body_v + DT*total/h.mass_kg
        torch.testing.assert_close(b.hydro.damping_force_body, -(drag @ next_v.unsqueeze(-1)).squeeze(-1))

    def test_configure_validates_and_applies_degradation(self):
        rt = self.make()
        actions = torch.tensor([[1., 0.]]).expand(self.n, 2)
        gains = mixer.one_sided_motor_gains(0.5, torch.arange(self.n))
        rt.configure(mixer.ThrusterDegradation(motor_gain=gains), resistance_scale=torch.full((self.n,), 1.5))
        _, pos, vel, quat = self.state(BLUEBOAT)
        step = rt.step(actions, pos, vel, quat, torch.zeros(self.n, 3))
        # One motor at gain 0.5, the other at 1: 1.5 x the per-motor forward limit
        # (D3: 60 N with its provisional 40 N motors; registry v2 8.2 kgf / 2).
        torch.testing.assert_close(step.motors.body_wrench[:, 0],
                                   torch.full((self.n,), 1.5*BLUEBOAT.thrusters.forward_limit_n[0]))
        for bad in (-1.0, float("nan"), torch.ones(self.n+1)):
            with self.assertRaises(ValueError):
                rt.configure(resistance_scale=bad)
        with self.assertRaises(ValueError):
            rt.configure(mixer.ThrusterDegradation(thrust_cap=2.0))
        with self.assertRaises(ValueError):
            self.make(engine_gravity=(0.0, 0.0, -9.8))
        self.make(engine_gravity=(0.0, 0.0, -9.81))

    def test_resistance_scale_scope_surge_sway_yaw_only(self):
        h = BLUEBOAT.hydro
        v = torch.ones(1, 3)
        base_lin, base_ang = hydro.damping_matrices(h, v, v)
        lin, ang = hydro.damping_matrices(h, v, v, 2.0)
        torch.testing.assert_close(torch.diagonal(lin, dim1=-2, dim2=-1),
                                   torch.diagonal(base_lin, dim1=-2, dim2=-1)*torch.tensor([2., 2., 1.]))
        torch.testing.assert_close(torch.diagonal(ang, dim1=-2, dim2=-1),
                                   torch.diagonal(base_ang, dim1=-2, dim2=-1)*torch.tensor([1., 1., 2.]))

    def test_wave_band_guard_fails_closed(self):
        import types
        # C2 SeaState v2 default upper edge is 3 fp (sea_state.py:154): too narrow.
        with self.assertRaises(ValueError):
            runtime.check_wave_band(BLUEBOAT, types.SimpleNamespace(tp_range=(1.0, 1.4), v2_f_max_over_fp=3.0))
        self.assertAlmostEqual(runtime.check_wave_band(
            BLUEBOAT, types.SimpleNamespace(tp_range=(1.0, 1.4), v2_f_max_over_fp=4.0)), 4.0)
        # Long periods need a wider relative band because the filter edge is fixed in Hz.
        need = runtime.check_wave_band(BLUEBOAT, types.SimpleNamespace(tp_range=(2.8, 3.5), v2_f_max_over_fp=8.0))
        self.assertAlmostEqual(need, hydro.required_high_frequency_hz(BLUEBOAT.hydro, 1/3.5)*3.5)
        self.assertGreater(need, 7.0)
        wamv_need = runtime.check_wave_band(WAMV, types.SimpleNamespace(tp_range=(2.8, 3.5), v2_f_max_over_fp=8.0))
        self.assertLess(wamv_need, need)

    def test_environmental_truth_is_not_an_observation_output(self):
        # The runtime returns wrenches/diagnostics only; it has no observation API.
        self.assertFalse(any(name.startswith("obs") or name.startswith("observ")
                             for name in dir(runtime.HydroV2Runtime)))


class RuntimeConstructionContract(unittest.TestCase):
    """E2 runtime items: D2 items 13/15 and controller decision 12."""

    def make(self, profile=BLUEBOAT, n=4, **kwargs):
        return runtime.HydroV2Runtime(profile, n, "cpu", DT, dtype=torch.float64, **kwargs)

    def test_rigid_body_gravity_flag_rejected_when_disabled(self):
        with self.assertRaisesRegex(ValueError, "disable_gravity"):
            self.make(rigid_body_disable_gravity=True)
        with self.assertRaises(ValueError):
            self.make(rigid_body_disable_gravity=1)
        rt = self.make(rigid_body_disable_gravity=False, engine_gravity=(0.0, 0.0, -9.81))
        self.assertTrue(rt.construction_checks["rigid_body_gravity_flag_checked"])
        self.assertTrue(rt.construction_checks["engine_gravity_checked"])
        self.assertFalse(self.make().construction_checks["rigid_body_gravity_flag_checked"])
        with self.assertRaises(ValueError):                  # existing engine-gravity check kept
            self.make(engine_gravity=(0.0, 0.0, -9.8), rigid_body_disable_gravity=False)

    def test_com_source_root_link_requires_zero_offset(self):
        self.assertEqual(self.make().com_source, "root_link")
        shifted = replace(BLUEBOAT, com_offset_actor_m=(0.006, 0.047, 0.0068))
        with self.assertRaisesRegex(ValueError, "origin is the COM"):
            self.make(shifted)
        self.make(shifted, com_source="root_com")            # explicit alternative stays possible
        with self.assertRaises(ValueError):
            self.make(com_source="actor")
        with self.assertRaises(ValueError):
            reference.require_zero_com_offset(torch.zeros(3))

    def test_step_from_root_link_equals_step_with_link_pose_as_com(self):
        gen = torch.Generator().manual_seed(12)
        n = 4
        pos = torch.randn(n, 3, generator=gen)*.02
        pos[:, 2] += BLUEBOAT.hydro.equilibrium_com_z_m
        quat = torch.cat((torch.ones(n, 1), torch.randn(n, 3, generator=gen)*.05), -1)
        quat = quat/quat.norm(dim=-1, keepdim=True)
        pose = torch.cat((pos, quat), -1)
        vel = torch.randn(n, 6, generator=gen)*.3
        actions = torch.rand(n, 2, generator=gen)*2-1
        a = self.make().step_from_root_link(actions, pose, vel, torch.zeros(n, 3))
        b = self.make().step(actions, pos, vel, quat, torch.zeros(n, 3))
        self.assertTrue(torch.equal(a.force_actor, b.force_actor))
        self.assertTrue(torch.equal(a.torque_actor, b.torque_actor))
        rt = self.make(replace(BLUEBOAT, com_offset_actor_m=(0.0, 0.0, 0.001)), com_source="root_com")
        with self.assertRaises(ValueError):
            rt.step_from_root_link(actions, pose, vel, torch.zeros(n, 3))

    def test_step_has_no_python_level_host_synchronisation(self):
        # D2 item 13: validate/cache at construction; step must not read tensor
        # values on the host. Every Python-level way to do that is trapped.
        n = 5
        gen = torch.Generator().manual_seed(7)
        amp = torch.rand(n, 16, generator=gen)*.05
        omega = 2*math.pi*(0.3+torch.rand(n, 16, generator=gen))
        waves = hydro.SpectralWaves(amp, omega, omega.square()/9.81, torch.rand(n, 16, generator=gen)*6,
                                    torch.rand(n, 16, generator=gen)*6)
        rt = runtime.HydroV2Runtime(WAMV, n, "cpu", DT, dtype=torch.float64, gyroscopic_predictor=True)
        rt.configure(mixer.ThrusterDegradation(thrust_cap=torch.full((n, 1), .8), additional_lag_s=.1),
                     resistance_scale=torch.full((n,), 1.25))
        pos = torch.zeros(n, 3)
        pos[:, 2] = WAMV.hydro.equilibrium_com_z_m
        args = (torch.rand(n, 2, generator=gen)*2-1, pos, torch.randn(n, 6, generator=gen)*.2,
                torch.tensor([[1., 0, 0, 0]]).expand(n, 4), torch.zeros(n, 3))
        kwargs = dict(current_velocity_w=torch.randn(n, 3, generator=gen)*.2, waves=waves,
                      wind_force_w=torch.randn(n, 3, generator=gen), wind_torque_w=torch.randn(n, 3, generator=gen))
        reference_step = runtime.HydroV2Runtime(WAMV, n, "cpu", DT, dtype=torch.float64, gyroscopic_predictor=True)
        reference_step.configure(mixer.ThrusterDegradation(thrust_cap=torch.full((n, 1), .8), additional_lag_s=.1),
                                 resistance_scale=torch.full((n,), 1.25))
        expected = reference_step.step(*args, **kwargs)
        saved = {}

        def trap(*unused, **also_unused):
            raise AssertionError("host synchronisation inside HydroV2Runtime.step")
        for name in ("item", "tolist", "__bool__", "__float__", "__int__", "__index__", "numpy"):
            saved[name] = getattr(torch.Tensor, name)
            setattr(torch.Tensor, name, trap)
        original_tensor, original_as_tensor = torch.tensor, torch.as_tensor

        def no_python_data(data, *more, **options):
            if not isinstance(data, torch.Tensor):
                raise AssertionError("Python data -> tensor copy inside HydroV2Runtime.step")
            return original_as_tensor(data, *more, **options)
        torch.tensor = no_python_data
        torch.as_tensor = no_python_data
        try:
            got = rt.step(*args, **kwargs)
        finally:
            for name, value in saved.items():
                setattr(torch.Tensor, name, value)
            torch.tensor, torch.as_tensor = original_tensor, original_as_tensor
        self.assertTrue(torch.equal(got.force_actor, expected.force_actor))
        self.assertTrue(torch.equal(got.torque_actor, expected.torque_actor))
        # The error-checking solve (a documented synchronisation) is not used.
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            rt.step(*args, **kwargs)
        names = {event.key for event in prof.key_averages()}
        self.assertNotIn("aten::_linalg_check_errors", names)
        self.assertNotIn("aten::is_nonzero", names)

    def test_cached_path_is_bit_identical_to_validating_path(self):
        gen = torch.Generator().manual_seed(99)
        for profile in (BLUEBOAT, WAMV):
            spec = profile.hydro
            n = 64
            q = torch.randn(n, 4, generator=gen)
            rotation = physics.rotation_matrix(q/q.norm(dim=-1, keepdim=True))
            pos = torch.randn(n, 3, generator=gen)*.05
            vel, omega = torch.randn(n, 3, generator=gen), torch.randn(n, 3, generator=gen)
            amp = torch.rand(n, 12, generator=gen)*.1
            w = 2*math.pi*(0.2+torch.rand(n, 12, generator=gen))
            sample = hydro.SpectralWaves(amp, w, w.square()/9.81, torch.rand(n, 12, generator=gen)*6,
                                         torch.rand(n, 12, generator=gen)*6).sample(
                pos[:, :2], torch.zeros(n, 2), .3, torch.zeros(n), spec)
            constants = hydro.prepare_hydro_constants(spec, "cpu", torch.float64)
            scale = torch.full((n,), 1.1)
            a = hydro.hydro_wrench(spec, rotation, pos, vel, omega, vel*.1, sample, DT,
                                   vel, omega, scale)
            b = hydro.hydro_wrench(spec, rotation, pos, vel, omega, vel*.1, sample, DT,
                                   vel, omega, scale, constants=constants)
            for x, y in ((a.force_body, b.force_body), (a.torque_body, b.torque_body),
                         (a.froude_krylov_force_body, b.froude_krylov_force_body)):
                self.assertTrue(torch.equal(x, y))
            with self.assertRaises(ValueError):
                hydro.hydro_wrench(replace(spec, froude_krylov_horizontal=False), rotation, pos, vel, omega,
                                   vel, sample, DT, constants=constants)


if __name__ == "__main__":
    unittest.main(verbosity=2)
