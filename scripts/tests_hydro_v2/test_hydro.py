"""CPU properties for the unified water model; execute through run_cpu_tests.py."""
import importlib.util
import math
from pathlib import Path
import sys
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
restore = load("restoring_v2")
hydro = load("hydro_v2")
mixer = load("thruster_mixer")
load("vehicles_v2")
load("journal_vehicle")
profiles = load("hydro_profiles")
# E2 (decision 1): profiles come from the D5 v2 registry.
SPEC = profiles.profile_for_vehicle("blueboat_v2").hydro
WAMV = profiles.profile_for_vehicle("wamv_vrx_v2_opt1").hydro
torch.set_default_dtype(torch.float64)


class HydroProperties(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(71021)
        self.n = 128
        self.zero = torch.zeros(self.n, 3)
        self.identity = torch.eye(3).expand(self.n, 3, 3)

    def random(self, *shape):
        return torch.randn(*shape, generator=self.gen)

    def rotation(self):
        q = self.random(self.n, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        return physics.rotation_matrix(q)

    def equilibrium(self, spec):
        pos = self.zero.clone()
        pos[:, 2] = spec.equilibrium_com_z_m
        return pos

    def test_disabled_equals_zero_amplitude_and_zero_current_bit_exact(self):
        for spec in (SPEC, WAMV):
            rotation, pos = self.rotation(), self.random(self.n, 3)
            velocity, omega = self.random(self.n, 3), self.random(self.n, 3)
            calm = hydro.WaveSample.calm(pos)
            zeros = torch.zeros(self.n, 16)
            sea = hydro.SpectralWaves(zeros, torch.ones_like(zeros), torch.ones_like(zeros), zeros, zeros)
            sample = sea.sample(pos[:, :2], pos[:, :2], torch.arange(self.n), torch.zeros(self.n), spec)
            a = hydro.hydro_wrench(spec, rotation, pos, velocity, omega, self.zero, calm, 1/120)
            b = hydro.hydro_wrench(spec, rotation, pos, velocity, omega, self.zero, sample, 1/120)
            self.assertTrue(torch.equal(a.force_body, b.force_body))
            self.assertTrue(torch.equal(a.torque_body, b.torque_body))

    def test_rotational_reference_flag_is_identical_in_calm_water(self):
        rotation, pos = self.rotation(), self.random(self.n, 3)
        velocity, omega = self.random(self.n, 3), self.random(self.n, 3)
        calm = hydro.WaveSample.calm(pos)
        inertial = replace(SPEC, rotational_relative_motion=False)
        a = hydro.hydro_wrench(SPEC, rotation, pos, velocity, omega, self.zero, calm, 1/120)
        b = hydro.hydro_wrench(inertial, rotation, pos, velocity, omega, self.zero, calm, 1/120)
        self.assertTrue(torch.equal(a.torque_body, b.torque_body))
        self.assertTrue(torch.equal(a.force_body, b.force_body))

    def test_uniform_drift_zero_drag_all_attitudes_and_vertical_flow(self):
        # At the hydrostatic datum buoyancy balances gravity for any attitude,
        # so with no other load the predicted relative velocity stays zero.
        for spec in (SPEC, WAMV):
            velocity = self.random(self.n, 3)
            result = hydro.hydro_wrench(spec, self.rotation(), self.equilibrium(spec), velocity, self.zero,
                                        velocity, hydro.WaveSample.calm(self.zero), 1/120)
            self.assertLess(result.damping_force_body.abs().max().item(), 1e-9)
            self.assertLess(result.relative_velocity_body.abs().max().item(), 1e-12)

    def test_sum_water_before_drag_including_cancellation(self):
        a = self.random(self.n, 3)
        b = self.random(self.n, 3)
        calm = hydro.WaveSample.calm(self.zero)
        wave = replace(calm, orbital_velocity_world=b)
        args = (SPEC, self.identity, self.equilibrium(SPEC), self.zero, self.zero)
        one = hydro.hydro_wrench(*args, a, wave, 1/120)
        two = hydro.hydro_wrench(*args, a+b, calm, 1/120)
        torch.testing.assert_close(one.force_body, two.force_body, rtol=0, atol=0)
        cancelled = hydro.hydro_wrench(*args, -b, wave, 1/120)
        self.assertLess(cancelled.damping_force_body.abs().max().item(), 1e-9)
        # Aligned, opposed and transverse flows exercise nonlinear cross terms.
        for current, orbit in (((1., 0, 0), (1., 0, 0)), ((1., 0, 0), (-1., 0, 0)), ((1., 0, 0), (0, 1., 0))):
            current, orbit = torch.tensor(current).expand_as(a), torch.tensor(orbit).expand_as(a)
            result = hydro.hydro_wrench(*args, current, replace(calm, orbital_velocity_world=orbit), 1/120)
            expected = hydro.hydro_wrench(*args, current+orbit, calm, 1/120)
            torch.testing.assert_close(result.force_body, expected.force_body)
        # C1 finding 6 numbers: aligned 1+1 m/s is ONE quadratic law on 2 m/s,
        # not the sum of two separate laws on 1 m/s each.
        both = hydro.hydro_wrench(*args, torch.tensor((1., 0, 0)).expand_as(a),
                                  replace(calm, orbital_velocity_world=torch.tensor((1., 0, 0)).expand_as(a)), 1/120)
        single = hydro.hydro_wrench(*args, torch.tensor((1., 0, 0)).expand_as(a), calm, 1/120)
        self.assertGreater(float(both.damping_force_body[0, 0]), 2.0*float(single.damping_force_body[0, 0]))

    def test_relative_energy_decay_and_negative_step_work(self):
        for dt in (1e-5, 1/120, .1, 5.0):
            a = self.random(self.n, 3, 3)
            mass = a.transpose(-1, -2) @ a + torch.eye(3)
            a = self.random(self.n, 3, 3)
            drag = a.transpose(-1, -2) @ a * 470
            velocity, water = self.random(self.n, 3), self.random(self.n, 3)
            force = physics.implicit_damping_wrench(velocity, mass, drag, dt, water_velocity=water)
            next_v = velocity + dt * torch.linalg.solve(mass, force.unsqueeze(-1)).squeeze(-1)
            old, new = velocity-water, next_v-water
            old_energy = torch.einsum("ni,nij,nj->n", old, mass, old)
            new_energy = torch.einsum("ni,nij,nj->n", new, mass, new)
            self.assertTrue((new_energy <= old_energy+1e-10).all())
            self.assertTrue(((force*new).sum(-1) <= 1e-10).all())

    def test_known_forcing_and_water_solve_residual(self):
        v, u, f = (self.random(self.n, 3) for _ in range(3))
        mass = torch.eye(3)*17.26
        drag = torch.diag_embed(self.random(self.n, 3).abs()*500)
        dt = .037
        damping = physics.implicit_damping_wrench(v, mass, drag, dt, f, u)
        next_v = v + dt*(f+damping)/17.26
        torch.testing.assert_close(damping, -(drag @ (next_v-u).unsqueeze(-1)).squeeze(-1))

    def test_c2_no_water_extension_preserves_formula_exactly(self):
        v, f = self.random(self.n, 3), self.random(self.n, 3)
        mass = torch.eye(3)*17.26
        drag = torch.diag_embed(self.random(self.n, 3).abs())
        rhs = (mass@v.unsqueeze(-1)).squeeze(-1) + (1/120)*f
        expected = -(drag@torch.linalg.solve(mass+drag/120, rhs.unsqueeze(-1))).squeeze(-1)
        torch.testing.assert_close(physics.implicit_damping_wrench(v, mass, drag, 1/120, f), expected, rtol=0, atol=0)

    def test_body_yaw_damping_not_world_roll_damping_at_tilt(self):
        # C1 finding 2: angular damping acts in hull axes at any attitude.
        rotation = self.rotation()
        body_omega = torch.zeros_like(self.zero)
        body_omega[:, 2] = 1
        world_omega = (rotation@body_omega.unsqueeze(-1)).squeeze(-1)
        spec = replace(SPEC, roll_stiffness_nm=0, pitch_stiffness_nm=0)
        result = hydro.hydro_wrench(spec, rotation, self.zero, self.zero, world_omega,
                                    self.zero, hydro.WaveSample.calm(self.zero), 1/120)
        coefficient = SPEC.angular_linear_damping[2] + SPEC.angular_quadratic_damping[2]
        izz = SPEC.inertia_kgm2[2][2]
        expected_yaw = -coefficient/(1+coefficient/(120*izz))
        torch.testing.assert_close(result.damping_torque_body[:, 2], torch.full((self.n,), expected_yaw))
        self.assertLess(result.damping_torque_body[:, :2].abs().max().item(), 2e-12)

    def test_hull_following_wave_surface_has_no_angular_damping(self):
        # Relative-motion reference: a hull rotating exactly with the local
        # surface normal and aligned to it feels neither restoring nor damping.
        slope = torch.zeros(self.n, 2)
        rate = self.random(self.n, 2)*.1
        omega_surface = hydro.surface_angular_velocity(slope, rate)
        sample = hydro.WaveSample(torch.zeros(self.n), slope, torch.zeros(self.n, 3), omega_surface)
        spec = replace(SPEC, angular_quadratic_damping=(0., 0., 0.))
        result = hydro.hydro_wrench(spec, self.identity, self.equilibrium(SPEC), self.zero, omega_surface,
                                    self.zero, sample, 1/120)
        self.assertLess(result.damping_torque_body.abs().max().item(), 1e-12)
        inertial = hydro.hydro_wrench(replace(spec, rotational_relative_motion=False), self.identity,
                                      self.equilibrium(SPEC), self.zero, omega_surface, self.zero, sample, 1/120)
        self.assertGreater(inertial.damping_torque_body.abs().max().item(), 1.0)

    def test_surface_angular_velocity_matches_finite_difference_of_normal(self):
        slope, rate = self.random(self.n, 2)*.2, self.random(self.n, 2)
        eps = 1e-6
        n0 = hydro.surface_normal(slope)
        n1 = hydro.surface_normal(slope + eps*rate)
        omega = hydro.surface_angular_velocity(slope, rate)
        torch.testing.assert_close(torch.cross(omega, n0, dim=-1), (n1-n0)/eps, atol=1e-5, rtol=1e-5)
        self.assertLess((omega*n0).sum(-1).abs().max().item(), 1e-12)

    def test_heave_equilibrium_derivative_and_no_duplicate_excitation(self):
        for spec in (SPEC, WAMV):
            calm = hydro.WaveSample.calm(self.zero)
            datum = self.equilibrium(spec)
            force, _ = hydro.hydrostatic_wrench(spec, self.identity, datum, calm)
            torch.testing.assert_close(force[:, 2], torch.full((self.n,), spec.mass_kg*spec.gravity))
            raised = replace(calm, elevation=torch.full((self.n,), 1e-4))
            f2, _ = hydro.hydrostatic_wrench(spec, self.identity, datum, raised)
            torch.testing.assert_close((f2[:, 2]-force[:, 2])/1e-4,
                                       torch.full((self.n,), spec.heave_stiffness_n_per_m))
            pos = datum.clone()
            pos[:, 2] += 1e-4
            f3, _ = hydro.hydrostatic_wrench(spec, self.identity, pos, raised)
            torch.testing.assert_close(f3, force, rtol=0, atol=1e-9)

    def test_restoring_aligns_to_wave_normal_not_world_up(self):
        slope = torch.tensor([[.1, -.05]]).expand(self.n, 2)
        normal = hydro.surface_normal(slope)
        sample = hydro.WaveSample(torch.zeros(self.n), slope, torch.zeros(self.n, 3))
        # Build a hull whose z axis equals the normal: zero restoring.
        z = normal
        x = torch.nn.functional.normalize(torch.cross(torch.tensor([0., 1., 0.]).expand_as(z), z, dim=-1), dim=-1)
        y = torch.cross(z, x, dim=-1)
        rotation = torch.stack((x, y, z), dim=-1)
        _, torque = hydro.hydrostatic_wrench(SPEC, rotation, self.equilibrium(SPEC), sample)
        self.assertLess(torque.abs().max().item(), 1e-10)
        _, level = hydro.hydrostatic_wrench(SPEC, self.identity, self.equilibrium(SPEC), sample)
        self.assertGreater(level.abs().max().item(), 1.0)

    def test_local_phase_clone_translation_and_vector_clock_independence(self):
        shape = (self.n, 12)
        waves = hydro.SpectralWaves(self.random(*shape)*.01, self.random(*shape).abs(),
                                    self.random(*shape).abs(), self.random(*shape), self.random(*shape))
        local, origin = self.random(self.n, 2), self.random(self.n, 2)*96
        times = torch.arange(self.n)*.137
        a = waves.sample(local, torch.zeros_like(local), times, torch.zeros(self.n), SPEC)
        b = waves.sample(local+origin, origin, times, torch.zeros(self.n), SPEC)
        torch.testing.assert_close(a.elevation, b.elevation, atol=1e-14, rtol=1e-12)
        times[0] = 0
        c = waves.sample(local, torch.zeros_like(local), times, torch.zeros(self.n), SPEC)
        torch.testing.assert_close(a.orbital_velocity_world[1:], c.orbital_velocity_world[1:], atol=0, rtol=0)

    def test_vertical_wave_velocity_matches_surface_time_derivative(self):
        spec = replace(SPEC, orbital_depth_m=0)
        waves = hydro.SpectralWaves(torch.full((self.n, 1), .1), torch.full((self.n, 1), 2.0),
                                    torch.full((self.n, 1), 4/spec.gravity), torch.zeros(self.n, 1),
                                    torch.ones(self.n, 1))
        xy = self.zero[:, :2]
        eps = 1e-6
        a = waves.sample(xy, xy, .7, torch.zeros(self.n), spec, footprint=False)
        plus = waves.sample(xy, xy, .7+eps, torch.zeros(self.n), spec, footprint=False)
        minus = waves.sample(xy, xy, .7-eps, torch.zeros(self.n), spec, footprint=False)
        torch.testing.assert_close(a.orbital_velocity_world[:, 2], (plus.elevation-minus.elevation)/(2*eps))
        # Slope-rate consistency: d(slope)/dt at a fixed point.
        torch.testing.assert_close(a.surface_angular_velocity_world,
                                   hydro.surface_angular_velocity(a.slope, (plus.slope-minus.slope)/(2*eps)))

    def test_filter_cutoff_and_invalid_spec(self):
        for spec in (SPEC, WAMV):
            for yaw, length in ((0., spec.length_m), (math.pi/2, spec.beam_m)):
                k = math.sqrt(12*math.log(2))/length
                h = hydro.footprint_transfer(torch.tensor([[k]]), torch.zeros(1, 1), torch.tensor([yaw]),
                                             spec.length_m, spec.beam_m)
                self.assertAlmostEqual(h.item(), 2**-.5, places=14)
        for kwargs in ({"waterplane_m2": 0}, {"inertia_kgm2": ((0, 0, 0),)*3}, {"linear_damping": (-1, 1, 1)},
                       {"mass_kg": 10000}):
            with self.assertRaises(ValueError):
                replace(SPEC, **kwargs)

    def test_no_vessel_numbers_in_kernels(self):
        # Kernels must not hard-code BlueBoat/WAM-V numbers (owner rule).
        # E2: also scans reference_point.py and the v2 registry numbers.
        for name in ("hydro_v2.py", "thruster_mixer.py", "restoring_v2.py", "hydro_v2_runtime.py",
                     "reference_point.py"):
            text = (SHARED/name).read_text(encoding="ascii")
            for token in ("17.26", "0.3607", ".3607", "283.385", "283.3847", "242.67", "242.665",
                          "1.027135", "470", "9.1,", "5.9", "278.16", "139.77", "40.207", "20.067"):
                self.assertNotIn(token, text, "%s contains vessel number %s" % (name, token))


if __name__ == "__main__":
    unittest.main(verbosity=2)
