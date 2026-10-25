"""CPU properties for conservative restoring and a single COM reference."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import torch


def load_file(name, filename):
    path = Path(__file__).resolve().parents[2] / "tasks" / "_shared" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


restoring = load_file("d3_test_restoring_v2", "restoring_v2.py")
reference = load_file("d3_test_reference_point", "reference_point.py")
DTYPE = torch.float64


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        vector.shape[:-1] + (3, 3))


def rotations(count=512, seed=173):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Haar-distributed proper rotations, covering inverted/coupled attitudes.
    q, r = torch.linalg.qr(torch.randn(count, 3, 3, generator=generator,
                                     dtype=DTYPE, device="cpu"))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q[..., :, 2] *= torch.linalg.det(q).unsqueeze(-1)
    return q


class RestoringProperties(unittest.TestCase):
    def test_no_flat_world_vertical_torque_at_random_attitudes(self):
        rotation = rotations()
        moment = restoring.restoring_moment(rotation, 280.0, 141.0)
        world = (rotation @ moment.unsqueeze(-1)).squeeze(-1)
        self.assertLess(world[:, 2].abs().max().item(), 2e-12)
        # Anisotropy needs a coupled BODY z torque to achieve this property.
        self.assertGreater(moment[:, 2].abs().max().item(), 10)

    def test_flat_potential_is_yaw_invariant_at_random_attitudes(self):
        rotation = rotations()
        yaw_vector = torch.zeros(512, 3, dtype=DTYPE)
        yaw_vector[:, 2] = torch.linspace(-9, 9, 512, dtype=DTYPE)
        rotated = torch.matrix_exp(skew(yaw_vector)) @ rotation
        torch.testing.assert_close(
            restoring.restoring_potential(rotation, 280.0, 141.0),
            restoring.restoring_potential(rotated, 280.0, 141.0),
            atol=1e-12, rtol=1e-12)

    def test_moment_matches_finite_difference_of_potential(self):
        rotation = rotations(64)
        eps = 2e-6
        for normal in (None, torch.tensor([0.25, -0.4, 1.0], dtype=DTYPE)):
            moment = restoring.restoring_moment(rotation, 280.0, 141.0, normal)
            world = (rotation @ moment.unsqueeze(-1)).squeeze(-1)
            for axis in range(3):
                increment = torch.zeros(3, dtype=DTYPE)
                increment[axis] = eps
                plus = torch.matrix_exp(skew(increment)) @ rotation
                minus = torch.matrix_exp(skew(-increment)) @ rotation
                derivative = (restoring.restoring_potential(
                    plus, 280.0, 141.0, normal) - restoring.restoring_potential(
                    minus, 280.0, 141.0, normal)) / (2 * eps)
                torch.testing.assert_close(derivative, -world[:, axis],
                                           atol=7e-8, rtol=2e-8)

    def test_tilted_surface_has_no_torque_about_its_normal(self):
        rotation = rotations()
        normal = torch.tensor([0.4, -0.7, 1.0], dtype=DTYPE)
        moment = restoring.restoring_moment(rotation, 280.0, 141.0, normal)
        world = (rotation @ moment.unsqueeze(-1)).squeeze(-1)
        self.assertLess((world * normal).sum(-1).abs().max().item(), 2e-12)
        # A fixed tilted plane does not have world-yaw symmetry.
        self.assertGreater(world[:, 2].abs().max().item(), 1.0)

    def test_joint_rotation_of_hull_and_wave_normal_preserves_energy(self):
        rotation = rotations(64)
        change = rotations(64, seed=721)
        normal = torch.tensor([0.2, 0.3, 1.0], dtype=DTYPE)
        normal_changed = (change @ normal.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(
            restoring.restoring_potential(rotation, 280.0, 141.0, normal),
            restoring.restoring_potential(change @ rotation, 280.0, 141.0,
                                          normal_changed),
            atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(
            restoring.restoring_moment(rotation, 280.0, 141.0, normal),
            restoring.restoring_moment(change @ rotation, 280.0, 141.0,
                                       normal_changed),
            atol=1e-11, rtol=1e-11)

    def test_upright_zero_and_small_angle_stiffness(self):
        identity = torch.eye(3, dtype=DTYPE)
        self.assertEqual(restoring.restoring_potential(identity, 280, 141), 0)
        torch.testing.assert_close(restoring.restoring_moment(identity, 280, 141),
                                   torch.zeros(3, dtype=DTYPE))
        for axis, stiffness in ((0, 280), (1, 141)):
            for angle in (-1e-5, 1e-5):
                vector = torch.zeros(3, dtype=DTYPE)
                vector[axis] = angle
                rotation = torch.matrix_exp(skew(vector))
                moment = restoring.restoring_moment(rotation, 280, 141)
                self.assertAlmostEqual(moment[axis].item() / angle,
                                       -stiffness, places=6)
                self.assertLess((moment * vector).sum().item(), 0)
                self.assertGreater(restoring.restoring_potential(
                    rotation, 280, 141).item(), 0)

    def test_nonsingular_at_inversion_and_nonnegative_energy(self):
        rotation = rotations(4096)
        for kr, kp in ((280, 141), (1, 1000), (1000, 1), (0, 1), (0, 0)):
            potential = restoring.restoring_potential(rotation, kr, kp)
            self.assertTrue(torch.isfinite(potential).all())
            self.assertGreaterEqual(potential.min().item(), -1e-12)
            self.assertLessEqual(potential.max().item(), kr + kp + 1e-10)
        inverted = torch.diag(torch.tensor([1., -1., -1.], dtype=DTYPE))
        self.assertEqual(restoring.restoring_potential(inverted, 280, 141), 421)
        torch.testing.assert_close(restoring.restoring_moment(inverted, 280, 141),
                                   torch.zeros(3, dtype=DTYPE))

    def test_tensor_stiffness_and_input_validation(self):
        rotation = rotations(2)
        torque = restoring.restoring_moment(rotation, torch.tensor([280, 1]),
                                            torch.tensor([141, 10]))
        self.assertEqual(torque.shape, (2, 3))
        for bad in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                restoring.restoring_moment(rotation, bad, 1)
        with self.assertRaises(ValueError):
            restoring.restoring_moment(rotation, 1, 1, torch.zeros(3))


class ReferencePointProperties(unittest.TestCase):
    def test_actor_com_pose_and_velocity_round_trip(self):
        rotation = rotations()
        generator = torch.Generator().manual_seed(723)
        position, velocity, omega, offset = [
            torch.randn(512, 3, generator=generator, dtype=DTYPE)
            for _ in range(4)]
        actor_position = reference.actor_position_from_com(position, rotation,
                                                            offset)
        actor_velocity = reference.actor_velocity_from_com(velocity, omega,
                                                            rotation, offset)
        torch.testing.assert_close(reference.com_position_world(
            actor_position, rotation, offset), position)
        torch.testing.assert_close(reference.com_velocity_world(
            actor_velocity, omega, rotation, offset), velocity)

    def test_stationary_com_spin_has_zero_metric_motion(self):
        angles = torch.linspace(0, 2 * torch.pi, 721, dtype=DTYPE)
        vectors = torch.zeros(721, 3, dtype=DTYPE)
        vectors[:, 2] = angles
        rotation = torch.matrix_exp(skew(vectors))
        position = torch.tensor([17., -3., 0.1], dtype=DTYPE).expand(721, -1)
        offset = torch.tensor([0.006, 0.047, 0.0068], dtype=DTYPE)
        actor_position = reference.actor_position_from_com(position, rotation,
                                                            offset)
        com = reference.com_position_world(actor_position, rotation, offset)
        actor_length = torch.linalg.vector_norm(torch.diff(actor_position,
                                                           dim=0), dim=-1).sum()
        com_length = torch.linalg.vector_norm(torch.diff(com, dim=0), dim=-1).sum()
        self.assertGreater(actor_length.item(), 0.29)
        self.assertLess(com_length.item(), 1e-11)
        self.assertAlmostEqual(torch.linalg.vector_norm(
            actor_position[360] - actor_position[0]).item(),
            2 * (0.006**2 + 0.047**2)**0.5, places=12)
        omega = torch.tensor([0., 0., 1.], dtype=DTYPE).expand(721, -1)
        actor_velocity = reference.actor_velocity_from_com(
            torch.zeros_like(position), omega, rotation, offset)
        torch.testing.assert_close(reference.com_velocity_world(
            actor_velocity, omega, rotation, offset), torch.zeros_like(position))

    def test_footprint_exactly_matches_actor_geometry_at_any_attitude(self):
        rotation = rotations(16)
        position = torch.tensor([1., 2., 3.], dtype=DTYPE).expand(16, -1)
        offset = torch.tensor([0.006, 0.047, 0.0068], dtype=DTYPE)
        corners = torch.tensor([[x, y, z] for x in (-0.600826, 0.597951)
                                for y in (-0.449360, 0.449360)
                                for z in (-0.223200, 0.152962)], dtype=DTYPE)
        actor_position = reference.actor_position_from_com(position, rotation,
                                                            offset)
        direct = corners @ rotation.transpose(-1, -2) + actor_position[:, None]
        actual = reference.collision_vertices_world(position, rotation, corners,
                                                      offset)
        torch.testing.assert_close(actual, direct)
        torch.testing.assert_close(reference.collision_footprint_world_xy(
            position, rotation, corners, offset), direct[..., :2])
        rebased = reference.com_centered_vertices_body(corners, offset)
        self.assertNotAlmostEqual(rebased[:, 1].min().item(),
                                  -rebased[:, 1].max().item(), places=4)

    def test_per_environment_offsets_and_clone_translation(self):
        rotation = rotations(3)
        com = torch.tensor([[1., 2., 0.], [1., 2., 0.], [1., 2., 0.]],
                           dtype=DTYPE)
        origins = torch.tensor([[0., 0., 0.], [96., 192., 0.], [-80., 40., 0.]],
                               dtype=DTYPE)
        torch.testing.assert_close(reference.environment_local_position(
            com + origins, origins), com)
        offset = torch.tensor([[.006, .047, .0068], [0., 0., 0.], [.5, .2, 0.]],
                              dtype=DTYPE)
        vertices = torch.tensor([[1., 1., 1.], [-1., -1., -1.]], dtype=DTYPE)
        transformed = reference.collision_vertices_world(com, rotation, vertices,
                                                         offset)
        self.assertEqual(transformed.shape, (3, 2, 3))
        for i in range(3):
            torch.testing.assert_close(transformed[i],
                reference.collision_vertices_world(com[i], rotation[i], vertices,
                                                    offset[i]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
