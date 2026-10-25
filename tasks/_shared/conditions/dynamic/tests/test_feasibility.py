"""CPU witness tests: feasible scenes admitted, infeasible rejected, replay is strict.

Fixtures are synthetic kinematic envelopes, never vessel calibrations.
"""
import math
import unittest
from dataclasses import replace

import fixtures as f

d = f.d
from dynamic_cpu.model import TimedState  # noqa: E402


class WitnessTests(unittest.TestCase):
    def setUp(self):
        self.env = f.synthetic_envelope()
        self.encounter = d.Encounter(d.EpisodeKey(31, 0, 0, "d7_reach"), "reach", 0,
                                     self.env.t_avoid_s, None, (), 30.0)
        self.request = d.WitnessRequest(d.Pose(0, 0, 0, 0), self.env, self.encounter,
                                        (-5, 20, -5, 5), d.Circle(10, 0, 0.5),
                                        horizon_s=15.0, beam_width=8)

    def physical_request(self, intruder, **kwargs):
        """Dose-1 request with a nominal prefix that ends at reveal + reaction."""
        encounter = replace(self.encounter, dose=1, nominal_ttc_s=2.0, intruders=(intruder,))
        request = replace(self.request, encounter=encounter, **kwargs)
        release = intruder.reveal_s + request.envelope.reaction_s
        prefix, pose, t = [TimedState(0.0, request.initial_pose)], request.initial_pose, 0.0
        while t < release - 1e-12:
            step = min(request.dt_s, release - t)
            pose = d.integrate(pose, 0.0, 0.0, step)
            t += step
            prefix.append(TimedState(t, pose, 0.0, 0.0))
        return replace(request, prefix=tuple(prefix))

    def test_exact_unicycle_arc_and_acceleration(self):
        pose = d.integrate(d.Pose(0, 0, 0, 1), 0, 1, math.pi / 2)
        self.assertAlmostEqual(pose.x, 1)
        self.assertAlmostEqual(pose.y, 1)
        self.assertAlmostEqual(pose.yaw, math.pi / 2)
        straight = d.integrate(d.Pose(0, 0, 0, 1), 1, 0, 2)
        self.assertAlmostEqual(straight.x, 4)
        self.assertAlmostEqual(straight.speed, 3)
        tiny = d.integrate(d.Pose(0, 0, 0, 1), 1, 1e-12, 2)
        self.assertAlmostEqual(tiny.x, 4)
        self.assertAlmostEqual(tiny.y, 0, places=9)
        astern = d.integrate(d.Pose(0, 0, 0, -1), 0, 0, 2)
        self.assertAlmostEqual(astern.x, -2)

    def test_clear_transit_accepts_and_replays(self):
        result = d.find_witness(self.request)
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(d.replay_witness(self.request, result.trajectory).accepted)
        self.assertGreater(result.min_clearance_m, self.request.clearance_m)

    def test_static_detour_is_found_by_tangent_guidance(self):
        request = replace(self.request, static_obstacles=(d.Circle(5, 0, 0.5),), horizon_s=25.0)
        result = d.find_witness(request)
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(any(abs(item.pose.y) > 0.8 for item in result.trajectory))

    def test_moving_crossing_accepts(self):
        obj = d.Intruder("crossing", 5, -5, 0, 1, 0.3, 0.0, "occluder_exit")
        request = self.physical_request(obj, initial_pose=d.Pose(0, 0, 0, 1))
        result = d.find_witness(request)
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(d.replay_witness(request, result.trajectory).accepted)

    def test_head_on_needs_astern_or_aside(self):
        """Head-on in a 2 m channel: the 2 x 1 m hull cannot pivot (needs 2 x 1.118 m),
        so it is infeasible without astern motion and feasible with it."""
        channel = (-20, 20, -1.0, 1.0)
        obj = d.Intruder("head_on", 9, 0, -0.5, 0, 0.25, 0.5, "detection_horizon")
        forward_only = self.physical_request(obj, initial_pose=d.Pose(0, 0, 0, 0.5), bounds=channel,
                                             goal=d.Circle(-12, 0, 1.0), horizon_s=30.0)
        self.assertFalse(d.find_witness(forward_only).accepted)
        astern_env = replace(self.env, max_reverse_speed_mps=1.0, max_astern_accel_mps2=0.8)
        with_astern = self.physical_request(obj, initial_pose=d.Pose(0, 0, 0, 0.5), bounds=channel,
                                            goal=d.Circle(-12, 0, 1.0), horizon_s=30.0)
        with_astern = replace(with_astern, envelope=astern_env,
                              encounter=replace(with_astern.encounter, t_avoid_s=astern_env.t_avoid_s))
        result = d.find_witness(with_astern)
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(any(s.pose.speed < 0 for s in result.trajectory))

    def test_initial_physical_collision_rejected_even_before_reveal(self):
        obj = d.Intruder("drifting", 0, 0, 0.1, 0, 0.3, 5, "occluder_exit")
        result = d.find_witness(self.physical_request(obj))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "initial_geometry_unsafe")

    def test_narrow_sealed_path_rejected(self):
        request = replace(self.request, bounds=(-2, 12, -0.4, 0.4),
                          static_obstacles=(d.Circle(4, 0, 0.5),), horizon_s=8, beam_width=4)
        result = d.find_witness(request)
        self.assertFalse(result.accepted)
        self.assertIn(result.reason, ("initial_geometry_unsafe", "no_witness_in_frozen_search"))

    def test_fully_blocked_goal_rejected_by_search(self):
        wall = tuple(d.Circle(6, y, 0.6) for y in (-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5))
        request = replace(self.request, static_obstacles=wall, horizon_s=12, beam_width=4)
        result = d.find_witness(request)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "no_witness_in_frozen_search")
        self.assertTrue(result.search_exhausted)

    def test_deadline_does_not_admit_unreachable_terminal(self):
        self.assertFalse(d.find_witness(replace(self.request, horizon_s=0.5)).accepted)

    def test_deadline_before_encounter_rejected(self):
        obj = d.Intruder("crossing", 5, -5, 0, 1, 0.3, 5, "occluder_exit")
        request = self.physical_request(obj, horizon_s=6)
        self.assertEqual(d.find_witness(request).reason, "deadline_precedes_encounter")

    def test_station_holds_through_full_window(self):
        request = replace(self.request, goal=d.Circle(0, 0, 0.5), hold_circle=d.Circle(0, 0, 1),
                          dwell_s=2, final_speed_mps=0.1, horizon_s=10, minimum_completion_s=10)
        result = d.find_witness(request)
        self.assertTrue(result.accepted, result.reason)
        self.assertAlmostEqual(result.trajectory[-1].time_s, 10)
        self.assertTrue(all(abs(s.pose.x) < 1e-12 and abs(s.pose.y) < 1e-12 for s in result.trajectory))

    def test_hold_constraint_starts_at_the_settled_clock(self):
        request = replace(self.request, initial_pose=d.Pose(-4, 0, 0, 0), bounds=(-10, 10, -5, 5),
                          goal=d.Circle(0, 0, 1.0), hold_circle=d.Circle(0, 0, 1.5),
                          hold_from_s=12.0, horizon_s=20, minimum_completion_s=20)
        self.assertTrue(d.find_witness(request).accepted)
        self.assertFalse(d.find_witness(replace(request, hold_from_s=0.0)).accepted)

    def test_docking_terminal_speed_heading_dwell(self):
        obj = d.Intruder("crossing", 2, -2, 0, 1, 0.3, 0.0, "detection_horizon")
        request = self.physical_request(obj, goal=d.Circle(6, 0, 0.6), final_speed_mps=0.1,
                                        final_heading_rad=0, dwell_s=1, horizon_s=20)
        result = d.find_witness(request)
        self.assertTrue(result.accepted, result.reason)
        self.assertLessEqual(abs(result.trajectory[-1].pose.speed), 0.1 + 1e-9)
        self.assertLessEqual(abs(result.trajectory[-1].pose.yaw), 0.15)
        self.assertTrue(d.replay_witness(request, result.trajectory).accepted)
        self.assertAlmostEqual(result.trajectory[len(request.prefix) - 1].time_s, self.env.reaction_s)

    def test_station_evades_moving_intruder_inside_hold_circle(self):
        obj = d.Intruder("drifting", 0, -5, 0, 1, 0.3, 0.0, "detection_horizon")
        request = self.physical_request(obj, goal=d.Circle(0, 0, 3), hold_circle=d.Circle(0, 0, 3),
                                        horizon_s=12, minimum_completion_s=12, beam_width=64)
        result = d.find_witness(request)
        self.assertTrue(result.accepted, result.reason)
        self.assertAlmostEqual(result.trajectory[-1].time_s, 12)
        self.assertTrue(all(math.hypot(s.pose.x, s.pose.y) < 3 for s in result.trajectory))

    def test_ordered_mandatory_gates(self):
        request = replace(self.request, waypoints=(d.Circle(2, 0, 0.5), d.Circle(6, 0, 0.5)))
        result = d.find_witness(request)
        self.assertTrue(result.accepted, result.reason)
        other = replace(request, waypoints=(d.Circle(6, 4, 0.1), d.Circle(2, 0, 0.5)))
        self.assertEqual(d.replay_witness(other, result.trajectory).reason, "task_completion_not_certified")

    def test_nominal_prefix_cannot_change(self):
        obj = d.Intruder("crossing", 5, -5, 0, 1, 0.3, 2, "occluder_exit")
        request = self.physical_request(obj)
        first = TimedState(0, request.initial_pose)
        changed = TimedState(0.25, d.integrate(first.pose, 1, 0, 0.25), 1, 0)
        self.assertEqual(d.replay_witness(request, (first, changed)).reason, "nominal_prefix_changed")

    def test_replay_rejects_illegal_controls_and_kinematics(self):
        first = TimedState(0, self.request.initial_pose)
        cases = (
            (TimedState(0.25, d.integrate(first.pose, 2, 0, 0.25), 2, 0), "control_outside_envelope"),
            (TimedState(0.25, d.integrate(first.pose, 0, 2, 0.25), 0, 2), "control_outside_envelope"),
            (TimedState(0.25, d.Pose(0, 0, 0, 3), 0, 0), "speed_outside_envelope"),
            (TimedState(0.25, d.Pose(0, 0, 0, -0.25), -1, 0), "speed_outside_envelope"),
            (TimedState(0.25, d.Pose(2, 0, 0, 0), 0, 0), "kinematic_rollout_mismatch"),
            (TimedState(0, first.pose), "invalid_time_or_nonfinite_state"),
        )
        for state, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(d.replay_witness(self.request, (first, state)).reason, reason)

    def test_astern_acceleration_bound_is_enforced(self):
        env = replace(self.env, max_reverse_speed_mps=1.0, max_astern_accel_mps2=0.4)
        request = replace(self.request, envelope=env,
                          encounter=replace(self.encounter, t_avoid_s=env.t_avoid_s))
        first = TimedState(0, request.initial_pose)
        too_hard = TimedState(0.25, d.integrate(first.pose, -0.8, 0, 0.25), -0.8, 0)
        self.assertEqual(d.replay_witness(request, (first, too_hard)).reason, "control_outside_envelope")

    def test_between_tick_linear_tunneling_rejected(self):
        obj = d.Intruder("crossing", -2, 0, 4, 0, 0.1, 0, "detection_horizon")
        env = replace(self.env, reaction_s=0.0)
        request = replace(self.physical_request(obj), envelope=env, dt_s=1, horizon_s=1,
                          goal=d.Circle(0, 0, 1), collision_substeps=1, prefix=())
        request = replace(request, encounter=replace(request.encounter, t_avoid_s=env.t_avoid_s))
        trajectory = (TimedState(0, request.initial_pose), TimedState(1, request.initial_pose))
        self.assertEqual(d.replay_witness(request, trajectory).reason, "swept_clearance_not_certified")

    def test_native_orbit_between_safe_endpoints_rejected(self):
        obj = d.NativeMotion("orbit", 1, 0, 0.1, 2 * math.pi, orbit_radius_m=1)
        self.assertAlmostEqual(obj.position(0)[0], obj.position(1)[0])
        env = replace(self.env, reaction_s=0.0)
        request = replace(self.request, envelope=env, dt_s=1, horizon_s=1, goal=d.Circle(0, 0, 1),
                          collision_substeps=1,
                          encounter=replace(self.encounter, task="moving_basin", intruders=(obj,),
                                            t_avoid_s=env.t_avoid_s))
        trajectory = (TimedState(0, request.initial_pose), TimedState(1, request.initial_pose))
        self.assertEqual(d.replay_witness(request, trajectory).reason, "swept_clearance_not_certified")

    def test_full_footprint_boundary_rejected(self):
        request = replace(self.request, initial_pose=d.Pose(-4.7, 0, 0, 0))
        self.assertEqual(d.find_witness(request).reason, "initial_geometry_unsafe")

    def test_search_is_deterministic(self):
        self.assertEqual(d.find_witness(self.request), d.find_witness(self.request))

    def test_mutable_admission_parameters_rejected(self):
        with self.assertRaisesRegex(ValueError, "immutable tuples"):
            d.find_witness(replace(self.request, static_obstacles=[]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
