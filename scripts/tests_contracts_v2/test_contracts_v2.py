"""Episode-contract regressions (pure module; no Isaac imports)."""

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


j = load("d4_test_contracts", ROOT / "tasks/_shared/journal_contracts_v2.py")
cfg = load("d4_test_cfg_for_contracts", ROOT / "tasks/_shared/journal_cfg_v2.py")

BB_HULL = cfg.BLUEBOAT.footprint.vertices_m          # COM-relative D5 polygon
C2_BERTH = j.Berth((0.0, 0.0), 0.0, -7.0, 0.85, 4 * 0.899 / 2)   # C2 berth_geometry.py:44-75 fixture


def state(x=0.0, y=0.0, z=0.0, *, yaw=0.0, roll=0.0, vx=0.0, vy=0.0):
    cr, sr, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(yaw / 2), math.sin(yaw / 2)
    return j.RigidState((x, y, z), (cr * cy, sr * cy, sr * sy, cr * sy), (vx, vy, 0.0), (0.0, 0.0, 0.0))


def dock_fixture(episode, t0=0.0):
    """Outside the mouth, inward crossing, settle, then a 5 s hold."""
    samples = [(t0 + 1.0, state(-7.5, vx=0.5)), (t0 + 2.0, state(-6.8, vx=0.7)), (t0 + 3.0, state(0.0))]
    samples.extend((t0 + float(t), state()) for t in range(4, 9))
    result = None
    for time_s, sample in samples:
        result = episode.step(time_s, sample)
    return result


class ValidityTests(unittest.TestCase):
    def test_c5_codes_preserved_and_validity_codes_appended(self):
        self.assertEqual([int(r) for r in j.TerminationReason], list(range(10)))
        self.assertEqual([j.TERMINATION_REASON_NAMES[k] for k in range(6)],
                         ["running", "success", "collision", "timeout", "out_of_bounds", "failure"])
        self.assertEqual(j.TERMINATION_REASON_NAMES[9], "non_finite_state")

    def test_capsize_yaw_independent_and_threshold(self):
        for yaw in (0.0, 1.3, math.pi):
            self.assertEqual(j.validity_reason(state(roll=math.pi, yaw=yaw)), j.TerminationReason.CAPSIZE)
        self.assertEqual(j.validity_reason(state(roll=math.pi / 3 + 1e-6)), j.TerminationReason.CAPSIZE)
        self.assertEqual(j.validity_reason(state(roll=math.pi / 3 - 1e-6)), j.TerminationReason.RUNNING)

    def test_heave_relative_to_surface_and_equilibrium(self):
        lim = j.ValidityLimits()
        self.assertEqual(j.validity_reason(state(z=lim.min_heave_m - 1e-6)), j.TerminationReason.SINKING)
        self.assertEqual(j.validity_reason(state(z=lim.max_heave_m + 1e-6)), j.TerminationReason.AIRBORNE)
        ok = j.RigidState((0, 0, 3.2), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0), 3.0, 0.2)
        self.assertEqual(j.validity_reason(ok), j.TerminationReason.RUNNING)
        with self.assertRaises(ValueError):
            j.ValidityLimits(min_heave_m=0.1)

    def test_nonfinite_and_malformed_fail_closed(self):
        for sample in (state(x=math.nan), state(z=math.inf), state(vx=math.nan),
                       j.RigidState((0, 0, 0), (0, 0, 0, 0), (0, 0, 0), (0, 0, 0)),
                       j.RigidState((0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0)), {}, None):
            self.assertEqual(j.validity_reason(sample), j.TerminationReason.NONFINITE)

    def test_vectorised_validity_matches_scalar(self):
        rng = np.random.default_rng(0)
        n = 400
        com = rng.normal(0, 0.4, (n, 3))
        q = rng.normal(0, 1, (n, 4))
        q[:, 1:3] *= rng.uniform(0, 1, (n, 1))
        lin = rng.normal(0, 1, (n, 3))
        ang = rng.normal(0, 1, (n, 3))
        water, nominal = np.zeros(n), np.zeros(n)
        com[5, 0] = np.nan
        q[6] = 0.0
        ang[7, 2] = np.inf
        lim = j.ValidityLimits()
        batch = j.validity_reasons_batch(com, q, lin, ang, water, nominal, lim.capsize_angle_rad,
                                         lim.min_heave_m, lim.max_heave_m)
        for i in range(n):
            scalar = j.validity_reason(j.RigidState(com[i], q[i], lin[i], ang[i], 0.0, 0.0), lim)
            self.assertEqual(int(batch[i]), int(scalar), i)
        self.assertEqual(set(batch[[5, 6, 7]]), {9})

    def test_priority(self):
        for reason in j.VALIDITY_REASONS:
            r = j.termination_decision(validity=reason, collision=True, out_of_bounds=True, success=True,
                                       timeout=True)
            self.assertEqual((r.terminated, r.truncated, r.reason), (True, False, reason))
        self.assertEqual(j.termination_decision(collision=True, success=True, timeout=True).reason,
                         j.TerminationReason.COLLISION)
        self.assertEqual(j.termination_decision(out_of_bounds=True, collision=True).reason,
                         j.TerminationReason.OUT_OF_BOUNDS)
        self.assertEqual(j.termination_decision(success=True, failure=True, timeout=True).reason,
                         j.TerminationReason.SUCCESS)
        r = j.termination_decision(timeout=True)
        self.assertEqual((r.terminated, r.truncated, r.reason), (False, True, j.TerminationReason.TIMEOUT))

    def test_first_fatal_substep_latched_then_nonfinite_flagged(self):
        e = j.ReachEpisode((0, 0), 1, 10)
        e.observe_physics(state(roll=math.pi))
        e.observe_physics(state(x=math.nan))
        d = e.step(1, state(x=math.nan))
        self.assertEqual(d.reason, j.TerminationReason.CAPSIZE)
        self.assertTrue(e.last_completed.metrics["nonfinite_seen"])
        self.assertTrue(e.last_completed.metrics["path_length_valid_prefix_only"])
        self.assertFalse(e.last_completed.episode_success)
        e.reset((2, 0))
        self.assertEqual(e.validity.reason, j.TerminationReason.RUNNING)

    def test_substep_contact_and_nonfinite_clearance_fail_closed(self):
        e = j.ReachEpisode((0, 0), 1, 10)
        e.observe_physics(state(), collision=True)
        self.assertEqual(e.step(1, state()).reason, j.TerminationReason.COLLISION)
        e = j.ReachEpisode((0, 0), 1, 10)
        e.observe_physics(state(), clearance_m=float("nan"))
        self.assertEqual(e.step(1, state()).reason, j.TerminationReason.COLLISION)
        e = j.ReachEpisode((0, 0), 1, 10)
        e.observe_physics(state(), clearance_m=0.0)
        self.assertEqual(e.step(1, state()).reason, j.TerminationReason.COLLISION)


class ReachTests(unittest.TestCase):
    def test_single_target_latches_and_survives_autoreset(self):
        e = j.ReachEpisode((3, 0), 0.5, 10)
        e.reset((0, 0))
        self.assertFalse(e.step(1, state(1)).done)
        self.assertEqual(e.step(2, state(2.5), clearance_m=0.7).reason, j.TerminationReason.SUCCESS)
        done = e.last_completed
        self.assertEqual((done.time_to_success, done.success_prefix_path_length_m,
                          done.full_episode_path_length_m), (2, 2.5, 2.5))
        f = done.evaluator_fields()
        self.assertEqual(f["episode_path_length"], 2.5)
        self.assertEqual(f["termination_reason_name"], "success")
        self.assertEqual(f["min_clearance_m"], 0.7)
        with self.assertRaises(TypeError):
            done.metrics["x"] = 1
        e.reset((10, 0))
        self.assertIs(e.last_completed, done)
        self.assertEqual(e.metrics.full_episode_path_length_m, 0)

    def test_fast_pass_through_goal_between_samples_counts(self):
        e = j.ReachEpisode((5, 0), 0.3, 10)
        e.reset((0, 0))
        self.assertEqual(e.step(1, state(10)).reason, j.TerminationReason.SUCCESS)

    def test_contact_or_oob_never_success(self):
        for kwargs, expected in ((dict(collision=True), j.TerminationReason.COLLISION),
                                 (dict(out_of_bounds=True), j.TerminationReason.OUT_OF_BOUNDS)):
            e = j.ReachEpisode((0, 0), 1, 1)
            self.assertEqual(e.step(1, state(), **kwargs).reason, expected)
            self.assertFalse(e.last_completed.episode_success)
            self.assertTrue(math.isnan(e.last_completed.time_to_success))

    def test_deadline_sample_required_and_overrun_rejected(self):
        e = j.ReachEpisode((1, 0), 0.5, 1)
        self.assertEqual(e.step(1, state(1)).reason, j.TerminationReason.SUCCESS)
        with self.assertRaises(RuntimeError):
            e.step(2, state(1))
        e.reset((0, 0))
        with self.assertRaises(ValueError):
            e.step(1.1, state(1))
        e.reset((0, 0))
        self.assertEqual(e.step(1, state(0.1)).reason, j.TerminationReason.TIMEOUT)

    def test_prefix_and_full_path_are_distinct_fields(self):
        m = j.EpisodeMetrics((0, 0))
        m.observe(1, (10, 0))
        m.latch_success(1)
        m.observe(2, (20, 0))
        m.latch_success(2)
        self.assertEqual((m.success_prefix_path_length_m, m.full_episode_path_length_m, m.time_to_success),
                         (10, 20, 1))


class DockingTests(unittest.TestCase):
    def dock(self, cls=j.DockingEpisode, **kwargs):
        return cls(C2_BERTH, BB_HULL, 20, **kwargs)

    def test_c1_27_exterior_berth_point_230_never_succeeds(self):
        e = self.dock()
        e.reset((2.30, 0))
        for t in range(1, 21):
            d = e.step(t, state(2.30))
        self.assertEqual(d.reason, j.TerminationReason.TIMEOUT)
        self.assertFalse(e.last_completed.episode_success)
        self.assertFalse(e.last_completed.metrics["berth_contained"])
        self.assertAlmostEqual(e.last_completed.metrics["docking_position_error_m"], 2.30)
        print("C1#27 exterior berth point (2.30, 0): success=False, contained=False")

    def test_full_asymmetric_hull_must_be_inside(self):
        self.assertTrue(C2_BERTH.contains_polygon((0, 0), 0, BB_HULL))
        self.assertFalse(C2_BERTH.contains_polygon((0.3, 0), 0, BB_HULL))          # bow past back face
        self.assertTrue(C2_BERTH.contains_polygon((0, -1.28), 0, BB_HULL))         # starboard side 0.418
        self.assertFalse(C2_BERTH.contains_polygon((0, 1.29), 0, BB_HULL))         # port side 0.512

    def test_valid_entry_and_hold_succeeds_and_reports_terminal_errors(self):
        e = self.dock()
        e.reset((-8.0, 0.0))
        self.assertEqual(dock_fixture(e).reason, j.TerminationReason.SUCCESS)
        m = e.last_completed.metrics
        self.assertEqual(e.last_completed.time_to_success, 8)
        self.assertEqual(m["docking_position_error_m"], 0)
        self.assertTrue(m["berth_entry_latched"])
        self.assertAlmostEqual(m["berth_entry_time_s"], 1 + 0.5 / 0.7)
        self.assertEqual(m["docking_valid_dwell_s"], 5)

    def test_interior_spawn_and_reverse_entry_fail(self):
        e = self.dock()
        e.reset((0, 0))
        for t in range(1, 21):
            e.step(t, state())
        self.assertFalse(e.last_completed.episode_success)
        e = self.dock()
        e.reset((-8, 0))
        e.step(1, state(-6.8, vx=-1))
        self.assertFalse(e.entered)
        e = self.dock()
        e.reset((-8, 0))
        e.step(1, state(-6.8, yaw=math.pi, vx=1))          # stern-first entry
        self.assertFalse(e.entered)

    def test_excursion_between_control_samples_breaks_hold(self):
        e = self.dock()
        e.reset((-8, 0))
        e.step(1, state(-6.8, vx=1))
        e.step(2, state())
        e.step(6, state())
        e.observe_physics(state(vx=1))
        self.assertFalse(e.step(7, state()).done)
        self.assertEqual(e.hold_started_s, 7)

    def test_c1_28_harbour_dock_phase_valid_trajectory_succeeds(self):
        e = self.dock(j.HarborMissionEpisode, spawn_phase=2)
        e.reset((-8.0, 0.0))
        self.assertEqual(dock_fixture(e).reason, j.TerminationReason.SUCCESS)
        m = e.last_completed.metrics
        self.assertTrue(e.last_completed.episode_success)
        self.assertIsNone(m["departure_time_s"])
        self.assertIsNone(m["transit_time_s"])
        self.assertEqual(m["contract_scope"], "harbour_dock_phase")
        print("C1#28 isolated HarborDockPhase with a valid docking fixture: success=True")

    def _harbour(self):
        gates = (j.VirtualGate("d1", (-30.0, 0.0), (1.0, 0.0), 3.0),
                 j.VirtualGate("d2", (-20.0, 0.0), (1.0, 0.0), 3.0))
        exit_gate = j.VirtualGate("exit", (-12.0, 0.0), (1.0, 0.0), 3.0)
        return j.HarborMissionEpisode(C2_BERTH, BB_HULL, 200, departure_gates=gates, transit_gate=exit_gate)

    def test_full_harbour_strict_order(self):
        e = self._harbour()
        e.reset((-35, 0))
        t = 0.0
        for x in (-32, -29, -22, -19, -14, -11, -8, -7.5):
            t += 1
            e.step(t, state(x, vx=1.0))
        self.assertIsNotNone(e.departure_time_s)
        self.assertIsNotNone(e.transit_time_s)
        d = dock_fixture(e, t0=t)
        self.assertEqual(d.reason, j.TerminationReason.SUCCESS)
        m = e.last_completed.metrics
        self.assertLess(m["departure_gate_times_s"][0], m["departure_gate_times_s"][1])
        self.assertLess(m["departure_time_s"], m["transit_time_s"])
        self.assertLessEqual(m["transit_time_s"], m["berth_entry_time_s"])

    def test_full_harbour_skipped_gate_or_docking_first_cannot_succeed(self):
        e = self._harbour()
        e.reset((-35, 6))                       # bypass departure gates laterally
        t = 0.0
        for x in (-32, -29, -22, -19):
            t += 1
            e.step(t, state(x, 6, vx=1.0))
        for x in (-14, -11, -8, -7.5):
            t += 1
            e.step(t, state(x, 0, vx=1.0))
        self.assertIsNone(e.departure_time_s)
        dock_fixture(e, t0=t)
        self.assertFalse(e.done and e.last_completed.episode_success)

    def test_full_mission_requires_gates(self):
        with self.assertRaises(ValueError):
            j.HarborMissionEpisode(C2_BERTH, BB_HULL, 20)


class StationTests(unittest.TestCase):
    def station(self):
        return j.StationKeepingEpisode((0, 0), 1, 120, hold_s=60, late_interval_s=(60, 120), top_dose_start_s=60)

    def test_c1_26_early_hold_before_top_dose_cannot_succeed(self):
        e = self.station()
        for t in range(1, 121):
            d = e.step(t, state(0 if t <= 60 else 2))
        self.assertEqual(d.reason, j.TerminationReason.TIMEOUT)
        m = e.last_completed.metrics
        self.assertFalse(e.last_completed.episode_success)
        # error ramps 0 -> 2 over [60, 61] (linear), then stays 2 until 120
        self.assertAlmostEqual(m["station_final_window_rms_error_m"], math.sqrt((4.0 / 3.0 + 59 * 4.0) / 60.0))
        self.assertEqual(m["station_final_window_in_region_fraction"], 0)
        self.assertEqual(m["station_longest_hold_s"], 60)
        print("C1#26 60 s hold before the top dose: success=False (late-interval rule)")

    def test_hold_spanning_late_interval_succeeds_with_timeout_reason(self):
        e = self.station()
        for t in range(1, 121):
            d = e.step(t, state())
        self.assertEqual(d.reason, j.TerminationReason.TIMEOUT)
        self.assertTrue(e.last_completed.episode_success)
        self.assertEqual(e.last_completed.time_to_success, 120)
        m = e.last_completed.metrics
        self.assertEqual(m["station_final_window_longest_hold_s"], 60)
        self.assertEqual(m["station_final_window_coverage_fraction"], 1)

    def test_top_dose_after_interval_start_rejected(self):
        with self.assertRaises(ValueError):
            j.StationKeepingEpisode((0, 0), 1, 120, top_dose_start_s=61)

    def test_early_failure_leaves_missing_window_missing(self):
        e = self.station()
        e.step(10, state(), collision=True)
        m = e.last_completed.metrics
        self.assertIsNone(m["station_final_window_rms_error_m"])
        self.assertEqual(m["station_final_window_observed_s"], 0)

    def test_substep_excursion_breaks_late_hold(self):
        e = self.station()
        e.step(119, state())
        e.observe_physics(state(2))
        e.step(120, state())
        self.assertFalse(e.last_completed.episode_success)

    def test_rms_is_time_weighted_linear_interpolation(self):
        e = self.station()
        e.step(60, state(0))
        e.step(120, state(3))                   # error ramps 0 -> 3 over the window
        self.assertAlmostEqual(e.last_completed.metrics["station_final_window_rms_error_m"], math.sqrt(3.0))

    def test_thrust_effort_integral(self):
        e = self.station()
        for k in range(120):
            e.observe_physics(state(), dt_s=0.5, motor_thrust_n=(2.0, -1.0))
        for t in range(1, 121):
            e.step(t, state())
        self.assertEqual(e.last_completed.metrics["integrated_squared_thrust_n2s"], (240.0, 60.0))


class PathAndThreadingTests(unittest.TestCase):
    def test_gates_ordered_and_swept(self):
        e = j.OrderedPathEpisode(((0, 0), (5, 0), (10, 0)), 0.5, 10)
        self.assertFalse(e.step(1, state(10, 3)).done)       # skipped gate 1 laterally
        self.assertEqual(e.next_gate, 1)
        e.step(2, state(5.0, 0.2))
        self.assertEqual(e.next_gate, 2)
        self.assertEqual(e.step(3, state(12, 0)).reason, j.TerminationReason.SUCCESS)  # swept through gate 2

    def test_xte_integral_units_and_active_segment(self):
        e = j.OrderedPathEpisode(((0, 0), (10, 0)), 2, 10)
        e.reset((0, 1))
        e.step(10, state(10, 1))
        self.assertEqual(e.last_completed.metrics["path_cross_track_error_integral_m_s"], 10)
        e = j.OrderedPathEpisode(((0, 0), (10, 0), (10, 10)), 0.5, 10)
        e.step(1, state(10, 9))
        self.assertEqual(e.metrics.path_cross_track_error_integral_m_s, 4.5)

    def test_c1_45_fifteen_frames_inside_portal_do_not_pay(self):
        p = j.ThreadingPassage("portal1", 0, 1, 2)
        p.update("portal1", (-1, 0), 0.6, 0.45)
        for _ in range(15):
            self.assertEqual(p.update("portal1", (0, 0), 0.6, 0.45), 0)
        self.assertFalse(p.paid)
        self.assertEqual(p.update("portal1", (1.7, 0), 0.6, 0.45), 5)
        self.assertEqual(p.update("portal1", (2, 0), 0.6, 0.45), 0)
        print("C1#45 fifteen in-portal frames: no bonus; bonus only after full-hull exit")

    def test_threading_requires_clean_identified_passage(self):
        p = j.ThreadingPassage("portal1", 0, 1, 2)
        p.update("portal1", (-1, 0), 0.6, 0.45)
        p.update("portal1", (0, 0), 0.6, 0.45, collision=True)
        self.assertEqual(p.update("portal1", (2, 0), 0.6, 0.45), 0)
        other = j.ThreadingPassage("portal1", 0, 1, 2)
        other.update("portal2", (-1, 0), 0.6, 0.45)
        self.assertEqual(other.update("portal2", (2, 0), 0.6, 0.45), 0)
        missed = j.ThreadingPassage("portal1", 0, 1, 2)
        missed.update("portal1", (-1, 4), 0.6, 0.45)
        self.assertEqual(missed.update("portal1", (2, 0), 0.6, 0.45), 0)
        self.assertTrue(missed.failed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
