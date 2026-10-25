"""CPU tests of the twelve task adapters, admission accounting, runtime and bridge."""
import json
import math
import unittest
from dataclasses import replace

import fixtures as f

d = f.d
BB, WAMV, WAMV_OPT2 = f.VESSELS   # E4: final D5 registry names (WAMV = default opt1)


def admitted(task, vessel, env_row=0, episode=0, **ctx_overrides):
    env = f.envelope(vessel)
    ctx = f.context(task, env, **ctx_overrides)
    key = f.key(task, env=env_row, episode=episode)
    ladder = d.build_ladder(ctx, key, f.suite(env))
    return ctx, key, ladder, d.admit_ladders([(key, ladder)])


class TestAllTwelveTasks(unittest.TestCase):
    """Every task admits a full D0..D4 ladder on BlueBoat and on WAM-V (current VRX)."""

    @classmethod
    def setUpClass(cls):
        cls.pools = {}
        for vessel in (BB, WAMV):
            for task in d.TASKS:
                row = 1 if task == "station_keeping" else 0
                cls.pools[(vessel, task)] = admitted(task, vessel, env_row=row)

    def test_twelve_adapters(self):
        self.assertEqual(len(d.ADAPTERS), 12)
        self.assertEqual(set(d.ADAPTERS), set(d.TASKS))
        self.assertEqual(d.make_adapter("harbour_mission").task, "harbour")

    def test_every_task_every_dose_admitted_on_both_vessels(self):
        for (vessel, task), (_, key, _, pool) in self.pools.items():
            with self.subTest(vessel=vessel, task=task):
                self.assertEqual(pool.keys_admitted, 1, [(r.dose, r.reason) for r in pool.rejections])
                # G23e: station keeping BlueBoat D1 is not applicable by design (never offered)
                na = (1,) if (vessel, task) == (BB, "station_keeping") else ()
                self.assertEqual(pool.per_dose_counts, tuple((0, 0) if d_ in na else (1, 1) for d_ in range(5)))
                for dose in (d_ for d_ in range(5) if d_ not in na):
                    a = pool.assignment(key, dose)
                    self.assertTrue(d.replay_witness(a.candidate.request, a.witness.trajectory).accepted)

    def test_dose_semantics_per_task(self):
        for (vessel, task), (ctx, key, _, pool) in self.pools.items():
            with self.subTest(vessel=vessel, task=task):
                env = ctx.envelope
                doses = [dose for dose in range(5) if (key, dose) in pool.assignments]   # G23e: BB SK has no D1
                encs = [pool.assignment(key, dose).candidate.request.encounter for dose in doses]
                if task == "moving_basin":
                    mus = [e.meta("mu") for e in encs]
                    self.assertEqual(tuple(mus), d.NATIVE_RATIOS)
                    self.assertTrue(all(len(e.intruders) == len(ctx.native_geometry) for e in encs))
                    continue
                self.assertEqual(encs[0].intruders, ())
                paths = {(e.intruders[0].x0, e.intruders[0].y0, e.intruders[0].vx, e.intruders[0].vy,
                          e.intruders[0].start_s, e.intruders[0].stop_s) for e in encs[1:]}
                self.assertEqual(len(paths), 1, "physical path must not depend on dose")
                for dose, e in zip(doses[1:], encs[1:]):
                    self.assertAlmostEqual(e.nominal_ttc_s, d.DOSE_TTC_RATIOS[dose] * env.t_avoid_s)
                    # intruder size is 0.5 of THIS vessel's beam (diameter)
                    self.assertAlmostEqual(e.intruders[0].radius_m, 0.25 * env.beam_m)
                ranges = [pool.assignment(key, dose).candidate.reveal_evidence.detection_range_m
                          for dose in doses[1:]]
                self.assertEqual(ranges, sorted(ranges, reverse=True), "less warning at higher dose")

    def test_station_encounter_on_the_approach_and_harbour_stage_declared(self):
        # G23e: the station-keeping intruder meets the own ship during the approach, before the scoring window
        # (the retired design revealed it at the settled clock); details in test_g23e_sk_approach.py
        for vessel in (BB, WAMV):
            ctx, key, _, pool = self.pools[(vessel, "station_keeping")]
            for dose in range(2, 5):
                enc = pool.assignment(key, dose).candidate.request.encounter
                self.assertLess(enc.intruders[0].reveal_s, enc.meta("conflict_time_s"))
                self.assertLess(enc.meta("conflict_time_s"), ctx.settled_phase_start_s)
                self.assertEqual(enc.meta("sk_design"), "sk_approach_crossing_v1")
            _, key, _, pool = self.pools[(vessel, "harbour")]
            self.assertIn(pool.assignment(key, 3).candidate.stage, ("exit_gate", "transit", "berth_approach"))

    def test_capacity_reservation(self):
        for (vessel, task), (ctx, key, _, pool) in self.pools.items():
            cand = pool.assignment(key, 4).candidate
            if task == "moving_basin":
                self.assertEqual(cand.capacity.dynamic_reserve, 0)
                self.assertEqual(cand.base_slots, ctx.base_slots)
            else:
                self.assertEqual(cand.capacity.dynamic_slots, (ctx.base_max,))
                self.assertEqual(cand.capacity.capacity, ctx.base_max + 1)


class TestAdmissionAccounting(unittest.TestCase):
    def test_regeneration_is_deterministic_and_keyed(self):
        env = f.envelope(BB)
        ctx = f.context("reach", env)
        first = d.build_ladder(ctx, f.key("reach"), f.suite(env))
        again = d.build_ladder(ctx, f.key("reach"), f.suite(env))
        self.assertEqual([d.digest(c) for c in first], [d.digest(c) for c in again])
        other = d.build_ladder(ctx, f.key("reach", env=1), f.suite(env))
        self.assertNotEqual(first[2].request.encounter.metadata, other[2].request.encounter.metadata)
        with self.assertRaises(ValueError):
            d.build_candidate(ctx, d.EpisodeKey(42, 0, 0, "waves"), 1, f.suite(env))

    def test_joint_admission_removes_the_key_at_every_dose(self):
        # E4: with the final registry and the 1.0 m/s journal reference no journal
        # vessel exceeds the 30 m fan any more (D7 used the classic WAM-V at 2.0 m/s,
        # head-on D1 34.6 m). The same mechanism is exercised with a deliberately
        # short 22 m sensor: WAM-V opt1 head-on D1 needs 25.3 m, D2 19.7 m.
        short = d.SensorSpec(ray_count=36, max_range_m=22.0, source="deliberately short test sensor")
        _, key, ladder, pool = admitted("ring", WAMV, sensor=short)
        self.assertEqual(pool.keys_admitted, 0)
        reasons = {(r.dose, r.reason) for r in pool.rejections}
        self.assertIn((1, "detection_range_exceeds_existing_sensor_range"), reasons)
        self.assertIn((0, "joint_ladder_member_rejected"), reasons)
        self.assertEqual(pool.per_dose_counts[0], (1, 0))
        loose = d.admit_ladders([(key, ladder)], joint=False)
        self.assertEqual(loose.per_dose_counts[1], (1, 0))
        self.assertEqual(loose.per_dose_counts[4][1], 1)

    def test_duplicate_keys_are_errors(self):
        _, key, ladder, _ = admitted("reach", BB)
        with self.assertRaises(ValueError):
            d.admit_ladders([(key, ladder), (key, ladder)])

    def test_task_without_existing_sensor_fails_closed(self):
        # E4: D7's fail-closed behaviour is now the journal_ray_sensor_all_tasks=False
        # branch (CONTEXT_R3 decision 14 makes True the default; see test_e4.py).
        env = f.envelope(BB)
        ctx = f.context("path_following", env, sensor=None)
        key = f.key("path_following")
        ladder = d.build_ladder(ctx, key, f.suite(env, journal_ray_sensor_all_tasks=False))
        self.assertIsInstance(ladder[0], d.Candidate)
        for dose in range(1, 5):
            self.assertEqual(ladder[dose].reason, "task_has_no_existing_range_sensor")
        opted = d.build_ladder(ctx, key, f.suite(env, allow_unsensed=True, journal_ray_sensor_all_tasks=False))
        pool = d.admit_ladders([(key, opted)])
        self.assertEqual(pool.keys_admitted, 1)
        self.assertFalse(pool.assignment(key, 2).candidate.perceivable)

    def test_zero_dose_config_and_c4_mapping(self):
        with self.assertRaises(ValueError):
            d.DynamicSuiteConfig(enabled=False, dose=2)
        cfg = f.suite(f.envelope(BB), dose=3)
        mapped = cfg.to_c4_override()
        self.assertFalse(mapped["enabled"])
        self.assertIn("dose=D3", mapped["definition"])


class TestRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.key, _, cls.pool = admitted("reach", BB)
        # G23c: rule S explicitly (layer 2 makes S2 the default; S2 has its own tests in test_g23c_s2.py)
        cls.cfg = d.DynamicRuntimeConfig(motion_rule=d.release.RUNTIME_RULE_VERSION)

    def assignment(self, dose):
        return self.pool.assignment(self.key, dose)

    def test_controller_independence(self):
        # G23b: controller independence of the object motion is the v2.5 rule, kept as the diagnostic
        # motion rule "pinned_v0"; the default rule S (release.py) is checked below and in test_g23b_release.py.
        a = self.assignment(2)
        plan = d.build_transit_plan(self.ctx.initial_pose, self.ctx.envelope,
                                    (self.ctx.goal,), 0.25, self.ctx.deadline_s)
        pinned = d.DynamicRuntimeConfig(motion_rule=d.PINNED_RULE_VERSION)
        nominal, parked = d.make_adapter("reach"), d.make_adapter("reach")
        for adapter in (nominal, parked):
            adapter.reset((0,), pinned, {0: a})
            adapter.activate((0,), pinned)
        for step in range(0, int(plan.end_s * 4)):
            t = step * 0.25
            one = nominal.update(d.RuntimeFrame(((0, t),), ((0, plan.pose_at(t)),)), pinned)[0]
            two = parked.update(d.RuntimeFrame(((0, t),), ((0, d.Pose(-50, -50, 2.0, 0.0)),)), pinned)[0]
            self.assertEqual([(s.slot, s.x, s.y, s.radius_m, s.physical) for s in one],
                             [(s.slot, s.x, s.y, s.radius_m, s.physical) for s in two])
        s1, s2 = nominal.finish(0, "collision"), parked.finish(0, "timeout")
        self.assertIsNotNone(s1["payload"]["runtime"]["first_detection_s"])
        reveal = a.candidate.request.encounter.intruders[0].reveal_s
        self.assertAlmostEqual(s1["payload"]["runtime"]["first_detection_s"], math.ceil(reveal * 4) / 4)
        self.assertIsNone(s2["payload"]["runtime"]["first_detection_s"])
        self.assertEqual(s2["payload"]["runtime"]["non_exposure_reason"], "not_detected_at_sampled_times")
        self.assertEqual(s1["payload"]["assignment"], s2["payload"]["assignment"])
        self.assertEqual(s1["payload"]["runtime"]["itt_count"], s2["payload"]["runtime"]["itt_count"], 1)

    def test_release_sync_rule_times_the_object_by_the_own_ship(self):
        # G23b default rule S: the nominal own ship releases at t_hold and sees the pinned motion from then
        # on (same first detection as above); a parked own ship never releases (object stays at its hold point).
        a = self.assignment(2)
        plan = d.build_transit_plan(self.ctx.initial_pose, self.ctx.envelope,
                                    (self.ctx.goal,), 0.25, self.ctx.deadline_s)
        obj = a.candidate.request.encounter.intruders[0]
        nominal, parked = d.make_adapter("reach"), d.make_adapter("reach")
        for adapter in (nominal, parked):
            adapter.reset((0,), self.cfg, {0: a})
            adapter.activate((0,), self.cfg)
        t_hold = nominal.episodes[0].release.t_hold
        for step in range(0, int(plan.end_s * 4)):
            t = step * 0.25
            one = nominal.update(d.RuntimeFrame(((0, t),), ((0, plan.pose_at(t)),)), self.cfg)[0]
            two = parked.update(d.RuntimeFrame(((0, t),), ((0, d.Pose(-50, -50, 2.0, 0.0)),)), self.cfg)[0]
            self.assertEqual((one[0].x, one[0].y), obj.position(max(t, t_hold)))
            self.assertEqual((two[0].x, two[0].y), obj.position(t_hold))
        s1, s2 = nominal.finish(0, "collision"), parked.finish(0, "timeout")
        reveal = obj.reveal_s
        self.assertAlmostEqual(s1["payload"]["runtime"]["first_detection_s"], math.ceil(reveal * 4) / 4)
        self.assertEqual(s1["payload"]["runtime"]["release"]["release_time_s"], t_hold)
        self.assertEqual(s2["payload"]["runtime"]["non_exposure_reason"], "not_released")
        self.assertEqual(s1["payload"]["assignment"], s2["payload"]["assignment"])

    def test_physical_before_visible(self):
        a = self.assignment(4)
        adapter = d.make_adapter("reach")
        adapter.reset((0,), self.cfg, {0: a})
        adapter.activate((0,), self.cfg)
        sample = adapter.update(d.RuntimeFrame(((0, 0.0),), ((0, self.ctx.initial_pose),)), self.cfg)[0][0]
        self.assertTrue(sample.physical)
        self.assertFalse(sample.visible)
        self.assertEqual(sample.slot, a.candidate.capacity.dynamic_slots[0])

    def test_zero_dose_is_no_hazard(self):
        adapter = d.make_adapter("reach")
        adapter.reset((0,), self.cfg, {0: self.assignment(0)})
        adapter.activate((0,), self.cfg)
        for t in (0.0, 5.0, 10.0):
            self.assertEqual(adapter.update(d.RuntimeFrame(((0, t),), ((0, self.ctx.initial_pose),)),
                                            self.cfg)[0], ())
        stamp = adapter.finish(0, "success")
        self.assertEqual(stamp["payload"]["runtime"]["non_exposure_reason"], "zero_dose")
        self.assertEqual(stamp["payload"]["assignment"]["intruders"], [])

    def test_lifecycle_guards_reasons_and_stamp_roundtrip(self):
        adapter = d.make_adapter("reach")
        adapter.reset((0,), self.cfg, {0: self.assignment(1)})
        with self.assertRaises(ValueError):
            adapter.update(d.RuntimeFrame(((0, 0.0),)), self.cfg)          # not armed
        adapter.activate((0,), self.cfg)
        with self.assertRaises(TypeError):
            adapter.update([0.5], self.cfg)                                  # C4 voyage fraction
        with self.assertRaises(ValueError):
            adapter.reset((0,), self.cfg, {0: self.assignment(1)})          # not finished
        with self.assertRaises(ValueError):
            adapter.finish(0, "crashed")                                     # unknown reason
        stamp = adapter.finish(0, "capsize")                                 # D4 validity reason
        self.assertEqual(stamp["payload"]["runtime"]["non_exposure_reason"], "sensor_observations_missing")
        self.assertEqual(d.read_stamp(json.dumps(stamp)), stamp)
        self.assertEqual(adapter.finish(0, "capsize"), stamp)
        tampered = json.loads(json.dumps(stamp))
        tampered["payload"]["assignment"]["dose"] = 4
        with self.assertRaises(ValueError):
            d.read_stamp(tampered)

    def test_modified_assignment_rejected(self):
        a = self.assignment(3)
        forged = replace(a, candidate=replace(a.candidate, settled_phase_start_s=1.0))
        with self.assertRaises(ValueError):
            d.make_adapter("reach").reset((0,), self.cfg, {0: forged})
        with self.assertRaises(ValueError):
            d.make_adapter("scatter").reset((0,), self.cfg, {0: a})


class TestBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.key, _, cls.pool = admitted("reach", BB)
        cls.plan = d.build_transit_plan(cls.ctx.initial_pose, cls.ctx.envelope, (cls.ctx.goal,), 0.25,
                                        cls.ctx.deadline_s)

    def run_nominal(self, dose, dt, ticks_per_call=1):
        bridge = d.HazardBridge(d.make_adapter("reach"), d.DynamicRuntimeConfig(motion_rule=d.release.RUNTIME_RULE_VERSION), dt,
                                self.ctx.envelope.top_speed_mps)
        bridge.reset_rows((0,), {0: self.pool.assignment(self.key, dose)})
        tick = 0
        while tick * dt <= self.plan.end_s:
            pose = self.plan.pose_at(tick * dt)
            bridge.step({0: tick}, {0: pose})
            bridge.step({0: tick}, {0: pose})      # _get_dones re-consumes the same tick
            tick += ticks_per_call
        return bridge

    def test_nominal_ownship_is_hit_and_latched_even_with_coarse_ticks(self):
        enc = self.pool.assignment(self.key, 2).candidate.request.encounter
        conflict, closing = enc.meta("conflict_time_s"), enc.meta("closing_speed_mps")
        for dt in (1.0 / 120.0, 1.0):
            bridge = self.run_nominal(2, dt)
            self.assertTrue(bridge.contact(0), dt)
            stamp = bridge.finish(0, "collision")
            first = stamp["payload"]["runtime"]["bridge"]["first_swept_contact_interval_start_s"]
            # never late; early by at most one tick plus the declared conservative inflation
            inflation = (self.ctx.envelope.top_speed_mps + enc.intruders[0].speed_bound_mps) * dt / 2
            self.assertLessEqual(first, conflict + 1e-9)
            self.assertGreaterEqual(first, conflict - dt - inflation / closing - 1e-6)

    def test_zero_dose_never_contacts_and_ticks_are_not_double_counted(self):
        dt = 0.5
        bridge = self.run_nominal(0, dt)
        self.assertFalse(bridge.contact(0))
        stamp = bridge.finish(0, "success")
        ticks = int(math.floor(self.plan.end_s / dt + 1e-9)) + 1
        self.assertEqual(stamp["payload"]["runtime"]["bridge"]["swept_intervals"], ticks - 1)
        self.assertEqual(stamp["payload"]["runtime"]["observer_updates"], ticks)
        with self.assertRaises(ValueError):
            bridge.step({0: 5}, {0: self.ctx.initial_pose})

    def test_tick_cannot_go_backwards(self):
        bridge = d.HazardBridge(d.make_adapter("reach"), d.DynamicRuntimeConfig(motion_rule=d.release.RUNTIME_RULE_VERSION), 0.1, 3.0)
        bridge.reset_rows((0,), {0: self.pool.assignment(self.key, 1)})
        bridge.step({0: 3}, {0: self.ctx.initial_pose})
        with self.assertRaises(ValueError):
            bridge.step({0: 2}, {0: self.ctx.initial_pose})


if __name__ == "__main__":
    unittest.main(verbosity=2)
