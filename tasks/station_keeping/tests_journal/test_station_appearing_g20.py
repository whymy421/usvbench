"""G20 / G20b / G20c / G20d appearing obstacle on station keeping (final rule: station_route_block, G20d), CPU only.

Run: python -B tasks/station_keeping/tests_journal/test_station_appearing_g20.py

The REAL station_keeping_env.py journal branch (stubbed isaaclab, toy integrator, REAL runtime /
condition layer / E4 bridge / committed seed-42 D7 pools) with the appearing hooks and a 2 x 2 appearing
pool admitted on the REAL layouts with the active ladder and PINNED in a temporary family folder
(appearing_pin_fixture; the production DIAG42 pools live in the pool bundle):
  * the appearing slot is its own storage (slot 1 = 0 static + 1 D7) and the header names the pinned pool;
  * the route-block rule (G20d, owner decisions 1B / 2B): the cylinder stands ON the nominal approach route at
    0.45 x the start-to-centre distance (it blocks the nominal approach), its surface keeps max(R_hold + hw,
    reach + hw) from the station centre (hold circle clear, the centre is clear at every heading; REV-G20c M6:
    whether the station is a G17 evade target is report only); BlueBoat D1 is not applicable by design (refused at reset);
  * trigger mechanics: the fixture's hold controller drives to the station; every OFFERED main dose (BlueBoat
    D2..D4; WAM-V D4) triggers at the first control step with gap <= d_k while the vessel moves, the cylinder is
    visible at once (the ray block sees it) and physical (the swept-contact latch feeds the contract collision);
  * D0 is bit-identical to the suite absent (observations, rewards, dones, wrenches, poses, the
    completed record's conditions / D7 stamp / contract result / every other pairing primitive);
  * a key past the 2 x 2 grid runs D0 with the single reason beyond_offered_grid;
  * the bridge replays the nominal approach of every admitted key at the control rate and triggers
    within [d_k - one control step of nominal motion, d_k].
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import station_journal_fixtures as F

RESTORE = None
UNPIN = None
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
CONTROL_DT = 1.0 / 60.0
POOLS = {}


def setUpModule():
    global RESTORE, UNPIN, js, pairing, dyn, AC, AP, APOOL, ART, AST, PIN, AL
    RESTORE = F.install_isaaclab_stubs()
    js = F.ns("station_keeping.journal_station")
    pairing = F.ns("_shared.conditions.pairing")
    dyn = F.ns("_shared.conditions.dynamic")
    AC = F.ns("_shared.conditions.appearing.config")
    AP = F.ns("_shared.conditions.appearing.placement")
    APOOL = F.ns("_shared.conditions.appearing.pool")
    ART = F.ns("_shared.conditions.appearing.runtime")
    AST = F.ns("_shared.conditions.appearing.stamp")
    AL = F.ns("_shared.conditions.appearing.ladder")
    spec = importlib.util.spec_from_file_location(
        "appearing_pin_fixture", F.TASKS / "_shared" / "conditions" / "appearing" / "tests" / "appearing_pin_fixture.py")
    PIN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PIN)
    adm, fams = F.ns("_shared.conditions.appearing.admission"), F.ns("_shared.conditions.appearing.families")
    for vessel in VESSELS:
        POOLS[vessel] = PIN.build(adm, fams, "station_keeping", vessel, 2, 2, seed=F.SEED)
    _, UNPIN = PIN.install(APOOL, fams, list(POOLS.values()))


def tearDownModule():
    if UNPIN is not None:
        UNPIN()
    if RESTORE is not None:
        RESTORE()


def pinned(vessel):
    return APOOL.load_pinned_pool("station_keeping", "station_keeping", vessel, F.SEED, 2)[0]


def first_admitted(pool, num_envs, episode=0):
    for env in range(num_envs):
        if pool.is_admitted(env, episode):
            return env
    raise AssertionError("no admitted row")


class Wiring(unittest.TestCase):
    def test_slot_storage_and_header(self):
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, tokens=("appearing=D2",)))
        self.assertEqual(env._appearing_slot, 1)
        self.assertEqual(tuple(env._appearing_centers_w.shape), (2, 1, 2))
        header = env.journal_certificate_header()["appearing_hazards_header"]
        self.assertTrue(header["pool_pinned"])
        self.assertTrue(header["pool_file"].endswith("station_keeping__blueboat_v2__s42.json.xz"))
        self.assertEqual(header["grid"], {"num_envs": 2, "episodes": 2})
        self.assertEqual(header["trigger_gaps_m"], AC.trigger_gaps_m(env._appearing_bridge.t_avoid_s, "blueboat_v2"))
        self.assertEqual(header["ladder_option"], AC.LADDER_OPTION)
        self.assertEqual(len(header["per_dose_counts"]), AC.DOSE_COUNT)
        self.assertIn("collision_model", header)

    def test_route_block_keeps_the_station_clear(self):
        checked = 0
        for vessel in VESSELS:
            pool = pinned(vessel)
            for env_i, episode in pool.offered:
                doc = pool.placement(env_i, episode)
                if doc is None:
                    continue
                site, cyl = doc["site"], doc["cylinder"]
                self.assertEqual(site["rule"], "station_route_block")
                context = js.station_task_context(js.build_station_layout(2, vessel, F.SEED, env_i, episode), vessel)
                fp = context.envelope.footprint
                hw = 0.5 * fp.beam_m
                cx, cy = site["station_centre"]
                self.assertEqual((cx, cy), (context.hold_circle.x, context.hold_circle.y))
                surface = math.hypot(cyl["x_m"] - cx, cyl["y_m"] - cy) - cyl["radius_m"]
                self.assertGreaterEqual(surface, context.hold_circle.radius + hw - 1e-9)       # hold circle clear
                self.assertEqual(site["g17_evade_target_clear"],                         # report only (M6)
                                 surface >= AP.SK_G17_EVADE_CLEARANCE_M[vessel])
                for k in range(16):                                                       # every heading
                    centre = dyn.Pose(cx, cy, 2.0 * math.pi * k / 16.0, 0.0)
                    self.assertGreaterEqual(AP.gap_at(centre, cyl["x_m"], cyl["y_m"], cyl["radius_m"], fp), hw - 1e-9)
                self.assertLess(site["nominal_min_gap_m"], 0.0)                          # blocks the route
                start = context.initial_pose
                self.assertAlmostEqual(math.hypot(cyl["x_m"] - cx, cyl["y_m"] - cy),
                                       0.45 * math.hypot(cx - start.x, cy - start.y), places=9)
                checked += 1
        self.assertGreater(checked, 0)


class TriggerAtDeclaredGap(unittest.TestCase):
    def run_dose(self, vessel, dose, num_envs=2):
        pool = pinned(vessel)
        row = first_admitted(pool, num_envs)
        env = F.make_env(F.journal_cfg(vessel, num_envs=num_envs, tokens=("appearing=D%d" % dose,)))
        env.reset()
        bridge = env._appearing_bridge
        self.assertTrue(bridge.rows[row]["active"])
        x, y, r = bridge.rows[row]["cylinder"]
        footprint, t_avoid = bridge.footprint, bridge.t_avoid_s
        d_k = AC.trigger_gap_m(dose, t_avoid)
        origin = env.scene.env_origins[row, :2]
        limit = int(200.0 / CONTROL_DT)
        for _ in range(limit):
            pose = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
            self.assertFalse(bool(env._appearing_visible[row, 0]))
            rays_before = env._journal_ray_ranges()[row].clone()
            _, _, te, tr, _ = env.step(F.hold_controller(env))
            self.assertFalse(bool((te | tr)[row]), "episode ended before the trigger")
            if bridge.triggered(row):
                trig = bridge.rows[row]["trigger"]
                now = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
                gap = AP.gap_at(now, x, y, r, footprint)
                self.assertAlmostEqual(trig["gap_m"], gap, places=9)
                self.assertLessEqual(trig["gap_m"], d_k)
                self.assertGreater(trig["previous_check_gap_m"], d_k)
                motion = math.hypot(now.x - pose.x, now.y - pose.y) + \
                    footprint.reach_m * abs(math.remainder(now.yaw - pose.yaw, 2 * math.pi))
                self.assertGreater(trig["gap_m"], d_k - motion - 1e-9)
                self.assertGreater(abs(now.speed), 0.0)                       # the vessel still moves
                self.assertEqual(trig["closing_speed_mps"], now.speed * math.cos(trig["bearing_rad"]))
                self.assertTrue(bool(env._appearing_visible[row, 0] & env._appearing_physical[row, 0]))
                centre = env._appearing_centers_w[row, 0] - origin
                self.assertAlmostEqual(float(centre[0]), x, places=4)
                self.assertAlmostEqual(float(centre[1]), y, places=4)
                rays_after = env._journal_ray_ranges()[row]
                self.assertTrue(bool((rays_after < rays_before - 1e-6).any()))   # visible to the ray block
                return env, row
        self.fail("the trigger gap was never reached")

    def test_every_dose_triggers_at_the_declared_gap_blueboat(self):
        for dose in AL.offered_main_doses("station_keeping", "blueboat_v2"):   # D2..D4 (D1 not applicable)
            with self.subTest(dose=dose):
                self.run_dose("blueboat_v2", dose)

    def test_wamv_triggers_at_the_declared_gap(self):
        # G20d: the cylinder blocks the approach, so even the smallest gap D4 (1.8 m) triggers for the fixture
        # controller's straight-to-the-station line
        self.run_dose("wamv_vrx_v2_opt1", AC.MAIN_DOSES[-1])

    def test_blueboat_d1_is_refused_at_reset(self):
        # G20d (decision 2B): D1 is not applicable by design for BlueBoat station keeping (never offered)
        pinned("blueboat_v2")
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, tokens=("appearing=D1",)))
        with self.assertRaisesRegex(ValueError, "not applicable by design"):
            env.reset()

    def test_contact_terminates_with_a_collision(self):
        env, row = self.run_dose("blueboat_v2", 2)
        x, y, _ = env._appearing_bridge.rows[row]["cylinder"]
        origin = env.scene.env_origins[row, :2]
        for _ in range(int(60.0 / CONTROL_DT)):
            pose = env.robot.data.root_link_pose_w
            fwd = js.bow_axis_xy(pose[:, 3:7])
            rel = torch.tensor([x, y]) + origin - pose[:, :2]
            err = torch.atan2(fwd[:, 0] * rel[:, 1] - fwd[:, 1] * rel[:, 0], (fwd * rel).sum(-1))
            action = torch.stack((torch.full_like(err, 0.6), torch.clamp(1.5 * err, -1.0, 1.0)), -1)
            _, _, te, tr, _ = env.step(action)
            if bool((te | tr)[row]):
                break
        result = env.journal_episode_result(row)
        self.assertEqual(result["termination_reason_name"], "collision")
        stamp = env.journal_episode_record(row)["appearing_hazards"]["payload"]
        self.assertTrue(stamp["runtime"]["exposure_realized"])
        self.assertTrue(stamp["runtime"]["swept_contact_latched"])
        self.assertEqual(stamp["runtime"]["termination_reason"], "collision")


class ZeroDoseIdentity(unittest.TestCase):
    def episode_run(self, vessel, tokens, steps=40):
        env = F.make_env(F.journal_cfg(vessel, num_envs=2, tokens=tokens))
        obs, _ = env.reset()
        out = [obs["policy"].clone()]
        for k in range(steps):
            if k == steps - 3:
                env.robot.data.root_link_pose_w[1, 0] += 1.0e3            # row 1 ends (out of bounds)
            o, r, te, tr, _ = env.step(F.hold_controller(env))
            out += [o["policy"].clone(), r.clone(), te.clone(), tr.clone(),
                    env.robot.data.root_link_pose_w.clone(), env.robot.applied[-1][0].clone(),
                    env.robot.applied[-1][1].clone()]
        return env, out, copy.deepcopy(env.journal_episode_record(1))

    def check(self, vessel):
        absent, a, ra = self.episode_run(vessel, ())
        present, b, rb = self.episode_run(vessel, ("appearing=D0",))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertNotIn("appearing_hazards", ra)
        self.assertEqual(rb["appearing_hazards"]["payload"]["assignment"]["dose"], 0)
        self.assertFalse(rb["appearing_hazards"]["payload"]["runtime"]["exposure_realized"])
        for field in ("conditions", "dynamic_hazards"):
            self.assertEqual(json.dumps(ra[field], sort_keys=True), json.dumps(rb[field], sort_keys=True), field)
        self.assertTrue(F.same(absent.journal_episode_result(1), present.journal_episode_result(1)))
        hb = dict(rb["scenario_hashes"])
        hb.pop(pairing.APPEARING_PRIMITIVE)
        self.assertEqual(ra["scenario_hashes"], hb)
        self.assertEqual(absent.dynamic_hazard_header, present.dynamic_hazard_header)

    def test_d0_is_bit_identical_to_the_suite_absent_blueboat(self):
        self.check("blueboat_v2")

    def test_d0_is_bit_identical_to_the_suite_absent_wamv(self):
        self.check("wamv_vrx_v2_opt1")


class BeyondGrid(unittest.TestCase):
    def test_beyond_grid_key_runs_d0_with_the_single_reason(self):
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, tokens=("appearing=D3",)))
        env.reset()
        row = 1
        for _ in range(2):                                   # episodes 0 and 1 end early (out of bounds)
            env.step(F.hold_controller(env))
            env.robot.data.root_link_pose_w[row, 0] += 1.0e3
            _, _, te, tr, _ = env.step(F.hold_controller(env))
            self.assertTrue(bool(te[row] | tr[row]))
        self.assertEqual(int(env._jscenario.episode_indices()[row]), 2)       # one past the 2 x 2 grid
        self.assertFalse(env._appearing_bridge.rows[row]["active"])
        env.step(F.hold_controller(env))
        env.robot.data.root_link_pose_w[row, 0] += 1.0e3
        env.step(F.hold_controller(env))
        record = env.journal_episode_record(row)
        kind, reasons = AST.exposure_of(record["appearing_hazards"])
        self.assertEqual((kind, reasons), ("not_admitted", ("beyond_offered_grid",)))
        payload = record["appearing_hazards"]["payload"]
        self.assertEqual((payload["runtime"]["assigned"], payload["runtime"]["itt_count"]), (False, 0))


class NominalReplayTrigger(unittest.TestCase):
    def test_every_admitted_key_triggers_within_one_control_step(self):
        for vessel in VESSELS:
            pool = pinned(vessel)
            checked = 0
            for env_i, episode in pool.offered:
                if not pool.is_admitted(env_i, episode):
                    continue
                context = js.station_task_context(js.build_station_layout(2, vessel, F.SEED, env_i, episode),
                                                  vessel)
                plan = AP.nominal_plan(context, AC.AppearingSuiteConfig(enabled=True, dose=1, vessel_id=vessel))
                envelope = context.envelope
                for dose in AC.offered_doses("station_keeping", vessel):                 # G20d
                    a = pool.assignment(dyn.EpisodeKey(F.SEED, env_i, episode, AC.hazard_group("station_keeping")),
                                        dose)
                    a = a.__class__(**{**a.__dict__, "key": dyn.EpisodeKey(F.SEED, 0, episode, a.key.hazard_group)})
                    bridge = ART.AppearingBridge("station_keeping", 1, 1.0 / 120.0, envelope.top_speed_mps,
                                                 envelope.footprint, envelope.t_avoid_s)
                    bridge.reset_rows([0], {0: a})
                    if a.candidate["nominal_trigger_s"] is None:      # calibration rung not reached nominally
                        self.assertIn(dose, AC.SUPPLEMENTARY_DOSES)
                        continue
                    tick = 0
                    bridge.step({0: 0}, {0: plan.pose_at(0.0)})
                    while not bridge.triggered(0) and tick / 120.0 < plan.end_s:
                        for _ in range(2):
                            tick += 1
                            bridge.step({0: tick}, {0: plan.pose_at(min(plan.end_s, tick / 120.0))})
                        bridge.step({0: tick}, {0: plan.pose_at(min(plan.end_s, tick / 120.0))}, check_trigger=True)
                    self.assertTrue(bridge.triggered(0), (vessel, env_i, episode, dose))
                    trig = bridge.rows[0]["trigger"]
                    self.assertLessEqual(trig["gap_m"], a.trigger_gap_m)
                    self.assertGreater(trig["gap_m"], a.trigger_gap_m - envelope.reference_speed_mps * CONTROL_DT
                                       - 1e-6)
                    self.assertAlmostEqual(trig["time_s"], a.candidate["nominal_trigger_s"], delta=CONTROL_DT + 1e-9)
                    # still moving (main rungs): the nominal speed at the crossing is >= 0.25 u_ref (placement),
                    # the sample at most one control step later
                    if dose in AC.MAIN_DOSES:
                        self.assertGreaterEqual(trig["pose"][3], AP.SK_MIN_TRIGGER_SPEED_FRACTION
                                                * AC.REFERENCE_SPEED_MPS - envelope.max_decel_mps2 * CONTROL_DT - 1e-9)
                    checked += 1
            self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
