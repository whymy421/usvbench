"""G20b appearing obstacle on journal path hazard (REAL env methods, stubbed Isaac Lab, CPU).

The REAL PathHazardEnv journal branch on ph_fixtures' stub DirectRLEnv with the REAL F0 mixin, HydroV2Runtime,
condition layer, E4 bridge and the G3 AppearingBridge, and a 2 x 2 appearing pool per vessel admitted on the
REAL layouts with the active ladder (4 x 2) and PINNED in a temporary family folder (appearing_pin_fixture; the
production DIAG42 pools live in the pool bundle):
  * wiring: the appearing column follows the D7 slot (slot 15 = 14 static + 1 D7), the header names the pinned
    pool and the collision model, and the env is no longer refused (C7);
  * trigger mechanics: following the nominal plan, every witnessed dose triggers at the first control step whose
    gap is <= d_k, within one control step of motion; the cylinder is physical + visible at once (obstacle_active,
    rays) and driving on terminates the contract with a collision;
  * D0 is bit-identical to the suite absent (observations, rewards, dones, wrenches, poses, condition / D7
    stamps, every other pairing primitive);
  * a forced not-admitted key and a key beyond the 4 x 2 grid run D0 with the D20 / D24 stamps.
Run: python -B tasks/path_hazard/tests_journal/test_journal_appearing_g20b.py
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import ph_fixtures as F  # noqa: E402

F.install_stubs()
M = F.modules()
AC = importlib.import_module("tasks._shared.conditions.appearing.config")
AP = importlib.import_module("tasks._shared.conditions.appearing.placement")
APOOL = importlib.import_module("tasks._shared.conditions.appearing.pool")
AST = importlib.import_module("tasks._shared.conditions.appearing.stamp")
ADM = importlib.import_module("tasks._shared.conditions.appearing.admission")
FAMS = importlib.import_module("tasks._shared.conditions.appearing.families")
TASK = "path_hazard"
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
CONTROL_DT = 1.0 / 60.0
_spec = importlib.util.spec_from_file_location(
    "appearing_pin_fixture", F.REPO / "tasks" / "_shared" / "conditions" / "appearing" / "tests" / "appearing_pin_fixture.py")
PIN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PIN)
PINNED = {}
UNPIN = None


def setUpModule():
    global UNPIN
    for vessel in VESSELS:
        PINNED[vessel] = PIN.build(ADM, FAMS, TASK, vessel, 4, 2, seed=F.SEED)
    _, UNPIN = PIN.install(APOOL, FAMS, list(PINNED.values()))


def tearDownModule():
    if UNPIN is not None:
        UNPIN()


def make(vessel="blueboat_v2", tokens=(), num_envs=2, appearing_pool=None):
    cfg = F.journal_cfg(vessel, tokens=tokens, num_envs=num_envs, episodes=3)
    if appearing_pool is not None:
        cfg.journal_appearing_pool = appearing_pool
    return F.make_env(cfg)


def yaw_of(env):
    q = env.robot.data.root_link_pose_w[:, 3:7]
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def plan_targets(env, row, spacing_s=1.0):
    """Dense targets along the key's nominal plan up to the site, then the cylinder centre (env-local)."""
    a = env._appearing_bridge.rows[row]["assignment"]
    x, y, _ = a.cylinder
    t_site = a.placement["site"]["t_site_s"]
    context = FAMS.admission_inputs(TASK, env.cfg.vehicle, 2, F.SEED, row, int(a.key.episode))[0]
    plan = AP.nominal_plan(context, ADM.admission_suite(env.cfg.vehicle))
    scale = float(env._jph_scale)
    out, t = [], spacing_s
    while t < t_site:
        pose = plan.pose_at(t)
        out.append((pose.x, pose.y, 0.6 * scale))
        t += spacing_s
    return out + [(x, y, 0.0)]


def pursue(env, targets, cursor, thrust=0.5):
    com = env.robot.data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]
    yaw = yaw_of(env)
    actions = []
    for i in range(env.num_envs):
        goal = targets[i][min(cursor[i], len(targets[i]) - 1)]
        while cursor[i] < len(targets[i]) - 1 and \
                math.hypot(goal[0] - float(com[i, 0]), goal[1] - float(com[i, 1])) <= max(goal[2], 0.3):
            cursor[i] += 1
            goal = targets[i][cursor[i]]
        bearing = math.atan2(goal[1] - float(com[i, 1]), goal[0] - float(com[i, 0]))
        error = math.atan2(math.sin(bearing - float(yaw[i])), math.cos(bearing - float(yaw[i])))
        actions.append([thrust * max(0.2, math.cos(error)), max(-1.0, min(1.0, 1.5 * error))])
    return torch.tensor(actions, dtype=torch.float32)


def admitted_row(pool, num_envs=4, episode=0):
    for env in range(num_envs):
        if pool.is_admitted(env, episode):
            return env
    return None


def end_episode(env, rows):
    for row in rows:
        env.robot.data.root_link_pose_w[row, 0] += 1.0e3                 # out of bounds -> the episode ends


class Wiring(unittest.TestCase):
    def test_slot_header_and_no_c7(self):
        env = make(tokens=("appearing=D2",))
        base = M.plans.STATIC_CAPACITY[TASK]
        self.assertEqual(env._appearing_slot, base + M.plans.D7_RESERVED_SLOTS[TASK])
        self.assertEqual(env.obstacle_centers.shape[1], env._appearing_slot + 1)
        self.assertEqual(env._jph_d7_slots, tuple(range(base, env._appearing_slot)))
        header = env.journal_certificate_header()["appearing_hazards_header"]
        self.assertTrue(header["pool_pinned"])
        self.assertTrue(header["pool_file"].endswith("path_hazard__blueboat_v2__s42.json.xz"))
        self.assertEqual(header["grid"], {"num_envs": 4, "episodes": 2})
        self.assertEqual(header["trigger_gaps_m"], AC.trigger_gaps_m(env._appearing_bridge.t_avoid_s, "blueboat_v2"))
        self.assertEqual(len(header["k_equivalent"]), AC.DOSE_COUNT)
        self.assertIn("collision_model", header)
        absent = make()
        self.assertEqual(absent.obstacle_centers.shape[1], env._appearing_slot)     # no extra column


class TriggerAtDeclaredGap(unittest.TestCase):
    def run_dose(self, vessel, dose):
        pool = PINNED[vessel]
        row = admitted_row(pool)
        self.assertIsNotNone(row, "no admitted key among the first 4 envs at episode 0 (%s)" % vessel)
        env = make(vessel, tokens=("appearing=D%d" % dose,), num_envs=row + 1)
        env.reset()
        slot = env._appearing_slot
        bridge = env._appearing_bridge
        targets = [plan_targets(env, i) if bridge.rows[i]["active"] else [(0.0, 0.0, 0.0)]
                   for i in range(env.num_envs)]
        cursor = [0] * env.num_envs
        x, y, r = bridge.rows[row]["cylinder"]
        footprint = bridge.footprint
        d_k = AC.trigger_gap_m(dose, bridge.t_avoid_s)
        for _ in range(int(600.0 / CONTROL_DT)):
            pose = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
            self.assertFalse(bool(env.obstacle_active[row, slot]))
            rays_before = env._journal_ray_ranges()[row].clone()
            _, _, te, tr, _ = env.step(pursue(env, targets, cursor))
            if bool((te | tr)[row]):
                self.fail("episode ended before the trigger: %s"
                          % env.journal_episode_record(row)["appearing_hazards"]["payload"]["runtime"]
                          ["termination_reason"])
            if bridge.triggered(row):
                trig = bridge.rows[row]["trigger"]
                now = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
                gap = AP.gap_at(now, x, y, r, footprint)
                self.assertAlmostEqual(trig["gap_m"], gap, places=9)
                self.assertLessEqual(trig["gap_m"], d_k)
                self.assertGreater(trig["previous_check_gap_m"], d_k)          # the FIRST step with gap <= d_k
                motion = math.hypot(now.x - pose.x, now.y - pose.y) + \
                    footprint.reach_m * abs(math.remainder(now.yaw - pose.yaw, 2 * math.pi))
                self.assertGreater(trig["gap_m"], d_k - motion - 1e-9)        # within one control step
                self.assertTrue(bool(env.obstacle_active[row, slot]))         # physical + visible at once
                self.assertAlmostEqual(float(env._jph_centers[row, slot, 0]), x, places=9)
                rays_after = env._journal_ray_ranges()[row]
                self.assertTrue(bool((rays_after < rays_before - 1e-6).any()) or gap > float(env.cfg.ray_max_range_m))
                return env, row
        self.fail("the trigger gap was never reached")

    def test_every_dose_triggers_at_the_declared_gap_blueboat(self):
        for dose in AC.WITNESSED_DOSES:
            with self.subTest(dose=dose):
                self.run_dose("blueboat_v2", dose)

    def test_wamv_triggers_at_the_declared_gap(self):
        self.run_dose("wamv_vrx_v2_opt1", AC.MAX_DOSE)

    def test_contact_terminates_with_a_collision(self):
        env, row = self.run_dose("blueboat_v2", 1)
        x, y, _ = env._appearing_bridge.rows[row]["cylinder"]
        targets = [[(x, y, 0.0)] for _ in range(env.num_envs)]
        cursor = [0] * env.num_envs
        for _ in range(int(120.0 / CONTROL_DT)):
            _, _, te, tr, _ = env.step(pursue(env, targets, cursor, thrust=0.6))
            if bool((te | tr)[row]):
                break
        stamp = env.journal_episode_record(row)["appearing_hazards"]["payload"]
        self.assertEqual(stamp["runtime"]["termination_reason"], "collision")
        self.assertTrue(stamp["runtime"]["exposure_realized"])


class ZeroDoseIdentity(unittest.TestCase):
    def check(self, vessel):
        def run(tokens):
            env = make(vessel, tokens=tokens)
            obs, _ = env.reset()
            out = [obs["policy"].clone()]
            records = []
            for k in range(40):
                if k == 30:
                    end_episode(env, (1,))
                o, r, te, tr, _ = env.step(F.actions(k, env.num_envs))
                out += [o["policy"].clone(), r.clone(), te.clone(), tr.clone(),
                        env.robot.data.root_link_pose_w.clone()]
                for i in (te | tr).nonzero().flatten().tolist():
                    records.append(copy.deepcopy(env.journal_episode_record(i)))
            return env, out, records
        absent, a, ra = run(())
        present, b, rb = run(("appearing=D0",))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(len(ra), len(rb))
        self.assertGreater(len(ra), 0)
        for x, y in zip(ra, rb):
            self.assertNotIn("appearing_hazards", x)
            self.assertEqual(y["appearing_hazards"]["payload"]["assignment"]["dose"], 0)
            self.assertFalse(y["appearing_hazards"]["payload"]["runtime"]["exposure_realized"])
            for field in ("conditions", "dynamic_hazards"):
                self.assertEqual(json.dumps(x[field], sort_keys=True), json.dumps(y[field], sort_keys=True), field)
            hy = dict(y["scenario_hashes"])
            hy.pop(M.pairing.APPEARING_PRIMITIVE)
            self.assertEqual(x["scenario_hashes"], hy)
        for i in range(2):
            ca, cb = absent.journal_completed[i], present.journal_completed[i]
            if ca is not None:
                self.assertEqual(json.dumps(ca["contract"], sort_keys=True), json.dumps(cb["contract"], sort_keys=True))
        self.assertEqual(absent.dynamic_hazard_header, present.dynamic_hazard_header)

    def test_d0_is_bit_identical_to_the_suite_absent_blueboat(self):
        self.check("blueboat_v2")

    def test_d0_is_bit_identical_to_the_suite_absent_wamv(self):
        self.check("wamv_vrx_v2_opt1")


class NotAdmittedAndBeyondGrid(unittest.TestCase):
    def test_forced_not_admitted_key_runs_d0(self):
        pool = PIN.forced(APOOL, PINNED["blueboat_v2"], ((1, 0),))
        env = make(tokens=("appearing=D3",), appearing_pool=pool)
        env.reset()
        self.assertFalse(env._appearing_bridge.rows[1]["active"])
        env.step(F.actions(0, 2))
        end_episode(env, (1,))
        env.step(F.actions(1, 2))
        record = env.journal_episode_record(1)
        kind, reasons = AST.exposure_of(record["appearing_hazards"])
        self.assertEqual((kind, reasons), ("not_admitted", ("forced_test_rejection",)))
        payload = record["appearing_hazards"]["payload"]
        self.assertEqual((payload["assignment"]["dose"], payload["assignment"]["requested_dose"]), (0, 3))
        self.assertEqual((payload["runtime"]["assigned"], payload["runtime"]["itt_count"]), (False, 0))

    def test_beyond_grid_key_runs_d0_with_the_single_reason(self):
        env = make(tokens=("appearing=D2",))
        env.reset()
        for _ in range(2):                                   # episodes 0 and 1 end early (out of bounds)
            env.step(F.actions(0, 2))
            end_episode(env, (0, 1))
            env.step(F.actions(1, 2))
        self.assertFalse(env._appearing_bridge.rows[0]["active"])
        env.step(F.actions(2, 2))
        end_episode(env, (0,))
        env.step(F.actions(3, 2))
        record = env.journal_episode_record(0)
        kind, reasons = AST.exposure_of(record["appearing_hazards"])
        self.assertEqual((kind, reasons), ("not_admitted", ("beyond_offered_grid",)))
        self.assertIsNone(record["appearing_hazards"]["payload"]["assignment"]["geometry_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
