"""G20b appearing obstacle on journal harbour mission (REAL env methods, stubbed Isaac Lab, CPU).

The REAL HarborMissionEnv journal branch on harbour_journal_fixtures' stub DirectRLEnv with the REAL F0 mixin,
HydroV2Runtime, condition layer, harbour E4 bridge and the G3 AppearingBridge, and appearing pools admitted on
the REAL layouts with the active ladder (BlueBoat 4 x 2, WAM-V 2 x 1) PINNED in a temporary family folder
(appearing_pin_fixture; the production DIAG42 pools live in the pool bundle):
  * wiring: the appearing column follows the D7 slot (slot 81 = 80 static + 1 D7) with the D7 masks; a D7
    publish never clears it; the header names the pinned pool and the collision model;
  * trigger mechanics: following the nominal plan, every witnessed dose triggers at the first control step whose
    gap is <= d_k, within one control step of motion; the cylinder is physical + visible at once (masks, rays)
    and its swept-contact latch terminates the contract with a collision;
  * D0 is bit-identical to the suite absent (observations, rewards, dones, poses, condition / D7 stamps, the
    contract result, every other pairing primitive);
  * a forced not-admitted key and a key beyond the grid run D0 with the D20 / D24 stamps.
Run: python -B tasks/harbor_mission/test_journal_harbour_appearing.py
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
import torch  # noqa: E402

import harbour_journal_fixtures as H  # noqa: E402

AC = H.load("_shared.conditions.appearing.config")
AP = H.load("_shared.conditions.appearing.placement")
APOOL = H.load("_shared.conditions.appearing.pool")
AST = H.load("_shared.conditions.appearing.stamp")
ADM = H.load("_shared.conditions.appearing.admission")
FAMS = H.load("_shared.conditions.appearing.families")
TASK = "harbour_mission"
CONTROL_DT = 1.0 / 60.0
GRIDS = {"blueboat_v2": (4, 2), "wamv_vrx_v2_opt1": (2, 1)}
_spec = importlib.util.spec_from_file_location(
    "appearing_pin_fixture", H.REPO / "tasks" / "_shared" / "conditions" / "appearing" / "tests" / "appearing_pin_fixture.py")
PIN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PIN)
PINNED = {}
UNPIN = None


def setUpModule():
    global UNPIN
    for vessel, (envs, episodes) in GRIDS.items():
        PINNED[vessel] = PIN.build(ADM, FAMS, TASK, vessel, envs, episodes, seed=H.SEED)
    _, UNPIN = PIN.install(APOOL, FAMS, list(PINNED.values()))


def tearDownModule():
    if UNPIN is not None:
        UNPIN()


def make(vessel="blueboat_v2", tokens=(), num_envs=2, **extra):
    return H.make_env(vessel, tokens, num_envs=num_envs, **extra)


def yaw_of(env):
    q = env.robot.data.root_link_pose_w[:, 3:7]
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def plan_targets(env, row, spacing_s=1.0):
    a = env._appearing_bridge.rows[row]["assignment"]
    x, y, _ = a.cylinder
    t_site = a.placement["site"]["t_site_s"]
    context = FAMS.admission_inputs(TASK, env.cfg.vehicle, 2, H.SEED, row, int(a.key.episode))[0]
    plan = AP.nominal_plan(context, ADM.admission_suite(env.cfg.vehicle))
    scale = float(env._jlayouts[row].params["length_scale"])
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


def admitted_row(pool, num_envs, episode=0):
    for env in range(num_envs):
        if pool.is_admitted(env, episode):
            return env
    return None


def end_episode(env, rows):
    for row in rows:
        env.robot.data.root_link_pose_w[row, 0] += 1.0e3


def fixed_actions(k, n):
    base = torch.tensor([[0.7, 0.1], [0.5, -0.2], [0.8, 0.0], [0.6, 0.3]], dtype=torch.float32)
    return base[:n] * (1.0 if k % 5 else 0.6)


class Wiring(unittest.TestCase):
    def test_slot_masks_header(self):
        env = make(tokens=("appearing=D2",))
        storage = H.plans.STATIC_CAPACITY[TASK] + H.plans.D7_RESERVED_SLOTS[TASK]
        self.assertEqual(env._appearing_slot, storage)
        self.assertEqual(tuple(env.obstacle_centers.shape[1:]), (storage + 1, 2))
        self.assertEqual(env._d7_physical.shape[1], storage + 1)
        header = env.journal_certificate_header()["appearing_hazards_header"]
        self.assertTrue(header["pool_pinned"])
        self.assertTrue(header["pool_file"].endswith("harbour_mission__blueboat_v2__s42.json.xz"))
        self.assertEqual(header["grid"], {"num_envs": 4, "episodes": 2})
        self.assertEqual(header["hazard_group"], "g3_appearing_harbour_mission")
        self.assertIn("collision_model", header)
        absent = make()
        self.assertEqual(absent.obstacle_centers.shape[1], storage)
        self.assertEqual(absent._jwidth, storage)


class TriggerAtDeclaredGap(unittest.TestCase):
    def run_dose(self, vessel, dose):
        pool = PINNED[vessel]
        row = admitted_row(pool, GRIDS[vessel][0])
        self.assertIsNotNone(row, "no admitted key at episode 0 (%s)" % vessel)
        env = make(vessel, ("appearing=D%d" % dose,), num_envs=max(2, row + 1))
        env.reset()
        slot = env._appearing_slot
        bridge = env._appearing_bridge
        targets = [plan_targets(env, i) if bridge.rows[i]["active"] else [(0.0, 0.0, 0.0)]
                   for i in range(env.num_envs)]
        cursor = [0] * env.num_envs
        x, y, r = bridge.rows[row]["cylinder"]
        footprint = bridge.footprint
        d_k = AC.trigger_gap_m(dose, bridge.t_avoid_s)
        for _ in range(int(900.0 / CONTROL_DT)):
            pose = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
            self.assertFalse(bool(env._d7_visible[row, slot] | env._d7_physical[row, slot]))
            rays_before = env._journal_true_rays()[row].clone()
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
                self.assertGreater(trig["previous_check_gap_m"], d_k)
                motion = math.hypot(now.x - pose.x, now.y - pose.y) + \
                    footprint.reach_m * abs(math.remainder(now.yaw - pose.yaw, 2 * math.pi))
                self.assertGreater(trig["gap_m"], d_k - motion - 1e-9)
                self.assertTrue(bool(env._d7_visible[row, slot] & env._d7_physical[row, slot]))
                self.assertFalse(bool(env.obstacle_active[row, slot]))        # D7-like channel, not static
                origin = env.scene.env_origins[row, :2]
                self.assertAlmostEqual(float(env.obstacle_centers[row, slot, 0] - origin[0]), x, places=3)
                rays_after = env._journal_true_rays()[row]
                self.assertTrue(bool((rays_after < rays_before - 1e-6).any()) or gap > float(env.cfg.ray_max_range_m))
                # a D7 publish of every row clears only the D7 slots, never the appearing column
                env._journal_publish_d7({i: () for i in range(env.num_envs)})
                self.assertTrue(bool(env._d7_visible[row, slot]))
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
        self.assertTrue(stamp["runtime"]["swept_contact_latched"])


class ZeroDoseIdentity(unittest.TestCase):
    def check(self, vessel):
        def run(tokens):
            env = make(vessel, tokens)
            obs, _ = env.reset()
            out = [obs["policy"].clone()]
            records, results = [], []
            for k in range(40):
                if k == 30:
                    end_episode(env, (1,))
                o, r, te, tr, _ = env.step(fixed_actions(k, env.num_envs))
                out += [o["policy"].clone(), r.clone(), te.clone(), tr.clone(),
                        env.robot.data.root_link_pose_w.clone()]
                for i in (te | tr).nonzero().flatten().tolist():
                    records.append(copy.deepcopy(env.journal_episode_record(i)))
                    results.append(json.dumps(env.episode_contract_result[i], sort_keys=True, default=str))
            return env, out, records, results
        absent, a, ra, qa = run(())
        present, b, rb, qb = run(("appearing=D0",))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(qa, qb)
        self.assertEqual(len(ra), len(rb))
        self.assertGreater(len(ra), 0)
        for x, y in zip(ra, rb):
            self.assertNotIn("appearing_hazards", x)
            self.assertEqual(y["appearing_hazards"]["payload"]["assignment"]["dose"], 0)
            for field in ("conditions", "dynamic_hazards"):
                self.assertEqual(json.dumps(x[field], sort_keys=True), json.dumps(y[field], sort_keys=True), field)
            hy = dict(y["scenario_hashes"])
            hy.pop(H.pairing.APPEARING_PRIMITIVE)
            self.assertEqual(x["scenario_hashes"], hy)
        self.assertEqual(absent.dynamic_hazard_header, present.dynamic_hazard_header)

    def test_d0_is_bit_identical_to_the_suite_absent_blueboat(self):
        self.check("blueboat_v2")

    def test_d0_is_bit_identical_to_the_suite_absent_wamv(self):
        self.check("wamv_vrx_v2_opt1")


class NotAdmittedAndBeyondGrid(unittest.TestCase):
    def test_forced_not_admitted_key_runs_d0(self):
        pool = PIN.forced(APOOL, PINNED["blueboat_v2"], ((1, 0),))
        env = make(tokens=("appearing=D3",), journal_appearing_pool=pool)
        env.reset()
        self.assertFalse(env._appearing_bridge.rows[1]["active"])
        env.step(fixed_actions(0, 2))
        end_episode(env, (1,))
        env.step(fixed_actions(1, 2))
        record = env.journal_episode_record(1)
        kind, reasons = AST.exposure_of(record["appearing_hazards"])
        self.assertEqual((kind, reasons), ("not_admitted", ("forced_test_rejection",)))
        payload = record["appearing_hazards"]["payload"]
        self.assertEqual((payload["runtime"]["assigned"], payload["runtime"]["itt_count"]), (False, 0))

    def test_beyond_grid_key_runs_d0_with_the_single_reason(self):
        env = make(tokens=("appearing=D2",))
        env.reset()
        for _ in range(2):                                   # episodes 0 and 1 end early (out of bounds)
            env.step(fixed_actions(0, 2))
            end_episode(env, (0, 1))
            env.step(fixed_actions(1, 2))
        self.assertFalse(env._appearing_bridge.rows[0]["active"])
        env.step(fixed_actions(2, 2))
        end_episode(env, (0,))
        env.step(fixed_actions(3, 2))
        record = env.journal_episode_record(0)
        kind, reasons = AST.exposure_of(record["appearing_hazards"])
        self.assertEqual((kind, reasons), ("not_admitted", ("beyond_offered_grid",)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
