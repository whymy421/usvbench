"""G3 appearing obstacle on the HazardNav journal scatter task (REAL env methods, stubbed Isaac Lab, CPU).

Every test constructs the REAL HazardNavEnv from the REAL E5 scatter template
cfg on journal_test_support's stub DirectRLEnv, with the REAL F0 mixin (now
carrying the D7 and the appearing channels), HydroV2Runtime, schema-3
condition layer, E4 bridge and the G3 AppearingBridge, and (G20b) a 2 x 2
appearing pool per vessel admitted on the REAL layouts with the active ladder
and PINNED in a temporary family folder (appearing_pin_fixture; the production
pools live in the pool bundle):
  * every witnessed dose triggers at the declared gap within one control step;
  * the cylinder is absent before the trigger (no slot, rays blind to it) and
    physical plus visible after (obstacle_active, rays, reward clearance);
    driving on terminates the contract with a collision;
  * D0 is bit-identical to the suite absent (observations, rewards, dones,
    wrenches, poses, condition / D7 stamps, every other pairing primitive);
  * a not-admitted key and a beyond-grid key run D0 with the D20 stamp;
  * pairing isolates the appearing treatment and the appearing primitive is
    dose-independent for every other suite;
  * both hazard suites (and all suites) on at once run and stamp;
  * the validator seam (pairing recompute, header checks) accepts the records.
Run: python -B tasks/hazard_nav/test_journal_appearing.py
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import journal_test_support as S  # noqa: E402

RESTORE = S.install_isaaclab_stubs()
FAM = S.load_family()
J, D = FAM.journal, FAM.dynamic
import importlib  # noqa: E402

AC = importlib.import_module(S.ROOT + "._shared.conditions.appearing.config")
AP = importlib.import_module(S.ROOT + "._shared.conditions.appearing.placement")
APOOL = importlib.import_module(S.ROOT + "._shared.conditions.appearing.pool")
AST = importlib.import_module(S.ROOT + "._shared.conditions.appearing.stamp")
TASK = "scatter"
ALL_ON = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "perception.bundle=D2",
          "dynamic=D2")


def _pin_fixture():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "appearing_pin_fixture", S.REPO / "tasks" / "_shared" / "conditions" / "appearing" / "tests"
        / "appearing_pin_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIN = _pin_fixture()
_ADM = importlib.import_module(S.ROOT + "._shared.conditions.appearing.admission")
_FAMS = importlib.import_module(S.ROOT + "._shared.conditions.appearing.families")
# G20b: the production pools live in the pool bundle; a 2 x 2 grid per vessel is admitted on the REAL layouts
# with the active ladder and pinned in a temporary family folder (the env's pinned-pool path runs unchanged)
PINNED = {v: PIN.build(_ADM, _FAMS, TASK, v, 2, 2, seed=S.SEED) for v in ("blueboat_v2", "wamv_vrx_v2_opt1")}
PIN.install(APOOL, _FAMS, list(PINNED.values()))


def pinned(vessel):
    return APOOL.load_pinned_pool("hazard_nav", TASK, vessel, S.SEED, 2)[0]


def env_for(vessel="blueboat_v2", tokens=(), num_envs=2, episodes=(0, 1, 2)):
    pool = S.admitted_pool(FAM, TASK, vessel, num_envs=num_envs, episodes=episodes)
    cfg = S.journal_cfg(FAM, TASK, vessel, tokens=tokens, num_envs=num_envs, pool=pool)
    return S.make_env(FAM, cfg)


def force_timeout(env, steps):
    env._journal_twin.deadline[:] = steps * FAM.plans.CONTROL_DT_S


def yaw_of(env):
    q = env.robot.data.root_link_pose_w[:, 3:7]
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def route_targets(env, row):
    """Nominal route circles before the site, then the cylinder centre (env-local)."""
    assignment = env._appearing_bridge.rows[row]["assignment"]
    x, y, _ = assignment.cylinder
    t_site = assignment.placement["site"]["t_site_s"]
    context = J.d7_context(env._journal_scenarios[row])
    plan = AP.nominal_plan(context, AC.AppearingSuiteConfig(enabled=True, dose=1, vessel_id=env.cfg.vehicle))
    targets = [(c.x, c.y, c.radius) for c, t in zip(context.route, plan.target_times) if t < t_site]
    return targets + [(x, y, 0.0)]


def pursue(env, targets, cursor, thrust=0.6):
    com = env.robot.data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]
    yaw = yaw_of(env)
    actions = []
    for i in range(env.num_envs):
        goal = targets[i][min(cursor[i], len(targets[i]) - 1)]
        if cursor[i] < len(targets[i]) - 1 and math.hypot(goal[0] - float(com[i, 0]), goal[1] - float(com[i, 1])) \
                <= max(goal[2], 0.5):
            cursor[i] += 1
            goal = targets[i][cursor[i]]
        bearing = math.atan2(goal[1] - float(com[i, 1]), goal[0] - float(com[i, 0]))
        error = math.atan2(math.sin(bearing - float(yaw[i])), math.cos(bearing - float(yaw[i])))
        actions.append([thrust * max(0.2, math.cos(error)), max(-1.0, min(1.0, 1.5 * error))])
    return torch.tensor(actions, dtype=torch.float32)


def admitted_row(pool, num_envs=2, episode=0):
    for env in range(num_envs):
        if pool.is_admitted(env, episode):
            return env
    raise AssertionError("no admitted row among the first %d envs at episode %d" % (num_envs, episode))


def not_admitted_row(pool, num_envs=64, episode=0):
    for env in range(num_envs):
        if not pool.is_admitted(env, episode):
            return env
    return None


class TriggerAtDeclaredGap(unittest.TestCase):
    def run_dose(self, vessel, dose, num_envs=2):
        pool = pinned(vessel)
        row = admitted_row(pool, num_envs)
        env = env_for(vessel, tokens=("appearing=D%d" % dose,), num_envs=num_envs)
        env.reset()
        slot = env._appearing_slot
        self.assertEqual(slot, env.obstacle_centers.shape[1] - 1)
        self.assertEqual(slot, FAM.plans.STATIC_CAPACITY[TASK] + 1)      # after the D7 slot
        targets = [route_targets(env, i) if env._appearing_bridge.rows[i]["active"] else [(0.0, 0.0, 0.0)]
                   for i in range(env.num_envs)]
        cursor = [0] * env.num_envs
        bridge = env._appearing_bridge
        envelope = J.vessel_envelope(vessel)
        x, y, r = bridge.rows[row]["cylinder"]
        d_k = AC.trigger_gap_m(dose, envelope.t_avoid_s)
        prev_pose, prev_obs = None, None
        for _ in range(6000):
            pose = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
            before = bool(env.obstacle_active[row, slot])
            self.assertFalse(before)
            obs, _, te, tr, _ = env.step(pursue(env, targets, cursor))
            self.assertFalse(bool((te | tr)[row]), "episode ended before the trigger")
            if bridge.triggered(row):
                trig = bridge.rows[row]["trigger"]
                now = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
                gap = AP.gap_at(now, x, y, r, envelope.footprint)
                self.assertAlmostEqual(trig["gap_m"], gap, places=9)
                self.assertLessEqual(trig["gap_m"], d_k)
                self.assertGreater(trig["previous_check_gap_m"], d_k)          # the FIRST step with gap <= d_k
                motion = math.hypot(now.x - pose.x, now.y - pose.y) + \
                    envelope.footprint.reach_m * abs(math.remainder(now.yaw - pose.yaw, 2 * math.pi))
                self.assertGreater(trig["gap_m"], d_k - motion - 1e-9)        # within one control step
                self.assertTrue(bool(env.obstacle_active[row, slot]))         # physical + visible at once
                self.assertEqual(float(env.obstacle_radii[row, slot]), float(torch.tensor(r, dtype=torch.float32)))
                return env, row, prev_obs, obs
            prev_obs = obs
        self.fail("the trigger gap was never reached")

    def test_every_dose_triggers_at_the_declared_gap_blueboat(self):
        for dose in AC.WITNESSED_DOSES:
            with self.subTest(dose=dose):
                self.run_dose("blueboat_v2", dose)

    def test_wamv_triggers_at_the_declared_gap(self):
        self.run_dose("wamv_vrx_v2_opt1", AC.MAX_DOSE)

    def test_absent_before_physical_and_visible_after_and_contact_terminates(self):
        env, row, before_obs, after_obs = self.run_dose("blueboat_v2", 2)
        slot = env._appearing_slot
        # visible: the 36-ray block of the observation now sees the cylinder
        mask_without = env.obstacle_active.clone()
        mask_without[:, slot] = False
        rays_with = env._journal_rays(env.obstacle_active | env._d7_visible)[row]
        rays_without = env._journal_rays(mask_without | env._d7_visible)[row]
        self.assertTrue(bool((rays_with < rays_without).any()))
        # physical: the reward clearance and the contract clearance include it
        clearance = float(env._journal_clearance()[row])
        assignment = env._appearing_bridge.rows[row]
        pose = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
        x, y, r = assignment["cylinder"]
        self.assertLessEqual(clearance, math.hypot(pose.x - x, pose.y - y) - r + 1e-6)
        # keep driving into it: the contract terminates with a collision
        cursor = [99] * env.num_envs
        targets = [[(x, y, 0.0)] for _ in range(env.num_envs)]
        for _ in range(3000):
            _, _, te, tr, _ = env.step(pursue(env, targets, cursor))
            if bool((te | tr)[row]):
                break
        record = env.journal_completed_records[row]
        self.assertEqual(record["termination_name"], "collision")
        stamp = record["appearing_hazards"]["payload"]
        self.assertTrue(stamp["runtime"]["exposure_realized"])
        self.assertEqual(stamp["runtime"]["termination_reason"], "collision")
        self.assertEqual(record["appearing_exposure"], "admitted")


class ZeroDoseIdentity(unittest.TestCase):
    def test_d0_is_bit_identical_to_the_suite_absent(self):
        def run(tokens):
            env = env_for(tokens=tokens)
            obs, _ = env.reset()
            out = [obs["policy"].clone()]
            records = []
            force_timeout(env, 5)
            for k in range(12):
                o, r, te, tr, _ = env.step(S.zero_action(env, (0.6, 0.25 * (-1) ** k)))
                out += [o["policy"].clone(), r.clone(), te.clone(), tr.clone(),
                        env.robot.data.root_link_pose_w.clone(), env.robot.applied[0].clone(),
                        env.robot.applied[1].clone()]
                for i in (te | tr).nonzero().flatten().tolist():
                    records.append(copy.deepcopy(env.journal_completed_records[i]))
                if bool((te | tr).any()):
                    force_timeout(env, 5)
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
            for field in ("conditions", "dynamic_hazards", "contract", "scenario", "termination_code"):
                # JSON text equality (NaN-aware: the contract reports NaN for an unsuccessful episode)
                self.assertEqual(json.dumps(x[field], sort_keys=True), json.dumps(y[field], sort_keys=True), field)
            hy = dict(y["scenario_hashes"])
            hy.pop(FAM.pairing.APPEARING_PRIMITIVE)
            self.assertEqual(x["scenario_hashes"], hy)       # D0 = "suite absent" for every other primitive
        self.assertEqual(absent.dynamic_hazard_header, present.dynamic_hazard_header)
        self.assertEqual(absent.journal_certificate_header()["conditions"],
                         present.journal_certificate_header()["conditions"])
        self.assertNotIn("appearing_hazards_header", absent.journal_certificate_header())


class NotAdmittedAndBeyondGrid(unittest.TestCase):
    def test_not_admitted_key_runs_d0_with_the_d20_stamp(self):
        # G20b: a forced D16 record (key (1, 0) of the 2 x 2 grid replaced by a not-admitted record, the pool
        # digest recomputed), handed to the env as cfg.journal_appearing_pool (not pinned, so not a certificate)
        pool = PIN.forced(APOOL, pinned("blueboat_v2"), ((1, 0),))
        row = not_admitted_row(pool, num_envs=2)
        self.assertEqual(row, 1)
        d7 = S.admitted_pool(FAM, TASK, "blueboat_v2", num_envs=2, episodes=(0, 1, 2))
        cfg = S.journal_cfg(FAM, TASK, "blueboat_v2", tokens=("appearing=D3",), num_envs=2, pool=d7)
        cfg.journal_appearing_pool = pool
        env = S.make_env(FAM, cfg)
        env.reset()
        self.assertFalse(env._appearing_bridge.rows[row]["active"])
        force_timeout(env, 3)
        for _ in range(3):
            env.step(S.zero_action(env, (0.5, 0.0)))
            self.assertFalse(bool(env.obstacle_active[row, env._appearing_slot]))
        record = env.journal_completed_records[row]
        kind, reasons = AST.exposure_of(record["appearing_hazards"])
        self.assertEqual(kind, "not_admitted")
        self.assertEqual(record["appearing_exposure"], "not_admitted")
        payload = record["appearing_hazards"]["payload"]
        self.assertEqual(payload["exposure"], "not_admitted")
        self.assertEqual((payload["assignment"]["dose"], payload["assignment"]["requested_dose"]), (0, 3))
        self.assertIsNone(payload["assignment"]["cylinder"])
        self.assertEqual((payload["runtime"]["assigned"], payload["runtime"]["itt_count"]), (False, 0))
        self.assertTrue(set(reasons) <= set(pool.reason_counts()))
        self.assertEqual(payload["not_admitted"]["reasons"], list(reasons))

    def test_beyond_grid_key_runs_d0_with_the_single_reason(self):
        env = env_for(tokens=("appearing=D2",), episodes=(0, 1, 2))
        env.reset()
        seen = {}
        for _ in range(3):                       # episodes 0, 1 (in the grid), then 2 (beyond the 64 x 2 grid)
            force_timeout(env, 2)
            for _ in range(2):
                _, _, te, tr, _ = env.step(S.zero_action(env, (0.4, 0.1)))
            for i in range(env.num_envs):
                record = env.journal_completed_records[i]
                seen[(i, record["scenario_episode"])] = record
        for i in range(env.num_envs):
            record = seen[(i, 2)]
            kind, reasons = AST.exposure_of(record["appearing_hazards"])
            self.assertEqual((kind, reasons), ("not_admitted", ("beyond_offered_grid",)))
            payload = record["appearing_hazards"]["payload"]
            self.assertEqual((payload["assignment"]["dose"], payload["runtime"]["itt_count"]), (0, 0))
            self.assertIsNone(payload["assignment"]["geometry_id"])
        header = env.journal_certificate_header()["appearing_hazards_header"]
        self.assertEqual(header["grid"], {"num_envs": 2, "episodes": 2})
        self.assertTrue(header["pool_pinned"])
        self.assertEqual(header["per_dose_counts_fields"], ["offered", "admitted", "not_admitted"])
        self.assertEqual(len(header["per_dose_counts"]), AC.DOSE_COUNT)


class Pairing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = {}
        for tokens in (("appearing=D0",), ("appearing=D3",), ("appearing=D3", "waves=D2"),
                       ("appearing=D3", "dynamic=D2"), ("appearing=D1", "waves=D2")):
            env = env_for(tokens=tokens)
            env.reset()
            force_timeout(env, 2)
            for _ in range(2):
                env.step(S.zero_action(env, (0.4, 0.0)))
            cls.records[tokens] = [copy.deepcopy(env.journal_completed_records[i]) for i in range(env.num_envs)]

    def test_appearing_treatment_is_isolated(self):
        P = FAM.pairing
        pool = pinned("blueboat_v2")
        for row in range(2):
            a, b = self.records[("appearing=D0",)][row], self.records[("appearing=D3",)][row]
            if pool.is_admitted(row, 0):
                summary = P.validate_pair(a, b, "appearing")
                self.assertEqual(summary["changed_primitives"], [P.APPEARING_PRIMITIVE])
                with self.assertRaises(P.PairingError):
                    P.validate_pair(a, b, "waves")
            else:      # a not-admitted key hashes identically at every appearing dose (D20 form): unexposed
                self.assertEqual(a["scenario_hashes"], b["scenario_hashes"])
                with self.assertRaises(P.PairingError):
                    P.validate_pair(a, b, "appearing")

    def test_appearing_primitive_is_dose_independent_for_every_other_suite(self):
        P = FAM.pairing
        for row in range(2):
            base = self.records[("appearing=D3",)][row]
            for other, treatment in ((("appearing=D3", "waves=D2"), "waves"),
                                     (("appearing=D3", "dynamic=D2"), "dynamic")):
                rec = self.records[other][row]
                self.assertEqual(rec["scenario_hashes"][P.APPEARING_PRIMITIVE],
                                 base["scenario_hashes"][P.APPEARING_PRIMITIVE])
                summary = P.validate_pair(base, rec, treatment)
                self.assertNotIn(P.APPEARING_PRIMITIVE, summary["changed_primitives"])
            if pinned("blueboat_v2").is_admitted(row, 0):
                mixed = self.records[("appearing=D1", "waves=D2")][row]
                with self.assertRaises(P.PairingError):       # an appearing difference never pairs as waves
                    P.validate_pair(self.records[("appearing=D3", "waves=D2")][row], mixed, "waves")

    def test_tampered_appearing_stamp_is_rejected(self):
        P = FAM.pairing
        a = copy.deepcopy(self.records[("appearing=D3",)][0])
        a["appearing_hazards"]["payload"]["assignment"]["dose"] = 2
        with self.assertRaises(P.PairingError):
            P.validate_pair(a, self.records[("appearing=D0",)][0], "appearing")


class BothHazardSuites(unittest.TestCase):
    def check(self, vessel, tokens):
        env = env_for(vessel, tokens=tokens)
        env.reset()
        self.assertEqual(env.obstacle_centers.shape[1], FAM.plans.STATIC_CAPACITY[TASK] + 2)
        for _ in range(3):
            obs, rew, te, tr, _ = env.step(S.zero_action(env, (0.7, -0.2)))
            self.assertTrue(torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all())
        force_timeout(env, 4)
        env.step(S.zero_action(env))
        header = env.journal_certificate_header(treatment="appearing")
        self.assertEqual(header["pairing"]["changed_primitives"], [FAM.pairing.APPEARING_PRIMITIVE])
        self.assertIn("d7_assignment", header["pairing"]["unchanged_primitives"])
        self.assertEqual(header["appearing_hazards_header"]["dose"], 2)
        self.assertEqual(header["appearing_hazards_header"]["slot"], FAM.plans.STATIC_CAPACITY[TASK] + 1)
        self.assertIn("pool_digest", header["dynamic_hazards_header"])
        for i in range(env.num_envs):
            rec = env.journal_completed_records[i]
            self.assertEqual(rec["termination_name"], "timeout")
            D.read_stamp(rec["dynamic_hazards"])
            AST.read_appearing_stamp(rec["appearing_hazards"])
            self.assertIn(FAM.pairing.APPEARING_PRIMITIVE, rec["scenario_hashes"])
            self.assertIn("d7_assignment", rec["scenario_hashes"])
            if "dynamic=D2" in tokens:
                self.assertEqual(rec["dynamic_hazards"]["payload"]["assignment"]["dose"], 2)
        return env

    def test_both_hazard_suites_on_blueboat(self):
        self.check("blueboat_v2", ("dynamic=D2", "appearing=D2"))

    def test_all_suites_on_wamv(self):
        self.check("wamv_vrx_v2_opt1", ALL_ON + ("appearing=D2",))

    def test_wired_task_without_a_pinned_pool_fails_closed(self):
        # every HazardNav task is wired; reach has no pin in the temporary family folder of this test
        pool = S.admitted_pool(FAM, "reach", "blueboat_v2", num_envs=2, episodes=(0, 1))
        cfg = S.journal_cfg(FAM, "reach", "blueboat_v2", tokens=("appearing=D1",), num_envs=2, pool=pool)
        with self.assertRaisesRegex(ValueError, "not pinned"):
            S.make_env(FAM, cfg)


class ValidatorSeam(unittest.TestCase):
    def test_validator_recomputes_the_appearing_primitive_and_checks_the_header(self):
        sys.path.insert(0, str(S.REPO / "scripts"))
        import validate_cert_v3 as V
        import appearing_exposure as AE
        env = env_for(tokens=("appearing=D2", "dynamic=D1"))
        env.reset()
        force_timeout(env, 2)
        for _ in range(2):
            env.step(S.zero_action(env, (0.4, 0.0)))
        header = env.journal_certificate_header(treatment="appearing")
        records = []
        for i in range(env.num_envs):
            rec = env.journal_completed_records[i]
            records.append({"env": i, "ep": 0, "conditions": rec["conditions"],
                            "dynamic_hazards": rec["dynamic_hazards"], "scenario_hashes": rec["scenario_hashes"],
                            "appearing_hazards": rec["appearing_hazards"]})
        document = {"header": header, "records": records}
        self.assertEqual(V.journal_pairing_errors(document), [])
        self.assertEqual(AE.record_errors(document), [])
        self.assertEqual(AE.header_errors(header["appearing_hazards_header"], records), [])
        AE.grid_check(header["appearing_hazards_header"], 2, 2)
        with self.assertRaises(ValueError):
            AE.grid_check(header["appearing_hazards_header"], 2, 3)
        with self.assertRaises(ValueError):
            AE.grid_check(header["appearing_hazards_header"], 3, 2)
        bad = copy.deepcopy(document)
        bad["records"][0]["scenario_hashes"][FAM.pairing.APPEARING_PRIMITIVE] = "0" * 64
        self.assertTrue(V.journal_pairing_errors(bad))
        bad = copy.deepcopy(document)
        bad["records"][0]["appearing_hazards"]["payload"]["runtime"]["trigger_checks"] += 1
        self.assertTrue(AE.record_errors(bad))
        unpinned = dict(header["appearing_hazards_header"], pool_pinned=False)
        self.assertTrue(AE.header_errors(unpinned, records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
