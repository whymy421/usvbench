"""G20 / G20b appearing obstacle on a rolled-out HazardNav task (forced crossing; REAL env methods, stubbed
Isaac Lab, CPU).

The REAL HazardNavEnv on journal_test_support's stub DirectRLEnv with the REAL F0 mixin, HydroV2Runtime,
condition layer, E4 bridge and G3 AppearingBridge, and a 2 x 2 appearing pool per vessel admitted on the
REAL layouts with the active ladder and PINNED in a temporary family folder (appearing_pin_fixture; the
production DIAG42 pools live in the pool bundle):
  * trigger mechanics: following the nominal plan, every witnessed dose triggers at the first control step
    whose gap is <= d_k, within one control step of motion, and the cylinder is physical + visible at once;
  * D0 is bit-identical to the suite absent (observations, rewards, dones, wrenches, poses, condition /
    D7 stamps, every other pairing primitive);
  * a key beyond the 2 x 2 grid runs D0 with the single reason beyond_offered_grid;
  * the bridge replays the nominal plan of every admitted key of both pools at the control rate and
    triggers within [d_k - (one control step of nominal motion), d_k].
Run: python -B tasks/hazard_nav/test_journal_appearing_g20.py
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
ART = importlib.import_module(S.ROOT + "._shared.conditions.appearing.runtime")
AST = importlib.import_module(S.ROOT + "._shared.conditions.appearing.stamp")
TASK = "forced_crossing"
CONTROL_DT = 1.0 / 60.0


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


def plan_targets(env, row, spacing_s=1.0):
    """Dense targets along the key's nominal plan up to the site, then the cylinder centre (env-local)."""
    assignment = env._appearing_bridge.rows[row]["assignment"]
    x, y, _ = assignment.cylinder
    t_site = assignment.placement["site"]["t_site_s"]
    context = J.d7_context(env._journal_scenarios[row])
    plan = AP.nominal_plan(context, AC.AppearingSuiteConfig(enabled=True, dose=1, vessel_id=env.cfg.vehicle))
    out, t = [], spacing_s
    while t < t_site:
        pose = plan.pose_at(t)
        out.append((pose.x, pose.y, 0.6 * env._journal_record.geometry.length_scale))
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


def admitted_row(pool, num_envs=2, episode=0):
    for env in range(num_envs):
        if pool.is_admitted(env, episode):
            return env
    raise AssertionError("no admitted row among the first %d envs at episode %d" % (num_envs, episode))


class TriggerAtDeclaredGap(unittest.TestCase):
    def run_dose(self, vessel, dose, num_envs=2):
        pool = pinned(vessel)
        row = admitted_row(pool, num_envs)
        env = env_for(vessel, tokens=("appearing=D%d" % dose,), num_envs=num_envs)
        env.reset()
        slot = env._appearing_slot
        self.assertEqual(slot, env.obstacle_centers.shape[1] - 1)
        self.assertEqual(slot, FAM.plans.STATIC_CAPACITY[TASK] + 1)      # after the D7 slot
        targets = [plan_targets(env, i) if env._appearing_bridge.rows[i]["active"] else [(0.0, 0.0, 0.0)]
                   for i in range(env.num_envs)]
        cursor = [0] * env.num_envs
        bridge = env._appearing_bridge
        envelope = J.vessel_envelope(vessel)
        x, y, r = bridge.rows[row]["cylinder"]
        d_k = AC.trigger_gap_m(dose, envelope.t_avoid_s)
        for _ in range(int(400.0 / CONTROL_DT)):
            pose = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
            self.assertFalse(bool(env.obstacle_active[row, slot]))
            _, _, te, tr, _ = env.step(pursue(env, targets, cursor))
            if bool((te | tr)[row]):
                self.fail("episode ended before the trigger: %s"
                          % env.journal_completed_records[row]["termination_name"])
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
                return env, row
        self.fail("the trigger gap was never reached")

    def test_every_dose_triggers_at_the_declared_gap_blueboat(self):
        for dose in AC.WITNESSED_DOSES:
            with self.subTest(dose=dose):
                self.run_dose("blueboat_v2", dose)

    def test_wamv_triggers_at_the_declared_gap(self):
        self.run_dose("wamv_vrx_v2_opt1", AC.MAX_DOSE)


class ZeroDoseIdentity(unittest.TestCase):
    def check(self, vessel):
        def run(tokens):
            env = env_for(vessel, tokens=tokens)
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
                self.assertEqual(json.dumps(x[field], sort_keys=True), json.dumps(y[field], sort_keys=True), field)
            hy = dict(y["scenario_hashes"])
            hy.pop(FAM.pairing.APPEARING_PRIMITIVE)
            self.assertEqual(x["scenario_hashes"], hy)
        self.assertEqual(absent.dynamic_hazard_header, present.dynamic_hazard_header)
        self.assertEqual(absent.journal_certificate_header()["conditions"],
                         present.journal_certificate_header()["conditions"])
        header = present.journal_certificate_header()["appearing_hazards_header"]
        self.assertTrue(header["pool_pinned"])
        self.assertEqual(header["grid"], {"num_envs": 2, "episodes": 2})

    def test_d0_is_bit_identical_to_the_suite_absent_blueboat(self):
        self.check("blueboat_v2")

    def test_d0_is_bit_identical_to_the_suite_absent_wamv(self):
        self.check("wamv_vrx_v2_opt1")


class BeyondGrid(unittest.TestCase):
    def test_beyond_grid_key_runs_d0_with_the_single_reason(self):
        env = env_for(tokens=("appearing=D2",), episodes=(0, 1, 2))
        env.reset()
        seen = {}
        for _ in range(3):
            force_timeout(env, 2)
            for _ in range(2):
                env.step(S.zero_action(env, (0.4, 0.1)))
            for i in range(env.num_envs):
                record = env.journal_completed_records[i]
                seen[(i, record["scenario_episode"])] = record
        for i in range(env.num_envs):
            kind, reasons = AST.exposure_of(seen[(i, 2)]["appearing_hazards"])
            self.assertEqual((kind, reasons), ("not_admitted", ("beyond_offered_grid",)))


class NominalReplayTrigger(unittest.TestCase):
    """Bridge-level trigger mechanics on every admitted key: the nominal plan sampled at the control rate."""

    def test_every_admitted_key_triggers_within_one_control_step(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            pool = pinned(vessel)
            envelope = J.vessel_envelope(vessel)
            checked = 0
            for env_i, episode in pool.offered:
                if not pool.is_admitted(env_i, episode):
                    continue
                context = J.d7_context(J.build_scenario(TASK, vessel, 2, S.SEED, env_i, episode))
                plan = AP.nominal_plan(context, AC.AppearingSuiteConfig(enabled=True, dose=1, vessel_id=vessel))
                for dose in AC.WITNESSED_DOSES:
                    key = D.EpisodeKey(S.SEED, env_i, episode, AC.hazard_group(TASK))
                    a = pool.assignment(key, dose)
                    bridge = ART.AppearingBridge(TASK, AC.appearing_slot(80, 1), 1.0 / 120.0,
                                                 envelope.top_speed_mps, envelope.footprint, envelope.t_avoid_s)
                    a = a.__class__(**{**a.__dict__, "key": D.EpisodeKey(S.SEED, 0, episode, a.key.hazard_group)})
                    bridge.reset_rows([0], {0: a})
                    tick = 0
                    bridge.step({0: 0}, {0: plan.pose_at(0.0)})
                    while not bridge.triggered(0) and tick * (1.0 / 120.0) < plan.end_s:
                        for _ in range(2):
                            tick += 1
                            bridge.step({0: tick}, {0: plan.pose_at(min(plan.end_s, tick / 120.0))})
                        bridge.step({0: tick}, {0: plan.pose_at(min(plan.end_s, tick / 120.0))},
                                    check_trigger=True)
                    self.assertTrue(bridge.triggered(0), (vessel, env_i, episode, dose))
                    trig = bridge.rows[0]["trigger"]
                    d_k = a.trigger_gap_m
                    self.assertLessEqual(trig["gap_m"], d_k)
                    self.assertGreater(trig["gap_m"], d_k - envelope.reference_speed_mps * CONTROL_DT - 1e-6)
                    self.assertAlmostEqual(trig["time_s"], a.candidate["nominal_trigger_s"], delta=CONTROL_DT + 1e-9)
                    checked += 1
            self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
