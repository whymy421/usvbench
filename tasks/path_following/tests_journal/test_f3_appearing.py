"""G3 appearing obstacle on the journal path-following task (REAL env methods, stubbed Isaac Lab, CPU).

The REAL PathFollowingEnv (F3 wiring) on f3_fixtures' fake DirectRLEnv, with the
REAL F0 mixin carrying the D7 and the appearing channels and (G20b) appearing
pools admitted on the REAL layouts with the active ladder (BlueBoat 4 x 2, WAM-V
16 x 1) PINNED in a temporary family folder (appearing_pin_fixture; the
production pools live in the pool bundle):
  * every witnessed dose triggers at the declared gap within one control step;
  * absent before the trigger, physical + visible after: the point-return ray
    block sees it, the swept-contact latch enters the contract collision
    ledger and driving on terminates the episode with a collision;
  * D0 is bit-identical to the suite absent;
  * a not-admitted key (BlueBoat leg-0 sites) and a beyond-grid key run D0
    with the D20 stamp;
  * pairing isolates the appearing treatment; both hazard suites on run and stamp.
Run: python -B test_f3_appearing.py
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

import f3_fixtures as F  # noqa: E402

J = F.jpf
AC = F.load("_shared.conditions.appearing.config")
AP = F.load("_shared.conditions.appearing.placement")
APOOL = F.load("_shared.conditions.appearing.pool")
AST = F.load("_shared.conditions.appearing.stamp")
TASK = "path_following"
F64 = torch.float64


def _pin_fixture():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "appearing_pin_fixture", F.TASKS / "_shared" / "conditions" / "appearing" / "tests" / "appearing_pin_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIN = _pin_fixture()
_ADM, _FAMS = F.load("_shared.conditions.appearing.admission"), F.load("_shared.conditions.appearing.families")
PINNED = {"blueboat_v2": PIN.build(_ADM, _FAMS, TASK, "blueboat_v2", 4, 2, seed=F.SEED),
          "wamv_vrx_v2_opt1": PIN.build(_ADM, _FAMS, TASK, "wamv_vrx_v2_opt1", 16, 1, seed=F.SEED)}
PIN.install(APOOL, _FAMS, list(PINNED.values()))


def pinned(vessel):
    return APOOL.load_pinned_pool("path_following", TASK, vessel, F.SEED, 2)[0]


def make(vessel, tokens, num_envs=2, episodes=2):
    return F.make_env(F.journal_cfg(vessel, num_envs=num_envs, episodes=episodes, tokens=tokens))


def end_in(env, steps):
    env._journal_oracle.deadline[:] = (env.episode_length_buf.to(F64) + steps) * F.plans.CONTROL_DT_S


def yaw_of(env):
    q = env.robot.data.root_link_pose_w[:, 3:7]
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def targets_for(env, row):
    row_state = env._appearing_bridge.rows[row]
    if not row_state["active"]:
        return [(float(env.waypoints[row, 0, 0]), float(env.waypoints[row, 0, 1]))]
    leg = row_state["assignment"].placement["site"]["leg_index"]
    x, y, _ = row_state["cylinder"]
    gates = [(float(env.waypoints[row, g, 0]), float(env.waypoints[row, g, 1])) for g in range(leg)]
    return gates + [(x, y)]


def pursue(env, targets, cursor, thrust=0.7, reach=1.0):
    com = env.robot.data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]
    yaw = yaw_of(env)
    out = []
    for i in range(env.num_envs):
        goal = targets[i][min(cursor[i], len(targets[i]) - 1)]
        if cursor[i] < len(targets[i]) - 1 and math.hypot(goal[0] - float(com[i, 0]), goal[1] - float(com[i, 1])) \
                <= reach:
            cursor[i] += 1
            goal = targets[i][cursor[i]]
        bearing = math.atan2(goal[1] - float(com[i, 1]), goal[0] - float(com[i, 0]))
        error = math.atan2(math.sin(bearing - float(yaw[i])), math.cos(bearing - float(yaw[i])))
        out.append([thrust * max(0.2, math.cos(error)), max(-1.0, min(1.0, 1.5 * error))])
    return torch.tensor(out, dtype=torch.float32)


def find_row(pool, num_envs, predicate):
    for env in range(num_envs):
        if predicate(env):
            return env
    return None


class TriggerAtDeclaredGap(unittest.TestCase):
    def run_dose(self, vessel, dose, num_envs, row):
        env = make(vessel, ("appearing=D%d" % dose,), num_envs=num_envs)
        env.reset()
        self.assertEqual(env._appearing_slot, 1)
        bridge = env._appearing_bridge
        self.assertTrue(bridge.rows[row]["active"])
        envelope = J.d7_envelope(vessel)
        x, y, r = bridge.rows[row]["cylinder"]
        d_k = AC.trigger_gap_m(dose, envelope.t_avoid_s)
        targets = [targets_for(env, i) for i in range(env.num_envs)]
        cursor = [0] * env.num_envs
        for _ in range(20000):
            before = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
            self.assertFalse(bool(env._appearing_physical[row, 0] or env._appearing_visible[row, 0]))
            obs, _, te, tr, _ = env.step(pursue(env, targets, cursor))
            self.assertFalse(bool((te | tr)[row]), "episode ended before the trigger")
            if bridge.triggered(row):
                trig = bridge.rows[row]["trigger"]
                now = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
                self.assertAlmostEqual(trig["gap_m"], AP.gap_at(now, x, y, r, envelope.footprint), places=9)
                self.assertLessEqual(trig["gap_m"], d_k)
                self.assertGreater(trig["previous_check_gap_m"], d_k)
                motion = math.hypot(now.x - before.x, now.y - before.y) + \
                    envelope.footprint.reach_m * abs(math.remainder(now.yaw - before.yaw, 2 * math.pi))
                self.assertGreater(trig["gap_m"], d_k - motion - 1e-9)
                self.assertTrue(bool(env._appearing_physical[row, 0] and env._appearing_visible[row, 0]))
                return env, row, targets, cursor
        self.fail("the trigger gap was never reached")

    def test_every_dose_triggers_at_the_declared_gap_blueboat(self):
        pool = pinned("blueboat_v2")
        row = find_row(pool, 4, lambda e: pool.is_admitted(e, 0))
        for dose in AC.WITNESSED_DOSES:
            with self.subTest(dose=dose):
                self.run_dose("blueboat_v2", dose, 4, row)

    def test_wamv_triggers_at_the_declared_gap(self):
        pool = pinned("wamv_vrx_v2_opt1")
        row = find_row(pool, 16, lambda e: pool.placement(e, 0)["site"]["leg_index"] == 0)
        self.assertIsNotNone(row)
        self.run_dose("wamv_vrx_v2_opt1", 3, row + 1, row)

    def test_visible_to_rays_and_contact_terminates_with_collision(self):
        pool = pinned("blueboat_v2")
        row = find_row(pool, 4, lambda e: pool.is_admitted(e, 0))
        env, row, targets, cursor = self.run_dose("blueboat_v2", 2, 4, row)
        com, quat, _, _ = env._journal_truth()
        yaw = J.yaw_from_quat(quat)
        with_cyl = env._journal_ray_ranges(com[:, :2], yaw)[row]
        hidden = J.point_return_rays(com[:, :2], yaw, env._d7_centers_local, env._d7_radii, env._d7_visible,
                                     env._journal_ray_count, env._journal_ray_range)[row]
        self.assertTrue(bool((with_cyl < hidden).any()))                 # the ray block sees the cylinder
        x, y, _ = env._appearing_bridge.rows[row]["cylinder"]
        targets[row], cursor[row] = [(x, y)], 0
        for _ in range(4000):
            _, _, te, tr, _ = env.step(pursue(env, targets, cursor))
            if bool((te | tr)[row]):
                break
        record = env.journal_path_record(row)
        self.assertEqual(record["contract"]["termination_reason"], 2)    # collision (E5 code 2)
        payload = record["appearing_hazards"]["payload"]
        self.assertTrue(payload["runtime"]["swept_contact_latched"])
        self.assertEqual(payload["runtime"]["termination_reason"], "collision")
        self.assertEqual(record["appearing_hazards_exposure"], "admitted")
        self.assertFalse(env._d7_bridge.contact(row))                    # the D7 channel saw nothing


class ZeroDoseIdentity(unittest.TestCase):
    def test_d0_is_bit_identical_to_the_suite_absent(self):
        runs = []
        for tokens in ((), ("appearing=D0",)):
            env = make("blueboat_v2", tokens, num_envs=2)
            obs, _ = env.reset()
            out, records = [obs["policy"].clone()], []
            end_in(env, 6)
            for k in range(14):
                o, r, te, tr, _ = env.step(F.seek_actions(env))
                out += [o["policy"].clone(), r.clone(), te.clone(), tr.clone(), env.robot.applied[-1][0].clone(),
                        env.robot.applied[-1][1].clone(), env.robot.data.root_link_pose_w.clone()]
                done = (te | tr).nonzero().flatten().tolist()
                records += [copy.deepcopy(env.journal_path_record(i)) for i in done]
                if done:
                    end_in(env, 6)
            runs.append((env, out, records))
        (ea, a, ra), (eb, b, rb) = runs
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertGreater(len(ra), 0)
        for x, y in zip(ra, rb):
            for field in ("conditions", "dynamic_hazards", "contract", "family"):
                self.assertEqual(json.dumps(x[field], sort_keys=True), json.dumps(y[field], sort_keys=True), field)
            hy = dict(y["scenario_hashes"])
            hy.pop(F.pairing.APPEARING_PRIMITIVE)
            self.assertEqual(x["scenario_hashes"], hy)
            self.assertEqual(y["appearing_hazards"]["payload"]["runtime"]["non_exposure_reason"], "zero_dose")
        self.assertEqual(ea.dynamic_hazard_header, eb.dynamic_hazard_header)


class NotAdmittedAndBeyondGrid(unittest.TestCase):
    def test_not_admitted_key_runs_d0_with_the_d20_stamp(self):
        # Phase 1b: the start-distance rule admits every path-following key of the pinned seed-42 pools, so the
        # D16 path is exercised with a FORCED rejection (as CONTEXT_R4 D24 (f) asks for station keeping): the
        # pinned pool with key (env 1, episode 0) replaced by a not-admitted record, the pool digest recomputed.
        pool = pinned("blueboat_v2")
        row = 1
        keys = []
        for ident in pool.offered:
            item = pool._keys[ident]
            forced = ident == (row, 0)
            keys.append({"env": ident[0], "episode": ident[1], "geometry_id": item["geometry_id"],
                         "placement": None if forced else item["placement"],
                         "reasons": [(0, "forced_rejection_for_test")] if forced else list(item["reasons"]),
                         "doses": {} if forced else dict(item["doses"])})
        forced_pool = APOOL.AppearingPool(task=TASK, vessel_id="blueboat_v2", seed=F.SEED, tier=2,
                                          suite_identity=pool.suite_identity, num_envs=pool.grid["num_envs"],
                                          episodes=2, keys=keys)
        self.assertNotEqual(forced_pool.pool_digest, pool.pool_digest)
        cfg = F.journal_cfg("blueboat_v2", num_envs=row + 1, episodes=2, tokens=("appearing=D4",))
        cfg.journal_appearing_pool = forced_pool
        env = F.make_env(cfg)
        env.reset()
        self.assertFalse(env._appearing_bridge.rows[row]["active"])
        geometry = env._journal_layouts[row].geometry_hash          # the episode-0 layout (before the reset)
        end_in(env, 3)
        for _ in range(3):
            env.step(F.seek_actions(env))
            self.assertFalse(bool(env._appearing_physical[row, 0]))
        record = env.journal_path_record(row)
        kind, reasons = AST.exposure_of(record["appearing_hazards"])
        self.assertEqual((kind, reasons), ("not_admitted", ("forced_rejection_for_test",)))
        payload = record["appearing_hazards"]["payload"]
        self.assertEqual((payload["assignment"]["dose"], payload["assignment"]["requested_dose"]), (0, 4))
        self.assertEqual((payload["runtime"]["assigned"], payload["runtime"]["itt_count"]), (False, 0))
        self.assertEqual(payload["assignment"]["geometry_id"], geometry)

    def test_beyond_grid_key_runs_d0_with_the_single_reason(self):
        env = make("blueboat_v2", ("appearing=D1",), num_envs=2)
        env.reset()
        seen = {}
        for episode in range(3):
            end_in(env, 2)
            for _ in range(2):
                env.step(F.seek_actions(env))
            for i in range(env.num_envs):
                seen[(i, episode)] = copy.deepcopy(env.journal_path_record(i))
        for i in range(env.num_envs):
            kind, reasons = AST.exposure_of(seen[(i, 2)]["appearing_hazards"])
            self.assertEqual((kind, reasons), ("not_admitted", ("beyond_offered_grid",)))
        header = env.journal_certificate_header(treatment="appearing")["appearing_hazards_header"]
        self.assertEqual(header["grid"], {"num_envs": 4, "episodes": 2})
        self.assertEqual(header["per_dose_counts"][1], list(pinned("blueboat_v2").per_dose_counts[1]))
        self.assertEqual(header["per_dose_counts"][1][0], 8)
        self.assertEqual(len(header["per_dose_counts"]), AC.DOSE_COUNT)
        self.assertTrue(header["pool_pinned"])


class PairingAndBothSuites(unittest.TestCase):
    def test_pairing_isolates_the_appearing_treatment(self):
        records = {}
        for tokens in (("appearing=D0", "dynamic=D1"), ("appearing=D2", "dynamic=D1"),
                       ("appearing=D2", "dynamic=D3")):
            env = make("wamv_vrx_v2_opt1", tokens, num_envs=1, episodes=2)
            env.reset()
            end_in(env, 2)
            for _ in range(2):
                env.step(F.seek_actions(env))
            records[tokens] = env.journal_path_record(0)
        P = F.pairing
        summary = P.validate_pair(records[("appearing=D0", "dynamic=D1")], records[("appearing=D2", "dynamic=D1")],
                                  "appearing")
        self.assertEqual(summary["changed_primitives"], [P.APPEARING_PRIMITIVE])
        summary = P.validate_pair(records[("appearing=D2", "dynamic=D1")], records[("appearing=D2", "dynamic=D3")],
                                  "dynamic")
        self.assertEqual(summary["changed_primitives"], ["d7_assignment"])
        with self.assertRaises(P.PairingError):
            P.validate_pair(records[("appearing=D0", "dynamic=D1")], records[("appearing=D2", "dynamic=D3")],
                            "appearing")

    def test_both_hazard_suites_and_all_suites_on(self):
        tokens = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "dynamic=D2", "appearing=D3")
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            env = make(vessel, tokens, num_envs=2)
            env.reset()
            end_in(env, 3)
            for _ in range(3):
                obs, rew, _, _, _ = env.step(F.seek_actions(env))
                self.assertTrue(torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all())
            header = env.journal_certificate_header(treatment="appearing")
            self.assertIn("appearing_hazards_header", header)
            self.assertEqual(header["appearing_hazards_header"]["dose"], 3)
            for i in range(2):
                record = env.journal_path_record(i)
                F.dynamic.read_stamp(record["dynamic_hazards"])
                AST.read_appearing_stamp(record["appearing_hazards"])
                self.assertEqual(sorted(record["scenario_hashes"]), list(env.journal_pairing_inventory))
                self.assertEqual(record["dynamic_hazards"]["payload"]["assignment"]["dose"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
