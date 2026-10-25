"""F4 round T: episodes beyond the offered grid (CONTEXT_R4 D24), CPU only.

Run: python -B tasks/station_keeping/tests_journal/test_station_journal_d24.py

The REAL station_keeping_env.py journal branch (stubbed isaaclab, toy
integrator, REAL runtime / condition layer / E4 bridge / committed seed-42
pools: offered grid 64 envs x 3 episodes) is driven like an evaluator whose
fast env autoresets past its quota: row 1 ends episodes 0, 1 and 2 early (out
of bounds) and autoresets into task episode 3 = E, one past the offered grid.
That episode must run to its layout deadline at every dynamic level D0..D4
without an exception and stamp the family's not-admitted form (D16 / D20)
with the single reason "beyond_offered_grid" (D24 (a)); its layout is the E5
layout of that key; F0 pairing.d7_primitive_hash accepts the stamp and gives
the same value at every level. Keys outside the env range, with another seed
or another hazard group stay preflight errors (D24 (b)). The D7 header
publishes grid = {"num_envs": 64, "episodes": 3} (D24 (c)) and keeps the pool's
[offered, admitted, not_admitted] per-dose counts unchanged (D22).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import station_journal_fixtures as F

RESTORE = None
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
D20_REASON_SOURCES = (("not_admitted", "reasons"), ("not_admitted_reasons",),
                      ("assignment", "not_admitted_reasons"), ("assignment", "rejection_reasons"),
                      ("runtime", "rejection_reasons"), ("runtime", "reasons"), ("rejection_reasons",),
                      ("reasons",))


def setUpModule():
    global RESTORE, js, pairing, dyn
    RESTORE = F.install_isaaclab_stubs()
    js = F.ns("station_keeping.journal_station")
    pairing = F.ns("_shared.conditions.pairing")
    dyn = F.ns("_shared.conditions.dynamic")


def tearDownModule():
    if RESTORE is not None:
        RESTORE()


def d20_reasons(payload):
    """CONTEXT_R4 D20 (c) reading: first non-empty source, items normalised, joint echo dropped."""
    for path in D20_REASON_SOURCES:
        value = payload
        for part in path:
            value = value.get(part) if isinstance(value, dict) else None
        if not value:
            continue
        out = []
        for item in value:
            if isinstance(item, str):
                names = [item]
            elif isinstance(item, dict):
                names = [item["reason"]]
            else:
                names = [v for v in item if isinstance(v, str)]
            out.extend(n for n in names if n != "joint_ladder_member_rejected" and n not in out)
        return out
    return []


def committed_pool(vessel):
    cfg = F.journal_cfg(vessel)
    return js.StationD7Pool.load(cfg.journal_d7_pool, cfg.journal_d7_pool_sha256)


class PoolGridTests(unittest.TestCase):
    def test_committed_pools_offered_grid_and_counts(self):
        for vessel in VESSELS:
            with self.subTest(vessel=vessel):
                pool = committed_pool(vessel)
                self.assertEqual(pool.offered_grid, {"num_envs": 64, "episodes": 3})
                self.assertEqual(pool.grid["episodes_admitted"], 3)             # 2 scored + 1 overflow (round 4)
                na = (1,) if vessel == "blueboat_v2" else ()                  # G23e: not applicable by design
                self.assertEqual(pool.per_dose_counts,                          # D22 triples
                                 tuple((0, 0, 0) if d in na else (192, 192, 0) for d in range(5)))
                self.assertEqual(pool.e4_per_dose_counts,                       # stored E4 columns unchanged
                                 tuple((0, 0) if d in na else (192, 192) for d in range(5)))
                self.assertEqual(pool.keys_not_admitted, 0)
                for env, episode in ((0, 0), (63, 2), (17, 1)):
                    self.assertEqual(pool.key_status(js.episode_key(42, env, episode)), js.ADMITTED)
                for env, episode in ((0, 3), (63, 3), (5, 4), (12, 10 ** 6)):
                    self.assertEqual(pool.key_status(js.episode_key(42, env, episode)), js.BEYOND_OFFERED_GRID)

    def test_beyond_grid_assignment_is_the_not_admitted_record(self):
        pool = committed_pool("blueboat_v2")
        key = js.episode_key(42, 5, 3)
        records = [pool.assignment(key, d) for d in range(5)]           # no bind_context needed
        for d, record in enumerate(records):
            self.assertIsInstance(record, js.NotAdmittedAssignment)
            self.assertEqual((record.kind, record.dose, record.requested_dose), (js.BEYOND_OFFERED_GRID, 0, d))
            self.assertEqual(record.reasons, ((0, "beyond_offered_grid", 0),))
            summary = record.summary()
            self.assertEqual((summary["dose"], summary["intruders"], summary["task"]), (0, [], "station_keeping"))
            self.assertEqual(summary["key"], {"eval_seed": 42, "env": 5, "episode": 3,
                                              "hazard_group": "d7_station_keeping"})
            self.assertIsNone(summary["geometry_id"])
        self.assertEqual(len({r.candidate_digest for r in records}), 1)          # dose independent
        other = pool.assignment(js.episode_key(42, 5, 4), 2)
        self.assertNotEqual(other.candidate_digest, records[0].candidate_digest)  # key dependent
        with self.assertRaises(KeyError):                                         # no layout record of it
            pool.bind_context(key, js.station_task_context(js.build_station_layout(2, "blueboat_v2", 42, 5, 3),
                                                           "blueboat_v2"))
        with self.assertRaises(KeyError):
            pool.assignment(key, 5)

    def test_keys_outside_the_grid_stay_preflight_errors(self):
        pool = committed_pool("blueboat_v2")
        dock = dyn.EpisodeKey(42, 0, 0, "d7_docking")
        for key in (js.episode_key(42, 64, 0), js.episode_key(42, 64, 5), js.episode_key(43, 0, 0),
                    js.episode_key(43, 0, 3), js.episode_key(42, 1000, 3),
                    dyn.EpisodeKey(42, 0, 3, "d7_docking"), dock):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "preflight error"):
                    pool.key_status(key)
                with self.assertRaises(KeyError):                 # the mixin's B02 preflight conversion
                    pool.assignment(key, 0)

    def test_manifest_integrity_of_the_not_admitted_derivation(self):
        cfg = F.journal_cfg("blueboat_v2")
        document = json.loads(Path(cfg.journal_d7_pool).read_text(encoding="ascii"))

        def forged(mutate):
            doc = copy.deepcopy(document)
            mutate(doc)
            doc["pool_digest"] = js.pool_manifest_digest(doc)
            return doc

        def partial_ladder(doc):
            doc["accepted"] = [r for r in doc["accepted"] if not (r[1] == 3 and r[2] == 1 and r[4] == 2)]
        with self.assertRaisesRegex(ValueError, "whole D0..D4 ladder"):
            js.StationD7Pool(forged(partial_ladder))

        def dropped_without_reason(doc):
            doc["accepted"] = [r for r in doc["accepted"] if not (r[1] == 3 and r[2] == 1)]
            doc["keys_admitted"] -= 1
            doc["per_dose_counts"] = [[192, 191]] * 5
        with self.assertRaisesRegex(ValueError, "no E4 rejection reason"):
            js.StationD7Pool(forged(dropped_without_reason))

        def not_a_grid(doc):
            doc["offered"] = [k for k in doc["offered"] if not (k["env"] == 7 and k["episode"] == 0)]
            doc["accepted"] = [r for r in doc["accepted"] if not (r[1] == 7 and r[2] == 0)]
            doc["keys_offered"] -= 1
            doc["keys_admitted"] -= 1
            del doc["layouts"]["7/0"]
        with self.assertRaisesRegex(ValueError, "full grid"):
            js.StationD7Pool(forged(not_a_grid))

        def wrong_grid(doc):
            doc["grid"]["episodes_admitted"] = 4
        with self.assertRaisesRegex(ValueError, "grid"):
            js.StationD7Pool(forged(wrong_grid))


class BeyondGridEnvTests(unittest.TestCase):
    ROW = 1

    def run_level(self, dose, vessel="blueboat_v2"):
        env = F.make_env(F.journal_cfg(vessel, num_envs=2, tokens=("dynamic=D%d" % dose,)))
        header0 = copy.deepcopy(env.journal_certificate_header()["dynamic_hazards_header"])
        env.reset()
        row = self.ROW
        early = []
        for _ in range(3):                                   # an early finisher: episodes 0, 1, 2 end early
            for k in range(2):
                env.step(F.hold_controller(env))
            env.robot.data.root_link_pose_w[row, 0] += 1.0e3
            _, _, te, tr, _ = env.step(F.hold_controller(env))
            self.assertTrue(bool(te[row] | tr[row]))
            early.append(dyn.read_stamp(env.journal_episode_record(row)["dynamic_hazards"]))
        for stamp in early:                                  # admitted pool keys: ordinary E4 stamps
            self.assertNotIn("exposure", stamp["payload"])
            self.assertEqual(stamp["payload"]["assignment"]["dose"], dose)
        self.assertEqual(int(env._jscenario.episode_indices()[row]), 3)      # = E, one past the grid
        self.assertEqual(env._d7_bridge.exposure(row), js.NOT_ADMITTED)
        record = env._d7_bridge.current_record(row)
        self.assertEqual((record.kind, record.requested_dose), (js.BEYOND_OFFERED_GRID, dose))
        layout = js.build_station_layout(2, vessel, 42, row, 3)
        self.assertEqual(env._jlayouts[row].geometry_hash, layout.geometry_hash)   # the key's E5 layout
        pose = env.robot.data.root_link_pose_w
        self.assertAlmostEqual(float(pose[row, 0] - env.scene.env_origins[row, 0]), float(layout.start[0]),
                               places=5)
        self.assertAlmostEqual(float(pose[row, 1] - env.scene.env_origins[row, 1]), float(layout.start[1]),
                               places=5)
        published = False
        steps = 0
        while True:                                          # the full beyond-grid episode, to its deadline
            _, _, te, tr, _ = env.step(F.hold_controller(env))
            steps += 1
            if not bool(te[row] | tr[row]):
                published |= bool(env._d7_physical[row, 0] | env._d7_visible[row, 0])
            if bool(te[row] | tr[row]):
                break
        self.assertFalse(published)                          # no D7 object for the not-admitted row
        self.assertEqual(steps, int(layout.deadline_steps))
        result = env.journal_episode_result(row)
        self.assertEqual(result["termination_reason_name"], "timeout")
        header1 = env.journal_certificate_header()["dynamic_hazards_header"]
        return env.journal_episode_record(row), header0, header1, layout

    def test_beyond_grid_episode_runs_at_every_level(self):
        hashes, digests = set(), set()
        with self.assertRaisesRegex(ValueError, "not applicable by design"):   # G23e: BlueBoat SK has no D1
            self.run_level(1)
        for dose in (0, 2, 3, 4):
            with self.subTest(dose=dose):
                record, header0, header1, layout = self.run_level(dose)
                envelope = dyn.read_stamp(record["dynamic_hazards"])
                payload = envelope["payload"]
                self.assertEqual(payload["exposure"], "not_admitted")
                self.assertEqual(payload["runtime"]["exposure"], "not_admitted")
                self.assertEqual(payload["assignment"]["exposure"], "not_admitted")
                self.assertEqual(payload["not_admitted_reasons"], [[0, "beyond_offered_grid", 0]])
                self.assertEqual(payload["not_admitted"]["reasons"], [[0, "beyond_offered_grid", 0]])
                self.assertEqual(payload["not_admitted"]["kind"], "beyond_offered_grid")
                self.assertEqual(d20_reasons(payload), ["beyond_offered_grid"])
                assignment = payload["assignment"]
                self.assertEqual((assignment["dose"], assignment["intruders"], assignment["task"]),
                                 (0, [], "station_keeping"))
                self.assertEqual(assignment["key"], {"eval_seed": 42, "env": 1, "episode": 3,
                                                     "hazard_group": "d7_station_keeping"})
                self.assertEqual(payload["runtime"]["suite_dose"], dose)
                self.assertEqual(payload["runtime"]["termination_reason"], "timeout")
                self.assertFalse(payload["runtime"]["assigned"])
                bridge = payload["runtime"]["bridge"]
                self.assertEqual(bridge["swept_intervals"], bridge["last_tick"])   # every interval stepped once
                self.assertEqual(js.record_exposure(record["dynamic_hazards"]), "not_admitted")
                d7 = pairing.d7_primitive_hash(record["dynamic_hazards"])
                self.assertEqual(record["scenario_hashes"][pairing.D7_PRIMITIVE], d7)
                self.assertEqual(record["scenario_hashes"]["layout"], layout.geometry_hash)
                self.assertEqual(record["conditions"]["runtime"]["episode_index"], 3)
                hashes.add(d7)
                digests.add(assignment["candidate_digest"])
                # D24 (c) + D22: grid published, per-dose counts are the pool's and do not move
                self.assertEqual(header1["grid"], {"num_envs": 64, "episodes": 3})
                self.assertEqual(header1["per_dose_counts"],                # G23e: D1 not applicable by design
                                 [[0, 0, 0] if d == 1 else [192, 192, 0] for d in range(5)])
                self.assertEqual(header1["per_dose_counts_columns"], ["offered", "admitted", "not_admitted"])
                self.assertEqual(header0, header1)
        self.assertEqual(len(hashes), 1)                    # same primitive at every level
        self.assertEqual(len(digests), 1)

    def test_wamv_beyond_grid_reset_and_header(self):
        env = F.make_env(F.journal_cfg("wamv_vrx_v2_opt1", num_envs=2, tokens=("dynamic=D3",)))
        header = env.journal_certificate_header()["dynamic_hazards_header"]
        self.assertEqual(header["grid"], {"num_envs": 64, "episodes": 3})
        self.assertEqual(header["per_dose_counts"], [[192, 192, 0]] * 5)
        env.reset()
        for _ in range(3):
            env.step(F.hold_controller(env))
            env.robot.data.root_link_pose_w[0, 1] += 1.0e3
            env.step(F.hold_controller(env))
        self.assertEqual(int(env._jscenario.episode_indices()[0]), 3)
        self.assertEqual(env._d7_bridge.current_record(0).kind, js.BEYOND_OFFERED_GRID)
        for _ in range(3):
            env.step(F.hold_controller(env))
        env.robot.data.root_link_pose_w[0, 1] += 1.0e3
        env.step(F.hold_controller(env))
        payload = dyn.read_stamp(env.journal_episode_record(0)["dynamic_hazards"])["payload"]
        self.assertEqual((payload["exposure"], payload["not_admitted_reasons"]),
                         ("not_admitted", [[0, "beyond_offered_grid", 0]]))
        self.assertEqual(payload["runtime"]["termination_reason"], "out_of_bounds")
        self.assertEqual(int(env._jscenario.episode_indices()[0]), 4)       # and on: still beyond the grid
        self.assertEqual(env._d7_bridge.current_record(0).key.episode, 4)

    def test_preflight_errors_at_setup_and_reset(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, seed=43))
        with self.assertRaisesRegex(ValueError, "covers 64 envs"):
            F.make_env(F.journal_cfg("blueboat_v2", num_envs=65))
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2))
        env.reset()
        env.cfg.seed = 43                                    # a key of another seed reaches the reset
        with self.assertRaisesRegex(ValueError, "preflight error"):
            env._reset_idx(torch.tensor([0]))
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2))
        env.reset()
        env.cfg.seed = 43                                    # the F0 mixin's own B02 conversion (KeyError)
        nxt = env._conditions.scenario.episode_indices()[torch.tensor([0])] + 1
        with self.assertRaisesRegex(ValueError, "preflight error"):
            env.journal_reset(torch.tensor([0]), torch.zeros(1), nxt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
