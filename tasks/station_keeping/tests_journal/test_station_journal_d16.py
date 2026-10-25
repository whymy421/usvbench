"""F4 round T: the D16 not-admitted path with a FORCED, REAL E4 rejection, CPU only.

Run: python -B tasks/station_keeping/tests_journal/test_station_journal_d16.py

The seed-42 station-keeping pools admit every offered key, so the not-admitted
path is exercised on a small REAL E4 admission (journal_station.
admit_station_grid, one joint admit_grid pass, BlueBoat, tier 2, keys (0, 0)
and (1, 0)) whose ``context_fn`` hook hands E4 the real context of key (1, 0)
with a 5 s deadline, so E4 itself rejects that key at every dose (the
placement reason "conflict_after_deadline_or_plan_end"). The env keeps the
key's E5 layout (real deadline) and the stored rejection records drive the
not-admitted record (no re-admission, the manifest digest is unchanged).

Checks (CONTEXT_R4 D16, D20, D22): the not-admitted key runs a full episode at
D0..D4 with exposure "not_admitted" (payload, payload.runtime and
payload.assignment), the E4 reasons, a dose-0 assignment with key, task, dose
0 and a dose-independent candidate digest; the header per-dose counts are
[offered, admitted, not_admitted] = [2, 1, 1]; F0 pairing.validate_pair pairs
the key with itself under waves, wind, current, actuator and perception, and
raises "no treatment difference" for the dynamic treatment (unexposed), while
the admitted key of the same runs is exposed (its dynamic pair validates).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import station_journal_fixtures as F

RESTORE = None
VESSEL = "blueboat_v2"
FORCED = (1, 0)                       # (env, episode) E4 is made to reject
FORCED_DEADLINE_S = 5.0
# G23e: with the approach design the 5 s deadline ends the nominal before the meeting point
E4_REASON = "sk_nominal_never_reaches_meeting_distance"
DOSES = (0, 2, 3, 4)                    # G23e: BlueBoat station keeping D1 is not applicable by design
SUITES = {"waves": "waves=D2", "wind": "wind=D2", "current": "current=D2", "actuator": "actuator=D2",
          "perception": "perception=D2"}


def setUpModule():
    global RESTORE, js, pairing, dyn
    RESTORE = F.install_isaaclab_stubs()
    js = F.ns("station_keeping.journal_station")
    pairing = F.ns("_shared.conditions.pairing")
    dyn = F.ns("_shared.conditions.dynamic")


def tearDownModule():
    if RESTORE is not None:
        RESTORE()


def force(layout, vessel, key):
    """context_fn: the REAL context of the key; only the forced key gets a 5 s E4 deadline."""
    context = js.station_task_context(layout, vessel)
    if (key.env, key.episode) == FORCED:
        context = dataclasses.replace(context, deadline_s=FORCED_DEADLINE_S)
    return context


class ForcedRejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw, layouts, suite = js.admit_station_grid(VESSEL, 2, F.SEED, [(0, 0), FORCED], context_fn=force)
        cls.raw, cls.layouts, cls.suite = raw, layouts, suite
        cls.levels = {}

    def pool(self):
        """A fresh in-memory pool (manifest + witnesses through a JSON round trip) per env."""
        return js.pool_from_admission(self.raw, self.layouts, vessel_key=VESSEL, seed=F.SEED, tier=2,
                                      suite=self.suite, grid={"num_envs": 2, "episodes": 1})

    def env(self, tokens):
        return F.make_env(F.journal_cfg(VESSEL, num_envs=2, tokens=tokens, pool=self.pool()))

    # ------------------------------------------------------------- helpers --
    def level(self, dose):
        """Full episodes of both rows at dynamic dose ``dose``: {row: first completed record}."""
        if dose in self.levels:
            return self.levels[dose]
        env = self.env(("dynamic=D%d" % dose,))
        env.reset()
        self.assertEqual(env._d7_bridge.exposure(1), js.NOT_ADMITTED)
        self.assertEqual(env._d7_bridge.exposure(0), js.ADMITTED)
        geometry = env._jlayouts[1].geometry_hash              # the forced key's layout (episode 0)
        records, results, steps, published = {}, {}, 0, False
        while len(records) < 2:
            _, _, te, tr, _ = env.step(F.hold_controller(env))
            steps += 1
            done = te | tr
            if 1 not in records:
                published |= bool(env._d7_physical[1, 0] | env._d7_visible[1, 0])
            for row in (0, 1):
                if bool(done[row]) and row not in records:
                    records[row] = env.journal_episode_record(row)
                    results[row] = (env.journal_episode_result(row), steps)
        header = env.journal_certificate_header()["dynamic_hazards_header"]
        self.levels[dose] = (records, results, published, header, geometry)
        return self.levels[dose]

    def suite_record(self, token):
        """Same scenario under dynamic=D2 plus one other suite; the episode is ended out of bounds."""
        env = self.env(("dynamic=D2", token))
        env.reset()
        for _ in range(3):
            env.step(F.hold_controller(env))
        env.robot.data.root_link_pose_w[1, 0] += 1.0e3
        _, _, te, tr, _ = env.step(F.hold_controller(env))
        self.assertTrue(bool(te[1] | tr[1]))
        return env.journal_episode_record(1)

    # --------------------------------------------------------------- tests --
    def test_real_e4_rejection_and_the_pool_view(self):
        raw = self.raw
        self.assertEqual((raw.keys_offered, raw.keys_admitted), (2, 1))
        self.assertEqual(sorted({(r.key.env, r.key.episode) for r in raw.rejections}), [FORCED])
        pool = self.pool()
        key = js.episode_key(F.SEED, *FORCED)
        self.assertEqual(pool.key_status(key), js.NOT_ADMITTED)
        self.assertEqual(pool.key_status(js.episode_key(F.SEED, 0, 0)), js.ADMITTED)
        self.assertEqual(pool.key_status(js.episode_key(F.SEED, 0, 1)), js.BEYOND_OFFERED_GRID)
        self.assertEqual(pool.offered_grid, {"num_envs": 2, "episodes": 1})
        reasons = [[d, E4_REASON, 0] for d in range(5)]
        self.assertEqual(pool.not_admitted_reasons(*FORCED), reasons)
        # G23e: D1 not applicable by design: published (0, 0, 0); E4 offered D1 only to the key rejected before
        # the design could be evaluated (its deadline ends the nominal before the meeting point)
        self.assertEqual(pool.per_dose_counts, tuple((0, 0, 0) if d == 1 else (2, 1, 1) for d in range(5)))
        self.assertEqual(pool.e4_per_dose_counts, tuple((1, 0) if d == 1 else (2, 1) for d in range(5)))
        self.assertEqual(pool.not_applicable_doses, (1,))
        fields = pool.header_fields()
        self.assertEqual(fields["not_admitted"], [[1, 0, E4_REASON]])
        self.assertEqual(fields["grid"], {"num_envs": 2, "episodes": 1})
        with self.assertRaisesRegex(ValueError, "bind the env's real layout context"):
            pool.assignment(key, 2)
        layout = js.build_station_layout(2, VESSEL, F.SEED, *FORCED)
        other = js.build_station_layout(2, VESSEL, F.SEED, 1, 1)
        with self.assertRaisesRegex(ValueError, "differs from the admitted layout"):
            pool.bind_context(key, js.station_task_context(other, VESSEL))
        pool.bind_context(key, js.station_task_context(layout, VESSEL))
        records = [pool.assignment(key, d) for d in range(5)]
        for d, record in enumerate(records):
            self.assertIsInstance(record, js.NotAdmittedAssignment)
            self.assertEqual((record.kind, record.dose, record.requested_dose), (js.E4_REJECTED, 0, d))
            self.assertEqual(record.reason_list(), reasons)
            self.assertEqual(record.geometry_id, layout.geometry_hash)
        self.assertEqual(len({r.candidate_digest for r in records}), 1)

    def test_not_admitted_key_runs_full_episodes_d0_to_d4(self):
        layout = js.build_station_layout(2, VESSEL, F.SEED, *FORCED)
        hashes, digests, admitted_hashes = set(), set(), set()
        with self.assertRaisesRegex(ValueError, "not applicable by design"):      # G23e
            self.env(("dynamic=D1",))
        for dose in DOSES:
            with self.subTest(dose=dose):
                records, results, published, header, geometry = self.level(dose)
                self.assertEqual(geometry, layout.geometry_hash)          # E5 layout kept (no redraw)
                self.assertFalse(published)                               # no D7 object for the row
                result, steps = results[1]
                self.assertEqual(result["termination_reason_name"], "timeout")   # a FULL episode
                self.assertEqual(steps, int(layout.deadline_steps))
                record = records[1]
                payload = dyn.read_stamp(record["dynamic_hazards"])["payload"]
                self.assertEqual(payload["exposure"], "not_admitted")
                self.assertEqual(payload["runtime"]["exposure"], "not_admitted")
                self.assertEqual(payload["assignment"]["exposure"], "not_admitted")
                self.assertEqual(payload["not_admitted_reasons"], [[d, E4_REASON, 0] for d in range(5)])
                self.assertEqual(payload["not_admitted"]["kind"], "e4_rejected")
                assignment = payload["assignment"]
                self.assertEqual((assignment["dose"], assignment["intruders"], assignment["task"]),
                                 (0, [], "station_keeping"))
                self.assertEqual(assignment["key"], {"eval_seed": 42, "env": 1, "episode": 0,
                                                     "hazard_group": "d7_station_keeping"})
                self.assertEqual(assignment["geometry_id"], layout.geometry_hash)
                self.assertEqual(payload["runtime"]["suite_dose"], dose)
                self.assertEqual(payload["runtime"]["termination_reason"], "timeout")
                bridge = payload["runtime"]["bridge"]
                self.assertEqual(bridge["swept_intervals"], bridge["last_tick"])
                d7 = pairing.d7_primitive_hash(record["dynamic_hazards"])
                self.assertEqual(record["scenario_hashes"][pairing.D7_PRIMITIVE], d7)
                hashes.add(d7)
                digests.add(assignment["candidate_digest"])
                self.assertEqual(header["per_dose_counts"],                       # D22 (G23e: D1 excluded)
                                 [[0, 0, 0] if d == 1 else [2, 1, 1] for d in range(5)])
                self.assertEqual(header["not_applicable_doses"], [1])
                self.assertEqual(header["keys_not_admitted"], 1)
                self.assertEqual(header["grid"], {"num_envs": 2, "episodes": 1})  # D24 (c)
                admitted = dyn.read_stamp(records[0]["dynamic_hazards"])["payload"]
                self.assertNotIn("exposure", admitted)                        # E4 stamp unchanged
                self.assertEqual(admitted["assignment"]["dose"], dose)
                admitted_hashes.add(records[0]["scenario_hashes"][pairing.D7_PRIMITIVE])
        self.assertEqual(len(hashes), 1)                  # same primitive at every level
        self.assertEqual(len(digests), 1)
        self.assertEqual(len(admitted_hashes), len(DOSES))   # the admitted key IS exposed per dose

    def test_pairing_other_suites_and_unexposed_for_dynamic(self):
        base = self.level(2)[0][1]
        for treatment, token in SUITES.items():
            with self.subTest(treatment=treatment):
                record = self.suite_record(token)
                payload = dyn.read_stamp(record["dynamic_hazards"])["payload"]
                self.assertEqual(payload["exposure"], "not_admitted")
                summary = pairing.validate_pair(base, record, treatment)
                self.assertTrue(set(summary["changed_primitives"]) <= set(pairing.treatment_primitives(treatment)))
                self.assertIn(pairing.D7_PRIMITIVE, summary["unchanged_primitives"])
                self.assertIn("layout", summary["unchanged_primitives"])
        for low, high in ((0, 3), (0, 4), (2, 4)):
            with self.subTest(dynamic=(low, high)):
                with self.assertRaisesRegex(pairing.PairingError, "no treatment difference"):
                    pairing.validate_pair(self.level(low)[0][1], self.level(high)[0][1], "dynamic")
        # contrast: the admitted key of the same runs pairs cleanly for the dynamic treatment
        summary = pairing.validate_pair(self.level(0)[0][0], self.level(3)[0][0], "dynamic")
        self.assertEqual(summary["changed_primitives"], [pairing.D7_PRIMITIVE])


if __name__ == "__main__":
    unittest.main(verbosity=2)
