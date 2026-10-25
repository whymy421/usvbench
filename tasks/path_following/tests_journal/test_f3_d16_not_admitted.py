"""F3 round 4b: CONTEXT_R4 D16 not-admitted keys on the REAL PathFollowingEnv (stubbed Isaac Lab, CPU only).

Real E4 rejections, no synthetic pool: at evaluation seed 51 E4 rejects the
BlueBoat key (env 0, episode 0) and the WAM-V key (env 0, episode 0), both
('nominal_observer_detects_before_reveal') (G7: until the G7 unforced-window rule
the BlueBoat probe was seed 20, a 'no_unforced_d1_window_before_site_end' key E4 now admits)
(found by an offline search of the first env's key; the fixture admits the
pool of the test grid on the real layouts).

* a not-admitted key runs a FULL episode at every dynamic level D0..D4: no D7
  object ever, identical wrench sequence at every level (it runs D0), stamp
  exposure 'not_admitted' with E4's reasons, requested dose recorded, header
  per-dose counts [offered, admitted, not_admitted];
* it pairs (F0 pairing.validate_pair) with the same key at D0 of every other
  suite (waves, wind, current, actuator, perception) at every dynamic level,
  and is unexposed for the dynamic treatment (validate_pair refuses a
  'dynamic' pair: no treatment difference), unlike an admitted key;
* its layout is the E5 layout of the key (no redraw, no attempt stream);
* an aborted not-admitted row stamps 'aborted' without a certificate record.
Run: python -B test_f3_d16_not_admitted.py
"""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import f3_fixtures as F  # noqa: E402

J = F.jpf
NA = J.EXPOSURE_NOT_ADMITTED
BLUEBOAT_NA_SEED = 51          # BlueBoat key (51, env 0, episode 0) is not admitted by E4 (G7: was seed 20)
WAMV_NA_SEED = 51              # WAM-V key (51, env 0, episode 0) is not admitted by E4
OTHER_SUITES = {"waves": "waves=D2", "wind": "wind=D2", "current": "current=D2", "actuator": "actuator=D2",
                "perception": "perception=D2"}


def short_record(vessel, seed, tokens, steps=3):
    """Record of env 0's first episode, ended out of bounds at control step ``steps`` (code 4)."""
    cfg = F.journal_cfg(vessel, num_envs=1, episodes=2, tokens=tokens, seed=seed)
    env = F.make_env(cfg)
    env.reset()
    exposure = env._journal_d7_exposure[0]
    for k in range(steps + 1):
        if k == steps:
            env.robot.data.root_link_pose_w[0, 0] += 5000.0
        env.step(F.seek_actions(env))
    assert int(env.termination_reason[0]) == 4
    return env, env.journal_path_record(0), exposure


class NotAdmittedEpisodeTests(unittest.TestCase):
    def test_full_episode_at_every_dynamic_level(self):
        vessel, seed = "blueboat_v2", BLUEBOAT_NA_SEED
        runs = []
        for dose in range(5):
            cfg = F.journal_cfg(vessel, num_envs=1, episodes=2, tokens=("dynamic=D%d" % dose,), seed=seed)
            env = F.make_env(cfg)
            env.reset()
            pool = env._d7_pool
            self.assertFalse(pool.is_admitted(0, 0))
            self.assertEqual(env._journal_d7_exposure[0], NA)
            self.assertEqual(env._d7_bridge.exposure(0), NA)
            layout = env._journal_layouts[0]
            self.assertEqual(layout.geometry_hash, J.journal_layout(vessel, 2, seed, 0, 0).geometry_hash)
            self.assertEqual(layout.geometry_hash, pool.geometry(0, 0))
            forces, code = [], 0
            for _ in range(int(layout.deadline_steps) + 5):
                env.step(F.seek_actions(env))
                forces.append(torch.cat(env.robot.applied[-1], -1)[0].clone())
                code = int(env.termination_reason[0])
                if code:
                    break
                self.assertFalse(bool(env._d7_physical.any()) or bool(env._d7_visible.any()))   # no D7 object
                self.assertEqual(env._journal_d7_samples.get(0), ())
                self.assertFalse(env._d7_bridge.contact(0))
            self.assertIn(code, (1, 3), "the episode must end naturally (success or timeout)")
            record = env.journal_path_record(0)
            stamp = record["dynamic_hazards"]
            F.dynamic.read_stamp(stamp)                                  # E4 envelope + digest
            payload = stamp["payload"]
            self.assertEqual(payload["exposure"], NA)
            self.assertEqual(J.stamp_exposure(stamp), NA)
            self.assertEqual(record["dynamic_hazards_exposure"], NA)
            assignment, runtime = payload["assignment"], payload["runtime"]
            self.assertEqual((assignment["dose"], assignment["requested_dose"]), (0, dose))
            self.assertEqual(assignment["intruders"], [])
            self.assertEqual(assignment["rejection_reasons"], ["nominal_observer_detects_before_reveal"])
            self.assertEqual(assignment["key"], {"eval_seed": seed, "env": 0, "episode": 0,
                                                 "hazard_group": J.HAZARD_GROUP})
            self.assertEqual(assignment["geometry_id"], layout.geometry_hash)
            self.assertEqual(assignment["pool_digest"], pool.pool_digest)
            self.assertFalse(runtime["assigned"])
            self.assertEqual(runtime["non_exposure_reason"], NA)
            self.assertEqual(runtime["termination_reason"], {1: "success", 3: "timeout"}[code])
            self.assertEqual(record["contract"]["termination_reason"], code)
            self.assertEqual(record["scenario_hashes"]["d7_assignment"], F.pairing.d7_primitive_hash(stamp))
            header = env.journal_certificate_header()["dynamic_hazards_header"]
            self.assertEqual(header["per_dose_counts_fields"], ["offered", "admitted", "not_admitted"])
            self.assertEqual(header["per_dose_counts"], [list(x) for x in pool.per_dose_counts])
            self.assertTrue(all(x[2] >= 1 and x[0] == x[1] + x[2] for x in header["per_dose_counts"]))
            self.assertEqual(header["keys_not_admitted"], pool.keys_not_admitted)
            self.assertIn([0, 0, ["nominal_observer_detects_before_reveal"]], header["not_admitted_keys"])
            self.assertEqual(header["finished_episodes_by_exposure"][NA], 1)
            self.assertEqual(header["exposure_rule"], J.D16_RULE)
            self.assertFalse(header["pool_pinned"])                     # a test pool given by cfg path
            runs.append((forces, record))
        reference_forces, reference = runs[0]
        for dose, (forces, record) in enumerate(runs):
            self.assertEqual(len(forces), len(reference_forces), dose)
            for a, b in zip(forces, reference_forces):
                self.assertTrue(torch.equal(a, b), dose)                 # every level runs D0
            self.assertEqual(record["scenario_hashes"], reference["scenario_hashes"], dose)
            self.assertEqual(record["contract"], reference["contract"], dose)
        print("not-admitted BlueBoat key: %d control steps, code %d at every dynamic level"
              % (len(reference_forces), reference["contract"]["termination_reason"]))

    def test_wamv_not_admitted_key_and_aborted_row(self):
        env, record, exposure = short_record("wamv_vrx_v2_opt1", WAMV_NA_SEED, ("dynamic=D4", "waves=D1"))
        self.assertEqual(exposure, NA)
        payload = record["dynamic_hazards"]["payload"]
        self.assertEqual((payload["exposure"], payload["assignment"]["requested_dose"]), (NA, 4))
        self.assertEqual(payload["assignment"]["rejection_reasons"], ["nominal_observer_detects_before_reveal"])
        self.assertEqual(env.journal_episode_record(0)["dynamic_hazards"], record["dynamic_hazards"])
        cfg = F.journal_cfg("wamv_vrx_v2_opt1", num_envs=1, episodes=2, tokens=("dynamic=D2",), seed=WAMV_NA_SEED)
        env = F.make_env(cfg)
        env.reset()
        for _ in range(3):
            env.step(F.seek_actions(env))
        env.reset()                                                     # manual reset mid-episode
        aborted = env.journal_aborted_dynamic_hazards[-1]
        F.dynamic.read_stamp(aborted)
        self.assertEqual(aborted["payload"]["exposure"], NA)
        self.assertEqual(aborted["payload"]["runtime"]["termination_reason"], "aborted")
        self.assertEqual(aborted["payload"]["runtime"]["last_time_s"], 6 * F.DT)
        self.assertEqual(env.dynamic_hazard_header["finished_episodes_by_exposure"],
                         {"assigned": 0, NA: 0, "aborted": 1})
        self.assertEqual(int(env._scenario.episode_indices()[0]), 1)     # the next key (episode 1)
        self.assertEqual(env._journal_d7_exposure[0],
                         J.EXPOSURE_ASSIGNED if env._d7_pool.is_admitted(0, 1) else NA)


class NotAdmittedPairingTests(unittest.TestCase):
    def test_pairs_with_the_same_key_at_d0_of_every_other_suite(self):
        vessel, seed = "blueboat_v2", BLUEBOAT_NA_SEED
        _, base, exposure = short_record(vessel, seed, ("dynamic=D0",))
        self.assertEqual(exposure, NA)
        for dose in range(5):
            dyn = "dynamic=D%d" % dose
            _, only_dynamic, _ = short_record(vessel, seed, (dyn,))
            self.assertEqual(only_dynamic["dynamic_hazards_exposure"], NA)
            self.assertEqual(only_dynamic["scenario_hashes"], base["scenario_hashes"], dose)
            if dose:
                with self.assertRaises(F.pairing.PairingError):         # unexposed for the dynamic treatment
                    F.pairing.validate_pair(only_dynamic, base, "dynamic")
            for suite, token in OTHER_SUITES.items():
                _, record, _ = short_record(vessel, seed, (dyn, token))
                summary = F.pairing.validate_pair(record, base, suite)
                self.assertTrue(set(summary["changed_primitives"]) <= set(F.pairing.SUITE_PRIMITIVES[suite]))
                self.assertIn("d7_assignment", summary["unchanged_primitives"])
                self.assertIn("route", summary["unchanged_primitives"])
                self.assertEqual(record["dynamic_hazards"]["payload"]["assignment"]["requested_dose"], dose)

    def test_admitted_key_is_a_dynamic_treatment_pair(self):
        _, d0, exposure0 = short_record("blueboat_v2", F.SEED, ("dynamic=D0",))
        _, d2, exposure2 = short_record("blueboat_v2", F.SEED, ("dynamic=D2",))
        self.assertEqual((exposure0, exposure2), (J.EXPOSURE_ASSIGNED, J.EXPOSURE_ASSIGNED))
        summary = F.pairing.validate_pair(d2, d0, "dynamic")
        self.assertEqual(summary["changed_primitives"], ["d7_assignment"])
        self.assertEqual(d2["dynamic_hazards_exposure"], J.EXPOSURE_ASSIGNED)


class NoRedrawTests(unittest.TestCase):
    def test_no_attempt_machinery_is_left(self):
        params = list(inspect.signature(J.journal_layout).parameters)
        self.assertEqual(params, ["vehicle", "tier", "seed", "env_index", "episode_index"])
        for name in ("layout_group", "LAYOUT_ATTEMPTS_MAX", "LAYOUT_ATTEMPT_RULE"):
            self.assertFalse(hasattr(J, name), name)
        source = inspect.getsource(J)
        self.assertNotIn("d7_attempt", source)
        self.assertNotIn("layout_attempt", source)

    def test_pool_answers_every_dose_of_a_not_admitted_key(self):
        path = F.pool_file("blueboat_v2", 1, 2, seed=BLUEBOAT_NA_SEED)
        pool = J.read_pool_file(path)
        key = F.dynamic.EpisodeKey(BLUEBOAT_NA_SEED, 0, 0, J.HAZARD_GROUP)
        digests = set()
        for dose in range(5):
            item = pool.assignment(key, dose)
            self.assertIsInstance(item, J.NotAdmittedAssignment)
            self.assertEqual(item.requested_dose, dose)
            digests.add(item.candidate_digest)
        self.assertEqual(len(digests), 1)                                # dose-independent identity
        self.assertEqual(pool.keys_not_admitted, 1 + int(not pool.is_admitted(0, 1)))
        with self.assertRaises(KeyError):
            pool.assignment(key, 5)
        with self.assertRaises(KeyError):                                # outside the grid: preflight error
            pool.assignment(F.dynamic.EpisodeKey(BLUEBOAT_NA_SEED, 1, 0, J.HAZARD_GROUP), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
