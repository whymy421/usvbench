"""F2 D16 tests: D7-rejected keys never change the task layout (CONTEXT_R4 round-4b D16).

Run: python -B tasks/path_hazard/tests_journal/test_journal_d16.py   (CPU only)

With the in-memory WAM-V fixture pool (seed 42, 2 envs x 3 episodes, the same admission function as
the offline tool) key (42, 0, 0) is NOT admitted by E4 (nominal_prefix_unsafe) and key (42, 1, 0) is
admitted, so env row 0 is a not-admitted key and row 1 an admitted one in every run below.

* a not-admitted key runs a full episode (reset -> contract termination -> latch) at EVERY dynamic
  level D0..D4 and with the suite absent: no intruder, no D7 sample, record
  ``dynamic_hazards.payload.exposure == "not_admitted"`` with E4's reasons, the header counts
  not_admitted per dose; never an exception, never a redraw;
* its records pair (pairing.validate_pair) with the same key at D0 of every other suite (waves,
  wind, current, actuator, perception, perception.bundle at D1), and are UNEXPOSED for the dynamic
  treatment (validate_pair(..., "dynamic") raises "no treatment difference"), while the admitted
  row pairs as a dynamic treatment;
* the layout of a not-admitted key equals the E5 layout of that key (build_layout on the keyed
  stream "journal_layout", computed here independently of the family module);
* NotAdmittedBridge stamps verify with the E4 and certificate readers and keep the bridge rules.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ph_fixtures as F  # noqa: E402

M = F.modules()
J = M.journal
STUBS = F.install_stubs()
CERT = importlib.import_module("tasks._shared.conditions.certificate")
VESSEL = "wamv_vrx_v2_opt1"
NA, ADM = 0, 1                     # env rows: key (42,0,0) not admitted, key (42,1,0) admitted
OTHER_SUITES = ("waves", "wind", "current", "actuator", "perception", "perception.bundle")
_RUNS = {}


def pool():
    return F.test_pool(VESSEL, 2, 3)


def episode(tokens):
    """One full journal episode of both rows (contract timeout after 3 control steps), cached.

    Returns (env, records, d7_seen, first): ``records`` are the latched episode-0 records,
    ``d7_seen[i]`` whether row i ever published a D7 sample during episode 0 (every publication is
    recorded with the row's episode index; the terminating step auto-resets the rows into episode
    1, whose keys are admitted), ``first`` a snapshot of episode 0 taken right after the reset.
    """
    if tokens in _RUNS:
        return _RUNS[tokens]
    env = F.make_env(F.journal_cfg(VESSEL, tokens=tokens, pool=pool()))
    published = []
    original = env._journal_publish_d7

    def publish(samples):
        episodes = env._scenario.episode_indices().tolist()
        published.extend((int(i), int(episodes[int(i)]), len(samples[i])) for i in samples)
        return original(samples)
    env._journal_publish_d7 = publish
    env.reset()
    n = env.num_envs
    first = {"layouts": list(env._jph_layouts), "centers": env._jph_centers.clone(),
             "waypoints": env._jph_waypoints.clone(), "params": [dict(p) for p in env._scenario_params],
             "not_admitted": [env._d7_bridge.is_not_admitted(i) for i in range(n)]}
    routes = torch.cat((torch.zeros((n, 1, 2), dtype=torch.float64), env._jph_waypoints), dim=1)
    env._jph_oracle.reset(torch.arange(n), routes, env._jph_gate_radius,
                          torch.full((n,), 3.0 / 60.0, dtype=torch.float64), routes[:, 0])
    for k in range(3):
        env.step(F.actions(k, n))
    records = [env.journal_episode_record(i) for i in range(n)]
    d7_seen = [any(c > 0 for r, e, c in published if r == i and e == 0) for i in range(n)]
    assert all(any(r == i and e == 0 for r, e, _ in published) for i in range(n))   # publication did run
    _RUNS[tokens] = (env, records, d7_seen, first)
    return _RUNS[tokens]


class Fixture(unittest.TestCase):
    def test_fixture_keys(self):
        p = pool()
        self.assertFalse(p.is_admitted(42, NA, 0))
        self.assertTrue(p.is_admitted(42, ADM, 0))
        reasons = p.not_admitted[(42, NA, 0)]
        self.assertTrue(J.primary_reasons(reasons))
        for dose in range(5):
            record = p.assignment(J.episode_key(42, NA, 0), dose)
            self.assertIsInstance(record, J.NotAdmittedAssignment)          # never an exception
            self.assertEqual(record.reasons, tuple(tuple(r) for r in reasons))
            self.assertEqual(p.assignment(J.episode_key(42, ADM, 0), dose).candidate.request.encounter.dose, dose)
        with self.assertRaises(KeyError):
            p.assignment(J.episode_key(42, 7, 0), 0)                        # outside the offered grid
        self.assertEqual(p.per_dose_counts, ((6, p.keys_admitted, p.keys_not_admitted),) * 5)


class NotAdmittedEpisodes(unittest.TestCase):
    def test_full_episode_at_every_dynamic_level(self):
        p = pool()
        reasons = [list(r) for r in p.not_admitted[(42, NA, 0)]]
        for dose in range(5):
            env, records, d7_seen, _ = episode(("dynamic=D%d" % dose,))
            self.assertEqual(env.termination_reason.tolist(), [3, 3])        # both rows ran to the contract end
            self.assertFalse(d7_seen[NA])                                    # no intruder, ever
            self.assertEqual(d7_seen[ADM], dose > 0)
            payload = records[NA]["dynamic_hazards"]["payload"]
            M.dynamic.read_stamp(records[NA]["dynamic_hazards"])             # E4 reader accepts it
            self.assertEqual(payload["exposure"], "not_admitted")
            self.assertEqual(J.record_exposure(records[NA]["dynamic_hazards"]), "not_admitted")
            self.assertEqual(payload["not_admitted_reasons"], reasons)
            self.assertEqual(payload["assignment"]["dose"], 0)
            self.assertIsNone(payload["assignment"]["candidate_digest"])
            self.assertEqual(payload["assignment"]["intruders"], [])
            self.assertEqual(payload["runtime"]["suite_dose"], dose)
            self.assertEqual(payload["runtime"]["non_exposure_reason"], "not_admitted")
            self.assertIs(payload["runtime"]["assigned"], False)            # D27: never assigned,
            self.assertEqual(payload["runtime"]["itt_count"], 0)            # outside the ITT denominator
            self.assertEqual(payload["runtime"]["termination_reason"], "timeout")
            self.assertEqual(payload["runtime"]["bridge"]["last_tick"], 6)
            self.assertEqual(env.journal_completed[NA]["d7_exposure"], "not_admitted")
            self.assertEqual(env.journal_completed[ADM]["d7_exposure"], "admitted")
            admitted = records[ADM]["dynamic_hazards"]["payload"]
            self.assertNotIn("exposure", admitted)
            self.assertEqual(admitted["assignment"]["dose"], dose)
            self.assertIs(admitted["runtime"]["assigned"], True)              # E4 stamp of an admitted key
            self.assertEqual(admitted["runtime"]["itt_count"], 1)
            header = env.journal_certificate_header(treatment="dynamic")["dynamic_hazards_header"]
            self.assertEqual(header["per_dose_counts"], [[6, p.keys_admitted, p.keys_not_admitted]] * 5)
            self.assertEqual(header["per_dose_counts_columns"], ["offered", "admitted", "not_admitted"])
            self.assertEqual(header["keys_not_admitted"], p.keys_not_admitted)
            self.assertIn("pool_manifest_sha256", header)
            fields = CERT.episode_certificate_fields(env, NA)
            self.assertEqual(fields["dynamic_hazards"]["payload"]["exposure"], "not_admitted")

    def test_suite_absent_equals_d0_for_a_not_admitted_key(self):
        _, absent, _, _ = episode(())
        _, d0, _, _ = episode(("dynamic=D0",))
        self.assertEqual(absent[NA]["scenario_hashes"], d0[NA]["scenario_hashes"])
        self.assertEqual(absent[ADM]["scenario_hashes"], d0[ADM]["scenario_hashes"])

    def test_pairs_with_the_same_key_at_d0_of_every_other_suite(self):
        pairing = M.pairing
        for suite in OTHER_SUITES:
            _, other, _, _ = episode((suite + "=D1",))
            self.assertIn("exposure", other[NA]["dynamic_hazards"]["payload"])
            for dose in range(5):
                _, dyn, _, _ = episode(("dynamic=D%d" % dose,))
                result = pairing.validate_pair(dyn[NA], other[NA], suite)
                self.assertTrue(set(result["changed_primitives"]) <= set(pairing.treatment_primitives(suite)))
                self.assertNotIn(pairing.D7_PRIMITIVE, result["changed_primitives"])
                if dose == 0:    # the admitted row pairs for the other suites too (reference)
                    pairing.validate_pair(dyn[ADM], other[ADM], suite)

    def test_unexposed_for_the_dynamic_treatment(self):
        pairing = M.pairing
        _, d0, _, _ = episode(("dynamic=D0",))
        for dose in range(1, 5):
            _, dyn, _, _ = episode(("dynamic=D%d" % dose,))
            self.assertEqual(dyn[NA]["scenario_hashes"], d0[NA]["scenario_hashes"])
            with self.assertRaisesRegex(pairing.PairingError, "no treatment difference"):
                pairing.validate_pair(d0[NA], dyn[NA], "dynamic")
            result = pairing.validate_pair(d0[ADM], dyn[ADM], "dynamic")
            self.assertEqual(result["changed_primitives"], [pairing.D7_PRIMITIVE])


class LayoutIsTheE5Layout(unittest.TestCase):
    def test_not_admitted_key_keeps_its_e5_layout(self):
        layouts = __import__("tasks._shared.journal_layouts_v2", fromlist=["build_layout"])
        rng_mod = __import__("tasks._shared.scenario_rng", fromlist=["numpy_generator"])
        rng = rng_mod.numpy_generator(int(rng_mod.SCENARIO_PROTOCOL_VERSION), 42, NA, 0, "journal_layout")
        expected = layouts.build_layout("path_hazard", 2, VESSEL, rng, max_attempts=200)
        hashes = set()
        for tokens in ((), ("dynamic=D0",), ("dynamic=D2",), ("dynamic=D4",), ("waves=D1",)):
            _, records, _, first = episode(tokens)
            layout = first["layouts"][NA]
            self.assertEqual(layout.geometry_hash, expected.geometry_hash)
            self.assertEqual(layout.geometry_hash, pool().geometry_hashes[(42, NA, 0)])
            count = expected.obstacle_count
            self.assertTrue(np.array_equal(first["centers"][NA, :count].numpy(), np.asarray(expected.centers)))
            self.assertTrue(np.array_equal(first["waypoints"][NA].numpy(), np.asarray(expected.waypoints)))
            self.assertEqual(first["params"][NA]["layout"]["d7_exposure"], "not_admitted")
            hashes.add(tuple(records[NA]["scenario_hashes"][k] for k in ("layout", "route", "spawn_pose")))
        self.assertEqual(len(hashes), 1)
        source = (F.REPO / "tasks" / "path_hazard" / "path_hazard_journal.py").read_text(encoding="utf-8")
        for token in ("attempt=", "layout_attempt", "admit_with_redraws", "journal_layout/a"):
            self.assertNotIn(token, source)


class BridgeRules(unittest.TestCase):
    def test_not_admitted_bridge_rows(self):
        env, records, _, first = episode(("dynamic=D3",))
        bridge = env._d7_bridge
        self.assertIsInstance(bridge, J.NotAdmittedBridge)
        self.assertEqual(first["not_admitted"], [True, False])      # episode 0: row 0 not admitted
        self.assertFalse(bridge.is_not_admitted(NA))                # its episode 1 key is admitted (E4 row)
        stamp = records[NA]["dynamic_hazards"]
        CERT.verify_dynamic_hazard_stamp(stamp)
        other = J.NotAdmittedBridge(bridge.inner, 3)
        with self.assertRaises(ValueError):          # row 0 runs an unfinished (admitted) E4 episode
            other.reset_rows((NA,), {NA: pool().assignment(J.episode_key(42, NA, 0), 3)})
        r = 5                                        # a row the E4 bridge has never seen
        record = J.NotAdmittedAssignment(J.episode_key(42, r, 0), pool().not_admitted[(42, NA, 0)],
                                         pool().pool_digest)
        other.reset_rows((r,), {r: record})
        self.assertEqual(other.step({r: 0}, {}), {r: ()})
        self.assertEqual(other.step({r: 0}, {}), {r: ()})           # a consumed tick again: no error
        self.assertFalse(other.contact(r))
        with self.assertRaises(ValueError):
            other.reset_rows((r,), {r: record})                    # unfinished row
        with self.assertRaises(ValueError):
            other.finish(r, "not_a_reason")
        with self.assertRaises(ValueError):
            other.step({r: -1}, {})
        s = other.finish(r, "success")
        self.assertIs(other.finish(r, "timeout"), s)               # finished: the same stamp back
        with self.assertRaises(ValueError):
            other.step({r: 99}, {})                                # finished row
        self.assertEqual(s["payload"]["runtime"]["termination_reason"], "success")
        self.assertEqual((s["payload"]["runtime"]["assigned"], s["payload"]["runtime"]["itt_count"]), (False, 0))
        CERT.verify_dynamic_hazard_stamp(s)
        at_d0 = J.NotAdmittedBridge(bridge.inner, 0)                # same key, suite at D0: same primitive
        at_d0.reset_rows((r,), {r: record})
        s0 = at_d0.finish(r, "timeout")
        self.assertEqual(M.pairing.d7_primitive_hash(s), M.pairing.d7_primitive_hash(s0))
        self.assertNotEqual(M.pairing.d7_primitive_hash(s), M.pairing.d7_primitive_hash(stamp))   # other key
        with self.assertRaises(ValueError):                        # a record filed under another row
            J.NotAdmittedBridge(bridge.inner, 3).reset_rows((6,), {6: record})


if __name__ == "__main__":
    unittest.main(verbosity=2)
