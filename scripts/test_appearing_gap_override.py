"""G20b (owner decision 2026-10-03): the diagnostic token appearing.gap_m (calibration sweep). CPU only.

Checks the G16 windcoef / T25 rollpitch pattern for the appearing trigger-gap override:
* resolution: appearing.gap_m=<x> is carried by the resolved stack (tokens round trip, stamp block with
  gap_override_m and diagnostic_only), needs the appearing suite at a dose >= 1 (C2), a positive number (C5);
  every other appearing.<field> stays C6; absent token -> stamp and tokens unchanged;
* bounds: the vessel's calibration rung D5 (0.76 / 1.25 m, G20c: no smaller gap is witnessed) up to the task's
  start gap (calibration tasks: the largest sweep gap, BlueBoat 7.5 m, WAM-V 15.0 m; docking: D1); the override
  never enters the suite identity (the same pools serve the sweep);
* runtime: the REAL AppearingBridge triggers at the override gap and stamps gap_override_m, dose_gap_m, the
  position against the witnessed gaps and k_equivalent; the stamp reader rejects an inconsistent override;
* refusals in official runs: eval_protocol preflight (tokens and suite file), run_cell's request detector, the
  cert_v3 official check, validate_cert_v3 (certificate_status reason, header errors);
* G20c (REV-G20b m4, m5): a suite file spelling the key with JSON \\u escapes is decoded and refused (preflight
  and run_cell); a missing registration counts as official; the supplementary dose appearing=D5 is refused
  outside diagnostic runs at every layer (preflight tokens / suite file, resolved stack, run_cell, cert_v3,
  validate_cert_v3).
Run: python -B scripts/test_appearing_gap_override.py   (or pytest)
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, str(Path(_SCRIPTS).parent / "tasks" / "_shared" / "conditions" / "appearing" / "tests"))
import g3_fixtures as G  # noqa: E402

C, ST, RT, LAD = G.config, G.stamp, G.runtime, G.ladder
JC = G.load("_shared.conditions.journal_config")
PHYSICS_DT, DECIMATION = 1.0 / 120.0, 2


class Resolution(unittest.TestCase):
    def test_token_round_trip_and_stamp(self):
        r = JC.resolve_journal_conditions(["appearing=D1", "appearing.gap_m=1.4"], seed=3)
        self.assertEqual((r.appearing_dose, r.appearing_gap_m), (1, 1.4))
        self.assertIn("appearing.gap_m=1.4", r.tokens())
        block = r.stamp()["appearing"]
        self.assertEqual((block["gap_override_m"], block["diagnostic_only"]), (1.4, True))
        self.assertEqual(JC.resolve_journal_conditions(r.tokens(), seed=3).stamp(), r.stamp())
        self.assertEqual(r.appearing_config(G.BB).gap_override_m, 1.4)
        plain = JC.resolve_journal_conditions(["appearing=D1"], seed=3)
        self.assertNotIn("gap_override_m", plain.stamp()["appearing"])
        self.assertNotIn("appearing.gap_m", " ".join(plain.tokens()))
        self.assertEqual(r.appearing_config(G.BB).identity(), plain.appearing_config(G.BB).identity())

    def test_declared_conflicts(self):
        for tokens, code in ((["appearing.gap_m=1.4"], "C2"), (["appearing=D0", "appearing.gap_m=1.4"], "C2"),
                             (["appearing=D1", "appearing.gap_m=-1"], "C5"),
                             (["appearing=D1", "appearing.gap_m=\"x\""], "C5"),
                             (["appearing=D1", "appearing.trigger_m=1.4"], "C6")):
            with self.assertRaises(JC.SuiteConflict) as caught:
                JC.resolve_journal_conditions(tokens, seed=3)
            self.assertEqual(caught.exception.code, code, tokens)

    def test_bounds(self):
        bb, wv = G.envelope(G.BB), G.envelope(G.WAMV)
        self.assertEqual(LAD.GAP_OVERRIDE_MIN_M, LAD.CALIBRATION_RUNG_M)
        self.assertEqual(LAD.GAP_OVERRIDE_MIN_M, {G.BB: 0.76, G.WAMV: 1.25})
        for value in (0.66, 0.67, 0.70, 0.75):                      # below the BlueBoat calibration rung
            with self.assertRaises(ValueError):
                C.AppearingSuiteConfig(enabled=True, dose=1, vessel_id=G.BB, gap_override_m=value)
        C.AppearingSuiteConfig(enabled=True, dose=1, vessel_id=G.BB, gap_override_m=0.76)
        top_scatter = C.start_gap_m("scatter", bb.t_avoid_s, G.BB)
        top_docking = C.start_gap_m("docking", bb.t_avoid_s, G.BB)
        self.assertEqual(C.check_gap_override("scatter", bb.t_avoid_s, G.BB, top_scatter), top_scatter)
        for task, value in (("scatter", top_scatter + 0.01), ("docking", top_docking + 0.01), ("scatter", 0.6)):
            with self.assertRaises(ValueError):
                C.check_gap_override(task, bb.t_avoid_s, G.BB, value)
        self.assertEqual(C.check_gap_override("path_following", wv.t_avoid_s, G.WAMV, 1.25), 1.25)
        for value in (1.03, 1.05, 1.2):                             # below the WAM-V calibration rung
            with self.assertRaises(ValueError):
                C.check_gap_override("path_following", wv.t_avoid_s, G.WAMV, value)
        if C.LADDER_KIND == "metres":
            # G47 (v2.6.4): scatter D1 7.5 m; G48/G49: BlueBoat docking D1 is not applicable by design, so its start
            # gap (the upper bound of appearing.gap_m) is D2 = 3.5 m
            self.assertEqual((top_scatter, top_docking), (7.5, 3.5))
            self.assertEqual(LAD.CALIBRATION_SWEEP_M[G.BB], (0.76, 1.0, 1.4, 2.0, 2.9, 4.0, 5.0, 6.6, 7.5))
            self.assertEqual(LAD.CALIBRATION_SWEEP_M[G.WAMV], (1.25, 1.4, 1.8, 2.5, 3.6, 5.0, 7.5, 11.0, 15.0))
            for vessel, sweep in LAD.CALIBRATION_SWEEP_M.items():      # sweep floor = D5, the witnessed rung
                self.assertEqual(min(sweep), LAD.CALIBRATION_RUNG_M[vessel])
                self.assertEqual(min(sweep), C.trigger_gap_m(5, G.envelope(vessel).t_avoid_s, vessel))
            self.assertEqual(C.start_gap_m("scatter", wv.t_avoid_s, G.WAMV), 15.0)
            for vessel, sweep in LAD.CALIBRATION_SWEEP_M.items():      # every sweep gap is a legal token value
                env = G.envelope(vessel)
                for value in sweep:
                    for task in LAD.CALIBRATION_TASKS:
                        C.check_gap_override(task, env.t_avoid_s, vessel, value)


class Runtime(unittest.TestCase):
    def run_override(self, override, dose=1):
        pool = G.reference_pool("path_following", G.BB)
        row, episode = G.first_admitted(pool)
        env = G.envelope(G.BB)
        a = pool.assignment(G.key("path_following", row, episode), dose)
        bridge = RT.AppearingBridge("path_following", 1, PHYSICS_DT, env.top_speed_mps, env.footprint, env.t_avoid_s,
                                    gap_override_m=override)
        bridge.reset_rows((row,), {row: a})
        x, y, _ = a.cylinder
        start = (x - 40.0, y)
        G.straight_run(bridge, row, start, 0.0, 1.0, PHYSICS_DT, DECIMATION, 4000)
        return bridge.finish(row, "collision"), a, bridge

    def test_bridge_triggers_at_the_override_and_stamps_it(self):
        for override in (0.76, 1.4, 6.6):
            stamp, a, _ = self.run_override(override)
            rt = stamp["payload"]["runtime"]
            self.assertEqual(rt["trigger_gap_m"], override)
            self.assertEqual(rt["gap_override_m"], override)
            self.assertEqual(rt["dose_gap_m"], a.trigger_gap_m)
            self.assertLessEqual(rt["trigger"]["gap_m"], override)
            self.assertGreater(rt["trigger"]["gap_m"], override - PHYSICS_DT * DECIMATION - 1e-9)
            self.assertAlmostEqual(rt["k_equivalent"], override / G.envelope(G.BB).t_avoid_s, places=12)
            low = C.smallest_trigger_gap_m(G.envelope(G.BB).t_avoid_s, G.BB)
            high = C.largest_trigger_gap_m(G.envelope(G.BB).t_avoid_s, G.BB)
            want = ("below_smallest_witnessed_gap" if override < low else
                    "above_largest_witnessed_gap" if override > high else "between_witnessed_gaps")
            self.assertEqual(rt["gap_override_position"], want)
            self.assertEqual(ST.read_appearing_stamp(stamp), stamp)
            bad = copy.deepcopy(stamp)
            bad["payload"]["runtime"]["trigger_gap_m"] = a.trigger_gap_m
            with self.assertRaises(ValueError):
                ST.read_appearing_stamp({"payload": bad["payload"], "sha256": ST.digest(bad["payload"])})

    def test_override_bounds_at_the_bridge(self):
        env = G.envelope(G.BB)
        with self.assertRaises(ValueError):
            RT.AppearingBridge("docking", 65, PHYSICS_DT, env.top_speed_mps, env.footprint, env.t_avoid_s,
                               gap_override_m=C.start_gap_m("docking", env.t_avoid_s, G.BB) + 0.5)


class OfficialRefusals(unittest.TestCase):
    def test_eval_protocol_preflight(self):
        import eval_protocol as EP
        official = {"purpose": "pilot", "namespace": "analysis"}
        diagnostic = {"purpose": "diagnostic", "namespace": "diagnostic"}
        args = SimpleNamespace(suite=["appearing=D1", "appearing.gap_m=1.4"], suite_file=None,
                               seed_registration=official)
        self.assertEqual(EP.appearing_gap_override_requested(args)[0], ["appearing.gap_m=1.4"])
        with self.assertRaises(EP.OverrideError):
            EP.appearing_gap_override_preflight(args)
        EP.appearing_gap_override_preflight(SimpleNamespace(suite=args.suite, suite_file=None,
                                                            seed_registration=diagnostic))
        EP.appearing_gap_override_preflight(SimpleNamespace(suite=["appearing=D1"], suite_file=None,
                                                            seed_registration=official))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "suites.json"
            path.write_text('{"suites": {"appearing": "D1"}, "parameters": {"appearing.gap_m": 2.0}}', encoding="ascii")
            file_args = SimpleNamespace(suite=[], suite_file=str(path), seed_registration=official)
            self.assertTrue(EP.appearing_gap_override_requested(file_args)[1])
            with self.assertRaises(EP.OverrideError):
                EP.appearing_gap_override_preflight(file_args)

    def test_escaped_suite_file_keys_are_decoded(self):
        """G20c (REV-G20b m4): a \\u-escaped key passes a raw text search; the decoded suite file does not."""
        import eval_protocol as EP
        import run_cell as RC
        official = {"purpose": "pilot", "namespace": "analysis"}
        with tempfile.TemporaryDirectory() as tmp:
            esc, nested = Path(tmp) / "esc.json", Path(tmp) / "nested.json"
            esc.write_text('{"suite": ["appearing=D1", "appearing.\\u0067ap_m=2.0"]}', encoding="ascii")
            nested.write_text('{"suites": {"appearing": "D1"}, "overrides": {"appearing": {"\\u0067ap_m": 2.0}}}',
                              encoding="ascii")
            for path in (esc, nested):
                self.assertNotIn("gap_m", path.read_text(encoding="ascii"))           # the raw search misses it
                args = SimpleNamespace(suite=[], suite_file=str(path), seed_registration=official)
                self.assertTrue(EP.appearing_gap_override_requested(args)[1], path.name)
                with self.assertRaises(EP.OverrideError):
                    EP.appearing_gap_override_preflight(args)
                self.assertTrue(RC.appearing_gap_override_requested(SimpleNamespace(suite=[], suite_file=str(path)))[1])
                EP.appearing_gap_override_preflight(SimpleNamespace(suite=[], suite_file=str(path),
                                                                    seed_registration={"purpose": "diagnostic"}))
            for registration in (None, {}, {"purpose": None}):          # no registration = official
                with self.assertRaises(EP.OverrideError):
                    EP.appearing_gap_override_preflight(SimpleNamespace(suite=[], suite_file=str(esc),
                                                                        seed_registration=registration))

    def test_supplementary_dose_is_diagnostic_only(self):
        """G20c (REV-G20b m5): appearing=D5 (the calibration rung) is refused outside diagnostic runs."""
        import eval_protocol as EP
        import run_cell as RC
        import cert_v3 as CV
        import validate_cert_v3 as V
        import appearing_exposure as AE
        official, diagnostic = {"purpose": "selection"}, {"purpose": "diagnostic"}
        self.assertEqual(C.SUPPLEMENTARY_DOSES, (5,))
        args = SimpleNamespace(suite=["appearing=D5"], suite_file=None, seed_registration=official)
        self.assertEqual(EP.appearing_supplementary_dose_requested(args)[0], ["appearing=D5"])
        with self.assertRaises(EP.OverrideError):
            EP.appearing_gap_override_preflight(args)
        EP.appearing_gap_override_preflight(SimpleNamespace(suite=["appearing=D5"], suite_file=None,
                                                            seed_registration=diagnostic))
        for main in ("appearing=D0", "appearing=D4", "waves=D4"):          # the main table passes
            EP.appearing_gap_override_preflight(SimpleNamespace(suite=[main], suite_file=None,
                                                                seed_registration=official))
        with tempfile.TemporaryDirectory() as tmp:
            for i, text in enumerate(('{"suites": {"appearing": "D5"}}', '{"suite": ["appea\\u0072ing=D5"]}')):
                path = Path(tmp) / ("supp%d.json" % i)
                path.write_text(text, encoding="ascii")
                file_args = SimpleNamespace(suite=[], suite_file=str(path), seed_registration=official)
                self.assertTrue(EP.appearing_supplementary_dose_requested(file_args)[1], text)
                with self.assertRaises(EP.OverrideError):
                    EP.appearing_gap_override_preflight(file_args)
        self.assertEqual(RC.appearing_supplementary_dose_requested.__module__, "eval_protocol")
        block = {"dose": 5, "suite": {"vessel_id": G.BB}, "supplementary_doses": [5]}
        for registration in ({"purpose": "final_test"}, None):
            with self.assertRaises(ValueError):
                CV.appearing_supplementary_dose_official_check(block, registration)
        CV.appearing_supplementary_dose_official_check(block, diagnostic)
        CV.appearing_supplementary_dose_official_check(dict(block, dose=4), {"purpose": "final_test"})
        for purpose in ("pilot", None):
            with self.assertRaises(ValueError):
                AE.supplementary_dose_official_check(block, purpose)
        AE.supplementary_dose_official_check(block, "diagnostic")
        AE.supplementary_dose_official_check(dict(block, dose=4), "pilot")
        status = V.certificate_status({"header": {"seed_namespace": {"purpose": "pilot"},
                                                  "appearing_hazards_header": block}})
        self.assertFalse(status["official"])
        self.assertTrue(any("supplementary appearing dose" in r for r in status["reasons"]))

    def test_run_cell_detects_the_token(self):
        import run_cell as RC
        self.assertEqual(RC.appearing_gap_override_requested.__module__, "eval_protocol")
        tokens, _ = RC.appearing_gap_override_requested(SimpleNamespace(suite=["appearing.gap_m=2.9"], suite_file=None))
        self.assertEqual(tokens, ["appearing.gap_m=2.9"])

    def test_cert_v3_and_validator(self):
        import cert_v3 as CV
        import validate_cert_v3 as V
        import appearing_exposure as AE
        block = {"gap_override_m": 1.4, "suite": {"vessel_id": G.BB}}
        with self.assertRaises(ValueError):
            CV.appearing_gap_override_official_check(block, {"purpose": "final_test"})
        CV.appearing_gap_override_official_check(block, {"purpose": "diagnostic"})
        CV.appearing_gap_override_official_check({"suite": {}}, {"purpose": "final_test"})
        status = V.certificate_status({"header": {"seed_namespace": {"purpose": "pilot"},
                                                  "appearing_hazards_header": block}})
        self.assertFalse(status["official"])
        self.assertTrue(any("appearing.gap_m" in r for r in status["reasons"]))
        with self.assertRaises(ValueError):
            AE.gap_override_official_check(block, "pilot")
        with self.assertRaises(ValueError):                 # G20c (REV-G20b m4): no registration = official
            AE.gap_override_official_check(block, None)
        AE.gap_override_official_check(block, "diagnostic")
        with self.assertRaises(ValueError):
            AE.ladder_check({"gap_override_m": 0.5, "suite": {"vessel_id": G.BB}})
        t_avoid = G.envelope(G.BB).t_avoid_s
        header = {"suite": {"vessel_id": G.BB}, "suite_version": C.APPEARING_SUITE_VERSION,
                  "ladder_option": C.LADDER_OPTION, "t_avoid_s": t_avoid,
                  "trigger_gaps_m": C.trigger_gaps_m(t_avoid, G.BB), "hazard_group": C.hazard_group("scatter")}
        AE.ladder_check(header)
        for value in (0.76, 4.0, 7.5):                       # [calibration rung, start gap] of BlueBoat scatter
            AE.ladder_check(dict(header, gap_override_m=value))
        for value in (0.75, 7.51):
            with self.assertRaises(ValueError):
                AE.ladder_check(dict(header, gap_override_m=value))
        with self.assertRaises(ValueError):                  # docking: start gap = D1 = 7.5 m (G47 / v2.6.4)
            AE.ladder_check(dict(header, gap_override_m=7.6, hazard_group=C.hazard_group("docking")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
