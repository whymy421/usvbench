"""G20b: realized time pressure in the appearing stamp + its reader, validator and schema; bridge-level
trigger mechanics on every journal task (CPU only; run: python -B test_g20b_trigger_stamp.py).

Covers:
  * the trigger record carries closing_speed_mps = surge x cos(relative bearing) and realized_ttc_s =
    max(gap, 0) / closing speed (None when not closing), produced by the REAL AppearingBridge;
  * stamp round trip: read_appearing_stamp accepts real stamps; a stamp whose time-pressure fields do not
    recompute (digest re-sealed) or lack them, or whose dose lies outside the active ladder, is rejected;
  * scripts/appearing_exposure.py (loaded the validator's way, by path): trigger_errors, record_errors,
    dose_count / per-dose rows and the header's ladder check follow the active ladder;
  * cert_v3.schema.json: the appearing stamp def accepts real stamps and rejects a trigger without the
    time-pressure fields (jsonschema Draft 2020-12, as validate_cert_v3);
  * G20c (REV-G20b M1): the reader pins every gap / k field to the ladder with the pinned t_avoid (REV tamper
    cases T1-T6 rejected; T11: a self-consistent trigger-speed edit inside the vessel envelope stays undetectable
    by design - the digest is a checksum, not a signature - and one outside the envelope is rejected), the gap
    override lies in [calibration rung, start gap]; the header's ladder fields are mandatory and every record
    agrees with the header (H1-H3); exposure_realized per controller and dose (realized_exposure_table);
  * trigger mechanics on all twelve tasks x both vessels: the nominal plan of every admitted key of a small
    grid replayed through the bridge at the control rate (1/120 s physics, decimation 2) triggers at the first
    control step whose gap is <= d_k, within one control step of nominal motion, at every witnessed dose.
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
import g3_fixtures as F  # noqa: E402

C, ST, P, POOL, RT = F.config, F.stamp, F.placement, F.pool_module, F.runtime
PHYSICS_DT, DECIMATION = 1.0 / 120.0, 2
CONTROL_DT = PHYSICS_DT * DECIMATION
SCRIPTS = F.REPO / "scripts"


def exposure_module():
    spec = importlib.util.spec_from_file_location("_g20b_appearing_exposure", SCRIPTS / "appearing_exposure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reseal(envelope):
    payload = envelope["payload"]
    return {"payload": payload, "sha256": ST.digest(payload)}


def triggered_stamp(dose=2, speed=1.0, heading=0.3, task="path_following", vessel=F.BB, gap_override_m=None):
    pool = F.reference_pool(task, vessel)
    row, episode = F.first_admitted(pool)
    envelope = F.envelope(vessel)
    a = pool.assignment(F.key(task, row, episode), dose)
    x, y, _ = a.cylinder
    slot = C.appearing_slot(F.plans.STATIC_CAPACITY[task], F.plans.D7_RESERVED_SLOTS[task])
    bridge = RT.AppearingBridge(task, slot, PHYSICS_DT, envelope.top_speed_mps, envelope.footprint,
                                envelope.t_avoid_s, gap_override_m=gap_override_m)
    bridge.reset_rows((row,), {row: a})
    distance = 40.0 * envelope.length_m
    start = (x - distance * math.cos(heading), y - distance * math.sin(heading))
    F.straight_run(bridge, row, start, heading, speed, PHYSICS_DT, DECIMATION, int(4 * distance / max(speed, 0.1)
                                                                                    / CONTROL_DT) + 10)
    return bridge.finish(row, "collision"), a


class TriggerRecord(unittest.TestCase):
    def test_closing_speed_and_ttc_are_recorded_and_recompute(self):
        for dose in C.WITNESSED_DOSES:
            stamp, a = triggered_stamp(dose=dose)
            trig = stamp["payload"]["runtime"]["trigger"]
            speed, bearing, gap = trig["pose"][3], trig["bearing_rad"], trig["gap_m"]
            self.assertAlmostEqual(bearing, 0.0, places=9)                    # dead ahead on the line
            self.assertEqual(trig["closing_speed_mps"], speed * math.cos(bearing))
            self.assertEqual(trig["realized_ttc_s"], max(gap, 0.0) / (speed * math.cos(bearing)))
            self.assertLessEqual(gap, a.trigger_gap_m)
            # at u_ref straight at the cylinder the realized TTC is the gap / u_ref = k_eq * t_avoid (one step)
            self.assertAlmostEqual(trig["realized_ttc_s"], a.trigger_gap_m / C.REFERENCE_SPEED_MPS,
                                   delta=CONTROL_DT + 1e-9)
            self.assertAlmostEqual(stamp["payload"]["runtime"]["k_equivalent"],
                                   a.trigger_gap_m / F.envelope(F.BB).t_avoid_s, places=12)
            self.assertEqual(ST.read_appearing_stamp(stamp), stamp)

    def test_not_closing_gives_no_ttc(self):
        self.assertIsNone(ST.realized_ttc_s(1.0, 0.0))
        self.assertIsNone(ST.realized_ttc_s(1.0, -0.4))
        self.assertEqual(ST.realized_ttc_s(-0.2, 2.0), 0.0)                    # contact already: TTC 0
        out = ST.trigger_pressure(1.5, math.pi, 1.0)                          # cylinder astern: opening
        self.assertLess(out["closing_speed_mps"], 0.0)
        self.assertIsNone(out["realized_ttc_s"])

    def test_tampered_or_missing_fields_are_rejected(self):
        stamp, _ = triggered_stamp(dose=1)
        for field, value in (("closing_speed_mps", 0.123), ("realized_ttc_s", 9.0), ("realized_ttc_s", None)):
            bad = copy.deepcopy(stamp)
            bad["payload"]["runtime"]["trigger"][field] = value
            with self.assertRaisesRegex(ValueError, "does not recompute"):
                ST.read_appearing_stamp(reseal(bad))
        for field in ST.TRIGGER_PRESSURE_FIELDS:
            bad = copy.deepcopy(stamp)
            del bad["payload"]["runtime"]["trigger"][field]
            with self.assertRaisesRegex(ValueError, "lacks " + field):
                ST.read_appearing_stamp(reseal(bad))
        bad = copy.deepcopy(stamp)
        bad["payload"]["runtime"]["trigger"]["closing_speed_mps"] = 0.5          # not re-sealed: digest
        with self.assertRaisesRegex(ValueError, "digest"):
            ST.read_appearing_stamp(bad)
        bad = copy.deepcopy(stamp)
        bad["payload"]["assignment"]["dose"] = C.MAX_DOSE + 1
        with self.assertRaisesRegex(ValueError, "outside D0"):
            ST.read_appearing_stamp(reseal(bad))
        bad = copy.deepcopy(stamp)
        bad["payload"]["runtime"]["trigger"] = None                             # realized without a record
        with self.assertRaises(ValueError):
            ST.read_appearing_stamp(reseal(bad))

    def test_schema_version_and_ladder_text(self):
        stamp, _ = triggered_stamp(dose=3)
        payload = stamp["payload"]
        self.assertEqual(payload["schema"], "usvbench.appearing_stamp/3")         # G20d: + visible_at_trigger
        self.assertIn("visibility", payload["semantics"])
        self.assertIsInstance(payload["runtime"]["visible_at_trigger"], bool)
        self.assertEqual(payload["suite_version"], C.APPEARING_SUITE_VERSION)
        self.assertIn("time_pressure", payload["semantics"])
        self.assertIn("ladder option " + C.LADDER_OPTION, payload["semantics"]["dose"])


class ValidatorAndSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AE = exposure_module()
        cls.stamp, cls.assignment = triggered_stamp(dose=2)

    def test_exposure_reader_follows_the_ladder(self):
        AE = self.AE
        self.assertEqual(AE.dose_count(), C.DOSE_COUNT)
        self.assertEqual(AE.trigger_errors(self.stamp), [])
        block = {"pool_pinned": True, "pool_manifest_sha256": "0" * 64, "grid": {"num_envs": 2, "episodes": 2},
                 "per_dose_counts": [[2, 2, 0]] * C.DOSE_COUNT,
                 "per_dose_counts_fields": ["offered", "admitted", "not_admitted"], "dose": 2,
                 "suite_version": C.APPEARING_SUITE_VERSION,
                 "gap_ratios": None if C.GAP_RATIOS is None else list(C.GAP_RATIOS),
                 "ladder_option": C.LADDER_OPTION, "suite": {"vessel_id": F.BB},
                 "t_avoid_s": F.envelope(F.BB).t_avoid_s,
                 "trigger_gaps_m": C.trigger_gaps_m(F.envelope(F.BB).t_avoid_s, F.BB)}
        records = [{"appearing_hazards": self.stamp}]
        self.assertEqual(AE.header_errors(block, records), [])
        for patch in ({"gap_ratios": [None, 4.0, 3.0, 2.0, 1.0]}, {"suite_version": "g3_appearing_gap_v1"},
                      {"trigger_gaps_m": [None, 9.0] + [1.0] * (C.DOSE_COUNT - 2)}, {"ladder_option": "Z"},
                      {"per_dose_counts": [[2, 2, 0]] * (C.DOSE_COUNT + 1)}, {"dose": C.MAX_DOSE + 1}):
            self.assertTrue(AE.header_errors(dict(block, **patch), records), patch)

    def test_record_errors_report_an_inconsistent_trigger(self):
        AE = self.AE
        document = {"records": [{"appearing_hazards": self.stamp}]}
        self.assertEqual(AE.record_errors(document), [])
        bad = copy.deepcopy(self.stamp)
        bad["payload"]["runtime"]["trigger"]["realized_ttc_s"] = 1.0e3
        errors = AE.record_errors({"records": [{"appearing_hazards": reseal(bad)}]})
        self.assertEqual(len(errors), 1)
        self.assertIn("does not recompute", errors[0])

    def test_schema_accepts_real_stamps_and_requires_the_fields(self):
        from jsonschema import Draft202012Validator
        schema = json.loads((SCRIPTS / "cert_v3.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator({"$defs": schema["$defs"], "$ref": "#/$defs/appearing_hazard_stamp"})
        self.assertEqual(list(validator.iter_errors(self.stamp)), [])
        untriggered = copy.deepcopy(self.stamp)
        untriggered["payload"]["runtime"]["trigger"] = None
        self.assertEqual(list(validator.iter_errors(untriggered)), [])
        bad = copy.deepcopy(self.stamp)
        del bad["payload"]["runtime"]["trigger"]["closing_speed_mps"]
        self.assertTrue(list(validator.iter_errors(bad)))
        bad = copy.deepcopy(self.stamp)
        bad["payload"]["schema"] = "usvbench.appearing_stamp/1"
        self.assertTrue(list(validator.iter_errors(bad)))
        record_def = schema["$defs"]
        self.assertIn("appearing_hazard_stamp", record_def)


def header_block(dose, vessel=F.BB, task="path_following", **extra):
    t_avoid = F.envelope(vessel).t_avoid_s
    block = {"pool_pinned": True, "pool_manifest_sha256": "0" * 64, "grid": {"num_envs": 2, "episodes": 2},
             "per_dose_counts": [[2, 2, 0]] * C.DOSE_COUNT,
             "per_dose_counts_fields": ["offered", "admitted", "not_admitted"], "dose": dose,
             "suite_version": C.APPEARING_SUITE_VERSION, "gap_ratios": None if C.GAP_RATIOS is None else
             list(C.GAP_RATIOS), "ladder_option": C.LADDER_OPTION, "suite": {"vessel_id": vessel},
             "t_avoid_s": t_avoid, "trigger_gaps_m": C.trigger_gaps_m(t_avoid, vessel),
             "hazard_group": C.hazard_group(task),
             # G20d (decision 2B): the dose applicability of (task, vessel), as pool.header_block writes it
             "dose_applicability": {"D%d" % d: (F.ladder.NOT_APPLICABLE if d in F.ladder.not_applicable_doses(task, vessel)
                                                else "offered") for d in range(1, C.DOSE_COUNT)},
             "not_applicable_doses": {"D%d" % d: F.ladder.not_applicable_reason(task, vessel, d)
                                      for d in F.ladder.not_applicable_doses(task, vessel)}}
    block.update(extra)
    return block


class TamperPins(unittest.TestCase):
    """G20c (REV-G20b M1): the REV tamper cases (evidence_rev/tamper_probe.json) against the pinned reader."""

    @classmethod
    def setUpClass(cls):
        cls.AE = exposure_module()
        cls.stamp, cls.assignment = triggered_stamp(dose=4)

    def rejected(self, envelope, why=None):
        with self.assertRaises(ValueError) as caught:
            ST.read_appearing_stamp(envelope)
        if why is not None:
            self.assertIn(why, str(caught.exception))
        self.assertTrue(self.AE.record_errors({"records": [{"appearing_hazards": envelope}]}))

    def test_real_stamps_pass_the_pins(self):
        self.assertEqual(ST.pin_errors(self.stamp["payload"]), [])
        rt = self.stamp["payload"]["runtime"]
        self.assertEqual(rt["trigger_gap_m"], C.trigger_gap_m(4, F.envelope(F.BB).t_avoid_s, F.BB))
        for dose in C.WITNESSED_DOSES:
            stamp, _ = triggered_stamp(dose=dose)
            self.assertEqual(ST.pin_errors(stamp["payload"]), [], dose)

    def test_t1_to_t6_are_rejected(self):
        payload = self.stamp["payload"]
        bad = copy.deepcopy(self.stamp)                                      # T1: both gaps 5.0 at D4
        bad["payload"]["runtime"]["trigger_gap_m"] = 5.0
        bad["payload"]["assignment"]["trigger_gap_m"] = 5.0
        self.rejected(reseal(bad), "ladder gap")
        bad = copy.deepcopy(self.stamp)                                      # T2: runtime k_equivalent
        bad["payload"]["runtime"]["k_equivalent"] = 99.0
        self.rejected(reseal(bad), "k_equivalent")
        bad = copy.deepcopy(self.stamp)                                      # T3: assignment k / gap_ratio
        bad["payload"]["assignment"]["k_equivalent"] = 99.0
        bad["payload"]["assignment"]["gap_ratio"] = 7.0
        self.rejected(reseal(bad))
        bad = copy.deepcopy(self.stamp)                                      # T4: D4 run relabelled D1
        bad["payload"]["assignment"]["dose"] = 1
        self.rejected(reseal(bad), "ladder gap")
        bad = copy.deepcopy(self.stamp)                                      # T5: override run, fields stripped
        bad["payload"]["runtime"]["trigger_gap_m"] = 0.7
        bad["payload"]["runtime"]["trigger"]["gap_m"] = 0.69
        trig = bad["payload"]["runtime"]["trigger"]
        trig.update(ST.trigger_pressure(trig["gap_m"], trig["bearing_rad"], trig["pose"][3]))
        self.rejected(reseal(bad), "trigger_gap_m")
        bad = copy.deepcopy(self.stamp)                                      # T6: vessel relabelled
        bad["payload"]["assignment"]["vessel_id"] = F.WAMV
        self.rejected(reseal(bad))
        self.assertEqual(payload, self.stamp["payload"])                     # the fixture stays untouched

    def test_t11_trigger_speed_envelope(self):
        top = F.envelope(F.BB).top_speed_mps
        self.assertEqual(ST.LADDER.VESSEL_TOP_SPEED_MPS[F.BB], top)
        for speed, ok in ((3.0, True), (top + 0.5, False), (-ST.LADDER.VESSEL_MAX_REVERSE_SPEED_MPS[F.BB] - 0.5,
                                                             False)):
            bad = copy.deepcopy(self.stamp)
            bad["payload"]["runtime"]["trigger"]["pose"][3] = speed
            trig = bad["payload"]["runtime"]["trigger"]
            trig.update(ST.trigger_pressure(trig["gap_m"], trig["bearing_rad"], trig["pose"][3]))
            if ok:      # documented limit: a self-consistent edit inside the envelope is not detectable
                ST.read_appearing_stamp(reseal(bad))
            else:
                self.rejected(reseal(bad), "envelope")

    def test_override_pins(self):
        stamp, a = triggered_stamp(dose=1, gap_override_m=1.4)
        self.assertEqual(ST.read_appearing_stamp(stamp), stamp)
        for patch in ({"dose_gap_m": 2.9}, {"gap_override_m": 0.70, "trigger_gap_m": 0.70},
                      {"gap_override_m": 9.0, "trigger_gap_m": 9.0}):
            bad = copy.deepcopy(stamp)
            bad["payload"]["runtime"].update(patch)
            self.rejected(reseal(bad))
        bad = copy.deepcopy(self.stamp)                                       # override fields without override
        bad["payload"]["runtime"]["dose_gap_m"] = bad["payload"]["runtime"]["trigger_gap_m"]
        self.rejected(reseal(bad), "without a gap override")

    def test_h1_to_h3_header_cases(self):
        AE = self.AE
        records = [{"appearing_hazards": self.stamp}]
        self.assertEqual(AE.header_errors(header_block(4), records), [])
        stripped = {k: v for k, v in header_block(4).items()
                    if k not in ("suite_version", "ladder_option", "gap_ratios", "trigger_gaps_m", "t_avoid_s")}
        errors = AE.header_errors(stripped, records)                         # H1: mandatory fields
        self.assertTrue(any("lacks" in m for m in errors), errors)
        errors = AE.header_errors(header_block(4, trigger_gaps_m=[None, 6.58, 4.93, 3.29, 2.06], t_avoid_s=None),
                                  records)                                   # H2: ladder A gaps, no t_avoid
        self.assertTrue(errors)
        self.assertTrue(AE.header_errors(header_block(4, t_avoid_s=3.74273175), records))   # WAM-V t_avoid
        no_vessel = header_block(4)
        no_vessel["suite"] = {}
        self.assertTrue(any("suite.vessel_id" in m for m in AE.header_errors(no_vessel, records)))
        bad = copy.deepcopy(self.stamp)                                      # H3: record gap 5.0 at D4
        bad["payload"]["runtime"]["trigger_gap_m"] = 5.0
        document = {"records": [{"appearing_hazards": reseal(bad)}]}
        self.assertTrue(AE.record_errors(document) + AE.header_errors(header_block(4), document["records"]))
        # a valid WAM-V stamp under a BlueBoat header: the record x header cross-check catches it
        wamv, _ = triggered_stamp(dose=4, vessel=F.WAMV)
        errors = AE.header_errors(header_block(4), [{"appearing_hazards": wamv}])
        self.assertTrue(any("differs from the header" in m for m in errors), errors)
        # a header override with a record that has none (and the reverse)
        self.assertTrue(AE.header_errors(header_block(4, gap_override_m=1.4), records))
        for purpose in (None, "pilot", "final_test"):
            with self.assertRaises(ValueError):
                AE.gap_override_official_check(header_block(4, gap_override_m=1.4), purpose)

    def test_realized_exposure_table(self):
        AE = self.AE
        stamp, _ = triggered_stamp(dose=2)
        pool = F.reference_pool("path_following", F.BB)
        row, episode = F.first_admitted(pool)
        envelope = F.envelope(F.BB)
        a = pool.assignment(F.key("path_following", row, episode), 2)
        slot = C.appearing_slot(F.plans.STATIC_CAPACITY["path_following"], F.plans.D7_RESERVED_SLOTS["path_following"])
        bridge = RT.AppearingBridge("path_following", slot, PHYSICS_DT, envelope.top_speed_mps, envelope.footprint,
                                    envelope.t_avoid_s)
        bridge.reset_rows((row,), {row: a})
        x, y, _ = a.cylinder
        bridge.step({row: 0}, {row: F.dynamic.Pose(x - 50.0, y, 0.0, 0.0)}, check_trigger=True)
        missed = bridge.finish(row, "timeout")                               # assigned, never triggered
        self.assertFalse(missed["payload"]["runtime"]["exposure_realized"])
        doc = {"header": {"controller": {"name": "los_pid_avoid_v1"}, "appearing_hazards_header": header_block(2)},
               "records": [{"appearing_hazards": stamp}, {"appearing_hazards": missed}, {"appearing_hazards": stamp}]}
        rows = AE.realized_exposure_table([doc])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["task"], row["vessel_id"], row["controller"], row["dose"]),
                         ("path_following", F.BB, "los_pid_avoid_v1", 2))
        self.assertEqual((row["assigned"], row["realized"], row["not_admitted"]), (3, 2, 0))
        self.assertAlmostEqual(row["share"], 2.0 / 3.0)
        self.assertEqual(row["wilson_lower_90"], AE.wilson_lower_bound(2, 3))
        self.assertEqual(row["trigger_gap_m"], C.trigger_gap_m(2, envelope.t_avoid_s, F.BB))
        self.assertAlmostEqual(AE.wilson_lower_bound(32, 32), 32.0 / (32.0 + AE.WILSON_Z_ONE_SIDED_90 ** 2),
                               places=12)
        self.assertIsNone(AE.wilson_lower_bound(0, 0))


class NominalReplayEveryTask(unittest.TestCase):
    """Bridge-level trigger mechanics: the nominal plan of every admitted key of a 2 x 1 grid, all tasks."""

    def test_every_task_triggers_within_one_control_step_at_every_dose(self):
        checked = {}
        for task in C.PLACEMENT_TASKS:
            for vessel in (F.BB, F.WAMV):
                pool = F.small_pool(task, vessel, num_envs=2, episodes=1)
                envelope = F.envelope(vessel)
                slot = C.appearing_slot(F.plans.STATIC_CAPACITY[task], F.plans.D7_RESERVED_SLOTS[task])
                for env_i, episode in pool.offered:
                    if not pool.is_admitted(env_i, episode):
                        continue
                    context = F.inputs(task, vessel)(env_i, episode)[0]
                    plan = P.nominal_plan(context, F.admission.admission_suite(vessel))
                    for dose in C.offered_doses(task, vessel):          # G20d: BlueBoat SK D1 is never offered
                        a = pool.assignment(F.key(task, env_i, episode), dose)
                        if a.exposure != "admitted":                    # G20d 5B: docking D5 may lack a witness
                            self.assertIn(dose, C.optional_doses(task, vessel))
                            continue
                        bridge = RT.AppearingBridge(task, slot, PHYSICS_DT, envelope.top_speed_mps,
                                                    envelope.footprint, envelope.t_avoid_s)
                        bridge.reset_rows((env_i,), {env_i: a})
                        tick, last = 0, plan.pose_at(0.0)
                        bridge.step({env_i: 0}, {env_i: last})
                        while not bridge.triggered(env_i) and tick * PHYSICS_DT < plan.end_s:
                            for _ in range(DECIMATION):
                                tick += 1
                                last = plan.pose_at(min(plan.end_s, tick * PHYSICS_DT))
                                bridge.step({env_i: tick}, {env_i: last})
                            bridge.step({env_i: tick}, {env_i: last}, check_trigger=True)
                        with self.subTest(task=task, vessel=vessel, key=(env_i, episode), dose=dose):
                            if a.candidate["nominal_trigger_s"] is None:
                                # the calibration rung D5 need not be reached by the nominal plan (witnessed
                                # with the cylinder present from t = 0): the replay never triggers it
                                self.assertIn(dose, C.SUPPLEMENTARY_DOSES)
                                self.assertFalse(bridge.triggered(env_i))
                                continue
                            self.assertTrue(bridge.triggered(env_i))
                            trig = bridge.rows[env_i]["trigger"]
                            d_k = a.trigger_gap_m
                            self.assertLessEqual(trig["gap_m"], d_k)
                            if trig["previous_check_gap_m"] is not None:
                                self.assertGreater(trig["previous_check_gap_m"], d_k)
                            self.assertGreater(trig["gap_m"], d_k - envelope.max_speed_mps * CONTROL_DT - 1e-6)
                            self.assertAlmostEqual(trig["time_s"], a.candidate["nominal_trigger_s"],
                                                   delta=CONTROL_DT + 1e-9)
                            self.assertGreater(trig["time_s"], 0.0)              # never at step 0
                            self.assertEqual(trig["closing_speed_mps"],
                                             trig["pose"][3] * math.cos(trig["bearing_rad"]))
                        checked[(task, vessel)] = checked.get((task, vessel), 0) + 1
        missing = [(t, v) for t in C.PLACEMENT_TASKS for v in (F.BB, F.WAMV) if (t, v) not in checked]
        self.assertLessEqual(len(missing), 2, missing)          # a 2-key grid may hold no admitted key rarely


if __name__ == "__main__":
    unittest.main(verbosity=2)
