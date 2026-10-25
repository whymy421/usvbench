"""G35 (controller decision J19, v2.6.1): an appearing trigger outside the vessel's speed envelope is RECORDED in the
stamp, never refused (CPU only; run: python -B test_g35_speed_envelope.py, or pytest).

Regression of G33 (training smoke S3, path following BlueBoat, release f7eec671ce0d-0d729fdf5560): a trigger at a
surge speed of -1.5084264743885187 m/s (BlueBoat envelope [-0.9743162233660108, 3.000326502719915] m/s,
ladder.py:70-71) made read_appearing_stamp raise inside journal_dones -> pairing.appearing_primitive_hash, which the
trainer turned into an infrastructure failure. Covers:
  * the REAL AppearingBridge driven astern at that speed into the cylinder: the stamp carries
    runtime.trigger_speed_outside_envelope = true and runtime.trigger_speed_envelope {speed_mps, envelope_mps, rule};
    read_appearing_stamp, exposure_of and pairing.appearing_primitive_hash accept it, and the appearing primitive
    hash equals the one of an in-envelope run of the same key and dose (pairing is unaffected);
  * in-envelope stamps (ahead, astern inside the reverse bound, the tolerance edge) carry neither field;
  * WAM-V astern beyond its reverse bound is flagged likewise; above top speed is flagged;
  * forgeries stay rejected: an out-of-envelope speed without the flag (the v2.6 T11 case), a flag on an
    in-envelope stamp, a record that does not recompute, a lone flag or a lone record;
  * scripts/appearing_exposure.py trigger_speed_summary / trigger_speed_table count the flagged episodes.
"""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g3_fixtures as F  # noqa: E402

C, ST, RT, PAIRING = F.config, F.stamp, F.runtime, F.pairing
PHYSICS_DT, DECIMATION = 1.0 / 120.0, 2
CONTROL_DT = PHYSICS_DT * DECIMATION
G33_SPEED = -1.5084264743885187          # G33 S3 STATUS.json / S3.a4be12f58c.log (BlueBoat, path following)


def exposure_module():
    spec = importlib.util.spec_from_file_location("_g35_appearing_exposure", F.REPO / "scripts" / "appearing_exposure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reseal(envelope):
    payload = envelope["payload"]
    return {"payload": payload, "sha256": ST.digest(payload)}


def run_stamp(speed, dose=1, task="path_following", vessel=F.BB, heading=0.3):
    """A REAL bridge row driven in a straight line at ``speed`` (signed surge) until it triggers. Astern
    (speed < 0) the cylinder lies behind the boat: start = cylinder + distance x heading direction."""
    pool = F.reference_pool(task, vessel)
    row, episode = F.first_admitted(pool)
    envelope = F.envelope(vessel)
    a = pool.assignment(F.key(task, row, episode), dose)
    x, y, _ = a.cylinder
    slot = C.appearing_slot(F.plans.STATIC_CAPACITY[task], F.plans.D7_RESERVED_SLOTS[task])
    bridge = RT.AppearingBridge(task, slot, PHYSICS_DT, envelope.top_speed_mps, envelope.footprint,
                                envelope.t_avoid_s)
    bridge.reset_rows((row,), {row: a})
    distance = 40.0 * envelope.length_m
    sign = 1.0 if speed >= 0 else -1.0
    start = (x - sign * distance * math.cos(heading), y - sign * distance * math.sin(heading))
    steps = int(4 * distance / max(abs(speed), 0.1) / CONTROL_DT) + 10
    F.straight_run(bridge, row, start, heading, speed, PHYSICS_DT, DECIMATION, steps)
    assert bridge.triggered(row), "the row never reached d_k"
    return bridge.finish(row, "collision")


class OutsideEnvelopeIsRecorded(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AE = exposure_module()
        cls.flagged = run_stamp(G33_SPEED)
        cls.inside = run_stamp(1.0)

    def test_g33_case_is_recorded_not_refused(self):
        runtime = self.flagged["payload"]["runtime"]
        self.assertEqual(runtime["trigger"]["pose"][3], G33_SPEED)
        self.assertIs(runtime[ST.SPEED_ENVELOPE_FLAG], True)
        record = runtime[ST.SPEED_ENVELOPE_RECORD]
        self.assertEqual(record["speed_mps"], G33_SPEED)
        self.assertEqual(record["envelope_mps"], [-0.9743162233660108, 3.000326502719915])
        self.assertEqual(record["rule"], ST.SPEED_ENVELOPE_RULE)
        self.assertEqual(ST.read_appearing_stamp(self.flagged), self.flagged)          # the v2.6 raise site
        self.assertEqual(ST.exposure_of(self.flagged), (ST.ADMITTED, ()))
        flagged_hash = PAIRING.appearing_primitive_hash(self.flagged)                   # journal_dones path
        self.assertEqual(flagged_hash, PAIRING.appearing_primitive_hash(self.inside))   # same key and dose
        self.assertEqual(self.AE.record_errors({"records": [{"appearing_hazards": self.flagged}]}), [])

    def test_in_envelope_stamps_carry_no_field(self):
        back = ST.LADDER.VESSEL_MAX_REVERSE_SPEED_MPS[F.BB]
        for speed in (1.0, -0.5, -(back + 0.5e-6)):            # ahead; astern inside; inside the 1e-6 tolerance
            stamp = run_stamp(speed)
            runtime = stamp["payload"]["runtime"]
            self.assertEqual(runtime["trigger"]["pose"][3], speed)
            self.assertNotIn(ST.SPEED_ENVELOPE_FLAG, runtime, speed)
            self.assertNotIn(ST.SPEED_ENVELOPE_RECORD, runtime, speed)
            self.assertEqual(ST.read_appearing_stamp(stamp), stamp)
            self.assertEqual(ST.speed_envelope_errors(stamp["payload"]), [])

    def test_wamv_astern_and_overspeed_are_flagged(self):
        back = ST.LADDER.VESSEL_MAX_REVERSE_SPEED_MPS[F.WAMV]
        stamp = run_stamp(-(back + 0.2), vessel=F.WAMV)
        self.assertIs(stamp["payload"]["runtime"][ST.SPEED_ENVELOPE_FLAG], True)
        self.assertEqual(ST.read_appearing_stamp(stamp), stamp)
        top = ST.LADDER.VESSEL_TOP_SPEED_MPS[F.BB]
        self.assertTrue(ST.speed_outside_envelope(F.BB, top + 2e-6))
        self.assertFalse(ST.speed_outside_envelope(F.BB, top + 0.5e-6))
        self.assertFalse(ST.speed_outside_envelope("no_such_vessel", 99.0))

    def test_forgeries_stay_rejected(self):
        def rejected(envelope, why):
            with self.assertRaises(ValueError) as caught:
                ST.read_appearing_stamp(envelope)
            self.assertIn(why, str(caught.exception))
            self.assertTrue(self.AE.record_errors({"records": [{"appearing_hazards": envelope}]}))

        bad = copy.deepcopy(self.flagged)                       # out of the envelope, flag stripped (v2.6 T11)
        del bad["payload"]["runtime"][ST.SPEED_ENVELOPE_FLAG]
        del bad["payload"]["runtime"][ST.SPEED_ENVELOPE_RECORD]
        rejected(reseal(bad), "envelope")
        bad = copy.deepcopy(self.flagged)                       # lone record
        del bad["payload"]["runtime"][ST.SPEED_ENVELOPE_FLAG]
        rejected(reseal(bad), "not recorded")
        bad = copy.deepcopy(self.flagged)                       # flag false
        bad["payload"]["runtime"][ST.SPEED_ENVELOPE_FLAG] = False
        rejected(reseal(bad), "not recorded")
        bad = copy.deepcopy(self.flagged)                       # record does not recompute
        bad["payload"]["runtime"][ST.SPEED_ENVELOPE_RECORD]["speed_mps"] = -1.2
        rejected(reseal(bad), "not recorded")
        bad = copy.deepcopy(self.flagged)
        bad["payload"]["runtime"][ST.SPEED_ENVELOPE_RECORD]["envelope_mps"] = [-2.0, 3.0]
        rejected(reseal(bad), "not recorded")
        bad = copy.deepcopy(self.inside)                        # flag on an in-envelope stamp
        bad["payload"]["runtime"].update(ST.speed_envelope_fields(ST.ADMITTED, {"vessel_id": F.BB},
                                                                  {"trigger": {"pose": [0.0, 0.0, 0.0, -2.0]}}))
        rejected(reseal(bad), "inside")
        bad = copy.deepcopy(self.flagged)                       # not re-sealed: the digest
        bad["payload"]["runtime"][ST.SPEED_ENVELOPE_FLAG] = False
        rejected(bad, "digest")
        self.assertIs(self.flagged["payload"]["runtime"][ST.SPEED_ENVELOPE_FLAG], True)   # fixture untouched

    def test_summary_counts_the_flagged_episodes(self):
        document = {"records": [{"env": 0, "ep": 0, "appearing_hazards": self.flagged},
                                {"env": 1, "ep": 0, "appearing_hazards": self.inside}, {"env": 2, "ep": 0}]}
        out = self.AE.trigger_speed_summary(document)
        self.assertEqual((out["records_with_stamp"], out["admitted"], out["realized"], out["outside_envelope"]),
                         (2, 2, 2, 1))
        self.assertEqual(out["episodes"], [{"index": 0, "env": 0, "ep": 0, "speed_mps": G33_SPEED,
                                            "envelope_mps": [-0.9743162233660108, 3.000326502719915]}])
        block = {"suite": {"vessel_id": F.BB}, "hazard_group": C.hazard_group("path_following"), "dose": 1}
        document["header"] = {"controller": {"name": "ppo"}, "appearing_hazards_header": block}
        rows = self.AE.trigger_speed_table([document])
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["realized"], rows[0]["outside_envelope"], rows[0]["share_of_realized"]),
                         (2, 1, 0.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
