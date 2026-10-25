"""G7 item B on REAL gate-task keys (CPU; ring, fortress, forced crossing, iceberg; both vessels; seed 42).

Per (task, vessel), for one key the pre-G7 pool rejected ONLY for "no_unforced_d1_window_before_site_end"
and one key it admitted (key lists from G5's diagnostic, DESIGN_GATE_LEADIN.md section 3):
* window-rejected key: the exact-zero rule (pre-G7) still rejects it, every command in its window is a
  nonzero sub-tolerance pursuit correction, and under the G7 predicate every dose D0..D4 builds;
* admitted key: its candidates are bit-identical under the pre-G7 and the G7 rule at D0..D4;
* real manoeuvre: a window placed over the key's start turn is still rejected under the G7 predicate.
The full 422-key evidence is the G7 probe (work/evidence/probe_ttc_*.json) and the re-admitted pools.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
NS = "_g7gate"
if NS not in sys.modules:
    _pkg = ModuleType(NS)
    _pkg.__path__ = [str(REPO / "tasks")]
    sys.modules[NS] = _pkg
    for _folder in (REPO / "tasks").iterdir():
        if _folder.is_dir() and _folder.name != "__pycache__":
            _sub = ModuleType(NS + "." + _folder.name)
            _sub.__path__ = [str(_folder)]
            sys.modules[NS + "." + _folder.name] = _sub


def imp(name):
    return importlib.import_module(NS + "." + name)


J = imp("hazard_nav.journal_hazard_nav")
dyn = imp("_shared.conditions.dynamic")
adapters = imp("_shared.conditions.dynamic.adapters")
reference = imp("_shared.conditions.dynamic.reference")
model = imp("_shared.conditions.dynamic.model")
WINDOW = "no_unforced_d1_window_before_site_end"
# (task, vessel): (window-rejected key, admitted key) of the pre-G7 pinned pools
KEYS = {
    ("ring", "blueboat_v2"): ((1, 0), (0, 0)), ("fortress", "blueboat_v2"): ((0, 1), (0, 0)),
    ("forced_crossing", "blueboat_v2"): ((0, 1), (0, 0)), ("iceberg", "blueboat_v2"): ((0, 1), (0, 0)),
    ("ring", "wamv_vrx_v2_opt1"): ((3, 0), (0, 0)), ("fortress", "wamv_vrx_v2_opt1"): ((0, 0), (0, 1)),
    ("forced_crossing", "wamv_vrx_v2_opt1"): ((0, 0), (3, 0)), ("iceberg", "wamv_vrx_v2_opt1"): ((0, 0), (1, 1)),
}


def exact_zero(accel, yaw_rate):
    return accel == 0.0 and yaw_rate == 0.0


class PreG7Rule:
    """Context manager: the exact-zero window predicate of suite version d7_ttc_dose_v2."""

    def __enter__(self):
        self.saved = reference.is_unforced_command, adapters.is_unforced_command
        reference.is_unforced_command = adapters.is_unforced_command = exact_zero

    def __exit__(self, *exc):
        reference.is_unforced_command, adapters.is_unforced_command = self.saved


def ladder(task, vessel, env, episode):
    sc = J.build_scenario(task, vessel, 2, 42, env, episode)
    ctx, key = J.d7_context(sc), J.d7_key(sc)
    suite = dyn.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
    out = []
    for dose in range(5):
        try:
            out.append(adapters.build_candidate(ctx, key, dose, suite.at_dose(dose)))
        except dyn.PlacementError as error:
            out.append(error)
    return ctx, suite, out


class TestGateWindow(unittest.TestCase):
    def test_window_rejected_keys_now_build_and_admitted_keys_keep_their_candidates(self):
        for (task, vessel), (rejected, admitted) in KEYS.items():
            with self.subTest(task=task, vessel=vessel, key=rejected):
                with PreG7Rule():
                    _, _, old = ladder(task, vessel, *rejected)
                self.assertTrue(all(isinstance(c, dyn.PlacementError) and c.reason == WINDOW for c in old), old)
                ctx, suite, new = ladder(task, vessel, *rejected)
                self.assertTrue(all(isinstance(c, adapters.Candidate) for c in new), new)
                # the window held only nonzero sub-tolerance commands
                plan = adapters._nominal_plan(adapters.PROFILES[task], ctx, suite)
                enc = new[0].request.encounter
                t_c, start = enc.meta("conflict_time_s"), enc.meta("d1_window_start_s")
                inside = [s for b, s in zip(plan.states, plan.states[1:]) if s.time_s > start and b.time_s < t_c]
                nonzero = [s for s in inside if s.accel_mps2 != 0.0 or s.yaw_rate_rps != 0.0]
                self.assertTrue(nonzero)
                self.assertTrue(all(model.is_unforced_command(s.accel_mps2, s.yaw_rate_rps) for s in inside))
            with self.subTest(task=task, vessel=vessel, key=admitted):
                with PreG7Rule():
                    _, _, old = ladder(task, vessel, *admitted)
                _, _, new = ladder(task, vessel, *admitted)
                self.assertEqual([dyn.digest(c) for c in old], [dyn.digest(c) for c in new])

    def test_a_window_over_the_start_turn_is_still_rejected(self):
        for (task, vessel), (_, admitted) in KEYS.items():
            with self.subTest(task=task, vessel=vessel):
                sc = J.build_scenario(task, vessel, 2, 42, *admitted)
                ctx = J.d7_context(sc)
                suite = dyn.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
                plan = adapters._nominal_plan(adapters.PROFILES[task], ctx, suite)
                window = dyn.DOSE_TTC_RATIOS[1] * ctx.envelope.t_avoid_s
                real = [s.time_s for s in plan.states[1:] if not model.is_unforced_command(s.accel_mps2,
                                                                                           s.yaw_rate_rps)]
                self.assertTrue(real, "the nominal plan starts with a real turn")
                t = max(real[0], window) + 1.0e-3
                with self.assertRaises(dyn.PlacementError) as ctx_err:
                    adapters._fit_conflict(plan, t, window, t, 0.0)
                self.assertEqual(ctx_err.exception.reason, WINDOW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
