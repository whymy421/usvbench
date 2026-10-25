"""F4: recorded env traces replayed through the SCALAR oracle (bit for bit), CPU only.

Run: python -B tasks/station_keeping/tests_journal/test_station_journal_trace.py

The REAL station_keeping_env.py journal branch (stubbed isaaclab, toy
integrator, REAL runtime / condition layer / D7 bridge / committed pool) runs
full BlueBoat episodes to their layout deadline with a simple controller on
the estimated state; every oracle input the env fed its tensor twin
(journal_contract_trace: reset, observe per physics substep, step per control
step) is replayed through journal_contracts_v2.StationKeepingEpisode. The
per-step codes and every completed EpisodeResult.evaluator_fields() must be
identical bit for bit. Also: the episode ends exactly at the layout deadline
(timeout code 3, fixed horizon, success decided only at T), the D7 bridge
swept every physics interval once, and a WAM-V + suites trace replays too.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import station_journal_fixtures as F

RESTORE = None


def setUpModule():
    global RESTORE, dyn
    RESTORE = F.install_isaaclab_stubs()
    dyn = F.ns("_shared.conditions.dynamic")


def tearDownModule():
    if RESTORE is not None:
        RESTORE()


def run(env, steps=None, until_all_completed=True):
    env.reset()
    done_once = torch.zeros(env.num_envs, dtype=torch.bool)
    k = 0
    while True:
        _, _, terminated, truncated, _ = env.step(F.hold_controller(env))
        done_once |= terminated | truncated
        k += 1
        if steps is not None and k >= steps:
            break
        if steps is None and until_all_completed and bool(done_once.all()):
            break
    return k


class TraceParityTests(unittest.TestCase):
    def check(self, env):
        trace = env.journal_contract_trace
        codes, completed = F.replay_scalar(trace, env.cfg.journal_plan.validity_limits)
        self.assertEqual(codes, F.trace_codes(trace))
        twin_completed = F.trace_completed(trace)
        self.assertEqual([i for i, _ in completed], [i for i, _ in twin_completed])
        for (i, scalar), (_, twin) in zip(completed, twin_completed):
            F.same(twin, scalar, "env%d" % i)
        return codes, completed

    def test_blueboat_full_episodes_replay_bitwise(self):
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2), trace=True)
        steps = run(env)
        codes, completed = self.check(env)
        self.assertEqual(len(completed), 2)
        for i, result in completed:
            self.assertEqual(result["termination_reason"], 3)            # fixed horizon: timeout at T
            self.assertEqual(result["deadline_s"], result["station_late_interval_s"][1])
        # the episode ended exactly at the scenario's layout deadline (time index = deadline steps)
        stops = {}
        for row_index, row in enumerate(codes):
            for i, c in enumerate(row):
                if c and i not in stops:
                    stops[i] = row_index + 1
        first_layout_steps = [int(s) for s in F.trace_reset_steps(env.journal_contract_trace)]
        self.assertEqual(sorted(stops.values()), sorted(first_layout_steps))
        # hold controller succeeded at least once, with time_to_success = the deadline T (decided at T)
        successes = [r for _, r in completed if r["episode_success"]]
        self.assertTrue(successes, [r["station_final_window_rms_error_m"] for _, r in completed])
        for r in successes:
            self.assertEqual(r["time_to_success"], r["deadline_s"])
        # D7: every physics interval swept exactly once; stamps verify
        for i in range(2):
            record = env.journal_episode_record(i)
            envelope = dyn.read_stamp(record["dynamic_hazards"])
            bridge = envelope["payload"]["runtime"]["bridge"]
            self.assertEqual(bridge["swept_intervals"], bridge["last_tick"])
            self.assertEqual(envelope["payload"]["runtime"]["termination_reason"], "timeout")
        self.assertGreater(steps, 6000)

    def test_blueboat_dynamic_d4_full_episodes_replay_bitwise(self):
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, tokens=("dynamic=D4",)), trace=True)
        seen_rays = []
        bound = env._get_observations

        def observe():
            out = bound()
            seen_rays.append(bool((env.journal_estimate.rays < env.cfg.ray_max_range_m).any()))
            return out
        env._get_observations = observe
        run(env)
        _, completed = self.check(env)
        self.assertGreaterEqual(len(completed), 2)          # a collision may end an episode early
        last = {i: r for i, r in completed}                 # latest completion per env = its record
        codes = sorted({r["termination_reason"] for _, r in completed})
        self.assertTrue(set(codes) <= {2, 3}, codes)
        detected = 0
        for i in range(2):
            runtime = dyn.read_stamp(env.journal_episode_record(i)["dynamic_hazards"])["payload"]["runtime"]
            self.assertEqual(runtime["bridge"]["swept_intervals"], runtime["bridge"]["last_tick"])
            self.assertEqual(runtime["termination_reason"], {2: "collision", 3: "timeout"}[
                last[i]["termination_reason"]])
            if runtime["first_detection_s"] is not None:
                detected += 1
                # G23e: the intruder meets the own ship on the approach: revealed (and detected) before the
                # scoring window T - hold (the retired design revealed it at the settled clock, D7 S4)
                self.assertLess(runtime["first_detection_s"], last[i]["station_late_interval_s"][0])
                self.assertEqual(runtime["release"]["plan_kind"], "approach")
        self.assertGreater(detected, 0)
        self.assertTrue(any(seen_rays))              # the intruder reached the 36-ray observation block

    def test_wamv_with_suites_partial_trace_replays_bitwise(self):
        env = F.make_env(F.journal_cfg("wamv_vrx_v2_opt1", num_envs=2,
                                       tokens=("current=D3", "wind=D2", "perception=D2", "dynamic=D3")),
                         trace=True)
        run(env, steps=300)
        self.check(env)


if __name__ == "__main__":
    unittest.main(verbosity=2)
