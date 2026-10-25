"""G28 item 1: the outage start rule through the REAL journal mixin and the family wiring (CPU).

* a journal env with perception.dropout starts the outage at f x (route length / cruise speed), f keyed in
  [0.40, 0.60]; the per-episode condition stamp names the rule, f and T_ref;
* an outage cell whose reset names no start rule is refused (no silent 5-20 s fallback); the critical-phase
  path (station keeping) is unchanged;
* non-outage cells are byte-identical whether or not the route length is passed (observations, applied wrench,
  dones and latched records);
* every family env that has no critical phase passes reference_route_length_m to journal_reset, station
  keeping passes outage_start_s (source check of the six family envs).

Run: python -B scripts/tests_journal_glue/test_g28_outage_wiring.py   (or pytest)
"""
from __future__ import annotations

import ast
import math
import os
import sys
import unittest

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

KR = G.shared("conditions.keyed_rng")
PV = G.shared("conditions.perception_v3")
REPO = os.path.dirname(os.path.dirname(_HERE))
ACTIONS = torch.tensor([[0.6, 0.4], [0.3, -0.2]])


class NoRuleEnv(G.FakeJournalEnv):
    """The fake env WITHOUT the G28 route length (what a family that forgot the wiring would do)."""

    def _reset_idx(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        data = self.robot.data
        for i in ids.tolist():
            origin = self.scene.env_origins[i]
            data.root_link_pose_w[i] = torch.tensor([origin[0], origin[1], G.WATER_Z + self.nominal_height,
                                                     1.0, 0.0, 0.0, 0.0])
            data.root_com_vel_w[i] = torch.tensor([self.spawn_speed, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.episode_counter[i] += 1
            self.episode_length_buf[i] = 0
            self.oracles[i].reset((0.0, 0.0))
        self.journal_reset(ids, torch.full((ids.numel(),), self.axis_rad, dtype=torch.float64),
                           self.episode_counter[ids].clone(), **getattr(self, "reset_kwargs", {}))


def run(env, steps):
    torch.manual_seed(5)
    trace = {"obs": [env.reset()], "dones": [], "applied": []}
    original = env.robot.set_external_force_and_torque

    def capture(forces, torques):
        trace["applied"].append((forces.clone(), torques.clone()))
        return original(forces, torques)
    env.robot.set_external_force_and_torque = capture
    for _ in range(steps):
        obs, terminated, truncated = env.step(ACTIONS)
        trace["obs"].append(obs)
        trace["dones"].append((terminated.clone(), truncated.clone()))
    trace["conditions"] = [dict(c) for c in env.episode_conditions]
    return trace


class MixinRule(unittest.TestCase):
    def test_reach_outage_starts_on_the_route_rule(self):
        cfg = G.journal_cfg("blueboat_v2", tokens=["perception.dropout=D3"])
        env = G.FakeJournalEnv(cfg, 2, deadline_s=0.5)
        env.reset()
        p = env._conditions.perception
        cruise = G.plans.vessel_record("blueboat_v2").cruise_speed_m_s
        t_ref = math.hypot(*env.goal_xy) / cruise
        for i in range(2):
            key = KR.stream_key(cfg.seed, i, int(p.episode[i]), "perception.outage_start")
            u = float(KR.uniforms(np.asarray([np.uint64(key)]), np.zeros(1))[0])
            f = 0.40 + u * (0.60 - 0.40)
            self.assertEqual(int(p.outage_start[i]), int(math.floor(f * t_ref / p.dt)))
            record = p.episode_parameters(i)
            self.assertEqual(record["outage_source"], PV.OUTAGE_ROUTE_SOURCE)
            self.assertEqual(record["outage_reference_duration_s"], t_ref)
        # the latched condition stamp of a finished episode carries the rule
        for _ in range(40):
            env.step(ACTIONS)
        stamp = env.episode_conditions[0]
        perception = stamp["runtime"]["perception"]
        self.assertEqual(perception["outage_source"], "reference_route_fraction")
        self.assertTrue(0.40 <= perception["outage_start_fraction"] <= 0.60)

    def test_outage_cell_without_a_start_rule_is_refused(self):
        cfg = G.journal_cfg("blueboat_v2", tokens=["perception.dropout=D2"])
        env = NoRuleEnv(cfg, 2)
        with self.assertRaisesRegex(ValueError, "no fallback window"):
            env.reset()

    def test_critical_phase_path_is_kept(self):
        cfg = G.journal_cfg("blueboat_v2", tokens=["perception.dropout=D2"])
        env = NoRuleEnv(cfg, 2)
        env.reset_kwargs = {"outage_start_s": torch.full((2,), 12.0, dtype=torch.float64)}
        env.reset()
        p = env._conditions.perception
        self.assertTrue(bool((p.outage_start == int(math.floor(12.0 / p.dt + 1e-9))).all()))
        self.assertEqual(p.episode_parameters(0)["outage_source"], "task_critical_phase")

    def test_non_outage_cells_are_identical_with_and_without_the_route_length(self):
        for vessel, tokens in (("blueboat_v2", []), ("blueboat_v2", ["perception.noise=D2", "perception.latency=D2"]),
                               ("wamv_vrx_v2_opt1", ["waves=D2", "wind=D2"]), ("blueboat_v2", ["perception.bias=D3"])):
            cfg = G.journal_cfg(vessel, tokens=tokens)
            with_rule = run(G.FakeJournalEnv(cfg, 2, deadline_s=0.4), 40)
            cfg = G.journal_cfg(vessel, tokens=tokens)
            without = run(NoRuleEnv(cfg, 2, deadline_s=0.4), 40)
            for a, b in zip(with_rule["obs"], without["obs"]):
                self.assertTrue(torch.equal(a, b), tokens)
            for (fa, ta), (fb, tb) in zip(with_rule["applied"], without["applied"]):
                self.assertTrue(torch.equal(fa, fb) and torch.equal(ta, tb), tokens)
            for (ta, ua), (tb, ub) in zip(with_rule["dones"], without["dones"]):
                self.assertTrue(torch.equal(ta, tb) and torch.equal(ua, ub), tokens)
            self.assertEqual(with_rule["conditions"], without["conditions"], tokens)


class FamilyWiring(unittest.TestCase):
    FAMILIES = {"tasks/hazard_nav/hazard_nav_env.py": "reference_route_length_m",
                "tasks/path_following/journal_path_following.py": "reference_route_length_m",
                "tasks/path_hazard/path_hazard_env.py": "reference_route_length_m",
                "tasks/docking/docking_env.py": "reference_route_length_m",
                "tasks/harbor_mission/harbor_mission_env.py": "reference_route_length_m",
                "tasks/station_keeping/station_keeping_env.py": "outage_start_s"}

    def test_every_family_names_a_start_rule(self):
        for relative, keyword in self.FAMILIES.items():
            with open(os.path.join(REPO, relative), encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "journal_reset"]
            self.assertEqual(len(calls), 1, relative)
            names = {k.arg: k.value for k in calls[0].keywords}
            self.assertIn(keyword, names, relative)
            self.assertFalse(isinstance(names[keyword], ast.Constant) and names[keyword].value is None, relative)


if __name__ == "__main__":
    unittest.main()
