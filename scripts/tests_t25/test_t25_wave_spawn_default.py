"""T25 (owner decision T1.35a): journal ids spawn on the filtered wave surface by default (CPU).

* the default is a named constant (wave_spawn.JOURNAL_SPAWN_ON_WAVES_DEFAULT = True) citing the decision;
  cfg.journal_spawn_on_waves = False keeps the calm spawn; a non-bool is refused;
* waves (Hs > 0): a journal env without the attribute writes the reset rows onto the filtered surface (the
  same rows as an explicit True run) and stamps header green_water_header.spawn;
* calm water, wave-free stacks and Hs = 0: the default run is bitwise the explicit False run (applied
  wrenches, observations, dones, poses, records); only an Hs = 0 header carries the spawn stamp;
* real writer + validator: a waves certificate built with the default validates and carries the stamp.

Run: python -B scripts/tests_t25/test_t25_wave_spawn_default.py   (or pytest)
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

SPAWN = G.shared("conditions.wave_spawn")
ACTIONS = torch.tensor([[0.5, 0.3], [0.2, -0.4]])


def run(tokens, flag, vessel="blueboat_v2", steps=40, deadline_s=0.5):
    torch.manual_seed(11)
    extra = {} if flag is None else {"journal_spawn_on_waves": flag}
    cfg = G.journal_cfg(vessel, tokens=tokens, **extra)
    env = G.FakeJournalEnv(cfg, 2, deadline_s=deadline_s)
    writes = []
    original = env.robot.write_root_state_to_sim
    env.robot.write_root_state_to_sim = lambda state, ids: (writes.append(state.clone()), original(state, ids))
    trace = {"obs": [env.reset()], "start": env.robot.data.root_link_pose_w.clone(), "dones": [], "records": []}
    for _ in range(steps):
        obs, terminated, truncated = env.step(ACTIONS)
        trace["obs"].append(obs)
        trace["dones"].append((terminated.clone(), truncated.clone()))
        for i in (terminated | truncated).nonzero().flatten().tolist():
            trace["records"].append(json.loads(json.dumps(env.journal_episode_record(i), sort_keys=True)))
    trace["header"] = json.loads(json.dumps(env.journal_certificate_header(), sort_keys=True))
    trace["writes"] = writes
    return env, trace


def assert_bitwise(case, a, b):
    ea, ta = a
    eb, tb = b
    case.assertEqual(len(ea.robot.applied), len(eb.robot.applied))
    for (fa, qa), (fb, qb) in zip(ea.robot.applied, eb.robot.applied):
        case.assertTrue(torch.equal(fa, fb) and torch.equal(qa, qb))
    for oa, ob in zip(ta["obs"], tb["obs"]):
        case.assertTrue(torch.equal(oa, ob))
    for (x, y), (u, v) in zip(ta["dones"], tb["dones"]):
        case.assertTrue(torch.equal(x, u) and torch.equal(y, v))
    case.assertTrue(torch.equal(ea.robot.data.root_link_pose_w, eb.robot.data.root_link_pose_w))
    case.assertEqual(ta["records"], tb["records"])


class Default(unittest.TestCase):
    def test_named_constant_and_cfg_switch(self):
        self.assertIs(SPAWN.JOURNAL_SPAWN_ON_WAVES_DEFAULT, True)
        self.assertIn("T1.35a", SPAWN.SPAWN_DEFAULT_DECISION)
        self.assertIs(SPAWN.enabled(SimpleNamespace()), True)
        self.assertIs(SPAWN.enabled(SimpleNamespace(journal_spawn_on_waves=False)), False)
        with self.assertRaises(ValueError):
            SPAWN.enabled(SimpleNamespace(journal_spawn_on_waves=1))

    def test_waves_spawn_on_the_surface_by_default(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            default = run(["waves=D4"], None, vessel)
            explicit = run(["waves=D4"], True, vessel)
            off = run(["waves=D4"], False, vessel)
            assert_bitwise(self, default, explicit)
            self.assertTrue(default[1]["writes"] and not off[1]["writes"])
            self.assertTrue(default[0]._journal_spawn_on_waves and not off[0]._journal_spawn_on_waves)
            self.assertFalse(torch.equal(default[1]["start"][:, 2], off[1]["start"][:, 2]))   # heave on the surface
            self.assertEqual(default[1]["header"]["green_water_header"]["spawn"]["version"], SPAWN.SPAWN_VERSION)
            self.assertNotIn("spawn", off[1]["header"]["green_water_header"])

    def test_calm_wave_free_and_zero_hs_are_bitwise_unchanged(self):
        for tokens in ([], ["wind=D2", "current=D1"], ["waves=D0"]):
            default = run(tokens, None)
            off = run(tokens, False)
            assert_bitwise(self, default, off)
            self.assertEqual(default[1]["writes"], [])
            header_d, header_o = default[1]["header"], off[1]["header"]
            if tokens == ["waves=D0"]:
                self.assertEqual(header_d["green_water_header"].pop("spawn")["version"], SPAWN.SPAWN_VERSION)
            self.assertEqual(header_d, header_o)


class PartialFirstReset(unittest.TestCase):
    def test_first_partial_reset_spawns_like_a_full_reset(self):
        """A fresh env reset in two parts (rows 3, 1 then 2, 0) gives the same spawn rows as one full reset, in
        waves and in calm water (wave_source_for_reset needs only the reset rows configured)."""
        for tokens in (["waves=D3"], []):
            cfg_a = G.journal_cfg("wamv_vrx_v2_opt1", tokens=tokens)
            cfg_b = G.journal_cfg("wamv_vrx_v2_opt1", tokens=tokens)
            a = G.FakeJournalEnv(cfg_a, 4)
            b = G.FakeJournalEnv(cfg_b, 4)
            a._reset_idx(torch.tensor([0, 1, 2, 3]))
            b._reset_idx(torch.tensor([3, 1]))
            b._reset_idx(torch.tensor([2, 0]))
            self.assertTrue(torch.equal(a.robot.data.root_link_pose_w, b.robot.data.root_link_pose_w), tokens)
            self.assertTrue(torch.equal(a.robot.data.root_com_vel_w, b.robot.data.root_com_vel_w), tokens)
            self.assertIs(b._conditions.wave_source_for_reset(torch.tensor([0, 1, 2, 3])), b._conditions.wave_source())
        fresh = G.FakeJournalEnv(G.journal_cfg("wamv_vrx_v2_opt1", tokens=["waves=D3"]), 4)
        with self.assertRaises(RuntimeError):
            fresh._conditions.wave_source_for_reset(torch.tensor([0]))      # never configured


class Certificate(unittest.TestCase):
    def test_default_waves_certificate_validates_with_the_stamp(self):
        import test_journal_evaluators as T
        doc = T.journal_document(("waves=D3",))
        self.assertEqual(T._check(doc), [])
        self.assertEqual(doc["header"]["green_water"]["spawn"]["version"], SPAWN.SPAWN_VERSION)
        calm = T.journal_document(("wind=D1",))
        self.assertEqual(T._check(calm), [])
        self.assertNotIn("green_water", calm["header"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
