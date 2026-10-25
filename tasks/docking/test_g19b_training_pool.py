"""G19b (IMPLEMENTATION_PLAN P3, F5): training-only online-admission D7 pool of journal docking. CPU only.

Uses the REAL DockingEnv harness of test_journal_docking_env.py (its stubbed Isaac Lab, toy integrator).
(a) non-component keys never call the admission function and serve the D24 object (beyond_grid True, the env's
geometry check skipped); (b) a component key's served D1 assignment has the digests of an OFFLINE admission of
that key (the harness's joint admission routine, and the non-joint training rule); (c) a forced D1 rejection gives
the D16 object with the scenario geometry; (d) evaluation seeds are refused; (e) prefetch on / off gives identical
served objects; (f) the REAL DockingEnv with the all-D1 stack + mixture + adapter resets through every mixture
component and steps; (g) the header fields the env and mixin read.
Run: python -B tasks/docking/test_g19b_training_pool.py
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import test_journal_docking_env as H  # noqa: E402  (installs the stubs; its tests are not collected here)

import importlib  # noqa: E402
TP = importlib.import_module("tasks.docking.training_pool")
TH = importlib.import_module("tasks._shared.training_hazards")
TM = importlib.import_module("tasks._shared.conditions.training_mixture")
AC = importlib.import_module("tasks._shared.conditions.appearing.config")
P, JD, D, JC = H.pool_io, H.jd, H.dynamic, H.JC
SEED, VESSEL, TASK = 90001, "blueboat_v2", "docking"


def mixture(seed=SEED, purpose="training_smoke"):
    return TM.TrainingMixture(seed=seed, task=TASK, vessel=VESSEL, appearing_wired=TASK in AC.WIRED_TASKS,
                              purpose=purpose)


def pool(m=None, cache=None, admit_grid=None, num_envs=6):
    m = mixture() if m is None else m
    return TP.TrainingDockingD7Pool(mixture=m, task=TASK, vessel=VESSEL, seed=m.seed, tier=2, num_envs=num_envs,
                                    suite=D.DynamicSuiteConfig(enabled=True, dose=1, vessel_id=VESSEL),
                                    cache=cache or TH.TrainingHazardCache(), purpose=m.purpose, admit_grid=admit_grid)


def key(env, episode, seed=SEED):
    return D.EpisodeKey(seed, env, episode, P.HAZARD_GROUP)


def keys(m, want, n_envs=6, episodes=12):
    return [(e, k) for e in range(n_envs) for k in range(episodes) if (m.hazard_dose("dynamic", e, k) == 1) == want]


class Counter:
    def __init__(self, function):
        self.function, self.calls = function, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.function(*args, **kwargs)


class Adapter(unittest.TestCase):
    def test_a_b_c_d_e(self):
        counter = Counter(D.admit_grid)
        p = pool(admit_grid=counter)
        self.assertTrue(P.suite_matches(p, D.DynamicSuiteConfig(enabled=True, dose=1, vessel_id=VESSEL)))
        for env, episode in keys(p.gate.mixture, False):                                    # (a)
            self.assertTrue(p.beyond_grid(env, episode, SEED))
            got = p.assignment(key(env, episode), 1)
            self.assertIsInstance(got, P.NotAdmittedAssignment)
            self.assertTrue(got.beyond_grid)
            self.assertEqual(got.geometry_id, JD.build_scenario(VESSEL, 2, SEED, env, episode, True).geometry_id)
        self.assertEqual(counter.calls, 0)
        env, episode = keys(p.gate.mixture, True)[0]                                        # (b)
        k = key(env, episode)
        self.assertFalse(p.beyond_grid(env, episode, SEED))
        scenario = JD.build_scenario(VESSEL, 2, SEED, env, episode, True)
        self.assertEqual(p.geometry_id(env, episode), scenario.geometry_id)
        served = p.assignment(k, 1)
        suite = D.DynamicSuiteConfig(enabled=True, vessel_id=VESSEL)
        ladder = D.build_ladder(JD.d7_context(D, scenario), k, suite)
        for reference_pool in (D.admit_ladders([(k, ladder)], joint=True), D.admit_ladders([(k, ladder)], joint=False)):
            if (k, 1) in reference_pool.assignments and not isinstance(served, P.NotAdmittedAssignment):
                self.assertEqual(served.candidate_digest, reference_pool.assignments[(k, 1)].candidate_digest)
                self.assertEqual(served.witness_digest, reference_pool.assignments[(k, 1)].witness_digest)

        def forced(jobs, joint=False, workers=1):                                          # (c)
            real = D.admit_grid(jobs, joint=joint, workers=workers)
            kk = jobs[0][1]
            return dataclasses.replace(
                real, assignments={kd: a for kd, a in real.assignments.items() if kd != (kk, 1)},
                rejections=tuple(real.rejections) + (D.Rejection(kk, 1, "forced_test_rejection", 0),))
        q = pool(admit_grid=forced)
        got = q.assignment(k, 1)
        self.assertIsInstance(got, P.NotAdmittedAssignment)
        self.assertFalse(got.beyond_grid)
        self.assertEqual(got.geometry_id, scenario.geometry_id)
        self.assertEqual(q.exposure(env, episode), P.EXPOSURE_NOT_ADMITTED)
        self.assertIn("forced_test_rejection", [r for _, r, _ in got.reasons])
        for seed in (11001, 21001, 31001, 41041):                                          # (d)
            with self.assertRaises(TH.TrainingHazardError):
                pool(m=mixture(seed=seed))
        outs = []                                                                          # (e)
        for prefetch in (False, True):
            cache = TH.TrainingHazardCache(prefetch=prefetch)
            r = pool(cache=cache, num_envs=2)
            outs.append([getattr(r.assignment(key(e, j), 1), "candidate_digest", None)
                         for e in range(2) for j in range(10)])
            cache.close()
        self.assertEqual(outs[0], outs[1])


class StubEnvResets(unittest.TestCase):
    N, RESETS = 6, 14

    def test_f_g_reset_path_through_every_component(self):
        m = mixture()
        cache = TH.TrainingHazardCache()
        cfg = H.CFG_CLASS[VESSEL]()
        cfg.berth_walls = True
        cfg.sim.gravity = (0.0, 0.0, -9.81)
        cfg.scene.num_envs = self.N
        cfg.visual.enable_water = False
        JC.apply_journal_conditions(cfg, JC.resolve_journal_conditions(presets=m.required_tokens(), seed=SEED))
        cfg.journal_training_mixture = m
        pools = TP.training_pools(cfg, m, cache=cache, purpose="training_smoke")
        cfg.journal_d7_pool = pools["d7"]
        appearing_served = []
        if pools["appearing"] is not None:                       # G20b wiring: the shared appearing adapter
            cfg.journal_appearing_pool = pools["appearing"]
            original_a = pools["appearing"].assignment
            pools["appearing"].assignment = (lambda key, dose: appearing_served.append((key, original_a(key, dose)))
                                             or appearing_served[-1][1])
        served = []
        original = pools["d7"].assignment
        pools["d7"].assignment = lambda kk, dose: served.append((kk, original(kk, dose))) or served[-1][1]
        env = H.docking_env.DockingEnv(cfg)
        for field in ("per_dose_counts", "pool_digest", "keys_offered", "joint"):
            self.assertIn(field, env.dynamic_hazard_header)
        layer = env._conditions
        self.assertIs(layer.training_mixture, m)
        seen = set()
        ids = torch.arange(self.N)
        for r in range(self.RESETS):
            del served[:]
            env._reset_idx(ids)
            by_env = {int(kk.env): a for kk, a in served}
            self.assertEqual(sorted(by_env), list(range(self.N)))
            for i in range(self.N):
                comp = layer.training_component(i)
                self.assertEqual(comp, m.component(i, int(layer.episode_index[i])))
                seen.add(comp)
                beyond = isinstance(by_env[i], P.NotAdmittedAssignment) and by_env[i].beyond_grid
                self.assertEqual(beyond, comp != "dynamic", (r, i))
        self.assertEqual(seen, set(TM.COMPONENTS) - ({"appearing"} if not m.appearing_wired else set()))
        if pools["appearing"] is not None:
            self.assertTrue(appearing_served)
            for key, got in appearing_served:
                self.assertEqual(bool(got.beyond_offered_grid), m.component(key.env, key.episode) != "appearing", key)
            self.assertTrue(any(not got.beyond_offered_grid for _, got in appearing_served))
        env._get_observations()          # what DirectRLEnv.reset does after _reset_idx (one estimate per group)
        env.step(torch.zeros((self.N, 2)))
        log = cache.admission_log()["suites"]["dynamic"]
        print("G19b F5 STUB ENV: %d resets x %d envs, components %d, D7 admitted %d / not %d"
              % (self.RESETS, self.N, len(seen), log["admitted"], log["not_admitted"]), flush=True)


def _code9_support(levels):
    scripts = Path(__file__).resolve().parents[levels] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import g19b_training_test_support as GS
    return GS


class Code9RealEnv(unittest.TestCase):
    """Follow-up C1 (docking)."""

    def test_code9_reward_is_replaced_in_the_real_env(self):
        GS = _code9_support(2)
        cfg = H.CFG_CLASS[VESSEL]()
        cfg.berth_walls = True
        cfg.sim.gravity = (0.0, 0.0, -9.81)
        cfg.scene.num_envs = 2
        cfg.visual.enable_water = False
        JC.apply_journal_conditions(cfg, JC.resolve_journal_conditions(presets=[], seed=SEED))
        pools = TP.training_pools(cfg, None, cache=TH.TrainingHazardCache(), purpose="training_smoke")
        cfg.journal_d7_pool = pools["d7"]
        monitor, steps = GS.code9_run(H.docking_env.DockingEnv(cfg), "docking")
        GS.check_code9(self, monitor, steps, -25.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
