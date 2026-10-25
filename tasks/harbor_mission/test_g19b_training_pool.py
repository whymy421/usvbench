"""G19b (IMPLEMENTATION_PLAN P3, F6): training-only online-admission D7 pool of the journal harbour mission.

CPU only. (a) non-component keys never call the admission function and serve the D24 NotAdmitted record;
(b) a component key's served D1 assignment has the digests of an OFFLINE admission of that key
(admit_harbour_grid, joint, and the non-joint training rule); (c) a forced D1 rejection gives the D16 record with
the layout geometry and stage; (d) evaluation seeds are refused; (e) prefetch on / off gives identical served
objects; (f) the REAL HarborMissionEnv (harbour fixtures stub) with the all-D1 stack + mixture + adapter resets
through every mixture component (its geometry check on the D7 row identity passes) and steps; (g) header fields.
Run: python -B tasks/harbor_mission/test_g19b_training_pool.py
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import harbour_journal_fixtures as F  # noqa: E402

F.install_isaaclab_stubs()
HJ = F.load("harbor_mission.harbor_journal")
P = F.load("harbor_mission.harbor_d7_pool")
TP = F.load("harbor_mission.training_pool")
TH = F.load("_shared.training_hazards")
TM = F.load("_shared.conditions.training_mixture")
AC = F.load("_shared.conditions.appearing.config")
D = F.load("_shared.conditions.dynamic")
SEED, VESSEL, TASK = 90001, "blueboat_v2", "harbour_mission"


def mixture(seed=SEED, purpose="training_smoke"):
    return TM.TrainingMixture(seed=seed, task=TASK, vessel=VESSEL, appearing_wired=TASK in AC.WIRED_TASKS,
                              purpose=purpose)


def pool(m=None, cache=None, admit_grid=None, num_envs=6):
    m = mixture() if m is None else m
    return TP.TrainingHarbourD7Pool(mixture=m, task=TASK, vessel=VESSEL, seed=m.seed, tier=2, num_envs=num_envs,
                                    suite=D.DynamicSuiteConfig(enabled=True, dose=1, vessel_id=VESSEL),
                                    cache=cache or TH.TrainingHazardCache(), purpose=m.purpose, admit_grid=admit_grid)


def key(env, episode, seed=SEED):
    return D.EpisodeKey(seed, env, episode, P.ADMITTED_GROUP)


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
        for env, episode in keys(p.gate.mixture, False):                                    # (a)
            got = p.assignment(key(env, episode), 1)
            self.assertIsInstance(got, P.NotAdmitted)
            self.assertTrue(got.beyond_grid)
            self.assertIsNone(got.geometry_id)
        self.assertEqual(counter.calls, 0)
        got = p.assignment(D.EpisodeKey(SEED, 0, 0, P.MIXIN_GROUP), 1)                     # pre-v2 group alias
        self.assertEqual(got.key.hazard_group if hasattr(got, "key") else P.ADMITTED_GROUP, P.ADMITTED_GROUP)
        env, episode = keys(p.gate.mixture, True)[0]                                        # (b)
        k = key(env, episode)
        served = p.assignment(k, 1)
        suite = D.DynamicSuiteConfig(enabled=True, vessel_id=VESSEL)
        offline, info = HJ.admit_harbour_grid(D, VESSEL, 2, SEED, [(env, episode)], suite, F.envelope(VESSEL))
        layout = HJ.journal_layout(VESSEL, 2, SEED, env, episode)
        if isinstance(served, P.NotAdmitted):
            self.assertEqual(served.geometry_id, layout.geometry_hash)
        else:
            self.assertEqual(served.candidate.geometry_id, layout.geometry_hash)
            if (k, 1) in offline.assignments:
                self.assertEqual(served.candidate_digest, offline.assignments[(k, 1)].candidate_digest)
                self.assertEqual(served.witness_digest, offline.assignments[(k, 1)].witness_digest)

        def forced(jobs, joint=False, workers=1):                                          # (c)
            real = D.admit_grid(jobs, joint=joint, workers=workers)
            kk = jobs[0][1]
            return dataclasses.replace(
                real, assignments={kd: a for kd, a in real.assignments.items() if kd != (kk, 1)},
                rejections=tuple(real.rejections) + (D.Rejection(kk, 1, "forced_test_rejection", 0),))
        got = pool(admit_grid=forced).assignment(k, 1)
        self.assertIsInstance(got, P.NotAdmitted)
        self.assertFalse(got.beyond_grid)
        self.assertEqual(got.geometry_id, layout.geometry_hash)
        self.assertEqual(got.stage, info["stage"][k])
        self.assertIn("forced_test_rejection", got.failing_reasons())
        for seed in (11001, 21001, 31001, 41041):                                          # (d)
            with self.assertRaises(TH.TrainingHazardError):
                pool(m=mixture(seed=seed))
        outs = []                                                                          # (e)
        for prefetch in (False, True):
            cache = TH.TrainingHazardCache(prefetch=prefetch)
            q = pool(cache=cache, num_envs=2)
            outs.append([getattr(q.assignment(key(e, j), 1), "candidate_digest", None)
                         for e in range(2) for j in range(10)])
            cache.close()
        self.assertEqual(outs[0], outs[1])


class StubEnvResets(unittest.TestCase):
    N, RESETS = 6, 14

    def test_f_g_reset_path_through_every_component(self):
        m = mixture()
        cache = TH.TrainingHazardCache()
        cfg = F.journal_cfg(VESSEL, tuple(m.required_tokens()), num_envs=self.N, seed=SEED, pool=False,
                            journal_training_mixture=m)
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
        pools["d7"].assignment = lambda k, dose: served.append((k, original(k, dose))) or served[-1][1]
        spec = F.vehicles_v2.get_vehicle_v2(VESSEL)
        F.FakeRigidObject.MASS = spec.mass_kg
        F.FakeRigidObject.INERTIA = spec.inertia_com_kg_m2
        env = F.env_module().HarborMissionEnv(cfg)
        for field in ("grid", "per_dose_counts", "keys_not_admitted", "pool_digest", "keys_offered", "joint"):
            self.assertIn(field, env.dynamic_hazard_header)
        layer = env._conditions
        self.assertIs(layer.training_mixture, m)
        seen = set()
        ids = torch.arange(self.N)
        for r in range(self.RESETS):
            del served[:]
            env._reset_idx(ids)
            by_env = {int(k.env): a for k, a in served}
            self.assertEqual(sorted(by_env), list(range(self.N)))
            for i in range(self.N):
                comp = layer.training_component(i)
                self.assertEqual(comp, m.component(i, int(layer.episode_index[i])))
                seen.add(comp)
                beyond = isinstance(by_env[i], P.NotAdmitted) and by_env[i].beyond_grid
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
        print("G19b F6 STUB ENV: %d resets x %d envs, components %d, D7 admitted %d / not %d"
              % (self.RESETS, self.N, len(seen), log["admitted"], log["not_admitted"]), flush=True)


def _code9_support(levels):
    scripts = Path(__file__).resolve().parents[levels] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import g19b_training_test_support as GS
    return GS


class Code9RealEnv(unittest.TestCase):
    """Follow-up C1 (harbour mission)."""

    def test_code9_reward_is_replaced_in_the_real_env(self):
        GS = _code9_support(2)
        cfg = F.journal_cfg(VESSEL, (), num_envs=2, seed=SEED, pool=False)
        pools = TP.training_pools(cfg, None, cache=TH.TrainingHazardCache(), purpose="training_smoke")
        cfg.journal_d7_pool = pools["d7"]
        spec = F.vehicles_v2.get_vehicle_v2(VESSEL)
        F.FakeRigidObject.MASS = spec.mass_kg
        F.FakeRigidObject.INERTIA = spec.inertia_com_kg_m2
        monitor, steps = GS.code9_run(F.env_module().HarborMissionEnv(cfg), "harbor_mission")
        GS.check_code9(self, monitor, steps, -25.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
