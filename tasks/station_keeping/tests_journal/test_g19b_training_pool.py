"""G19b (IMPLEMENTATION_PLAN P3, F4): training-only online-admission D7 pool of journal station keeping. CPU only.

(a) non-component keys never call the admission function and serve the D24 object (key_status
    beyond_offered_grid, no context bind); (b) a component key's served D1 assignment has the digests of an
    OFFLINE admission of that key (admit_station_grid, joint, and the non-joint training rule); (c) a forced D1
    rejection gives the family's D16 object (kind e4_rejected, the layout geometry); (d) evaluation seeds are
    refused; (e) prefetch on / off gives identical served objects; (f) the REAL StationKeepingEnv (station
    fixtures' stub DirectRLEnv) with the all-D1 stack + mixture + adapter resets through every mixture
    component: key_status / bind_context / journal_reset accept every served object, the layer's component of
    each row is TrainingMixture.component, and the env steps; (g) the header fields the env and mixin read.
Episodes of this task always run to their deadline (thousands of control steps), so (f) drives the reset path
(the place where the pool is used) instead of finishing episodes.
Run: python -B tasks/station_keeping/tests_journal/test_g19b_training_pool.py
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import station_journal_fixtures as F  # noqa: E402

F.install_isaaclab_stubs()
J = F.ns("station_keeping.journal_station")
TP = F.ns("station_keeping.training_pool")
TH = F.ns("_shared.training_hazards")
TM = F.ns("_shared.conditions.training_mixture")
AC = F.ns("_shared.conditions.appearing.config")
D = J._dynamic()
SEED, VESSEL, TASK = 90001, "blueboat_v2", "station_keeping"
LEVEL = TM.training_level(TASK, VESSEL, "dynamic")      # G31 / owner decision J10: BlueBoat SK trains D7 at D2


def mixture(seed=SEED, purpose="training_smoke"):
    return TM.TrainingMixture(seed=seed, task=TASK, vessel=VESSEL, appearing_wired=TASK in AC.WIRED_TASKS,
                              purpose=purpose)


def suite():
    return D.DynamicSuiteConfig(enabled=True, dose=LEVEL, vessel_id=VESSEL)


def pool(m=None, cache=None, admit_grid=None, num_envs=6):
    m = mixture() if m is None else m
    return TP.TrainingStationD7Pool(mixture=m, task=TASK, vessel=VESSEL, seed=m.seed, tier=2, num_envs=num_envs,
                                    suite=suite(), cache=cache or TH.TrainingHazardCache(), purpose=m.purpose,
                                    admit_grid=admit_grid)


def keys(m, want, n_envs=6, episodes=12):
    return [(e, k) for e in range(n_envs) for k in range(episodes)
            if (m.hazard_dose("dynamic", e, k) == LEVEL) == want]


def bind(p, env, episode):
    layout = J.build_station_layout(2, VESSEL, SEED, env, episode)
    p.bind_context(J.episode_key(SEED, env, episode), J.station_task_context(layout, VESSEL))


class Counter:
    def __init__(self, function):
        self.function, self.calls = function, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.function(*args, **kwargs)


class Adapter(unittest.TestCase):
    def test_a_non_component_keys(self):
        counter = Counter(D.admit_grid)
        p = pool(admit_grid=counter)
        for env, episode in keys(p.gate.mixture, False):
            k = J.episode_key(SEED, env, episode)
            self.assertEqual(p.key_status(k), J.BEYOND_OFFERED_GRID)
            got = p.assignment(k, LEVEL)
            self.assertIsInstance(got, J.NotAdmittedAssignment)
            self.assertEqual(got.kind, J.BEYOND_OFFERED_GRID)
        self.assertEqual(counter.calls, 0)
        with self.assertRaises(ValueError):
            p.key_status(J.episode_key(SEED, 6, 0))
        with self.assertRaises(KeyError):
            p.assignment(J.episode_key(42, 0, 0), LEVEL)

    def test_b_offline_digests_and_c_forced_rejection(self):
        p = pool()
        env, episode = keys(p.gate.mixture, True)[0]
        k = J.episode_key(SEED, env, episode)
        status = p.key_status(k)
        bind(p, env, episode)
        served = p.assignment(k, LEVEL)
        joint, _, _ = J.admit_station_grid(VESSEL, 2, SEED, [(env, episode)])
        self.assertEqual(LEVEL, 2)                    # G31 J10: D1 is not applicable by design (G23e)
        if status == J.ADMITTED:
            if (k, LEVEL) in joint.assignments:
                self.assertEqual(served.candidate_digest, joint.assignments[(k, LEVEL)].candidate_digest)
                self.assertEqual(served.witness_digest, joint.assignments[(k, LEVEL)].witness_digest)
        else:
            self.assertIsInstance(served, J.NotAdmittedAssignment)

        def forced(jobs, joint=False, workers=1):
            real = D.admit_grid(jobs, joint=joint, workers=workers)
            kk = jobs[0][1]
            return dataclasses.replace(
                real, assignments={kd: a for kd, a in real.assignments.items() if kd != (kk, LEVEL)},
                rejections=tuple(real.rejections) + (D.Rejection(kk, LEVEL, "forced_test_rejection", 0),))
        q = pool(admit_grid=forced)
        self.assertEqual(q.key_status(k), J.NOT_ADMITTED)
        bind(q, env, episode)
        got = q.assignment(k, LEVEL)
        self.assertEqual(got.kind, J.E4_REJECTED)
        self.assertEqual(got.geometry_id, J.layout_hash(J.build_station_layout(2, VESSEL, SEED, env, episode)))
        self.assertIn("forced_test_rejection", [r for _, r, _ in got.reasons])

    def test_d_evaluation_seeds_refused_and_e_prefetch(self):
        for seed in (11001, 21001, 31001, 41041):
            with self.assertRaises(TH.TrainingHazardError):
                pool(m=mixture(seed=seed))
        outs = []
        for prefetch in (False, True):
            cache = TH.TrainingHazardCache(prefetch=prefetch)
            p = pool(cache=cache, num_envs=2)
            out = []
            for env in range(2):
                for episode in range(10):
                    k = J.episode_key(SEED, env, episode)
                    if p.key_status(k) != J.BEYOND_OFFERED_GRID:
                        bind(p, env, episode)
                    got = p.assignment(k, LEVEL)
                    out.append(getattr(got, "candidate_digest", None))
            outs.append(out)
            cache.close()
        self.assertEqual(outs[0], outs[1])


class StubEnvResets(unittest.TestCase):
    N, RESETS = 6, 14

    def test_f_g_reset_path_through_every_component(self):
        m = mixture()
        cache = TH.TrainingHazardCache()
        cfg = F.journal_cfg(VESSEL, num_envs=self.N, seed=SEED, tokens=m.required_tokens(),
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
        pools["d7"].assignment = lambda key, dose: served.append((key, original(key, dose))) or served[-1][1]
        env = F.make_env(cfg)
        for field in ("grid", "per_dose_counts", "keys_not_admitted", "pool_digest", "keys_offered", "joint"):
            self.assertIn(field, env.dynamic_hazard_header)
        layer = env._conditions
        self.assertIs(layer.training_mixture, m)
        seen = set()
        ids = torch.arange(self.N)
        for r in range(self.RESETS):
            del served[:]
            env._reset_idx(ids)
            for i in range(self.N):
                episode = int(layer.episode_index[i])
                comp = layer.training_component(i)
                self.assertEqual(comp, m.component(i, episode))
                seen.add(comp)
            by_env = {int(k.env): a for k, a in served}
            self.assertEqual(sorted(by_env), list(range(self.N)))
            for i, got in by_env.items():
                beyond = isinstance(got, J.NotAdmittedAssignment) and got.kind == J.BEYOND_OFFERED_GRID
                self.assertEqual(beyond, layer.training_component(i) != "dynamic", (r, i))
        self.assertEqual(seen, set(TM.COMPONENTS) - ({"appearing"} if not m.appearing_wired else set()))
        if pools["appearing"] is not None:
            self.assertTrue(appearing_served)
            for key, got in appearing_served:
                self.assertEqual(bool(got.beyond_offered_grid), m.component(key.env, key.episode) != "appearing", key)
            self.assertTrue(any(not got.beyond_offered_grid for _, got in appearing_served))
        env._get_observations()          # what DirectRLEnv.reset does after _reset_idx (one estimate per group)
        env.step(torch.zeros((self.N, 2)))
        log = cache.admission_log()["suites"]["dynamic"]
        self.assertGreater(log["keys_served_d1"], 0)
        print("G19b F4 STUB ENV: %d resets x %d envs, components %d, D7 admitted %d / not %d"
              % (self.RESETS, self.N, len(seen), log["admitted"], log["not_admitted"]), flush=True)


def _code9_support(levels):
    scripts = Path(__file__).resolve().parents[levels] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import g19b_training_test_support as GS
    return GS


class Code9RealEnv(unittest.TestCase):
    """Follow-up C1 (station keeping)."""

    def test_code9_reward_is_replaced_in_the_real_env(self):
        GS = _code9_support(3)
        cfg = F.journal_cfg(VESSEL, num_envs=2, seed=SEED)
        pools = TP.training_pools(cfg, None, cache=TH.TrainingHazardCache(), purpose="training_smoke")
        cfg.journal_d7_pool = pools["d7"]
        monitor, steps = GS.code9_run(F.make_env(cfg), "station_keeping")
        GS.check_code9(self, monitor, steps, 0.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
