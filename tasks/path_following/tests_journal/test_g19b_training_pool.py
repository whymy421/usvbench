"""G19b (IMPLEMENTATION_PLAN P3, F3): training-only online-admission D7 pool of path following + the shared
appearing adapter, CPU only (REAL PathFollowingEnv on f3_fixtures' fake DirectRLEnv, REAL E4 / G3 admission).

(a) non-component keys never call the admission function and serve the D24 D0 object on the key's E5 layout;
(b) a component key's served D1 assignment has the digests of an OFFLINE admission of that key
    (admit_path_following_keys, the family's own joint tool; and the non-joint rule the training uses);
(c) a forced D1 rejection gives the family's D16 not-admitted object;
(d) evaluation seeds are refused;
(e) prefetch on / off gives identical served objects;
(f) the REAL env with the all-D1 stack + mixture + adapters (cfg.journal_d7_pool = the adapter: the env does
    not load a pool file, G19b hunk) runs every mixture component, latches the stamps (component per episode =
    TrainingMixture.component), D7 / appearing rows run D1 only on their own component; (g) header fields.
Run: python -B tasks/path_following/tests_journal/test_g19b_training_pool.py
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import f3_fixtures as F  # noqa: E402

J = F.jpf
D = F.dynamic
TP = F.load("path_following.training_pool")
TH = F.load("_shared.training_hazards")
TM = F.load("_shared.conditions.training_mixture")
AC = F.load("_shared.conditions.appearing.config")
AK = F.load("_shared.conditions.appearing.placement")
AADM = F.load("_shared.conditions.appearing.admission")
SEED = 90001                      # training_smoke namespace (seed registry v3)
VESSEL = "blueboat_v2"
TASK = "path_following"
F64 = torch.float64


def mixture(seed=SEED, purpose="training_smoke"):
    return TM.TrainingMixture(seed=seed, task=TASK, vessel=VESSEL, appearing_wired=TASK in AC.WIRED_TASKS,
                              purpose=purpose)


def suite():
    return D.DynamicSuiteConfig(enabled=True, dose=1, vessel_id=VESSEL)


def d7_pool(m=None, cache=None, admit_grid=None, num_envs=6):
    m = mixture() if m is None else m
    return TP.TrainingPathFollowingD7Pool(mixture=m, task=TASK, vessel=VESSEL, seed=m.seed, tier=2,
                                          num_envs=num_envs, suite=suite(), cache=cache or TH.TrainingHazardCache(),
                                          purpose=m.purpose, admit_grid=admit_grid)


def key(env, episode, seed=SEED):
    return D.EpisodeKey(seed, env, episode, J.HAZARD_GROUP)


def component_keys(m, name, n_envs=6, episodes=12, want=True):
    return [(e, k) for e in range(n_envs) for k in range(episodes) if (m.hazard_dose(name, e, k) == 1) == want]


class Counter:
    def __init__(self, function):
        self.function, self.calls = function, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.function(*args, **kwargs)


class D7Adapter(unittest.TestCase):
    def test_a_non_component_keys_never_admit(self):
        counter = Counter(D.admit_grid)
        pool = d7_pool(admit_grid=counter)
        off = component_keys(pool.gate.mixture, "dynamic", want=False)
        for env, episode in off:
            k = key(env, episode)
            self.assertTrue(pool.beyond_grid(k))
            got = pool.assignment(k, 1)
            self.assertIsInstance(got, J.NotAdmittedAssignment)
            self.assertTrue(got.beyond_offered_grid)
            self.assertEqual(got.geometry_id, J.journal_layout(VESSEL, 2, SEED, env, episode).geometry_hash)
        self.assertEqual((counter.calls, pool.admission_calls), (0, 0))
        self.assertEqual(pool.grid_record(), {"num_envs": 6, "episodes": 0})
        self.assertEqual(pool.meta["suite"], J.suite_identity(J.admission_suite(VESSEL)))
        for bad in (key(6, 0), key(0, 0, seed=42)):
            with self.assertRaises(KeyError):
                pool.assignment(bad, 1)

    def test_b_component_key_matches_the_offline_admission(self):
        pool = d7_pool()
        checked = 0
        for env, episode in component_keys(pool.gate.mixture, "dynamic")[:3]:
            k = key(env, episode)
            self.assertFalse(pool.beyond_grid(k))
            geometry = pool.geometry(env, episode)
            served = pool.assignment(k, 1)
            layout = J.journal_layout(VESSEL, 2, SEED, env, episode)
            self.assertEqual(geometry, layout.geometry_hash)
            self.assertEqual(J.assignment_geometry(served), layout.geometry_hash)
            joint = J.admit_path_following_keys([(VESSEL, env, episode)], SEED, tier=2)[VESSEL]["pool"]
            non_joint = D.admit_grid([(J.d7_task_context(layout, VESSEL), k, J.admission_suite(VESSEL))],
                                     joint=False)
            if (k, 1) not in non_joint.assignments:
                self.assertIsInstance(served, J.NotAdmittedAssignment)
                continue
            for reference in [non_joint.assignments[(k, 1)]] + ([joint.assignments[(k, 1)]]
                                                                if (k, 1) in joint.assignments else []):
                self.assertEqual(served.candidate_digest, reference.candidate_digest)
                self.assertEqual(served.witness_digest, reference.witness_digest)
            checked += 1
        self.assertGreater(checked, 0)

    def test_c_forced_rejection_gives_the_d16_object(self):
        def forced(jobs, joint=False, workers=1):
            real = D.admit_grid(jobs, joint=joint, workers=workers)
            k = jobs[0][1]
            return dataclasses.replace(
                real, assignments={kd: a for kd, a in real.assignments.items() if kd != (k, 1)},
                rejections=tuple(real.rejections) + (D.Rejection(k, 1, "forced_test_rejection", 0),))
        pool = d7_pool(admit_grid=forced)
        env, episode = component_keys(pool.gate.mixture, "dynamic")[0]
        got = pool.assignment(key(env, episode), 1)
        self.assertIsInstance(got, J.NotAdmittedAssignment)
        self.assertFalse(got.beyond_offered_grid)
        self.assertIn("forced_test_rejection", J.primary_reasons(got.rejections))
        self.assertEqual(got.geometry_id, J.journal_layout(VESSEL, 2, SEED, env, episode).geometry_hash)
        self.assertFalse(pool.is_admitted(env, episode))

    def test_d_evaluation_seeds_are_refused(self):
        for seed in (11001, 11004, 21001, 31008, 41041, 41161):
            with self.assertRaises(TH.TrainingHazardError):
                d7_pool(m=mixture(seed=seed, purpose="training_smoke"))
            with self.assertRaises(TH.TrainingHazardError):
                d7_pool(m=mixture(seed=seed, purpose="training"))

    def test_e_prefetch_gives_identical_served_objects(self):
        m = mixture()
        outs = []
        for prefetch in (False, True):
            cache = TH.TrainingHazardCache(prefetch=prefetch)
            pool = d7_pool(m=m, cache=cache, num_envs=3)
            out = []
            for env in range(3):
                for episode in range(12):
                    got = pool.assignment(key(env, episode), 1)
                    out.append(("na", got.rejections, got.geometry_id) if isinstance(got, J.NotAdmittedAssignment)
                               else ("a", got.candidate_digest, got.witness_digest))
            outs.append(out)
            cache.close()
        self.assertEqual(outs[0], outs[1])


def training_cfg(m, num_envs, cache):
    _, _, templates = F.env_modules()
    cfg = templates.PathFollowingBlueBoatJournalEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.visual.enable_water = False
    cfg.visual.enable_waypoint_markers = False
    cfg.visual.enable_path_line = False
    resolved = F.journal_config.resolve_journal_conditions(presets=m.required_tokens(), seed=m.seed)
    F.journal_config.apply_journal_conditions(cfg, resolved)
    templates.journal_validate(cfg)
    pools = TP.training_pools(cfg, m, cache=cache, purpose=m.purpose)
    cfg.journal_d7_pool = pools["d7"]
    cfg.journal_appearing_pool = pools["appearing"]
    cfg.journal_training_mixture = m
    return cfg


class StubEnvThroughEveryComponent(unittest.TestCase):
    N, EPISODES = 6, 14

    def test_f_g_env_runs_every_component_and_latches_stamps(self):
        m = mixture()
        cache = TH.TrainingHazardCache()
        env = F.make_env(training_cfg(m, self.N, cache))
        self.assertIs(env._conditions.training_mixture, m)
        header = env.dynamic_hazard_header
        for field in ("pool_digest", "per_dose_counts", "keys_offered", "keys_admitted", "joint", "grid",
                      "keys_not_admitted", "pool_pinned"):
            self.assertIn(field, header)
        self.assertIs(header["pool_pinned"], False)
        self.assertEqual(header["pool_file"]["source"], "training_online")
        self.assertEqual(env.appearing_hazard_header["training"]["mixture_sha256"], m.digest())
        env.reset()
        seen, checked = set(), 0
        for step in range(self.EPISODES):
            env._journal_oracle.deadline[:] = (env.episode_length_buf.to(F64) + 1) * F.plans.CONTROL_DT_S
            env.step(torch.tensor([[0.4, 0.0]] * self.N))
            for i in range(self.N):
                rec = env.journal_episode_record(i)
                runtime = rec["conditions"]["runtime"]
                self.assertEqual(runtime["episode_index"], step)
                comp = runtime["training_mixture"]["component"]
                self.assertEqual(comp, m.component(i, step))
                seen.add(comp)
                d7 = rec["dynamic_hazards"]["payload"]
                beyond = d7["assignment"].get("rejections") == [[0, J.BEYOND_GRID_REASON]]
                self.assertEqual(beyond, comp != "dynamic", (i, step, comp))
                appearing = rec["appearing_hazards"]["payload"]["assignment"]
                self.assertEqual(appearing.get("not_admitted_reasons") == [[0, "beyond_offered_grid"]],
                                 comp != "appearing", (i, step, comp))
                checked += 1
        self.assertEqual(seen, set(TM.COMPONENTS))
        log = cache.admission_log()["suites"]
        self.assertEqual(log["dynamic"]["keys_served_d1"],
                         len(component_keys(m, "dynamic", self.N, self.EPISODES + 1)))
        self.assertEqual(log["appearing"]["keys_served_d1"],
                         len(component_keys(m, "appearing", self.N, self.EPISODES + 1)))
        with self.assertRaises(RuntimeError):
            env.journal_certificate_header()
        print("G19b F3 STUB ENV: %d episodes, components %d/%d, D7 admitted %d / not %d, appearing %d / %d"
              % (checked, len(seen), len(TM.COMPONENTS), log["dynamic"]["admitted"], log["dynamic"]["not_admitted"],
                 log["appearing"]["admitted"], log["appearing"]["not_admitted"]), flush=True)


class Appearing(unittest.TestCase):
    def test_appearing_component_key_matches_the_offline_pool(self):
        m = mixture()
        counter = Counter(AK.admit_key)
        pool = TH.TrainingAppearingPool(mixture=m, task=TASK, vessel=VESSEL, seed=SEED, tier=2, num_envs=4,
                                        inputs=TP.appearing_inputs(TASK, VESSEL, 2, SEED),
                                        cache=TH.TrainingHazardCache(), purpose=m.purpose, admit=counter)
        akey = lambda e, k: D.EpisodeKey(SEED, e, k, AC.hazard_group(TASK))  # noqa: E731
        for e, k in component_keys(m, "appearing", 4, 12, want=False):
            self.assertTrue(pool.assignment(akey(e, k), 1).beyond_offered_grid)
        self.assertEqual(counter.calls, 0)
        e, k = component_keys(m, "appearing", 4, 12)[0]
        served = pool.assignment(akey(e, k), 1)
        offline = AADM.build_pool(TASK, VESSEL, SEED, 2, e + 1, k + 1,
                                  lambda env, ep: J.appearing_admission_inputs(VESSEL, 2, SEED, env, ep))
        reference = offline.assignment(akey(e, k), 1)
        self.assertEqual((served.exposure, served.position_digest, served.candidate_digest, served.geometry_id),
                         (reference.exposure, reference.position_digest,
                          reference.candidate_digest if reference.exposure == "admitted" else served.candidate_digest,
                          reference.geometry_id))
        self.assertEqual(counter.calls, 1)


def _code9_support(levels):
    scripts = Path(__file__).resolve().parents[levels] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import g19b_training_test_support as GS
    return GS


class Code9RealEnv(unittest.TestCase):
    """Follow-up C1 (path following)."""

    def test_code9_reward_is_replaced_in_the_real_env(self):
        GS = _code9_support(3)
        _, _, templates = F.env_modules()
        cfg = templates.PathFollowingBlueBoatJournalEnvCfg()
        cfg.scene.num_envs = 2
        cfg.visual.enable_water = False
        cfg.visual.enable_waypoint_markers = False
        cfg.visual.enable_path_line = False
        F.journal_config.apply_journal_conditions(cfg, F.journal_config.resolve_journal_conditions(presets=[],
                                                                                                   seed=SEED))
        templates.journal_validate(cfg)
        pools = TP.training_pools(cfg, None, cache=TH.TrainingHazardCache(), purpose="training_smoke")
        cfg.journal_d7_pool = pools["d7"]
        monitor, steps = GS.code9_run(F.make_env(cfg), "path_following")
        GS.check_code9(self, monitor, steps, 0.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
