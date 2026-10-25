"""G19b (IMPLEMENTATION_PLAN P3, F1): training-only online-admission D7 pool of HazardNav + the shared appearing
adapter, CPU only (REAL HazardNavEnv on journal_test_support's stub DirectRLEnv, REAL E4 / G3 admission).

(a) non-component keys never call the admission function (call counter 0) and serve the D24 D0 object;
(b) a component key's served D1 assignment has the candidate / witness digests of an OFFLINE admission of that
    key (E4 admit_ladders on the same scenario; appearing: the G3 offline build_pool);
(c) a forced D1 rejection gives the family's D16 not-admitted object (with the scenario geometry);
(d) evaluation seeds (selection, pilot, calibration, final test) and unfrozen campaign seeds are refused;
(e) prefetch on / off gives identical served objects;
(f) the REAL env with the all-D1 stack + mixture + adapters steps through every mixture component, latches the
    episode stamps (component per episode = TrainingMixture.component), D7 / appearing rows run D1 only on their
    own component, and the mixin header fields exist (g); the env never produces a certificate header.
Run: python -B tasks/hazard_nav/test_g19b_training_pool.py
"""
from __future__ import annotations

import copy
import dataclasses
import importlib
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import journal_test_support as S  # noqa: E402

RESTORE = S.install_isaaclab_stubs()
FAM = S.load_family()
J, D = FAM.journal, FAM.dynamic
imp = lambda name: importlib.import_module(S.ROOT + "." + name)  # noqa: E731
TP = imp("hazard_nav.training_pool")
TH = imp("_shared.training_hazards")
TM = imp("_shared.conditions.training_mixture")
AC = imp("_shared.conditions.appearing.config")
AADM = imp("_shared.conditions.appearing.admission")
SEED = 90001                      # training_smoke namespace (seed registry v3): usable while unfrozen
VESSEL = "blueboat_v2"
TASK = "scatter"


def mixture(task=TASK, seed=SEED, purpose="training_smoke"):
    return TM.TrainingMixture(seed=seed, task=task, vessel=VESSEL, appearing_wired=task in AC.WIRED_TASKS,
                              purpose=purpose)


def suite():
    return D.DynamicSuiteConfig(enabled=True, dose=1, vessel_id=VESSEL)


def d7_pool(m=None, cache=None, admit_grid=None, num_envs=6, **kwargs):
    m = mixture() if m is None else m
    return TP.TrainingHazardD7Pool(mixture=m, task=m.task, vessel=VESSEL, seed=m.seed, tier=2, num_envs=num_envs,
                                   suite=suite(), cache=cache or TH.TrainingHazardCache(), purpose=m.purpose,
                                   admit_grid=admit_grid, **kwargs)


def key(env, episode, task=TASK, seed=SEED):
    return D.EpisodeKey(seed, env, episode, "d7_" + task)


def component_keys(m, suite_name, n_envs=6, episodes=12, want=True):
    return [(e, k) for e in range(n_envs) for k in range(episodes)
            if (m.hazard_dose(suite_name, e, k) == 1) == want]


class Counter:
    def __init__(self, function):
        self.function, self.calls = function, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.function(*args, **kwargs)


class D7Adapter(unittest.TestCase):
    def test_a_non_component_keys_never_admit_and_run_the_d24_object(self):
        counter = Counter(D.admit_grid)
        pool = d7_pool(admit_grid=counter)
        off = component_keys(pool.gate.mixture, "dynamic", want=False)
        self.assertGreater(len(off), 40)
        for env, episode in off:
            got = pool.assignment(key(env, episode), 1)
            self.assertIsInstance(got, J.NotAdmittedAssignment)
            self.assertTrue(got.record.beyond_offered_grid)
            self.assertTrue(pool.beyond_offered_grid(key(env, episode)))
            scenario = J.build_scenario(TASK, VESSEL, 2, SEED, env, episode)
            self.assertIs(type(J.check_assignment_geometry(pool, scenario, 1)), J.NotAdmittedAssignment)
        self.assertEqual((counter.calls, pool.admission_calls), (0, 0))
        log = pool.gate.cache.admission_log()["suites"]["dynamic"]
        self.assertEqual(log["keys_served_d1"], 0)
        self.assertEqual(log["d0_keys_served"], len(off))
        self.assertIs(J.family_pool(pool), pool)
        for bad in (key(6, 0), key(0, 0, seed=42), key(0, 0, task="ring")):
            with self.assertRaises(KeyError):
                pool.assignment(bad, 1)
        with self.assertRaises(TH.TrainingHazardError):
            pool.assignment(key(0, 0), 2)

    def test_b_component_key_matches_the_offline_admission(self):
        pool = d7_pool()
        checked = 0
        for env, episode in component_keys(pool.gate.mixture, "dynamic")[:4]:
            k = key(env, episode)
            served = pool.assignment(k, 1)
            scenario = J.build_scenario(TASK, VESSEL, 2, SEED, env, episode)
            ladder = D.build_ladder(J.d7_context(scenario), k, suite())
            non_joint = D.admit_ladders([(k, ladder)], joint=False)       # the training rule (D1 taken)
            joint = D.admit_ladders([(k, ladder)], joint=True)            # the evaluation pools' rule
            if (k, 1) not in non_joint.assignments:
                self.assertIsInstance(served, J.NotAdmittedAssignment)
                continue
            for reference in [non_joint.assignments[(k, 1)]] + ([joint.assignments[(k, 1)]]
                                                                if (k, 1) in joint.assignments else []):
                self.assertEqual(served.candidate_digest, reference.candidate_digest)
                self.assertEqual(served.witness_digest, reference.witness_digest)
            self.assertIs(J.check_assignment_geometry(pool, scenario, 1), served)
            checked += 1
        self.assertGreater(checked, 0)
        log = pool.gate.cache.admission_log()["suites"]["dynamic"]
        self.assertEqual(log["keys_served_d1"], 4)
        self.assertEqual(log["admitted"] + log["not_admitted"], 4)

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
        self.assertFalse(got.record.beyond_offered_grid)
        self.assertIn("forced_test_rejection", [r for _, r, _ in got.reasons])
        scenario = J.build_scenario(TASK, VESSEL, 2, SEED, env, episode)
        self.assertEqual(got.record.geometry_id, scenario.geometry_id)
        self.assertEqual(J.check_assignment_geometry(pool, scenario, 1), got)
        log = pool.gate.cache.admission_log()["suites"]["dynamic"]
        self.assertEqual(log["not_admitted"], 1)
        self.assertIn("forced_test_rejection", log["not_admitted_reasons"])

    def test_d_evaluation_and_unfrozen_campaign_seeds_are_refused(self):
        for seed in (11001, 21001, 31001, 41041, 7, 55555):
            with self.assertRaises(TH.TrainingHazardError):
                d7_pool(m=mixture(seed=seed, purpose="training"))
            with self.assertRaises(TH.TrainingHazardError):
                d7_pool(m=mixture(seed=seed, purpose="training_smoke"))
        module = TH.seed_registry_module()
        proposal = module.load_registry()                      # G52: the in-tree registry is frozen since v2.6.6
        proposal["status"], proposal["competence_gate"]["status"] = "proposal", "OWNER_DECISION_PENDING"
        with self.assertRaises(TH.TrainingHazardError):        # campaign seed, registry a proposal
            d7_pool(m=mixture(seed=42, purpose="training"), registry=proposal)
        frozen = module.load_registry()
        frozen["status"], frozen["competence_gate"]["status"] = "frozen", "frozen"
        pool = d7_pool(m=mixture(seed=42, purpose="training"), registry=frozen)
        self.assertEqual(pool.gate.registry_stamp["candidate_index"], 0)
        with self.assertRaises(TH.TrainingHazardError):        # mixture of another seed
            TP.TrainingHazardD7Pool(mixture=mixture(seed=90002), task=TASK, vessel=VESSEL, seed=SEED, tier=2,
                                    num_envs=2, suite=suite(), cache=TH.TrainingHazardCache(),
                                    purpose="training_smoke")

    def test_e_prefetch_gives_identical_served_objects(self):
        m = mixture()
        keys = [(e, k) for e in range(3) for k in range(12)]
        served = []
        for prefetch in (False, True):
            cache = TH.TrainingHazardCache(prefetch=prefetch)
            pool = d7_pool(m=m, cache=cache, num_envs=3)
            out = []
            for env, episode in keys:
                got = pool.assignment(key(env, episode), 1)
                out.append(("na", got.reasons) if isinstance(got, J.NotAdmittedAssignment)
                           else ("a", got.candidate_digest, got.witness_digest))
            log = cache.admission_log()
            cache.close()
            served.append((out, log))
        self.assertEqual(served[0][0], served[1][0])
        self.assertEqual(served[0][1]["suites"]["dynamic"]["keys_served_d1"],
                         served[1][1]["suites"]["dynamic"]["keys_served_d1"])
        self.assertGreater(served[1][1]["suites"]["dynamic"]["prefetched_served"], 0)


class AppearingAdapter(unittest.TestCase):
    def adapter(self, cache=None, admit=None, num_envs=6, m=None):
        m = mixture() if m is None else m
        return TH.TrainingAppearingPool(mixture=m, task=TASK, vessel=VESSEL, seed=m.seed, tier=2,
                                        num_envs=num_envs, inputs=TP.appearing_inputs(TASK, VESSEL, 2, m.seed),
                                        cache=cache or TH.TrainingHazardCache(), purpose=m.purpose, admit=admit)

    def test_appearing_a_b_c_d_e(self):
        AK = imp("_shared.conditions.appearing.placement")
        counter = Counter(AK.admit_key)
        pool = self.adapter(admit=counter)
        m = pool.mixture
        akey = lambda e, k: D.EpisodeKey(SEED, e, k, AC.hazard_group(TASK))  # noqa: E731
        for env, episode in component_keys(m, "appearing", want=False):                 # (a)
            got = pool.assignment(akey(env, episode), 1)
            self.assertTrue(got.beyond_offered_grid and got.dose == 0)
        self.assertEqual(counter.calls, 0)
        env, episode = component_keys(m, "appearing")[0]                                 # (b)
        served = pool.assignment(akey(env, episode), 1)
        offline = AADM.build_pool(TASK, VESSEL, SEED, 2, env + 1, episode + 1,
                                  lambda e, k: J.appearing_admission_inputs(TASK, VESSEL, 2, SEED, e, k))
        reference = offline.assignment(akey(env, episode), 1)
        self.assertEqual(type(served).__name__, type(reference).__name__)
        self.assertEqual(served.position_digest, reference.position_digest)
        if served.exposure == "admitted":
            self.assertEqual(served.candidate_digest, reference.candidate_digest)
            self.assertEqual(served.dose, 1)
        self.assertEqual(served.geometry_id, reference.geometry_id)
        self.assertEqual(counter.calls, 1)
        self.assertEqual(pool.suite_identity, offline.suite_identity)

        def forced(context, k, cfg, **kwargs):                                           # (c)
            out = AK.admit_key(context, k, cfg, **kwargs)
            return dict(out, doses={}, reasons=[(1, "forced_test_rejection")])
        rejected = self.adapter(admit=forced).assignment(akey(env, episode), 1)
        self.assertEqual(rejected.exposure, "not_admitted")
        self.assertFalse(rejected.beyond_offered_grid)
        self.assertEqual(rejected.geometry_id, reference.geometry_id)
        for seed in (11001, 21001, 31001, 41041):                                         # (d)
            with self.assertRaises(TH.TrainingHazardError):
                self.adapter(m=mixture(seed=seed, purpose="training_smoke"))
        outs = []                                                                         # (e)
        for prefetch in (False, True):
            cache = TH.TrainingHazardCache(prefetch=prefetch)
            p = self.adapter(cache=cache, num_envs=2)
            outs.append([(a.exposure, a.position_digest, a.candidate_digest)
                         for a in (p.assignment(akey(e, k), 1) for e in range(2) for k in range(12))])
            cache.close()
        self.assertEqual(outs[0], outs[1])
        if "reach" not in AC.WIRED_TASKS:                 # v2.5; G20b wires reach (then no refusal applies)
            with self.assertRaises(TH.TrainingHazardError):
                TH.TrainingAppearingPool(mixture=mixture("reach"), task="reach", vessel=VESSEL, seed=SEED, tier=2,
                                         num_envs=2, inputs=lambda e, k: None, cache=TH.TrainingHazardCache(),
                                         purpose="training_smoke")


class StubEnvThroughEveryComponent(unittest.TestCase):
    N, EPISODES = 6, 14

    def test_f_g_env_runs_every_component_and_latches_stamps(self):
        m = mixture()
        cache = TH.TrainingHazardCache()
        cfg = S.journal_cfg(FAM, TASK, VESSEL, tokens=m.required_tokens(), seed=SEED, num_envs=self.N)
        pools = TP.training_pools(cfg, m, cache=cache, purpose="training_smoke")
        cfg.journal_d7_pool = pools["d7"]
        cfg.journal_appearing_pool = pools["appearing"]
        cfg.journal_training_mixture = m
        env = S.make_env(FAM, cfg)
        self.assertIs(env._conditions.training_mixture, m)
        header = env.dynamic_hazard_header                                                # (g)
        for field in ("pool_digest", "per_dose_counts", "keys_offered", "keys_admitted", "joint", "grid",
                      "keys_not_admitted"):
            self.assertIn(field, header)
        self.assertEqual(env.appearing_hazard_header["training"]["mixture_sha256"], m.digest())
        env.reset()
        seen, records = set(), []
        for step in range(self.EPISODES):
            env._journal_twin.deadline[:] = 1 * FAM.plans.CONTROL_DT_S
            env.step(S.zero_action(env, (0.4, 0.0)))
            for i in range(self.N):
                rec = env.journal_completed_records[i]
                records.append((i, step, rec))
        for i, step, rec in records:
            self.assertEqual(rec["scenario_episode"], step)
            comp = rec["conditions"]["runtime"]["training_mixture"]["component"]
            self.assertEqual(comp, m.component(i, step))
            seen.add(comp)
            d7 = D.read_stamp(rec["dynamic_hazards"])["payload"]
            if comp != "dynamic":
                self.assertEqual(d7["not_admitted"]["reasons"], [[0, "beyond_offered_grid", 0]], (i, step))
            else:
                self.assertNotEqual(d7.get("not_admitted", {}).get("reasons"), [[0, "beyond_offered_grid", 0]])
            appearing = rec["appearing_hazards"]["payload"]["assignment"]
            if comp != "appearing":
                self.assertEqual(appearing["not_admitted_reasons"], [[0, "beyond_offered_grid"]], (i, step))
            else:
                self.assertNotEqual(appearing.get("not_admitted_reasons"), [[0, "beyond_offered_grid"]])
        self.assertEqual(seen, set(TM.COMPONENTS))
        counts = env._conditions.training_mixture_counts()
        self.assertEqual(sum(counts.values()), self.N * (self.EPISODES + 1))
        log = cache.admission_log()["suites"]
        self.assertEqual(log["dynamic"]["keys_served_d1"],
                         len(component_keys(m, "dynamic", self.N, self.EPISODES + 1)))
        self.assertEqual(log["appearing"]["keys_served_d1"],
                         len(component_keys(m, "appearing", self.N, self.EPISODES + 1)))
        with self.assertRaises(RuntimeError):
            env.journal_certificate_header()
        print("G19b F1 STUB ENV: %d episodes, components %d/%d, D7 keys admitted %d / not %d, appearing %d / %d"
              % (len(records), len(seen), len(TM.COMPONENTS), log["dynamic"]["admitted"],
                 log["dynamic"]["not_admitted"], log["appearing"]["admitted"], log["appearing"]["not_admitted"]),
              flush=True)


def _code9_support(levels):
    scripts = Path(__file__).resolve().parents[levels] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import g19b_training_test_support as GS
    return GS


class EnvLevelGateEquality(unittest.TestCase):
    """Follow-up (review): at env level, a row whose episode component is 'calm' (every gate off, D7 D0) evolves
    bitwise like the same row of a calm-arm env (no stack): pose, COM velocity, reward and observation, over 6
    episodes of 5 control steps (REAL HazardNavEnv, reach, training adapters on both sides)."""

    def make(self, m, tokens):
        cfg = S.journal_cfg(FAM, "reach", VESSEL, tokens=tokens, seed=SEED, num_envs=self.N)
        if m is not None:
            cfg.journal_training_mixture = m
        pools = TP.training_pools(cfg, m, cache=TH.TrainingHazardCache(), purpose="training_smoke")
        cfg.journal_d7_pool = pools["d7"]
        if pools["appearing"] is not None:                 # reach is appearing-wired once G20b lands
            cfg.journal_appearing_pool = pools["appearing"]
        return S.make_env(FAM, cfg)

    N, EPISODES, STEPS = 6, 6, 5

    def test_calm_component_rows_equal_a_calm_env(self):
        m = mixture(task="reach")
        mix, calm = self.make(m, m.required_tokens()), self.make(None, ())
        self.assertIs(mix._conditions.training_mixture, m)
        mix.reset()
        calm.reset()
        compared = 0
        for episode in range(self.EPISODES):
            for _ in range(self.STEPS):
                for env in (mix, calm):
                    env._journal_twin.deadline[:] = self.STEPS * FAM.plans.CONTROL_DT_S
                comps = [mix._conditions.training_component(i) for i in range(self.N)]
                out_m = mix.step(S.zero_action(mix, (0.6, 0.2)))
                out_c = calm.step(S.zero_action(calm, (0.6, 0.2)))
                obs_m = out_m[0]["policy"] if isinstance(out_m[0], dict) else out_m[0]
                obs_c = out_c[0]["policy"] if isinstance(out_c[0], dict) else out_c[0]
                after = [mix._conditions.training_component(i) for i in range(self.N)]
                for i in range(self.N):
                    if comps[i] == "calm":       # this step's reward belongs to the episode that ran it
                        self.assertTrue(torch.equal(out_m[1][i], out_c[1][i]), (episode, i))
                    if after[i] != "calm":       # the post-step state belongs to the (possibly new) episode
                        continue
                    self.assertTrue(torch.equal(mix.robot.data.root_link_pose_w[i], calm.robot.data.root_link_pose_w[i]),
                                    (episode, i))
                    self.assertTrue(torch.equal(mix.robot.data.root_com_vel_w[i], calm.robot.data.root_com_vel_w[i]))
                    self.assertTrue(torch.equal(obs_m[i], obs_c[i]), (episode, i))
                    compared += 1
        self.assertGreater(compared, self.N * self.STEPS)
        print("G19b ENV GATE EQUALITY: %d calm-component row steps bitwise equal to the calm env" % compared,
              flush=True)


class Code9RealEnv(unittest.TestCase):
    """Follow-up C1: a NaN pose ends the row with code 9 and a NaN reward; the training wrapper replaces it."""

    def test_code9_reward_is_replaced_in_the_real_env(self):
        GS = _code9_support(2)
        for task in ("reach", "scatter"):
            cfg = S.journal_cfg(FAM, task, VESSEL, tokens=(), seed=SEED, num_envs=2)
            pools = TP.training_pools(cfg, None, cache=TH.TrainingHazardCache(), purpose="training_smoke")
            cfg.journal_d7_pool = pools["d7"]
            env = S.make_env(FAM, cfg)
            monitor, steps = GS.code9_run(env, "hazard_nav")
            GS.check_code9(self, monitor, steps, -25.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    try:
        unittest.main(verbosity=2)
    finally:
        RESTORE()
