"""F2 D24 tests: episodes beyond the offered grid run at D0 and are never counted (CONTEXT_R4 D24).

Run: python -B tasks/path_hazard/tests_journal/test_journal_d24.py   (CPU only)

The evaluators keep stepping every env until the slowest reaches its per-env quota, so a fast env
autoresets into task episode indices past the admission grid. With the in-memory WAM-V fixture pool
(seed 42, 2 envs x 3 episodes, grid {"num_envs": 2, "episodes": 3}; the same admission function as
the offline tool):

* pool: key (42, env < 2, episode >= 3) is NOT ADMITTED with the single reason
  "beyond_offered_grid" at every dose (never KeyError); it is no pool key (counts, not-admitted
  list and geometry hashes unchanged); env >= 2, another seed or another hazard group stays a
  KeyError (preflight error); the offered keys must form one full grid;
* env: row 0 finishes episodes 0, 1, 2 early (forced 3-step contract deadline) while row 1 stays in
  episode 0; row 0 then resets into episode 3 = E and runs a full episode at every dynamic level
  D0..D4 (and with the suite absent) without an exception: no intruder, its layout is the E5 layout
  of key (42, 0, 3), the record stamps exposure "not_admitted" with the single reason
  "beyond_offered_grid", and F0 pairing.d7_primitive_hash accepts the stamp with ONE value at every
  level; the env goes on into episode 4 (also beyond the grid);
* dynamic_hazard_header["grid"] equals the pool's offered grid;
* env preflight: an env outside the grid or another eval seed fails closed at reset with a clear
  ValueError;
* the real pinned pools (F2 evidence files, when next to this repository copy) expose grid 64 x 2.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ph_fixtures as F  # noqa: E402

M = F.modules()
J = M.journal
STUBS = F.install_stubs()
CERT = importlib.import_module("tasks._shared.conditions.certificate")
VESSEL = "wamv_vrx_v2_opt1"
GRID = {"num_envs": 2, "episodes": 3}
FAST, SLOW = 0, 1                 # env rows: row 0 finishes early, row 1 stays in episode 0
EVIDENCE = F.REPO.parent / "evidence" / "d7_pool"
_RUNS = {}


def pool():
    return F.test_pool(VESSEL, 2, 3)


def d20_reasons(payload):
    """CONTEXT_R4 D20 (c) reason reading of a not-admitted payload (first non-empty source)."""
    sources = (payload.get("not_admitted", {}).get("reasons") if isinstance(payload.get("not_admitted"), dict)
               else None, payload.get("not_admitted_reasons"), payload["assignment"].get("not_admitted_reasons"),
               payload["assignment"].get("rejection_reasons"), payload["runtime"].get("rejection_reasons"),
               payload["runtime"].get("reasons"), payload.get("rejection_reasons"), payload.get("reasons"))
    items = next(s for s in sources if s)
    out = []
    for item in items:
        if isinstance(item, str):
            names = [item]
        elif isinstance(item, dict):
            names = [item["reason"]]
        else:
            names = [v for v in item if isinstance(v, str)]
        for name in names:
            if name != "joint_ladder_member_rejected" and name not in out:
                out.append(name)
    return out


def expected_layout(env_index, episode_index):
    """E5 build_layout on the keyed stream "journal_layout", computed independently of the family module."""
    layouts = importlib.import_module("tasks._shared.journal_layouts_v2")
    rng_mod = importlib.import_module("tasks._shared.scenario_rng")
    rng = rng_mod.numpy_generator(int(rng_mod.SCENARIO_PROTOCOL_VERSION), 42, env_index, episode_index,
                                  "journal_layout")
    return layouts.build_layout("path_hazard", 2, VESSEL, rng, max_attempts=200)


def force_short_deadline(env, row):
    """Contract deadline of 3 control steps for ``row`` (its current episode ends at timeout, code 3)."""
    route = torch.as_tensor(J.route_points(env._jph_layouts[row]), dtype=torch.float64).unsqueeze(0)
    env._jph_oracle.reset(torch.tensor([row]), route, env._jph_gate_radius,
                          torch.full((1,), 3.0 / 60.0, dtype=torch.float64), route[:, 0])


def beyond_grid_run(tokens):
    """Row 0 runs episodes 0..3 (3 = one past the grid) and resets into 4; row 1 stays in episode 0.

    Returns a dict with the env, the episode-3 snapshot taken right after its reset, the latched
    episode-3 record, the episode-3 completion entry and whether row 0 published a D7 sample in it.
    """
    if tokens in _RUNS:
        return _RUNS[tokens]
    env = F.make_env(F.journal_cfg(VESSEL, tokens=tokens, pool=pool()))
    published = []
    original = env._journal_publish_d7

    def publish(samples):
        episodes = env._scenario.episode_indices().tolist()
        published.extend((int(i), int(episodes[int(i)]), len(samples[i])) for i in samples)
        return original(samples)
    env._journal_publish_d7 = publish
    env.reset()
    n = env.num_envs
    force_short_deadline(env, FAST)
    episode_of = lambda row: int(env._scenario.episode_indices()[row])  # noqa: E731
    out = {"env": env, "snapshot": None, "record": None, "completed": None, "codes": []}
    for k in range(40):
        before = episode_of(FAST)
        env.step(F.actions(k, n))
        self_slow = episode_of(SLOW)
        assert self_slow == 0, "the slow row must stay in episode 0"
        after = episode_of(FAST)
        if after == before:
            continue
        assert after == before + 1
        out["codes"].append(int(env.termination_reason[FAST]))
        if before == GRID["episodes"]:          # episode 3 just finished (latched), row 0 is in episode 4
            out["record"] = env.journal_episode_record(FAST)
            out["completed"] = dict(env.journal_completed[FAST])
            out["cert_fields"] = CERT.episode_certificate_fields(env, FAST)
            out["episode4_not_admitted"] = env._d7_bridge.is_not_admitted(FAST)
            break
        force_short_deadline(env, FAST)
        if after == GRID["episodes"]:           # reset into the first beyond-grid episode
            out["snapshot"] = {"layout": env._jph_layouts[FAST], "exposure": env._jph_exposure[FAST],
                               "not_admitted": env._d7_bridge.is_not_admitted(FAST),
                               "centers": env._jph_centers[FAST].clone(),
                               "waypoints": env._jph_waypoints[FAST].clone(),
                               "params": dict(env._scenario_params[FAST]["layout"])}
    assert out["record"] is not None, "row 0 never finished its beyond-grid episode"
    out["d7_seen"] = any(c > 0 for r, e, c in published if r == FAST and e == GRID["episodes"])
    assert any(r == FAST and e == GRID["episodes"] for r, e, _ in published)       # publication did run
    _RUNS[tokens] = out
    return out


class PoolGrid(unittest.TestCase):
    def test_grid_and_beyond_grid_keys(self):
        p = pool()
        self.assertEqual(p.grid, GRID)
        self.assertEqual((p.eval_seed, p.hazard_group), (42, "d7_path_hazard"))
        counts, not_admitted, hashes = p.per_dose_counts, dict(p.not_admitted), dict(p.geometry_hashes)
        for env_index in range(2):
            for episode in (3, 4, 17, 1000):
                self.assertEqual(p.key_status(42, env_index, episode), J.BEYOND_OFFERED_GRID)
                self.assertFalse(p.is_admitted(42, env_index, episode))
                self.assertFalse(p.is_offered(42, env_index, episode))
                key = J.episode_key(42, env_index, episode)
                records = [p.assignment(key, dose) for dose in range(5)]          # never KeyError
                for record in records:
                    self.assertIsInstance(record, J.NotAdmittedAssignment)
                    self.assertEqual(record, records[0])                         # dose independent
                    self.assertEqual(record.key, key)
                    self.assertEqual(record.reasons, ((0, "beyond_offered_grid", 0),))
                    summary = record.summary()
                    self.assertEqual((summary["dose"], summary["candidate_digest"], summary["intruders"]),
                                     (0, None, []))
                    self.assertEqual(summary["key"]["episode"], episode)
        for env_index in range(2):          # inside the grid: unchanged D16 classification
            for episode in range(3):
                status = p.key_status(42, env_index, episode)
                self.assertEqual(status, "not_admitted" if (42, env_index, episode) in p.not_admitted
                                 else "admitted")
        # beyond-grid keys are no pool keys: nothing counted or stored
        self.assertEqual(p.per_dose_counts, counts)
        self.assertEqual(p.per_dose_counts, ((6, p.keys_admitted, p.keys_not_admitted),) * 5)
        self.assertEqual(p.not_admitted, not_admitted)
        self.assertEqual(p.geometry_hashes, hashes)
        self.assertEqual(p.header_fields()["grid"], GRID)
        self.assertEqual(p.header_fields()["per_dose_counts"], [list(r) for r in counts])

    def test_keys_outside_the_env_range_or_with_another_seed_or_group_stay_preflight_errors(self):
        p = pool()
        dynamic = M.dynamic
        bad = [(42, 2, 0), (42, 2, 3), (42, 9, 0), (43, 0, 0), (43, 0, 3), (7, 1, 1)]
        for seed, env_index, episode in bad:
            with self.assertRaises(KeyError):
                p.key_status(seed, env_index, episode)
            with self.assertRaises(KeyError):
                p.is_admitted(seed, env_index, episode)
            for dose in range(5):
                with self.assertRaises(KeyError):
                    p.assignment(J.episode_key(seed, env_index, episode), dose)
        with self.assertRaises(KeyError):
            p.key_status(42, 0, -1)
        for episode in (0, 3):                                             # another hazard group
            other = dynamic.EpisodeKey(42, 0, episode, "d7_path_following")
            with self.assertRaises(KeyError):
                p.assignment(other, 0)
            with self.assertRaises(KeyError):
                p.key_status(42, 0, episode, "d7_path_following")

    def test_offered_keys_must_form_one_full_grid(self):
        p = pool()
        self.assertEqual(J.offered_grid(p.offered), (42, "d7_path_hazard", 2, 3))
        holey = [k for k in p.offered if (k.env, k.episode) != (1, 1)]
        with self.assertRaisesRegex(ValueError, "full grid"):
            J.offered_grid(holey)
        mixed = list(p.offered) + [J.episode_key(43, 0, 0)]
        with self.assertRaisesRegex(ValueError, "ONE eval seed"):
            J.offered_grid(mixed)
        with self.assertRaises(ValueError):
            J.offered_grid([])
        self.assertEqual(J.check_grid({"num_envs": 64, "episodes": 2}), {"num_envs": 64, "episodes": 2})
        for bad in (None, {}, {"num_envs": 2}, {"num_envs": 0, "episodes": 2}, {"num_envs": True, "episodes": 2},
                    {"num_envs": 2, "episodes": 2.0}, {"num_envs": 2, "episodes": 2, "x": 1}):
            with self.assertRaises(ValueError):
                J.check_grid(bad)


class BeyondGridEpisode(unittest.TestCase):
    LEVELS = tuple(("dynamic=D%d" % dose,) for dose in range(5))

    def test_full_episode_at_every_dynamic_level(self):
        expected = expected_layout(FAST, GRID["episodes"])
        primitives, family = set(), set()
        for dose, tokens in enumerate(self.LEVELS + ((),)):
            run = beyond_grid_run(tokens)
            env, snap, record = run["env"], run["snapshot"], run["record"]
            self.assertEqual(run["codes"], [3, 3, 3, 3])                    # episodes 0..3 ended at timeout
            # the beyond-grid episode: not admitted, E5 layout of key (42, 0, 3), no intruder
            self.assertIsNotNone(snap)
            self.assertTrue(snap["not_admitted"])
            self.assertEqual(snap["exposure"], "not_admitted")
            self.assertEqual(snap["params"]["d7_exposure"], "not_admitted")
            self.assertEqual(snap["layout"].geometry_hash, expected.geometry_hash)
            self.assertNotIn((42, FAST, GRID["episodes"]), pool().geometry_hashes)   # no pool record of it
            count = expected.obstacle_count
            self.assertTrue(np.array_equal(snap["centers"][:count].numpy(), np.asarray(expected.centers)))
            self.assertTrue(np.array_equal(snap["waypoints"].numpy(), np.asarray(expected.waypoints)))
            self.assertFalse(run["d7_seen"])
            self.assertTrue(run["episode4_not_admitted"])                   # episode 4: beyond the grid too
            # the record (D16/D20 stamp, single reason "beyond_offered_grid")
            stamp = record["dynamic_hazards"]
            M.dynamic.read_stamp(stamp)                                      # E4 reader accepts it
            CERT.verify_dynamic_hazard_stamp(stamp)                          # certificate verifier too
            payload = stamp["payload"]
            self.assertEqual(payload["exposure"], "not_admitted")
            self.assertEqual(J.record_exposure(stamp), "not_admitted")
            self.assertEqual(payload["not_admitted_reasons"], [[0, "beyond_offered_grid", 0]])
            self.assertEqual(d20_reasons(payload), ["beyond_offered_grid"])
            assignment = payload["assignment"]
            self.assertEqual(assignment["key"], {"eval_seed": 42, "env": FAST, "episode": GRID["episodes"],
                                                 "hazard_group": "d7_path_hazard"})
            self.assertEqual((assignment["task"], assignment["dose"], assignment["candidate_digest"],
                              assignment["intruders"]), ("path_hazard", 0, None, []))
            runtime = payload["runtime"]
            self.assertEqual(runtime["suite_dose"], dose if tokens else 0)
            self.assertEqual(runtime["run_dose"], 0)
            self.assertEqual(runtime["termination_reason"], "timeout")
            self.assertEqual(runtime["non_exposure_reason"], "not_admitted")
            self.assertIs(runtime["assigned"], False)                        # D27: never assigned,
            self.assertEqual(runtime["itt_count"], 0)                        # outside the ITT denominator
            self.assertEqual(run["completed"]["d7_exposure"], "not_admitted")
            self.assertEqual(run["completed"]["geometry_hash"], expected.geometry_hash)
            self.assertEqual(run["cert_fields"]["dynamic_hazards"]["payload"]["exposure"], "not_admitted")
            # F0 pairing accepts the stamp; one value at every level (and with the suite absent)
            value = M.pairing.d7_primitive_hash(stamp)
            self.assertEqual(record["scenario_hashes"][M.pairing.D7_PRIMITIVE], value)
            primitives.add(value)
            family.add(tuple(record["scenario_hashes"][k] for k in ("layout", "route", "spawn_pose")))
            # header: the offered grid; per-dose counts unchanged
            header = env.journal_certificate_header(treatment="dynamic")["dynamic_hazards_header"]
            self.assertEqual(header["grid"], GRID)
            self.assertEqual(header["grid"], pool().grid)
            self.assertEqual(env.dynamic_hazard_header["grid"], GRID)
            self.assertEqual(header["per_dose_counts"], [[6, pool().keys_admitted, pool().keys_not_admitted]] * 5)
        self.assertEqual(len(primitives), 1)
        self.assertEqual(len(family), 1)

    def test_unexposed_for_the_dynamic_treatment(self):
        pairing = M.pairing
        d0 = beyond_grid_run(("dynamic=D0",))["record"]
        for dose in range(1, 5):
            dyn = beyond_grid_run(("dynamic=D%d" % dose,))["record"]
            self.assertEqual(dyn["scenario_hashes"], d0["scenario_hashes"])
            with self.assertRaisesRegex(pairing.PairingError, "no treatment difference"):
                pairing.validate_pair(d0, dyn, "dynamic")


class EnvPreflight(unittest.TestCase):
    def test_env_outside_the_grid_fails_closed(self):
        env = F.make_env(F.journal_cfg(VESSEL, num_envs=3, pool=pool()))
        with self.assertRaisesRegex(ValueError, r"outside the offered D7 grid .*env 2 is outside the pool's env range"):
            env.reset()

    def test_another_eval_seed_fails_closed(self):
        cfg = F.journal_cfg(VESSEL, pool=pool(), seed=43)
        self.assertEqual(int(cfg.seed), 43)
        env = F.make_env(cfg)
        with self.assertRaisesRegex(ValueError, r"outside the offered D7 grid .*eval seed 43 is not the pool's"):
            env.reset()

    def test_header_grid_is_required(self):
        class NoGrid:
            def __init__(self, inner):
                self.inner = inner
                self.pool_digest = inner.pool_digest

            def assignment(self, key, dose):
                return self.inner.assignment(key, dose)

            def is_admitted(self, *key):
                return self.inner.is_admitted(*key)
        with self.assertRaisesRegex(ValueError, "offered grid"):
            F.make_env(F.journal_cfg(VESSEL, pool=NoGrid(pool())))


@unittest.skipUnless(EVIDENCE.is_dir(), "F2 evidence folder not next to this repository copy")
class PinnedPools(unittest.TestCase):
    def test_pinned_pools_expose_the_64_by_2_grid(self):
        manifest = J.pinned_manifest()
        self.assertEqual((manifest["eval_seed"], manifest["envs"], manifest["episodes"]), (42, 64, 2))
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            suite = M.dynamic.DynamicSuiteConfig(enabled=False, vessel_id=vessel, dose=0)
            p = J.resolve_pool("", vessel=vessel, tier=2, suite=suite, path_variable=str(EVIDENCE))
            self.assertEqual(p.grid, {"num_envs": 64, "episodes": 2})
            self.assertEqual(p.header_fields()["grid"], {"num_envs": 64, "episodes": 2})
            self.assertEqual(p.key_status(42, 63, 2), J.BEYOND_OFFERED_GRID)
            record = p.assignment(J.episode_key(42, 63, 2), 0)
            self.assertEqual(record.reasons, ((0, "beyond_offered_grid", 0),))
            with self.assertRaises(KeyError):
                p.assignment(J.episode_key(42, 64, 0), 0)
            with self.assertRaises(KeyError):
                p.assignment(J.episode_key(41, 0, 2), 0)
            self.assertEqual([list(r) for r in p.per_dose_counts], manifest["vessels"][vessel]["per_dose_counts"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
