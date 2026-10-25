"""F3: the committed D7 admission pools (tasks/path_following/journal_d7/), CPU only (D16, D19).

The env loads ``journal_d7/d7_pool_path_following_<vessel>_seed42.d7pool.xz``
by path, pinned in ``journal_d7/POOL_MANIFEST.json`` (fail closed). This test
re-derives, for both vessels and every key of the 64-env x 2-episode grid at
seed 42, the REAL E5 journal layout and checks it is the geometry the pool
recorded for the key (D16: no redraw, no attempt streams); every admitted
key carries its D0..D4 ladder on that layout, every other key an explicit
not_admitted record with E4's reasons. It also checks the pinned manifest,
the file digest, the pool digest (recomputed from the assignments by the
loader), the provenance block, the suite identity the runtime uses, that the
admission is deterministic (serial == spawn process pool), that the env runs
the committed grid from the default path (a not-admitted row runs its E5
layout with exposure not_admitted) and that the pool bytes are reproducible.
Run: python -B test_f3_pool_files.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f3_fixtures as F  # noqa: E402

J = F.jpf
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
SEED, ENVS, EPISODES, TIER = 42, 64, 2, 2
FAMILY_POOL_BYTES_MAX = 8 * 1024 * 1024          # D19 / F_ROUND_4B item 3: <= 8 MB per family in the repo


class CommittedPoolTests(unittest.TestCase):
    def test_pinned_manifest_matches_the_committed_files(self):
        manifest, manifest_sha = J.read_pool_manifest()
        self.assertEqual(manifest_sha, hashlib.sha256(J.pool_manifest_path().read_bytes()).hexdigest())
        self.assertEqual(manifest["pool_schema"], J.POOL_SCHEMA)
        total = 0
        for vessel in VESSELS:
            name = J.pool_file_name(vessel, SEED)
            entry = manifest["pools"][name]
            path = J.POOL_DIR / name
            data = path.read_bytes()
            total += len(data)
            self.assertEqual(entry["file_sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(entry["bytes"], len(data))
            sidecar = Path(str(path) + ".sha256").read_text(encoding="ascii").split()[0]
            self.assertEqual(entry["container_sha256"], sidecar)
            pool = J.read_pool_file(path)
            self.assertEqual(entry["pool_digest"], pool.pool_digest)
            self.assertEqual((entry["keys_offered"], entry["keys_admitted"], entry["keys_not_admitted"]),
                             (pool.keys_offered, pool.keys_admitted, pool.keys_not_admitted))
            self.assertEqual(entry["per_dose_counts"], [list(x) for x in pool.per_dose_counts])
        self.assertLessEqual(total, FAMILY_POOL_BYTES_MAX)

    def test_pools_cover_the_grid_on_the_e5_layouts(self):
        for vessel in VESSELS:
            path = J.POOL_DIR / J.pool_file_name(vessel, SEED)
            self.assertTrue(path.is_file(), path)
            J._POOL_CACHE.clear()
            pool = J.read_pool_file(path)
            sidecar = Path(str(path) + ".sha256").read_text(encoding="ascii").split()[0]
            self.assertEqual(pool.sha256, sidecar)
            meta = pool.meta
            self.assertEqual((meta["task"], meta["vessel"], meta["seed"], meta["num_envs"], meta["episodes"],
                              meta["geometry_tier"]), ("path_following", vessel, SEED, ENVS, EPISODES, TIER))
            self.assertEqual(meta["geometry_version_digest"], F.plans.GEOMETRY_VERSION_DIGEST)
            self.assertEqual(meta["suite"], J.suite_identity(J.admission_suite(vessel)))
            self.assertNotIn("layout_attempts", meta)
            self.assertNotIn("layout_retry_history", meta)
            for tokens in ((), ("dynamic=D2",), ("dynamic=D4", "waves=D1")):
                runtime = F.journal_config.resolve_journal_conditions(presets=list(tokens), seed=SEED)
                self.assertEqual(J.suite_identity(runtime.dynamic_config(vessel)), meta["suite"], tokens)
            self.assertEqual(pool.keys_offered, ENVS * EPISODES)
            self.assertEqual(pool.keys_admitted + pool.keys_not_admitted, ENVS * EPISODES)
            self.assertEqual([list(x) for x in pool.per_dose_counts],
                             [[ENVS * EPISODES, pool.keys_admitted, pool.keys_not_admitted]] * 5)
            not_admitted = {(e, k): reasons for e, k, reasons in meta["not_admitted"]}
            self.assertEqual(len(not_admitted), pool.keys_not_admitted)
            for env in range(ENVS):
                for episode in range(EPISODES):
                    layout = J.journal_layout(vessel, TIER, SEED, env, episode)
                    self.assertEqual(pool.geometry(env, episode), layout.geometry_hash)
                    key = F.dynamic.EpisodeKey(SEED, env, episode, J.HAZARD_GROUP)
                    for dose in range(5):
                        assignment = pool.assignment(key, dose)
                        self.assertEqual(J.assignment_geometry(assignment), layout.geometry_hash)
                        if (env, episode) in not_admitted:
                            self.assertIsInstance(assignment, J.NotAdmittedAssignment)
                            self.assertEqual(assignment.requested_dose, dose)
                            self.assertEqual([list(x) for x in assignment.rejections], not_admitted[(env, episode)])
                            self.assertTrue(J.primary_reasons(assignment.rejections))
                            self.assertNotIn(((key, dose)), pool.assignments)
                            continue
                        self.assertEqual(assignment.candidate.request.encounter.key, key)
                        self.assertEqual(assignment.candidate.request.encounter.dose, dose)
                        self.assertEqual(assignment.candidate.sensor.max_range_m,
                                         layout.params["ray_sensor_max_range_m"])
            with self.assertRaises(KeyError):                    # outside the grid: preflight, never a redraw
                pool.assignment(F.dynamic.EpisodeKey(SEED, ENVS, 0, J.HAZARD_GROUP), 0)
            print("%s: %d admitted + %d not_admitted keys x 5 doses verified against the E5 layouts "
                  "(file sha256 %s, %d bytes)" % (vessel, pool.keys_admitted, pool.keys_not_admitted,
                                                  pool.sha256[:16], path.stat().st_size))

    def test_admission_is_deterministic_serial_equals_process_pool(self):
        committed = J.read_pool_file(J.POOL_DIR / J.pool_file_name("blueboat_v2", SEED))
        items = [(0, 0), (1, 0), (14, 1)]                        # (14, 1) is a not-admitted key (G7 pool)
        self.assertFalse(committed.is_admitted(14, 1))
        jobs = []
        for env, episode in items:
            layout = J.journal_layout("blueboat_v2", TIER, SEED, env, episode)
            jobs.append((J.d7_task_context(layout, "blueboat_v2"),
                         F.dynamic.EpisodeKey(SEED, env, episode, J.HAZARD_GROUP), J.admission_suite("blueboat_v2")))
        serial = F.dynamic.admit_grid(jobs, workers=1)
        again = F.dynamic.admit_grid(jobs, workers=1)
        parallel = F.dynamic.admit_grid(jobs, workers=2)
        self.assertEqual(serial.pool_digest, again.pool_digest)
        self.assertEqual(serial.pool_digest, parallel.pool_digest)
        derived = J.derive_not_admitted(serial, [job[1] for job in jobs])
        self.assertEqual(set(derived), {(14, 1)})
        self.assertEqual(derived[(14, 1)], committed.assignment(jobs[2][1], 0).rejections)
        for env, episode in items[:2]:
            key = F.dynamic.EpisodeKey(SEED, env, episode, J.HAZARD_GROUP)
            for dose in range(5):
                self.assertEqual(committed.assignment(key, dose).candidate_digest,
                                 serial.assignment(key, dose).candidate_digest)

    def test_env_runs_the_committed_grid_from_the_default_path(self):
        _, _, templates = F.env_modules()
        manifest_sha = J.read_pool_manifest()[1]
        for vessel, cls in (("blueboat_v2", templates.PathFollowingBlueBoatJournalEnvCfg),
                            ("wamv_vrx_v2_opt1", templates.PathFollowingWAMVJournalEnvCfg)):
            cfg = cls()
            cfg.scene.num_envs = ENVS
            cfg.visual.enable_water = cfg.visual.enable_waypoint_markers = cfg.visual.enable_path_line = False
            F.journal_config.apply_journal_conditions(
                cfg, F.journal_config.resolve_journal_conditions(presets=["dynamic=D3"], seed=SEED))
            self.assertFalse(hasattr(cfg, "journal_d7_pool_path"))          # the default file is used
            env = F.make_env(cfg)
            env.reset()
            for k in range(2):
                if k == 0:
                    env.robot.data.root_link_pose_w[14, 0] += 5000.0   # row 14 -> episode 1, not admitted (both)
                env.step(F.seek_actions(env))
            pool = env._d7_pool
            episodes = env._scenario.episode_indices().tolist()
            self.assertEqual(episodes[14], 1)
            expected = [J.EXPOSURE_ASSIGNED if pool.is_admitted(i, episodes[i]) else J.EXPOSURE_NOT_ADMITTED
                        for i in range(ENVS)]
            self.assertEqual(expected[14], J.EXPOSURE_NOT_ADMITTED)
            self.assertEqual(env._journal_d7_exposure, expected)
            self.assertEqual(env._d7_bridge.rows.keys(), set(range(ENVS)))
            self.assertEqual(env.journal_path_record(14)["dynamic_hazards_exposure"], J.EXPOSURE_ASSIGNED)
            for i in range(ENVS):
                self.assertEqual(env._d7_bridge.exposure(i), expected[i])
                self.assertEqual(env._journal_layouts[i].geometry_hash,
                                 J.journal_layout(vessel, TIER, SEED, i, episodes[i]).geometry_hash)   # D16
                if expected[i] == J.EXPOSURE_NOT_ADMITTED:
                    self.assertFalse(bool(env._d7_physical[i].any()) or bool(env._d7_visible[i].any()))
            header = env.journal_certificate_header()["dynamic_hazards_header"]
            self.assertEqual(header["pool_file"]["sha256"], pool.sha256)
            self.assertEqual(header["pool_file"]["source"], "pinned")
            self.assertTrue(header["pool_pinned"])
            self.assertEqual(header["grid"], {"num_envs": ENVS, "episodes": EPISODES})   # D24 (c)
            self.assertEqual(header["pool_manifest_sha256"], manifest_sha)
            self.assertTrue(header["pool_file"]["path"].endswith(J.pool_file_name(vessel, SEED)))
            self.assertEqual(header["per_dose_counts"], [list(x) for x in pool.per_dose_counts])
            self.assertEqual(header["per_dose_counts_fields"], ["offered", "admitted", "not_admitted"])
            self.assertEqual(header["keys_not_admitted"], pool.keys_not_admitted)
            self.assertEqual(len(header["not_admitted_keys"]), pool.keys_not_admitted)

    def test_file_bytes_are_reproducible(self):
        path = J.POOL_DIR / J.pool_file_name("blueboat_v2", SEED)
        pool = J.read_pool_file(path)
        tmp = Path(F._POOL_DIR) / "reproduce" / path.name
        J.write_pool_file(tmp, pool.pool, pool.offered, pool.meta)
        self.assertEqual(hashlib.sha256(tmp.read_bytes()).hexdigest(), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_unpinned_or_mismatched_default_pool_fails_closed(self):
        import os
        import shutil
        import tempfile
        cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=1)
        del cfg.journal_d7_pool_path
        folder = Path(tempfile.mkdtemp(prefix="f3_pooldir_"))
        old = os.environ.get(J.POOL_DIR_ENV)
        try:
            os.environ[J.POOL_DIR_ENV] = str(folder)
            name = J.pool_file_name("blueboat_v2", SEED)
            small = F.pool_file("blueboat_v2", 2, 1)                # sidecar-valid, but NOT the pinned pool
            shutil.copy(small, folder / name)
            shutil.copy(str(small) + ".sha256", str(folder / name) + ".sha256")
            J._POOL_CACHE.clear()
            with self.assertRaises(ValueError):
                J.load_journal_d7_pool(cfg)
            shutil.copy(J.POOL_DIR / name, folder / name)           # the pinned file via the runtime variable
            shutil.copy(str(J.POOL_DIR / name) + ".sha256", str(folder / name) + ".sha256")
            pool, source = J.load_journal_d7_pool(cfg)
            self.assertEqual(source, "env")
            self.assertEqual(pool.pool_digest, J.read_pool_manifest()[0]["pools"][name]["pool_digest"])
            cfg.seed = 7                                             # a seed that is not pinned
            with self.assertRaises(ValueError) as caught:
                J.load_journal_d7_pool(cfg)
            self.assertIn("not pinned", str(caught.exception))
        finally:
            if old is None:
                os.environ.pop(J.POOL_DIR_ENV, None)
            else:
                os.environ[J.POOL_DIR_ENV] = old
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
