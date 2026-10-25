"""F3: the REAL PathFollowingEnv journal wiring on a stubbed Isaac Lab (CPU only).

Covers the F-family brief item 8: legacy byte-identity (env methods and the
legacy cfg), EXACT CALL ORDER through the real HydroV2Runtime, schema-3
condition layer and E4 bridge, the submitted wrench, D8 water level, zero-dose
identity, reset-row isolation, ALL SUITES ON at once (both vessels) with
stamps, D9 codes 6-9 published, D11 pairing hashes, the journal observation
(native 7 from the position estimate + 36 rays, no velocity channel, velocity
doses rejected), the point-return ray proxy, spawn at the COM, and the D7
pool preflight (fail closed).
Run: python -B test_f3_env_journal.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import lzma
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import f3_fixtures as F  # noqa: E402

J = F.jpf
ENV_FILE = F.TASKS / "path_following" / "path_following_env.py"
CFG_FILE = F.TASKS / "path_following" / "path_following_env_cfg.py"
ALL_SUITES = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "dynamic=D2")
ZERO_SUITES = ("waves=D0", "wind=D0", "current=D0", "actuator=D0", "perception=D0", "dynamic=D0")
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")


def class_methods(path, class_name):
    text = path.read_text(encoding="utf-8")
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {n.name: ast.get_source_segment(text, n) for n in node.body if isinstance(n, ast.FunctionDef)}
    raise AssertionError(class_name + " not found in " + str(path))


def strip_journal_branch(source):
    """Remove the one journal branch (from its `if` line through its `return`) from a legacy method."""
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if 'getattr(' in line and '"journal_contract_version"' in line)
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > indent):
        end += 1
    return "\n".join(lines[:start] + lines[end:])


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_steps(env, steps, actions=None, before_step=None):
    applied, observations, codes = [], [], []
    for k in range(steps):
        if before_step is not None:
            before_step(k, env)
        a = actions(env) if callable(actions) else F.seek_actions(env)
        obs, _, _, _, _ = env.step(a)
        applied.append(env.robot.applied[-1])
        observations.append(obs["policy"].clone())
        codes.append(env.termination_reason.clone())
    return applied, observations, codes


def force_oob(row, step):
    def hook(k, env):
        if k == step:
            env.robot.data.root_link_pose_w[row, 0] += 5000.0
    return hook


class LegacyIdentityTests(unittest.TestCase):
    def test_env_methods_byte_identical_to_fbase(self):
        methods = class_methods(ENV_FILE, "PathFollowingEnv")
        for name, expected in F.FBASE_METHOD_SHA256.items():
            source = methods[name]
            if name == "__init__":
                source = strip_journal_branch(source)
            self.assertEqual(sha(source), expected, name)
        if (F.FBASE / "tasks/path_following/path_following_env.py").is_file():
            base = class_methods(F.FBASE / "tasks/path_following/path_following_env.py", "PathFollowingEnv")
            self.assertEqual(set(base), set(methods))
            for name in base:
                source = strip_journal_branch(methods[name]) if name == "__init__" else methods[name]
                self.assertEqual(source, base[name], name)

    def test_legacy_cfg_post_init_and_objects_unchanged(self):
        new = class_methods(CFG_FILE, "PathFollowingEnvCfg")["__post_init__"]
        base_file = F.FBASE / "tasks/path_following/path_following_env_cfg.py"
        if not base_file.is_file():
            self.skipTest("F-base not present on this machine")
        base = class_methods(base_file, "PathFollowingEnvCfg")["__post_init__"]
        self.assertEqual(strip_journal_branch(new), base)
        cfg_module, _, _ = F.env_modules()
        F.load("_shared.vehicles")
        fbase_root = "_f3fbase"
        if fbase_root not in sys.modules:
            F._namespace(fbase_root, F.FBASE / "tasks")
            for folder in sorted((F.FBASE / "tasks").iterdir()):
                if folder.is_dir() and folder.name != "__pycache__":
                    F._namespace(fbase_root + "." + folder.name, folder)
        import importlib
        base_module = importlib.import_module(fbase_root + ".path_following.path_following_env_cfg")

        def plain(value, depth=0):
            if depth > 8:
                return repr(value)
            if hasattr(value, "__dict__") and not isinstance(value, type):
                return {k: plain(v, depth + 1) for k, v in sorted(vars(value).items())}
            if isinstance(value, (list, tuple)):
                return [plain(v, depth + 1) for v in value]
            return value if isinstance(value, (int, float, str, bool, type(None))) else repr(value)

        for name in ("PathFollowingEnvCfg", "PathFollowingBlueBoatEnvCfg", "PathFollowingBlueBoatWaveEnvCfg"):
            a, b = plain(getattr(cfg_module, name)()), plain(getattr(base_module, name)())
            self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str),
                             name)


class CallOrderTests(unittest.TestCase):
    def test_exact_call_order_real_runtime_layer_bridge(self):
        for vessel in VESSELS:
            trace = []
            cfg = F.journal_cfg(vessel, num_envs=2, episodes=2, tokens=("waves=D2",))
            env = F.make_env(cfg, trace)
            env.reset()
            levels, elevations = [], []
            original = env._journal_observe_physics

            def observe(robot_data, env_origins, water_level_local, _orig=original):
                levels.append(torch.as_tensor(water_level_local).clone())
                return _orig(robot_data, env_origins, water_level_local)
            env._journal_observe_physics = observe
            for k in range(3):
                if k == 2:
                    env.robot.data.root_link_pose_w[0, 0] += 5000.0
                env.step(F.seek_actions(env))
                step = env.journal_last_step
                forces, torques = env.robot.applied[-1]
                self.assertTrue(torch.equal(forces[:, 0], step.force_actor.to(forces.dtype)))
                self.assertTrue(torch.equal(torques[:, 0], step.torque_actor.to(torques.dtype)))
                elevations.append(step.waves.elevation.clone())
            expected = ["setup", "reset", "estimates"]
            for k in range(3):
                expected += ["pre_physics", "substep", "substep", "pre_dones", "dones"]
                if k == 2:
                    expected.append("reset")
                expected.append("estimates")
            self.assertEqual(trace, expected, vessel)
            self.assertEqual(env.call_log.count("physics"), 6)
            # D8: first substep of the episode sees the constant surface, later ones the previous elevation
            self.assertTrue(torch.equal(levels[0], torch.zeros_like(levels[0])))
            self.assertTrue(torch.equal(levels[2], elevations[0].to(levels[2].dtype)))
            self.assertEqual(env.termination_reason.tolist(), [4, 0])
            self.assertEqual(int(env._hydro_v2.ticks[0]), 0)
            self.assertEqual(int(env._hydro_v2.ticks[1]), 6)

    def test_spawn_is_the_com_at_equilibrium_with_layout_heading(self):
        cfg = F.journal_cfg("wamv_vrx_v2_opt1", num_envs=2, episodes=2)
        env = F.make_env(cfg)
        env.reset()
        ids, state = env.robot.writes[-1]
        self.assertEqual(ids.tolist(), [0, 1])
        z_eq = env.journal_vehicle.hydro_profile.hydro.equilibrium_com_z_m
        origins = env.scene.env_origins
        for i in range(2):
            layout = env._journal_layouts[i]
            self.assertAlmostEqual(float(state[i, 2] - origins[i, 2]), z_eq, places=6)
            self.assertTrue(torch.allclose(state[i, :2] - origins[i, :2],
                                           torch.as_tensor(layout.start, dtype=torch.float32)))
            yaw = J.yaw_from_quat(state[i, 3:7].to(torch.float64))
            self.assertAlmostEqual(float(yaw), float(math.atan2(math.sin(layout.start_heading_rad),
                                                                math.cos(layout.start_heading_rad))), places=6)
            self.assertTrue(torch.equal(state[i, 7:], torch.zeros(6)))
        self.assertEqual(env.journal_vehicle.com_offset_m, (0.0, 0.0, 0.0))


class SuiteTests(unittest.TestCase):
    def test_zero_dose_identity(self):
        runs = []
        for tokens in ((), ZERO_SUITES):
            cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=2, tokens=tokens)
            env = F.make_env(cfg)
            env.reset()
            runs.append(run_steps(env, 25, before_step=force_oob(1, 20)))
        (fa, oa, ca), (fb, ob, cb) = runs
        for k in range(25):
            self.assertTrue(torch.equal(fa[k][0], fb[k][0]) and torch.equal(fa[k][1], fb[k][1]), k)
            self.assertTrue(torch.equal(oa[k], ob[k]), k)
            self.assertTrue(torch.equal(ca[k], cb[k]), k)

    def test_all_suites_on_both_vessels_run_and_stamp(self):
        for vessel in VESSELS:
            cfg = F.journal_cfg(vessel, num_envs=4, episodes=2, tokens=ALL_SUITES)
            env = F.make_env(cfg)
            env.reset()

            def hook(k, env):
                if k == 2:
                    env.robot.data.root_link_pose_w[0, 0] += 5000.0                             # 4 out of bounds
                    env.robot.data.root_link_pose_w[1, 3:7] = torch.tensor([0.5, 0.8660254, 0.0, 0.0])  # 6 capsize
                    env.robot.data.root_link_pose_w[2, 2] -= 50.0                               # 7 sinking
                    env.robot.data.root_com_vel_w[3, 0] = float("nan")                          # 9 non-finite
            run_steps(env, 3, before_step=hook)
            self.assertEqual(env.termination_reason.tolist(), [4, 6, 7, 9], vessel)
            self.assertEqual(env.extras["termination_reason"].tolist(), [4, 6, 7, 9])
            header = env.journal_certificate_header()
            self.assertEqual(header["conditions"]["schema"], "usvbench.conditions/3")
            enabled = {k: v["enabled"] for k, v in header["conditions"]["parameters"].items()
                       if isinstance(v, dict) and "enabled" in v}
            self.assertTrue(all(enabled.values()), enabled)
            self.assertEqual(env._d7_suite.dose, 2)
            for i in range(4):
                record = env.journal_path_record(i)
                self.assertEqual(record["conditions"]["schema"], "usvbench.conditions/3")
                F.dynamic.read_stamp(record["dynamic_hazards"])
                self.assertEqual(record["dynamic_hazards"]["payload"]["assignment"]["dose"], 2)
                self.assertEqual(sorted(record["scenario_hashes"]), list(env.journal_pairing_inventory))
                recomputed = F.pairing.condition_primitive_hashes(record["conditions"])
                for name, value in recomputed.items():
                    self.assertEqual(record["scenario_hashes"][name], value)
                self.assertEqual(record["scenario_hashes"]["d7_assignment"],
                                 F.pairing.d7_primitive_hash(record["dynamic_hazards"]))
                for name in ("route", "spawn_pose"):
                    self.assertEqual(record["scenario_hashes"][name],
                                     F.pairing.primitive_hash(name, record["family"]["primitives"][name]))
                self.assertEqual(record["contract"]["termination_reason"], [4, 6, 7, 9][i])
                self.assertEqual(record["conditions"]["runtime"]["estimates"]["checked_groups"], ["pos", "ray", "vel"])

    def test_two_wave_levels_pair_cleanly(self):
        records = []
        for tokens in (("waves=D1", "wind=D2"), ("waves=D3", "wind=D2")):
            cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=2, tokens=tokens)
            env = F.make_env(cfg)
            env.reset()
            run_steps(env, 4, before_step=force_oob(0, 3))
            records.append(env.journal_path_record(0))
        summary = F.pairing.validate_pair(records[0], records[1], "waves")
        self.assertEqual(summary["changed_primitives"], ["waves"])
        self.assertIn("route", summary["unchanged_primitives"])
        with self.assertRaises(F.pairing.PairingError):
            F.pairing.validate_pair(records[0], records[1], "wind")

    def test_reset_row_isolation(self):
        tokens = ("waves=D2", "wind=D1", "perception=D2", "actuator=D1", "dynamic=D1")
        runs = []
        for hook in (None, force_oob(0, 5)):
            cfg = F.journal_cfg("blueboat_v2", num_envs=3, episodes=2, tokens=tokens)
            env = F.make_env(cfg)
            env.reset()
            runs.append((run_steps(env, 14, before_step=hook), env))
        (fa, oa, ca), env_a = runs[0]
        (fb, ob, cb), env_b = runs[1]
        self.assertEqual(int(cb[5][0]), 4)
        for k in range(14):
            for row in (1, 2):
                self.assertTrue(torch.equal(fa[k][0][row], fb[k][0][row]) and torch.equal(fa[k][1][row], fb[k][1][row]))
                self.assertTrue(torch.equal(oa[k][row], ob[k][row]), (k, row))
        for row in (1, 2):
            for name in ("full_path", "xte_integral", "next_gate", "previous_xy"):
                self.assertTrue(torch.equal(getattr(env_a._journal_oracle, name)[row],
                                            getattr(env_b._journal_oracle, name)[row]), name)
            self.assertEqual(env_a._conditions.estimate_counts(row), env_b._conditions.estimate_counts(row))
        self.assertEqual(int(env_b._scenario.episode_indices()[0]), 1)
        self.assertEqual(int(env_b._scenario.episode_indices()[1]), 0)


class ObservationTests(unittest.TestCase):
    def test_layout_is_native7_from_the_estimate_plus_36_rays(self):
        cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=2, tokens=("perception.bias=D4",))
        env = F.make_env(cfg)
        obs, _ = env.reset()
        self.assertEqual(tuple(obs["policy"].shape), (2, 43))
        self.assertEqual(cfg.observation_space, 43)
        run_steps(env, 3)
        policy = env.step(F.seek_actions(env))[0]["policy"]
        est = env.journal_estimate
        truth = env.robot.data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]
        self.assertGreater(float((est.position - truth).norm(dim=-1).min()), 0.1)   # D4 bias moves the estimate
        forward = J.quat_apply(env.robot.data.root_link_quat_w, env.forward_vec)[:, :2]
        forward = forward / forward.norm(dim=-1, keepdim=True)
        target = env.waypoints[torch.arange(2), (env._journal_oracle.next_gate - 1).clamp(0, 3)]
        rpos = target - est.position
        dist = rpos.norm(dim=-1, keepdim=True)
        direction = rpos / dist
        dot = (forward * direction).sum(-1)
        self.assertTrue(torch.equal(policy[:, 0], dot))
        self.assertTrue(torch.equal(policy[:, 2], (dist / (20.0 * env._journal_length_scale))[:, 0]))
        self.assertTrue(torch.equal(policy[:, 7:], est.rays / cfg.ray_max_range_m))
        self.assertEqual(tuple(est.velocity.shape), (2, 3))       # estimated for MPPI/classical, never observed

    def test_velocity_doses_rejected_no_velocity_channel(self):
        contract = F.observation_groups.lookup("path_following", "journal_nav7_ray36")
        self.assertNotIn("vel", contract.groups)
        with self.assertRaises(ValueError) as caught:
            F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, episodes=2, tokens=("perception.bundle=D2",)))
        self.assertIn("vel", str(caught.exception))
        cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=2)
        resolved = F.journal_config.resolve_journal_conditions(presets=["perception.bundle=D2"], seed=F.SEED)
        with self.assertRaises(F.journal_config.SuiteConflict) as conflict:
            F.journal_config.admit_journal_conditions(cfg, resolved)
        self.assertEqual(conflict.exception.code, "C4")

    def test_point_return_ray_proxy(self):
        com = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        yaw = torch.tensor([0.0, math.pi / 2, 0.0, 0.0], dtype=torch.float64)
        centers = torch.tensor([[[10.0, 0.0]], [[10.0, 0.0]], [[0.2, 0.1]], [[1.0, 6.0]]], dtype=torch.float64)
        radii = torch.tensor([[1.0], [1.0], [0.5], [1.0]], dtype=torch.float64)
        visible = torch.tensor([[True], [True], [True], [False]])
        rays = J.point_return_rays(com, yaw, centers, radii, visible, 36, 30.0)
        self.assertEqual(float(rays[0, 0]), 9.0)                  # dead ahead: nearest surface point
        self.assertEqual(int((rays[0] < 30.0).sum()), 1)          # one return per object (centre ray only)
        self.assertEqual(int(rays[1].argmin()), 27)               # 270 deg in the body frame (object to starboard)
        self.assertEqual(float(rays[2].min()), 0.0)               # observer inside the disk
        self.assertTrue(torch.equal(rays[3], torch.full((36,), 30.0, dtype=torch.float64)))   # invisible
        centers2 = torch.tensor([[[5.0 * math.cos(math.radians(5.0)), 5.0 * math.sin(math.radians(5.0))]]],
                                dtype=torch.float64)
        tie = J.point_return_rays(com[:1], yaw[:1], centers2, radii[:1], visible[:1], 36, 30.0)
        self.assertEqual(int(tie[0].argmin()), 1)                 # bearing exactly between rays 0 and 1 -> 1

    def test_d7_object_reaches_only_the_ray_block(self):
        cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=2, tokens=("dynamic=D4",))
        env = F.make_env(cfg)
        env.reset()
        seen = False
        for k in range(3600):
            a = F.seek_actions(env)
            speed = env.robot.data.root_com_vel_w[:, :2].norm(dim=-1)
            a[:, 0] = (0.3 + 2.0 * (1.0 - speed)).clamp(-1.0, 1.0)      # hold ~1 m/s (the nominal reference)
            obs = env.step(a)[0]["policy"]
            if bool(env._d7_visible[0, 0]) and int(env.episode_length_buf[0]) > 0:
                self.assertLess(float(obs[0, 7:].min()), 1.0)
                expected = J.point_return_rays(
                    (env.robot.data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]).to(torch.float64),
                    J.yaw_from_quat(env.robot.data.root_link_pose_w[:, 3:7].to(torch.float64)),
                    env._d7_centers_local, env._d7_radii, env._d7_visible, 36, cfg.ray_max_range_m).float()
                self.assertTrue(torch.equal(env.journal_estimate.rays[0], expected[0]))
                seen = True
                break
        self.assertTrue(seen, "the D4 intruder should become visible to the 1 m/s driver")
        self.assertFalse(hasattr(env, "obstacle_active"))


class PoolPreflightTests(unittest.TestCase):
    def test_pool_is_required_and_verified(self):
        cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=1)
        good = Path(cfg.journal_d7_pool_path)
        tmp = Path(tempfile.mkdtemp(prefix="f3_badpool_"))
        try:
            cfg.journal_d7_pool_path = str(tmp / "absent.json.xz")
            with self.assertRaises(FileNotFoundError):
                F.make_env(cfg)
            cfg.journal_d7_pool_path = str(good)
            cfg.journal_d7_pool_sha256 = "0" * 64
            with self.assertRaises(ValueError):
                F.make_env(cfg)
            del cfg.journal_d7_pool_sha256
            data, blob = J.split_pool_container(lzma.decompress(good.read_bytes()))
            start = data.index('["D","Intruder",["') + len('["D","Intruder",["')
            first = data.index(",", data.index('"', start) + 1) + 1        # after the path_type string
            last = data.index(",", first)
            x0 = float(data[first:last])
            text = data[:first] + repr(x0 + 1.0e-6) + data[last:]          # move one intruder by 1 um
            json.loads(text)
            container = J.join_pool_container(text, blob)
            bad = tmp / good.name
            bad.write_bytes(lzma.compress(container))
            Path(str(bad) + ".sha256").write_text(hashlib.sha256(container).hexdigest() + "\n",
                                                  encoding="ascii")
            J._POOL_CACHE.clear()
            cfg.journal_d7_pool_path = str(bad)
            with self.assertRaises((ValueError, TypeError)):
                F.make_env(cfg)
            flipped = bytearray(blob)
            flipped[len(flipped) // 2] ^= 0x01                             # one bit of one witness trajectory
            container = J.join_pool_container(data, flipped)
            bad.write_bytes(lzma.compress(container))
            Path(str(bad) + ".sha256").write_text(hashlib.sha256(container).hexdigest() + "\n",
                                                  encoding="ascii")
            J._POOL_CACHE.clear()
            with self.assertRaises((ValueError, TypeError)):
                F.make_env(cfg)
            cfg.journal_d7_pool_path = str(good)
            env = F.make_env(cfg)
            env.reset()
            env.robot.data.root_link_pose_w[0, 0] += 5000.0
            # CONTEXT_R4 D24 (a) replaces the round-4 fail-closed rule for this key: episode 1 lies beyond the
            # 2 x 1 grid, so it runs D0 as a not-admitted key "beyond_offered_grid" (test_f3_d24_beyond_grid.py)
            env.step(F.seek_actions(env))
            self.assertEqual(int(env._scenario.episode_indices()[0]), 1)
            self.assertEqual(env._journal_d7_exposure[0], J.EXPOSURE_NOT_ADMITTED)
            # D24 (b): an env outside the grid's env range stays a preflight error (fail closed)
            cfg.scene.num_envs = 3
            env = F.make_env(cfg)
            with self.assertRaises(ValueError) as beyond:
                env.reset()
            self.assertIn("no admitted D7", str(beyond.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_opt_in_online_admission_beyond_the_committed_grid(self):
        cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=1, tokens=("dynamic=D2",))
        cfg.journal_d7_online_admission = True
        env = F.make_env(cfg)
        env.reset()
        committed = env.dynamic_hazard_header["pool_digest"]
        env.robot.data.root_link_pose_w[0, 0] += 5000.0
        env.step(F.seek_actions(env))                          # env 0 -> episode 1: outside the grid
        self.assertEqual(int(env._scenario.episode_indices()[0]), 1)
        keys = env.dynamic_hazard_header["online_admitted_keys"]
        self.assertEqual([(k["env"], k["episode"]) for k in keys], [(0, 1)])
        self.assertEqual(keys[0]["exposure"], J.EXPOSURE_ASSIGNED)
        self.assertNotEqual(keys[0]["pool_digest"], committed)
        self.assertTrue(env.journal_certificate_header()["dynamic_hazards_header"]["online_admission"])
        env.robot.data.root_link_pose_w[0, 0] += 5000.0
        env.step(F.seek_actions(env))
        stamp = env.journal_path_record(0)["dynamic_hazards"]["payload"]["assignment"]
        self.assertEqual((stamp["key"]["episode"], stamp["dose"]), (1, 2))
        self.assertEqual(stamp["pool_digest"], keys[0]["pool_digest"])   # never the committed digest

    def test_seed_and_suite_must_match_the_pool(self):
        cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=1, seed=43)
        cfg.journal_d7_pool_path = str(F.pool_file("blueboat_v2", 2, 1, seed=42))
        with self.assertRaises(ValueError) as caught:
            F.make_env(cfg)
        self.assertIn("seed", str(caught.exception))
        cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=1,
                            tokens=("dynamic=D1", "dynamic.intruder_speed_mps=0.7"))
        with self.assertRaises(ValueError) as caught:
            F.make_env(cfg)
        self.assertIn("suite", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
