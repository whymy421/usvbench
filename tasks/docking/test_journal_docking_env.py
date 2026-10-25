"""F5: the REAL DockingEnv journal branch on CPU with a stubbed Isaac Lab (never imports Isaac Lab).

The real modules run: tasks/docking/docking_env.py (DockingEnv, journal
branch), docking_env_cfg.py and E5's journal_runtime_cfg_v2 templates
(DockingBlueBoatJournalEnvCfg / DockingWAMVJournalEnvCfg), F0's
JournalEnvMixin, E2's HydroV2Runtime, E3's schema-3 condition layer and
E4's HazardBridge with a REAL D7 pool (tools_f5/admit_d7_pool.py logic on
a 4-env x 2-episode grid at seed 42: BlueBoat keys (1, 0), (1, 1), (2, 1)
are NOT admitted by E4 and run the D16 empty D0 row). Only isaaclab / gymnasium are stubbed: a
minimal DirectRLEnv reproducing Isaac Lab's step order (pre_physics,
decimation x (apply_action, physics), dones, rewards, reset of done rows,
observations) around a toy 6-DOF rigid-body integrator (a plumbing
stand-in, not a physics claim).

Covered (F_FAMILY_COMMON item 8): legacy byte identity of the env methods vs
F-base; zero-dose identity; substep call order through the real
runtime/layer/bridge; contract parity (scalar E5 oracle vs the env's tensor
twin on recorded env traces); reset-row isolation; ALL SUITES ON at D2 +
dynamic D2 runs 3 control steps and stamps; pairing hashes present and a
clean single-treatment pair; both vessels; keyed RNG streams (C1 finding 29);
D2 item 14 level table; D7 publication / swept contact / B08 finish before
reset; COM spawn height and the zero COM offset check; round 4b (CONTEXT_R4
D16): not-admitted keys run full episodes at every dynamic level with
exposure "not_admitted", pair with the same key at D0 of every other suite,
keep the E5 layout of their key, and the pinned repo pools + manifest verify;
round T (CONTEXT_R4 D22/D24): the header per-dose counts are [offered,
admitted, not_admitted] rows and the header publishes the offered grid; a task
episode beyond the grid resets and runs a full episode at every dynamic level,
stamps exposure "not_admitted" with the single reason "beyond_offered_grid",
keeps the E5 layout of its key and pairs; keys outside the env range or with
another seed stay preflight errors.

Run: python -B tasks/docking/test_journal_docking_env.py
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import math
import os
from pathlib import Path
import sys
import tempfile
import types
from types import ModuleType, SimpleNamespace
import unittest

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FBASE = Path(os.environ.get("F5_FBASE_REPO", r"D:\claude code\codex_tasks\20260930_fbase\work\repo"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

def _d7_shown(env, row):
    """G23c L2 (rule S2): a row's D7 intruder is physical exactly when its runtime rule has revealed it
    (hidden before the reveal; rows without a runtime rule: physical when they carry an intruder)."""
    episode = env._d7_bridge.adapter.episodes.get(int(row))
    if episode is None:
        return False
    if episode.release is None:
        return bool(episode.objects)
    return bool(episode.release.released_at(episode.last_time_s))



# ------------------------------------------------------------------ stubs --
def _install_stubs():
    """isaaclab / gymnasium stand-ins (pattern: scripts/tests_journal_glue/glue_fixtures.py)."""
    if "tasks" not in sys.modules:
        pkg = ModuleType("tasks")
        pkg.__path__ = [str(REPO / "tasks")]
        sys.modules["tasks"] = pkg
    for folder in (REPO / "tasks").iterdir():
        name = "tasks." + folder.name
        if folder.is_dir() and folder.name not in ("_shared", "__pycache__") and name not in sys.modules:
            pkg = ModuleType(name)
            pkg.__path__ = [str(folder)]
            sys.modules[name] = pkg

    class Cfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def func(self, *args, **kwargs):
            return None

    class WithState(Cfg):
        InitialStateCfg = Cfg

    def configclass(cls):
        def __init__(self, **kwargs):
            for klass in reversed(cls.__mro__):
                for name, value in vars(klass).items():
                    if name.startswith("__") or isinstance(value, (staticmethod, classmethod, property)):
                        continue
                    if callable(value) and not isinstance(value, type):
                        continue
                    setattr(self, name, copy.deepcopy(value))
            for name, value in kwargs.items():
                setattr(self, name, value)
            post = getattr(self, "__post_init__", None)
            if post is not None:
                post()
        cls.__init__ = __init__
        return cls

    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        mod.__getattr__ = lambda attr: Cfg
        sys.modules[name] = mod
        return mod

    root = module("isaaclab")
    root.sim = module("isaaclab.sim", SimulationCfg=Cfg)
    module("isaaclab.assets", ArticulationCfg=WithState, RigidObjectCfg=WithState, RigidObject=FakeRobot)
    module("isaaclab.envs", DirectRLEnvCfg=type("DirectRLEnvCfg", (), {}), ViewerCfg=Cfg, DirectRLEnv=FakeDirectRLEnv)
    module("isaaclab.scene", InteractiveSceneCfg=Cfg)
    utils = module("isaaclab.utils", configclass=configclass)
    utils.math = module("isaaclab.utils.math")
    root.utils = utils
    gym = module("gymnasium")
    gym.spaces = SimpleNamespace(Box=Cfg)
    gym.register = lambda **kwargs: None


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)


def rotation(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
        torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
        torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)


class FakeRobot:
    """RigidObject stand-in: root-link pose, COM velocity, root-state write, toy integrator."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_bodies = 1
        self.data = SimpleNamespace()
        self.applied = []
        self.writes = []

    def bind(self, n, mass, inertia, gravity):
        d = self.data
        d.root_link_pose_w = torch.zeros((n, 7))
        d.root_link_pose_w[:, 3] = 1.0
        d.root_com_vel_w = torch.zeros((n, 6))
        d.default_root_state = torch.zeros((n, 13))
        d.default_root_state[:, 3] = 1.0
        d.body_com_pos_b = torch.zeros((n, 1, 3))
        self._ALL_INDICES = torch.arange(n)
        self.mass = float(mass)
        self.inertia = torch.as_tensor(inertia, dtype=torch.float64)
        self.gravity = torch.as_tensor(gravity, dtype=torch.float64)

    def write_root_state_to_sim(self, root_state, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        self.data.root_link_pose_w[ids] = root_state[:, :7].to(torch.float32)
        self.data.root_com_vel_w[ids] = root_state[:, 7:13].to(torch.float32)
        self.writes.append((ids.clone(), root_state.clone()))

    def set_external_force_and_torque(self, forces, torques):
        if tuple(forces.shape[1:]) != (1, 3) or tuple(torques.shape[1:]) != (1, 3):
            raise ValueError("expected [N,1,3] body wrenches")
        self.applied.append((forces.clone(), torques.clone()))

    def integrate(self, dt):
        force_b, torque_b = (x[:, 0].to(torch.float64) for x in self.applied[-1])
        pose = self.data.root_link_pose_w.to(torch.float64)
        vel = self.data.root_com_vel_w.to(torch.float64)
        q = pose[:, 3:7]
        r = rotation(q)
        v = vel[:, :3] + dt * ((r @ force_b.unsqueeze(-1)).squeeze(-1) / self.mass + self.gravity)
        omega_b = (r.transpose(-1, -2) @ vel[:, 3:].unsqueeze(-1)).squeeze(-1)
        inertia = self.inertia.expand(len(q), 3, 3)
        gyro = torch.cross(omega_b, (inertia @ omega_b.unsqueeze(-1)).squeeze(-1), dim=-1)
        omega_b = omega_b + dt * torch.linalg.solve(inertia, (torque_b - gyro).unsqueeze(-1)).squeeze(-1)
        p = pose[:, :3] + dt * v
        half = 0.5 * dt * omega_b
        q = quat_mul(q, torch.cat((torch.ones_like(half[:, :1]), half), -1))
        q = q / q.norm(dim=-1, keepdim=True)
        omega_w = (rotation(q) @ omega_b.unsqueeze(-1)).squeeze(-1)
        self.data.root_link_pose_w = torch.cat((p, q), -1).to(torch.float32)
        self.data.root_com_vel_w = torch.cat((v, omega_w), -1).to(torch.float32)


class FakeScene:
    def __init__(self, n, spacing):
        self.env_origins = torch.stack((torch.arange(n, dtype=torch.float32) * float(spacing), torch.zeros(n),
                                        torch.zeros(n)), -1)
        self.rigid_objects = {}

    def clone_environments(self, copy_from_source=False):
        return None


class FakeDirectRLEnv:
    """Isaac Lab DirectRLEnv call order (isaaclab/envs/direct_rl_env.py step/reset), CPU."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        from tasks._shared import vehicles_v2
        self.cfg = cfg
        self.num_envs = int(cfg.scene.num_envs)
        self.device = torch.device("cpu")
        self.extras = {}
        self.scene = FakeScene(self.num_envs, cfg.scene.env_spacing)
        self._setup_scene()
        spec = vehicles_v2.get_vehicle_v2(cfg.vehicle)
        self.robot.bind(self.num_envs, spec.mass_kg, spec.inertia_com_kg_m2, cfg.sim.gravity)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.max_episode_length = math.ceil(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))

    def _reset_idx(self, env_ids):
        self.episode_length_buf[torch.as_tensor(env_ids, dtype=torch.long)] = 0

    def reset(self):
        self._reset_idx(self.robot._ALL_INDICES)
        return self._get_observations(), self.extras

    def step(self, action):
        self._pre_physics_step(action)
        for _ in range(self.cfg.decimation):
            self._apply_action()
            self.robot.integrate(self.cfg.sim.dt)
        self.episode_length_buf += 1
        terminated, truncated = self._get_dones()
        reward = self._get_rewards()
        done = (terminated | truncated).nonzero().flatten()
        if done.numel():
            self._reset_idx(done)
        return self._get_observations(), reward, terminated, truncated, self.extras


_install_stubs()
from tasks._shared import journal_cfg_v2 as plans  # noqa: E402
from tasks._shared import journal_contracts_v2 as contracts  # noqa: E402
from tasks._shared.conditions import dynamic, pairing  # noqa: E402
from tasks._shared.conditions import journal_config as JC  # noqa: E402
from tasks._shared import journal_runtime_cfg_v2 as templates  # noqa: E402
from tasks.docking import docking_env, docking_env_cfg  # noqa: E402
from tasks.docking import journal_d7_pool as pool_io  # noqa: E402
from tasks.docking import journal_docking as jd  # noqa: E402
from tasks.docking import level_table  # noqa: E402

SEED = 42
N = 4
ALL_ON = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "perception.bundle=D2", "dynamic=D2")
ALL_D0 = ("waves=D0", "wind=D0", "current=D0", "actuator=D0", "perception=D0", "perception.bundle=D0", "dynamic=D0")
CFG_CLASS = {"blueboat_v2": templates.DockingBlueBoatJournalEnvCfg,
             "wamv_vrx_v2_opt1": templates.DockingWAMVJournalEnvCfg}
_POOLS = {}
# G7 (item B): under the G7 unforced-window rule E4 admits every seed-42 docking key, so the D16 tests force
# a REAL E4 rejection on the keys they always used (harbour precedent, harbour_journal_fixtures.forced_pool):
# the E4 context of these keys gets a deadline no nominal plan can meet; the scenario, its geometry id and
# the env stay the key's E5 ones (no redraw).
FORCED_REJECT = {("blueboat_v2", True): ((1, 0), (1, 1), (2, 1))}
FORCED_DEADLINE_S = 5.0


def _admission_context(vessel, walls, ek, scenario, envelope):
    import dataclasses
    ctx = jd.d7_context(dynamic, scenario, envelope)
    if ek in FORCED_REJECT.get((vessel, walls), ()):
        ctx = dataclasses.replace(ctx, deadline_s=FORCED_DEADLINE_S)
    return ctx


def small_pool(vessel, walls=True):
    """REAL pool (the tool's algorithm: one joint admit on the plain keys, D16) for a 4-env x 2-episode grid."""
    if (vessel, walls) not in _POOLS:
        envelope = jd.d7_envelope(dynamic, vessel)
        suite = dynamic.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
        cells = [(e, k) for e in range(N) for k in range(2)]
        scenarios = {ek: jd.build_scenario(vessel, 2, SEED, ek[0], ek[1], walls) for ek in cells}
        offered = [dynamic.EpisodeKey(SEED, e, k, "d7_docking") for e, k in cells]
        pool = dynamic.admit_ladders([(key, dynamic.build_ladder(
            _admission_context(vessel, walls, ek, scenarios[ek], envelope), key, suite))
            for ek, key in zip(cells, offered)])
        doc = pool_io.pool_to_document(pool, vessel=vessel, tier=2, seed=SEED, num_envs=N, episodes=2,
                                       offered=offered, suite=suite,
                                       geometry_ids={ek: sc.geometry_id for ek, sc in scenarios.items()},
                                       berth_walls=walls)
        _POOLS[(vessel, walls)] = (doc, pool_io.pool_from_document(doc))
    return _POOLS[(vessel, walls)][1]


def make_cfg(vessel="blueboat_v2", tokens=(), seed=SEED, num_envs=N, walls=True):
    cfg = CFG_CLASS[vessel]()
    cfg.berth_walls = walls
    cfg.sim.gravity = (0.0, 0.0, -9.81)          # Isaac Lab SimulationCfg default (not in the stub)
    cfg.scene.num_envs = num_envs
    cfg.visual.enable_water = False
    JC.apply_journal_conditions(cfg, JC.resolve_journal_conditions(presets=list(tokens), seed=seed))
    cfg.journal_d7_pool = small_pool(vessel, walls)
    return cfg


def make_env(vessel="blueboat_v2", tokens=(), **kwargs):
    return docking_env.DockingEnv(make_cfg(vessel, tokens, **kwargs))


def actions(k, n=N):
    base = torch.tensor([[0.6, 0.2], [0.8, -0.3], [0.3, 0.5], [1.0, 0.0]])
    return (base[:n] * (1.0 - 0.05 * (k % 5)))


# ======================================================================== tests
class LegacyIdentity(unittest.TestCase):
    PINNED = {  # sha256 of the F-base source segments (tasks/docking/docking_env.py, class DockingEnv)
        "_pre_physics_step": "04f32d2602456d69167f2d3314f9d56d59abf984275320e72f90ffeec84b0a4d",
        "_apply_action": "62a7839b42df8837f5e2058cad44fac7997254610a8aac4075aa54f591526a23",
        "_get_observations": "2d06b5d08e3981e21f0dd455a9eacb234ee415a5f05e713dc737a27c11b55463",
        "_get_rewards": "3ffcb4f42a9395c98796a236bac3a9290dd93c9346d111691b6c0a4827245db7",
        "_get_dones": "adb4bce05ef23aa1b1d7b4754b1aa5194614775973702f657f851bef10f351b1",
        "_reset_idx": "c07fb31bcc197d0bea4d41d75a37312577831ae2ae940dcd6b900f597ff67533",
        "_setup_scene": "dc26772030d46da6635bdadea941ca7db3da9b0ae289fe0911a3e3987dcdae33",
        "_compute_buoyancy_forces": "84228fe9f506992f67dc0c235dd2a51585a38a0a7d2025ad08d767bc46dc8443",
        "_compute_current_forces": "be8e8e6376dfd200e1f9ced67b1efcb74eecb8ad1055900a57d443e4bdfddb92",
        "_task_state": "3fc9135910ee5d9bc0bae4a2f3fd697223e0c5e6bbfcb8b1a3dc7c68ed35b995",
        "_wall_clearance": "83e94a2ad01a0fb140353004dc2079e5254d978639d0fcdd726d10058691aeae",
        "_wall_ray_ranges_m": "c8e6bf3d78b189df1cc4fa99b8ac61270398048a3b8a1a4e4f9dc4c13f706b57",
    }

    @staticmethod
    def segments(path):
        src = Path(path).read_text(encoding="utf-8")
        cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "DockingEnv")
        return {f.name: ast.get_source_segment(src, f) for f in cls.body if isinstance(f, ast.FunctionDef)}

    def test_legacy_methods_byte_identical_to_fbase(self):
        ours = self.segments(REPO / "tasks" / "docking" / "docking_env.py")
        for name, digest in self.PINNED.items():
            self.assertEqual(hashlib.sha256(ours[name].encode()).hexdigest(), digest, name)
        if (FBASE / "tasks" / "docking" / "docking_env.py").is_file():
            base = self.segments(FBASE / "tasks" / "docking" / "docking_env.py")
            for name, text in base.items():
                if name != "__init__":
                    self.assertEqual(ours[name], text, name)
            # __init__: the legacy body follows the journal early return unchanged
            legacy = base["__init__"].split("\n", 1)[1]
            self.assertIn(legacy, ours["__init__"])

    def test_legacy_cfg_keeps_legacy_vehicle_and_physics(self):
        cfg = docking_env_cfg.DockingBlueBoatWallEnvCfg()
        self.assertEqual(cfg.vehicle, "blueboat")
        self.assertFalse(hasattr(cfg.underwater_physics_cfg, "hydrodynamics_model"))
        self.assertEqual(cfg.observation_space, 42)
        self.assertEqual(cfg.docking_level_table_in_force(), "eval_level_distances")


class JournalCfg(unittest.TestCase):
    def test_both_templates_build_the_journal_branch(self):
        for vessel, cls in CFG_CLASS.items():
            cfg = cls()
            plan = cfg.journal_plan
            self.assertEqual(cfg.underwater_physics_cfg.hydrodynamics_model, "hydro_v2")
            self.assertEqual(cfg.underwater_physics_cfg.physics_version, 2)
            self.assertEqual(cfg.ray_count, 36)
            self.assertAlmostEqual(cfg.ray_max_range_m, 30.0 * plan.params["length_scale"])
            self.assertEqual(cfg.observation_space, 42)
            self.assertIs(cfg.journal_observation_hooks_wired, True)
            self.assertEqual(cfg.observation_contract, "journal_dock6_ray36")
            self.assertEqual(cfg.vessel_profile, plans.vessel_record(vessel).key if vessel == "blueboat_v2"
                             else cfg.vessel_profile)
            self.assertTrue(cfg.robot_cfg.spawn.usd_path.replace("\\", "/").endswith(
                "assets_v2/" + ("blueboat_v2.usda" if vessel == "blueboat_v2" else "wamv_vrx_v2_competition.usdc")))
            self.assertIs(cfg.robot_cfg.spawn.rigid_props.disable_gravity, False)
            self.assertEqual(cfg.episode_length_s, plan.max_deadline_s)
            self.assertEqual(cfg.docking_level_table_in_force(), "journal_plan")
            JC.check_journal_cfg(cfg)

    def test_level_table_override_refusal_d2_item_14(self):
        cfg = templates.DockingBlueBoatJournalEnvCfg()
        cfg.validate_after_overrides()
        for field, value in (("eval_level_distances_v2", (3.0, 9.0, 16.0, 26.0)), ("eval_level", 2),
                             ("spawn_distance_override_m", 4.0)):
            bad = templates.DockingBlueBoatJournalEnvCfg()
            setattr(bad, field, value)
            with self.assertRaisesRegex(ValueError, "journal docking reads no level table"):
                bad.validate_after_overrides()
        legacy = docking_env_cfg.DockingBlueBoatEnvCfg()
        legacy.underwater_physics_cfg.physics_version = 2
        self.assertEqual(legacy.docking_level_table_in_force(), "eval_level_distances_v2")
        legacy.eval_level_distances_v2 = (3.0, 9.0, 16.0, 26.0)
        legacy.validate_after_overrides()                     # the table in force may change
        legacy.eval_level_distances = (30.0, 20.0, 10.0, 3.0)
        with self.assertRaisesRegex(ValueError, "has no effect"):
            legacy.validate_after_overrides()
        stamp = level_table.level_table_stamp(templates.DockingWAMVJournalEnvCfg())
        self.assertEqual(stamp["docking_level_table_in_force"], "journal_plan")

    def test_pinned_repo_pools_and_manifest_verify_d16_d19(self):
        """The pools ship in tasks/docking/journal_pools (D19); every key is admitted or explicitly not."""
        import json
        folder = Path(docking_env_cfg.JOURNAL_D7_POOL_DIR)
        manifest = folder / "POOL_MANIFEST.json"
        self.assertEqual(pool_io.file_sha256(manifest), docking_env_cfg.JOURNAL_D7_POOL_MANIFEST_SHA256)
        listed = json.loads(manifest.read_text(encoding="ascii"))["pools"]
        self.assertEqual(sorted(listed), sorted(docking_env_cfg.JOURNAL_D7_POOLS))
        # G49 (v2.6.5, J29): the pools of the formal seeds 11002 41041 41047 41051 41057 are pinned too; they travel
        # in the pool bundle only (verified by scripts/test_g49_seed_pools.py); the in-repo seed-42 pools keep the
        # 8 MB repo limit and the checks below
        self.assertEqual(sorted({entry["eval_seed"] for entry in listed.values()}),
                         [42, 11001, 11002, 41041, 41047, 41051, 41057])     # G53: + gate seed 11001
        in_repo = {name: entry for name, entry in listed.items() if entry["eval_seed"] == 42}
        self.assertEqual(len(in_repo), 2)
        self.assertLessEqual(sum(entry["bytes"] for entry in in_repo.values()), 8 * 1024 * 1024)
        for name, digest in docking_env_cfg.JOURNAL_D7_POOLS.items():
            if name not in in_repo:
                continue
            vessel = "blueboat_v2" if "blueboat" in name else "wamv_vrx_v2_opt1"
            pool = pool_io.load_pool(folder / name, expected_sha256=digest, vessel=vessel, tier=2, seed=42)
            self.assertEqual(listed[name]["pool_digest"], pool.pool_digest)
            self.assertEqual(listed[name]["keys_not_admitted"], pool.keys_not_admitted)
            self.assertEqual(pool.keys_admitted + pool.keys_not_admitted, 128)
            not_admitted = 0
            for env in range(64):
                for k in range(2):
                    self.assertTrue(pool.covers(env, k))
                    key = dynamic.EpisodeKey(42, env, k, "d7_docking")
                    status = pool.exposure(env, k)
                    for dose in range(5):
                        item = pool.assignment(key, dose)
                        if status == "not_admitted":
                            self.assertIsInstance(item, pool_io.NotAdmittedAssignment)
                            self.assertEqual((item.dose, item.requested_dose), (0, dose))
                            self.assertTrue(item.reasons)
                        else:
                            self.assertEqual(item.candidate.request.encounter.key, key)   # plain key, no remap
                    not_admitted += status == "not_admitted"
            self.assertEqual(not_admitted, pool.keys_not_admitted)
            self.assertEqual([c["not_admitted"] for c in pool.per_dose_exposure_counts()],
                             [pool.keys_not_admitted] * 5)
            # D24: the offered grid; episode index 2 is beyond it (not admitted), env 64 is outside it
            self.assertEqual(pool.grid, {"num_envs": 64, "episodes": 2})
            item = pool.assignment(dynamic.EpisodeKey(42, 63, 2, "d7_docking"), 3)
            self.assertIsInstance(item, pool_io.NotAdmittedAssignment)
            self.assertEqual((item.dose, item.reasons), (0, ((0, "beyond_offered_grid", 0),)))
            with self.assertRaises(KeyError):
                pool.assignment(dynamic.EpisodeKey(42, 64, 0, "d7_docking"), 3)


class PoolFile(unittest.TestCase):
    def test_file_round_trip_and_fail_closed(self):
        doc = (small_pool("blueboat_v2"), _POOLS[("blueboat_v2", True)][0])[1]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "docking_blueboat_v2_g2_s42_4x2.json.xz"
            sha = pool_io.write_pool(path, doc)
            pool = pool_io.load_pool(path, vessel="blueboat_v2", tier=2, seed=42)
            self.assertEqual(pool.pool_digest, doc["pool_digest"])
            self.assertEqual(pool.file_sha256, sha)
            with self.assertRaisesRegex(ValueError, "differs from the pinned"):
                pool_io.load_pool(path, expected_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "eval seed"):
                pool_io.load_pool(path, seed=43)
            tampered = copy.deepcopy(doc)
            tampered["assignments"][0]["candidate"]["f"]["request"]["f"]["dwell_s"] = 1.0
            with self.assertRaises(ValueError):
                pool_io.pool_from_document(tampered)
            tampered = copy.deepcopy(doc)
            tampered["manifest_accepted"] = tampered["manifest_accepted"][1:]
            with self.assertRaisesRegex(ValueError, "pool digest"):
                pool_io.pool_from_document(tampered)
            # D16 records: the not-admitted list must equal the E4 rejections exactly
            self.assertTrue(doc["not_admitted"])
            tampered = copy.deepcopy(doc)
            tampered["not_admitted"][0]["reasons"][0][1] = "made_up_reason"
            with self.assertRaisesRegex(ValueError, "disagrees with the E4 rejections"):
                pool_io.pool_from_document(tampered)
            tampered = copy.deepcopy(doc)
            row = next(r for r in tampered["task_episodes"] if r[3] == "not_admitted")
            row[3] = "admitted"
            with self.assertRaisesRegex(ValueError, "manifest disagrees"):
                pool_io.pool_from_document(tampered)
            old = copy.deepcopy(doc)
            old["schema"] = "usvbench.docking.d7_pool/1"                  # round-4 attempt pools are refused
            with self.assertRaisesRegex(ValueError, "not a docking D7 pool"):
                pool_io.pool_from_document(old)
            with self.assertRaises(FileNotFoundError):
                pool_io.load_pool(Path(folder) / "missing.json.xz", expected_sha256="0" * 64)

    def test_env_refuses_an_unpinned_or_missing_pool(self):
        cfg = make_cfg()
        cfg.journal_d7_pool = None
        with tempfile.TemporaryDirectory() as folder:
            cfg.d7_pool_dir = folder
            with self.assertRaises(FileNotFoundError):
                docking_env.DockingEnv(cfg)
            cfg.d7_pool_file = str(Path(folder) / "unpinned.json.xz")
            with self.assertRaisesRegex(ValueError, "no pinned SHA-256"):
                docking_env.DockingEnv(cfg)


class KeyedStreams(unittest.TestCase):
    def test_scenario_is_a_pure_function_of_the_key_c1_finding_29(self):
        a = jd.build_scenario("blueboat_v2", 2, 42, 3, 1)
        b = jd.build_scenario("blueboat_v2", 2, 42, 3, 1)
        self.assertEqual(a.family_hashes(), b.family_hashes())
        self.assertEqual(a.berth_heading_rad, jd.berth_heading(42, 3, 1))
        other_seed = jd.build_scenario("blueboat_v2", 2, 43, 3, 1)
        self.assertNotEqual(a.family_hashes()["berth"], other_seed.family_hashes()["berth"])
        # berth and spawn are separate streams: the spawn stream's draws do not depend on the berth stream
        rng = jd.spawn_generator(42, 3, 1)
        layout = jd.layouts.build_layout("docking", 2, "blueboat_v2", rng)
        np.testing.assert_array_equal(layout.start, a.layout.start)
        self.assertFalse(hasattr(jd, "encounter_episode") or hasattr(jd, "stream_group"))   # D16: no attempts

    def test_suites_do_not_move_berth_or_spawn(self):
        plain, stacked = make_env(), make_env(tokens=("current=D3", "waves=D2", "perception=D2", "dynamic=D1"))
        plain.reset()
        stacked.reset()
        for i in range(N):
            self.assertEqual(plain._journal_family_scenario_hashes(i), stacked._journal_family_scenario_hashes(i))
        self.assertTrue(torch.equal(plain.robot.writes[0][1], stacked.robot.writes[0][1]))

    def test_reset_order_does_not_change_the_scenario(self):
        a, b = make_env(), make_env()
        a._reset_idx(torch.tensor([0, 1, 2, 3]))
        b._reset_idx(torch.tensor([3, 1]))
        b._reset_idx(torch.tensor([2, 0]))
        for i in range(N):
            self.assertEqual(a._journal_family_scenario_hashes(i), b._journal_family_scenario_hashes(i))


def record_calls(obj, names, log, prefix):
    for name in names:
        original = getattr(obj, name)

        def wrapper(*args, _original=original, _name=name, **kwargs):
            log.append(prefix + "." + _name)
            return _original(*args, **kwargs)
        setattr(obj, name, wrapper)


class CallOrder(unittest.TestCase):
    def test_substep_order_through_the_real_runtime_layer_and_bridge(self):
        env = make_env(tokens=ALL_ON)
        env.journal_trace = trace = []
        inner = []
        record_calls(env._hydro_v2, ("reset", "step_from_root_link"), inner, "runtime")
        record_calls(env._conditions, ("reset_runtime", "configure_runtime", "physics_inputs", "episode_stamp",
                                       "estimate_position", "estimate_velocity", "estimate_rays"), inner, "conditions")
        record_calls(env._d7_bridge, ("reset_rows", "step", "finish"), inner, "bridge")
        record_calls(env._jd_oracle, ("reset", "observe_physics", "step"), inner, "oracle")
        env.reset()
        self.assertEqual(inner, ["oracle.reset", "runtime.reset", "conditions.reset_runtime",
                                 "conditions.configure_runtime", "bridge.reset_rows", "bridge.step",
                                 "conditions.estimate_position", "conditions.estimate_velocity",
                                 "conditions.estimate_rays"])
        inner.clear()
        env.step(actions(0))
        sub = ["oracle.observe_physics", "bridge.step", "conditions.physics_inputs", "runtime.step_from_root_link"]
        self.assertEqual(inner[:8], sub * 2)
        self.assertEqual(inner[8:11], ["bridge.step", "oracle.step", "oracle.observe_physics"])   # step observes the final sample
        self.assertEqual(inner[-3:], ["conditions.estimate_position", "conditions.estimate_velocity",
                                      "conditions.estimate_rays"])
        self.assertEqual(trace, ["reset", "estimates", "pre_physics", "substep", "substep", "pre_dones", "dones",
                                 "estimates"])

    def test_actions_are_clamped_once_and_forces_are_body_wrenches(self):
        env = make_env()
        env.reset()
        env.step(torch.full((N, 2), 3.0))
        self.assertTrue(torch.equal(env.actions, torch.ones(N, 2)))
        forces, torques = env.robot.applied[-1]
        self.assertEqual(tuple(forces.shape), (N, 1, 3))


class ZeroDose(unittest.TestCase):
    def test_all_suites_at_d0_are_bit_identical_to_no_suites(self):
        runs = []
        for tokens in ((), ALL_D0):
            env = make_env(tokens=tokens)
            obs = [env.reset()[0]["policy"]]
            for k in range(10):
                obs.append(env.step(actions(k))[0]["policy"])
            runs.append((env, obs))
        (a, oa), (b, ob) = runs
        self.assertEqual(len(a.robot.applied), len(b.robot.applied))
        for (fa, ta), (fb, tb) in zip(a.robot.applied, b.robot.applied):
            self.assertTrue(torch.equal(fa, fb) and torch.equal(ta, tb))
        for x, y in zip(oa, ob):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(any(bool(torch.count_nonzero(f)) for f, _ in a.robot.applied))


def force_out_of_bounds(env, rows):
    """Teleport rows far outside the admitted bounds (a deterministic terminal event)."""
    pose = env.robot.data.root_link_pose_w
    for i in rows:
        pose[i, 0] = env.scene.env_origins[i, 0] + 1.0e3
        pose[i, 1] = env.scene.env_origins[i, 1] + 1.0e3


class AllSuitesOn(unittest.TestCase):
    def run_vessel(self, vessel):
        env = make_env(vessel, tokens=ALL_ON)
        env.reset()
        for k in range(3):
            if k == 2:
                force_out_of_bounds(env, range(N))
            _, _, terminated, truncated, _ = env.step(actions(k))
        self.assertEqual(env.termination_reason.tolist(), [4] * N)
        self.assertTrue(bool(terminated.all()) and not bool(truncated.any()))
        for i in range(N):
            record = env.journal_episode_record(i)
            stamp = record["conditions"]
            self.assertEqual(stamp["schema"], "usvbench.conditions/3")
            for suite in ("waves", "current", "wind", "actuator", "perception"):
                self.assertTrue(stamp["parameters"][suite]["enabled"], suite)
            rt = stamp["runtime"]
            self.assertEqual(rt["runtime_ticks"], 6)
            self.assertEqual(rt["estimates"]["counts"], {"pos": 3, "vel": 3, "ray": 3})
            self.assertEqual(stamp["context"]["vehicle"], vessel)
            dynamic.read_stamp(record["dynamic_hazards"])
            payload = record["dynamic_hazards"]["payload"]
            admitted = env._jd_pool.exposure(i, 0) == "admitted"
            self.assertEqual(payload["assignment"]["dose"], 2 if admitted else 0)
            self.assertEqual(payload.get("exposure", "admitted"), env._jd_pool.exposure(i, 0))
            self.assertEqual(payload["runtime"]["termination_reason"], "out_of_bounds")
            self.assertEqual(payload["runtime"]["bridge"]["last_tick"], 6)
            self.assertEqual(payload["assignment"]["key"]["episode"], 0)          # plain task key (D16)
            self.assertEqual(sorted(record["scenario_hashes"]), sorted(env.journal_pairing_inventory))
            self.assertIn("berth", record["scenario_hashes"])
            self.assertIn("spawn_pose", record["scenario_hashes"])
            result = env.journal_episode_results[i]
            self.assertEqual(result["termination_reason"], 4)
            self.assertEqual(result["termination_reason_name"], "out_of_bounds")
        header = env.journal_certificate_header()
        self.assertEqual(header["dynamic_hazards_header"]["pool_digest"], env._d7_pool.pool_digest)
        self.assertIn("docking_pool", header["dynamic_hazards_header"])
        # D22: [offered, admitted, not_admitted] rows; D24 (c): the pool's offered grid
        pool = env._d7_pool
        self.assertEqual(header["dynamic_hazards_header"]["per_dose_counts"],
                         [[o, a, o - a] for o, a in pool.per_dose_counts])
        self.assertEqual([row[2] for row in header["dynamic_hazards_header"]["per_dose_counts"]],
                         [pool.keys_not_admitted] * 5)
        self.assertEqual(header["dynamic_hazards_header"]["per_dose_counts_fields"],
                         ["offered", "admitted", "not_admitted"])
        self.assertEqual(header["dynamic_hazards_header"]["grid"], {"num_envs": N, "episodes": 2})
        self.assertEqual(header["dynamic_hazards_header"]["grid"], pool.grid)
        # every row was reset after its latch: the new episode is running
        self.assertEqual(int(env._jd_task_episode[0]), 1)
        self.assertFalse(env._d7_bridge.rows[0]["finished"])
        return env

    def test_blueboat_all_suites_on(self):
        self.run_vessel("blueboat_v2")

    def test_wamv_all_suites_on(self):
        self.run_vessel("wamv_vrx_v2_opt1")


class Pairing(unittest.TestCase):
    def test_single_treatment_pair_is_clean(self):
        records = []
        for tokens in (("waves=D1", "dynamic=D1"), ("waves=D3", "dynamic=D1")):
            env = make_env(tokens=tokens)
            env.reset()
            env.step(actions(0))
            force_out_of_bounds(env, range(N))
            env.step(actions(1))
            records.append(env.journal_episode_record(1))
        summary = pairing.validate_pair(records[0], records[1], "waves")
        self.assertEqual(summary["changed_primitives"], ["waves"])
        with self.assertRaises(pairing.PairingError):
            pairing.validate_pair(records[0], records[1], "wind")


class Parity(unittest.TestCase):
    def test_env_trace_scalar_oracle_matches_the_tensor_twin(self):
        env = make_env(tokens=("waves=D2", "dynamic=D3"))
        scalars = [None] * N
        original_observe, original_step = env._jd_oracle.observe_physics, env._jd_oracle.step
        original_reset = env._jd_oracle.reset
        seen = []

        def rigid(state, i):
            return contracts.RigidState(state["com"][i].tolist(), state["quat"][i].tolist(), state["lin"][i].tolist(),
                                        state["ang"][i].tolist(), water_level_m=float(state["water"][i]),
                                        nominal_com_height_m=float(state["nominal"][i]))

        def reset(ids, com_xy, **kw):
            original_reset(ids, com_xy, **kw)
            for row, i in enumerate(torch.as_tensor(ids).tolist()):
                b = env._jd_scenarios[i].berth
                berth = contracts.Berth(tuple(b["center_xy"]), b["heading_rad"], b["opening_x_m"], b["back_x_m"],
                                        b["half_width_m"])
                s = contracts.DockingEpisode(berth, env._jd_record.footprint.vertices_m, float(env._jd_deadline[i]),
                                             hold_s=b["hold_s"], position_tolerance_m=b["position_tolerance_m"],
                                             heading_tolerance_rad=b["heading_tolerance_rad"],
                                             speed_tolerance_mps=b["speed_tolerance_m_s"],
                                             validity_limits=contracts.ValidityLimits(**env._jd_limits))
                s.reset(tuple(com_xy[row].tolist()))
                scalars[i] = s

        def observe(state, collision=None, clearance_m=None, rows=None):
            for i in range(N):
                scalars[i].observe_physics(rigid(state, i), collision=bool(collision[i]),
                                           clearance_m=float(clearance_m[i]))
            return original_observe(state, collision=collision, clearance_m=clearance_m, rows=rows)

        def step(time_s, state, collision=None, out_of_bounds=None, clearance_m=None):
            expected = []
            for i in range(N):
                d = scalars[i].step(float(time_s[i]), rigid(state, i), collision=bool(collision[i]),
                                    out_of_bounds=bool(out_of_bounds[i]), clearance_m=float(clearance_m[i]))
                expected.append(int(d.reason) if d.done else 0)
            codes = original_step(time_s, state, collision=collision, out_of_bounds=out_of_bounds,
                                  clearance_m=clearance_m)
            self.assertEqual(codes.tolist(), expected)
            for i in range(N):
                if expected[i]:
                    ref = scalars[i].last_completed.evaluator_fields()
                    for key, value in ref.items():
                        got = env._jd_oracle.last_completed[i][key]
                        self.assertTrue(got == value or (isinstance(value, float) and math.isnan(value)
                                                         and math.isnan(got)), (key, got, value))
                    seen.append(expected[i])
            return codes

        env._jd_oracle.reset, env._jd_oracle.observe_physics, env._jd_oracle.step = reset, observe, step
        env.reset()
        for k in range(40):
            if k == 30:
                force_out_of_bounds(env, [1])
            env.step(actions(k))
        self.assertIn(4, seen)


class ResetIsolation(unittest.TestCase):
    def test_resetting_one_row_leaves_the_others_untouched(self):
        env = make_env(tokens=("perception=D2", "wind=D2", "dynamic=D2"))
        env.reset()
        for k in range(2):
            env.step(actions(k))
        before = (int(env._hydro_v2.ticks[0]), env._conditions.estimate_counts(0), int(env._d7_tick[0]),
                  float(env._jd_oracle.full_len[0]), env._jd_scenarios[0].family_hashes(),
                  env._d7_bridge.rows[0]["last_tick"])
        env._reset_idx(torch.tensor([1]))
        after = (int(env._hydro_v2.ticks[0]), env._conditions.estimate_counts(0), int(env._d7_tick[0]),
                 float(env._jd_oracle.full_len[0]), env._jd_scenarios[0].family_hashes(),
                 env._d7_bridge.rows[0]["last_tick"])
        self.assertEqual(before, after)
        self.assertEqual(int(env._hydro_v2.ticks[1]), 0)
        self.assertEqual(int(env._jd_task_episode[1]), 1)
        self.assertEqual(len(env.journal_aborted_dynamic_hazards), 1)   # mid-episode manual reset = aborted


class SpawnAndSensors(unittest.TestCase):
    def test_spawn_puts_the_com_at_the_calm_equilibrium_and_the_admitted_pose(self):
        env = make_env("wamv_vrx_v2_opt1")
        env.reset()
        record = plans.vessel_record("wamv_vrx_v2_opt1")
        pose = env.robot.data.root_link_pose_w
        for i in range(N):
            sc = env._jd_scenarios[i]
            self.assertAlmostEqual(float(pose[i, 2]), record.com_height_above_waterline_m, places=5)
            self.assertAlmostEqual(float(pose[i, 0] - env.scene.env_origins[i, 0]), sc.start_xy[0], places=3)
            self.assertAlmostEqual(float(pose[i, 1] - env.scene.env_origins[i, 1]), sc.start_xy[1], places=3)
            self.assertAlmostEqual(math.atan2(2 * float(pose[i, 3] * pose[i, 6]), 1 - 2 * float(pose[i, 6]) ** 2),
                                   sc.start_heading_rad, places=5)
            self.assertGreater(float(env._jd_prev_clear[i]), 0.0)     # spawn is clear of the walls

    def test_nonzero_com_offset_is_refused(self):
        class Offset(docking_env.DockingEnv):
            def _setup_scene(self):
                super()._setup_scene()
                original = self.robot.bind

                def bind(n, *a):
                    original(n, *a)
                    self.robot.data.body_com_pos_b[:, 0, 0] = 0.01
                self.robot.bind = bind
        with self.assertRaisesRegex(ValueError, "origin at the COM"):
            Offset(make_cfg())

    def test_d7_slot_publication_contact_and_rays(self):
        env = make_env(tokens=("dynamic=D4",))
        env.reset()
        admitted = torch.tensor([env._jd_pool.exposure(i, 0) == "admitted" for i in range(N)])
        self.assertTrue(bool((~admitted).any()) and bool(admitted.any()))
        # one intruder per ADMITTED row at D4, physical once revealed (G23c L2, rule S2: hidden before the reveal)
        shown = torch.tensor([bool(admitted[i]) and _d7_shown(env, i) for i in range(N)])
        self.assertTrue(torch.equal(env._d7_physical[:, 0], shown))
        self.assertEqual(tuple(env._d7_centers.shape), (N, 1, 2))
        self.assertEqual(env._jd_d7_slot, plans.STATIC_CAPACITY["docking"])
        # rays: walls and the D7 disk combine by elementwise minimum, D7 only when visible
        env._d7_visible[:] = True
        env._d7_centers[0, 0] = env.robot.data.root_link_pose_w[0, :2].to(torch.float64) \
            - env.scene.env_origins[0, :2].to(torch.float64) + torch.tensor([3.0, 0.0], dtype=torch.float64)
        env._d7_radii[0, 0] = 0.5
        from tasks.docking.journal_kernels import fan_angles, ray_circle_ranges, yaw_from_quat
        pose = env.robot.data.root_link_pose_w.to(torch.float64)
        xy = pose[:, :2] - env.scene.env_origins[:, :2].to(torch.float64)
        ang = fan_angles(yaw_from_quat(pose[:, 3:7]), 36)
        walls = ray_circle_ranges(xy, ang, env._jd_wall_centers, env._jd_wall_radii, env.cfg.ray_max_range_m)
        d7 = ray_circle_ranges(xy, ang, env._d7_centers, env._d7_radii, env.cfg.ray_max_range_m,
                               active=env._d7_visible)
        combined = torch.minimum(walls, d7)
        self.assertTrue(bool((combined <= walls).all()))
        self.assertLess(float(combined[0].min()), 3.0)
        env._d7_visible[:] = False
        hidden = ray_circle_ranges(xy, ang, env._d7_centers, env._d7_radii, env.cfg.ray_max_range_m,
                                   active=env._d7_visible)
        self.assertTrue(bool((hidden == env.cfg.ray_max_range_m).all()))

    def test_swept_d7_contact_is_merged_into_the_contract_collision(self):
        env = make_env(tokens=("dynamic=D1",))
        env.reset()
        self.assertEqual(env._jd_pool.exposure(3, 0), "admitted")
        env._d7_bridge.rows[3]["contact"] = True                         # latched swept contact of row 3
        env.step(actions(0))
        self.assertEqual(int(env.termination_reason[3]), 2)
        self.assertEqual(env.episode_dynamic_hazards[3]["payload"]["runtime"]["termination_reason"], "collision")

    def test_virtual_berth_keeps_the_36_ray_block_against_the_d7_slot(self):
        """berth_walls=False (D7 spec D5 / R4 D14): no wall geometry, rays see the D7 slot only."""
        env = make_env(walls=False, tokens=("dynamic=D4",))
        obs = env.reset()[0]["policy"]
        self.assertEqual(env._jd_walls, 0)
        self.assertEqual(tuple(obs.shape), (N, 42))
        self.assertTrue(bool(torch.isinf(env._jd_prev_clear).all()))      # no wall clearance
        env._d7_visible[:] = True
        env._d7_centers[:, 0] = (env.robot.data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]).to(
            torch.float64) + torch.tensor([2.0, 0.0], dtype=torch.float64)
        env._d7_radii[:, 0] = 0.4
        env._journal_publish_d7 = lambda samples: None     # keep the hand-placed visible disk for this step
        env.step(torch.zeros(N, 2))
        self.assertTrue(bool((env.journal_estimate.rays.min(dim=1).values < env.cfg.ray_max_range_m).all()))
        with self.assertRaisesRegex(ValueError, "berth_walls"):
            cfg = make_cfg()
            cfg.journal_d7_pool = small_pool("blueboat_v2", walls=False)
            docking_env.DockingEnv(cfg)

    def test_observation_reads_the_position_estimate(self):
        env = make_env(tokens=("perception.bias=D4", "perception.enabled=true"))
        obs = env.reset()[0]["policy"]
        est = env.journal_estimate.position
        truth = env.robot.data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]
        self.assertFalse(torch.equal(est, truth))
        expected = torch.linalg.vector_norm(est, dim=-1) / env.cfg.observation_distance_scale_m
        self.assertTrue(torch.allclose(obs[:, 2], expected))
        self.assertEqual(tuple(obs.shape), (N, 42))


class NotAdmittedD16(unittest.TestCase):
    """Round 4b, CONTEXT_R4 D16: E4-rejected keys keep their scenario and run the dynamic suite at D0."""

    NA_ROW = 1                     # BlueBoat keys (1, 0) and (1, 1) are rejected by E4 (G7: forced, FORCED_REJECT)
    SUITES = {"waves": "waves=D2", "wind": "wind=D2", "current": "current=D2", "actuator": "actuator=D2",
              "perception": "perception=D2"}

    @classmethod
    def setUpClass(cls):
        pool = small_pool("blueboat_v2")
        cls.pool = pool
        cls.rejected = sorted(cell for cell in ((e, k) for e in range(N) for k in range(2))
                              if pool.exposure(*cell) == "not_admitted")

    def test_small_grid_has_rejected_keys_with_e4_reasons(self):
        self.assertEqual(self.rejected, [(1, 0), (1, 1), (2, 1)])
        for cell in self.rejected:
            reasons = self.pool.not_admitted_reasons(*cell)
            self.assertTrue(reasons)
            for dose, reason, _n in reasons:
                self.assertIn(dose, range(5))
                self.assertNotEqual(reason, "joint_ladder_member_rejected")

    def test_layout_of_a_not_admitted_key_is_the_e5_layout_of_that_key(self):
        """No redraw: the env's scenario equals E5 build_layout on the key's own keyed streams."""
        env = make_env(tokens=("dynamic=D3",))
        env.reset()
        for i in range(N):
            scenario = env._jd_scenarios[i]
            e5 = jd.layouts.build_layout("docking", 2, "blueboat_v2", jd.spawn_generator(SEED, i, 0))
            self.assertEqual(scenario.layout.geometry_hash, e5.geometry_hash)
            np.testing.assert_array_equal(scenario.layout.start, e5.start)
            self.assertEqual(scenario.berth_heading_rad, jd.berth_heading(SEED, i, 0))
            self.assertEqual(scenario.geometry_id, self.pool.geometry_id(i, 0))
            self.assertEqual(scenario.family_hashes(), jd.build_scenario("blueboat_v2", 2, SEED, i, 0).family_hashes())
        # identical to the scenario of the same key with the dynamic suite off
        plain = make_env()
        plain.reset()
        self.assertEqual(plain._journal_family_scenario_hashes(self.NA_ROW),
                         env._journal_family_scenario_hashes(self.NA_ROW))

    def run_full_episode(self, tokens, max_steps=4000, terminal_after=None):
        """Zero thrust until the not-admitted row finishes its FIRST episode.

        terminal_after=None: the natural end (timeout at the scenario deadline, ~2.7k control steps);
        terminal_after=n: the row is moved out of bounds after n steps (code 4 on step n + 1).
        """
        env = make_env(tokens=tokens, num_envs=2)
        env.reset()
        row = self.NA_ROW
        self.assertEqual(env._d7_bridge.exposure(row), "not_admitted")
        self.deadline_steps = env._jd_scenarios[row].deadline_steps
        # G23c L2 (rule S2): row 0 carries its intruder (physical once revealed)
        self.assertEqual(bool(env._d7_physical[0, 0]), _d7_shown(env, 0))
        self.admitted_row_physical = bool(env._d7_bridge.adapter.episodes[0].objects)
        steps = 0
        while env.journal_episode_results[row] is None:
            self.assertLess(steps, max_steps)
            if terminal_after is not None and steps == terminal_after:
                force_out_of_bounds(env, [row])
            env.step(torch.zeros(2, 2))
            steps += 1
            self.assertFalse(bool(env._d7_physical[row, 0]) or bool(env._d7_visible[row, 0]))
            self.assertFalse(bool(env._jd_d7_contact[row]))
        return env, steps

    def test_natural_full_episode_of_a_not_admitted_key_at_d4(self):
        """Reset -> timeout at the scenario's own deadline with the suite at D4: never an intruder."""
        from tasks.docking.journal_d7_exposure import record_exposure
        env, steps = self.run_full_episode(("dynamic=D4",))
        self.assertLessEqual(abs(steps - self.deadline_steps), 1)
        self.assertTrue(self.admitted_row_physical)
        result = env.journal_episode_results[self.NA_ROW]
        self.assertEqual(result["termination_reason_name"], "timeout")
        self.assertEqual((result["d7_exposure"], result["d7_requested_dose"], result["d7_run_dose"]),
                         ("not_admitted", 4, 0))
        record = env.journal_episode_record(self.NA_ROW)
        self.assertEqual(record_exposure(record["dynamic_hazards"]), "not_admitted")
        self.assertEqual(record["dynamic_hazards"]["payload"]["runtime"]["termination_reason"], "timeout")
        self.assertEqual(record["dynamic_hazards"]["payload"]["runtime"]["bridge"]["last_tick"], 2 * steps)

    def test_episode_at_every_dynamic_level_stamps_not_admitted_and_pairs(self):
        from tasks._shared.conditions.certificate import verify_dynamic_hazard_stamp
        from tasks.docking.journal_d7_exposure import record_exposure

        row = self.NA_ROW
        records, d7_hashes = {}, set()
        for level in range(5):
            env, steps = self.run_full_episode(("dynamic=D%d" % level,), terminal_after=3)
            self.assertEqual(steps, 4)
            self.assertEqual(self.admitted_row_physical, level > 0)             # row 0 (admitted) has its intruder
            result = env.journal_episode_results[row]
            self.assertEqual(result["termination_reason_name"], "out_of_bounds")
            self.assertEqual((result["d7_exposure"], result["d7_requested_dose"], result["d7_run_dose"]),
                             ("not_admitted", level, 0))
            self.assertTrue(result["d7_not_admitted_reasons"])
            record = env.journal_episode_record(row)
            envelope = verify_dynamic_hazard_stamp(record["dynamic_hazards"])
            dynamic.read_stamp(record["dynamic_hazards"])
            payload = envelope["payload"]
            self.assertEqual(payload["exposure"], "not_admitted")
            self.assertEqual(record_exposure(record["dynamic_hazards"]), "not_admitted")
            self.assertEqual(payload["assignment"]["dose"], 0)
            self.assertEqual(payload["assignment"]["requested_dose"], level)
            self.assertEqual(payload["assignment"]["intruders"], [])
            self.assertEqual(payload["assignment"]["key"], {"eval_seed": SEED, "env": row, "episode": 0,
                                                            "hazard_group": "d7_docking"})
            self.assertEqual(payload["runtime"]["non_exposure_reason"], "not_admitted")
            self.assertEqual(payload["runtime"]["termination_reason"], "out_of_bounds")
            self.assertEqual(payload["runtime"]["bridge"]["last_tick"], steps * 2)
            self.assertEqual([[r["dose"], r["reason"], r["sampled_states"]] for r in payload["not_admitted"]["reasons"]],
                             [list(r) for r in self.pool.not_admitted_reasons(row, 0)])
            self.assertEqual(payload["not_admitted"]["requested_dose"], level)
            # the header counts not_admitted per dose; the manifest digest is stamped
            header = env.journal_certificate_header()["dynamic_hazards_header"]
            self.assertEqual([c["not_admitted"] for c in header["per_dose_exposure_counts"]], [3] * 5)
            self.assertEqual(header["per_dose_counts"], [[8, 5, 3]] * 5)            # D22 rows
            self.assertEqual(header["keys_not_admitted"], 3)
            self.assertEqual(header["pool_manifest_sha256"], docking_env_cfg.JOURNAL_D7_POOL_MANIFEST_SHA256)
            records[level] = record
            d7_hashes.add(record["scenario_hashes"][pairing.D7_PRIMITIVE])
            # the row keeps running: its next episode (1, 1) is also not admitted, no exception
            self.assertEqual(env._d7_bridge.exposure(row), "not_admitted")
            self.assertEqual(int(env._jd_task_episode[row]), 1)
            env.step(torch.zeros(2, 2))
        self.assertEqual(len(d7_hashes), 1)                   # the same D7 primitive at every level
        # unexposed for the dynamic treatment: no treatment difference between Dk and D0
        for level in range(1, 5):
            with self.assertRaisesRegex(pairing.PairingError, "no treatment difference"):
                pairing.validate_pair(records[0], records[level], "dynamic")
        # paired with the same key at D0 of every other suite (dynamic suite absent there)
        for suite, token in self.SUITES.items():
            env = make_env(tokens=(token,), num_envs=2)
            env.reset()
            env.step(torch.zeros(2, 2))
            force_out_of_bounds(env, [row])
            env.step(torch.zeros(2, 2))
            other = env.journal_episode_record(row)
            self.assertEqual(verify_dynamic_hazard_stamp(other["dynamic_hazards"])["payload"]["exposure"],
                             "not_admitted")
            for level in range(5):
                summary = pairing.validate_pair(records[level], other, suite)
                self.assertTrue(summary["changed_primitives"])
                self.assertLessEqual(set(summary["changed_primitives"]), set(pairing.SUITE_PRIMITIVES[suite]))
                self.assertNotIn(pairing.D7_PRIMITIVE, summary["changed_primitives"])
                self.assertEqual(summary["identity"]["env_index"], row)

    def test_admitted_and_not_admitted_rows_switch_without_exception(self):
        """Row 2: episode 0 admitted (intruder at D4), episode 1 not admitted (empty D0 row)."""
        env = make_env(tokens=("dynamic=D4",))
        env.reset()
        self.assertEqual(env._d7_bridge.exposure(2), "admitted")
        self.assertTrue(bool(env._d7_bridge.adapter.episodes[2].objects))
        self.assertEqual(bool(env._d7_physical[2, 0]), _d7_shown(env, 2))   # G23c L2: physical once revealed
        env.step(actions(0))
        force_out_of_bounds(env, [2])
        env.step(actions(1))
        first = env.journal_episode_record(2)
        self.assertNotIn("exposure", first["dynamic_hazards"]["payload"])      # E4 stamp unchanged
        self.assertEqual(first["dynamic_hazards"]["payload"]["assignment"]["dose"], 4)
        self.assertEqual(env._d7_bridge.exposure(2), "not_admitted")
        self.assertFalse(bool(env._d7_physical[2, 0]))
        env.step(actions(2))
        self.assertFalse(bool(env._d7_physical[2, 0]) or bool(env._jd_d7_contact[2]))
        self.assertEqual(bool(env._d7_physical[0, 0]), _d7_shown(env, 0))   # the admitted rows keep theirs
        # the second episode is the last one of the 4x2 grid: stamp it through the bridge directly
        # (a further env reset asks for episode 2, beyond the offered grid: D24, class BeyondGridD24)
        second = env._d7_bridge.finish(2, "timeout")
        self.assertEqual(second["payload"]["exposure"], "not_admitted")
        self.assertEqual(second["payload"]["assignment"]["key"]["episode"], 1)
        self.assertEqual(second["payload"]["assignment"]["requested_dose"], 4)
        # a key outside the pool's env range is still a preflight error (never a redraw)
        with self.assertRaises(KeyError):
            self.pool.assignment(dynamic.EpisodeKey(SEED, 9, 0, "d7_docking"), 1)


class BeyondGridD24(unittest.TestCase):
    """Round T, CONTEXT_R4 D24: task episodes beyond the pool's offered grid run at D0, never counted.

    The evaluators keep stepping every env until the slowest reaches its quota, so a fast env autoresets
    into k >= grid["episodes"]. The small BlueBoat pool offers 4 envs x 2 episodes; row 0 is admitted at
    k = 0, 1 and row 1 is NOT admitted by E4 at k = 0, 1 (D16), so both exposures lead into k = E = 2.
    """

    ROWS = (0, 1)
    REASON = [{"dose": 0, "reason": "beyond_offered_grid", "sampled_states": 0}]

    @classmethod
    def setUpClass(cls):
        cls.pool = small_pool("blueboat_v2")
        cls.E = int(cls.pool.grid["episodes"])

    def advance_past_the_grid(self, env):
        """Forced terminals end episodes 0 .. E-1 of both rows; the env itself autoresets them into k = E."""
        for k in range(self.E):
            env.step(torch.zeros(2, 2))
            force_out_of_bounds(env, self.ROWS)
            env.step(torch.zeros(2, 2))
            for row in self.ROWS:
                self.assertEqual(int(env._jd_task_episode[row]), k + 1)
                self.assertEqual(env.journal_episode_results[row]["task_episode"], k)
        for row in self.ROWS:
            env.journal_episode_results[row] = None            # test bookkeeping: wait for the k = E latch

    def check_layout_of_the_beyond_grid_key(self, env, row):
        """The env built the E5 layout of the key (seed, row, E) with the same keyed streams as any key."""
        scenario = env._jd_scenarios[row]
        self.assertEqual(scenario.episode, self.E)
        e5 = jd.layouts.build_layout("docking", 2, "blueboat_v2", jd.spawn_generator(SEED, row, self.E))
        self.assertEqual(scenario.layout.geometry_hash, e5.geometry_hash)
        np.testing.assert_array_equal(scenario.layout.start, e5.start)
        self.assertEqual(scenario.berth_heading_rad, jd.berth_heading(SEED, row, self.E))
        built = jd.build_scenario("blueboat_v2", 2, SEED, row, self.E)
        self.assertEqual(scenario.geometry_id, built.geometry_id)
        self.assertEqual(scenario.family_hashes(), built.family_hashes())
        # the pool holds no geometry record of the key: the env skipped the pool geometry check
        self.assertFalse(self.pool.covers(row, self.E))
        with self.assertRaisesRegex(ValueError, "beyond the offered grid"):
            self.pool.geometry_id(row, self.E)
        self.assertEqual(env._d7_bridge.exposure(row), "not_admitted")
        self.assertFalse(bool(env._d7_physical[row, 0]) or bool(env._d7_visible[row, 0]))
        return scenario

    def test_episode_beyond_the_grid_runs_at_every_dynamic_level(self):
        """k = E resets and runs a natural full episode (reset -> timeout) at D0..D4 for both rows."""
        from tasks._shared.conditions.certificate import verify_dynamic_hazard_stamp
        from tasks.docking.journal_d7_exposure import record_exposure

        records = {row: {} for row in self.ROWS}
        d7_hashes = {row: set() for row in self.ROWS}
        for level in range(5):
            env = make_env(tokens=("dynamic=D%d" % level,), num_envs=2)
            env.reset()
            self.assertEqual(env._d7_bridge.exposure(0), "admitted")
            self.assertEqual(env._d7_bridge.exposure(1), "not_admitted")
            self.advance_past_the_grid(env)
            scenarios = {row: self.check_layout_of_the_beyond_grid_key(env, row) for row in self.ROWS}
            limit = 2 * max(s.deadline_steps for s in scenarios.values()) + 10
            latched, steps = {}, 0
            while len(latched) < len(self.ROWS):
                self.assertLess(steps, limit)
                env.step(torch.zeros(2, 2))
                steps += 1
                for row in self.ROWS:
                    if row not in latched:
                        self.assertFalse(bool(env._d7_physical[row, 0]) or bool(env._d7_visible[row, 0]))
                        self.assertFalse(bool(env._jd_d7_contact[row]))
                        if env.journal_episode_results[row] is not None:
                            latched[row] = (steps, env.journal_episode_results[row],
                                            env.journal_episode_record(row))
            header = env.journal_certificate_header()["dynamic_hazards_header"]
            self.assertEqual(header["grid"], {"num_envs": N, "episodes": self.E})
            self.assertEqual(header["per_dose_counts"], [[8, 5, 3]] * 5)       # beyond-grid keys are not pool keys
            for row in self.ROWS:
                row_steps, result, record = latched[row]
                self.assertLessEqual(abs(row_steps - scenarios[row].deadline_steps), 1)
                self.assertEqual(result["termination_reason_name"], "timeout")
                self.assertEqual(result["task_episode"], self.E)
                self.assertEqual((result["d7_exposure"], result["d7_requested_dose"], result["d7_run_dose"]),
                                 ("not_admitted", level, 0))
                self.assertEqual(result["d7_not_admitted_reasons"], [[0, "beyond_offered_grid", 0]])
                envelope = verify_dynamic_hazard_stamp(record["dynamic_hazards"])
                dynamic.read_stamp(record["dynamic_hazards"])
                self.assertEqual(record_exposure(record["dynamic_hazards"]), "not_admitted")
                payload = envelope["payload"]
                self.assertEqual(payload["exposure"], "not_admitted")
                self.assertEqual(payload["not_admitted"]["reasons"], self.REASON)       # exactly one reason
                self.assertEqual(payload["not_admitted"]["top_reason"], "beyond_offered_grid")
                self.assertEqual(payload["not_admitted"]["rule"], pool_io.BEYOND_GRID_RULE)
                self.assertEqual(payload["not_admitted"]["requested_dose"], level)
                assignment = payload["assignment"]
                self.assertEqual(assignment["key"], {"eval_seed": SEED, "env": row, "episode": self.E,
                                                     "hazard_group": "d7_docking"})
                self.assertEqual((assignment["task"], assignment["dose"], assignment["requested_dose"]),
                                 ("docking", 0, level))
                self.assertEqual(assignment["intruders"], [])
                self.assertEqual(assignment["not_admitted_reasons"], self.REASON)
                self.assertEqual(assignment["geometry_id"], scenarios[row].geometry_id)
                self.assertIsNotNone(assignment["candidate_digest"])
                self.assertEqual(payload["runtime"]["non_exposure_reason"], "not_admitted")
                self.assertEqual(payload["runtime"]["termination_reason"], "timeout")
                self.assertEqual(payload["runtime"]["bridge"]["last_tick"], 2 * row_steps)
                # F0 pairing accepts the stamp; the D7 primitive is the record's scenario hash
                value = pairing.d7_primitive_hash(record["dynamic_hazards"])
                self.assertEqual(value, record["scenario_hashes"][pairing.D7_PRIMITIVE])
                self.assertEqual(sorted(record["scenario_hashes"]), sorted(env.journal_pairing_inventory))
                d7_hashes[row].add(value)
                records[row][level] = record
            # trailing episodes keep running: the rows are now in k = E + 1 (also beyond the grid)
            env.step(torch.zeros(2, 2))
            for row in self.ROWS:
                self.assertEqual(int(env._jd_task_episode[row]), self.E + 1)
                self.assertEqual(env._d7_bridge.exposure(row), "not_admitted")
        for row in self.ROWS:
            self.assertEqual(len(d7_hashes[row]), 1, row)              # the same value at every level
            for level in range(1, 5):
                with self.assertRaisesRegex(pairing.PairingError, "no treatment difference"):
                    pairing.validate_pair(records[row][0], records[row][level], "dynamic")
        self.assertNotEqual(d7_hashes[0], d7_hashes[1])                 # key dependent
        # paired with the same key in another suite's run (dynamic suite absent there)
        env = make_env(tokens=("waves=D2",), num_envs=2)
        env.reset()
        self.advance_past_the_grid(env)
        env.step(torch.zeros(2, 2))
        force_out_of_bounds(env, self.ROWS)
        env.step(torch.zeros(2, 2))
        for row in self.ROWS:
            other = env.journal_episode_record(row)
            self.assertEqual(verify_dynamic_hazard_stamp(other["dynamic_hazards"])["payload"]["not_admitted"]
                             ["reasons"], self.REASON)
            for level in range(5):
                summary = pairing.validate_pair(records[row][level], other, "waves")
                self.assertTrue(summary["changed_primitives"])
                self.assertLessEqual(set(summary["changed_primitives"]), set(pairing.SUITE_PRIMITIVES["waves"]))
                self.assertNotIn(pairing.D7_PRIMITIVE, summary["changed_primitives"])

    def test_pool_answers_beyond_grid_keys_without_a_geometry_record(self):
        pool = self.pool
        for env_index in range(N):
            for k in (self.E, self.E + 1, self.E + 57):
                self.assertTrue(pool.beyond_grid(env_index, k))
                self.assertTrue(pool.beyond_grid(env_index, k, SEED))
                self.assertEqual(pool.exposure(env_index, k), "not_admitted")
                self.assertEqual(pool.not_admitted_reasons(env_index, k), ((0, "beyond_offered_grid", 0),))
                key = dynamic.EpisodeKey(SEED, env_index, k, "d7_docking")
                items = [pool.assignment(key, dose) for dose in range(5)]
                for dose, item in enumerate(items):
                    self.assertIsInstance(item, pool_io.NotAdmittedAssignment)
                    self.assertTrue(item.beyond_grid)
                    self.assertEqual((item.dose, item.requested_dose, item.key), (0, dose, key))
                    self.assertEqual(item.reasons, ((0, "beyond_offered_grid", 0),))
                    self.assertEqual(item.geometry_id,
                                     jd.build_scenario("blueboat_v2", 2, SEED, env_index, k).geometry_id)
                # dose independent, key dependent candidate digest
                self.assertEqual(len({item.record_digest for item in items}), 1)
        digests = {pool.assignment(dynamic.EpisodeKey(SEED, e, self.E, "d7_docking"), 0).record_digest
                   for e in range(N)}
        self.assertEqual(len(digests), N)
        # inside the grid nothing changed: not beyond the grid, D16 reasons are E4's
        for env_index in range(N):
            for k in range(self.E):
                self.assertFalse(pool.beyond_grid(env_index, k))
                self.assertTrue(all(r[1] != "beyond_offered_grid" for r in pool.not_admitted_reasons(env_index, k)))
        self.assertEqual(pool.per_dose_counts, ((8, 5),) * 5)
        self.assertEqual(pool.header()["grid"], {"num_envs": N, "episodes": self.E})

    def test_keys_outside_the_env_range_or_with_another_seed_fail_closed(self):
        pool = self.pool
        group = "d7_docking"
        bad_keys = (dynamic.EpisodeKey(SEED, N, 0, group),            # env outside the grid
                    dynamic.EpisodeKey(SEED, N, self.E, group),       # env outside the grid, beyond episodes
                    dynamic.EpisodeKey(SEED, N + 40, self.E + 3, group),
                    dynamic.EpisodeKey(SEED + 1, 0, self.E, group),   # another eval seed
                    dynamic.EpisodeKey(SEED + 1, 0, 0, group),
                    dynamic.EpisodeKey(SEED, 0, self.E, "d7_path_hazard"))   # another hazard group
        for key in bad_keys:
            for dose in range(5):
                with self.assertRaises(KeyError, msg=repr(key)):
                    pool.assignment(key, dose)
        with self.assertRaises(KeyError):
            pool.assignment(dynamic.EpisodeKey(SEED, 0, self.E, group), 5)     # bad dose stays refused
        self.assertFalse(pool.beyond_grid(N, self.E))
        self.assertFalse(pool.beyond_grid(0, self.E, SEED + 1))
        self.assertFalse(pool.beyond_grid(0, self.E, SEED, "d7_path_hazard"))
        self.assertFalse(pool.beyond_grid(0, True))
        for env_index, k in ((N, 0), (N, self.E), (-1, 0)):
            with self.assertRaisesRegex(ValueError, "holds no docking episode"):
                pool.exposure(env_index, k)
        with self.assertRaisesRegex(ValueError, "not beyond the offered grid"):
            pool.beyond_grid_geometry_id(N, self.E)
        # env level: the mixin turns the pool's KeyError into a clear preflight error ...
        env = make_env(num_envs=2)
        env.reset()
        env.cfg.seed = SEED + 1                   # a key with another eval seed reaches the D7 assignment
        with self.assertRaisesRegex(ValueError, "no admitted D7 assignment .*preflight error"):
            env.journal_reset(torch.tensor([0]), torch.zeros(1, dtype=torch.float64), torch.tensor([1]),
                              outage_start_s=None)
        # ... an env count beyond the pool grid or another eval seed is refused at construction
        with self.assertRaisesRegex(ValueError, "covers 4 envs; the env has 5"):
            make_env(num_envs=N + 1)
        cfg = make_cfg(seed=SEED + 1, num_envs=2)
        cfg.journal_d7_pool = pool
        with self.assertRaisesRegex(ValueError, "another vessel or evaluation seed"):
            docking_env.DockingEnv(cfg)


class G6PublishedTargets(unittest.TestCase):
    """G6: the berth is published as dock_point / dock_heading (world frame, legacy names) so
    eval_mppi (mode "dock"), classical_baseline / journal_controllers (DockingLaw) and
    _horizontal_distance (cert_v3 d0) reach it (P5: eval_mppi refused the journal docking env)."""

    def test_dock_point_and_heading_follow_the_scenario(self):
        scripts = str(REPO / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import eval_mppi
        import journal_controllers as jc
        env = make_env(num_envs=2)
        env.reset()
        for i in range(2):
            theta = env._jd_scenarios[i].berth_heading_rad
            self.assertTrue(torch.equal(env.dock_point[i], env.scene.env_origins[i, :2]))
            self.assertAlmostEqual(float(env.dock_heading[i, 0]), math.cos(theta), places=6)
            self.assertAlmostEqual(float(env.dock_heading[i, 1]), math.sin(theta), places=6)
        self.assertEqual(env.dock_point.dtype, torch.float32)
        self.assertEqual(env.dock_heading.dtype, torch.float32)
        self.assertEqual(eval_mppi._detect_mode(env), "dock")
        if hasattr(env.robot.data, "root_pos_w") and hasattr(env.robot.data, "root_com_pos_w"):  # fake data may lack them
            self.assertTrue(bool(torch.isfinite(env._horizontal_distance()).all()))
        env.step(actions(0, 2))
        vessel = env.cfg.vehicle
        controller = jc.JournalBaselineController(
            env, "docking", jc.load_gains(vessel), dt=env.control_step_s, zone_radius_m=2.5, slow_radius_m=6.0,
            tolerance_m=float(env.cfg.success_position_tolerance_m), scale=jc.length_scale(vessel))
        out = controller.act()
        self.assertEqual(tuple(out.shape), (2, 2))
        self.assertTrue(bool(torch.isfinite(out).all()))
        before = env.dock_heading.clone()
        env.cfg.seed = SEED
        env._reset_idx(torch.tensor([1]))                       # next task episode of env 1
        theta = env._jd_scenarios[1].berth_heading_rad
        self.assertAlmostEqual(float(env.dock_heading[1, 0]), math.cos(theta), places=6)
        self.assertTrue(torch.equal(env.dock_heading[0], before[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
