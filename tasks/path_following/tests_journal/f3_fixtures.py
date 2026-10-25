"""CPU fixtures for the F3 path-following journal tests (never imports Isaac Lab).

* ``load(name)``: repository task modules through bare namespace packages
  (root ``_f3t``; no family ``__init__`` runs, so no gym registration and no
  simulator import).
* ``install_isaaclab_stubs()``: minimal isaaclab/gymnasium stand-ins so the
  REAL ``path_following_env_cfg.py``, ``path_following_env.py`` and E5's
  ``journal_runtime_cfg_v2.py`` execute on CPU. ``DirectRLEnv`` is replaced by
  ``FakeDirectRLEnv`` which reproduces Isaac Lab 2.x's ``step()`` order
  (pre_physics, decimation x (apply_action, write, physics, update),
  episode counter, dones, rewards, reset of done rows, observations) around a
  toy 6-DOF rigid body (semi-implicit Euler of the submitted body-frame
  wrench plus engine gravity; a plumbing stand-in, not a physics claim; same
  integrator as scripts/tests_journal_glue/glue_fixtures.py).
* ``journal_cfg(vessel, ...)``: the REAL E5 template cfg with the F3 parent
  edits, conditions from ``resolve_journal_conditions`` and a D7 pool file
  admitted for exactly the keys a test needs (written to a temp folder).
* ``make_env(cfg)``: the REAL ``PathFollowingEnv`` on the fake base.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace

sys.dont_write_bytecode = True
import torch

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "tasks"
ROOT = "_f3t"
DT = 1.0 / 120.0
DECIMATION = 2
SEED = 42
FBASE = Path(r"D:\claude code\codex_tasks\20260930_fbase\work\repo")
# SHA-256 of the four legacy method sources of PathFollowingEnv in F-base (ast.get_source_segment),
# recorded from D:\claude code\codex_tasks\20260930_fbase\work\repo (MANIFEST-verified) so the
# byte-identity test also runs where F-base is absent.
FBASE_METHOD_SHA256 = {
    "_pre_physics_step": "90d7ebb341cdbfe6cad09f8b064cdf422118d7fda3e58b0d814cdb993f7681f8",
    "_apply_action": "5a176be44da18f7386edbb5bcc23df0e3dce197ab2a5e85542d7e24dd09b7f3f",
    "_get_observations": "44cd457bb2018c63a10831c5ad605d2cad2fb359085e24afc26fe0cd656b7d6f",
    "_get_rewards": "800c2c827a37ef46b36afcdf899c82ef89ac54ba05c406994a9aa355b559a060",
    "_get_dones": "04147640c22290a68597a2af222f91574d9d521e477c93ff3b5c22f7c2f11ded",
    "_reset_idx": "4ad15fc4378aebd5141e9d93a43aa5b18bf0b1ad2fc0864f0285acd361c3af8b",
    "_setup_scene": "494a849d1b03cb1a73caac100060f5d1efe89526385dcec8cba788363ade888d",
    "_compute_buoyancy_forces": "32661f08a94d8e845322c0b4e682e58acb4c0c3896eb35881b4e06f0b8bfb02d",
    "__init__": "1adbdabfb9aa3fd83ce0c7be8e83cee6ba876b75b53619efb7b31cf5faa53f0e",
}


def _namespace(name, folder):
    module = ModuleType(name)
    module.__path__ = [str(folder)]
    module.__package__ = name
    sys.modules[name] = module


if ROOT not in sys.modules:
    _namespace(ROOT, TASKS)
    for _folder in sorted(TASKS.iterdir()):
        if _folder.is_dir() and _folder.name != "__pycache__":
            _namespace(ROOT + "." + _folder.name, _folder)


def load(name):
    return importlib.import_module(ROOT + "." + name)


jpf = load("path_following.journal_path_following")
plans = load("_shared.journal_cfg_v2")
contracts = load("_shared.journal_contracts_v2")
vehicles_v2 = load("_shared.vehicles_v2")
journal_config = load("_shared.conditions.journal_config")
observation_groups = load("_shared.conditions.observation_groups")
pairing = load("_shared.conditions.pairing")
dynamic = jpf._dynamic()


# ------------------------------------------------------------- isaaclab stubs --
def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), -1)


def rotation(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
        torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
        torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)


class FakeRobotData:
    def __init__(self, n):
        self.root_link_pose_w = torch.zeros((n, 7), dtype=torch.float32)
        self.root_link_pose_w[:, 3] = 1.0
        self.root_com_vel_w = torch.zeros((n, 6), dtype=torch.float32)
        self.default_root_state = torch.zeros((n, 13), dtype=torch.float32)
        self.default_root_state[:, 3] = 1.0

    @property
    def root_link_quat_w(self):
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_pos_w(self):
        return self.root_link_pose_w[:, :3]

    @property
    def root_com_pos_w(self):
        raise AssertionError("journal code must not read root_com_pos_w (CONTEXT_R3 decision 12)")


class FakeRigidObject:
    """Stand-in for isaaclab.assets.RigidObject (one body, origin = COM)."""

    def __init__(self, cfg):
        self.cfg = cfg
        usd = Path(cfg.spawn.usd_path).name
        self.spec = next((s for s in vehicles_v2.VEHICLES_V2.values() if Path(s.asset_relpath).name == usd), None)
        self.num_bodies = 1
        self.applied = []
        self.writes = []
        self.data = None

    def bind(self, n):
        self.data = FakeRobotData(n)
        self._ALL_INDICES = torch.arange(n)
        self.mass = float(self.spec.mass_kg)
        self.inertia = torch.as_tensor(self.spec.inertia_com_kg_m2, dtype=torch.float64)

    def write_root_state_to_sim(self, root_state, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        self.writes.append((ids.clone(), root_state.clone()))
        self.data.root_link_pose_w[ids] = root_state[:, :7].to(torch.float32)
        self.data.root_com_vel_w[ids] = root_state[:, 7:13].to(torch.float32)

    def set_external_force_and_torque(self, forces, torques):
        if forces.shape[1:] != (1, 3) or torques.shape[1:] != (1, 3):
            raise ValueError("expected [N,1,3] body wrenches")
        self.applied.append((forces.clone(), torques.clone()))

    def integrate(self, dt, gravity):
        force_b, torque_b = (x[:, 0].to(torch.float64) for x in self.applied[-1])
        pose = self.data.root_link_pose_w.to(torch.float64)
        vel = self.data.root_com_vel_w.to(torch.float64)
        q = pose[:, 3:7]
        r = rotation(q)
        acc = (r @ force_b.unsqueeze(-1)).squeeze(-1) / self.mass + torch.as_tensor(gravity, dtype=torch.float64)
        v = vel[:, :3] + dt * acc
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
        side = int(math.ceil(math.sqrt(n)))
        rows = torch.arange(n) // side
        cols = torch.arange(n) % side
        self.env_origins = torch.stack((cols.float() * spacing, -rows.float() * spacing, torch.zeros(n)), -1)
        self.num_envs = n
        self.rigid_objects = {}
        self.articulations = {}

    def clone_environments(self, copy_from_source=False):
        return None

    def write_data_to_sim(self):
        return None

    def update(self, dt):
        return None

    def reset(self, env_ids=None):
        return None


class FakeDirectRLEnv:
    """Isaac Lab 2.x DirectRLEnv step()/reset() order on the toy rigid body."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.cfg = cfg
        self.render_mode = render_mode
        self.device = torch.device("cpu")
        self.num_envs = int(cfg.scene.num_envs)
        self.scene = FakeScene(self.num_envs, float(cfg.scene.env_spacing))
        self.physics_dt = float(cfg.sim.dt)
        self.step_dt = self.physics_dt * int(cfg.decimation)
        self._setup_scene()
        self.robot.bind(self.num_envs)
        self.max_episode_length = int(math.ceil(float(cfg.episode_length_s) / self.step_dt))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool)
        self.common_step_counter = 0
        self.extras = {}
        self.call_log = []

    def step(self, action):
        self._pre_physics_step(action)
        for _ in range(int(self.cfg.decimation)):
            self._apply_action()
            self.scene.write_data_to_sim()
            self.robot.integrate(self.physics_dt, self.cfg.sim.gravity)
            self.call_log.append("physics")
            self.scene.update(self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        terminated, truncated = self._get_dones()
        self.reset_terminated[:], self.reset_time_outs[:] = terminated, truncated
        reset_buf = terminated | truncated
        reward = self._get_rewards()
        ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self._reset_idx(ids)
        obs = self._get_observations()
        return obs, reward, terminated, truncated, self.extras

    def reset(self, seed=None, options=None):
        self._reset_idx(None)
        return self._get_observations(), self.extras

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self.scene.reset(env_ids)
        self.episode_length_buf[env_ids] = 0


def _quat_apply(q, v):
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _quat_apply_inverse(q, v):
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def _quat_from_angle_axis(angle, axis):
    half = 0.5 * angle
    return torch.cat((torch.cos(half), torch.sin(half) * axis), -1)


_STUB_STATE = {}


def install_isaaclab_stubs():
    """Install the stand-ins (idempotent); returns the stub module map."""
    if _STUB_STATE:
        return _STUB_STATE

    class Cfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class WithState(Cfg):
        InitialStateCfg = Cfg

    class SimulationCfg(Cfg):
        def __init__(self, **kwargs):
            self.dt = 1.0 / 60.0
            self.gravity = (0.0, 0.0, -9.81)
            super().__init__(**kwargs)

    class DomeLightCfg(Cfg):
        def func(self, path, cfg):
            return None

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
    root.sim = module("isaaclab.sim", SimulationCfg=SimulationCfg, DomeLightCfg=DomeLightCfg)
    root.assets = module("isaaclab.assets", ArticulationCfg=WithState, RigidObjectCfg=WithState,
                         RigidObject=FakeRigidObject, Articulation=FakeRigidObject)
    root.envs = module("isaaclab.envs", DirectRLEnvCfg=type("DirectRLEnvCfg", (), {}),
                       DirectRLEnv=FakeDirectRLEnv, ViewerCfg=Cfg)
    root.scene = module("isaaclab.scene", InteractiveSceneCfg=Cfg)
    utils = root.utils = module("isaaclab.utils", configclass=configclass)
    utils.math = module("isaaclab.utils.math", quat_apply=_quat_apply, quat_apply_inverse=_quat_apply_inverse,
                        quat_from_angle_axis=_quat_from_angle_axis)
    gym = module("gymnasium")
    gym.spaces = SimpleNamespace(Box=Cfg)
    _STUB_STATE.update(root=root)
    return _STUB_STATE


def env_modules():
    install_isaaclab_stubs()
    return (load("path_following.path_following_env_cfg"), load("path_following.path_following_env"),
            load("_shared.journal_runtime_cfg_v2"))


# ------------------------------------------------------------------- pools --
_POOL_DIR = Path(tempfile.mkdtemp(prefix="f3_pools_"))
_POOLS = {}


def pool_file(vessel, num_envs, episodes, seed=SEED, tier=2):
    """Admit exactly (seed, env < num_envs, episode < episodes) on the real layouts; returns the path."""
    key = (vessel, num_envs, episodes, seed, tier)
    if key not in _POOLS:
        admitted = jpf.admit_path_following_grid([vessel], seed, num_envs, episodes, tier=tier, workers=1)[vessel]
        meta = jpf.pool_meta(vessel, seed, num_envs, episodes, tier, admitted)
        path = _POOL_DIR / ("%s_%d_%d_%d" % (vessel, num_envs, episodes, seed)) / jpf.pool_file_name(vessel, seed)
        jpf.write_pool_file(path, admitted["pool"], admitted["offered"], meta)
        _POOLS[key] = path
    return _POOLS[key]


# -------------------------------------------------------------------- cfgs --
def journal_cfg(vessel="blueboat_v2", *, num_envs=2, episodes=2, tokens=(), seed=SEED, visual=False):
    """REAL E5 template cfg (F3 parent edits) + resolved conditions + a test D7 pool."""
    _, _, templates = env_modules()
    cls = {"blueboat_v2": templates.PathFollowingBlueBoatJournalEnvCfg,
           "wamv_vrx_v2_opt1": templates.PathFollowingWAMVJournalEnvCfg}[vessel]
    cfg = cls()
    cfg.scene.num_envs = int(num_envs)
    if not visual:
        cfg.visual.enable_water = False
        cfg.visual.enable_waypoint_markers = False
        cfg.visual.enable_path_line = False
    resolved = journal_config.resolve_journal_conditions(presets=list(tokens), seed=seed)
    journal_config.apply_journal_conditions(cfg, resolved)
    cfg.journal_d7_pool_path = str(pool_file(vessel, num_envs, episodes, seed))
    templates.journal_validate(cfg)
    return cfg


def make_env(cfg, trace=None):
    _, env_module, _ = env_modules()
    cls = env_module.PathFollowingEnv
    if trace is not None:
        cls = type("TracedPathFollowingEnv", (cls,), {"journal_trace": trace})
    return cls(cfg)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seek_actions(env, surge=1.0, gain=1.5):
    """Simple truth-based line-of-sight actions toward the active gate (test driver only)."""
    data = env.robot.data
    com = data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]
    q = data.root_link_pose_w[:, 3:7]
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    active = (env._journal_oracle.next_gate - 1).clamp(0, 3)
    target = env.waypoints[torch.arange(env.num_envs), active]
    bearing = torch.atan2(target[:, 1] - com[:, 1], target[:, 0] - com[:, 0])
    error = torch.atan2(torch.sin(bearing - yaw), torch.cos(bearing - yaw))
    thrust = torch.full_like(error, surge) * torch.cos(error).clamp(min=0.2)
    return torch.stack((thrust, (gain * error).clamp(-1.0, 1.0)), -1)
