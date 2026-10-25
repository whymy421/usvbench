"""CPU test support for the HazardNav journal wiring (F1). Never imports Isaac Lab.

* ``install_isaaclab_stubs()``: minimal isaaclab/gymnasium stand-ins (same
  pattern as scripts/test_jbase_integration.py and F0's glue_fixtures) plus a
  STUB ``DirectRLEnv`` that follows Isaac Lab's call order (pre-physics,
  decimation x (apply_action + engine step), episode counter, dones, rewards,
  reset of done rows, observations) around the REAL HazardNavEnv methods, the
  REAL F0 JournalEnvMixin, HydroV2Runtime, schema-3 condition layer and E4
  HazardBridge. The "engine" is a toy semi-implicit rigid-body integrator
  (body-frame wrench + engine gravity) -- plumbing, not a physics claim.
* ``load_family()``: the real cfg/env modules loaded under a bare namespace
  package (tasks/__init__.py and the family __init__ files never run).
* ``journal_cfg(task, vessel, tokens)``: an E5 template cfg (the REAL class
  from journal_runtime_cfg_v2) with conditions applied (F0 journal_config).
* ``admitted_pool(...)``: a REAL E4 AdmissionPool admitted on the REAL
  journal scenarios (build_scenario + d7_context) of a small key grid.
Not a test module itself (no ``test_`` prefix).
"""
from __future__ import annotations

import copy
import importlib
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

sys.dont_write_bytecode = True
import torch

REPO = Path(__file__).resolve().parents[2]
ROOT = "_f1_tasks"
DT = 1.0 / 120.0
GRAVITY = (0.0, 0.0, -9.81)
SEED = 42


# ------------------------------------------------------------ quaternion math --
def quat_apply(q, v):
    qv = q[..., 1:]
    cross = torch.cross(qv, v, dim=-1)
    return v + 2.0 * (q[..., :1] * cross + torch.cross(qv, cross, dim=-1))


def quat_apply_inverse(q, v):
    qv = q[..., 1:]
    cross = torch.cross(qv, v, dim=-1)
    return v - 2.0 * q[..., :1] * cross + 2.0 * torch.cross(qv, cross, dim=-1)


def quat_from_angle_axis(angle, axis):
    theta = (angle / 2.0).unsqueeze(-1)
    xyz = torch.nn.functional.normalize(axis, dim=-1) * torch.sin(theta)
    return torch.cat((torch.cos(theta), xyz), dim=-1)


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


# ------------------------------------------------------------- fake engine --
class FakeRobotData:
    def __init__(self, n):
        self.root_link_pose_w = torch.zeros((n, 7), dtype=torch.float32)
        self.root_link_pose_w[:, 3] = 1.0
        self.root_com_vel_w = torch.zeros((n, 6), dtype=torch.float32)
        self.default_root_state = torch.zeros((n, 13), dtype=torch.float32)
        self.default_root_state[:, 3] = 1.0
        self.com_pos_b = torch.zeros((n, 1, 3), dtype=torch.float32)

    @property
    def root_com_pos_w(self):          # zero COM offset (v2 assets); read only at construction
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self):
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_pos_w(self):
        return self.root_link_pose_w[:, :3]

    @property
    def root_ang_vel_w(self):
        return self.root_com_vel_w[:, 3:]


class FakeRobot:
    def __init__(self, cfg, n, vessel):
        from importlib import import_module
        spec = import_module(ROOT + "._shared.vehicles_v2").get_vehicle_v2(vessel)
        self.cfg = cfg
        self.num_bodies = 1
        self.data = FakeRobotData(n)
        self.mass = float(spec.mass_kg)
        self.inertia = torch.as_tensor(spec.inertia_com_kg_m2, dtype=torch.float64)
        self.gravity = torch.as_tensor(GRAVITY, dtype=torch.float64)
        self.applied = None
        self.writes = []
        self._ALL_INDICES = torch.arange(n, dtype=torch.long)

    def write_root_state_to_sim(self, root_state, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        self.data.root_link_pose_w[ids] = root_state[:, :7].to(torch.float32)
        self.data.root_com_vel_w[ids] = root_state[:, 7:13].to(torch.float32)
        self.writes.append(ids.clone())

    def set_external_force_and_torque(self, forces, torques):
        if forces.shape[1:] != (1, 3) or torques.shape[1:] != (1, 3):
            raise ValueError("expected [N,1,3] body wrenches")
        self.applied = (forces.clone(), torques.clone())

    def integrate(self, dt):
        force_b, torque_b = (x[:, 0].to(torch.float64) for x in self.applied)
        pose = self.data.root_link_pose_w.to(torch.float64)
        vel = self.data.root_com_vel_w.to(torch.float64)
        q = pose[:, 3:7]
        r = rotation(q)
        acc = (r @ force_b.unsqueeze(-1)).squeeze(-1) / self.mass + self.gravity
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
        self.num_envs = n
        side = int(math.ceil(math.sqrt(n)))
        rows = torch.arange(n) // side
        cols = torch.arange(n) % side
        self.env_origins = torch.stack((cols.float() * spacing, rows.float() * spacing,
                                        torch.zeros(n)), -1)
        self.articulations, self.rigid_objects = {}, {}

    def clone_environments(self, copy_from_source=False):
        return None


class StubDirectRLEnv:
    """Isaac Lab DirectRLEnv call order on the CPU (see module docstring)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.cfg = cfg
        self.num_envs = int(cfg.scene.num_envs)
        self.device = torch.device("cpu")
        self.physics_dt = float(cfg.sim.dt)
        self.step_dt = self.physics_dt * int(cfg.decimation)
        self.scene = FakeScene(self.num_envs, float(getattr(cfg.scene, "env_spacing", 96.0)))
        self._setup_scene()
        self.max_episode_length = int(math.ceil(float(cfg.episode_length_s) / self.step_dt))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self.extras = {}
        self.common_step_counter = 0
        self.call_log = None

    def _log(self, name):
        if self.call_log is not None:
            self.call_log.append(name)

    def _reset_idx(self, env_ids):
        self.episode_length_buf[env_ids] = 0

    def reset(self):
        ids = torch.arange(self.num_envs, dtype=torch.long)
        self._log("reset_idx")
        self._reset_idx(ids)
        self._log("get_observations")
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.extras

    def step(self, action):
        self._log("pre_physics_step")
        self._pre_physics_step(action)
        for _ in range(int(self.cfg.decimation)):
            self._log("apply_action")
            self._apply_action()
            self.robot.integrate(self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._log("get_dones")
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self._log("get_rewards")
        self.reward_buf = self._get_rewards()
        ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self._log("reset_idx")
            self._reset_idx(ids)
        self._log("get_observations")
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated.clone(), self.reset_time_outs.clone(), self.extras


# ----------------------------------------------------------- isaaclab stubs --
class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def func(self, *args, **kwargs):
        return None


class _WithState(_Cfg):
    InitialStateCfg = _Cfg


def _configclass(cls):
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


def _stub_robot(cfg):
    return _PENDING_ROBOT.pop()


_PENDING_ROBOT = []


def install_isaaclab_stubs():
    """Install stand-ins; returns restore() (removes every module added afterwards)."""
    saved = dict(sys.modules)

    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        mod.__getattr__ = lambda attr: _Cfg
        sys.modules[name] = mod
        return mod

    root = module("isaaclab")
    root.sim = module("isaaclab.sim", SimulationCfg=_Cfg)
    root.assets = module("isaaclab.assets", ArticulationCfg=_WithState, RigidObjectCfg=_WithState,
                         RigidObject=_stub_robot, Articulation=_stub_robot)
    root.envs = module("isaaclab.envs", DirectRLEnvCfg=type("DirectRLEnvCfg", (), {}), ViewerCfg=_Cfg,
                       DirectRLEnv=StubDirectRLEnv)
    root.scene = module("isaaclab.scene", InteractiveSceneCfg=_Cfg)
    utils = root.utils = module("isaaclab.utils", configclass=_configclass)
    utils.math = module("isaaclab.utils.math", quat_apply=quat_apply, quat_apply_inverse=quat_apply_inverse,
                        quat_from_angle_axis=quat_from_angle_axis)
    gym = module("gymnasium")
    gym.spaces = SimpleNamespace(Box=_Cfg)

    def restore():
        for name in [name for name in sys.modules if name not in saved]:
            del sys.modules[name]
        sys.modules.update(saved)
    return restore


def load_family():
    """Namespace-load the real modules; returns SimpleNamespace(env, cfg, templates, journal, shared...)."""
    if ROOT not in sys.modules:
        package = ModuleType(ROOT)
        package.__path__ = [str(REPO / "tasks")]
        sys.modules[ROOT] = package
        for folder in (REPO / "tasks").iterdir():
            if folder.is_dir() and folder.name != "__pycache__":
                sub = ModuleType(ROOT + "." + folder.name)
                sub.__path__ = [str(folder)]
                sys.modules[ROOT + "." + folder.name] = sub
    imp = lambda name: importlib.import_module(ROOT + "." + name)
    return SimpleNamespace(
        env=imp("hazard_nav.hazard_nav_env"), cfg=imp("hazard_nav.hazard_nav_env_cfg"),
        templates=imp("_shared.journal_runtime_cfg_v2"), journal=imp("hazard_nav.journal_hazard_nav"),
        plans=imp("_shared.journal_cfg_v2"), contracts=imp("_shared.journal_contracts_v2"),
        journal_config=imp("_shared.conditions.journal_config"), dynamic=imp("_shared.conditions.dynamic"),
        pairing=imp("_shared.conditions.pairing"), termination=imp("_shared.termination_reason"),
        mixin=imp("_shared.journal_env_mixin"))


_CAMEL = {"reach": "Reach", "scatter": "Scatter", "ring": "Ring", "fortress": "Fortress",
          "forced_crossing": "ForcedCrossing", "iceberg": "Iceberg", "moving_basin": "MovingBasin"}
_TAG = {"blueboat_v2": "BlueBoat", "wamv_vrx_v2_opt1": "WAMV"}


def template_class(fam, task, vessel):
    return getattr(fam.templates, "%s%sJournalEnvCfg" % (_CAMEL[task], _TAG[vessel]))


def journal_cfg(fam, task, vessel="blueboat_v2", *, tokens=(), seed=SEED, num_envs=2, pool=None):
    """The REAL E5 template cfg with a resolved condition stack (F0 journal_config)."""
    cfg = template_class(fam, task, vessel)()
    cfg.sim.gravity = GRAVITY          # isaaclab SimulationCfg default; the stub Cfg has none
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = None
    resolved = fam.journal_config.resolve_journal_conditions(presets=list(tokens), seed=seed)
    fam.journal_config.apply_journal_conditions(cfg, resolved)
    if pool is not None:
        cfg.journal_d7_pool = pool
    return cfg


_POOLS = {}


def admitted_pool(fam, task, vessel, num_envs=2, episodes=(0, 1, 2), seed=SEED, suite=None):
    """REAL E4 pool over keys (seed, env, episode, d7_<task>) of the real journal scenarios."""
    key = (task, vessel, num_envs, tuple(episodes), seed)
    if key not in _POOLS:
        d = fam.dynamic
        J = fam.journal
        suite = suite if suite is not None else d.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
        ladders = []
        for env in range(num_envs):
            for episode in episodes:
                sc = J.build_scenario(task, vessel, 2, seed, env, episode)
                k = J.d7_key(sc)
                ladders.append((k, d.build_ladder(J.d7_context(sc), k, suite)))
        _POOLS[key] = d.admit_ladders(ladders)
    return _POOLS[key]


def make_env(fam, cfg):
    """Construct the REAL HazardNavEnv on the stub DirectRLEnv (robot = FakeRobot)."""
    _PENDING_ROBOT.append(FakeRobot(cfg, int(cfg.scene.num_envs), cfg.vehicle))
    return fam.env.HazardNavEnv(cfg)


def zero_action(env, value=(0.0, 0.0)):
    return torch.tensor([list(value)] * env.num_envs, dtype=torch.float32)
