"""CPU fixtures for the path-hazard journal tests (task F2). Never imports Isaac Lab.

* ``install_stubs()``: minimal isaaclab / gymnasium stand-ins (F0 glue_fixtures pattern) plus a
  ``DirectRLEnv`` stand-in that reproduces Isaac Lab's call order (reset -> _reset_idx + obs;
  step -> _pre_physics_step, decimation x (_apply_action, engine step), episode counter, _get_dones,
  _get_rewards, _reset_idx of done rows, _get_observations) and a toy 6-DOF rigid-body engine
  (semi-implicit Euler of the body wrench + engine gravity; a plumbing stand-in, not physics).
* The REAL modules are loaded: tasks/path_hazard/path_hazard_env.py, path_hazard_env_cfg.py,
  path_hazard_journal.py, the E5 template module tasks/_shared/journal_runtime_cfg_v2.py, and the
  real F0 mixin, E2 HydroV2Runtime, E3 condition layer and E4 HazardBridge behind them.
* ``test_pool``: an in-memory D7 pool admitted on the REAL journal layouts (same function as the
  offline tool, workers=1), cached per (vessel, envs, episodes); it holds admitted ladders AND
  not-admitted records (D16).
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

torch.set_num_threads(1)   # as the D1 harness; multithreaded reductions of tiny tensors are ~1000x slower here

REPO = Path(__file__).resolve().parents[3]
SEED = 42
DT = 1.0 / 120.0


def _namespaces():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if "tasks" not in sys.modules:
        module = ModuleType("tasks")
        module.__path__ = [str(REPO / "tasks")]
        sys.modules["tasks"] = module
    for folder in (REPO / "tasks").iterdir():
        name = "tasks." + folder.name
        if folder.is_dir() and folder.name not in ("_shared", "__pycache__") and name not in sys.modules:
            module = ModuleType(name)
            module.__path__ = [str(folder)]
            sys.modules[name] = module


# ------------------------------------------------------------------ math --
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


def quat_apply(q, v):
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def quat_apply_inverse(q, v):
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v - q[..., :1] * t + torch.cross(qv, t, dim=-1)


def quat_from_angle_axis(angle, axis):
    half = 0.5 * angle
    axis = axis / axis.norm(dim=-1, keepdim=True)
    return torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)


# ---------------------------------------------------------------- engine --
class FakeRobotData:
    def __init__(self, n, init_pos, init_rot):
        self.root_link_pose_w = torch.zeros((n, 7), dtype=torch.float32)
        self.root_link_pose_w[:, :3] = torch.tensor(init_pos, dtype=torch.float32)
        self.root_link_pose_w[:, 3:7] = torch.tensor(init_rot, dtype=torch.float32)
        self.root_com_vel_w = torch.zeros((n, 6), dtype=torch.float32)
        self.com_pos_b = torch.zeros((n, 3), dtype=torch.float32)
        self.default_root_state = torch.zeros((n, 13), dtype=torch.float32)
        self.default_root_state[:, :7] = self.root_link_pose_w

    @property
    def root_com_pos_w(self):          # origin = COM for v2 assets (legacy __init__ reads it once)
        return self.root_link_pose_w[:, :3]

    @property
    def root_pos_w(self):
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self):
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_ang_vel_w(self):
        return self.root_com_vel_w[:, 3:6]


class FakeRigidObject:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_bodies = 1
        self.applied = []
        self.data = None

    def bind(self, n, mass, inertia, gravity):
        init = self.cfg.init_state
        self.data = FakeRobotData(n, init.pos, init.rot)
        self.mass, self.inertia = float(mass), torch.as_tensor(inertia, dtype=torch.float64)
        self.gravity = torch.as_tensor(gravity, dtype=torch.float64)
        self._ALL_INDICES = torch.arange(n, dtype=torch.long)

    def reset(self, env_ids=None):
        return None

    def write_root_state_to_sim(self, root_state, env_ids=None):
        ids = self._ALL_INDICES if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long)
        self.data.root_link_pose_w[ids] = root_state[:, :7].to(torch.float32)
        self.data.root_com_vel_w[ids] = root_state[:, 7:13].to(torch.float32)

    def set_external_force_and_torque(self, forces, torques):
        if forces.shape[1:] != (1, 3) or torques.shape[1:] != (1, 3):
            raise ValueError("expected [N,1,3] body wrenches")
        self.applied.append((forces.detach().clone(), torques.detach().clone()))

    def integrate(self, dt):
        force_b, torque_b = (x[:, 0].to(torch.float64) for x in self.applied[-1])
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
        side = int(math.ceil(math.sqrt(n)))
        rows = [(float(i % side) * spacing, float(i // side) * spacing, 0.0) for i in range(n)]
        self.env_origins = torch.tensor(rows, dtype=torch.float32)
        self.articulations, self.rigid_objects = {}, {}

    def clone_environments(self, copy_from_source=False):
        return None

    def reset(self, env_ids=None):
        for obj in self.rigid_objects.values():
            obj.reset(env_ids)


class FakeDirectRLEnv:
    """Isaac Lab DirectRLEnv call order (isaaclab/envs/direct_rl_env.py step/reset), CPU."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.cfg = cfg
        self.num_envs = int(cfg.scene.num_envs)
        self.device = torch.device("cpu")
        self.extras = {}
        self.physics_dt = float(cfg.sim.dt)
        self.step_dt = float(cfg.sim.dt) * int(cfg.decimation)
        self.scene = FakeScene(self.num_envs, float(cfg.scene.env_spacing))
        self._setup_scene()
        mass, inertia = self._fake_mass_properties(cfg)
        self.robot.bind(self.num_envs, mass, inertia, getattr(cfg.sim, "gravity", (0.0, 0.0, -9.81)))
        self.max_episode_length_s = float(cfg.episode_length_s)
        self.max_episode_length = math.ceil(self.max_episode_length_s / self.step_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.common_step_counter = 0
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self.call_log = None

    @staticmethod
    def _fake_mass_properties(cfg):
        vehicles_v2 = importlib.import_module("tasks._shared.vehicles_v2")
        if cfg.vehicle in vehicles_v2.VEHICLES_V2:
            spec = vehicles_v2.get_vehicle_v2(cfg.vehicle)
            return spec.mass_kg, spec.inertia_com_kg_m2
        return 17.26, ((1.789, 0.0, 0.0), (0.0, 1.094, 0.0), (0.0, 0.0, 2.376))

    def _log(self, name):
        if self.call_log is not None:
            self.call_log.append(name)

    def _reset_idx(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        self.scene.reset(ids)
        self.episode_length_buf[ids] = 0

    def reset(self, seed=None, options=None):
        self._log("reset_idx")
        self._reset_idx(self.robot._ALL_INDICES)
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
        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self._log("get_rewards")
        self.reward_buf = self._get_rewards()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._log("reset_idx")
            self._reset_idx(reset_ids)
        self._log("get_observations")
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


# ----------------------------------------------------------------- stubs --
class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _WithState(_Cfg):
    InitialStateCfg = _Cfg


class _SimulationCfg(_Cfg):
    def __init__(self, **kwargs):
        kwargs.setdefault("gravity", (0.0, 0.0, -9.81))
        super().__init__(**kwargs)


class _DomeLightCfg(_Cfg):
    def func(self, path, cfg):
        return None


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


_STUBBED = {}


def install_stubs():
    """Install the isaaclab / gymnasium stand-ins once (idempotent)."""
    if _STUBBED:
        return _STUBBED
    _namespaces()

    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        mod.__getattr__ = lambda attr: _Cfg
        sys.modules[name] = mod
        return mod

    root = module("isaaclab")
    root.sim = module("isaaclab.sim", SimulationCfg=_SimulationCfg, DomeLightCfg=_DomeLightCfg)
    root.assets = module("isaaclab.assets", ArticulationCfg=_WithState, RigidObjectCfg=_WithState,
                         RigidObject=FakeRigidObject, Articulation=FakeRigidObject)
    root.envs = module("isaaclab.envs", DirectRLEnvCfg=type("DirectRLEnvCfg", (), {}), DirectRLEnv=FakeDirectRLEnv,
                       ViewerCfg=_Cfg)
    root.scene = module("isaaclab.scene", InteractiveSceneCfg=_Cfg)
    utils = root.utils = module("isaaclab.utils", configclass=_configclass)
    utils.math = module("isaaclab.utils.math", quat_apply=quat_apply, quat_apply_inverse=quat_apply_inverse,
                        quat_from_angle_axis=quat_from_angle_axis)
    if "gymnasium" not in sys.modules:
        gym = module("gymnasium", register=lambda **kwargs: None)
        gym.spaces = SimpleNamespace(Box=_Cfg)
    _STUBBED["env"] = importlib.import_module("tasks.path_hazard.path_hazard_env")
    _STUBBED["cfg"] = importlib.import_module("tasks.path_hazard.path_hazard_env_cfg")
    _STUBBED["templates"] = importlib.import_module("tasks._shared.journal_runtime_cfg_v2")
    return _STUBBED


def modules():
    _namespaces()
    return SimpleNamespace(
        journal=importlib.import_module("tasks.path_hazard.path_hazard_journal"),
        contracts=importlib.import_module("tasks._shared.journal_contracts_v2"),
        plans=importlib.import_module("tasks._shared.journal_cfg_v2"),
        geometry=importlib.import_module("tasks._shared.journal_geometry_v2"),
        journal_config=importlib.import_module("tasks._shared.conditions.journal_config"),
        pairing=importlib.import_module("tasks._shared.conditions.pairing"),
        termination=importlib.import_module("tasks._shared.termination_reason"),
        dynamic=importlib.import_module("tasks._shared.conditions.dynamic"),
        conditions=importlib.import_module("tasks._shared.conditions"),
    )


# ------------------------------------------------------------------ pools --
_POOLS = {}


def test_pool(vessel, envs=2, episodes=3, seed=SEED):
    key = (vessel, envs, episodes, seed)
    if key not in _POOLS:
        fam = modules().journal
        keys = [(vessel, e, p) for e in range(envs) for p in range(episodes)]
        pools, _ = fam.admit_pool(keys, seed=seed, tier=2, workers=1)
        _POOLS[key] = pools[vessel]
    return _POOLS[key]


# -------------------------------------------------------------------- cfg --
TEMPLATE = {"blueboat_v2": "PathHazardBlueBoatJournalEnvCfg", "wamv_vrx_v2_opt1": "PathHazardWAMVJournalEnvCfg"}
ALL_ON = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "perception.bundle=D2", "dynamic=D2")
ALL_D0 = ("waves=D0", "wind=D0", "current=D0", "actuator=D0", "perception=D0", "perception.bundle=D0", "dynamic=D0")


def journal_cfg(vessel="blueboat_v2", *, tokens=(), num_envs=2, pool=None, seed=SEED, episodes=3):
    stubs = install_stubs()
    cfg = getattr(stubs["templates"], TEMPLATE[vessel])()
    cfg.scene.num_envs = int(num_envs)
    for flag in ("enable_water", "enable_obstacles", "enable_waypoint_markers", "enable_path_line"):
        setattr(cfg.visual, flag, False)
    m = modules()
    resolved = m.journal_config.resolve_journal_conditions(presets=list(tokens), seed=seed)
    m.journal_config.apply_journal_conditions(cfg, resolved)
    cfg.journal_d7_pool = pool if pool is not None else test_pool(vessel, num_envs, episodes, seed)
    return cfg


def make_env(cfg, trace=False):
    env_module = install_stubs()["env"]
    cls = env_module.PathHazardEnv
    env = cls.__new__(cls)
    if trace:
        env.journal_trace = []
    env.__init__(cfg)
    return env


def actions(k, n):
    base = torch.tensor([[0.8, 0.15], [0.6, -0.25], [0.9, 0.05], [0.4, 0.3]], dtype=torch.float32)
    return base[:n] * (1.0 if k % 7 else 0.5)
