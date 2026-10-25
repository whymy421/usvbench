"""CPU fixtures for the F6 harbour journal tests (stubbed Isaac Lab; the simulator is never imported).

* ``install_isaaclab_stubs()``: minimal isaaclab / gymnasium stand-ins so the REAL
  harbour cfg module, the REAL E5 journal cfg templates and the REAL
  ``HarborMissionEnv`` class execute on CPU (pattern: scripts/test_jbase_integration.py,
  F0 scripts/tests_journal_glue/glue_fixtures.py). The stub DirectRLEnv follows
  Isaac Lab's step order: _pre_physics_step, decimation x (_apply_action,
  write, physics, scene update), episode_length_buf += 1, _get_dones,
  _get_rewards, _reset_idx(done rows), _get_observations.
* The "engine" is a toy 6-DOF rigid-body integrator (semi-implicit Euler of the
  body-frame wrench plus engine gravity) using the v2 registry mass/inertia -
  plumbing, not a physics claim.
* ``load(name)``: repo modules through bare namespace packages (family
  ``__init__`` files - gym registration - are never executed).
* ``journal_cfg(vessel, tokens)``: the E5 template class instantiated through
  the REAL parent cfg (task F6 edits), with the stacked conditions applied.
* ``small_pool(vessel, envs, episodes)``: a REAL E4 admission pool for the
  first keys of the seed-42 grid, admitted exactly like the offline tool.
* ``forced_pool(vessel, envs, episodes, reject)``: the same admission with the
  keys in ``reject`` given an E4 context whose deadline no nominal plan can
  meet, so E4 REALLY rejects them (explicit not-admitted records, D16); the
  keys keep their E5 layout and geometry id.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

sys.dont_write_bytecode = True
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = "_f6_tasks"
SEED = 42
FBASE = Path(r"D:\claude code\codex_tasks\20260930_fbase\work\repo")


# ------------------------------------------------------------------ math --
def rotation(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
        torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
        torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)


def quat_apply(q, v):
    return (rotation(q) @ v.unsqueeze(-1)).squeeze(-1)


def quat_apply_inverse(q, v):
    return (rotation(q).transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)


def quat_from_angle_axis(angle, axis):
    angle = torch.as_tensor(angle).reshape(-1)
    axis = torch.as_tensor(axis, dtype=angle.dtype).reshape(1, 3)
    half = 0.5 * angle.unsqueeze(-1)
    return torch.cat((torch.cos(half), axis * torch.sin(half)), -1)


# ---------------------------------------------------------- fake engine --
class FakeData:
    def __init__(self, n, init_state):
        pos = tuple(getattr(init_state, "pos", (0.0, 0.0, 0.0)))
        rot = tuple(getattr(init_state, "rot", (1.0, 0.0, 0.0, 0.0)))
        self.root_link_pose_w = torch.tensor([list(pos) + list(rot)] * n, dtype=torch.float32)
        self.root_com_vel_w = torch.zeros((n, 6), dtype=torch.float32)
        self.default_root_state = torch.tensor([list(pos) + list(rot) + [0.0] * 6] * n, dtype=torch.float32)
        self.com_pos_b = torch.zeros((n, 1, 3), dtype=torch.float32)

    @property
    def root_com_pos_w(self):
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
    NUM_ENVS = 1
    MASS = 1.0
    INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    GRAVITY = (0.0, 0.0, -9.81)

    def __init__(self, cfg):
        n = FakeRigidObject.NUM_ENVS
        self.cfg = cfg
        self.num_bodies = 1
        self._ALL_INDICES = torch.arange(n)
        self.data = FakeData(n, cfg.init_state)
        self.mass = float(FakeRigidObject.MASS)
        self.inertia = torch.as_tensor(FakeRigidObject.INERTIA, dtype=torch.float64)
        self.gravity = torch.as_tensor(FakeRigidObject.GRAVITY, dtype=torch.float64)
        self.applied = None
        self.history = []
        self.writes = []

    def write_root_state_to_sim(self, root_state, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        self.data.root_link_pose_w[ids] = root_state[:, :7].to(torch.float32)
        self.data.root_com_vel_w[ids] = root_state[:, 7:13].to(torch.float32)
        self.writes.append((ids.clone(), root_state.clone()))

    def set_external_force_and_torque(self, forces, torques):
        if tuple(forces.shape[1:]) != (1, 3) or tuple(torques.shape[1:]) != (1, 3):
            raise ValueError("expected [N,1,3] body wrenches")
        self.applied = (forces.clone(), torques.clone())
        self.history.append(self.applied)

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


class StubDirectRLEnv:
    """Isaac Lab DirectRLEnv call order around the family hooks (CPU)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.cfg = cfg
        self.num_envs = int(cfg.scene.num_envs)
        self.device = torch.device("cpu")
        self.physics_dt = float(cfg.sim.dt)
        self.step_dt = self.physics_dt * int(cfg.decimation)
        self.max_episode_length = int(math.ceil(float(cfg.episode_length_s) / self.step_dt))
        self.extras = {}
        spacing = float(cfg.scene.env_spacing)
        n = self.num_envs
        cols = int(math.ceil(math.sqrt(n)))
        origins = torch.zeros((n, 3))
        for i in range(n):
            origins[i, 0] = (i // cols) * spacing - 0.5 * spacing
            origins[i, 1] = (i % cols) * spacing - 0.25 * spacing
        self.scene = SimpleNamespace(env_origins=origins, articulations={}, rigid_objects={},
                                     clone_environments=lambda **_: None)
        FakeRigidObject.NUM_ENVS = n
        self._setup_scene()
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.common_step_counter = 0
        self.reset_terminated = torch.zeros(n, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(n, dtype=torch.bool)
        self.reset_buf = torch.zeros(n, dtype=torch.bool)
        self.reward_buf = torch.zeros(n)
        self.call_log = []

    def _reset_idx(self, env_ids):
        self.episode_length_buf[torch.as_tensor(env_ids, dtype=torch.long)] = 0

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs))
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.extras

    def step(self, action):
        self._pre_physics_step(action)
        for _ in range(int(self.cfg.decimation)):
            self._apply_action()
            self.robot.integrate(self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        terminated, truncated = self._get_dones()
        self.reset_terminated[:] = terminated
        self.reset_time_outs[:] = truncated
        self.reset_buf = terminated | truncated
        self.reward_buf = self._get_rewards()
        ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self._reset_idx(ids)
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated.clone(), self.reset_time_outs.clone(), self.extras


# ------------------------------------------------------------ stubs --
def install_isaaclab_stubs():
    """Install isaaclab / gymnasium stand-ins in sys.modules (idempotent)."""
    if "isaaclab" in sys.modules and getattr(sys.modules["isaaclab"], "_f6_stub", False):
        return

    class Cfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class WithState(Cfg):
        InitialStateCfg = Cfg

    class SimulationCfg(Cfg):
        def __init__(self, **kwargs):
            self.dt, self.render_interval, self.gravity, self.device = 1.0 / 60.0, 1, (0.0, 0.0, -9.81), "cpu"
            super().__init__(**kwargs)

    class DomeLightCfg(Cfg):
        def func(self, *args, **kwargs):
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

    root = module("isaaclab", _f6_stub=True)
    root.sim = module("isaaclab.sim", SimulationCfg=SimulationCfg, DomeLightCfg=DomeLightCfg)
    root.assets = module("isaaclab.assets", ArticulationCfg=WithState, RigidObjectCfg=WithState,
                         RigidObject=FakeRigidObject, Articulation=FakeRigidObject)
    root.envs = module("isaaclab.envs", DirectRLEnvCfg=type("DirectRLEnvCfg", (), {}), ViewerCfg=Cfg,
                       DirectRLEnv=StubDirectRLEnv)
    root.scene = module("isaaclab.scene", InteractiveSceneCfg=Cfg)
    utils = module("isaaclab.utils", configclass=configclass)
    utils.math = module("isaaclab.utils.math", quat_apply=quat_apply, quat_apply_inverse=quat_apply_inverse,
                        quat_from_angle_axis=quat_from_angle_axis)
    root.utils = utils
    gym = module("gymnasium")
    gym.spaces = SimpleNamespace(Box=Cfg)
    gym.register = lambda **kwargs: None


def _namespaces():
    if ROOT in sys.modules:
        return
    package = ModuleType(ROOT)
    package.__path__ = [str(REPO / "tasks")]
    sys.modules[ROOT] = package
    for folder in (REPO / "tasks").iterdir():
        if folder.is_dir() and folder.name != "__pycache__":
            sub = ModuleType(ROOT + "." + folder.name)
            sub.__path__ = [str(folder)]
            sys.modules[ROOT + "." + folder.name] = sub


def load(name):
    """A repo module (dotted path below tasks/) through the bare namespace packages."""
    _namespaces()
    return importlib.import_module(ROOT + "." + name)


# pure modules (no simulator import)
hj = load("harbor_mission.harbor_journal")
hpool = load("harbor_mission.harbor_d7_pool")
plans = load("_shared.journal_cfg_v2")
contracts = load("_shared.journal_contracts_v2")
dynamic = load("_shared.conditions.dynamic")
journal_config = load("_shared.conditions.journal_config")
pairing = load("_shared.conditions.pairing")
vehicles_v2 = load("_shared.vehicles_v2")
geometry = load("_shared.journal_geometry_v2")


def env_module():
    install_isaaclab_stubs()
    return load("harbor_mission.harbor_mission_env")


def templates():
    install_isaaclab_stubs()
    return load("_shared.journal_runtime_cfg_v2")


# ------------------------------------------------------------ pools --
_POOLS = {}


def envelope(vessel):
    record = plans.vessel_record(vessel)
    env, _ = dynamic.envelope_from_spec_v2(vehicles_v2.get_vehicle_v2(vessel),
                                           reference_speed_mps=record.cruise_speed_m_s,
                                           turning_radius_m=record.turning_radius_design_m)
    return env


def small_pool(vessel="blueboat_v2", envs=2, episodes=2, seed=SEED, tier=2):
    """REAL E4 pool for the first envs x episodes keys (same routine as the offline tool)."""
    key = (vessel, envs, episodes, seed, tier)
    if key not in _POOLS:
        suite = journal_config.resolve_journal_conditions(presets=("dynamic=D2",), seed=seed).dynamic_config(vessel)
        grid = [(e, k) for e in range(envs) for k in range(episodes)]
        pool, info = hj.admit_harbour_grid(dynamic, vessel, tier, seed, grid, suite, envelope(vessel))
        if info["not_admitted"]:
            raise RuntimeError("fixture keys not admitted: %r" % (list(info["not_admitted"]),))
        meta = hj.pool_meta(info, vessel, tier, seed, suite, envelope(vessel), grid={"envs": envs,
                                                                                      "episodes": episodes})
        _POOLS[key] = (pool, meta)
    return _POOLS[key]


FORCED_DEADLINE_S = 5.0


def forced_pool(vessel="blueboat_v2", envs=2, episodes=2, reject=((1, 0),), seed=SEED, tier=2):
    """REAL E4 pool where the keys in ``reject`` are rejected by E4 (deadline 5 s in their E4 context)."""
    import dataclasses

    reject = tuple(sorted(tuple(r) for r in reject))
    key = ("forced", vessel, envs, episodes, reject, seed, tier)
    if key not in _POOLS:
        suite = journal_config.resolve_journal_conditions(presets=("dynamic=D2",), seed=seed).dynamic_config(vessel)
        grid = [(e, k) for e in range(envs) for k in range(episodes)]

        def context(layout, env_envelope, dyn, episode_key):
            ctx = hj.d7_task_context(layout, env_envelope, dyn)
            if (episode_key.env, episode_key.episode) in reject:
                ctx = dataclasses.replace(ctx, deadline_s=FORCED_DEADLINE_S)
            return ctx

        pool, info = hj.admit_harbour_grid(dynamic, vessel, tier, seed, grid, suite, envelope(vessel),
                                           context_fn=context)
        got = sorted((k.env, k.episode) for k in info["not_admitted"])
        if got != list(reject):
            raise RuntimeError("forced rejection produced not_admitted %r, wanted %r" % (got, reject))
        meta = hj.pool_meta(info, vessel, tier, seed, suite, envelope(vessel),
                            grid={"envs": envs, "episodes": episodes, "forced_reject": [list(r) for r in reject]})
        _POOLS[key] = (pool, meta, info)
    return _POOLS[key]


# ------------------------------------------------------------ cfgs --
TEMPLATE = {"blueboat_v2": "HarbourMissionBlueBoatJournalEnvCfg", "wamv_vrx_v2_opt1": "HarbourMissionWAMVJournalEnvCfg"}


def journal_cfg(vessel="blueboat_v2", tokens=(), num_envs=2, seed=SEED, pool=True, spawn_phase=0, pool_episodes=2,
                reject=None, **extra):
    module = templates()
    cls = getattr(module, TEMPLATE[vessel])
    if spawn_phase != 0:
        cls = type(cls.__name__ + "Dock", (cls,), {"spawn_phase": spawn_phase})
        cls = sys.modules["isaaclab.utils"].configclass(cls)
    cfg = cls()
    cfg.scene.num_envs = num_envs
    resolved = journal_config.resolve_journal_conditions(presets=list(tokens), seed=seed)
    journal_config.apply_journal_conditions(cfg, resolved)
    if pool and reject is not None:
        raw, meta, _info = forced_pool(vessel, envs=max(2, num_envs), episodes=pool_episodes, reject=reject,
                                       seed=seed)
        cfg.journal_d7_pool = hpool.HarbourPoolView(raw, meta)
    elif pool:
        raw, meta = small_pool(vessel, envs=max(2, num_envs), episodes=pool_episodes, seed=seed)
        cfg.journal_d7_pool = hpool.HarbourPoolView(raw, meta)
    for name, value in extra.items():
        setattr(cfg, name, value)
    return cfg


def make_env(vessel="blueboat_v2", tokens=(), num_envs=2, **kwargs):
    module = env_module()
    cfg = journal_cfg(vessel, tokens, num_envs=num_envs, **kwargs)
    spec = vehicles_v2.get_vehicle_v2(vessel)
    FakeRigidObject.MASS = spec.mass_kg
    FakeRigidObject.INERTIA = spec.inertia_com_kg_m2
    return module.HarborMissionEnv(cfg)


# ------------------------------------------------------- controllers --
def phase_target_controller(env):
    """A simple staged pursuit driven ONLY by journal_phase_targets + journal_estimate (+ true heading)."""
    targets = env.journal_phase_targets()
    est = env.journal_estimate
    position = est.position.to(torch.float64)
    forward = env._forward_2d().to(torch.float64)
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    index = targets["active_gate_index"]
    midpoint = targets["active_gate_midpoint"].clone()
    normal = targets["active_gate_normal"]
    berth = targets["berth_pose"]
    # aim a little beyond the gate plane (so the crossing has normal speed)
    ahead = torch.where((index < 3).unsqueeze(-1), midpoint + 2.0 * normal, midpoint)
    ahead = torch.where((index == 3).unsqueeze(-1), midpoint + 0.5 * normal, ahead)
    delta = ahead - position
    bearing = torch.atan2(delta[:, 1], delta[:, 0])
    error = torch.atan2(torch.sin(bearing - heading), torch.cos(bearing - heading))
    docking = index == 4
    error = torch.where(docking, torch.atan2(torch.sin(berth[:, 2] - heading), torch.cos(berth[:, 2] - heading))
                        + 0.8 * torch.atan2(-((position[:, 0] - berth[:, 0]) * -torch.sin(berth[:, 2])
                                              + (position[:, 1] - berth[:, 1]) * torch.cos(berth[:, 2])),
                                             torch.ones_like(heading)), error)
    distance = torch.linalg.vector_norm(berth[:, :2] - position, dim=-1)
    thrust = torch.where(index <= 2, torch.full_like(error, 0.6), torch.full_like(error, 0.35))
    along = (berth[:, 0] - position[:, 0]) * torch.cos(berth[:, 2]) + (berth[:, 1] - position[:, 1]) * torch.sin(berth[:, 2])
    surge = est.velocity[:, 0].to(torch.float64)
    thrust = torch.where(docking, torch.clamp(0.4 * along - 0.8 * surge, -0.5, 0.5), thrust)
    thrust = thrust * torch.clamp(torch.cos(error), min=0.0)
    yaw = torch.clamp(1.5 * error, -1.0, 1.0)
    del distance
    return torch.stack((thrust, yaw), -1).to(torch.float32)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("ascii")).hexdigest()
