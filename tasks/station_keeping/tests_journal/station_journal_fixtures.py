"""CPU fixtures for the F4 station-keeping journal tests (never imports Isaac Lab).

* ``ns(name)``: repository task modules loaded through bare namespace
  packages (no task package initializer runs: no gym registration, no
  simulator import).
* ``install_isaaclab_stubs()``: minimal isaaclab / gymnasium stand-ins so the
  REAL cfg modules and the REAL station_keeping_env.py execute on CPU. The
  DirectRLEnv stand-in reproduces Isaac Lab 2.x DirectRLEnv.step's call order
  (pre_physics -> decimation x (apply_action, sim step) -> episode counter ->
  dones -> rewards -> reset of done rows -> observations); the rigid object
  is a toy 6-DOF integrator (semi-implicit Euler, body-frame wrench, engine
  gravity) - plumbing, not a physics claim (same as F0 glue_fixtures).
* ``journal_cfg(vessel, ...)``: E5's real template class
  (StationKeeping<BlueBoat|WAMV>JournalEnvCfg) built through the F4 parent
  cfg branch, with an evaluation seed and resolved suites (F0 journal_config).
* ``replay_scalar(trace)``: replays an env's recorded oracle inputs through
  the SCALAR journal_contracts_v2.StationKeepingEpisode (the reference) and
  returns its per-step codes and completed results.
* ``same(a, b)``: bitwise comparison of evaluator-field dicts.
"""
from __future__ import annotations

import copy
import importlib
import math
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace

sys.dont_write_bytecode = True
import torch

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "tasks"
ROOT = "_f4_tasks"
DT = 1.0 / 120.0
SEED = 42


def _install_namespace():
    if ROOT in sys.modules:
        return
    package = ModuleType(ROOT)
    package.__path__ = [str(TASKS)]
    sys.modules[ROOT] = package
    for folder in TASKS.iterdir():
        if folder.is_dir() and folder.name != "__pycache__":
            sub = ModuleType(ROOT + "." + folder.name)
            sub.__path__ = [str(folder)]
            sys.modules[ROOT + "." + folder.name] = sub


def ns(name):
    _install_namespace()
    return importlib.import_module(ROOT + "." + name)


# ------------------------------------------------------------ fake engine --
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


class FakeRigidObject:
    """Stand-in for isaaclab.assets.RigidObject (one body, root-link pose = COM)."""

    instances = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_bodies = 1
        self.applied = []
        self.writes = []
        FakeRigidObject.instances.append(self)

    def bind(self, num_envs, vessel):
        vehicles_v2 = ns("_shared.vehicles_v2")
        spec = vehicles_v2.get_vehicle_v2(vessel)
        self.mass = float(spec.mass_kg)
        self.inertia = torch.as_tensor(spec.inertia_com_kg_m2, dtype=torch.float64)
        self.gravity = torch.tensor((0.0, 0.0, -9.81), dtype=torch.float64)
        n = int(num_envs)
        init = self.cfg.init_state
        default = torch.zeros((n, 13), dtype=torch.float32)
        default[:, :3] = torch.tensor(init.pos, dtype=torch.float32)
        default[:, 3:7] = torch.tensor(init.rot, dtype=torch.float32)
        self.data = SimpleNamespace(
            root_link_pose_w=default[:, :7].clone(), root_com_vel_w=torch.zeros((n, 6)),
            default_root_state=default, body_com_pos_b=torch.zeros((n, 1, 3)))
        self._ALL_INDICES = torch.arange(n, dtype=torch.long)

    def write_root_state_to_sim(self, root_state, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        self.writes.append(ids.clone())
        self.data.root_link_pose_w[ids] = root_state[:, :7].to(torch.float32)
        self.data.root_com_vel_w[ids] = root_state[:, 7:13].to(torch.float32)

    def set_external_force_and_torque(self, forces, torques):
        if forces.shape[1:] != (1, 3) or torques.shape[1:] != (1, 3):
            raise ValueError("expected [N,1,3] body wrenches")
        self.applied.append((forces.clone(), torques.clone()))

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
    def __init__(self, num_envs, spacing):
        n = int(num_envs)
        side = int(math.ceil(math.sqrt(n)))
        idx = torch.arange(n)
        self.env_origins = torch.stack(((idx // side).float() * spacing, (idx % side).float() * spacing,
                                        torch.zeros(n)), -1)
        self.rigid_objects = {}
        self.articulations = {}
        self.calls = []

    def clone_environments(self, copy_from_source=False):
        self.calls.append("clone")

    def reset(self, env_ids=None):
        self.calls.append("scene_reset")

    def write_data_to_sim(self):
        self.calls.append("write")

    def update(self, dt):
        self.calls.append("update")


class FakeDirectRLEnv:
    """Isaac Lab 2.x DirectRLEnv call order on CPU (the parts the family uses)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.cfg = cfg
        self.render_mode = render_mode
        self.num_envs = int(cfg.scene.num_envs)
        self.device = torch.device("cpu")
        self.scene = FakeScene(self.num_envs, float(cfg.scene.env_spacing))
        self.extras = {}
        self.physics_dt = float(cfg.sim.dt)
        self.step_dt = self.physics_dt * int(cfg.decimation)
        self._setup_scene()
        self.robot.bind(self.num_envs, cfg.vehicle)
        self.max_episode_length = math.ceil(float(cfg.episode_length_s) / self.step_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.common_step_counter = 0
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool)

    def _reset_idx(self, env_ids):
        self.scene.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

    def reset(self):
        self._reset_idx(self.robot._ALL_INDICES)
        self.scene.write_data_to_sim()
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.extras

    def step(self, action):
        self._pre_physics_step(action)
        for _ in range(int(self.cfg.decimation)):
            self._apply_action()
            self.scene.write_data_to_sim()
            self.robot.integrate(self.physics_dt)
            self.scene.update(self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)
            self.scene.write_data_to_sim()
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated.clone(), self.reset_time_outs.clone(), \
            self.extras


# ---------------------------------------------------------------- stubs --
_SAVED = None


def install_isaaclab_stubs():
    """Install stand-ins; returns restore() (removes every module added afterwards)."""
    global _SAVED
    saved = dict(sys.modules)

    class Cfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class WithState(Cfg):
        InitialStateCfg = Cfg

    class LightCfg(Cfg):
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

    root = module("isaaclab")
    root.sim = module("isaaclab.sim", SimulationCfg=Cfg, DomeLightCfg=LightCfg)
    root.utils = module("isaaclab.utils", configclass=configclass)
    root.utils.math = module("isaaclab.utils.math")
    root.assets = module("isaaclab.assets", ArticulationCfg=WithState, RigidObjectCfg=WithState,
                         RigidObject=FakeRigidObject, Articulation=FakeRigidObject)
    root.envs = module("isaaclab.envs", DirectRLEnvCfg=type("DirectRLEnvCfg", (), {}),
                       DirectRLEnv=FakeDirectRLEnv, ViewerCfg=Cfg)
    root.scene = module("isaaclab.scene", InteractiveSceneCfg=Cfg)
    gym = module("gymnasium")
    gym.spaces = SimpleNamespace(Box=Cfg)
    _SAVED = saved

    def restore():
        for name in [name for name in sys.modules if name not in saved]:
            del sys.modules[name]
        sys.modules.update(saved)
    return restore


# ------------------------------------------------------------------ cfgs --
def journal_cfg(vessel="blueboat_v2", *, num_envs=2, seed=SEED, tokens=(), pool=None, **extra):
    """E5's real template (under stubs) through the F4 parent branch + seed + suites."""
    templates = ns("_shared.journal_runtime_cfg_v2")
    journal_config = ns("_shared.conditions.journal_config")
    name = "StationKeeping%sJournalEnvCfg" % ("BlueBoat" if vessel == "blueboat_v2" else "WAMV")
    cfg = getattr(templates, name)()
    cfg.scene.num_envs = int(num_envs)
    cfg.sim.gravity = (0.0, 0.0, -9.81)      # Isaac Lab SimulationCfg default (the stub has no defaults)
    cfg.visual.enable_water = False
    cfg.visual.enable_hold_zone_marker = False
    for key, value in extra.items():
        setattr(cfg, key, value)
    resolved = journal_config.resolve_journal_conditions(presets=list(tokens), seed=seed)
    journal_config.apply_journal_conditions(cfg, resolved)
    if pool is not None:
        cfg.journal_d7_pool = pool
    return cfg


def env_module():
    return ns("station_keeping.station_keeping_env")


def make_env(cfg, trace=False):
    env = env_module().StationKeepingEnv(cfg)
    if trace:
        env.journal_contract_trace = []
    return env


def hold_controller(env):
    """Simple P controller on the ESTIMATED state (drives toward the station and brakes)."""
    js = ns("station_keeping.journal_station")
    est = env.journal_estimate
    pose = env.robot.data.root_link_pose_w
    fwd = js.bow_axis_xy(pose[:, 3:7])
    rel = env._jstation.to(torch.float32) - est.position.to(torch.float32)
    heading_err = torch.atan2(fwd[:, 0] * rel[:, 1] - fwd[:, 1] * rel[:, 0], (fwd * rel).sum(-1))
    distance = rel.norm(dim=-1)
    surge = est.velocity[:, 0].to(torch.float32)
    thrust = torch.clamp(0.4 * distance - 0.8 * surge, -1.0, 1.0) * torch.cos(heading_err).clamp(min=0.0)
    turn = torch.clamp(1.5 * heading_err - 0.3 * est.velocity[:, 2].to(torch.float32), -1.0, 1.0)
    return torch.stack((thrust, turn), -1)


# ------------------------------------------------------------ comparison --
def _bits(value):
    return struct.pack("<d", float(value))


def same(a, b, path="result"):
    """Bitwise equality of nested evaluator dicts (floats by their IEEE bits; nan == nan)."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
            raise AssertionError("%s: key sets differ: %s vs %s" % (path, sorted(a) if isinstance(a, dict) else a,
                                                                  sorted(b) if isinstance(b, dict) else b))
        for key in a:
            same(a[key], b[key], path + "." + key)
        return True
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError("%s: length differs" % path)
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, "%s[%d]" % (path, i))
        return True
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        if a != b or type(a) is not type(b):
            raise AssertionError("%s: %r != %r" % (path, a, b))
        return True
    if isinstance(a, int) and isinstance(b, int):
        if a != b:
            raise AssertionError("%s: %r != %r" % (path, a, b))
        return True
    if math.isnan(float(a)) and math.isnan(float(b)):
        return True
    if _bits(a) != _bits(b):
        raise AssertionError("%s: %r != %r (bitwise)" % (path, a, b))
    return True


def rigid_state(state, i):
    contracts = ns("_shared.journal_contracts_v2")
    return contracts.RigidState(tuple(state.com_xyz[i].tolist()), tuple(state.quat_wxyz[i].tolist()),
                                tuple(state.lin_vel[i].tolist()), tuple(state.ang_vel[i].tolist()),
                                water_level_m=float(state.water_level[i]),
                                nominal_com_height_m=float(state.nominal_height[i]))


def replay_scalar(trace, limits):
    """Replay recorded oracle inputs through the scalar oracle; returns (codes per step, completed list)."""
    contracts = ns("_shared.journal_contracts_v2")
    oracles = {}
    codes_out, completed = [], []
    validity = contracts.ValidityLimits(**limits)
    for event in trace:
        kind = event[0]
        if kind == "reset":
            _, ids, start, station, radius, deadline, hold_s, late, window = event
            for j, i in enumerate(ids.tolist()):
                oracle = contracts.StationKeepingEpisode(
                    tuple(station[j].tolist()), float(radius[j]), float(deadline[j]), hold_s=hold_s,
                    late_interval_s=tuple(late[j].tolist()), top_dose_start_s=0.0,
                    final_window_s=float(window[j]), validity_limits=validity)
                oracle.reset(tuple(start[j].tolist()))
                oracles[i] = oracle
        elif kind == "observe":
            state = event[1]
            for i, oracle in oracles.items():
                oracle.observe_physics(rigid_state(state, i))
        elif kind == "step":
            _, time_s, state, collision, oob = event
            row = []
            for i in range(len(time_s)):
                oracle = oracles.get(i)
                if oracle is None or oracle.done:
                    row.append(0)
                    continue
                decision = oracle.step(float(time_s[i]), rigid_state(state, i), collision=bool(collision[i]),
                                       out_of_bounds=bool(oob[i]))
                row.append(int(decision.reason) if decision.done else 0)
                if decision.done:
                    completed.append((i, oracle.last_completed.evaluator_fields()))
            codes_out.append(row)
    return codes_out, completed


def trace_codes(trace):
    return [event[1].tolist() for event in trace if event[0] == "codes"]


def trace_completed(trace):
    return [(event[1], event[2]) for event in trace if event[0] == "completed"]


def trace_reset_steps(trace):
    """Deadline (control steps) of every row of the FIRST reset event of a trace."""
    event = next(e for e in trace if e[0] == "reset")
    control_dt = ns("_shared.journal_cfg_v2").CONTROL_DT_S
    return [int(round(float(d) / control_dt)) for d in event[5]]
