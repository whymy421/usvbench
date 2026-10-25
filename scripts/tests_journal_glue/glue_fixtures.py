"""Shared CPU fixtures for the F0 journal-glue tests (never imports Isaac Lab).

* ``shared(name)``: modules of tasks/_shared loaded through a bare namespace
  package (tasks/__init__.py and the family packages are never executed).
* ``FakeJournalEnv``: a minimal DirectRLEnv stand-in that composes the REAL
  JournalEnvMixin and drives the REAL HydroV2Runtime, the REAL schema-3
  condition layer and the REAL E4 HazardBridge through Isaac Lab's call
  order (pre_physics, decimation x apply_action + physics, dones, reset of
  done rows, observations). The "engine" is a toy 6-DOF rigid-body
  integrator (semi-implicit Euler, body-frame wrench, engine gravity) - a
  plumbing stand-in, not a physics claim. The task contract is E5's scalar
  ReachEpisode oracle per env (the reference the families' tensor twins
  must reproduce).
* ``d7_pool``: a REAL E4 admission pool for a synthetic reach-shaped context
  (E4 fixture geometry, scaled per vessel), every dose D0..D4 admitted
  jointly.
* ``journal_cfg``: a SimpleNamespace journal cfg with the fields the mixin
  reads (the family cfg classes are exercised in test_stacking.py).
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
import torch

REPO = Path(__file__).resolve().parents[2]
SHARED = REPO / "tasks" / "_shared"
PACKAGE = "_f0_shared"
if PACKAGE not in sys.modules:
    _pkg = ModuleType(PACKAGE)
    _pkg.__path__ = [str(SHARED)]
    sys.modules[PACKAGE] = _pkg


def shared(name):
    return importlib.import_module(PACKAGE + "." + name)


mixin_module = shared("journal_env_mixin")
conditions = shared("conditions")
journal_config = shared("conditions.journal_config")
pairing = shared("conditions.pairing")
observation_groups = shared("conditions.observation_groups")
dynamic = shared("conditions.dynamic")
plans = shared("journal_cfg_v2")
contracts = shared("journal_contracts_v2")
termination = shared("termination_reason")
vehicles_v2 = shared("vehicles_v2")

DT = 1.0 / 120.0
DECIMATION = 2
WATER_Z = 0.0
SEED = 42
RAYS = 36


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("ascii")).hexdigest()


# ----------------------------------------------------------------- D7 pool --
_POOLS = {}


def reach_context(vessel):
    rec = plans.vessel_record(vessel)
    env, _ = dynamic.envelope_from_spec_v2(vehicles_v2.get_vehicle_v2(vessel),
                                           reference_speed_mps=rec.cruise_speed_m_s,
                                           turning_radius_m=rec.turning_radius_design_m)
    u = max(env.beam_m / 0.93, env.length_m / 1.2)       # E4 fixtures.scale
    v = env.reference_speed_mps
    return dynamic.TaskContext(
        task="reach", envelope=env, bounds=(-80 * u, 80 * u, -80 * u, 80 * u),
        sensor=dynamic.SensorSpec(ray_count=RAYS, max_range_m=30.0 * rec.length_scale,
                                  source="journal 36-ray fan (synthetic test context)"),
        initial_pose=dynamic.Pose(0.0, 0.0, 0.0, v), goal=dynamic.Circle(30 * u, 0.0, 2 * u),
        deadline_s=2.0 * 30 * u / v)


def d7_pool(vessel="blueboat_v2", num_envs=2, episodes=4, seed=SEED):
    key = (vessel, num_envs, episodes, seed)
    if key not in _POOLS:
        ctx = reach_context(vessel)
        suite = dynamic.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
        ladders = []
        for env in range(num_envs):
            for episode in range(episodes):
                k = dynamic.EpisodeKey(seed, env, episode, "d7_reach")
                ladders.append((k, dynamic.build_ladder(ctx, k, suite)))
        pool = dynamic.admit_ladders(ladders)
        if pool.keys_admitted != len(ladders):
            raise RuntimeError("synthetic D7 pool did not admit every key")
        _POOLS[key] = pool
    return _POOLS[key]


# ------------------------------------------------------------------ cfg --
def journal_cfg(vessel="blueboat_v2", *, task="reach", tokens=(), seed=SEED, hooks=True, **extra):
    plan = plans.plan_class(task, vessel)(2)
    cfg = SimpleNamespace(
        journal_task=task, journal_contract_version=2, journal_mode=True, journal_geometry_tier=2,
        journal_plan=plan, vehicle=vessel, seed=None, conditions=None, condition_presets=(),
        sim=SimpleNamespace(dt=DT, gravity=(0.0, 0.0, -9.81)), decimation=DECIMATION,
        underwater_physics_cfg=SimpleNamespace(physics_version=2, hydrodynamics_model="hydro_v2",
                                               enable_current=False),
        robot_cfg=SimpleNamespace(spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=False))),
        obs_sea_state=False, appear_count=0)
    if hooks:
        setattr(cfg, observation_groups.HOOKS_FLAG, True)
    for name, value in extra.items():
        setattr(cfg, name, value)
    resolved = journal_config.resolve_journal_conditions(presets=list(tokens), seed=seed)
    journal_config.apply_journal_conditions(cfg, resolved)
    return cfg


# ------------------------------------------------------------- fake engine --
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


class FakeRobot:
    def __init__(self, n, mass, inertia, gravity):
        self.data = FakeRobotData(n)
        self.mass = float(mass)
        self.inertia = torch.as_tensor(inertia, dtype=torch.float64)
        self.gravity = torch.as_tensor(gravity, dtype=torch.float64)
        self.applied = []

    def write_root_state_to_sim(self, root_state, env_ids):
        """T25: Isaac Lab Articulation root-state write (pose wxyz + COM velocity, world), used by the
        default-on wave-following spawn (owner decision T1.35a; same arithmetic as G18's SpawnEnv)."""
        ids = torch.as_tensor(env_ids, dtype=torch.long).flatten()
        self.data.root_link_pose_w[ids] = root_state[:, :7].to(torch.float32)
        self.data.root_com_vel_w[ids] = root_state[:, 7:].to(torch.float32)

    def set_external_force_and_torque(self, forces, torques):
        if forces.shape[1:] != (1, 3) or torques.shape[1:] != (1, 3):
            raise ValueError("expected [N,1,3] body wrenches")
        self.applied.append((forces.clone(), torques.clone()))

    def integrate(self, dt):
        """Semi-implicit Euler of the last body-frame wrench plus engine gravity."""
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
        dq = torch.cat((torch.ones_like(half[:, :1]), half), -1)
        q = quat_mul(q, dq)
        q = q / q.norm(dim=-1, keepdim=True)
        r = rotation(q)
        omega_w = (r @ omega_b.unsqueeze(-1)).squeeze(-1)
        self.data.root_link_pose_w = torch.cat((p, q), -1).to(torch.float32)
        self.data.root_com_vel_w = torch.cat((v, omega_w), -1).to(torch.float32)


class FakeJournalEnv(mixin_module.JournalEnvMixin):
    """DirectRLEnv call order around the real mixin (reach contract, scalar oracle)."""

    journal_family = "hazard_nav"

    def __init__(self, cfg, num_envs=2, *, pool=None, trace=None, deadline_s=30.0, goal_xy=(30.0, 0.0),
                 axis_rad=0.0, spawn_speed=0.0):
        self.cfg, self.num_envs, self.device, self.extras = cfg, int(num_envs), torch.device("cpu"), {}
        self.journal_trace = trace
        spec = vehicles_v2.get_vehicle_v2(cfg.vehicle)
        self.robot = FakeRobot(self.num_envs, spec.mass_kg, spec.inertia_com_kg_m2,
                               getattr(cfg.sim, "gravity", (0.0, 0.0, -9.81)))
        self.scene = SimpleNamespace(env_origins=torch.stack(
            (torch.arange(self.num_envs, dtype=torch.float32) * 50.0, torch.zeros(self.num_envs),
             torch.zeros(self.num_envs)), -1))
        self.journal_setup(d7_pool=pool if pool is not None else d7_pool(cfg.vehicle, self.num_envs))
        record = plans.vessel_record(cfg.vehicle)
        self.nominal_height = record.com_height_above_waterline_m
        self.limits = contracts.ValidityLimits(**cfg.journal_plan.validity_limits)
        self.deadline_s, self.goal_xy, self.axis_rad = float(deadline_s), goal_xy, float(axis_rad)
        self.spawn_speed = float(spawn_speed)
        self.oracles = [contracts.ReachEpisode(goal_xy, 2.0, self.deadline_s, self.limits)
                        for _ in range(self.num_envs)]
        self.episode_counter = torch.full((self.num_envs,), -1, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.observed_water_levels = []
        self.rays = torch.full((self.num_envs, RAYS), 30.0 * record.length_scale)
        self.observations = []

    # ---- family hooks
    def _state(self, i, water_level):
        pose, vel = self.robot.data.root_link_pose_w[i], self.robot.data.root_com_vel_w[i]
        origin = self.scene.env_origins[i]
        com = (pose[:3] - origin).tolist()
        return contracts.RigidState(com, pose[3:7].tolist(), vel[:3].tolist(), vel[3:].tolist(),
                                    water_level_m=float(water_level), nominal_com_height_m=self.nominal_height)

    def _journal_observe_physics(self, robot_data, env_origins, water_level_local):
        levels = torch.as_tensor(water_level_local).expand(self.num_envs)
        self.observed_water_levels.append(levels.clone())
        for i, oracle in enumerate(self.oracles):
            oracle.observe_physics(self._state(i, levels[i]))

    def _journal_family_scenario_hashes(self, env_id):
        i = int(env_id)
        episode = int(self.episode_counter[i])
        return {"layout": sha({"goal": list(self.goal_xy), "radius": 2.0}),
                "rotation": sha({"axis": self.axis_rad}),
                "spawn_pose": sha({"env": i, "episode": episode, "speed": self.spawn_speed})}

    # ---- DirectRLEnv call order
    def step(self, actions):
        self.actions = self.journal_pre_physics(actions)
        for _ in range(self.cfg.decimation):
            forces, torques = self.journal_substep(self.robot.data, self.scene.env_origins, WATER_Z)
            self.robot.set_external_force_and_torque(forces, torques)
            self.robot.integrate(self.cfg.sim.dt)
        self.episode_length_buf += 1
        terminated, truncated = self._get_dones()
        done = (terminated | truncated).nonzero().flatten()
        if done.numel():
            self._reset_idx(done)
        return self._get_observations(), terminated, truncated

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs))
        return self._get_observations()

    def _get_dones(self):
        view = self.journal_pre_dones(self.robot.data, self.scene.env_origins, WATER_Z)
        levels = torch.as_tensor(view.water_level_local).expand(self.num_envs)
        codes = torch.zeros(self.num_envs, dtype=torch.long)
        control_dt = self.cfg.sim.dt * self.cfg.decimation
        for i, oracle in enumerate(self.oracles):
            t = float(self.episode_length_buf[i]) * control_dt
            decision = oracle.step(min(t, self.deadline_s), self._state(i, levels[i]),
                                   collision=bool(view.d7_contact[i]))
            codes[i] = int(decision.reason) if decision.done else 0
        return self.journal_dones(codes)

    def _reset_idx(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        data = self.robot.data
        for i in ids.tolist():
            origin = self.scene.env_origins[i]
            data.root_link_pose_w[i] = torch.tensor([origin[0], origin[1], WATER_Z + self.nominal_height,
                                                     1.0, 0.0, 0.0, 0.0])
            data.root_com_vel_w[i] = torch.tensor([self.spawn_speed, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.episode_counter[i] += 1
            self.episode_length_buf[i] = 0
            self.oracles[i].reset((0.0, 0.0))
        self.journal_reset(ids, torch.full((ids.numel(),), self.axis_rad, dtype=torch.float64),
                           self.episode_counter[ids].clone(),
                           reference_route_length_m=torch.full((ids.numel(),), float(math.hypot(*self.goal_xy)),
                                                               dtype=torch.float64))   # G28 outage rule

    def _get_observations(self):
        data = self.robot.data
        com = data.root_link_pose_w[:, :2] - self.scene.env_origins[:, :2]
        yaw = torch.atan2(2 * (data.root_link_pose_w[:, 3] * data.root_link_pose_w[:, 6]),
                          1 - 2 * data.root_link_pose_w[:, 6] ** 2)
        c, s = torch.cos(yaw), torch.sin(yaw)
        v = data.root_com_vel_w
        body = torch.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 5]), -1)
        est = self.journal_estimates(com, body, self.rays)
        goal = torch.tensor(self.goal_xy, dtype=est.position.dtype)
        obs = torch.cat((goal - est.position, est.velocity, est.rays / float(self.rays[0, 0])), -1)
        self.observations.append(obs)
        return obs


def run(cfg, steps, *, num_envs=2, actions=None, **kwargs):
    """Reset + ``steps`` control steps; returns (env, list of applied wrench pairs)."""
    env = FakeJournalEnv(cfg, num_envs, **kwargs)
    env.reset()
    for k in range(steps):
        a = actions(k) if callable(actions) else (actions if actions is not None else
                                                   torch.tensor([[0.6, 0.4], [0.3, -0.2]])[:num_envs])
        env.step(a)
    return env


# ---------------------------------------------------- stubbed isaaclab cfgs --
def install_isaaclab_stubs():
    """Minimal isaaclab/gymnasium stand-ins so the REAL cfg modules execute on CPU.

    Same pattern as scripts/test_jbase_integration.py (_stub_isaaclab). Returns a
    restore() callable that removes every module added afterwards.
    """
    saved = dict(sys.modules)

    class Cfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

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
    module("isaaclab.assets", ArticulationCfg=WithState, RigidObjectCfg=WithState)
    module("isaaclab.envs", DirectRLEnvCfg=type("DirectRLEnvCfg", (), {}), ViewerCfg=Cfg)
    module("isaaclab.scene", InteractiveSceneCfg=Cfg)
    module("isaaclab.utils", configclass=configclass)
    gym = module("gymnasium")
    gym.spaces = SimpleNamespace(Box=Cfg)

    def restore():
        for name in [name for name in sys.modules if name not in saved]:
            del sys.modules[name]
        sys.modules.update(saved)
    return restore


# --------------------------------------------- the 24 E5 journal cfg templates --
REPRESENTATIVE = {"waves": "waves=D2", "wind": "wind=D2", "current": "current=D2",
                  "actuator": "actuator=D2", "perception": "perception=D2", "dynamic": "dynamic=D2"}
FAMILY_OBLIGATIONS = (
    "C6 (F1-F6): the parent __post_init__ resolves the v2 registry name; here get_vehicle is shimmed "
    "to the legacy entry for the parent's legacy-only fields (the journal path reads the v2 vessel "
    "through journal_vehicle / journal_env_mixin)",
    "C8 (F1): HazardNavEnvCfg.validate_after_overrides accepts layout_mode='journal' (J-base "
    "hazard_nav_env_cfg.py:683-700 whitelist); here the legacy checks run as 'scatter'",
    "C8 (F2): the PathHazard BlueBoat-only assert (path_hazard_env_cfg.py:254-255) accepts the journal "
    "vessels; here the legacy post-init runs with vehicle='blueboat' and the v2 name is restored",
    "CONTEXT_R3 decision 13: underwater_physics_cfg.hydrodynamics_model = 'hydro_v2'",
    "D14: ray_count / ray_max_range_m = the task plan's ray sensor (36 rays, 30 m x length scale)",
    "D5: journal_observation_hooks_wired = True once _get_observations calls the estimate API",
)


def build_templates():
    """Instantiate E5's 24 journal cfg classes on CPU; returns {class name: cfg}.

    The REAL cfg modules run under isaaclab stubs; the FAMILY_OBLIGATIONS
    (edits F1-F6 make in their own files, ANCHORS_E5 C6/C8) are applied
    explicitly by this harness and listed in the test report.
    """
    restore = install_isaaclab_stubs()
    try:
        root = "_f0_tasks"
        package = ModuleType(root)
        package.__path__ = [str(REPO / "tasks")]
        sys.modules[root] = package
        for folder in (REPO / "tasks").iterdir():
            if folder.is_dir() and folder.name != "__pycache__":
                package = ModuleType(root + "." + folder.name)
                package.__path__ = [str(folder)]
                sys.modules[root + "." + folder.name] = package
        templates = importlib.import_module(root + "._shared.journal_runtime_cfg_v2")
        legacy = importlib.import_module(root + "._shared.vehicles")

        def get_vehicle(name):
            if name in vehicles_v2.VEHICLES_V2:
                return legacy.get_vehicle("blueboat" if name.startswith("blueboat") else "wamv")
            return legacy.get_vehicle(name)

        for name, module in list(sys.modules.items()):
            if name.startswith(root + ".") and name.endswith("_env_cfg") and hasattr(module, "get_vehicle"):
                module.get_vehicle = get_vehicle
        hazard = sys.modules[root + ".hazard_nav.hazard_nav_env_cfg"].HazardNavEnvCfg
        validate = hazard.validate_after_overrides

        def validate_after_overrides(self):
            if getattr(self, "layout_mode", None) != "journal":
                return validate(self)
            self.layout_mode = "scatter"
            try:
                return validate(self)
            finally:
                self.layout_mode = "journal"
        hazard.validate_after_overrides = validate_after_overrides
        path_hazard = sys.modules[root + ".path_hazard.path_hazard_env_cfg"].PathHazardEnvCfg
        post_init = path_hazard.__post_init__

        def path_hazard_post_init(self):
            requested = self.vehicle
            if requested in vehicles_v2.VEHICLES_V2:
                self.vehicle = "blueboat"
            try:
                return post_init(self)
            finally:
                self.vehicle = requested
        path_hazard.__post_init__ = path_hazard_post_init
        out = {}
        for name in sorted(n for n in dir(templates) if n.endswith("JournalEnvCfg")):
            cfg = getattr(templates, name)()
            cfg.underwater_physics_cfg.hydrodynamics_model = "hydro_v2"
            params = cfg.journal_plan.params
            cfg.ray_count = int(params["ray_sensor_ray_count"])
            cfg.ray_max_range_m = float(params["ray_sensor_max_range_m"])
            setattr(cfg, observation_groups.HOOKS_FLAG, True)
            out[name] = cfg
        return out
    finally:
        restore()
