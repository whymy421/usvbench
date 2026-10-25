"""G6 test helper: the REAL Isaac Lab 2.3.2 configclass / dict / string utilities on CPU.

Used by scripts/tests_g6/test_g6_runtime_cfg.py. Loads, from an Isaac Lab checkout found by
``find_reference()`` (``$ISAACLAB_REFERENCE``, else ``../IsaacLab`` next to this repository,
``C:/usvb/IsaacLab`` or ``~/IsaacLab``), the unmodified files
    source/isaaclab/isaaclab/utils/configclass.py, dict.py, string.py   (SHA-256 pinned: 2.3.2)
    source/isaaclab/isaaclab/envs/utils/spaces.py                       (if present)
into sys.modules under their real names, plus minimal stated stubs for the pieces that
import omni/carb/pxr/warp/Kit (isaaclab.utils.array, the isaaclab.sim/assets/scene/envs
cfg classes re-declared with the REAL configclass and Isaac Lab 2.3.2 field names and
defaults, and gymnasium's registry/spaces). The modules are placed in sys.modules
directly: the D1 CPU runner's import hook refuses to IMPORT isaaclab/omni (it guards
against the simulator); these are pure-Python reference files and stand-ins, no simulator.
``install()`` returns a restore() callable that removes every module it added.
"""
from __future__ import annotations

from dataclasses import MISSING
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import warnings

PINNED = {
    "configclass.py": "c9871929e46adfa567471b5f7177e7b3c234c1b1d57ffff46ef3dd82fec16ac2",
    "dict.py": "ad452303120c131182c23fad82728e53a4a2a7f2ab4353e4a98be3d7c44f5c69",
    "string.py": "1d2d0dc41097f5694d0a5bd3ac36ab919c4b4929cb862416d8a854766cd6784c",
}
REPO = Path(__file__).resolve().parents[2]


def find_reference():
    """(root, reason): an Isaac Lab checkout whose utils match the 2.3.2 pins, else (None, why)."""
    candidates = [os.environ.get("ISAACLAB_REFERENCE"), REPO.parent / "IsaacLab", "C:/usvb/IsaacLab",
                  Path.home() / "IsaacLab"]
    reasons = []
    for candidate in candidates:
        if not candidate:
            continue
        utils = Path(candidate) / "source" / "isaaclab" / "isaaclab" / "utils"
        if not (utils / "configclass.py").is_file():
            reasons.append(f"{candidate}: no Isaac Lab utils")
            continue
        bad = [name for name, pin in PINNED.items()
               if hashlib.sha256((utils / name).read_bytes()).hexdigest() != pin]
        if bad:
            reasons.append(f"{candidate}: not the Isaac Lab 2.3.2 files ({', '.join(bad)})")
            continue
        return Path(candidate), "ok"
    return None, "; ".join(reasons) or "no Isaac Lab checkout found (set ISAACLAB_REFERENCE)"


def _module(name, package=False, path=None, **attrs):
    mod = types.ModuleType(name)
    if package:
        mod.__path__ = [str(path)] if path else []
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    parent, _, leaf = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], leaf, mod)
    return mod


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    parent, _, leaf = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], leaf, mod)
    return mod


# ------------------------------------------------------------------ gymnasium
def _install_gymnasium():
    import numpy as np

    gym = _module("gymnasium", package=True)

    class EnvSpec:
        def __init__(self, id, entry_point=None, kwargs=None, **extra):
            self.id, self.entry_point, self.kwargs = id, entry_point, dict(kwargs or {})
            self.extra = extra

    registry = {}

    def register(id, entry_point=None, kwargs=None, **extra):
        if id in registry:
            warnings.warn(f"Overriding environment {id} already in registry.")
        registry[id] = EnvSpec(id, entry_point, kwargs, **extra)

    def spec(env_id):
        if env_id not in registry:
            raise KeyError(f"No registered env with id: {env_id}")
        return registry[env_id]

    def make(*_args, **_kwargs):
        raise RuntimeError("G6 CPU test: gym.make needs Isaac Sim")

    gym.registry, gym.register, gym.spec, gym.make, gym.EnvSpec = registry, register, spec, make, EnvSpec
    spaces = _module("gymnasium.spaces", package=True)

    class Space:
        pass

    class Box(Space):
        def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
            self.dtype = np.dtype(dtype)
            shape = tuple(shape) if shape is not None else np.shape(low)
            self.shape = tuple(int(s) for s in shape)
            self.low = np.broadcast_to(np.asarray(low, dtype=self.dtype), self.shape).copy()
            self.high = np.broadcast_to(np.asarray(high, dtype=self.dtype), self.shape).copy()

        def __eq__(self, other):
            return (isinstance(other, Box) and self.shape == other.shape and self.dtype == other.dtype
                    and np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high))

        def __repr__(self):
            return f"Box({self.low.tolist()}, {self.high.tolist()}, {self.shape}, {self.dtype})"

    class Discrete(Space):
        def __init__(self, n, seed=None, start=0):
            self.n = int(n)

    class MultiDiscrete(Space):
        def __init__(self, nvec, dtype=np.int64, seed=None):
            self.nvec = np.asarray(nvec, dtype=dtype)

    class Tuple(Space):
        def __init__(self, spaces, seed=None):
            self.spaces = tuple(spaces)

    class Dict(Space):
        def __init__(self, spaces=None, seed=None, **kw):
            self.spaces = dict(spaces or {}, **kw)

    for cls in (Space, Box, Discrete, MultiDiscrete, Tuple, Dict):
        cls.__module__ = "gymnasium.spaces"
        setattr(spaces, cls.__name__, cls)
    gym.spaces = spaces
    return gym


def install(root):
    """Install the real utils + stubs; returns (restore, modules)."""
    saved = dict(sys.modules)

    def restore():
        for name in [name for name in sys.modules if name not in saved]:
            del sys.modules[name]
        sys.modules.update(saved)

    import numpy as np
    import torch

    root = Path(root)
    utils_dir = root / "source" / "isaaclab" / "isaaclab" / "utils"
    _install_gymnasium()
    _module("isaaclab", package=True)
    utils = _module("isaaclab.utils", package=True, path=utils_dir)     # __init__ not executed
    _module("isaaclab.utils.array", TENSOR_TYPES={"numpy": np.ndarray, "torch": torch.Tensor},
            TENSOR_TYPE_CONVERSIONS={})
    string_mod = _load_file("isaaclab.utils.string", utils_dir / "string.py")            # REAL
    dict_mod = _load_file("isaaclab.utils.dict", utils_dir / "dict.py")                  # REAL
    configclass_mod = _load_file("isaaclab.utils.configclass", utils_dir / "configclass.py")  # REAL
    for mod in (dict_mod, string_mod):                                    # real __init__: from .dict/.string import *
        for name, value in vars(mod).items():
            if not name.startswith("_") and not isinstance(value, types.ModuleType):
                setattr(utils, name, value)
    utils.configclass = configclass_mod.configclass
    configclass = configclass_mod.configclass

    # ---- isaaclab.sim (stubs with Isaac Lab 2.3.2 fields)
    sim = _module("isaaclab.sim", package=True)
    _module("isaaclab.sim.spawners", package=True)
    _module("isaaclab.sim.spawners.from_files", package=True)
    _module("isaaclab.sim.spawners.materials", package=True)
    from_files = _module("isaaclab.sim.spawners.from_files.from_files")
    physics_materials = _module("isaaclab.sim.spawners.materials.physics_materials")

    def spawn_from_usd(prim_path, cfg, translation=None, orientation=None, **kwargs):
        raise RuntimeError("stub spawner (needs Kit)")

    def spawn_rigid_body_material(prim_path, cfg):
        raise RuntimeError("stub material spawner (needs Kit)")

    spawn_from_usd.__module__ = from_files.__name__
    spawn_rigid_body_material.__module__ = physics_materials.__name__
    from_files.spawn_from_usd = spawn_from_usd
    physics_materials.spawn_rigid_body_material = spawn_rigid_body_material

    @configclass
    class PhysxCfg:
        solver_type: int = 1
        solve_articulation_contact_last: bool = False
        min_position_iteration_count: int = 1
        max_position_iteration_count: int = 255
        min_velocity_iteration_count: int = 0
        max_velocity_iteration_count: int = 255
        enable_ccd: bool = False
        enable_stabilization: bool = False
        enable_external_forces_every_iteration: bool = False
        enable_enhanced_determinism: bool = False
        bounce_threshold_velocity: float = 0.5
        friction_offset_threshold: float = 0.04
        friction_correlation_distance: float = 0.025
        gpu_max_rigid_contact_count: int = 2 ** 23
        gpu_max_rigid_patch_count: int = 5 * 2 ** 15
        gpu_found_lost_pairs_capacity: int = 2 ** 21
        gpu_found_lost_aggregate_pairs_capacity: int = 2 ** 25
        gpu_total_aggregate_pairs_capacity: int = 2 ** 21
        gpu_collision_stack_size: int = 2 ** 26
        gpu_heap_capacity: int = 2 ** 26
        gpu_temp_buffer_capacity: int = 2 ** 24
        gpu_max_num_partitions: int = 8
        gpu_max_soft_body_contacts: int = 2 ** 20
        gpu_max_particle_contacts: int = 2 ** 20

    @configclass
    class RenderCfg:
        enable_translucency: bool | None = None
        enable_reflections: bool | None = None
        enable_global_illumination: bool | None = None
        antialiasing_mode: str | None = None
        enable_dlssg: bool | None = None
        enable_dl_denoiser: bool | None = None
        dlss_mode: int | None = None
        enable_direct_lighting: bool | None = None
        samples_per_pixel: int | None = None
        enable_shadows: bool | None = None
        enable_ambient_occlusion: bool | None = None
        dome_light_upper_lower_strategy: int | None = None
        carb_settings: dict | None = None
        rendering_mode: str | None = None

    @configclass
    class PhysicsMaterialCfg:
        func: object = MISSING

    @configclass
    class RigidBodyMaterialCfg(PhysicsMaterialCfg):
        func: object = spawn_rigid_body_material
        static_friction: float = 0.5
        dynamic_friction: float = 0.5
        restitution: float = 0.0
        friction_combine_mode: str = "average"
        restitution_combine_mode: str = "average"
        compliant_contact_stiffness: float = 0.0
        compliant_contact_damping: float = 0.0

    @configclass
    class SimulationCfg:
        physics_prim_path: str = "/physicsScene"
        device: str = "cuda:0"
        dt: float = 1.0 / 60.0
        render_interval: int = 1
        gravity: tuple = (0.0, 0.0, -9.81)
        enable_scene_query_support: bool = False
        use_fabric: bool = True
        physx: PhysxCfg = PhysxCfg()
        physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
        render: RenderCfg = RenderCfg()
        create_stage_in_memory: bool = False
        logging_level: str = "WARNING"
        save_logs_to_file: bool = True
        log_dir: str | None = None

    @configclass
    class ArticulationRootPropertiesCfg:
        articulation_enabled: bool | None = None
        enabled_self_collisions: bool | None = None
        solver_position_iteration_count: int | None = None
        solver_velocity_iteration_count: int | None = None
        sleep_threshold: float | None = None
        stabilization_threshold: float | None = None
        fix_root_link: bool | None = None

    @configclass
    class RigidBodyPropertiesCfg:
        rigid_body_enabled: bool | None = None
        kinematic_enabled: bool | None = None
        disable_gravity: bool | None = None
        linear_damping: float | None = None
        angular_damping: float | None = None
        max_linear_velocity: float | None = None
        max_angular_velocity: float | None = None
        max_depenetration_velocity: float | None = None
        max_contact_impulse: float | None = None
        enable_gyroscopic_forces: bool | None = None
        retain_accelerations: bool | None = None
        solver_position_iteration_count: int | None = None
        solver_velocity_iteration_count: int | None = None
        sleep_threshold: float | None = None
        stabilization_threshold: float | None = None

    @configclass
    class MassPropertiesCfg:
        mass: float | None = None
        density: float | None = None

    @configclass
    class CollisionPropertiesCfg:
        collision_enabled: bool | None = None
        contact_offset: float | None = None
        rest_offset: float | None = None
        torsional_patch_radius: float | None = None
        min_torsional_patch_radius: float | None = None

    @configclass
    class SpawnerCfg:
        func: object = MISSING
        visible: bool = True
        semantic_tags: list | None = None
        copy_from_source: bool = True

    @configclass
    class RigidObjectSpawnerCfg(SpawnerCfg):
        mass_props: MassPropertiesCfg | None = None
        rigid_props: RigidBodyPropertiesCfg | None = None
        collision_props: CollisionPropertiesCfg | None = None
        activate_contact_sensors: bool = False

    @configclass
    class DeformableObjectSpawnerCfg(SpawnerCfg):
        deformable_props: object | None = None

    @configclass
    class FileCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
        scale: tuple | None = None
        articulation_props: ArticulationRootPropertiesCfg | None = None
        fixed_tendons_props: object | None = None
        spatial_tendons_props: object | None = None
        joint_drive_props: object | None = None
        visual_material_path: str = "material"
        visual_material: object | None = None

    @configclass
    class UsdFileCfg(FileCfg):
        func: object = spawn_from_usd
        usd_path: str = MISSING
        variants: object | None = None

    for cls in (PhysxCfg, RenderCfg, PhysicsMaterialCfg, RigidBodyMaterialCfg, SimulationCfg,
                ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg, MassPropertiesCfg,
                CollisionPropertiesCfg, SpawnerCfg, RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg,
                FileCfg, UsdFileCfg):
        cls.__module__ = "isaaclab.sim"
        setattr(sim, cls.__name__, cls)

    # ---- isaaclab.assets
    assets = _module("isaaclab.assets", package=True)
    _module("isaaclab.assets.articulation", package=True)
    _module("isaaclab.assets.rigid_object", package=True)
    art_mod = _module("isaaclab.assets.articulation.articulation")
    rig_mod = _module("isaaclab.assets.rigid_object.rigid_object")
    Articulation = type("Articulation", (), {"__module__": art_mod.__name__})
    RigidObject = type("RigidObject", (), {"__module__": rig_mod.__name__})
    art_mod.Articulation, rig_mod.RigidObject = Articulation, RigidObject

    @configclass
    class AssetBaseCfg:
        @configclass
        class InitialStateCfg:
            pos: tuple = (0.0, 0.0, 0.0)
            rot: tuple = (1.0, 0.0, 0.0, 0.0)

        class_type: type = None
        prim_path: str = MISSING
        spawn: object | None = None
        init_state: InitialStateCfg = InitialStateCfg()
        collision_group: int = 0
        debug_vis: bool = False

    @configclass
    class ArticulationCfg(AssetBaseCfg):
        @configclass
        class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
            lin_vel: tuple = (0.0, 0.0, 0.0)
            ang_vel: tuple = (0.0, 0.0, 0.0)
            joint_pos: dict = {".*": 0.0}
            joint_vel: dict = {".*": 0.0}

        class_type: type = Articulation
        articulation_root_prim_path: str | None = None
        init_state: InitialStateCfg = InitialStateCfg()
        soft_joint_pos_limit_factor: float = 1.0
        actuators: dict = MISSING
        actuator_value_resolution_debug_print = False

    @configclass
    class RigidObjectCfg(AssetBaseCfg):
        @configclass
        class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
            lin_vel: tuple = (0.0, 0.0, 0.0)
            ang_vel: tuple = (0.0, 0.0, 0.0)

        class_type: type = RigidObject
        init_state: InitialStateCfg = InitialStateCfg()

    for cls in (AssetBaseCfg, ArticulationCfg, RigidObjectCfg):
        cls.__module__ = "isaaclab.assets"
        setattr(assets, cls.__name__, cls)

    # ---- isaaclab.scene
    scene = _module("isaaclab.scene", package=True)

    @configclass
    class InteractiveSceneCfg:
        num_envs: int = MISSING
        env_spacing: float = MISSING
        lazy_sensor_update: bool = True
        replicate_physics: bool = True
        filter_collisions: bool = True
        clone_in_fabric: bool = False

    InteractiveSceneCfg.__module__ = "isaaclab.scene"
    scene.InteractiveSceneCfg = InteractiveSceneCfg

    # ---- isaaclab.envs
    envs = _module("isaaclab.envs", package=True)
    from typing import TypeVar
    import gymnasium as gym
    SpaceType = TypeVar("SpaceType", gym.spaces.Space, int, set, tuple, list, dict)

    @configclass
    class ViewerCfg:
        eye: tuple = (7.5, 7.5, 7.5)
        lookat: tuple = (0.0, 0.0, 0.0)
        cam_prim_path: str = "/OmniverseKit_Persp"
        resolution: tuple = (1280, 720)
        origin_type: str = "world"
        env_index: int = 0
        asset_name: str | None = None
        body_name: str | None = None

    common = _module("isaaclab.envs.common", ViewerCfg=ViewerCfg, SpaceType=SpaceType)
    _module("isaaclab.envs.ui", package=True)
    ui_mod = _module("isaaclab.envs.ui.base_env_window")
    BaseEnvWindow = type("BaseEnvWindow", (), {"__module__": ui_mod.__name__})
    ui_mod.BaseEnvWindow = BaseEnvWindow
    sys.modules["isaaclab.envs.ui"].BaseEnvWindow = BaseEnvWindow

    @configclass
    class DirectRLEnvCfg:
        viewer: ViewerCfg = ViewerCfg()
        sim: SimulationCfg = SimulationCfg()
        ui_window_class_type: type | None = BaseEnvWindow
        seed: int | None = None
        decimation: int = MISSING
        is_finite_horizon: bool = False
        episode_length_s: float = MISSING
        scene: InteractiveSceneCfg = MISSING
        events: object | None = None
        observation_space: object = MISSING
        num_observations: int | None = None
        state_space: object | None = None
        num_states: int | None = None
        observation_noise_model: object | None = None
        action_space: object = MISSING
        num_actions: int | None = None
        action_noise_model: object | None = None
        rerender_on_reset: bool = False
        num_rerenders_on_reset: int = 0
        wait_for_textures: bool = True
        xr: object | None = None
        log_dir: str | None = None

    @configclass
    class ManagerBasedRLEnvCfg:
        seed: int | None = None

    for cls in (ViewerCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg):
        cls.__module__ = "isaaclab.envs"
        setattr(envs, cls.__name__, cls)
    envs.SpaceType = SpaceType
    envs.common = common
    _module("isaaclab.envs.utils", package=True)
    spaces_file = root / "source" / "isaaclab" / "isaaclab" / "envs" / "utils" / "spaces.py"
    spaces_mod = _load_file("isaaclab.envs.utils.spaces", spaces_file) if spaces_file.is_file() else None
    modules = {"configclass": configclass_mod, "dict": dict_mod, "string": string_mod, "spaces": spaces_mod}
    return restore, modules


TASK_UTILS_PINNED = {
    "hydra.py": "53553f5e19a2febf09ff923cb7fdc3cae55e4c131d32c7c901380b734359a4ee",
    "parse_cfg.py": "7869bb5d445a5957eb7184369709e85e71452d4cdb32a5a466c5ef20fa3c3bfb",
}


def install_task_utils(root):
    """After install(): the REAL isaaclab_tasks.utils package (__init__, importer, parse_cfg,
    hydra; hydra.py and parse_cfg.py SHA-256 pinned to 2.3.2). isaaclab_tasks/__init__.py (which
    imports every task and the simulator) is not executed. Returns the hydra module."""
    utils_dir = Path(root) / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "utils"
    for name, pin in TASK_UTILS_PINNED.items():
        if hashlib.sha256((utils_dir / name).read_bytes()).hexdigest() != pin:
            raise RuntimeError(f"{utils_dir / name} is not the Isaac Lab 2.3.2 file")
    _module("isaaclab_tasks", package=True, path=utils_dir.parent)
    spec = importlib.util.spec_from_file_location("isaaclab_tasks.utils", utils_dir / "__init__.py",
                                                  submodule_search_locations=[str(utils_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["isaaclab_tasks.utils"] = mod
    spec.loader.exec_module(mod)
    sys.modules["isaaclab_tasks"].utils = mod
    import importlib as _importlib
    return _importlib.import_module("isaaclab_tasks.utils.hydra")
