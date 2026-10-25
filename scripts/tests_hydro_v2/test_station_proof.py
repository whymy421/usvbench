"""CPU proof of the station-keeping hydro_v2 wiring (INTEGRATION_SPEC.md 2/3).

The env module imports Isaac Lab, so it is NOT imported. Instead the two new
methods are extracted from the source AST and executed against a stub env;
the class _apply_action and _compute_current_forces sources are compared
byte-for-byte with the C2 copy to prove the legacy path is untouched.

E2 change: environment family files are NOT edited in round E (they are
wired in part F), so the J-base station_keeping_env.py in this tree has no
hydro_v2 proof edit. The test therefore uses the in-tree env file when it
carries the wiring and otherwise D3's proof-edited copy (read-only; override
with USVBENCH_D3_PROOF_ROOT), and skips with an explicit reason if neither
exists. Either way the extracted wiring is executed against THIS tree's
hydro_v2 runtime and registry profile (blueboat_v2), which is what part F
will reuse.
"""
import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import unittest
import torch

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
SHARED = REPO / "tasks" / "_shared"
IN_TREE_STATION = REPO / "tasks" / "station_keeping" / "station_keeping_env.py"
D3_PROOF_STATION = Path(os.environ.get(
    "USVBENCH_D3_PROOF_ROOT", "D:/claude code/codex_tasks/20260930_d3_hydro_actuation_v2/work/repo")) \
    / "tasks" / "station_keeping" / "station_keeping_env.py"
if "_apply_action_hydro_v2" in IN_TREE_STATION.read_text(encoding="utf-8"):
    STATION = IN_TREE_STATION
elif D3_PROOF_STATION.is_file():
    STATION = D3_PROOF_STATION
    print("NOTE: in-tree station_keeping_env.py has no hydro_v2 wiring (part F); exercising D3's proof "
          "copy %s against this tree's runtime" % STATION, flush=True)
else:
    raise unittest.SkipTest("no hydro_v2 station wiring in this tree (part F) and D3 proof copy absent")
C2_STATION = Path("D:/claude code/codex_tasks/20260930_c2_physics_fix/work/repo/tasks/station_keeping/"
                  "station_keeping_env.py")
sys.path.insert(0, str(SHARED))


def load(name, path=None):
    spec = importlib.util.spec_from_file_location(name, path or SHARED / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


physics = load("physics_v2")
load("restoring_v2")
hydro = load("hydro_v2")
mixer = load("thruster_mixer")
load("reference_point")
load("vehicles_v2")
load("journal_vehicle")
profiles = load("hydro_profiles")
runtime = load("hydro_v2_runtime")
ramp = load("station_current_ramp", REPO / "tasks" / "station_keeping" / "current_ramp.py")
DATUM = profiles.profile_for_vehicle("blueboat_v2").hydro.equilibrium_com_z_m


def method_sources(path, class_name):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.name: ast.get_source_segment(text, item) for item in node.body
                    if isinstance(item, ast.FunctionDef)}
    raise AssertionError("class not found")


SOURCES = method_sources(STATION, "StationKeepingEnv")


def bind(name):
    namespace = {"torch": torch, "ramped_current_vec": ramp.ramped_current_vec}
    code = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in ("    " + SOURCES[name]).splitlines())
    exec(compile(code, str(STATION) + ":" + name, "exec"), namespace)
    return namespace[name]


class Robot:
    def __init__(self, n, gen):
        quat = torch.cat((torch.ones(n, 1), torch.randn(n, 3, generator=gen)*.05), -1)
        pos = torch.randn(n, 3, generator=gen)*.02
        pos[:, 2] += DATUM
        self.data = types.SimpleNamespace(
            root_com_pos_w=pos, root_com_vel_w=torch.randn(n, 6, generator=gen)*.3,
            root_link_quat_w=quat/quat.norm(dim=-1, keepdim=True))
        self.num_bodies = 1
        self.submitted = None

    def set_external_force_and_torque(self, forces, torques):
        self.submitted = (forces.clone(), torques.clone())


class StationProof(unittest.TestCase):
    def env(self, enable_current=True, ramp_pair=None, waves=True):
        n = 5
        gen = torch.Generator().manual_seed(99)
        profile = profiles.profile_for_vehicle("blueboat_v2")
        env = types.SimpleNamespace()
        env.num_envs, env.device = n, "cpu"
        env.robot = Robot(n, gen)
        env.actions = torch.rand(n, 2, generator=gen)*2-1
        env.physics_cfg = types.SimpleNamespace(enable_current=enable_current, water_surface_z=0.0)
        env.current_vec = torch.randn(n, 2, generator=gen)*.4
        env._current_ramp = ramp_pair
        env.episode_length_buf = torch.tensor([0, 10, 100, 1000, 7199])
        env.max_episode_length = 7200
        env.scene = types.SimpleNamespace(env_origins=torch.randn(n, 3, generator=gen)*40)
        env._hydro_v2 = runtime.HydroV2Runtime(profile, n, "cpu", 1/120)
        amp = torch.rand(n, 32, generator=gen)*.02
        omega = 2*math.pi*(.3+torch.rand(n, 32, generator=gen))
        env._hydro_v2_waves = hydro.SpectralWaves(amp, omega, omega.square()/9.81,
                                                  torch.rand(n, 32, generator=gen)*6.28,
                                                  torch.rand(n, 32, generator=gen)*6.28) if waves else None
        env._current_velocity_world = types.MethodType(bind("_current_velocity_world"), env)
        env._apply_action_hydro_v2 = types.MethodType(bind("_apply_action_hydro_v2"), env)
        return env, profile

    def test_submitted_wrench_equals_direct_runtime_step(self):
        for enable_current, ramp_pair, waves in ((True, None, True), (True, (.5, 3.), True), (False, None, False)):
            env, profile = self.env(enable_current, ramp_pair, waves)
            reference = runtime.HydroV2Runtime(profile, env.num_envs, "cpu", 1/120)
            for _ in range(3):
                env._apply_action_hydro_v2()
                current = env._current_velocity_world()
                data = env.robot.data
                expected = reference.step(env.actions, data.root_com_pos_w, data.root_com_vel_w,
                                          data.root_link_quat_w, env.scene.env_origins,
                                          current_velocity_w=current, waves=env._hydro_v2_waves,
                                          mean_water_level_local=0.0)
                forces, torques = env.robot.submitted
                self.assertEqual(forces.shape, (env.num_envs, 1, 3))
                self.assertTrue(torch.equal(forces[:, 0], expected.force_actor))
                self.assertTrue(torch.equal(torques[:, 0], expected.torque_actor))
            self.assertTrue(torch.equal(env._hydro_v2.ticks, torch.full((env.num_envs,), 3)))

    def test_current_is_water_velocity_data_with_ramp_semantics(self):
        env, _ = self.env(True, (.5, 3.))
        current = env._current_velocity_world()
        expected = ramp.ramped_current_vec(env.current_vec, env.episode_length_buf, 7199, .5, 3.)
        self.assertTrue(torch.equal(current[:, :2], expected))
        self.assertTrue(torch.equal(current[:, 2], torch.zeros(env.num_envs)))
        env, _ = self.env(False)
        self.assertTrue(torch.equal(env._current_velocity_world(), torch.zeros(env.num_envs, 3)))

    def test_legacy_methods_byte_identical_to_c2(self):
        if not C2_STATION.exists():
            self.skipTest("C2 copy not present on this machine")
        c2 = method_sources(C2_STATION, "StationKeepingEnv")
        for name in ("_apply_action", "_compute_current_forces", "_pre_physics_step"):
            self.assertEqual(SOURCES[name], c2[name], name)
        self.assertNotIn("_apply_action_hydro_v2", c2)

    def test_selector_defaults_to_legacy_and_rejects_unknown(self):
        self.assertEqual(runtime.hydrodynamics_model(types.SimpleNamespace()), "legacy")
        self.assertEqual(runtime.hydrodynamics_model(types.SimpleNamespace(hydrodynamics_model="hydro_v2")),
                         "hydro_v2")
        with self.assertRaises(ValueError):
            runtime.hydrodynamics_model(types.SimpleNamespace(hydrodynamics_model="c3"))
        init = SOURCES["__init__"]
        self.assertIn('hydrodynamics_model(self.physics_cfg) == "hydro_v2"', init)
        self.assertIn("physics_version", init)
        self.assertIn("engine_gravity", init)
        self.assertIn("check_wave_band(self._hydro_v2.profile, self.cfg.sea_state)", init)


if __name__ == "__main__":
    unittest.main(verbosity=2)
