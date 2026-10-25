"""F0 / CONTEXT_R4 D10: tasks/journal/__init__.py registers exactly E5's 24 journal ids.

Static (AST/regex, no import of any family package) plus an execution of the
module with a recording gymnasium stand-in and without gymnasium. CPU only;
run: python -B test_registration.py
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glue_fixtures as G  # noqa: E402

TASKS = G.REPO / "tasks"
JOURNAL_INIT = TASKS / "journal" / "__init__.py"
RUN_CELL_ID = re.compile(r"""\bid\s*=\s*(['"])(Isaac-[A-Za-z0-9_.\-]+)\1""")   # scripts/run_cell.py _ID
DRIFT_ID = re.compile(r'id="(Isaac-[^"]+)"')                                   # test_registry_drift / E5
CARRIER_FILES = {"hazard_nav": ("hazard_nav/hazard_nav_env.py", "HazardNavEnv"),
                 "path_following": ("path_following/path_following_env.py", "PathFollowingEnv"),
                 "station_keeping": ("station_keeping/station_keeping_env.py", "StationKeepingEnv"),
                 "docking": ("docking/docking_env.py", "DockingEnv"),
                 "path_hazard": ("path_hazard/path_hazard_env.py", "PathHazardEnv"),
                 "harbor_mission": ("harbor_mission/harbor_mission_env.py", "HarborMissionEnv")}
D1_CALM_CLASSES = ("HazardNavV3JournalEnvCfg", "HazardForcedCrossingJournalEnvCfg")


def execute_registration(gym_module):
    """Run tasks/journal/__init__.py as package 'tasks.journal' with ``gym_module``."""
    saved = {name: sys.modules.get(name) for name in ("gymnasium", "tasks", "tasks.journal")}
    try:
        if gym_module is None:
            sys.modules["gymnasium"] = None             # import gymnasium -> ImportError
        else:
            sys.modules["gymnasium"] = gym_module
        root = ModuleType("tasks")
        root.__path__ = [str(TASKS)]
        sys.modules["tasks"] = root
        spec = importlib.util.spec_from_file_location("tasks.journal", JOURNAL_INIT,
                                                      submodule_search_locations=[str(JOURNAL_INIT.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["tasks.journal"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class Recorder(ModuleType):
    def __init__(self):
        super().__init__("gymnasium")
        self.calls = []

    def register(self, **kwargs):
        self.calls.append(kwargs)


def class_fields(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = {s.target.id: ast.literal_eval(s.value) for s in node.body
                              if isinstance(s, ast.AnnAssign) and s.value is not None
                              and isinstance(s.target, ast.Name)
                              and isinstance(s.value, (ast.Constant, ast.UnaryOp))}
    return out


class Registration(unittest.TestCase):
    def test_literal_id_set_equals_journal_env_ids(self):
        text = JOURNAL_INIT.read_text(encoding="ascii")
        expected = set(G.plans.JOURNAL_ENV_IDS.values())
        run_cell_ids = [m.group(2) for m in RUN_CELL_ID.finditer(text)]
        drift_ids = DRIFT_ID.findall(text)
        self.assertEqual(len(run_cell_ids), 24)
        self.assertEqual(set(run_cell_ids), expected)
        self.assertEqual(sorted(run_cell_ids), sorted(drift_ids))
        tree = ast.parse(text)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "gym.register"]
        self.assertEqual(len(calls), 24)
        literal = []
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            self.assertIsInstance(kw["id"], ast.Constant)                 # literal, never an f-string
            literal.append(kw["id"].value)
        self.assertEqual(set(literal), expected)
        self.assertNotIn("for ", "".join(ast.unparse(n) for n in tree.body if isinstance(n, (ast.For, ast.While))))

    def test_disjoint_from_every_legacy_id(self):
        legacy = set()
        for init in TASKS.glob("*/__init__.py"):
            if init.parent.name == "journal":
                continue
            legacy.update(m.group(2) for m in RUN_CELL_ID.finditer(init.read_text(encoding="utf-8-sig")))
        self.assertGreater(len(legacy), 60)
        self.assertFalse(legacy & set(G.plans.JOURNAL_ENV_IDS.values()))

    def test_run_cell_maps_the_ids_to_the_journal_family(self):
        sys.path.insert(0, str(G.REPO / "scripts"))
        spec = importlib.util.spec_from_file_location("_f0_run_cell", G.REPO / "scripts" / "run_cell.py")
        run_cell = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_cell)
        found = run_cell.registered_tasks(TASKS)
        for gym_id in G.plans.JOURNAL_ENV_IDS.values():
            self.assertEqual(found[gym_id], "journal")
        self.assertEqual(sum(1 for v in found.values() if v == "journal"), 24)

    def test_entry_points_cfg_classes_and_agents_exist(self):
        module = execute_registration(Recorder())
        calls = module.RECORDED_REGISTRATIONS
        self.assertEqual(calls, ())                                       # real (recording) gym was used
        recorder = Recorder()
        module = execute_registration(recorder)
        self.assertEqual(len(recorder.calls), 24)
        templates = class_fields(G.SHARED / "journal_runtime_cfg_v2.py")
        by_id = {v: k for k, v in G.plans.JOURNAL_ENV_IDS.items()}
        for call in recorder.calls:
            task, vessel = by_id[call["id"]]
            carrier = G.plans.TASK_TABLE[task]["carrier"]
            self.assertTrue(call["disable_env_checker"])
            self.assertEqual(set(call["kwargs"]), {"env_cfg_entry_point", "skrl_cfg_entry_point"})
            module_path, _, cls = call["entry_point"].partition(":")
            rel, env_class = CARRIER_FILES[carrier]
            self.assertEqual(module_path, "tasks." + rel[:-3].replace("/", "."))
            self.assertEqual(cls, env_class)
            env_tree = ast.parse((TASKS / rel).read_text(encoding="utf-8"))
            self.assertIn(env_class, {n.name for n in env_tree.body if isinstance(n, ast.ClassDef)})
            cfg_module, _, cfg_class = call["kwargs"]["env_cfg_entry_point"].partition(":")
            self.assertEqual(cfg_module, "tasks._shared.journal_runtime_cfg_v2")
            self.assertEqual(templates[cfg_class]["journal_task"], task)
            self.assertEqual(templates[cfg_class]["vehicle"], vessel)
            self.assertNotIn(cfg_class, D1_CALM_CLASSES)
            agents, _, yaml = call["kwargs"]["skrl_cfg_entry_point"].partition(":")
            self.assertEqual(agents, "tasks.%s.agents" % carrier)
            self.assertTrue((TASKS / carrier / "agents" / yaml).is_file())
            # G19b P9 (TRAINING_DESIGN section 5 item 1): BlueBoat docking trains with its log_std -2.3 yaml
            expected = ("skrl_ppo_blueboat_cfg.yaml" if (task, vessel) == ("docking", "blueboat_v2")
                        else "skrl_ppo_cfg.yaml")
            self.assertEqual(yaml, expected, call["id"])

    def test_importable_without_gymnasium_and_without_isaac(self):
        before = set(sys.modules)
        module = execute_registration(None)
        self.assertEqual(len(module.RECORDED_REGISTRATIONS), 24)
        added = set(sys.modules) - before
        self.assertFalse({n for n in added if n.split(".")[0] in ("isaaclab", "isaacsim", "omni", "pxr")})


if __name__ == "__main__":
    unittest.main(verbosity=2)
