"""G6: journal cfgs under the REAL Isaac Lab 2.3.2 cfg machinery (P5 failures 1 and 2).

Failure 1 (hazard_nav, 14 ids): configclass's post-init re-assigns every non-callable
dir(obj) member and recognises a property only in the instance's OWN class dict
(isaaclab/utils/configclass.py:392-402), so HazardNavEnvCfg.obstacle_slot_count (a
read-only property of the parent) raised "has no setter" on every subclass. Cure: a
setter that accepts exactly the re-assignment of the derived value
(tasks/hazard_nav/hazard_nav_env_cfg.py).
Failure 2 (every id through hydra): cfg.journal_plan (a frozen JournalTaskPlan) reached
update_class_from_dict, which assigns every leaf (isaaclab/utils/dict.py:102, :163):
FrozenInstanceError. Cure: JournalPlanSlot keeps the plan under a dunder key that the dict
utilities skip; cert_v3.json_value renders it unchanged (resolved_config byte-identical).

* With an Isaac Lab 2.3.2 checkout (isaaclab_reference.find_reference; skipped otherwise):
  the real code reproduces both failure mechanisms on minimal classes and shows the cures;
  all 24 journal templates build, survive hydra.py's to_dict -> OmegaConf -> from_dict
  sequence unchanged in every field, validate(), deepcopy, and keep obstacle_slot_count live.
* Always: a configclass-safety lint over the 24 templates built by the existing CPU stubs
  (scripts/tests_journal_glue/glue_fixtures.build_templates) and the hidden-member map.
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g6/test_g6_runtime_cfg.py -q
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import importlib
import json
import math
from pathlib import Path
import re
import sys
import types

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
for entry in (str(HERE), str(SCRIPTS), str(REPO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
import isaaclab_reference as R  # noqa: E402

REF, WHY = R.find_reference()
needs_ref = pytest.mark.skipif(REF is None, reason="Isaac Lab 2.3.2 reference not found: " + str(WHY))
TEMPLATE_NAME = re.compile(r"^\w+JournalEnvCfg$")
# Isaac Lab's own behaviour-neutral normalization (dict.py:106-114 keeps only the OUTER
# tuple of a nested sequence): render-only colours, read as Gf.Vec3f(*color).
KNOWN_NORMALIZED = re.compile(r"^cfg\.visual\.waypoint_ordered_colors\[\d+\]$")


# ------------------------------------------------------------------ snapshots
def snap(value, seen=()):
    if value is None or isinstance(value, (bool, int, str)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ["float", "nan" if math.isnan(value) else value]
    if isinstance(value, (types.FunctionType, type)):
        return ["callable", value.__module__, value.__qualname__]
    if id(value) in seen:
        return ["cycle"]
    seen = seen + (id(value),)
    if type(value).__name__ == "Box":
        return ["Box", str(value.dtype), list(value.shape), value.low.tolist(), value.high.tolist()]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [snap(v, seen) for v in value]]
    if isinstance(value, (dict, types.MappingProxyType)):
        return [type(value).__name__, {repr(k): snap(v, seen) for k, v in value.items()}]
    if hasattr(value, "__dict__"):
        return ["obj", type(value).__qualname__, {k: snap(v, seen) for k, v in vars(value).items()}]
    return ["repr", repr(value)]


def diff(a, b, path="cfg", out=None):
    out = [] if out is None else out
    if a == b:
        return out
    if a[0] == b[0] == "obj" and a[1] == b[1]:
        for key in sorted(set(a[2]) | set(b[2])):
            if key not in a[2] or key not in b[2]:
                out.append(f"{path}.{key}: added/removed")
            else:
                diff(a[2][key], b[2][key], f"{path}.{key}", out)
        return out
    if a[0] == b[0] in ("dict", "mappingproxy"):
        for key in sorted(set(a[1]) | set(b[1])):
            if key not in a[1] or key not in b[1]:
                out.append(f"{path}[{key}]: added/removed")
            else:
                diff(a[1][key], b[1][key], f"{path}[{key}]", out)
        return out
    if a[0] == b[0] in ("list", "tuple") and len(a[1]) == len(b[1]):
        for index, (x, y) in enumerate(zip(a[1], b[1])):
            diff(x, y, f"{path}[{index}]", out)
        return out
    out.append(f"{path}: {json.dumps(a)[:160]} -> {json.dumps(b)[:160]}")
    return out


# ------------------------------------------------------------------ real Isaac Lab
@pytest.fixture(scope="module")
def real():
    restore, modules = R.install(REF)
    try:
        templates = importlib.import_module("tasks._shared.journal_runtime_cfg_v2")
        hazard = importlib.import_module("tasks.hazard_nav.hazard_nav_env_cfg")
        journal_config = importlib.import_module("tasks._shared.conditions.journal_config")
        yield types.SimpleNamespace(modules=modules, configclass=modules["configclass"].configclass,
                                    templates=templates, hazard=hazard, journal_config=journal_config)
    finally:
        restore()


def template_classes(templates):
    return [getattr(templates, n) for n in sorted(vars(templates)) if TEMPLATE_NAME.match(n)]


@needs_ref
def test_real_post_init_rejects_parent_read_only_property_and_the_setter_cures_it(real):
    cc = real.configclass

    @cc
    class Parent:
        x: int = 1

        @property
        def derived(self):
            return self.x + 1

    @cc
    class Child(Parent):
        y: int = 2

    Parent()                                           # own-class property: skipped (configclass.py:399-401)
    with pytest.raises(AttributeError, match="has no setter"):
        Child()                                        # P5 failure 1 mechanism

    @cc
    class Parent2:
        x: int = 1

        @property
        def derived(self):
            return self.x + 1

        @derived.setter
        def derived(self, value):
            if value != self.derived:
                raise AttributeError("derived is read-only")

    @cc
    class Child2(Parent2):
        y: int = 2

    child = Child2()
    assert child.derived == 2 and "derived" not in vars(child)
    with pytest.raises(AttributeError):
        child.derived = 99


@needs_ref
def test_real_from_dict_rejects_a_frozen_member_and_the_plan_slot_cures_it(real):
    cc = real.configclass

    @dataclasses.dataclass(frozen=True)
    class Plan:
        geometry_tier: int = 2

    @cc
    class Cfg:
        a: int = 1

    cfg = Cfg()
    cfg.journal_plan = Plan()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.from_dict(cfg.to_dict())                   # P5 failure 2 mechanism (dict.py:102 -> :163)

    @cc
    class Slotted(Cfg, real.templates.JournalPlanSlot):
        pass

    slotted = Slotted()
    plan = Plan()
    slotted.journal_plan = plan
    data = slotted.to_dict()
    assert "journal_plan" not in data and real.templates.PLAN_KEY not in data
    slotted.from_dict(data)
    assert slotted.journal_plan is plan
    assert copy.deepcopy(slotted).journal_plan == plan and copy.copy(slotted).journal_plan is plan
    assert Slotted().journal_plan is None


def hydra_round_trip(real, cfg):
    """hydra.py:47-93 without the ConfigStore / hydra.main plumbing."""
    spaces = real.modules["spaces"]
    dict_mod = real.modules["dict"]
    from omegaconf import OmegaConf
    spaces.replace_env_cfg_spaces_with_strings(cfg)
    data = dict_mod.replace_slices_with_strings({"env": cfg.to_dict()})
    data = OmegaConf.to_container(OmegaConf.create(data), resolve=True)
    data = dict_mod.replace_strings_with_slices(data)
    cfg.from_dict(data["env"])
    spaces.replace_strings_with_env_cfg_spaces(cfg)
    return cfg


@needs_ref
def test_all_24_templates_build_round_trip_and_validate(real):
    pytest.importorskip("omegaconf")
    if real.modules["spaces"] is None:
        pytest.skip("reference checkout has no envs/utils/spaces.py")
    import cert_v3
    classes = template_classes(real.templates)
    assert len(classes) == 24
    for cls in classes:
        cfg = cls()                                    # parse_cfg.py:114 does cfg_cls()
        cfg.sim.device, cfg.scene.num_envs = "cuda:0", 2   # parse_cfg.py:149-155
        plan = cfg.journal_plan
        assert type(plan).__name__.endswith("JournalPlan") and plan.geometry_tier == cfg.journal_geometry_tier
        before = snap(cfg)
        hydra_round_trip(real, cfg)
        changes = [c for c in diff(before, snap(cfg)) if not KNOWN_NORMALIZED.match(c.split(":")[0])]
        assert changes == [], (cls.__name__, changes[:5])
        assert cfg.journal_plan is plan
        cfg.validate()                                 # DirectRLEnv.__init__
        cfg.scene.validate()                           # InteractiveScene.__init__
        copy.deepcopy(cfg)
        real.templates.journal_validate(cfg)
        # resolved_config (cert_v3.json_value -> cfg.to_dict()): journal_plan rendered at the
        # position of the plan in __dict__, exactly as the former plain attribute
        config = cert_v3.json_value(cfg)
        expected = [cert_v3.HIDDEN_CFG_MEMBERS.get(k, k) for k in vars(cfg)
                    if not k.startswith("__") or k in cert_v3.HIDDEN_CFG_MEMBERS]
        assert list(config) == expected, cls.__name__
        assert config["journal_plan"] == {"geometry_tier": plan.geometry_tier}


@needs_ref
def test_obstacle_slot_count_stays_live_and_read_only(real):
    jc = real.journal_config
    for name in ("ScatterBlueBoatJournalEnvCfg", "ReachWAMVJournalEnvCfg"):
        cfg = getattr(real.templates, name)()
        width = cfg.obstacle_slot_count
        assert width == cfg.max_obstacles and "obstacle_slot_count" not in vars(cfg)
        cfg.obstacle_slot_count = width                # the post-init re-assignment: accepted
        with pytest.raises(AttributeError, match="cannot be set"):
            cfg.obstacle_slot_count = width + 1
        if cfg.journal_task == "scatter":
            cfg.seed = 42
            jc.apply_journal_conditions(cfg, jc.resolve_journal_conditions(presets=["appearing=D1"], seed=42))
            assert cfg.obstacle_slot_count == width + 1    # G3 appearing slot, computed live as before


# ------------------------------------------------------------------ always (stub cfgs)
def _stub_templates():
    sys.path.insert(0, str(SCRIPTS / "tests_journal_glue"))
    import glue_fixtures
    return glue_fixtures.build_templates(), glue_fixtures


def _unsafe(value, path, out, seen=()):
    """Members Isaac Lab's dict utilities cannot round-trip (see the module docstring)."""
    if value is None or isinstance(value, (bool, int, float, str)) or id(value) in seen:
        return
    seen = seen + (id(value),)
    if isinstance(value, types.MappingProxyType):
        out.append(path + ": MappingProxyType")
    elif isinstance(value, (types.MethodType, functools_partial())) or (
            isinstance(value, types.FunctionType) and ("<locals>" in value.__qualname__
                                                       or value.__name__ == "<lambda>")):
        out.append(path + ": unresolvable callable")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type) \
            and type(value).__dataclass_params__.frozen:
        out.append(path + ": frozen dataclass " + type(value).__qualname__)
    elif isinstance(value, (list, tuple)):
        if any(hasattr(v, "__dict__") and not isinstance(v, (type, types.FunctionType)) for v in value):
            out.append(path + ": sequence of objects")
        for index, item in enumerate(value):
            _unsafe(item, f"{path}[{index}]", out, seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            _unsafe(item, f"{path}[{key!r}]", out, seen)
    elif hasattr(value, "__dict__") and not isinstance(value, (type, types.FunctionType)):
        for key, item in vars(value).items():
            if not key.startswith("__"):
                _unsafe(item, f"{path}.{key}", out, seen)


def functools_partial():
    import functools
    return functools.partial


def test_templates_have_no_parent_property_without_setter():
    templates, _ = _stub_templates()
    assert len(templates) == 24
    for name, cfg in templates.items():
        cls = type(cfg)
        for attr in dir(cls):
            owner = next((k for k in cls.__mro__ if attr in vars(k)), None)
            descriptor = vars(owner).get(attr) if owner is not None else None
            if isinstance(descriptor, property) and owner is not cls:
                assert descriptor.fset is not None, f"{name}.{attr} (property of {owner.__name__}) has no setter"


def test_templates_hold_only_round_trippable_members_before_the_suites():
    templates, _ = _stub_templates()
    for name, cfg in templates.items():
        found = []
        for key, value in vars(cfg).items():
            if key.startswith("__"):
                continue
            if key in ("conditions", "journal_conditions"):
                continue                               # installed by build_templates' resolve; see next test
            _unsafe(value, "cfg." + key, found)
        assert found == [], (name, found)


def test_only_the_condition_stack_is_frozen_after_the_suites():
    """resolve_protocol installs cfg.conditions / cfg.journal_conditions (frozen) AFTER the
    hydra round trip in every evaluator; they are walked by validate() and to_dict() but never
    by from_dict. Any OTHER frozen member would be a new P5-class failure."""
    templates, glue = _stub_templates()
    jc = glue.journal_config
    resolved = jc.resolve_journal_conditions(presets=["waves=D1", "wind=D1", "current=D1", "actuator=D1",
                                                      "perception=D1", "dynamic=D1"], seed=42)
    for name, cfg in templates.items():
        cfg = copy.copy(cfg)
        cfg.seed = 42
        jc.apply_journal_conditions(cfg, resolved)
        found = []
        for key, value in vars(cfg).items():
            if not key.startswith("__"):
                _unsafe(value, "cfg." + key, found)
        assert {f.split(":")[0].split(".")[1].split("[")[0] for f in found} <= {"conditions", "journal_conditions"}, \
            (name, found)


def test_hidden_member_map_matches_the_plan_slot():
    tree = ast.parse((REPO / "tasks" / "_shared" / "journal_plan_slot.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = node.value
    plan_key = ast.literal_eval(values["PLAN_KEY"])
    assert plan_key.startswith("__") and plan_key.endswith("__")    # skipped by dict.py:59, configclass.py:288/394
    hidden = values["HIDDEN_CFG_MEMBERS"]
    assert isinstance(hidden, ast.Dict) and len(hidden.keys) == 1
    assert ast.literal_eval(hidden.values[0]) == "journal_plan" and hidden.keys[0].id == "PLAN_KEY"
    import cert_v3
    assert cert_v3.HIDDEN_CFG_MEMBERS == {plan_key: "journal_plan"}
    runtime = ast.parse((REPO / "tasks" / "_shared" / "journal_runtime_cfg_v2.py").read_text(encoding="utf-8"))
    classes = [n for n in runtime.body if isinstance(n, ast.ClassDef)]
    # the slot is the LAST base: bases[0] stays the family parent (tests_contracts_v2/test_cfg_v2.py)
    assert len(classes) == 24 and all(c.bases[-1].id == "JournalPlanSlot" and len(c.bases) == 2 for c in classes)


def test_json_value_cpu_path_renders_the_hidden_plan_like_the_plain_attribute():
    import cert_v3

    @dataclasses.dataclass(frozen=True)
    class Plan:
        geometry_tier: int = 2

    class Plain:
        pass

    class Slotted:
        @property
        def journal_plan(self):
            return self.__dict__.get("__journal_plan__")

    plain, slotted = Plain(), Slotted()
    for obj, key in ((plain, "journal_plan"), (slotted, "__journal_plan__")):
        obj.a = 1
        obj.__dict__[key] = Plan()
        obj.b = (1.0, 2.0)
    assert cert_v3.json_value(slotted) == cert_v3.json_value(plain)
    assert list(cert_v3.json_value(slotted)) == ["a", "journal_plan", "b"]
