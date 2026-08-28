"""Wave background variants: machine-check of the append-only claim.

Owner's order (2026-08-26): every task family gets a Wave variant whose sea is
a BACKGROUND FORCE ONLY -- unlike the station-keeping Wave id, no observation
channels are added, so every certified checkpoint loads zero-shot and the id
is a pure environmental stress axis.

What is machine-checked here, per variant:

* the Wave cfg subclass adds NOTHING beyond the sea-state field: its class
  body is exactly a docstring, one ``sea_state: SeaStateCfg = SeaStateCfg()``
  declaration, and a ``__post_init__`` whose every statement (after the
  mandatory ``super().__post_init__()``) assigns to ``self.sea_state.*``;
* those assignments land on REAL dataclass fields of ``SeaStateCfg``
  (reflection over ``dataclasses.fields``), pin the frozen evaluation box
  (H_s 0.30-0.60 m, T_p 2.0-2.5 s, frozen 2026-08-26; the certified
  station-keeping id's 1.5 s floor stays excluded as grandfathered history),
  pass the DNV-RP-C205 steepness admission and capture >= 95% of the
  analytic JONSWAP variance at every corner, and leave every other
  sea-state field at its default;
* the gym id is registered against that cfg class;
* ``NATIVE_LAYOUTS`` maps the Wave id to its parent id's layout verbatim
  (same observation contract, so the bridge treats both as the same policy
  input);
* the family env carries the shared wiring: the guarded ``SeaState``
  construction, the world-frame ``forces`` call, and the per-episode
  ``resample`` off a scenario stream.

The env cfg modules import ``isaaclab`` at module scope and cannot be imported
on a machine without Isaac Sim, so the cfg-side checks AST-lift the class
definitions out of the real source files -- the technique
``tasks/_shared/test_scenario_env_wiring.py`` uses -- while the band check
executes the lifted assignments against a real ``SeaStateCfg`` instance.

Run: python tasks/_shared/test_wave_variants.py
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import warnings
from pathlib import Path
import sys

SHARED = Path(__file__).resolve().parent
TASKS = SHARED.parent


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sea_state = _load_by_path("wave_variants_sea_state", SHARED / "sea_state.py")
obs = _load_by_path("wave_variants_obs_superset", SHARED / "obs_superset.py")


# The frozen evaluation box (2026-08-26). H_s is the station-keeping wave
# band (tasks/station_keeping/station_keeping_env_cfg.py:299); T_p is pinned
# to 2.0-2.5 s. Every corner passes the DNV-RP-C205 steepness admission
# (sea_state.validate_steepness_admission, worst corner Sp ~ 1/10.4 < 1/7)
# and captures >= 95% of the analytic JONSWAP variance on the default
# 104-component 0.04-1.60 Hz band -- both machine-checked below. The
# certified station-keeping id's 1.5 s floor stays excluded (grandfathered
# history, not a template); the rung ladder pins Hs per-rung later via --set.
EXPECTED_SEA_FIELDS = {
    "enable": True,
    "hs_range": (0.30, 0.60),
    "tp_range": (2.0, 2.5),
}

VARIANTS = [
    {
        "family": "hazard_nav",
        "cfg_file": "hazard_nav_env_cfg.py",
        "env_file": "hazard_nav_env.py",
        "wave_class": "HazardForcedCrossingWaveEnvCfg",
        "parent_class": "HazardForcedCrossingEnvCfg",
        "wave_id": "Isaac-USV-HazardCross-Wave-Direct-v1",
        "parent_id": "Isaac-USV-HazardCross-Direct-v1",
    },
    {
        "family": "hazard_nav",
        "cfg_file": "hazard_nav_env_cfg.py",
        "env_file": "hazard_nav_env.py",
        "wave_class": "HazardIcebergWaveEnvCfg",
        "parent_class": "HazardIcebergEnvCfg",
        "wave_id": "Isaac-USV-Iceberg-Wave-Direct-v1",
        "parent_id": "Isaac-USV-Iceberg-Direct-v1",
    },
    {
        "family": "path_following",
        "cfg_file": "path_following_env_cfg.py",
        "env_file": "path_following_env.py",
        "wave_class": "PathFollowingBlueBoatWaveEnvCfg",
        "parent_class": "PathFollowingBlueBoatEnvCfg",
        "wave_id": "Isaac-USV-PathFollow-BlueBoat-Wave-Direct-v1",
        "parent_id": "Isaac-USV-PathFollow-BlueBoat-Direct-v1",
    },
    {
        "family": "path_hazard",
        "cfg_file": "path_hazard_env_cfg.py",
        "env_file": "path_hazard_env.py",
        "wave_class": "PathHazardWaveEnvCfg",
        "parent_class": "PathHazardEnvCfg",
        "wave_id": "Isaac-USV-PathHazard-Wave-Direct-v1",
        "parent_id": "Isaac-USV-PathHazard-Direct-v1",
    },
    {
        "family": "docking",
        "cfg_file": "docking_env_cfg.py",
        "env_file": "docking_env.py",
        "wave_class": "DockingBlueBoatWaveEnvCfg",
        "parent_class": "DockingBlueBoatEnvCfg",
        "wave_id": "Isaac-USV-Dock-BlueBoat-Wave-Direct-v1",
        "parent_id": "Isaac-USV-Dock-BlueBoat-Direct-v1",
    },
    {
        "family": "harbor_mission",
        "cfg_file": "harbor_mission_env_cfg.py",
        "env_file": "harbor_mission_env.py",
        "wave_class": "HarborMissionWaveEnvCfg",
        "parent_class": "HarborMissionEnvCfg",
        "wave_id": "Isaac-USV-HarborMission-Wave-Direct-v1",
        "parent_id": "Isaac-USV-HarborMission-Direct-v1",
    },
]


def _class_def(variant) -> ast.ClassDef:
    path = TASKS / variant["family"] / variant["cfg_file"]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == variant["wave_class"]:
            return node
    raise AssertionError(f"{path}: class {variant['wave_class']} not found")


def _sea_state_assignments(variant) -> dict:
    """The ``self.sea_state.X = literal`` map; proves the body adds nothing else."""
    cls = _class_def(variant)
    label = "{}/{}".format(variant["family"], variant["wave_class"])

    bases = [b.id for b in cls.bases if isinstance(b, ast.Name)]
    assert bases == [variant["parent_class"]], (
        f"{label}: bases {bases}, expected [{variant['parent_class']}]"
    )

    body = list(cls.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # docstring

    assert len(body) == 2, (
        f"{label}: expected exactly [sea_state field, __post_init__], "
        f"got {len(body)} statements"
    )

    field = body[0]
    assert isinstance(field, ast.AnnAssign), f"{label}: first stmt not a field"
    assert isinstance(field.target, ast.Name) and field.target.id == "sea_state", (
        f"{label}: the only new field must be sea_state"
    )
    assert (
        isinstance(field.value, ast.Call)
        and isinstance(field.value.func, ast.Name)
        and field.value.func.id == "SeaStateCfg"
        and not field.value.args
        and not field.value.keywords
    ), f"{label}: sea_state default must be a bare SeaStateCfg()"

    post = body[1]
    assert isinstance(post, ast.FunctionDef) and post.name == "__post_init__", (
        f"{label}: second stmt must be __post_init__"
    )

    first = post.body[0]
    assert (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Call)
        and isinstance(first.value.func, ast.Attribute)
        and first.value.func.attr == "__post_init__"
        and isinstance(first.value.func.value, ast.Call)
        and isinstance(first.value.func.value.func, ast.Name)
        and first.value.func.value.func.id == "super"
    ), f"{label}: __post_init__ must start with super().__post_init__()"

    assigned = {}
    for stmt in post.body[1:]:
        assert isinstance(stmt, ast.Assign) and len(stmt.targets) == 1, (
            f"{label}: __post_init__ may only assign, got {ast.dump(stmt)[:80]}"
        )
        target = stmt.targets[0]
        assert (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "sea_state"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "self"
        ), f"{label}: __post_init__ touches {ast.dump(target)[:80]}, not sea_state"
        assigned[target.attr] = ast.literal_eval(stmt.value)
    return assigned


def test_wave_cfg_differs_from_parent_only_in_sea_state():
    for variant in VARIANTS:
        _sea_state_assignments(variant)


def test_wave_band_lands_on_real_sea_state_fields_and_pins_the_band():
    field_defaults = {
        f.name: getattr(sea_state.SeaStateCfg(), f.name)
        for f in dataclasses.fields(sea_state.SeaStateCfg)
    }
    for variant in VARIANTS:
        label = "{}/{}".format(variant["family"], variant["wave_class"])
        assigned = _sea_state_assignments(variant)
        unknown = set(assigned) - set(field_defaults)
        assert not unknown, f"{label}: not SeaStateCfg dataclass fields: {unknown}"
        assert assigned == EXPECTED_SEA_FIELDS, (
            f"{label}: sea-state assignments {assigned}, "
            f"expected {EXPECTED_SEA_FIELDS}"
        )
        # Every field the variant does not pin keeps the shared default, so
        # the variant is band-only (coupling constants stay task-agnostic).
        cfg = sea_state.SeaStateCfg(**assigned)
        for name, default in field_defaults.items():
            if name not in assigned:
                assert getattr(cfg, name) == default, (label, name)
        # The pinned band must be resolvable by the default spectrum: the
        # JONSWAP peak f_p = 1/T_p has to stay inside [f_min, f_max].
        tp_min = cfg.tp_range[0]
        assert 1.0 / tp_min <= cfg.f_max + 1.0e-9, (
            f"{label}: T_p {tp_min} s puts f_p above f_max {cfg.f_max} Hz"
        )
        # F2: every corner of the box passes the steepness admission
        # OUTRIGHT -- no exception, and no grandfather warning either (only
        # the certified station-keeping box may warn).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sea_state.validate_steepness_admission(cfg)
        assert not caught, (
            f"{label}: steepness admission warned: {caught[0].message}"
        )
        # F1/F6: every (Hs, Tp, gamma) corner keeps >= 95% of the analytic
        # JONSWAP variance on the cfg's component band (capture is checked
        # against the ANALYTIC spectrum; the Hs renormalisation would hide
        # any truncation from an amplitude-based check).
        for hs_corner in cfg.hs_range:
            for tp_corner in cfg.tp_range:
                for gamma_corner in cfg.gamma_range:
                    captured = sea_state.analytic_spectrum_capture(
                        cfg.n_components, cfg.f_min, cfg.f_max,
                        hs_corner, tp_corner, gamma_corner,
                    )
                    assert captured >= 0.95, (
                        f"{label}: corner Hs={hs_corner} Tp={tp_corner} "
                        f"gamma={gamma_corner} captures only "
                        f"{captured * 100.0:.2f}% of the analytic variance"
                    )


def test_wave_id_is_registered_with_the_wave_cfg():
    for variant in VARIANTS:
        init = (TASKS / variant["family"] / "__init__.py").read_text(encoding="utf-8")
        assert 'id="{}"'.format(variant["wave_id"]) in init, variant["wave_id"]
        assert variant["wave_class"] in init, variant["wave_class"]


def test_wave_layout_is_the_parent_layout_verbatim():
    for variant in VARIANTS:
        wave = obs.NATIVE_LAYOUTS[variant["wave_id"]]
        parent = obs.NATIVE_LAYOUTS[variant["parent_id"]]
        assert not isinstance(wave, obs.UnsupportedNativeLayout), variant["wave_id"]
        assert wave == parent, (
            "{}: layout differs from its parent {}".format(
                variant["wave_id"], variant["parent_id"]
            )
        )


def test_env_carries_the_shared_wave_wiring():
    guard = (
        'if getattr(self.cfg, "sea_state", None) is not None '
        "and self.cfg.sea_state.enable:"
    )
    for variant in VARIANTS:
        env = (TASKS / variant["family"] / variant["env_file"]).read_text(
            encoding="utf-8"
        )
        label = "{}/{}".format(variant["family"], variant["env_file"])
        assert guard in env, f"{label}: guarded SeaState construction missing"
        assert "self._sea.forces(" in env, f"{label}: wave force call missing"
        assert "self._sea.resample(env_ids, scenario=" in env, (
            f"{label}: per-episode scenario resample missing"
        )


def test_sources_are_pure_ascii_without_control_characters():
    files = {SHARED / "test_wave_variants.py"}
    for variant in VARIANTS:
        files.add(TASKS / variant["family"] / variant["cfg_file"])
        files.add(TASKS / variant["family"] / variant["env_file"])
        files.add(TASKS / variant["family"] / "__init__.py")
    for path in sorted(files):
        text = path.read_text(encoding="utf-8")
        for i, ch in enumerate(text):
            code = ord(ch)
            assert code < 128 and (code >= 32 or ch in "\n\r"), (
                f"{path.name}: non-ASCII or control char {code!r} at offset {i}"
            )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
