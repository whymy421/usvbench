"""CPU-pure tests for HazardNavEnvCfg post-override validation.

Isaac Lab is not available in the CPU test environment, so this lifts the
real ``validate_after_overrides`` method from the shipped config source and
executes it on a complete fabricated config object.

Run: python tasks/hazard_nav/test_cfg_validation.py
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace


CFG_PATH = Path(__file__).with_name("hazard_nav_env_cfg.py")


def _lift_validator():
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "HazardNavEnvCfg":
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "validate_after_overrides"
                ):
                    module = ast.Module(body=[item], type_ignores=[])
                    ast.fix_missing_locations(module)
                    namespace = {
                        "_math": math,
                        "SUPERSET_DIM": 51,
                        "SUPERSET_DIM_V2": 64,
                    }
                    exec(compile(module, str(CFG_PATH), "exec"), namespace)  # noqa: S102
                    return namespace["validate_after_overrides"]
    raise AssertionError("HazardNavEnvCfg.validate_after_overrides not found")


VALIDATE = _lift_validator()


def _cfg(**changes):
    values = {
        "obs_v2": False,
        "obs_v3": False,
        "superset_version": 1,
        "obs_feasibility": False,
        "feasibility_sectors": 9,
        "ray_count": 36,
        "observation_space": 39,
        "emit_superset_obs": False,
        "yaw_rate_obs_scale_rad_s": 1.0,
        "layout_mode": "scatter",
        "reward_progress_geodesic": False,
        "nav_targets_waypoint": False,
        "waypoint_lookahead_m": 3.0,
        "max_obstacles": 14,
        "min_goal_distance_m": 20.0,
        "max_goal_distance_m": 40.0,
        "reward_goal_entry_bonus": 0.0,
        "reward_goal_time_bonus": 0.0,
        "reward_reverse_action_scale": 0.0,
        "contact_dwell_tau_s": 0.15,
        "appear_count": 0,
        "appear_time_min_s": 5.0,
        "appear_time_max_s": 30.0,
        "appear_distance_m": 20.0,
        "appear_radius_m": 0.8,
        "appear_bfs_cell_m": 0.25,
        "appear_max_attempts": 36,
        "appear_stop_distance_m": 0.868,
        "mass_scale": 1.0,
        "mass_scale_choices": (),
        "drag_scale": 1.0,
        "drag_scale_choices": (),
        "thrust_cap_scale": 1.0,
        "thrust_cap_scale_choices": (),
        "motor_tau_s": 0.0,
        "motor_tau_s_choices": (),
        "payload_mass_kg": 0.0,
        "payload_mass_kg_choices": (),
        "thrust_imbalance": 0.0,
        "thrust_imbalance_choices": (),
    }
    values.update(changes)
    cfg = SimpleNamespace(**values)
    cfg.validate_after_overrides = lambda: VALIDATE(cfg)
    return cfg


def _expect_refusal(field, value, **changes):
    cfg = _cfg(**changes)
    try:
        cfg.validate_after_overrides()
    except (TypeError, ValueError) as exc:
        message = str(exc)
        assert field in message and repr(value) in message, message
        return
    raise AssertionError(f"expected {field}={value!r} to be rejected")


def test_validator_is_pure_for_a_valid_config():
    cfg = _cfg(thrust_imbalance=-0.5)
    before = dict(vars(cfg))
    cfg.validate_after_overrides()
    assert vars(cfg) == before


def test_appearance_count_and_active_window_domains():
    _expect_refusal("appear_count", -1, appear_count=-1)
    _expect_refusal(
        "appear_time_min_s", -30.0,
        appear_count=1, appear_time_min_s=-30.0, appear_time_max_s=-5.0,
    )
    _expect_refusal(
        "appear_time_min_s", 30.0,
        appear_count=1, appear_time_min_s=30.0, appear_time_max_s=5.0,
    )


def test_inactive_appearance_ignores_window_values():
    for minimum, maximum in ((-30.0, -5.0), (math.nan, math.inf)):
        cfg = _cfg(
            appear_count=0,
            appear_time_min_s=minimum,
            appear_time_max_s=maximum,
        )
        cfg.validate_after_overrides()


def test_active_appearance_auxiliary_domains_and_finiteness():
    for field, value in (
        ("appear_radius_m", 0.79),
        ("appear_bfs_cell_m", 0.0),
        ("appear_max_attempts", 0),
        ("appear_stop_distance_m", -0.1),
        ("appear_distance_m", math.inf),
        ("appear_time_max_s", math.nan),
    ):
        _expect_refusal(field, value, appear_count=1, **{field: value})


def test_actuator_scalar_boundaries():
    valid = _cfg(
        mass_scale=1.0e-12,
        drag_scale=0.0,
        thrust_cap_scale=0.0,
        motor_tau_s=0.0,
        payload_mass_kg=0.0,
        thrust_imbalance=-0.5,
    )
    valid.validate_after_overrides()
    for field, value in (
        ("mass_scale", 0.0),
        ("drag_scale", -0.1),
        ("thrust_cap_scale", -0.1),
        ("motor_tau_s", -0.1),
        ("payload_mass_kg", -0.1),
        ("thrust_imbalance", 1.5),
    ):
        _expect_refusal(field, value, **{field: value})
    _cfg(thrust_imbalance=-1.0).validate_after_overrides()
    _cfg(thrust_imbalance=1.0).validate_after_overrides()


def test_each_actuator_choices_tuple_validates_every_element():
    for field, bad in (
        ("mass_scale_choices", -0.1),
        ("drag_scale_choices", -0.1),
        ("thrust_cap_scale_choices", -0.1),
        ("motor_tau_s_choices", -0.1),
        ("payload_mass_kg_choices", -0.1),
        ("thrust_imbalance_choices", 1.5),
    ):
        _expect_refusal(field, bad, **{field: (1.0, bad)})


def test_actuator_scalars_and_choices_reject_non_finite_values():
    scalar_fields = (
        "mass_scale", "drag_scale", "thrust_cap_scale", "motor_tau_s",
        "payload_mass_kg", "thrust_imbalance",
    )
    for field in scalar_fields:
        _expect_refusal(field, math.nan, **{field: math.nan})
    _expect_refusal("mass_scale", True, mass_scale=True)
    choice_fields = tuple(f"{field}_choices" for field in scalar_fields)
    for field in choice_fields:
        _expect_refusal(field, math.inf, **{field: (1.0, math.inf)})


if __name__ == "__main__":
    failures = 0
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)
