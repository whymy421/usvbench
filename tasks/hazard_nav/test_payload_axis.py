# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Suite D deck-payload axis: identity, effect direction, distinguishability.

The payload is a POINT mass strapped on deck -- the single easiest real-boat
perturbation, and therefore the anchor axis for embodied validation. Model
documentation lives at the ``payload_*`` fields in ``hazard_nav_env_cfg.py``;
this file proves four things about the SHIPPED ``_apply_action``:

1. INERT BIT-IDENTITY. The whole method is AST-lifted from the source and
   compiled twice: verbatim, and with the payload block excised (which is the
   pre-payload method: the block is this axis's only addition). At zero
   payload -- scalar default AND drawn-zero choices envs -- the two produce
   bit-identical wrenches, so every certified id is untouched.
2. EFFECT DIRECTIONS. Centred mass slows every linear channel by exactly
   m/(m+mp) and leaves the yaw channel bit-identical; an off-centre mass
   produces the closed-form trim moment of the predicted sign and magnitude,
   through thrust AND through drag.
3. DISTINGUISHABILITY. No (mass_scale, thrust_imbalance) pair -- and no
   (mass_scale, thrust_imbalance, thrust_cap_scale, drag_scale) 4-tuple --
   reproduces the payload wrench across a fixed probe battery. The knob arms
   are evaluated through the SAME lifted shipped method (vectorised across
   envs via the per-env choices buffers), so the claim is about the code as
   it ships, not about a transcription. motor_tau_s is excluded by argument,
   not by search: it is a transient first-order filter on the COMMANDS whose
   steady state is the unfiltered command, so it cannot produce any
   steady-state effect at all, let alone touch the drag-borne moments the
   payload adds.
4. SOURCE PINS. The cfg fields exist with inert defaults, the reset path
   draws the choices tuple off the protocol's own group, and the group obeys
   the ``actuator_<param>`` naming contract.

Revert proof: set ``PAYLOAD_AXIS_ENV_SOURCE`` / ``PAYLOAD_AXIS_CFG_SOURCE``
to a pre-payload copy of the source files and this suite must fail (the
excision helper refuses a method without the block; the pins refuse a cfg
without the fields).

Run: python tasks/hazard_nav/test_payload_axis.py
"""

from __future__ import annotations

import ast
import copy
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from types import MethodType

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "_shared"))
import scenario_draws_hazard as _sdh  # noqa: E402

ENV_SOURCE = Path(
    os.environ.get("PAYLOAD_AXIS_ENV_SOURCE", _HERE / "hazard_nav_env.py")
)
CFG_SOURCE = Path(
    os.environ.get("PAYLOAD_AXIS_CFG_SOURCE", _HERE / "hazard_nav_env_cfg.py")
)

# Registry values, matching test_thrust_imbalance.py and the vehicle spec.
THRUST_FWD_N = 80.0
THRUST_REV_N = 48.0
YAW_TORQUE_NM = 23.0
HALF_BEAM_M = 0.45
PHYSICS_DT_S = 1.0 / 120.0
# USD-authored rigid body (assets/blueboat_physics.usd /World/BlueBoat).
HULL_MASS_KG = 17.26
HULL_IZZ_KGM2 = 2.376


# ------------------------------------------------------------- AST machinery --
def _method_def(path: Path, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HazardNavEnv":
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == method_name
                ):
                    return item
    raise AssertionError(f"{path}: no HazardNavEnv.{method_name}")


def _apply_action_def(path: Path) -> ast.FunctionDef:
    return _method_def(path, "_apply_action")


def _excise_payload_block(fn_def: ast.FunctionDef) -> ast.FunctionDef:
    """Delete the payload block -- i.e. reconstruct the pre-payload method."""
    fn_def = copy.deepcopy(fn_def)
    hits = [
        index
        for index, stmt in enumerate(fn_def.body)
        if isinstance(stmt, ast.If)
        and "payload_mass_kg" in ast.unparse(stmt.test)
    ]
    assert len(hits) == 1, (
        "payload block not found exactly once in the shipped _apply_action "
        f"(found {len(hits)}); the axis is missing or was restructured"
    )
    del fn_def.body[hits[0]]
    return fn_def


class _MathUtils:
    """Identity-quaternion-exact quat rotation, wxyz layout like isaaclab."""

    @staticmethod
    def quat_apply(quat, vec):
        w = quat[..., 0:1]
        xyz = quat[..., 1:4]
        t = 2.0 * torch.cross(xyz, vec, dim=-1)
        return vec + w * t + torch.cross(xyz, t, dim=-1)

    @staticmethod
    def quat_apply_inverse(quat, vec):
        w = quat[..., 0:1]
        xyz = -quat[..., 1:4]
        t = 2.0 * torch.cross(xyz, vec, dim=-1)
        return vec + w * t + torch.cross(xyz, t, dim=-1)


def _compile(fn_def: ast.FunctionDef, name: str):
    fn_def = copy.deepcopy(fn_def)
    fn_def.name = name
    module = ast.Module(body=[fn_def], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "np": np,
        "math": math,
        "math_utils": _MathUtils,
    }
    exec(compile(module, str(ENV_SOURCE), "exec"), namespace)  # noqa: S102
    return namespace[name]


SHIPPED = _compile(_apply_action_def(ENV_SOURCE), "shipped_apply_action")
EXCISED = _compile(
    _excise_payload_block(_apply_action_def(ENV_SOURCE)),
    "prepayload_apply_action",
)
# The two helpers _apply_action calls on self, lifted from the same source so
# the buoyancy wrench feeding the payload block is the shipped one.
BUOYANCY = _compile(
    _method_def(ENV_SOURCE, "_compute_buoyancy_forces"), "buoyancy"
)
ROOT_QUAT = _compile(_method_def(ENV_SOURCE, "_root_quat"), "root_quat")


# ------------------------------------------------------------ fabricated env --
class _Robot:
    def __init__(self, num_envs: int):
        self.num_bodies = 1
        quat = torch.zeros((num_envs, 4))
        quat[:, 0] = 1.0
        self.data = SimpleNamespace(
            root_pos_w=torch.zeros((num_envs, 3)),
            root_com_pos_w=torch.zeros((num_envs, 3)),
            root_com_vel_w=torch.zeros((num_envs, 6)),
            root_ang_vel_w=torch.zeros((num_envs, 3)),
            root_link_quat_w=quat,
        )
        self.applied = None

    def set_external_force_and_torque(self, forces, torques):
        self.applied = (forces.clone(), torques.clone())


def _cfg(**overrides) -> SimpleNamespace:
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=PHYSICS_DT_S),
        thrust_max_fwd=THRUST_FWD_N,
        thrust_max_rev=THRUST_REV_N,
        yaw_torque_max=YAW_TORQUE_NM,
        half_beam_m=HALF_BEAM_M,
        thrust_imbalance=0.0,
        thrust_imbalance_choices=(),
        mass_scale=1.0,
        mass_scale_choices=(),
        drag_scale=1.0,
        drag_scale_choices=(),
        thrust_cap_scale=1.0,
        thrust_cap_scale_choices=(),
        motor_tau_s=0.0,
        motor_tau_s_choices=(),
        payload_mass_kg=0.0,
        payload_mass_kg_choices=(),
        payload_offset_x_m=0.0,
        payload_offset_y_m=0.0,
        payload_ref_mass_kg=HULL_MASS_KG,
        payload_ref_izz_kgm2=HULL_IZZ_KGM2,
    )
    for key, value in overrides.items():
        assert hasattr(cfg, key), key
        setattr(cfg, key, value)
    return cfg


def _physics() -> SimpleNamespace:
    return SimpleNamespace(
        water_surface_z=0.0,
        rov_height=0.32,
        rov_volume=0.03,
        water_density=1000.0,
        gravity=9.81,
        buoyancy_center_offset=-0.05,
        surge_lin_damping=9.1,
        surge_quad_damping=5.9,
        sway_lin_damping=27.0,
        sway_quad_damping=18.0,
        heave_damping=5.0,
        yaw_lin_damping=4.0,
        yaw_quad_damping=6.0,
        rollpitch_rate_damping=1.0,
        restoring_stiffness_roll=10.0,
        restoring_stiffness_pitch=10.0,
    )


def _env(num_envs: int, cfg: SimpleNamespace) -> SimpleNamespace:
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        cfg=cfg,
        physics_cfg=_physics(),
        robot=_Robot(num_envs),
        actions=torch.zeros((num_envs, 2)),
        up_dir=torch.tensor([0.0, 0.0, 1.0]),
        _fwd_x=1.0,
        _fwd_y=0.0,
        _surge_axis_idx=0,
        _sway_axis_idx=1,
        _sea=None,
        _thrust_imbalance_per_env=torch.zeros(num_envs),
        _mass_scale_per_env=torch.ones(num_envs),
        _drag_scale_per_env=torch.ones(num_envs),
        _thrust_cap_scale_per_env=torch.ones(num_envs),
        _motor_tau_s_per_env=torch.zeros(num_envs),
        _payload_mass_kg_per_env=torch.zeros(num_envs),
        _applied_thrust_per_env=torch.zeros(num_envs),
        _applied_yaw_per_env=torch.zeros(num_envs),
    )
    env._compute_buoyancy_forces = MethodType(BUOYANCY, env)
    env._root_quat = MethodType(ROOT_QUAT, env)
    return env


def _run(fn, env, action, velocity_body, yaw_rate):
    """One control step of the lifted method; returns (forces, torques)."""
    env.actions = (
        torch.as_tensor(action, dtype=torch.float32)
        .expand(env.num_envs, 2)
        .clone()
    )
    vel = torch.zeros((env.num_envs, 6))
    vel[:, 0] = float(velocity_body[0])
    vel[:, 1] = float(velocity_body[1])
    vel[:, 5] = float(yaw_rate)
    env.robot.data.root_com_vel_w = vel
    env.robot.data.root_ang_vel_w = vel[:, 3:6].clone()
    fn(env)
    return env.robot.applied


# The fixed probe battery: (action, body velocity, yaw rate). Deterministic on
# purpose -- the distinguishability minima quoted below must be reproducible.
PROBES = (
    ((1.0, 0.0), (0.0, 0.0), 0.0),    # full thrust from rest
    ((0.0, 0.8), (0.0, 0.0), 0.0),    # yaw command from rest
    ((0.0, 0.0), (2.0, 0.0), 0.0),    # surge coast
    ((0.0, 0.0), (0.0, 1.0), 0.0),    # sway coast
    ((0.7, -0.3), (1.5, -0.5), 0.5),  # mixed manoeuvre
    ((1.0, 0.4), (2.5, 0.0), 0.0),    # thrust + turn at speed
    # Spin decay: the discriminating experiment against the c=d=k
    # impersonation. The payload leaves the yaw-damping moment alone (a
    # centred mass exactly, an offset mass up to the izz ratio), while any
    # drag_scale big enough to fake the slower linear channels must scale
    # this moment too -- so a fast spin exposes the impostor.
    ((0.0, 0.0), (0.0, 0.0), 2.0),    # pure spin decay
    ((0.0, 0.6), (1.0, 0.0), 1.0),    # yaw command against damping
)


def _planar(applied) -> np.ndarray:
    forces, torques = applied
    return np.stack(
        (
            forces[:, 0, 0].numpy(),
            forces[:, 0, 1].numpy(),
            torques[:, 0, 2].numpy(),
        ),
        axis=-1,
    )


def _probe_battery(fn, cfg, num_envs=1, setup=None) -> np.ndarray:
    """(probes, num_envs, 3) planar wrench of the lifted method."""
    rows = []
    for action, velocity, yaw_rate in PROBES:
        env = _env(num_envs, cfg)
        if setup is not None:
            setup(env)
        rows.append(_planar(_run(fn, env, action, velocity, yaw_rate)))
    return np.stack(rows, axis=0)


# -------------------------------------------------------------------- tests --
def test_inert_default_is_bit_identical():
    """Zero payload, scalar path: shipped == pre-payload, bit for bit."""
    rng = np.random.default_rng(20260826)
    for trial in range(64):
        action = rng.uniform(-1.0, 1.0, 2)
        velocity = rng.uniform(-3.0, 3.0, 2)
        yaw_rate = float(rng.uniform(-2.0, 2.0))
        shipped = _run(SHIPPED, _env(4, _cfg()), action, velocity, yaw_rate)
        excised = _run(EXCISED, _env(4, _cfg()), action, velocity, yaw_rate)
        assert torch.equal(shipped[0], excised[0]), (
            f"forces differ, trial {trial}"
        )
        assert torch.equal(shipped[1], excised[1]), (
            f"torques differ, trial {trial}"
        )
    # The other five knobs still ride through the payload-carrying method
    # unchanged at their non-default values.
    knob = _cfg(mass_scale=1.3, drag_scale=1.5, thrust_imbalance=0.1)
    for trial in range(16):
        action = rng.uniform(-1.0, 1.0, 2)
        velocity = rng.uniform(-3.0, 3.0, 2)
        yaw_rate = float(rng.uniform(-2.0, 2.0))
        shipped = _run(SHIPPED, _env(4, knob), action, velocity, yaw_rate)
        excised = _run(EXCISED, _env(4, knob), action, velocity, yaw_rate)
        assert torch.equal(shipped[0], excised[0]), trial
        assert torch.equal(shipped[1], excised[1]), trial
    print("   scalar inert path: 80 random steps bit-identical")


def test_drawn_zero_envs_are_bit_identical_in_choices_mode():
    """A (0.0, 3.0) pack: drawn-zero envs untouched, loaded envs not."""
    cfg = _cfg(payload_mass_kg_choices=(0.0, 3.0), payload_offset_y_m=0.4)
    drawn = torch.tensor([0.0, 3.0, 0.0, 3.0])

    action, velocity, yaw_rate = (0.9, 0.2), (1.0, 0.3), -0.4
    env = _env(4, cfg)
    env._payload_mass_kg_per_env = drawn.clone()
    shipped = _run(SHIPPED, env, action, velocity, yaw_rate)
    excised = _run(EXCISED, _env(4, _cfg()), action, velocity, yaw_rate)
    zero = drawn == 0.0
    assert torch.equal(shipped[0][zero], excised[0][zero]), "zero envs moved"
    assert torch.equal(shipped[1][zero], excised[1][zero]), "zero envs moved"
    loaded = ~zero
    assert not torch.equal(shipped[0][loaded], excised[0][loaded]), (
        "3 kg envs did not move: the choices path is dead"
    )
    print("   choices path: drawn-zero envs bit-identical, loaded envs move")


def test_centred_mass_slows_linear_channels_and_spares_yaw():
    """mp at the COM: linear forces * m/(m+mp) exactly, yaw bit-identical."""
    mp = 6.0
    cfg = _cfg(payload_mass_kg=mp)
    ratio = torch.tensor((HULL_MASS_KG + mp) / HULL_MASS_KG)
    for action, velocity, yaw_rate in PROBES:
        shipped = _run(SHIPPED, _env(1, cfg), action, velocity, yaw_rate)
        excised = _run(EXCISED, _env(1, _cfg()), action, velocity, yaw_rate)
        assert torch.equal(shipped[1], excised[1]), (
            "a centred point mass must leave every torque channel untouched "
            f"(reduced-mass term is zero at d=0): {action} {velocity}"
        )
        expected = excised[0] / ratio
        assert torch.allclose(shipped[0], expected, rtol=1e-6, atol=1e-6), (
            action,
            velocity,
        )
        # Direction: strictly smaller magnitude wherever the force is nonzero.
        nonzero = excised[0].abs() > 1e-9
        assert torch.all(
            shipped[0][nonzero].abs() < excised[0][nonzero].abs()
        ), "mass up must mean smaller applied force, i.e. slower acceleration"
    print(
        f"   centred {mp} kg: linear /{float(ratio):.5f} exact, yaw untouched"
    )


def test_offset_mass_produces_the_closed_form_trim():
    """Port-side weight under thrust: bow-to-port moment, no surge coupling."""
    mp, offset_y = 3.0, 0.4
    cfg = _cfg(payload_mass_kg=mp, payload_offset_y_m=offset_y)
    total = HULL_MASS_KG + mp
    delta_y = mp * offset_y / total
    mu = HULL_MASS_KG * mp / total
    izz_ratio = (HULL_IZZ_KGM2 + mu * offset_y**2) / HULL_IZZ_KGM2

    # Thrust-borne trim: full forward thrust from rest, zero yaw command.
    shipped = _run(SHIPPED, _env(1, cfg), (1.0, 0.0), (0.0, 0.0), 0.0)
    expected_tz = delta_y * THRUST_FWD_N / izz_ratio
    tz = float(shipped[1][0, 0, 2])
    assert tz > 0.0, "port-side weight must pull the bow to port (+z moment)"
    assert abs(tz - expected_tz) < 1e-5 * expected_tz + 1e-8, (tz, expected_tz)
    # No reciprocal surge coupling -- the anti-imbalance signature: surge is
    # exactly the mass-ratio-scaled thrust, nothing added or removed.
    fx = float(shipped[0][0, 0, 0])
    expected_fx = THRUST_FWD_N / (total / HULL_MASS_KG)
    assert abs(fx - expected_fx) < 1e-5 * expected_fx, (fx, expected_fx)

    # Drag-borne trim: bow weight, pure sway drift, no command at all.
    mp2, offset_x = 3.0, 0.3
    cfg2 = _cfg(payload_mass_kg=mp2, payload_offset_x_m=offset_x)
    delta_x = mp2 * offset_x / total
    izz_ratio2 = (HULL_IZZ_KGM2 + mu * offset_x**2) / HULL_IZZ_KGM2
    shipped2 = _run(SHIPPED, _env(1, cfg2), (0.0, 0.0), (0.0, 1.0), 0.0)
    sway_drag = -(27.0 + 18.0 * 1.0) * 1.0
    expected_tz2 = -delta_x * sway_drag / izz_ratio2
    tz2 = float(shipped2[1][0, 0, 2])
    assert abs(tz2 - expected_tz2) < 1e-5 * abs(expected_tz2) + 1e-8, (
        tz2,
        expected_tz2,
    )
    print(
        f"   trim: thrust-borne {tz:+.4f} Nm (predicted {expected_tz:+.4f}), "
        f"drag-borne {tz2:+.4f} Nm (predicted {expected_tz2:+.4f})"
    )


def _knob_battery(mass_grid, imb_grid, cap_grid, drag_grid) -> np.ndarray:
    """All knob combos at once through the shipped method's per-env buffers."""
    grids = np.array(
        np.meshgrid(mass_grid, imb_grid, cap_grid, drag_grid, indexing="ij")
    ).reshape(4, -1)
    count = grids.shape[1]
    cfg = _cfg(
        mass_scale_choices=(1.0,),
        thrust_imbalance_choices=(0.0,),
        thrust_cap_scale_choices=(1.0,),
        drag_scale_choices=(1.0,),
    )

    def setup(env):
        env._mass_scale_per_env = torch.as_tensor(
            grids[0], dtype=torch.float32
        )
        env._thrust_imbalance_per_env = torch.as_tensor(
            grids[1], dtype=torch.float32
        )
        env._thrust_cap_scale_per_env = torch.as_tensor(
            grids[2], dtype=torch.float32
        )
        env._drag_scale_per_env = torch.as_tensor(
            grids[3], dtype=torch.float32
        )

    return _probe_battery(SHIPPED, cfg, num_envs=count, setup=setup)


def _distinguishability(payload_cfg, mass_grid, imb_grid, cap_grid, drag_grid):
    """(min over combos of max mismatch, payload effect size), both in
    max-abs planar-wrench units over the probe battery. The coarse search is
    followed by one local refinement pass around its argmin (5x resolution,
    +/- one coarse step per axis), so "the grid was too coarse to find the
    impostor" is answered inside the test itself."""
    target = _probe_battery(SHIPPED, payload_cfg)[:, 0, :]      # (P, 3)
    nominal = _probe_battery(EXCISED, _cfg())[:, 0, :]          # (P, 3)
    effect = float(np.max(np.abs(target - nominal)))

    def _search(grids):
        knobs = _knob_battery(*grids)                            # (P, N, 3)
        mismatch = np.max(
            np.abs(knobs - target[:, None, :]), axis=(0, 2)
        )                                                        # (N,)
        best = int(np.argmin(mismatch))
        combos = np.array(
            np.meshgrid(*grids, indexing="ij")
        ).reshape(4, -1)
        return float(mismatch[best]), combos[:, best]

    grids = (mass_grid, imb_grid, cap_grid, drag_grid)
    coarse_min, centre = _search(grids)
    refined = tuple(
        np.linspace(centre[axis] - step, centre[axis] + step, 11)
        if grid.size > 1
        else grid
        for axis, (grid, step) in enumerate(
            zip(grids, [
                grid[1] - grid[0] if grid.size > 1 else 0.0 for grid in grids
            ])
        )
    )
    refined_min, _ = _search(refined)
    return min(coarse_min, refined_min), effect


def test_payload_is_not_reproducible_by_the_existing_knobs():
    """Numeric independence: grid-search the knob span, bound the residual.

    Two payload configurations, two searches each:
    (a) the dedicated (mass_scale x thrust_imbalance) plane, finely gridded --
        the two knobs the axis most resembles;
    (b) the full 4-knob box including thrust_cap_scale and drag_scale, which
        kills the c=d=k impersonation (scaling thrust and drag together mimics
        added mass on the linear channels but drags the yaw-damping moment
        with it, and the payload leaves that moment alone).
    The payload effect size is printed alongside so the residual can be
    judged as a fraction of the thing being faked.
    """
    fine_mass = np.linspace(0.85, 1.75, 181)
    fine_imb = np.linspace(-0.45, 0.45, 181)
    coarse_mass = np.linspace(0.85, 1.75, 19)
    coarse_imb = np.linspace(-0.45, 0.45, 19)
    coarse_cap = np.linspace(0.55, 1.45, 13)
    coarse_drag = np.linspace(0.55, 1.45, 13)

    centred = _cfg(payload_mass_kg=3.0)
    offset = _cfg(payload_mass_kg=3.0, payload_offset_y_m=0.4)

    for name, cfg in (("centred 3 kg", centred), ("3 kg at +0.4 m", offset)):
        plane_min, effect = _distinguishability(
            cfg, fine_mass, fine_imb, np.array([1.0]), np.array([1.0])
        )
        box_min, _ = _distinguishability(
            cfg, coarse_mass, coarse_imb, coarse_cap, coarse_drag
        )
        print(
            f"   {name}: effect {effect:.3f}, best (s,x)-plane residual "
            f"{plane_min:.3f}, best 4-knob residual {box_min:.3f} "
            "(max-abs planar wrench units)"
        )
        assert effect > 1.0, f"{name}: payload effect too small to matter"
        assert plane_min > 0.10 * effect, (
            f"{name}: a (mass_scale, thrust_imbalance) pair reproduces the "
            f"payload to within {plane_min:.4f} of an effect of {effect:.4f} "
            "-- the axis would be redundant"
        )
        assert box_min > 0.05 * effect, (
            f"{name}: a 4-knob combination reproduces the payload to within "
            f"{box_min:.4f} of an effect of {effect:.4f}"
        )


def test_composes_with_mass_scale():
    """Both axes hot: divisors compose to (s*m+mp)/m and (s*Iz+mu_s|d|^2)/Iz.

    The payload block takes its ratios against the mass_scale-d hull and the
    mass_scale block divides afterwards, so the totals must be exactly the
    loaded-hull divisors -- the subtlest line of the model, checked against
    hand-computed closed forms.
    """
    m, izz = HULL_MASS_KG, HULL_IZZ_KGM2
    s, mp, dy = 1.3, 3.0, 0.4

    cfg = _cfg(mass_scale=s, payload_mass_kg=mp)
    forces, torques = _run(SHIPPED, _env(1, cfg), (1.0, 0.5), (0.0, 0.0), 0.0)
    fx, tz = float(forces[0, 0, 0]), float(torques[0, 0, 2])
    expected_fx = THRUST_FWD_N / ((s * m + mp) / m)
    expected_tz = 0.5 * YAW_TORQUE_NM / s  # centred: yaw divisor is s alone
    assert abs(fx - expected_fx) < 1e-4, (fx, expected_fx)
    assert abs(tz - expected_tz) < 1e-4, (tz, expected_tz)

    cfg = _cfg(mass_scale=s, payload_mass_kg=mp, payload_offset_y_m=dy)
    _, torques = _run(SHIPPED, _env(1, cfg), (1.0, 0.0), (0.0, 0.0), 0.0)
    tz = float(torques[0, 0, 2])
    delta_y = mp * dy / (s * m + mp)
    mu_s = (s * m) * mp / (s * m + mp)
    expected_tz = (delta_y * THRUST_FWD_N) / ((s * izz + mu_s * dy * dy) / izz)
    assert abs(tz - expected_tz) < 1e-4, (tz, expected_tz)
    print(
        f"   composed with mass_scale {s}: fx {fx:.3f}, trim {tz:+.4f} Nm, "
        "both at closed form"
    )


def test_source_pins():
    """Cfg fields inert, reset draw on its own protocol group, naming rule."""
    cfg_text = CFG_SOURCE.read_text(encoding="utf-8")
    for line in (
        "payload_mass_kg: float = 0.0",
        "payload_mass_kg_choices: tuple = ()",
        "payload_offset_x_m: float = 0.0",
        "payload_offset_y_m: float = 0.0",
        "payload_ref_mass_kg: float = 17.26",
        "payload_ref_izz_kgm2: float = 2.376",
    ):
        assert line in cfg_text, f"cfg pin missing: {line}"

    env_text = ENV_SOURCE.read_text(encoding="utf-8")
    reset = env_text[env_text.index("def _reset_idx"):]
    assert "GROUP_PAYLOAD_MASS_KG," in reset, (
        "the payload choices draw is not on its own protocol group"
    )
    assert "self._payload_mass_kg_per_env[env_ids]" in reset

    assert _sdh.GROUP_PAYLOAD_MASS_KG == "actuator_payload_mass_kg"
    assert _sdh.GROUP_PAYLOAD_MASS_KG in _sdh.ACTUATOR_GROUPS
    entry = _sdh.actuator_scenario_entry(
        _sdh.GROUP_PAYLOAD_MASS_KG, [0.0, 3.0]
    )
    assert entry == {
        "actuator_payload_mass_kg": {"payload_mass_kg": [0.0, 3.0]}
    }
    print("   source pins PASS")


def main() -> None:
    print("1) inert bit-identity (shipped vs payload-block-excised)")
    test_inert_default_is_bit_identical()
    test_drawn_zero_envs_are_bit_identical_in_choices_mode()
    print("2) effect directions with closed-form magnitudes")
    test_centred_mass_slows_linear_channels_and_spares_yaw()
    test_offset_mass_produces_the_closed_form_trim()
    test_composes_with_mass_scale()
    print("3) distinguishability from the five existing axes")
    test_payload_is_not_reproducible_by_the_existing_knobs()
    print("4) source pins")
    test_source_pins()
    print("\n=> PASS")


if __name__ == "__main__":
    main()
