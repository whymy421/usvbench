"""No-Isaac validation of scripts/wave_probe_dump.py.

Two layers, neither launching Isaac Sim:

1. Source hygiene: the file AST-parses, carries no control characters
   (ord < 32 other than LF/CR) and no non-ASCII bytes.
2. Pure math: ``_roll_pitch_from_quat`` and ``_body_uvr`` are lifted out of the
   module source by AST -- so the tested code is the shipped code byte for
   byte, not a copy -- and exercised on fabricated CPU tensors.

Why the math is worth testing at all: the probe exists to decide whether roll
and pitch explain the wave drop. A sign error or an axis swap there would not
crash anything; it would quietly produce a clean-looking trace that answers the
question wrong, and the whole observation-contract plan is built on that answer.

Run:  python scripts/test_wave_probe_cpu.py
  or: python -m pytest scripts/test_wave_probe_cpu.py -v
"""

import ast
import io
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "wave_probe_dump.py")

import torch  # noqa: E402


def _load_math():
    """Extract the two pure helpers by AST, without importing the module.

    Importing wave_probe_dump would launch Isaac. Lifting the two function
    definitions keeps this a CPU test while still testing the shipped bytes.
    """
    tree = ast.parse(io.open(TARGET, encoding="utf-8").read())
    wanted = {"_roll_pitch_from_quat", "_body_uvr"}
    picked = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in picked} == wanted, "probe helpers were renamed"
    module = types.ModuleType("_probe_math")
    module.torch = torch
    exec(compile(ast.Module(body=picked, type_ignores=[]), TARGET, "exec"),
         module.__dict__)
    return module


M = _load_math()


def _quat_from_rpy(roll, pitch, yaw):
    """Scalar-first wxyz quaternion for an intrinsic Z-Y-X (yaw-pitch-roll)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def test_source_hygiene():
    raw = io.open(TARGET, "rb").read()
    ast.parse(raw.decode("utf-8"))
    assert not [b for b in raw if b < 32 and b not in (9, 10, 13)]
    assert not [b for b in raw if b > 127]


def test_level_hull_reads_zero_tilt():
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    roll, pitch = M._roll_pitch_from_quat(quat)
    assert abs(roll.item()) < 1e-6
    assert abs(pitch.item()) < 1e-6


def test_single_axis_tilt_is_exact():
    """One axis at a time, the probe's tilt IS the Euler angle, exactly.

    The probe reports tilt as the direction of world-up in the body frame --
    the same vector the restoring torque is built from -- not as ZYX Euler
    angles. Those two agree exactly whenever only one of roll/pitch is
    non-zero, which is the case this test pins.
    """
    for r_deg, p_deg in ((10, 0), (-10, 0), (0, 12), (0, -12), (25, 0), (0, 25)):
        r, p = math.radians(r_deg), math.radians(p_deg)
        quat = torch.tensor([_quat_from_rpy(r, p, 0.0)])
        roll, pitch = M._roll_pitch_from_quat(quat)
        assert abs(roll.item() - r) < 1e-6, (r_deg, p_deg, roll.item())
        assert abs(pitch.item() - p) < 1e-6, (r_deg, p_deg, pitch.item())


def test_coupled_tilt_tracks_euler_without_pretending_to_equal_it():
    """With BOTH axes tilted the two conventions differ at second order.

    This is a real property, not a defect: composing a roll and a pitch does
    not commute, so "the direction world-up points in the body frame" is not
    the same decomposition as "rotate by yaw, then pitch, then roll". The gap
    grows with tilt -- about 0.05 deg at (-8, 5) and about 0.9 deg at
    (20, -15). The probe deliberately keeps the world-up form because that is
    what tasks/_shared/restoring.py acts on, so the trace and the physics
    agree; the assertion here is that the two stay CLOSE and same-signed, not
    that they are equal. An earlier draft of this test demanded equality to
    1e-4 and failed the correct implementation.
    """
    for r_deg, p_deg, tol_deg in ((-8, 5, 0.2), (20, -15, 1.5), (12, 9, 0.5)):
        r, p = math.radians(r_deg), math.radians(p_deg)
        quat = torch.tensor([_quat_from_rpy(r, p, 0.0)])
        roll, pitch = M._roll_pitch_from_quat(quat)
        assert abs(math.degrees(roll.item()) - r_deg) < tol_deg, (
            r_deg, p_deg, math.degrees(roll.item()))
        assert abs(math.degrees(pitch.item()) - p_deg) < tol_deg, (
            r_deg, p_deg, math.degrees(pitch.item()))
        assert roll.item() * r > 0 and pitch.item() * p > 0, "sign flipped"


def test_tilt_is_yaw_invariant():
    """The whole point of reading world-up in the body frame.

    A hull tilted by a fixed amount must report the SAME roll and pitch no
    matter which way its bow points -- the exact property whose absence caused
    the July 'capsize by turning' bug in the restoring spring.
    """
    r, p = math.radians(9.0), math.radians(-6.0)
    base = None
    for yaw_deg in (0, 45, 90, 180, 270):
        quat = torch.tensor([_quat_from_rpy(r, p, math.radians(yaw_deg))])
        roll, pitch = M._roll_pitch_from_quat(quat)
        tilt = math.hypot(roll.item(), pitch.item())
        if base is None:
            base = tilt
        else:
            assert abs(tilt - base) < 1e-4, (yaw_deg, tilt, base)


def test_tilt_sign_agrees_with_the_restoring_torque():
    """Pin the sign convention to the physics, not to my arithmetic.

    tasks/_shared/restoring.py builds tau_x = -k_roll * b_y and
    tau_y = +k_pitch * b_x. A restoring torque opposes the tilt it corrects,
    so a POSITIVE roll must produce a NEGATIVE tau_x and a positive pitch a
    negative tau_y. The first draft of the probe had both signs inverted and
    every magnitude-only summary still looked perfectly reasonable, which is
    exactly why this assertion exists.
    """
    sys.path.insert(0, os.path.dirname(HERE))
    from tasks._shared.restoring import restoring_torque_body

    k_roll, k_pitch = 280.0, 141.0
    for r_deg, p_deg in ((12, 0), (-12, 0), (0, 9), (0, -9)):
        r, p = math.radians(r_deg), math.radians(p_deg)
        quat = torch.tensor([_quat_from_rpy(r, p, 0.0)])
        roll, pitch = M._roll_pitch_from_quat(quat)
        tau = restoring_torque_body(quat, k_roll, k_pitch)
        if abs(r_deg) > 0:
            assert roll.item() * tau[0, 0].item() < 0, (
                f"roll {r_deg} and tau_x {tau[0, 0].item():.3f} must oppose")
        if abs(p_deg) > 0:
            assert pitch.item() * tau[0, 1].item() < 0, (
                f"pitch {p_deg} and tau_y {tau[0, 1].item():.3f} must oppose")


def test_body_uvr_pure_surge():
    """Bow along +x, moving along +x: all surge, no sway."""
    quat = torch.tensor([_quat_from_rpy(0.0, 0.0, 0.0)])
    lin = torch.tensor([[1.5, 0.0, 0.0]])
    ang = torch.tensor([[0.0, 0.0, 0.3]])
    uvr, fwd = M._body_uvr(quat, lin, ang)
    assert abs(uvr[0, 0].item() - 1.5) < 1e-5
    assert abs(uvr[0, 1].item()) < 1e-5
    assert abs(uvr[0, 2].item() - 0.3) < 1e-6
    assert abs(fwd[0, 0].item() - 1.0) < 1e-5


def test_body_uvr_pure_sway():
    """Bow along +x, moving along +y: all sway, no surge.

    Sign convention must match tasks/_shared/kinematics.py, where sway is the
    component along left = (-fy, fx): motion to port is POSITIVE sway.
    """
    quat = torch.tensor([_quat_from_rpy(0.0, 0.0, 0.0)])
    lin = torch.tensor([[0.0, 2.0, 0.0]])
    ang = torch.tensor([[0.0, 0.0, 0.0]])
    uvr, _ = M._body_uvr(quat, lin, ang)
    assert abs(uvr[0, 0].item()) < 1e-5
    assert abs(uvr[0, 1].item() - 2.0) < 1e-5


def test_body_uvr_matches_kinematics_module():
    """The probe must report the SAME surge/sway the policy observation uses.

    If these two disagree, the trace cannot be read against the policy's own
    inputs, which is the entire diagnostic value of the dump.
    """
    sys.path.insert(0, os.path.dirname(HERE))
    from tasks._shared.kinematics import body_planar_rates

    for yaw_deg in (0, 30, 120, 200, 315):
        yaw = math.radians(yaw_deg)
        quat = torch.tensor([_quat_from_rpy(0.0, 0.0, yaw)])
        lin = torch.tensor([[0.7, -1.1, 0.0]])
        ang = torch.tensor([[0.0, 0.0, 0.25]])
        uvr, fwd = M._body_uvr(quat, lin, ang)
        ref = body_planar_rates(fwd, lin, ang[:, 2])
        for i, name in enumerate(("surge", "sway", "yaw_rate")):
            assert abs(uvr[0, i].item() - ref[0, i].item()) < 1e-5, (
                yaw_deg, name, uvr[0, i].item(), ref[0, i].item())


def test_probe_refuses_a_calm_task():
    """The refusal must be in the source: a calm trace answers nothing."""
    src = io.open(TARGET, encoding="utf-8").read()
    assert 'carries no sea state' in src
    assert "getattr(base, \"_sea\", None)" in src


def test_probe_defaults_to_the_dev_seed():
    """Seed 7 is the dev seed; this tool must never mint a headline number.

    The check is that the probe writes an .npz and never CONSTRUCTS a gcert
    filename -- an earlier draft asserted the string "gcert" was absent from
    the whole file and tripped over its own docstring, which says the tool
    writes "never a gcert JSON".
    """
    src = io.open(TARGET, encoding="utf-8").read()
    assert '"--eval-seed", type=int, default=7' in src
    assert "np.savez_compressed" in src
    code = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    body = code.split('"""', 2)[-1]  # drop the module docstring
    assert "gcert" not in body, "the probe must never write a certification file"


if __name__ == "__main__":
    failures = 0
    for key, value in sorted(dict(globals()).items()):
        if key.startswith("test_") and callable(value):
            try:
                value()
                print(f"PASS {key}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {key}: {type(exc).__name__}: {exc}")
    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)
