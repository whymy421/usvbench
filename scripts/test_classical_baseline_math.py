"""No-Isaac validation of scripts/classical_baseline.py.

Three layers, none of which launches Isaac Sim:

1. Source hygiene: the file AST-parses and contains no control characters
   (ord < 32 other than LF/CR) and no non-ASCII bytes.
2. Argparse path: the module top (docstring + parser) is executed with a stub
   ``isaaclab.app.AppLauncher`` injected into ``sys.modules``; ``--help`` must
   exit 0 and the validation guards must exit 2 on bad flag combinations.
   The stub never lets execution reach the real app launch.
3. Pure math: ``LOSPIDController``, ``DockingController._pid`` and
   ``_policy_obs`` are extracted from the module source by AST (so the
   certified math is tested byte-for-byte, not a copy) and exercised with
   fabricated CPU tensors.

Run:  python scripts/test_classical_baseline_math.py
  or: python -m pytest scripts/test_classical_baseline_math.py -v
"""

import ast
import math
import os
import sys
import types
from collections.abc import Mapping

import torch

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classical_baseline.py")


# ---------------------------------------------------------------- layer 1


def test_source_hygiene():
    data = open(BASELINE, "rb").read()
    bad = [(i, b) for i, b in enumerate(data) if (b < 32 and b not in (10, 13)) or b > 126]
    assert not bad, f"control/non-ascii bytes in classical_baseline.py: {bad[:5]}"
    ast.parse(data.decode("utf-8"), filename=BASELINE)


# ---------------------------------------------------------------- layer 2


def _module_top_source() -> str:
    """Source up to (excluding) the AppLauncher instantiation line."""
    src = open(BASELINE, "r", encoding="utf-8").read()
    cut = src.index("app_launcher = AppLauncher(args_cli)")
    return src[:cut]


def _run_argparse(argv: list[str]) -> int:
    """Execute the module top with a stubbed isaaclab; return the exit code."""
    stub_app = types.ModuleType("isaaclab.app")

    class AppLauncher:  # noqa: D401 - stub
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true")
            parser.add_argument("--device", type=str, default=None)

        def __init__(self, args):
            raise AssertionError("test must never launch the app")

    stub_app.AppLauncher = AppLauncher
    stub_root = types.ModuleType("isaaclab")
    stub_root.app = stub_app

    saved_modules = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.app")}
    saved_argv = sys.argv
    sys.modules["isaaclab"] = stub_root
    sys.modules["isaaclab.app"] = stub_app
    sys.argv = ["classical_baseline.py"] + argv
    try:
        exec(compile(_module_top_source(), BASELINE, "exec"), {"__name__": "__smoke__"})
    except SystemExit as exc:  # argparse --help (0) or parser.error (2)
        return int(exc.code or 0)
    finally:
        sys.argv = saved_argv
        for key, value in saved_modules.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return -1  # fell through: parsing succeeded and stopped before app launch


def test_argparse_help_and_guards():
    assert _run_argparse(["--help"]) == 0
    # Valid flag sets parse and stop right before the (stubbed) app launch.
    assert _run_argparse(
        ["--task", "Isaac-USV-Dock-BlueBoat-Direct-v1", "--protocol", "mission",
         "--level", "3", "--out", "r.json", "--headless"]
    ) == -1
    assert _run_argparse(
        ["--task", "Isaac-USV-StationKeep-BlueBoat-Current-Direct-v1",
         "--protocol", "mission", "--current-speed", "2.0", "--headless"]
    ) == -1
    # Guard rails exit 2.
    assert _run_argparse(["--task", "T", "--protocol", "throughput", "--out", "x.json"]) == 2
    assert _run_argparse(["--task", "T", "--protocol", "mission", "--level", "-1"]) == 2
    assert _run_argparse(["--task", "T", "--protocol", "mission", "--spawn-distance", "0"]) == 2
    assert _run_argparse(["--task", "T", "--protocol", "mission", "--current-speed", "-1"]) == 2
    assert _run_argparse(["--task", "T", "--protocol", "mission", "--num_envs", "0"]) == 2


# ---------------------------------------------------------------- layer 3


def _extract(names: set[str]) -> dict:
    """Exec only the named top-level defs from the baseline source."""
    tree = ast.parse(open(BASELINE, "r", encoding="utf-8").read(), filename=BASELINE)
    picked = [n for n in tree.body if getattr(n, "name", None) in names]
    assert {n.name for n in picked} == names, "baseline defs went missing"
    module = ast.Module(body=picked, type_ignores=[])
    namespace = {"torch": torch, "math": math, "Mapping": Mapping}
    exec(compile(module, BASELINE, "exec"), namespace)
    return namespace


NS = _extract({"_policy_obs", "LOSPIDController", "_boat_bow_2d",
               "_linear_velocity_xy", "DockingController"})
DT = 1.0 / 60.0


def _los(num_envs=4, ki=0.1, pass_through=False):
    return NS["LOSPIDController"](
        num_envs=num_envs, device=torch.device("cpu"), dtype=torch.float32,
        dt=DT, dist_scale=15.0, zone_radius=2.0, slow_radius=5.0,
        kp=2.0, ki=ki, kd=0.5, pass_through=pass_through,
    )


def test_policy_obs_selection():
    tensor = torch.zeros(4, 3)
    assert NS["_policy_obs"]({"policy": tensor}) is not None
    assert torch.equal(NS["_policy_obs"](tensor), tensor)
    try:
        NS["_policy_obs"](torch.zeros(4))
        raise AssertionError("1-D obs must be rejected")
    except RuntimeError:
        pass


def test_los_first_step():
    ctrl = _los()
    obs = torch.tensor([
        [1.0, 0.0, 10.0 / 15.0],                    # aligned, far
        [0.0, 1.0, 10.0 / 15.0],                    # 90 deg off, far
        [1.0, 0.0, 1.0 / 15.0],                     # inside the 2 m zone
        [math.cos(0.3), math.sin(0.3), 3.5 / 15.0], # slight error, braking band
    ])
    action = ctrl.act(obs)
    assert action.shape == (4, 2)
    thrust, yaw = action[:, 0], action[:, 1]
    # env0: error 0 -> yaw 0; profile (10-2)/(5-2) clamps to 1.
    assert abs(yaw[0].item()) < 1e-6 and abs(thrust[0].item() - 1.0) < 1e-6
    # env1: error pi/2, |error| >= pi/4 -> creep 0.2; yaw saturates at +1.
    assert abs(thrust[1].item() - 0.2) < 1e-6 and abs(yaw[1].item() - 1.0) < 1e-6
    # env2: inside the zone -> both commands zero, PID memory cleared.
    assert thrust[2].item() == 0.0 and yaw[2].item() == 0.0
    assert ctrl.integral[2].item() == 0.0 and not ctrl.initialized[2].item()
    # env3: yaw = kp*0.3 + ki*(0.3*dt), no derivative on the first step;
    # thrust = (3.5-2)/(5-2) = 0.5.
    expected_yaw = 2.0 * 0.3 + 0.1 * (0.3 * DT)
    assert abs(yaw[3].item() - expected_yaw) < 1e-5
    assert abs(thrust[3].item() - 0.5) < 1e-5


def test_los_derivative_and_integral():
    ctrl = _los()
    far = 10.0 / 15.0
    ctrl.act(torch.tensor([[math.cos(0.3), math.sin(0.3), far]] * 4))
    action = ctrl.act(torch.tensor([[math.cos(0.1), math.sin(0.1), far]] * 4))
    # integral = (0.3 + 0.1)*dt, derivative = (0.1-0.3)/dt = -12 -> saturates.
    expected = 2.0 * 0.1 + 0.1 * (0.4 * DT) + 0.5 * (-0.2 / DT)
    assert expected < -1.0 and abs(action[0, 1].item() - (-1.0)) < 1e-6
    assert abs(ctrl.integral[0].item() - 0.4 * DT) < 1e-6


def test_los_zone_reentry_has_no_derivative_kick():
    ctrl = _los(num_envs=2)
    ctrl.act(torch.tensor([[1.0, 0.0, 1.0 / 15.0]]* 2))     # inside: memory cleared
    action = ctrl.act(torch.tensor([[0.0, 1.0, 4.0 / 15.0]]* 2))  # drifted out
    # initialized was False -> derivative suppressed on re-engagement.
    expected = min(1.0, 2.0 * (math.pi / 2) + 0.1 * (math.pi / 2 * DT))
    assert abs(action[0, 1].item() - expected) < 1e-6


def test_los_reset():
    ctrl = _los()
    ctrl.act(torch.tensor([[0.0, 1.0, 10.0 / 15.0]] * 4))
    ctrl.reset(torch.tensor([True, False, True, False]))
    assert ctrl.integral[0].item() == 0.0 and not ctrl.initialized[0].item()
    assert ctrl.integral[1].item() != 0.0 and ctrl.initialized[1].item()


def test_los_pass_through_thrust():
    ctrl = _los(num_envs=3, pass_through=True)
    obs = torch.tensor([
        [1.0, 0.0, 1.0 / 15.0],    # aligned, even at close range: full thrust
        [0.0, 1.0, 10.0 / 15.0],   # 90 deg off: cos=0 -> creep 0.2
        [-1.0, 0.0, 10.0 / 15.0],  # reversed: cos clamps at 0 -> creep 0.2
    ])
    thrust = ctrl.act(obs)[:, 0]
    assert abs(thrust[0].item() - 1.0) < 1e-6
    assert abs(thrust[1].item() - 0.2) < 1e-6
    assert abs(thrust[2].item() - 0.2) < 1e-6
    # pass_through never clears PID memory (no hold zone).
    assert ctrl.initialized.all().item()


def _dock_pid(kp=0.0, ki=0.0, kd=1.0 / 60.0):
    return NS["DockingController"](
        base=None, num_envs=2, device=torch.device("cpu"),
        dtype=torch.float32, dt=DT, brake_radius=6.0, kp=kp, ki=ki, kd=kd,
    )


def test_dock_pid_wrapped_derivative():
    ctrl = _dock_pid()
    no_reset = torch.zeros(2, dtype=torch.bool)
    ctrl._pid(torch.tensor([3.0, 0.0]), no_reset)
    yaw = ctrl._pid(torch.tensor([-3.0, 0.5]), no_reset)
    # env0 jumps +3.0 -> -3.0 across the pi seam: the wrapped delta is
    # +(2*pi-6) ~= +0.2832, NOT -6.0. kd*delta/dt with kd=dt gives the delta.
    assert abs(yaw[0].item() - (2.0 * math.pi - 6.0)) < 1e-5
    assert abs(yaw[1].item() - 0.5) < 1e-5


def test_dock_pid_reset_mask_suppresses_derivative():
    ctrl = _dock_pid(kp=1.0)
    ctrl._pid(torch.tensor([0.5, 0.5]), torch.zeros(2, dtype=torch.bool))
    yaw = ctrl._pid(torch.tensor([0.3, 0.6]), torch.tensor([True, False]))
    # env0 was reset: pure kp term. env1 keeps its derivative (kd = dt),
    # yaw = 0.6 + (0.6 - 0.5) = 0.7, inside the +/-1 clamp.
    assert abs(yaw[0].item() - 0.3) < 1e-5
    assert abs(yaw[1].item() - 0.7) < 1e-5


def test_dock_controller_phase_init_and_reset():
    ctrl = _dock_pid()
    assert (ctrl.phase == ctrl.APPROACH).all().item()
    ctrl.phase[0] = ctrl.HOLD
    ctrl.integral[0] = 1.0
    ctrl.reset(torch.tensor([True, False]))
    assert ctrl.phase[0].item() == ctrl.APPROACH and ctrl.integral[0].item() == 0.0


def _main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except BaseException as exc:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print("all tests passed" if not failures else f"{failures} test(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
