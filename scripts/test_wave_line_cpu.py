"""No-Isaac validation of scripts/eval_obs_bridge.py and scripts/wave_acceptance.py.

Same three layers as scripts/test_classical_baseline_math.py, none of which
launches Isaac Sim:

1. Source hygiene: both files AST-parse, contain no control characters
   (ord < 32 other than LF/CR) and no non-ASCII bytes.
2. Argparse path: each module top (docstring + parser + pre-flight guards) is
   executed with a stub ``isaaclab.app.AppLauncher``; ``--help`` must exit 0,
   the refusal guards must exit 2, and valid flag sets must stop right before
   the (stubbed) app launch.
3. Pure math: the column-mapping logic is exercised on fabricated observation
   tensors against the frozen contract (tasks/_shared/obs_superset.py) and
   the prior-art bridge (scripts/obs_bridge.py), including the wave-task
   layout with channels the calm policies never saw. The per-episode record
   schema of eval_obs_bridge.py is diffed key-for-key against
   eval_v6_frozen.py.

Run:  python scripts/test_wave_line_cpu.py
  or: python -m pytest scripts/test_wave_line_cpu.py -v
"""

import ast
import math
import os
import re
import sys
import types

import torch

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPTS)
for path in (_REPO, _SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

BRIDGE_EVAL = os.path.join(_SCRIPTS, "eval_obs_bridge.py")
WAVE_PROBE = os.path.join(_SCRIPTS, "wave_acceptance.py")
FROZEN_EVAL = os.path.join(_SCRIPTS, "eval_v6_frozen.py")
SELF = os.path.abspath(__file__)

WAVE_ID = "Isaac-USV-StationKeep-BlueBoat-Wave-Direct-v1"
KIN_ID = "Isaac-USV-StationKeep-BlueBoat-Kin-Direct-v1"
NAV_ID = "Isaac-USV-StationKeep-BlueBoat-Direct-v1"
POOLED_ID = "Isaac-USV-HazardNav-Direct-v5"


# ---------------------------------------------------------------- layer 1


def test_source_hygiene():
    for target in (BRIDGE_EVAL, WAVE_PROBE, SELF):
        data = open(target, "rb").read()
        bad = [(i, b) for i, b in enumerate(data)
               if (b < 32 and b not in (10, 13)) or b > 126]
        assert not bad, f"control/non-ascii bytes in {target}: {bad[:5]}"
        ast.parse(data.decode("utf-8"), filename=target)


# ---------------------------------------------------------------- layer 2


def _module_top_source(target: str) -> str:
    """Source up to (excluding) the AppLauncher instantiation line."""
    src = open(target, "r", encoding="utf-8").read()
    cut = src.index("app = AppLauncher(args_cli).app")
    return src[:cut]


def _run_argparse(target: str, argv: list[str]) -> int:
    """Execute a module top with a stubbed isaaclab; return the exit code."""
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
    sys.argv = [os.path.basename(target)] + argv
    try:
        exec(compile(_module_top_source(target), target, "exec"),
             {"__name__": "__smoke__", "__file__": target})
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


def test_bridge_eval_argparse_help_and_guards():
    assert _run_argparse(BRIDGE_EVAL, ["--help"]) == 0
    # Valid: calm Kin champion bridged onto the wave task (SELF as a stand-in
    # existing checkpoint file). Stops right before the stubbed launch.
    assert _run_argparse(BRIDGE_EVAL, [
        "--checkpoint", SELF, "--train-task", KIN_ID, "--eval-task", WAVE_ID,
        "--episodes", "128", "--eval-seed", "42", "--level", "0", "--headless",
    ]) == -1
    # Nav-only (3-D) champion is also alignable onto the wave task.
    assert _run_argparse(BRIDGE_EVAL, [
        "--checkpoint", SELF, "--train-task", NAV_ID, "--eval-task", WAVE_ID,
        "--headless",
    ]) == -1
    # Refusals exit 2: unknown ids, explicitly unsupported pooled layouts,
    # missing checkpoint file, non-positive episode count.
    assert _run_argparse(BRIDGE_EVAL, [
        "--checkpoint", SELF, "--train-task", "Isaac-Not-A-Task-v1",
        "--eval-task", WAVE_ID]) == 2
    assert _run_argparse(BRIDGE_EVAL, [
        "--checkpoint", SELF, "--train-task", KIN_ID,
        "--eval-task", "Isaac-Not-A-Task-v1"]) == 2
    assert _run_argparse(BRIDGE_EVAL, [
        "--checkpoint", SELF, "--train-task", POOLED_ID,
        "--eval-task", WAVE_ID]) == 2
    assert _run_argparse(BRIDGE_EVAL, [
        "--checkpoint", SELF, "--train-task", KIN_ID,
        "--eval-task", POOLED_ID]) == 2
    assert _run_argparse(BRIDGE_EVAL, [
        "--checkpoint", os.path.join(_SCRIPTS, "no_such_ckpt.pt"),
        "--train-task", KIN_ID, "--eval-task", WAVE_ID]) == 2
    assert _run_argparse(BRIDGE_EVAL, [
        "--checkpoint", SELF, "--train-task", KIN_ID, "--eval-task", WAVE_ID,
        "--episodes", "0"]) == 2


def test_wave_probe_argparse_help_and_guards():
    assert _run_argparse(WAVE_PROBE, ["--help"]) == 0
    assert _run_argparse(WAVE_PROBE, [
        "--pairs", "0.30:1.5,0.45:2.25,0.60:3.0", "--headless"]) == -1
    assert _run_argparse(WAVE_PROBE, ["--headless"]) == -1  # default pairs
    assert _run_argparse(WAVE_PROBE, ["--pairs", "0.3-1.5"]) == 2
    assert _run_argparse(WAVE_PROBE, ["--pairs", "0.3:-2.0"]) == 2
    assert _run_argparse(WAVE_PROBE, ["--pairs", ""]) == 2
    assert _run_argparse(WAVE_PROBE, ["--steps", "10"]) == 2
    assert _run_argparse(WAVE_PROBE, ["--num-envs", "0"]) == 2


# ---------------------------------------------------------------- layer 3


def _extract(target: str, names: set[str], namespace: dict) -> dict:
    """Exec only the named top-level defs from a script source (byte-for-byte)."""
    tree = ast.parse(open(target, "r", encoding="utf-8").read(), filename=target)
    picked = [n for n in tree.body if getattr(n, "name", None) in names]
    assert {n.name for n in picked} == names, f"defs missing from {target}"
    module = ast.Module(body=picked, type_ignores=[])
    exec(compile(module, target, "exec"), namespace)
    return namespace


from tasks._shared.obs_superset import (  # noqa: E402
    NATIVE_LAYOUTS,
    SUPERSET_DIM_V2,
    extract_native,
    native_to_superset,
)
import obs_bridge  # noqa: E402  (scripts/obs_bridge.py - the prior art)

BRIDGE_NS = _extract(BRIDGE_EVAL, {"_bridge_mapping"}, {})
PROBE_NS = _extract(WAVE_PROBE, {"_parse_pairs", "_froude_rows"},
                    {"math": math, "FROUDE_SCALE": 10.0})

WAVE_LAYOUT = list(NATIVE_LAYOUTS[WAVE_ID])   # nav(3) + sea(54,55,56) + kin(3)
KIN_LAYOUT = list(NATIVE_LAYOUTS[KIN_ID])     # nav(3) + kin(3)
NAV_LAYOUT = list(NATIVE_LAYOUTS[NAV_ID])     # nav(3)


def test_frozen_layout_shapes():
    assert WAVE_LAYOUT == [0, 1, 2, 54, 55, 56, 51, 52, 53]
    assert KIN_LAYOUT == [0, 1, 2, 51, 52, 53]
    assert NAV_LAYOUT == [0, 1, 2]


def test_bridge_mapping_wave_to_calm():
    # Eval on wave (9-D), policy trained on calm Kin (6-D): every train
    # channel is present in the eval obs; the 3 sea channels are dropped.
    mapping = BRIDGE_NS["_bridge_mapping"](KIN_LAYOUT, WAVE_LAYOUT)
    assert mapping == [0, 1, 2, 6, 7, 8]
    # Nav-only train task keeps just the first three eval channels.
    assert BRIDGE_NS["_bridge_mapping"](NAV_LAYOUT, WAVE_LAYOUT) == [0, 1, 2]


def test_bridge_mapping_zero_fill_direction():
    # Reverse direction: wave-trained policy on the calm Kin task; the sea
    # channels have no source and must be zero-filled (None).
    mapping = BRIDGE_NS["_bridge_mapping"](WAVE_LAYOUT, KIN_LAYOUT)
    assert mapping == [0, 1, 2, None, None, None, 3, 4, 5]
    # Fabricated fully-disjoint layouts map to all-None (the script refuses).
    assert BRIDGE_NS["_bridge_mapping"]([54, 55], [0, 1, 2]) == [None, None]


def test_contract_projection_on_fabricated_tensors():
    # The exact per-step path eval_obs_bridge.py runs: eval native -> superset
    # scatter -> train native gather, on a batch with distinctive values.
    obs_wave = torch.arange(2 * 9, dtype=torch.float32).reshape(2, 9) + 1.0
    projected = extract_native(
        native_to_superset(obs_wave, WAVE_ID, dim=SUPERSET_DIM_V2), KIN_ID)
    assert projected.shape == (2, 6)
    expected = obs_wave[:, [0, 1, 2, 6, 7, 8]]
    assert torch.equal(projected, expected)
    # And the prior-art helper produces the identical result.
    assert torch.equal(obs_bridge.project(obs_wave, WAVE_ID, KIN_ID), expected)


def test_contract_projection_zero_fills_unseen_channels():
    obs_kin = torch.arange(6, dtype=torch.float32) + 1.0
    projected = obs_bridge.project(obs_kin, KIN_ID, WAVE_ID)
    assert projected.shape == (9,)
    assert torch.equal(projected[:3], obs_kin[:3])          # nav carried
    assert torch.equal(projected[3:6], torch.zeros(3))      # sea zero-filled
    assert torch.equal(projected[6:9], obs_kin[3:6])        # kinematics carried


def test_contract_refusals():
    obs_wave = torch.zeros(9)
    try:
        native_to_superset(obs_wave, "Isaac-Not-A-Task-v1")
        raise AssertionError("unknown id must raise KeyError")
    except KeyError:
        pass
    try:
        native_to_superset(torch.zeros(15), POOLED_ID)
        raise AssertionError("pooled layout must raise ValueError")
    except ValueError:
        pass
    try:
        native_to_superset(torch.zeros(8), WAVE_ID)  # wrong native width
        raise AssertionError("width mismatch must raise ValueError")
    except ValueError:
        pass


def test_parse_pairs():
    parse = PROBE_NS["_parse_pairs"]
    assert parse("0.30:1.5,0.60:3.0") == [(0.30, 1.5), (0.60, 3.0)]
    assert parse(" 0.45:2.25 ") == [(0.45, 2.25)]
    for bad in ("0.3-1.5", "0.3:", "0.3:0", "-1:2", ""):
        try:
            parse(bad)
            raise AssertionError(f"pairs {bad!r} must be rejected")
        except ValueError:
            pass


def test_apply_set_dotted_paths_and_tuples():
    ns = _extract(WAVE_PROBE, {"_apply_set"}, {})
    cfg = types.SimpleNamespace(
        seed=42, layout_max_attempts=100, flag=False,
        sea_state=types.SimpleNamespace(hs_range=(0.3, 0.6), f_max=0.5))
    ns["_apply_set"](cfg, "sea_state.hs_range=0.4,0.4")
    assert cfg.sea_state.hs_range == (0.4, 0.4)
    ns["_apply_set"](cfg, "sea_state.f_max=0.8")
    assert cfg.sea_state.f_max == 0.8
    ns["_apply_set"](cfg, "seed=7")
    assert cfg.seed == 7 and isinstance(cfg.seed, int)
    ns["_apply_set"](cfg, "flag=true")
    assert cfg.flag is True
    for bad in ("nope=1", "sea_state.nope=1", "sea_state.f_max"):
        try:
            ns["_apply_set"](cfg, bad)
            raise AssertionError(f"--set {bad!r} must be rejected")
        except SystemExit:
            pass


def test_froude_rows():
    rows = PROBE_NS["_froude_rows"]([(0.45, 2.25)])
    assert abs(rows[0]["full_hs_m"] - 4.5) < 1e-9
    assert abs(rows[0]["full_tp_s"] - 2.25 * math.sqrt(10.0)) < 1e-9


def _record_schema_keys(target: str) -> set[str]:
    """Keys written into per-episode records between 'rec = {' and append."""
    src = open(target, "r", encoding="utf-8").read()
    start = src.index("rec = {")
    stop = src.index("records.append(rec)")
    block = src[start:stop]
    literal_keys = set(re.findall(r'^\s*"(\w+)":', block, flags=re.M))
    assigned_keys = set(re.findall(r'rec\["(\w+)"\]', block))
    return literal_keys | assigned_keys


def test_record_schema_matches_eval_v6_frozen():
    frozen = _record_schema_keys(FROZEN_EVAL)
    bridged = _record_schema_keys(BRIDGE_EVAL)
    assert frozen == bridged, (
        f"record schema drifted: only-frozen={sorted(frozen - bridged)} "
        f"only-bridge={sorted(bridged - frozen)}")
    assert {"env", "ep", "success", "tts_s", "path_length_m"} <= frozen


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
