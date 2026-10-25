"""Run only path-loaded CPU tests; forbid Isaac imports and bytecode writes."""
import importlib.abc
import os
from pathlib import Path
import runpy
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[2]


class NoIsaac(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "isaaclab" or fullname.startswith("isaaclab."):
            raise ImportError("CPU test runner forbids importing Isaac Lab")


sys.meta_path.insert(0, NoIsaac())

TESTS = (
    "scripts/tests_physics_v2/test_damping_v2.py",
    "scripts/tests_physics_v2/test_waves_v2.py",
    "scripts/tests_physics_v2/test_geometry_docking_v2.py",
    "scripts/tests_physics_v2/test_appearance_v2.py",
    "tasks/_shared/test_sea_state.py",
    "tasks/_shared/test_wave_fix_package.py",
    "tasks/path_hazard/test_path_hazard_geometry.py",
    "tasks/path_hazard/test_no_bypass.py",
    "tasks/docking/test_berth_geometry.py",
    "tasks/hazard_nav/test_mid_episode_appearance.py",
    "tasks/hazard_nav/test_payload_axis.py",
    "tasks/hazard_nav/test_thrust_imbalance.py",
)

if len(sys.argv) == 3 and sys.argv[1] == "--one":
    target = (REPO / sys.argv[2]).resolve()
    if not target.is_relative_to(REPO):
        raise ValueError("test path must stay inside copied repo")
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target)]
    runpy.run_path(str(target), run_name="__main__")
else:
    failed = 0
    print("CPU only; Isaac imports blocked; CUDA hidden; bytecode disabled.", flush=True)
    print("Python " + sys.version.replace("\n", " "), flush=True)
    for test in TESTS:
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--one", test]
        print("\nCOMMAND: python -B repo/scripts/tests_physics_v2/run_cpu_tests.py --one " + test, flush=True)
        result = subprocess.run(cmd, cwd=REPO.parent, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        print(result.stdout, end="", flush=True)
        print("EXIT CODE: " + str(result.returncode), flush=True)
        failed += result.returncode != 0
    print("\nRESULT: %d/%d test programs passed" % (len(TESTS)-failed, len(TESTS)), flush=True)
    raise SystemExit(bool(failed))
