"""Path-loaded CPU tests only: no Isaac import, CUDA, or bytecode writes."""
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
        if fullname.split(".")[0] in ("isaaclab", "isaacsim", "omni"):
            raise ImportError("D3 CPU runner forbids Isaac/Omniverse imports")


sys.meta_path.insert(0, NoIsaac())

if len(sys.argv)==3 and sys.argv[1]=="--one":
    target = (REPO/sys.argv[2]).resolve()
    if not target.is_relative_to(REPO):
        raise ValueError("test path outside copied repo")
    sys.path.insert(0,str(target.parent))
    sys.argv=[str(target)]
    runpy.run_path(str(target),run_name="__main__")
else:
    programs=("test_hydro.py","test_restoring_reference.py","test_thruster_mixer.py",
              "test_profiles_runtime.py","test_station_proof.py","spectral_convergence.py",
              "cpu_scenarios.py","test_froude_krylov.py")  # E2 adds the Froude-Krylov program
    failed=0
    print("D3 CPU only; no Isaac/Omniverse imports; CUDA hidden; bytecode disabled.",flush=True)
    print("Python "+sys.version.replace("\n"," "),flush=True)
    for program in programs:
        relative="scripts/tests_hydro_v2/"+program
        print("\nCOMMAND: python -B repo/scripts/tests_hydro_v2/run_cpu_tests.py --one "+relative,flush=True)
        result=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--one",relative],
                              cwd=REPO.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              encoding="utf-8",errors="replace",env={**os.environ,"PYTHONIOENCODING":"utf-8"})
        print(result.stdout,end="",flush=True)
        print("EXIT CODE: "+str(result.returncode),flush=True)
        failed += result.returncode!=0
    print("RESULT: %d/%d programs passed"%(len(programs)-failed,len(programs)),flush=True)
    raise SystemExit(bool(failed))
