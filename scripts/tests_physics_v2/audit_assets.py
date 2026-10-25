"""Read USD mass metadata without Isaac, simulation, or asset mutation."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from pxr import Usd, UsdPhysics

ROOT = Path(r"C:/Users/BRADY/usvbench/assets")
rows = []
for name in ("blueboat_physics.usd", "boat_physics.usdc"):
    path = ROOT/name
    stage = Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        api = UsdPhysics.MassAPI(prim)
        q = api.GetPrincipalAxesAttr().Get()
        rows.append(dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         prim=str(prim.GetPath()), mass_kg=api.GetMassAttr().Get(),
                         diagonal_inertia_kgm2=list(api.GetDiagonalInertiaAttr().Get()),
                         principal_axes_wxyz=[q.GetReal(), *q.GetImaginary()],
                         note="Read-only authored metadata; zero inertia means PhysX computation is UNVERIFIED."))
print(json.dumps(rows, indent=2))
