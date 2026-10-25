"""Independent CPU read-back of the v2 assets against the registry (python -B).

Checks, per registered profile: file hash, stage units/up axis, single rigid
body at the default prim, MassAPI mass, COM at the origin, full inertia tensor
rebuilt from principal moments + principal-axes quaternion, visual bounds,
collider count, provenance metadata. Also re-derives C3's published 1:1 WAM-V
windage (WINDAGE.md table row 3: 211 kg five-mesh assembly, waterline 0.0888 m
in base_link) from the new asset's own meshes, as a regression of scale,
placement and the silhouette method. Writes assets/probe_v2.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from shapely import box as shp_box
from shapely import polygons as shp_polygons
from shapely import union_all

HERE = Path(__file__).resolve().parent
WORK = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(WORK / "repo"))
v2 = load("probe_vehicles_v2", WORK / "repo/vehicles_v2.py")


def tensor_from_usd(ma) -> np.ndarray:
    diag = np.array(ma.GetDiagonalInertiaAttr().Get(), dtype=float)
    q = ma.GetPrincipalAxesAttr().Get()
    m = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(q))), dtype=float)   # row-vector convention
    R = m.T                                                              # columns = principal axes in body
    return R @ np.diag(diag) @ R.T


def mesh_points(prim) -> np.ndarray:
    pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float)
    return (np.c_[pts, np.ones(len(pts))] @ xf)[:, :3]


def probe_asset(spec) -> dict:
    path = HERE / spec.asset_relpath
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == spec.asset_sha256, (spec.name, digest)
    stage = Usd.Stage.Open(str(path))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0 and UsdGeom.GetStageUpAxis(stage) == "Z"
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 1
    body = bodies[0]
    ma = UsdPhysics.MassAPI(body)
    mass = float(ma.GetMassAttr().Get())
    com = np.array(ma.GetCenterOfMassAttr().Get(), dtype=float)
    tensor = tensor_from_usd(ma)
    spec_I = np.array(spec.inertia_com_kg_m2)
    assert abs(mass - spec.mass_kg) <= 1e-5 * spec.mass_kg, (mass, spec.mass_kg)
    assert np.all(com == 0.0)
    err = float(np.max(np.abs(tensor - spec_I)))
    # USD stores the principal quaternion as float32; near identity its angle is
    # ill-conditioned (d angle/d w = 2/sin(angle/2)), bounding the tensor error
    # at about 1e-5 of the largest moment. PhysX reads the same float32 values.
    assert err <= 2e-5 * np.max(np.abs(spec_I)), err
    xf = np.array(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    assert np.allclose(xf, np.eye(4)), "rigid body prim must carry no transform"
    visuals, colliders = [], 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI) \
                and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
            visuals.append(mesh_points(prim))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            colliders += 1
    pts = np.vstack(visuals)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return {"asset": path.name, "sha256": digest, "mass_kg": mass, "com_m": com.tolist(),
            "inertia_rebuilt_kg_m2": tensor.tolist(), "max_abs_inertia_error_kg_m2": err,
            "visual_bounds_m": [lo.tolist(), hi.tolist()], "visual_mesh_count": len(visuals),
            "collider_count": colliders, "custom_data": {k: str(v) for k, v in body.GetCustomData().items()}}


def c3_regression(spec) -> dict:
    """Recompute C3 WINDAGE.md row 3 from the new competition asset's meshes."""
    stage = Usd.Stage.Open(str(HERE / spec.asset_relpath))
    com_base = np.array(v2._data.WAMV["competition"]["com_base_link_m"])
    names = ("base_link_v0", "left_engine_link_v0", "right_engine_link_v0",
             "left_propeller_link_v0", "right_propeller_link_v0")
    tris = []
    for n in names:
        prim = stage.GetPrimAtPath("/WAMV/Visuals/" + n)
        assert prim.IsValid(), n
        pts = mesh_points(prim)
        idx = np.array(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        tris.append(pts[idx])
    tris = np.concatenate(tris)

    def union(axes):
        p = tris[:, :, axes]
        e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        p = p[np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) > 1e-14]
        return union_all([union_all(shp_polygons(p[i:i + 4096]), grid_size=1e-6) for i in range(0, len(p), 4096)],
                         grid_size=1e-6)
    waterline = 0.0888023591 - com_base[2]          # C3 WINDAGE.md: h = 0.0888023591 m in base_link
    clip = shp_box(-100, waterline, 100, 100)
    f, s = union([1, 2]).intersection(clip), union([0, 2]).intersection(clip)
    got = {"frontal_area_m2": f.area, "lateral_area_m2": s.area,
           "lateral_zbar_m": s.centroid.y - waterline, "frontal_zbar_m": f.centroid.y - waterline}
    c3 = {"frontal_area_m2": 0.6783940, "lateral_area_m2": 2.1235695,
          "lateral_zbar_m": 0.2981497, "frontal_zbar_m": 0.6106468}
    rel = {k: abs(got[k] - c3[k]) / c3[k] for k in c3}
    assert max(rel.values()) < 2e-4, rel
    return {"recomputed": got, "c3_published": c3, "max_relative_difference": max(rel.values())}


def main() -> None:
    out = {"profiles": {}}
    for name, spec in v2.VEHICLES_V2.items():
        out["profiles"][name] = probe_asset(spec)
    out["c3_wamv_windage_regression"] = c3_regression(v2.get_vehicle_v2("wamv_vrx_v2_opt1"))
    out["status"] = "PASS"
    (HERE / "probe_v2.json").write_text(json.dumps(out, indent=1) + "\n", encoding="ascii")
    for name, r in out["profiles"].items():
        print(name, r["asset"], "mass", r["mass_kg"], "Ierr", r["max_abs_inertia_error_kg_m2"],
              "colliders", r["collider_count"], "visuals", r["visual_mesh_count"])
    print("C3 regression", json.dumps(out["c3_wamv_windage_regression"]))
    print("PASS")


if __name__ == "__main__":
    main()
