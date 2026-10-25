"""D5 v2 vessel asset builder (CPU only; run with python -B).

Builds, from read-only sources:
  wamv_vrx_v2_competition.usdc  VRX default launch assembly (vrx_sensors_enabled
                                 true, H thrusters): 283.385 kg
  wamv_vrx_v2_bare.usdc         VRX platform default (xacro defaults: base,
                                 two batteries, two engines, two propellers): 258 kg
  blueboat_v2.usda              legacy BlueBoat asset, canonicalized to
                                 bow +X / port +Y / up +Z, origin at the COM
and writes ../evidence/vessels_v2_derived.json with every derived number.

Frames. Every output asset uses FLU body axes (x bow, y port, z up) with the
actor origin AT THE AGGREGATE COM (USD MassAPI centerOfMass = 0). WAM-V axes
are the VRX base_link axes (bow +x, see wamv_aft_thrusters.xacro:4-5 engines
at x=-2.37). The VRX kinematic tree is expanded here by a small evaluator of
the pinned xacro macros (no ROS, no xacro binary): masses, inertias, joint
poses, meshes and collision primitives are read from the XML, not retyped.
Engine steering, propeller and sensor joints are frozen at zero.

Sources (pinned osrf/vrx 7609d1bd90ce7edb29d040a082f949e8b089c864, local
read-only checkout D:/claude code/scratch/vrx): see SOURCE_FILES below; the
SHA-256 of every file read is recorded in the output JSON and MANIFEST.json.
Nothing outside this task's work folder is written.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
from shapely import box as shp_box
from shapely import polygons as shp_polygons
from shapely import union_all

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / "repo"))
import vessel_physics_ref_v2 as ref  # noqa: E402

VRX = Path(r"D:\claude code\scratch\vrx")
URDF = VRX / "vrx_urdf"
COMMIT = "7609d1bd90ce7edb29d040a082f949e8b089c864"
XNS = "{http://ros.org/wiki/xacro}"
CNS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
NAMESPACE = "wamv"
RHO_WATER = 1000.0      # Surface.cc:63 default; WAM-V xacro sets no fluid_density
G_BENCH = 9.81          # benchmark gravity (D3 HydroSpec default)
G_VRX = 9.8             # SDF default world gravity used by VRX (report 5.2)

BLUEBOAT_SOURCE_USD = Path(r"C:\Users\BRADY\usvbench\assets\blueboat_physics.usd")
BLUEBOAT_SOURCE_SHA256 = "9acfd39ed31cad557ac701d57b7d56f9a420b773ffe2af01b4081dcba32a2865"
BLUEBOAT_SOURCE_WATERLINE_ABOVE_KEEL = 0.2232   # blueboat_build_usd.py:28
C3 = Path(r"D:\claude code\codex_tasks\20260930_c3_wind\work")

INPUT_SHA256: dict[str, str] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> Path:
    key = str(path.relative_to(VRX)).replace("\\", "/") if VRX in path.parents else str(path)
    INPUT_SHA256[key] = sha256(path)
    return path


# ----------------------------------------------------------------- xacro subset
_ALLOWED = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Call,
            ast.Compare, ast.Eq, ast.NotEq)
_FUNCS = {"cos": math.cos, "sin": math.sin, "radians": math.radians}


def ev(expr: str, env: dict):
    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError("unsupported xacro expression " + expr)
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name)
                                               and node.func.id in _FUNCS):
            raise ValueError("unsupported call in " + expr)
    scope = dict(_FUNCS, pi=math.pi)
    scope.update(env)
    return eval(compile(tree, "<xacro>", "eval"), {"__builtins__": {}}, scope)


def subst(text: str, env: dict) -> str:
    def rep(match):
        value = ev(match.group(1), env)
        return repr(float(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
    return re.sub(r"\$\{([^}]*)\}", rep, text)


def value_of(text: str, env: dict):
    text = text.strip()
    whole = re.fullmatch(r"\$\{([^}]*)\}", text)
    if whole:
        return ev(whole.group(1), env)
    try:
        return float(text)
    except ValueError:
        pass
    if text in ("True", "False", "true", "false"):
        return text.lower() == "true"
    return subst(text, env)


def floats(text: str, env: dict) -> np.ndarray:
    return np.array([float(t) for t in subst(text or "0 0 0", env).split()], dtype=float)


def rot(rpy) -> np.ndarray:
    return np.array(ref.rpy_matrix(*[float(a) for a in rpy]), dtype=float)


MACROS: dict[str, tuple[dict, ET.Element, str]] = {}


def load_macros(path: Path) -> ET.Element:
    root = ET.parse(record(path)).getroot()
    for macro in root.iter(XNS + "macro"):
        params = {}
        for m in re.finditer(r"(\w+)(?::=(?:'([^']*)'|(\S+)))?", macro.get("params", "")):
            name, quoted, bare = m.group(1), m.group(2), m.group(3)
            params[name] = quoted if quoted is not None else bare
        MACROS[macro.get("name")] = (params, macro, str(path.relative_to(VRX)).replace("\\", "/"))
    return root


def geometry(el: ET.Element, env: dict) -> dict:
    shape = el.find("geometry")[0]
    if shape.tag == "mesh":
        return {"type": "mesh", "filename": subst(shape.get("filename"), env),
                "scale": floats(shape.get("scale") or "1 1 1", env).tolist()}
    if shape.tag == "cylinder":
        return {"type": "cylinder", "radius": float(value_of(shape.get("radius"), env)),
                "length": float(value_of(shape.get("length"), env))}
    if shape.tag == "box":
        return {"type": "box", "size": floats(shape.get("size"), env).tolist()}
    if shape.tag == "sphere":
        return {"type": "sphere", "radius": float(value_of(shape.get("radius"), env))}
    raise ValueError("unsupported geometry " + shape.tag)


def origin(el: ET.Element | None, env: dict):
    o = None if el is None else el.find("origin")
    if o is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return floats(o.get("xyz"), env).tolist(), floats(o.get("rpy"), env).tolist()


def parse_link(el: ET.Element, env: dict, source: str) -> dict:
    link = {"name": subst(el.get("name"), env), "source": source, "visuals": [], "collisions": []}
    inertial = el.find("inertial")
    if inertial is not None:
        it = inertial.find("inertia")
        vals = {k: float(value_of(it.get(k), env)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        xyz, rpy = origin(inertial, env)
        link["inertial"] = {"mass": float(value_of(inertial.find("mass").get("value"), env)),
                            "xyz": xyz, "rpy": rpy,
                            "tensor": [[vals["ixx"], vals["ixy"], vals["ixz"]],
                                       [vals["ixy"], vals["iyy"], vals["iyz"]],
                                       [vals["ixz"], vals["iyz"], vals["izz"]]]}
    for child in el:
        if child.tag in ("visual", "collision"):
            xyz, rpy = origin(child, env)
            item = {"xyz": xyz, "rpy": rpy, "geometry": geometry(child, env),
                    "name": subst(child.get("name") or child.tag, env)}
            link["visuals" if child.tag == "visual" else "collisions"].append(item)
        elif child.tag.startswith(XNS) and child.tag[len(XNS):] in MACROS:
            # Macro emitting collision/visual elements inside a link (main_collisions).
            params, body, msource = MACROS[child.tag[len(XNS):]]
            menv = macro_env(params, child, env)
            for sub in body:
                if sub.tag in ("visual", "collision"):
                    xyz, rpy = origin(sub, menv)
                    item = {"xyz": xyz, "rpy": rpy, "geometry": geometry(sub, menv),
                            "name": subst(sub.get("name") or sub.tag, menv), "macro": msource}
                    link["visuals" if sub.tag == "visual" else "collisions"].append(item)
    return link


def macro_env(params: dict, call: ET.Element, outer: dict) -> dict:
    env = {"namespace": NAMESPACE}
    for name, default in params.items():
        if default is not None:
            env[name] = value_of(default, env)
    for key, raw in call.attrib.items():
        env[key] = value_of(raw, outer)
    missing = [p for p, d in params.items() if d is None and p not in env]
    if missing:
        raise ValueError("missing macro args " + ",".join(missing))
    return env


def expand(body: ET.Element, env: dict, source: str, links: list, joints: list) -> None:
    for el in body:
        tag = el.tag
        if tag == XNS + "property":
            env[el.get("name")] = value_of(el.get("value"), env)
        elif tag == "link":
            links.append(parse_link(el, env, source))
        elif tag == "joint":
            xyz, rpy = origin(el, env)
            joints.append({"name": subst(el.get("name"), env), "type": el.get("type"),
                           "parent": subst(el.find("parent").get("link"), env),
                           "child": subst(el.find("child").get("link"), env),
                           "xyz": xyz, "rpy": rpy, "source": source})
        elif tag in (XNS + "if", XNS + "unless"):
            cond = bool(value_of(el.get("value"), env))
            if cond == (tag == XNS + "if"):
                expand(el, env, source, links, joints)
        elif tag.startswith(XNS) and tag[len(XNS):] in MACROS:
            params, mbody, msource = MACROS[tag[len(XNS):]]
            expand(mbody, macro_env(params, el, env), msource, links, joints)
        # xacro:include, gazebo, plugin blocks carry no mass or geometry here.


def build_vrx_tree(with_sensors: bool):
    base_root = load_macros(URDF / "wamv_description/urdf/wamv_base.urdf.xacro")
    for rel in ("wamv_description/urdf/battery.xacro",
                "wamv_description/urdf/thrusters/engine.xacro",
                "wamv_description/urdf/cpu_cases.xacro",
                "wamv_gazebo/urdf/components/wamv_camera.xacro",
                "wamv_gazebo/urdf/components/wamv_gps.xacro",
                "wamv_gazebo/urdf/components/wamv_imu.xacro",
                "wamv_gazebo/urdf/components/wamv_3d_lidar.xacro",
                "wamv_gazebo/urdf/components/lidar.xacro",
                "wamv_gazebo/urdf/components/wamv_pinger.xacro",
                "wamv_gazebo/urdf/components/ball_shooter.xacro"):
        load_macros(URDF / rel)
    top = ET.parse(record(URDF / "wamv_gazebo/urdf/wamv_gazebo.urdf.xacro")).getroot()
    layout = ET.parse(record(URDF / "wamv_gazebo/urdf/thruster_layouts/wamv_aft_thrusters.xacro")).getroot()
    args = {a.get("name"): a.get("default") for a in top.findall(XNS + "arg")}
    assert args["thruster_config"] == "H" and args["vrx_sensors_enabled"] == "false"
    env = {"namespace": NAMESPACE}
    links, joints = [], []
    base_link = base_root.find("link")
    links.append(parse_link(base_link, env, "vrx_urdf/wamv_description/urdf/wamv_base.urdf.xacro"))
    for call in top.findall(XNS + "battery"):          # wamv_gazebo.urdf.xacro:44-45
        params, body, msource = MACROS["battery"]
        expand(body, macro_env(params, call, env), msource, links, joints)
    for call in layout.findall(XNS + "engine"):        # wamv_aft_thrusters.xacro:4-5
        params, body, msource = MACROS["engine"]
        expand(body, macro_env(params, call, env), msource, links, joints)
    if with_sensors:
        blocks = [el for el in top.iter(XNS + "if") if el.get("value") == "$(arg vrx_sensors_enabled)"]
        assert len(blocks) == 1                        # wamv_gazebo.urdf.xacro:156-184
        expand(blocks[0], env, "vrx_urdf/wamv_gazebo/urdf/wamv_gazebo.urdf.xacro", links, joints)
    return links, joints


def link_poses(links, joints):
    by_child = {j["child"]: j for j in joints}
    root = NAMESPACE + "/base_link"
    poses = {root: (np.eye(3), np.zeros(3))}
    names = [l["name"] for l in links]
    assert len(set(names)) == len(names)
    pending = [n for n in names if n != root]
    while pending:
        progressed = False
        for name in list(pending):
            j = by_child[name]
            if j["parent"] in poses:
                Rp, pp = poses[j["parent"]]
                poses[name] = (Rp @ rot(j["rpy"]), pp + Rp @ np.array(j["xyz"]))
                pending.remove(name)
                progressed = True
        assert progressed, "disconnected kinematic tree"
    return poses


def resolve_package(filename: str) -> Path:
    m = re.fullmatch(r"package://(wamv_description|vrx_gazebo|wamv_gazebo)/(.*)", filename)
    assert m, filename
    return URDF / m.group(1) / m.group(2)


def read_dae(path: Path):
    doc = ET.parse(record(path)).getroot()
    unit = float(doc.find("c:asset/c:unit", CNS).get("meter"))
    up = doc.find("c:asset/c:up_axis", CNS).text.strip()
    nodes = [n for n in doc.findall(".//c:visual_scene/c:node", CNS)
             if n.find("c:instance_geometry", CNS) is not None]
    assert len(nodes) == 1, path
    node = nodes[0]
    geom = doc.find(".//c:geometry[@id='" + node.find("c:instance_geometry", CNS).get("url")[1:] + "']", CNS)
    tri = geom.find("c:mesh/c:triangles", CNS)
    inputs = {x.get("semantic"): (int(x.get("offset")), x.get("source")[1:]) for x in tri.findall("c:input", CNS)}
    off, vid = inputs["VERTEX"]
    verts = geom.find(".//c:vertices[@id='" + vid + "']", CNS)
    sid = verts.find("c:input[@semantic='POSITION']", CNS).get("source")[1:]
    src = geom.find(".//c:source[@id='" + sid + "']", CNS)
    stride = int(src.find(".//c:accessor", CNS).get("stride"))
    pts = np.array(src.find("c:float_array", CNS).text.split(), dtype=float).reshape(-1, stride)[:, :3]
    mat = np.array(node.find("c:matrix", CNS).text.split(), dtype=float).reshape(4, 4)
    pts = (np.c_[pts, np.ones(len(pts))] @ mat.T)[:, :3] * unit
    if up == "Y_UP":   # COLLADA Y-up to Z-up: (x, y, z) -> (x, -z, y)
        pts = np.c_[pts[:, 0], -pts[:, 2], pts[:, 1]]
    else:
        assert up == "Z_UP", path
    width = max(v[0] for v in inputs.values()) + 1
    idx = np.array(tri.find("c:p", CNS).text.split(), dtype=np.int64).reshape(-1, width)[:, off]
    assert len(idx) == int(tri.get("count")) * 3 and idx.max() < len(pts)
    return pts, idx.reshape(-1, 3), {"unit_meter": unit, "up_axis": up}


def tessellate(geom: dict, n: int = 32):
    """Triangles of a primitive in its local frame (for windage/bounds only)."""
    if geom["type"] == "box":
        sx, sy, sz = (s / 2 for s in geom["size"])
        c = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                 (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
        return c, np.array(faces)
    if geom["type"] == "cylinder":
        r, h = geom["radius"], geom["length"] / 2
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        ring = np.c_[r * np.cos(a), r * np.sin(a)]
        pts = np.vstack([np.c_[ring, -h * np.ones(n)], np.c_[ring, h * np.ones(n)], [[0, 0, -h], [0, 0, h]]])
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
        return pts, np.array(faces)
    if geom["type"] == "sphere":
        r = geom["radius"]
        return tessellate({"type": "box", "size": [2 * r] * 3})
    raise ValueError(geom["type"])


def primitive_half_extent(geom: dict, R: np.ndarray) -> np.ndarray:
    if geom["type"] == "box":
        return np.abs(R) @ (np.array(geom["size"]) / 2)
    if geom["type"] == "cylinder":
        axis = R[:, 2]
        return np.abs(axis) * geom["length"] / 2 + geom["radius"] * np.sqrt(np.maximum(0.0, 1 - axis * axis))
    if geom["type"] == "sphere":
        return np.full(3, geom["radius"])
    raise ValueError(geom["type"])


def assemble(links, poses):
    parts = []
    for link in links:
        if "inertial" not in link:
            continue
        R, p = poses[link["name"]]
        ine = link["inertial"]
        Ri = R @ rot(ine["rpy"])
        parts.append((ine["mass"], tuple(p + R @ np.array(ine["xyz"])),
                      tuple(map(tuple, Ri @ np.array(ine["tensor"]) @ Ri.T)), link["name"], link["source"]))
    mass, com, tensor = ref.aggregate_inertia([(m, c, I) for m, c, I, _, _ in parts])
    # Independent numpy check of the pure-Python aggregation.
    m_np = sum(m for m, *_ in parts)
    c_np = sum(m * np.array(c) for m, c, *_ in parts) / m_np
    I_np = sum(np.array(I) + m * ((np.array(c) - c_np) @ (np.array(c) - c_np) * np.eye(3)
                                  - np.outer(np.array(c) - c_np, np.array(c) - c_np))
               for m, c, I, *_ in parts)
    assert abs(m_np - mass) < 1e-9 and np.allclose(c_np, com, atol=1e-12) and np.allclose(I_np, tensor, atol=1e-9)
    table = [{"link": n, "mass_kg": m, "com_base_m": list(c), "inertia_at_link_com_base_axes_kg_m2": [list(r) for r in I],
              "source": s} for m, c, I, n, s in parts]
    return mass, np.array(com), np.array(tensor), table


def silhouettes(tris: np.ndarray):
    def one(axes):
        pts = tris[:, :, axes]
        e1, e2 = pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]
        pts = pts[np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) > 1e-14]
        pieces = [union_all(shp_polygons(pts[i:i + 4096]), grid_size=1e-6) for i in range(0, len(pts), 4096)]
        return union_all(pieces, grid_size=1e-6)
    return one([1, 2]), one([0, 2])      # frontal (y,z) and lateral (x,z) as C3 windage_geometry.py:36-48


def windage_at(front, side, waterline_z: float) -> dict:
    clip = shp_box(-100, waterline_z, 100, 100)
    f, s = front.intersection(clip), side.intersection(clip)
    return {"waterline_actor_z_m": waterline_z, "frontal_area_m2": f.area, "lateral_area_m2": s.area,
            "frontal_centroid_height_above_water_m": f.centroid.y - waterline_z,
            "lateral_centroid_height_above_water_m": s.centroid.y - waterline_z,
            "lateral_centroid_x_from_com_m": s.centroid.x,
            "lateral_centroid_z_from_com_m": s.centroid.y}


def write_rigid_usd(path: Path, root_name: str, mass: float, tensor: np.ndarray, meta: dict,
                    visuals: list, colliders: list) -> dict:
    evals, evecs = np.linalg.eigh(tensor)
    for k in range(3):                     # canonical signs: principal axis k close to body axis k
        if evecs[k, k] < 0:
            evecs[:, k] *= -1
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1
    quat = Gf.Matrix3d(*evecs.T.flatten().tolist()).ExtractRotation().GetQuat()
    if path.exists():
        path.unlink()   # only this script's own outputs are ever rebuilt
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    body = UsdGeom.Xform.Define(stage, "/" + root_name)
    stage.SetDefaultPrim(body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
    ma = UsdPhysics.MassAPI.Apply(body.GetPrim())
    ma.CreateMassAttr(float(mass))
    ma.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(e) for e in evals]))
    ma.CreatePrincipalAxesAttr(Gf.Quatf(float(quat.GetReal()), Gf.Vec3f(*quat.GetImaginary())))
    for key, value in meta.items():
        body.GetPrim().SetCustomDataByKey(key, value)
    for name, pts, faces, color in visuals:
        mesh = UsdGeom.Mesh.Define(stage, "/" + root_name + "/Visuals/" + name)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1), dtype=np.int32)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        mesh.CreateExtentAttr([Gf.Vec3f(*pts.min(axis=0).tolist()), Gf.Vec3f(*pts.max(axis=0).tolist())])
    for name, center, R, geom in colliders:
        frame = UsdGeom.Xform.Define(stage, "/" + root_name + "/Collisions/" + name)
        frame.AddTranslateOp().Set(Gf.Vec3d(*center.tolist()))
        q = Gf.Matrix3d(*R.T.flatten().tolist()).ExtractRotation().GetQuat()
        frame.AddOrientOp().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*q.GetImaginary())))
        if geom["type"] == "cylinder":
            shape = UsdGeom.Cylinder.Define(stage, str(frame.GetPath()) + "/Shape")
            shape.CreateRadiusAttr(geom["radius"])
            shape.CreateHeightAttr(geom["length"])
            shape.CreateAxisAttr(UsdGeom.Tokens.z)
        else:
            assert geom["type"] == "box"
            shape = UsdGeom.Cube.Define(stage, str(frame.GetPath()) + "/Shape")
            shape.CreateSizeAttr(1.0)
            shape.AddScaleOp().Set(Gf.Vec3f(*[float(s) for s in geom["size"]]))
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        shape.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    stage.GetRootLayer().Save()
    return {"principal_moments_kg_m2": evals.tolist(),
            "principal_axes_quat_wxyz": [float(quat.GetReal()), *[float(x) for x in quat.GetImaginary()]]}


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def build_wamv(config: str) -> dict:
    with_sensors = config == "competition"
    links, joints = build_vrx_tree(with_sensors)
    poses = link_poses(links, joints)
    mass, com, tensor, table = assemble(links, poses)
    dyn = ET.parse(record(URDF / "wamv_gazebo/urdf/dynamics/wamv_gazebo_dynamics_plugin.xacro")).getroot()
    surfaces = [p for p in dyn.iter("plugin") if p.get("name") == "vrx::Surface"]
    assert len(surfaces) == 2
    points, hull_length, hull_radius = [], None, None
    for s in surfaces:
        hull_length = float(s.find("hull_length").text)
        hull_radius = float(s.find("hull_radius").text)
        pts = [tuple(float(t) for t in p.text.split()) for p in s.find("points").findall("point")]
        assert len(pts) == 2          # Surface.cc:254 divides hull_length by points.size() per plugin
        points += pts
    hydro = {}
    for label, g in (("g_bench_9p81", G_BENCH), ("g_vrx_9p8", G_VRX)):
        hydro[label] = ref.vrx_surface_hydrostatics(
            mass=mass, com_body=tuple(com), points_body=points, hull_length=hull_length,
            hull_radius=hull_radius, rho=RHO_WATER, g=g, points_per_plugin=2)
    # Visual triangles and colliders in the COM-centred actor frame.
    visuals, all_tris, collider_list, cbounds = [], [], [], []
    mesh_inputs = {}
    for link in links:
        R, p = poses[link["name"]]
        for k, vis in enumerate(link["visuals"]):
            Rv, pv = R @ rot(vis["rpy"]), p + R @ np.array(vis["xyz"])
            geom = vis["geometry"]
            if geom["type"] == "mesh":
                path = resolve_package(geom["filename"])
                local, faces, info = read_dae(path)
                local = local * np.array(geom["scale"])
                mesh_inputs[str(path.relative_to(VRX)).replace("\\", "/")] = info
            else:
                local, faces = tessellate(geom)
            pts = local @ Rv.T + pv - com
            label = sanitize(link["name"].split("/", 1)[1]) + "_v" + str(k)
            color = (0.85, 0.69, 0.18) if link["name"].endswith("base_link") else (0.18, 0.20, 0.23)
            visuals.append((label, pts, faces, color))
            all_tris.append(pts[faces])
        for k, col in enumerate(link["collisions"]):
            Rc, pc = R @ rot(col["rpy"]), p + R @ np.array(col["xyz"]) - com
            geom = col["geometry"]
            label = sanitize(link["name"].split("/", 1)[1] + "_" + col["name"]) + "_" + str(k)
            collider_list.append((label, pc, Rc, geom))
            h = primitive_half_extent(geom, Rc)
            cbounds.append((pc - h, pc + h))
    out = HERE / ("wamv_vrx_v2_" + config + ".usdc")
    vis_pts = np.concatenate([v[1] for v in visuals])
    meta = {"sourceRepository": "https://github.com/osrf/vrx", "sourceCommit": COMMIT,
            "configuration": config, "geometryScale": 1.0,
            "bodyFrame": "FLU: +X bow, +Y port, +Z up (VRX base_link axes); origin = aggregate COM",
            "comInVrxBaseLink_m": Gf.Vec3d(*com.tolist()),
            "license": "Apache-2.0 (see LICENSE_VRX.txt)",
            "modificationNotice": ("Converted by USVBench D5 from osrf/vrx " + COMMIT + " COLLADA/xacro to USD: "
                                   "scale 1.0, links merged into one rigid body with joints frozen at zero, "
                                   "origin moved to the aggregate COM, textures and Gazebo plugins omitted."),
            "builder": "assets/build_vessels_v2.py"}
    usd_info = write_rigid_usd(out, "WAMV", mass, tensor, meta, visuals, collider_list)
    front, side = silhouettes(np.concatenate(all_tris))
    h = hydro["g_bench_9p81"]
    waterline = -h["com_world_z_m"]           # upright approximation in the COM frame
    wind = windage_at(front, side, waterline)
    wind["sensitivity_waterline_minus_30mm"] = windage_at(front, side, waterline - 0.03)
    wind["sensitivity_waterline_plus_30mm"] = windage_at(front, side, waterline + 0.03)
    vlo, vhi = vis_pts.min(axis=0), vis_pts.max(axis=0)
    clo = np.min([b[0] for b in cbounds], axis=0)
    chi = np.max([b[1] for b in cbounds], axis=0)
    elo, ehi = np.minimum(vlo, clo), np.maximum(vhi, chi)
    capsule = ref.enclosing_capsule_for_rectangle(float(elo[0]), float(ehi[0]),
                                                  float(max(-elo[1], ehi[1])), 0.0)
    return {"config": config, "asset": out.name, "asset_sha256": sha256(out),
            "mass_kg": mass, "com_base_link_m": com.tolist(), "inertia_com_kg_m2": tensor.tolist(),
            **usd_info, "links": table, "link_count": len(links), "joint_count": len(joints),
            "joints": [{k: j[k] for k in ("name", "type", "parent", "child", "xyz", "rpy", "source")} for j in joints],
            "visual_count": len(visuals), "collider_count": len(collider_list),
            "mesh_inputs": mesh_inputs,
            "visual_bounds_com_m": [vlo.tolist(), vhi.tolist()],
            "collision_bounds_com_m": [clo.tolist(), chi.tolist()],
            "envelope_bounds_com_m": [elo.tolist(), ehi.tolist()],
            "footprint_rectangle_com_xy_m": [[float(elo[0]), float(elo[1])], [float(ehi[0]), float(elo[1])],
                                             [float(ehi[0]), float(ehi[1])], [float(elo[0]), float(ehi[1])]],
            "footprint_capsule": capsule,
            "surface_points_base_m": points, "surface_hull_length_m": hull_length,
            "surface_hull_radius_m": hull_radius, "hydrostatics": hydro, "windage": wind}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_blueboat() -> dict:
    """Canonical BlueBoat: reuse the verified CAD section hydrostatics of
    evidence/derive_blueboat.py (import only; its main() is not run)."""
    assert sha256(BLUEBOAT_SOURCE_USD) == BLUEBOAT_SOURCE_SHA256
    INPUT_SHA256[str(BLUEBOAT_SOURCE_USD)] = BLUEBOAT_SOURCE_SHA256
    derive = load_module("d5_derive_blueboat", WORK / "evidence/derive_blueboat.py")
    import trimesh
    stl = Path(r"C:\Users\BRADY\usvbench_gazebo\blueboat_cad\hull_set\PUB-BR-101381-001.STL")
    INPUT_SHA256[str(stl)] = sha256(stl)
    mesh = trimesh.load_mesh(stl, process=True)
    mesh.apply_translation([-mesh.bounds[:, 0].mean(), -mesh.bounds[0, 1], -mesh.bounds[:, 2].mean()])
    mesh.apply_scale(.001)
    fine = derive.hydro(mesh, 257)
    coarse = derive.hydro(mesh, 129)
    heights = np.linspace(1e-8, mesh.bounds[1, 1] - 1e-8, 513)
    areas = np.array([derive.props(derive.section(mesh, z))[0] for z in heights])
    from scipy.integrate import cumulative_trapezoid
    volume_curve = cumulative_trapezoid(2 * areas, heights, initial=0)
    max_volume = float(volume_curve[-1])

    def draft_for(mass_kg):
        return float(np.interp(mass_kg / RHO_WATER, volume_curve, heights))
    draft = fine["draft_m"]
    wl = derive.section(mesh, draft)       # columns: beam (x), length (z), metres
    wl_length = float(np.ptp(wl[:, 1]))
    wl_beam_one = float(np.ptp(wl[:, 0]))
    # Source frame S (bow -X, starboard +Y, z=0 at 0.2232 m above keel) -> hull
    # frame H (bow +X, port +Y, z=0 at the 17.26 kg waterline) -> actor frame A
    # (origin at COM).  blueboat_build_usd.py:3-5,28,53-55.
    src = Usd.Stage.Open(str(BLUEBOAT_SOURCE_USD))
    boat_src = src.GetPrimAtPath("/World/BlueBoat")
    ma_src = UsdPhysics.MassAPI(boat_src)
    com_s = np.array(ma_src.GetCenterOfMassAttr().Get(), dtype=float)
    diag = np.array(ma_src.GetDiagonalInertiaAttr().Get(), dtype=float)
    mass = float(ma_src.GetMassAttr().Get())
    flip = np.array([-1.0, -1.0, 1.0])
    z_shift = BLUEBOAT_SOURCE_WATERLINE_ABOVE_KEEL - draft
    com_h = com_s * flip + np.array([0.0, 0.0, z_shift])
    out = HERE / "blueboat_v2.usda"
    tmp = HERE / "_blueboat_flatten_tmp.usda"
    src.Flatten().Export(str(tmp))
    stage = Usd.Stage.Open(str(tmp))
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            assert not UsdGeom.Xformable(prim).GetOrderedXformOps(), "unexpected xform op " + str(prim.GetPath())
    boat = stage.GetPrimAtPath("/World/BlueBoat")
    ma = UsdPhysics.MassAPI(boat)
    ma.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    all_pts, tris, collision_pts, n_col = [], [], [], 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts = np.array(m.GetPointsAttr().Get(), dtype=np.float64) * flip + [0, 0, z_shift] - com_h
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        m.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*pts.min(axis=0).tolist()), Gf.Vec3f(*pts.max(axis=0).tolist())]))
        all_pts.append(pts)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_pts.append(pts)
            n_col += 1
        elif UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
            counts = np.array(m.GetFaceVertexCountsAttr().Get())
            assert np.all(counts == 3)
            idx = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            tris.append(pts[idx])
    for key, value in {"sourceAsset": str(BLUEBOAT_SOURCE_USD), "sourceSha256": BLUEBOAT_SOURCE_SHA256,
                       "bodyFrame": "FLU: +X bow, +Y port, +Z up; origin = COM",
                       "canonicalization": ("source rotated 180 deg about z (source bow -X, "
                                            "blueboat_build_usd.py:3-5,53) and translated so the COM is the origin; "
                                            "no scaling"),
                       "builder": "assets/build_vessels_v2.py"}.items():
        boat.SetCustomDataByKey(key, value)
    stage.GetRootLayer().Export(str(out))
    tmp.unlink()
    pts = np.vstack(all_pts)
    cpts = np.vstack(collision_pts)
    waterline = -com_h[2]
    front_mesh, side_mesh = silhouettes(np.concatenate(tris))
    usd_hull_windage = windage_at(front_mesh, side_mesh, waterline)
    # C3 saved full-resolution STL silhouettes (hull frame, waterline 0.2232 m above keel)
    from shapely import wkt
    INPUT_SHA256[str(C3 / "blueboat_front.wkt")] = sha256(C3 / "blueboat_front.wkt")
    INPUT_SHA256[str(C3 / "blueboat_side.wkt")] = sha256(C3 / "blueboat_side.wkt")
    front_c3 = wkt.loads((C3 / "blueboat_front.wkt").read_text())
    side_c3 = wkt.loads((C3 / "blueboat_side.wkt").read_text())
    wl_c3 = draft - BLUEBOAT_SOURCE_WATERLINE_ABOVE_KEEL
    c3_hull = windage_at(front_c3, side_c3, wl_c3)
    c3_hull["lateral_centroid_x_from_com_m"] = c3_hull["lateral_centroid_x_from_com_m"] - com_h[0]
    c3_hull["lateral_centroid_z_from_com_m"] = c3_hull["lateral_centroid_z_from_com_m"] - com_h[2]
    c3_check = windage_at(front_c3, side_c3, 0.0)
    # Superstructure band estimate (UNVERIFIED occupancy fractions; finalize_blueboat.py:17-19 method).
    public_height, public_beam, public_length = 0.46, 0.93, 1.20
    hull_top = float(mesh.bounds[1, 1])
    band = public_height - hull_top
    frac_front, frac_side = 0.35, 0.25
    extra_f, extra_l = frac_front * public_beam * band, frac_side * public_length * band
    extra_z = (hull_top - draft) + band / 2
    af, al = c3_hull["frontal_area_m2"] + extra_f, c3_hull["lateral_area_m2"] + extra_l
    selected = {
        "frontal_area_m2": af, "lateral_area_m2": al,
        "frontal_centroid_height_above_water_m": (c3_hull["frontal_area_m2"] * c3_hull["frontal_centroid_height_above_water_m"] + extra_f * extra_z) / af,
        "lateral_centroid_height_above_water_m": (c3_hull["lateral_area_m2"] * c3_hull["lateral_centroid_height_above_water_m"] + extra_l * extra_z) / al,
        # superstructure band assumed centred on the hull midpoint (x_hull = 0)
        "lateral_centroid_x_from_com_m": (c3_hull["lateral_centroid_x_from_com_m"] * c3_hull["lateral_area_m2"]
                                          + extra_l * (0.0 - com_h[0])) / al,
        "superstructure_band_height_m": band, "superstructure_frontal_fill": frac_front,
        "superstructure_lateral_fill": frac_side, "superstructure_frontal_area_m2": extra_f,
        "superstructure_lateral_area_m2": extra_l, "superstructure_centroid_height_above_water_m": extra_z,
        "upper_bound_full_band_frontal_area_m2": c3_hull["frontal_area_m2"] + public_beam * band,
        "upper_bound_full_band_lateral_area_m2": c3_hull["lateral_area_m2"] + public_length * band}
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    # Operational envelope: public deployed 1.20 x 0.93 m centred on the hull
    # (hull frame), expanded to contain every authored vertex, expressed about the COM.
    ex_lo = np.minimum(np.array([-0.60, -0.465]) - com_h[:2], lo[:2])
    ex_hi = np.maximum(np.array([0.60, 0.465]) - com_h[:2], hi[:2])
    capsule = ref.enclosing_capsule_for_rectangle(float(ex_lo[0]), float(ex_hi[0]),
                                                  float(max(-ex_lo[1], ex_hi[1])), 0.0)
    budget = {}
    with open(WORK / "evidence/blueboat_mass_budget_readback.txt", encoding="ascii") as f:
        budget["readback"] = f.read()
    return {"asset": out.name, "asset_sha256": sha256(out), "mass_kg": mass,
            "source_com_m": com_s.tolist(), "com_hull_frame_m": com_h.tolist(),
            "inertia_com_kg_m2": np.diag(diag).tolist(), "principal_axes_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "hydrostatics_fine": fine, "hydrostatics_coarse": coarse,
            "maximum_volume_m3": max_volume, "hull_height_m": hull_top,
            "draft_check": {"mass_16p9_kg_draft_m": draft_for(16.9), "mass_16p9_kg_freeboard_m": hull_top - draft_for(16.9),
                            "mass_29p5_kg_draft_m": draft_for(29.5), "mass_29p5_kg_freeboard_m": hull_top - draft_for(29.5)},
            "waterline_length_m": wl_length, "waterline_beam_per_hull_m": wl_beam_one,
            "waterline_footprint_beam_m": 2 * 0.3607 + wl_beam_one,
            "bounds_com_m": [lo.tolist(), hi.tolist()], "collision_mesh_count": n_col,
            "collision_bounds_com_m": [cpts.min(axis=0).tolist(), cpts.max(axis=0).tolist()],
            "footprint_rectangle_com_xy_m": [[float(ex_lo[0]), float(ex_lo[1])], [float(ex_hi[0]), float(ex_lo[1])],
                                             [float(ex_hi[0]), float(ex_hi[1])], [float(ex_lo[0]), float(ex_hi[1])]],
            "footprint_capsule": capsule,
            "windage_hull_only_c3_stl": c3_hull, "windage_hull_only_usd_decimated": usd_hull_windage,
            "windage_c3_regression_at_0p2232_draft": c3_check, "windage_selected": selected}


def main() -> None:
    commit = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(VRX), "rev-parse", "HEAD"],
                                     text=True).strip()
    assert commit == COMMIT, commit
    dirty = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(VRX), "status", "--porcelain"],
                                    text=True).strip()
    assert dirty == "", "VRX checkout is not clean"
    result = {"vrx_commit": commit, "vrx_checkout_clean": True,
              "wamv": {cfg: build_wamv(cfg) for cfg in ("competition", "bare")},
              "blueboat": build_blueboat()}
    record(VRX / "LICENSE")
    lic = HERE / "LICENSE_VRX.txt"
    assert sha256(lic) == sha256(VRX / "LICENSE"), "LICENSE_VRX.txt differs from the pinned VRX LICENSE"
    result["input_sha256"] = dict(sorted(INPUT_SHA256.items()))
    (WORK / "evidence/vessels_v2_derived.json").write_text(json.dumps(result, indent=1) + "\n", encoding="ascii")
    summary = {k: {kk: result["wamv"][k][kk] for kk in ("asset", "mass_kg", "com_base_link_m", "collider_count", "visual_count")}
               for k in result["wamv"]}
    summary["blueboat"] = {k: result["blueboat"][k] for k in ("asset", "mass_kg", "com_hull_frame_m", "collision_mesh_count")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
