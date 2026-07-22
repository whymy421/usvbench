"""Validate visual-only BlueBoat composition and unchanged physics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


EXT = (
    r"E:\anaconda\envs\isaaclab\Lib\site-packages\isaacsim\extscache"
    r"\omni.usd.libs-1.0.1+69cbf6ad.wx64.r.cp311"
)
sys.path.insert(0, EXT)
for subdir in ("bin", "lib"):
    dll_dir = os.path.join(EXT, subdir)
    if os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)

from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSICS_PATH = REPO_ROOT / "assets" / "blueboat_physics.usd"
VISUAL_PATH = REPO_ROOT / "assets" / "blueboat_visual.usd"
VISUAL_SCOPES = ("/World/BlueBoat/Visuals", "/World/BlueBoat/VisualsCAD")
CAD_SCOPE = "/World/BlueBoat/VisualsCAD"
HULL_PATHS = (
    f"{CAD_SCOPE}/Hull01_Starboard",
    f"{CAD_SCOPE}/Hull02_Port",
)
COLLISION_PATHS = (
    "/World/BlueBoat/Collisions/Hull01_Starboard",
    "/World/BlueBoat/Collisions/Hull02_Port",
)
EXPECTED_PHYSICS_PRIMS = {
    "/World/BlueBoat",
    *COLLISION_PATHS,
}
BANNED_VISUAL_APIS = {
    "PhysicsCollisionAPI",
    "PhysicsMeshCollisionAPI",
    "PhysicsMassAPI",
    "PhysicsRigidBodyAPI",
}


class Results:
    def __init__(self) -> None:
        self.failed = False

    def check(self, name: str, condition: bool, detail: str) -> None:
        label = "PASS" if condition else "FAIL"
        print(f"[{label}] {name}: {detail}")
        self.failed |= not condition


def physics_schemas(prim: Usd.Prim) -> tuple[str, ...]:
    return tuple(str(schema) for schema in prim.GetAppliedSchemas() if "Physics" in str(schema))


def mesh_counts(stage: Usd.Stage, path: str) -> tuple[int, int]:
    mesh = UsdGeom.Mesh.Get(stage, path)
    return len(mesh.GetPointsAttr().Get()), len(mesh.GetFaceVertexCountsAttr().Get())


def prim_bounds(
    cache: UsdGeom.BBoxCache, stage: Usd.Stage, path: str
) -> tuple[np.ndarray, np.ndarray]:
    bbox = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.asarray(bbox.GetMin(), dtype=np.float64), np.asarray(
        bbox.GetMax(), dtype=np.float64
    )


def collect_authored_attributes(layer: Sdf.Layer) -> list[tuple[str, str]]:
    attributes: list[tuple[str, str]] = []

    def visit(path: Sdf.Path) -> None:
        spec = layer.GetObjectAtPath(path)
        if isinstance(spec, Sdf.AttributeSpec):
            attributes.append((str(path), spec.name))

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return attributes


def mesh_component_z_ranges(
    points: np.ndarray, faces: np.ndarray
) -> list[tuple[int, float, float]]:
    parent = np.arange(len(points), dtype=np.int32)

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = int(parent[vertex])
        return vertex

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second, third in faces:
        union(int(first), int(second))
        union(int(first), int(third))

    face_roots = np.fromiter(
        (find(int(vertex)) for vertex in faces[:, 0]),
        dtype=np.int32,
        count=len(faces),
    )
    components = []
    for root in np.unique(face_roots):
        component_faces = faces[face_roots == root]
        z_values = points[np.unique(component_faces), 2]
        components.append((len(component_faces), float(z_values.min()), float(z_values.max())))
    return components


def main() -> int:
    results = Results()
    print(f"Physics asset: {PHYSICS_PATH}")
    print(f"Visual asset:  {VISUAL_PATH}")

    physics = Usd.Stage.Open(str(PHYSICS_PATH))
    visual = Usd.Stage.Open(str(VISUAL_PATH))
    results.check("open stages", bool(physics and visual), "both USD stages opened")
    if not physics or not visual:
        return 1

    # (a) Neither the replacement CAD nor the hidden legacy visuals may carry physics.
    visual_api_violations: list[str] = []
    for scope_path in VISUAL_SCOPES:
        scope = visual.GetPrimAtPath(scope_path)
        if not scope:
            visual_api_violations.append(f"missing {scope_path}")
            continue
        for prim in Usd.PrimRange(scope):
            applied = {str(value).split(":", 1)[0] for value in prim.GetAppliedSchemas()}
            banned = sorted(applied & BANNED_VISUAL_APIS)
            if banned:
                visual_api_violations.append(f"{prim.GetPath()}: {banned}")
    results.check(
        "visual scopes have no physics APIs",
        not visual_api_violations,
        "none" if not visual_api_violations else "; ".join(visual_api_violations),
    )

    legacy_visibility = UsdGeom.Imageable(
        visual.GetPrimAtPath("/World/BlueBoat/Visuals")
    ).GetVisibilityAttr().Get()
    results.check(
        "legacy visual override",
        legacy_visibility == UsdGeom.Tokens.invisible,
        f"visibility={legacy_visibility}",
    )

    # (b) Applied schemas, mass properties, and collision topology are unchanged.
    physics_prim_schemas = {
        str(prim.GetPath()): physics_schemas(prim)
        for prim in physics.Traverse()
        if physics_schemas(prim)
    }
    visual_prim_schemas = {
        str(prim.GetPath()): physics_schemas(prim)
        for prim in visual.Traverse()
        if physics_schemas(prim)
    }
    physics_set_ok = set(visual_prim_schemas) == EXPECTED_PHYSICS_PRIMS
    schemas_equal = visual_prim_schemas == physics_prim_schemas
    results.check(
        "physics API prim set",
        physics_set_ok,
        f"composed={sorted(visual_prim_schemas)}",
    )
    results.check(
        "physics applied schemas unchanged",
        schemas_equal,
        "identical" if schemas_equal else f"physics={physics_prim_schemas}; visual={visual_prim_schemas}",
    )

    physics_mass = UsdPhysics.MassAPI(physics.GetPrimAtPath("/World/BlueBoat"))
    visual_mass = UsdPhysics.MassAPI(visual.GetPrimAtPath("/World/BlueBoat"))
    mass_values_physics = (
        physics_mass.GetMassAttr().Get(),
        physics_mass.GetCenterOfMassAttr().Get(),
        physics_mass.GetDiagonalInertiaAttr().Get(),
    )
    mass_values_visual = (
        visual_mass.GetMassAttr().Get(),
        visual_mass.GetCenterOfMassAttr().Get(),
        visual_mass.GetDiagonalInertiaAttr().Get(),
    )
    mass_equal = mass_values_visual == mass_values_physics
    results.check(
        "mass properties unchanged",
        mass_equal,
        f"mass={mass_values_visual[0]}, com={mass_values_visual[1]}, inertia={mass_values_visual[2]}",
    )

    collision_details = []
    collisions_equal = True
    for path in COLLISION_PATHS:
        original_counts = mesh_counts(physics, path)
        composed_counts = mesh_counts(visual, path)
        collisions_equal &= original_counts == composed_counts
        collision_details.append(
            f"{Path(path).name} points/faces={composed_counts[0]}/{composed_counts[1]}"
        )
    results.check(
        "collision topology unchanged",
        collisions_equal,
        "; ".join(collision_details),
    )

    # (c) Visual size, hull length, and spacing sanity.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    hull_bounds = [prim_bounds(cache, visual, path) for path in HULL_PATHS]
    hull_lengths = [hi[0] - lo[0] for lo, hi in hull_bounds]
    hull_centers_y = [0.5 * (lo[1] + hi[1]) for lo, hi in hull_bounds]
    hull_length = max(hull_lengths)
    spacing = abs(hull_centers_y[0] - hull_centers_y[1])

    cad_meshes = [
        prim
        for prim in Usd.PrimRange(visual.GetPrimAtPath(CAD_SCOPE))
        if prim.IsA(UsdGeom.Mesh)
    ]
    cad_bounds = [prim_bounds(cache, visual, str(prim.GetPath())) for prim in cad_meshes]
    overall_lo = np.min(np.stack([item[0] for item in cad_bounds]), axis=0)
    overall_hi = np.max(np.stack([item[1] for item in cad_bounds]), axis=0)
    overall_extent = overall_hi - overall_lo
    results.check(
        "hull length sanity",
        1.0 <= hull_length <= 1.5,
        f"length={hull_length:.6f} m (individual={hull_lengths[0]:.6f}, {hull_lengths[1]:.6f})",
    )
    results.check(
        "hull spacing sanity",
        abs(spacing - 0.7214) <= 0.02,
        f"spacing={spacing:.6f} m, centers=({hull_centers_y[0]:.6f}, {hull_centers_y[1]:.6f})",
    )
    results.check(
        "overall visual bounds",
        bool(np.all(overall_extent <= 2.5)),
        f"extent=({overall_extent[0]:.6f}, {overall_extent[1]:.6f}, {overall_extent[2]:.6f}) m",
    )

    deck_mesh = UsdGeom.Mesh.Get(visual, f"{CAD_SCOPE}/DeckFrame")
    deck_points = np.asarray(deck_mesh.GetPointsAttr().Get(), dtype=np.float64)
    deck_faces = np.asarray(
        deck_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
    ).reshape(-1, 3)
    deck_components = mesh_component_z_ranges(deck_points, deck_faces)
    hull_top_z = max(item[1][2] for item in hull_bounds)
    floating_components = [
        component for component in deck_components if component[1] > hull_top_z + 0.2
    ]
    exploded_bridges = np.ptp(deck_points[deck_faces, 2], axis=1) > 0.5
    results.check(
        "no exploded or floating deck components",
        not floating_components and not np.any(exploded_bridges),
        f"components={len(deck_components)}, deck z-range={deck_points[:, 2].min():.6f}.."
        f"{deck_points[:, 2].max():.6f} m, cutoff={hull_top_z + 0.2:.6f} m, "
        f"exploded bridges={np.count_nonzero(exploded_bridges)}",
    )

    triangle_details = []
    total_triangles = 0
    for prim in cad_meshes:
        point_count, triangle_count = mesh_counts(visual, str(prim.GetPath()))
        total_triangles += triangle_count
        triangle_details.append(
            f"{prim.GetName()}={triangle_count:,} tris/{point_count:,} points"
        )
    print(f"[INFO] CAD geometry: {'; '.join(triangle_details)}; total={total_triangles:,} tris")

    # (d) Inspect only the new root layer's authored specs, not composed values.
    visual_layer = Sdf.Layer.FindOrOpen(str(VISUAL_PATH))
    authored_attributes = collect_authored_attributes(visual_layer)
    forbidden_authored = [
        path
        for path, name in authored_attributes
        if "physics:" in name.lower() or "mass" in name.lower()
    ]
    collision_authored = [
        path for path, _ in authored_attributes if "/Collisions/" in path
    ]
    relative_composition = list(visual_layer.subLayerPaths) == ["./blueboat_physics.usd"]
    results.check(
        "relative physics composition",
        relative_composition,
        f"subLayerPaths={list(visual_layer.subLayerPaths)}",
    )
    results.check(
        "new layer authors no physics/mass/collision opinions",
        not forbidden_authored and not collision_authored,
        f"physics/mass attrs={forbidden_authored or 'none'}, collision attrs={collision_authored or 'none'}",
    )

    with VISUAL_PATH.open("rb") as stream:
        header = stream.read(8)
    is_usdc = header == b"PXR-USDC"
    file_size = VISUAL_PATH.stat().st_size
    results.check(
        "binary USDC and size cap",
        is_usdc and file_size < 80 * 1024 * 1024,
        f"header={header!r}, size={file_size / (1024 * 1024):.2f} MiB",
    )

    print(f"[INFO] Measured hull length: {hull_length:.6f} m")
    print(f"[INFO] Measured hull spacing: {spacing:.6f} m")
    print(f"[INFO] Visual file size: {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MiB)")
    print("VALIDATION FAILED" if results.failed else "VALIDATION PASSED")
    return 1 if results.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
