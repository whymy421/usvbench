"""Build the composed, visual-only BlueBoat USD asset.

The two hull meshes are taken from the official bare-hull CAD and aligned to
the existing collision mesh bounds.  Deck/frame geometry is taken from the
official full-vehicle CAD, clipped above the bare hulls, and vertex-clustered
on a 4 mm grid.  All coordinates are baked into mesh points in metres.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh


EXT = (
    r"E:\anaconda\envs\isaaclab\Lib\site-packages\isaacsim\extscache"
    r"\omni.usd.libs-1.0.1+69cbf6ad.wx64.r.cp311"
)
sys.path.insert(0, EXT)
for subdir in ("bin", "lib"):
    dll_dir = os.path.join(EXT, subdir)
    if os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)

from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSICS_USD = REPO_ROOT / "assets" / "blueboat_physics.usd"
OUTPUT_USD = REPO_ROOT / "assets" / "blueboat_visual.usd"
CAD_ROOT = Path(r"C:\Users\BRADY\usvbench_gazebo\blueboat_cad")
HULL_FILES = (
    CAD_ROOT / "hull_set" / "PUB-BR-101381-001.STL",
    CAD_ROOT / "hull_set" / "PUB-BR-101381-002.STL",
)
FULL_VEHICLE = CAD_ROOT / "full_vehicle" / "PUB-BR-101447_RevA.STL"
HULL_PATHS = (
    "/World/BlueBoat/Collisions/Hull01_Starboard",
    "/World/BlueBoat/Collisions/Hull02_Port",
)
TARGET_HULL_Y = (0.3607, -0.3607)
GRID_MM = 4.0
EXPLODED_BRIDGE_Z_SPAN_M = 0.5
STL_RECORD_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
)


def bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def collision_bounds(stage: Usd.Stage, path: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = UsdGeom.Mesh.Get(stage, path)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    return bounds(points)


def load_and_align_hull(
    stl_path: Path,
    target_bounds: tuple[np.ndarray, np.ndarray],
    target_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map STL (width, up, length) to USD (-length, width, up)."""
    source = trimesh.load_mesh(stl_path, process=True)
    source_points = np.asarray(source.vertices, dtype=np.float64)
    source_faces = np.asarray(source.faces, dtype=np.int32)
    source_lo, source_hi = bounds(source_points)
    target_lo, target_hi = target_bounds

    source_width_center = 0.5 * (source_lo[0] + source_hi[0])
    source_length_center = 0.5 * (source_lo[2] + source_hi[2])
    target_length_center = 0.5 * (target_lo[0] + target_hi[0])

    points = np.empty_like(source_points, dtype=np.float32)
    points[:, 0] = (
        target_length_center - (source_points[:, 2] - source_length_center) * 0.001
    )
    points[:, 1] = target_y + (source_points[:, 0] - source_width_center) * 0.001
    points[:, 2] = target_lo[2] + (source_points[:, 1] - source_lo[1]) * 0.001

    # The axis map has negative determinant, so reverse winding.
    faces = source_faces[:, [0, 2, 1]].copy()
    return points, faces, np.array([source_lo[1], source_hi[1]], dtype=np.float64)


def binary_stl_memmap(path: Path) -> np.memmap:
    with path.open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
    expected_size = 84 + 50 * triangle_count
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected binary STL size for {path}")
    return np.memmap(
        path,
        dtype=STL_RECORD_DTYPE,
        mode="r",
        offset=84,
        shape=(triangle_count,),
    )


def scan_full_vehicle(stl: np.memmap) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, len(stl), 500_000):
        vertices = np.asarray(stl["vertices"][start : start + 500_000])
        chunk = vertices.reshape(-1, 3)
        lo = np.minimum(lo, chunk.min(axis=0))
        hi = np.maximum(hi, chunk.max(axis=0))
    return lo, hi


def cluster_deck_geometry(
    stl: np.memmap,
    full_bounds: tuple[np.ndarray, np.ndarray],
    hull_vertical_max_mm: float,
    target_x_center: float,
    target_z_bottom: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Clip away hulls, then grid-snap/re-index full-vehicle triangles."""
    lo, hi = full_bounds
    lateral_center = 0.5 * (lo[0] + hi[0])
    length_center = 0.5 * (lo[2] + hi[2])
    vertical_bottom = lo[1]
    cutoff = hull_vertical_max_mm + 0.1
    clustered_faces: list[np.ndarray] = []
    source_kept = 0

    for start in range(0, len(stl), 500_000):
        triangles = np.asarray(
            stl["vertices"][start : start + 500_000], dtype=np.float32
        )
        # The strict centroid clip removes the complete bare-hull envelope.
        keep = triangles[:, :, 1].mean(axis=1) > cutoff
        triangles = triangles[keep]
        source_kept += len(triangles)
        if not len(triangles):
            continue

        snapped = np.rint(triangles / GRID_MM).astype(np.int32)
        nondegenerate = (
            np.any(snapped[:, 0] != snapped[:, 1], axis=1)
            & np.any(snapped[:, 1] != snapped[:, 2], axis=1)
            & np.any(snapped[:, 0] != snapped[:, 2], axis=1)
        )
        clustered_faces.append(snapped[nondegenerate])

    quantized_faces = np.concatenate(clustered_faces, axis=0)
    quantized_points, inverse = np.unique(
        quantized_faces.reshape(-1, 3), axis=0, return_inverse=True
    )
    faces = inverse.reshape(-1, 3).astype(np.int32)

    # Grid clustering can map adjacent source facets onto the same triangle.
    canonical = np.sort(faces, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    faces = faces[np.sort(first)]

    cad_points = quantized_points.astype(np.float64) * GRID_MM
    points = np.empty_like(cad_points, dtype=np.float32)
    points[:, 0] = target_x_center - (cad_points[:, 2] - length_center) * 0.001
    points[:, 1] = (cad_points[:, 0] - lateral_center) * 0.001
    points[:, 2] = target_z_bottom + (cad_points[:, 1] - vertical_bottom) * 0.001
    faces = faces[:, [0, 2, 1]].copy()
    return points, faces, source_kept


def filter_floating_deck_components(
    points: np.ndarray,
    faces: np.ndarray,
    hull_top_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove exploded-position CAD geometry while preserving attached hardware.

    The source assembly contains a set of triangles spanning about 0.8 m from
    the deck to a detached accessory.  Those triangles make the two pieces
    topologically connected despite the empty space visible in the model.
    Remove only these impossible vertical bridges before applying the requested
    connected-component minimum-height rule.
    """
    face_z_span = np.ptp(points[faces, 2], axis=1)
    bridge_faces = face_z_span > EXPLODED_BRIDGE_Z_SPAN_M
    if np.any(bridge_faces):
        print(
            "Removed exploded bridge triangles: "
            f"{np.count_nonzero(bridge_faces):,} "
            f"(z-span {face_z_span[bridge_faces].min():.6f}.."
            f"{face_z_span[bridge_faces].max():.6f} m)"
        )
        faces = faces[~bridge_faces]

    parent = np.arange(len(points), dtype=np.int32)
    rank = np.zeros(len(points), dtype=np.uint8)

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = int(parent[vertex])
        return vertex

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    for first, second, third in faces:
        union(int(first), int(second))
        union(int(first), int(third))

    face_roots = np.fromiter(
        (find(int(vertex)) for vertex in faces[:, 0]),
        dtype=np.int32,
        count=len(faces),
    )
    component_limit = hull_top_z + 0.2
    keep_faces = np.zeros(len(faces), dtype=bool)
    component_stats = []
    for root in np.unique(face_roots):
        component_face_indices = np.flatnonzero(face_roots == root)
        component_vertices = np.unique(faces[component_face_indices])
        z_values = points[component_vertices, 2]
        minimum_z = float(z_values.min())
        maximum_z = float(z_values.max())
        keep = minimum_z <= component_limit
        component_stats.append(
            (
                minimum_z,
                maximum_z,
                len(component_face_indices),
                len(component_vertices),
                keep,
                component_face_indices,
            )
        )

    print(f"Deck component cutoff: min z <= {component_limit:.6f} m")
    for index, stats in enumerate(sorted(component_stats, key=lambda item: (item[0], item[1]))):
        minimum_z, maximum_z, face_count, vertex_count, keep, face_indices = stats
        keep_faces[face_indices] = keep
        action = "KEEP" if keep else "DROP"
        print(
            f"[{action}] component {index:02d}: triangles={face_count:,}, "
            f"points={vertex_count:,}, z=[{minimum_z:.6f}, {maximum_z:.6f}] m"
        )

    kept_faces = faces[keep_faces]
    used_points, inverse = np.unique(kept_faces.reshape(-1), return_inverse=True)
    compact_points = points[used_points]
    compact_faces = inverse.reshape(-1, 3).astype(np.int32)
    dropped_components = sum(not stats[4] for stats in component_stats)
    dropped_faces = len(faces) - len(kept_faces)
    print(
        f"Deck component filter: kept {len(kept_faces):,} triangles; "
        f"dropped {dropped_faces:,} triangles in {dropped_components} component(s)"
    )
    return compact_points, compact_faces


def define_mesh(
    stage: Usd.Stage,
    path: str,
    points: np.ndarray,
    faces: np.ndarray,
    color: tuple[float, float, float],
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    usd_points = Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, dtype=np.float32))
    mesh.CreatePointsAttr(usd_points)
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1), dtype=np.int32))
    )
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(usd_points))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    mesh.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.constant)
    return mesh


def main() -> None:
    for required in (PHYSICS_USD, *HULL_FILES, FULL_VEHICLE):
        if not required.is_file():
            raise FileNotFoundError(required)

    physics = Usd.Stage.Open(str(PHYSICS_USD))
    collision_target_bounds = [collision_bounds(physics, path) for path in HULL_PATHS]

    hull_data = []
    hull_vertical_ranges = []
    for stl_path, target, target_y in zip(
        HULL_FILES, collision_target_bounds, TARGET_HULL_Y
    ):
        points, faces, vertical_range = load_and_align_hull(stl_path, target, target_y)
        hull_data.append((points, faces))
        hull_vertical_ranges.append(vertical_range)

    full_stl = binary_stl_memmap(FULL_VEHICLE)
    full_bounds = scan_full_vehicle(full_stl)
    collision_lo, collision_hi = collision_target_bounds[0]
    target_x_center = 0.5 * (collision_lo[0] + collision_hi[0])
    deck_points, deck_faces, source_deck_triangles = cluster_deck_geometry(
        full_stl,
        full_bounds,
        max(value[1] for value in hull_vertical_ranges),
        target_x_center,
        collision_lo[2],
    )
    hull_top_z = max(target[1][2] for target in collision_target_bounds)
    deck_points, deck_faces = filter_floating_deck_components(
        deck_points, deck_faces, hull_top_z
    )

    if OUTPUT_USD.exists():
        OUTPUT_USD.unlink()
    stage = Usd.Stage.CreateNew(str(OUTPUT_USD))
    stage.GetRootLayer().subLayerPaths.append("./blueboat_physics.usd")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    old_visuals = UsdGeom.Imageable(stage.OverridePrim("/World/BlueBoat/Visuals"))
    old_visuals.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    scope = UsdGeom.Scope.Define(stage, "/World/BlueBoat/VisualsCAD")
    scope.GetPrim().SetCustomDataByKey(
        "build:alignment",
        "points baked in metres; USD (x,y,z)=(-STL length, STL width, STL up)",
    )
    scope.GetPrim().SetCustomDataByKey(
        "build:hullSource", "official PUB-BR-101381-001/-002.STL"
    )
    scope.GetPrim().SetCustomDataByKey(
        "build:deckSource",
        "official PUB-BR-101447_RevA.STL; above-hull clip; 4 mm grid",
    )

    define_mesh(
        stage,
        "/World/BlueBoat/VisualsCAD/Hull01_Starboard",
        hull_data[0][0],
        hull_data[0][1],
        (0.08, 0.08, 0.09),
    )
    define_mesh(
        stage,
        "/World/BlueBoat/VisualsCAD/Hull02_Port",
        hull_data[1][0],
        hull_data[1][1],
        (0.08, 0.08, 0.09),
    )
    define_mesh(
        stage,
        "/World/BlueBoat/VisualsCAD/DeckFrame",
        deck_points,
        deck_faces,
        (0.55, 0.58, 0.62),
    )

    stage.GetRootLayer().Save()
    del full_stl

    print(f"Wrote {OUTPUT_USD}")
    print(
        "Hull triangles:",
        f"{len(hull_data[0][1]):,}",
        f"{len(hull_data[1][1]):,}",
    )
    print(
        f"Deck/frame triangles: {source_deck_triangles:,} source -> "
        f"{len(deck_faces):,} clustered"
    )
    print(f"Deck/frame points: {len(deck_points):,}")
    print(
        f"Deck/frame z-range: {deck_points[:, 2].min():.6f}.."
        f"{deck_points[:, 2].max():.6f} m"
    )
    print(f"Output size: {OUTPUT_USD.stat().st_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
