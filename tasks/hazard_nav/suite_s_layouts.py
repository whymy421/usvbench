"""Suite S: frozen structural-generalization layouts for Hazard Navigation.

Training (Task A scatter) samples obstacles i.i.d. inside a corridor with a
pairwise inflated-gap floor, so accepted maps never contain long-range
STRUCTURE: no collinear rows, no correlated two-row weaves, no merged
super-obstacles, no walls. Suite S freezes numbered layouts drawn from exactly
such structure classes and asks the SAME task contract (same hull, same
sensors, same reward, same tier gap widths) on geometry families the scatter
prior assigns essentially zero mass. That separates "learned obstacle
avoidance" from "learned this map family" -- unlike BARN-style hidden sets,
which are new draws from the SAME training distribution.

Freeze protocol: a layout is a pure function of ``(salt, class_name, index,
level)`` -- hashed to a seed, generated, audited, then serialized to JSON with
an embedded SHA-256 checksum. The JSON asset is the frozen ground truth; the
generator merely reproduces it. Public instances are indices 0-9 per class;
hidden instances use a private salt held off-repo, so publishing their
checksums first commits to them without revealing them.

Admission reuses this package's guards (endpoint disks, BFS routing at
planning inflation) and adds the lesson the sealed ring taught: every gap is
either honestly passable at the tier width or PHYSICALLY sealed by surface
overlap -- nothing in between -- and a hull-half-beam BFS must not find a
materially shorter route than the planning-inflation BFS (that difference is
exactly how sub-tier leak passages show up).

Only the standard library and NumPy are needed at generation time; importing
``hazard_geometry`` also pulls Torch (unused here) because that module keeps
its analytic sensor math in the same file.

Runtime wiring: the ``HazardSuiteSEnvCfg`` family (hazard_nav_env_cfg.py)
names a class, and ``hazard_nav_env.py`` injects the frozen layouts on reset
through ``load_public_layouts`` / ``rotation_index`` below. The generator
remains standalone; envs only ever read the checksummed JSON assets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    from .hazard_geometry import (
        HALF_BEAM_M,
        HULL_BEAM_M,
        MIN_OBSTACLE_RADIUS_M,
        OBSTACLE_INFLATION_M,
        bfs_geodesic_length,
        difficulty_for_level,
        direct_segment_blocked,
        start_goal_disks_clear,
        wall_segment_cylinders,
    )
except ImportError:  # Direct execution: python tasks/hazard_nav/suite_s_layouts.py
    from hazard_geometry import (
        HALF_BEAM_M,
        HULL_BEAM_M,
        MIN_OBSTACLE_RADIUS_M,
        OBSTACLE_INFLATION_M,
        bfs_geodesic_length,
        difficulty_for_level,
        direct_segment_blocked,
        start_goal_disks_clear,
        wall_segment_cylinders,
    )


SUITE_S_SCHEMA = "usvbench.suite_s.layout"
SUITE_S_SCHEMA_VERSION = 1
SUITE_S_CLASSES: tuple[str, ...] = (
    "single_row",
    "staggered_rows",
    "diagonal_row",
    "clusters",
    "gap_wall",
)
SUITE_S_PUBLIC_INDICES: tuple[int, ...] = tuple(range(10))
SUITE_S_DEFAULT_LEVEL = 2

# Open gaps are built a hair wider than the tier bottleneck so admission never
# hinges on float round-off; the slack is far below anything a policy can use.
OPEN_GAP_SLACK_M = 0.02
# A pair only counts as sealed when the physical surfaces truly overlap. The
# ring shipped with 1.00 m holes in a 0.899 m hull because "sealed" was checked
# at planner inflation; here sealing is a surface-overlap FLOOR, full stop.
SEALED_MIN_OVERLAP_M = 0.05
# Sub-tier pair gaps are admissible only when a third cylinder physically
# covers the opening (wall second-neighbours are the canonical case).
COVERAGE_SAMPLES = 16
COVERAGE_DEPTH_M = 0.05
# Fine-grid seal comparison: a hull-half-beam route materially shorter than
# the planning-inflation route means a passage narrower than the tier leaked.
SEAL_CHECK_CELL_M = 0.25
SEAL_ROUTE_TOLERANCE_M = 2.5
MAX_GENERATION_ATTEMPTS = 200


@dataclass(frozen=True)
class SuiteSArena:
    """Arena conventions shared with the scatter task's local frame."""

    goal_distance_range_m: tuple[float, float] = (24.0, 36.0)
    # Structures stay this far from both endpoints; larger than the 5 m
    # protected disk plus the largest inflated radius with margin to spare.
    structure_margin_m: float = 9.0


DEFAULT_ARENA = SuiteSArena()


@dataclass(frozen=True)
class AdmissionAudit:
    """Measured admission verdict for one candidate layout."""

    obstacle_count: int
    sealed_pair_count: int
    covered_pair_count: int
    min_open_gap_inflated_m: float
    endpoints_clear: bool
    direct_blocked: bool
    geodesic_length_m: float | None
    plan_route_fine_m: float | None
    hull_route_fine_m: float | None
    detour_ratio: float | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class SuiteSLayout:
    """One frozen, admission-audited Suite S instance."""

    class_name: str
    index: int
    level: int
    salt: str
    seed: int
    start: np.ndarray
    goal: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    geodesic_length: float
    direct_blocked: bool
    attempts: int

    @property
    def obstacle_count(self) -> int:
        return int(self.radii.shape[0])


def layout_seed(
    class_name: str, index: int, level: int, salt: str = ""
) -> int:
    """Derive the deterministic RNG seed for ``(salt, class, index, level)``.

    SHA-256 rather than ``hash()``: Python string hashing is salted per
    process, which would silently unfreeze every layout.
    """
    token = (
        f"{SUITE_S_SCHEMA}:v{SUITE_S_SCHEMA_VERSION}:"
        f"{salt}:{class_name}:{int(index)}:{int(level)}"
    )
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _pitch_m(radius_m: float, bottleneck_m: float) -> float:
    """Center-to-center spacing giving an inflated gap of one tier width."""
    return (
        2.0 * radius_m
        + 2.0 * OBSTACLE_INFLATION_M
        + bottleneck_m
        + OPEN_GAP_SLACK_M
    )


def _goal_and_band(
    rng: np.random.Generator, arena: SuiteSArena
) -> tuple[float, np.ndarray, float, float]:
    distance = float(rng.uniform(*arena.goal_distance_range_m))
    goal = np.array((distance, 0.0), dtype=np.float64)
    return (
        distance,
        goal,
        arena.structure_margin_m,
        distance - arena.structure_margin_m,
    )


def _build_single_row(
    rng: np.random.Generator, bottleneck_m: float, arena: SuiteSArena
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """One picket line across the route: equal radii, equal tier-width gaps.

    Out-of-distribution because i.i.d. scatter draws are collinear with equal
    spacing with probability ~0: the policy faces several simultaneous,
    equally-priced gaps instead of isolated cylinders to skirt.
    """
    distance, goal, lo, hi = _goal_and_band(rng, arena)
    wall_x = float(np.clip(rng.uniform(0.45, 0.60) * distance, lo, hi))
    radius = float(rng.uniform(1.0, 1.6))
    pitch = _pitch_m(radius, bottleneck_m)
    count = 7
    # The middle picket sits (almost) on the start-goal axis, so the direct
    # segment is blocked by construction, not by luck.
    center_y = float(rng.uniform(-0.5, 0.5))
    offsets = pitch * (np.arange(count, dtype=np.float64) - 0.5 * (count - 1))
    centers = np.stack(
        [np.full(count, wall_x), center_y + offsets], axis=1
    )
    return goal, centers, np.full(count, radius, dtype=np.float64)


def _build_staggered_rows(
    rng: np.random.Generator, bottleneck_m: float, arena: SuiteSArena
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Two brick-staggered rows: every first-row gap is backed by a picket.

    Out-of-distribution because scatter has no cross-depth correlation; here
    the visible gap leads into a blocked pocket, forcing an S-shaped weave.
    The row separation is derived from the pitch so diagonal cross-row pairs
    still clear the tier bottleneck (sep >= sqrt(3)/2 * pitch suffices).
    """
    distance, goal, lo, hi = _goal_and_band(rng, arena)
    radius = float(rng.uniform(1.0, 1.5))
    pitch = _pitch_m(radius, bottleneck_m)
    row_gap = pitch * float(rng.uniform(0.95, 1.10))
    x1 = float(rng.uniform(0.35, 0.45) * distance)
    x1 = float(np.clip(x1, lo, hi - row_gap))
    x2 = x1 + row_gap
    if x2 > hi:
        return None
    base_y = float(rng.uniform(-0.5, 0.5))
    front = np.arange(5, dtype=np.float64) - 2.0
    back = np.arange(4, dtype=np.float64) - 1.5
    centers = np.concatenate(
        [
            np.stack([np.full(5, x1), base_y + pitch * front], axis=1),
            np.stack([np.full(4, x2), base_y + pitch * back], axis=1),
        ]
    )
    return goal, centers, np.full(9, radius, dtype=np.float64)


def _build_diagonal_row(
    rng: np.random.Generator, bottleneck_m: float, arena: SuiteSArena
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """A picket line tilted 25-35 degrees off the route-perpendicular.

    Out-of-distribution because the structure is anisotropic: every gap
    bearing deviates from the goal bearing the same way, so purely local
    avoidance drifts systematically off-route instead of averaging out.
    """
    distance, goal, lo, hi = _goal_and_band(rng, arena)
    mid_x = float(np.clip(rng.uniform(0.45, 0.60) * distance, lo, hi))
    theta = math.radians(float(rng.uniform(25.0, 35.0)))
    if rng.random() < 0.5:
        theta = -theta
    axis = np.array((math.sin(theta), math.cos(theta)), dtype=np.float64)
    radius = float(rng.uniform(1.0, 1.6))
    pitch = _pitch_m(radius, bottleneck_m)
    count = 7
    base = np.array((mid_x, float(rng.uniform(-0.5, 0.5))), dtype=np.float64)
    offsets = pitch * (np.arange(count, dtype=np.float64) - 0.5 * (count - 1))
    centers = base[None, :] + offsets[:, None] * axis[None, :]
    return goal, centers, np.full(count, radius, dtype=np.float64)


def _build_clusters(
    rng: np.random.Generator, bottleneck_m: float, arena: SuiteSArena
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Three lumpy super-obstacles built from overlapping cylinders.

    Out-of-distribution because scatter caps single radii at 2.0 m and floors
    every pairwise gap, so a 6-8 m merged blob with a lobed outline cannot
    occur; the policy must commit early to a channel between blobs. Members
    overlap their anchor by a surface-depth FLOOR (the wall lesson), and each
    placement is checked against ALL placed cylinders: sealed or tier-wide,
    nothing in between.
    """
    distance, goal, lo, hi = _goal_and_band(rng, arena)
    core_x = float(np.clip(rng.uniform(0.45, 0.58) * distance, lo, hi))
    lateral = float(rng.uniform(9.5, 12.0))
    cores = [
        np.array((core_x, float(rng.uniform(-1.0, 1.0)))),
        np.array(
            (
                float(np.clip(core_x + rng.uniform(-3.0, 3.0), lo, hi)),
                lateral + float(rng.uniform(-1.0, 1.0)),
            )
        ),
        np.array(
            (
                float(np.clip(core_x + rng.uniform(-3.0, 3.0), lo, hi)),
                -lateral + float(rng.uniform(-1.0, 1.0)),
            )
        ),
    ]

    placed_centers: list[np.ndarray] = []
    placed_radii: list[float] = []

    def admissible(candidate: np.ndarray, radius: float) -> bool:
        for other, other_radius in zip(placed_centers, placed_radii):
            gap = float(np.linalg.norm(candidate - other)) - radius - other_radius
            if gap <= -SEALED_MIN_OVERLAP_M:
                continue  # genuinely sealed against this cylinder
            if gap - 2.0 * OBSTACLE_INFLATION_M >= bottleneck_m:
                continue  # honestly tier-wide against this cylinder
            return False
        return True

    for core in cores:
        core_radius = float(rng.uniform(1.2, 1.9))
        if not admissible(core, core_radius):
            return None
        cluster_start = len(placed_centers)
        placed_centers.append(core.astype(np.float64))
        placed_radii.append(core_radius)
        for _ in range(int(rng.integers(2, 5))):
            for _ in range(40):
                anchor = int(rng.integers(cluster_start, len(placed_centers)))
                radius = float(rng.uniform(1.2, 1.9))
                overlap = float(rng.uniform(0.4, 0.8))
                bearing = float(rng.uniform(0.0, 2.0 * math.pi))
                reach = placed_radii[anchor] + radius - overlap
                candidate = placed_centers[anchor] + reach * np.array(
                    (math.cos(bearing), math.sin(bearing))
                )
                if admissible(candidate, radius):
                    placed_centers.append(candidate)
                    placed_radii.append(radius)
                    break
            # A member that finds no admissible spot is simply skipped; the
            # audit still judges the finished blob on its own merits.

    return (
        goal,
        np.asarray(placed_centers, dtype=np.float64),
        np.asarray(placed_radii, dtype=np.float64),
    )


def _build_gap_wall(
    rng: np.random.Generator, bottleneck_m: float, arena: SuiteSArena
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """An overlap-sealed wall across the route with one off-axis tier gate.

    Out-of-distribution because scatter never yields an unbroken barrier: the
    policy must find the single gate or pay a huge end-around detour. Unlike
    the forced-crossing basin there is open water past the wall tips, so this
    class measures gate-seeking under alternatives, not gate transit per se.
    Jamb centers sit half a physical gate plus one radius from the gate line,
    so the free width between SURFACES equals the requested gate width.
    """
    distance, goal, lo, hi = _goal_and_band(rng, arena)
    wall_x = float(np.clip(rng.uniform(0.45, 0.60) * distance, lo, hi))
    half_span = float(rng.uniform(9.0, 12.0))
    radius = MIN_OBSTACLE_RADIUS_M
    gate_physical = bottleneck_m + OPEN_GAP_SLACK_M + 2.0 * OBSTACLE_INFLATION_M
    jamb = 0.5 * gate_physical + radius
    gate_limit = half_span - jamb - 2.0  # each side stays a real wall
    if gate_limit < 2.5:
        return None
    gate_y = float(rng.uniform(2.5, gate_limit))
    if rng.random() < 0.5:
        gate_y = -gate_y
    lower = wall_segment_cylinders(
        np.array((wall_x, -half_span)),
        np.array((wall_x, gate_y - jamb)),
        radius_m=radius,
        overlap_m=0.30,
    )
    upper = wall_segment_cylinders(
        np.array((wall_x, gate_y + jamb)),
        np.array((wall_x, half_span)),
        radius_m=radius,
        overlap_m=0.30,
    )
    centers = np.vstack((lower, upper))
    return goal, centers, np.full(len(centers), radius, dtype=np.float64)


_BUILDERS = {
    "single_row": _build_single_row,
    "staggered_rows": _build_staggered_rows,
    "diagonal_row": _build_diagonal_row,
    "clusters": _build_clusters,
    "gap_wall": _build_gap_wall,
}


def _gap_covered(
    first: int,
    second: int,
    centers: np.ndarray,
    radii: np.ndarray,
) -> bool:
    """Whether a third cylinder physically plugs the opening of one pair."""
    direction = centers[second] - centers[first]
    span = float(np.linalg.norm(direction))
    if span <= 0.0:
        return False
    direction = direction / span
    inner_a = centers[first] + radii[first] * direction
    inner_b = centers[second] - radii[second] * direction
    fractions = np.linspace(0.0, 1.0, COVERAGE_SAMPLES)
    points = inner_a[None, :] + fractions[:, None] * (inner_b - inner_a)[None, :]
    others = [
        k for k in range(len(radii)) if k not in (first, second)
    ]
    if not others:
        return False
    distances = np.linalg.norm(
        points[:, None, :] - centers[others][None, :, :], axis=-1
    )
    depths = radii[others][None, :] - distances
    return bool(np.all(np.max(depths, axis=1) >= COVERAGE_DEPTH_M))


def audit_layout(
    start: np.ndarray,
    goal: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    level: int,
) -> AdmissionAudit:
    """Measure one candidate against the Suite S admission contract.

    1. Every pairwise opening is sealed (surface overlap), covered by a third
       cylinder, or at least one tier bottleneck wide after inflation.
    2. Endpoint disks are clear (scatter convention, 5 m).
    3. BFS routes at planning inflation, both at the standard 0.5 m grid and
       at the 0.25 m seal-check grid.
    4. A hull-half-beam BFS finds no materially shorter route -- the measured
       (not assumed) proof that no sub-tier passage leaked in.
    5. The direct start-goal segment is blocked: the structure must actually
       intervene, otherwise the instance measures nothing structural.
    """
    difficulty = difficulty_for_level(level)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    radii = np.asarray(radii, dtype=np.float64)
    failures: list[str] = []

    delta = centers[:, None, :] - centers[None, :, :]
    surface_gap = np.linalg.norm(delta, axis=-1) - radii[:, None] - radii[None, :]
    sealed_pairs = 0
    covered_pairs = 0
    min_open_gap = math.inf
    for first in range(len(radii)):
        for second in range(first + 1, len(radii)):
            gap = float(surface_gap[first, second])
            if gap <= -SEALED_MIN_OVERLAP_M:
                sealed_pairs += 1
                continue
            inflated_gap = gap - 2.0 * OBSTACLE_INFLATION_M
            if inflated_gap + 1.0e-9 >= difficulty.bottleneck_m:
                min_open_gap = min(min_open_gap, inflated_gap)
                continue
            if _gap_covered(first, second, centers, radii):
                covered_pairs += 1
                continue
            failures.append(
                f"pair ({first},{second}) inflated gap {inflated_gap:.3f} m is "
                f"neither sealed nor >= tier {difficulty.bottleneck_m:.3f} m"
            )

    endpoints_clear = start_goal_disks_clear(start, goal, centers, radii)
    if not endpoints_clear:
        failures.append("an inflated obstacle enters a protected endpoint disk")

    geodesic = bfs_geodesic_length(start, goal, centers, radii)
    if geodesic is None:
        failures.append("no BFS route at planning inflation on the 0.5 m grid")

    plan_fine = bfs_geodesic_length(
        start, goal, centers, radii, cell_m=SEAL_CHECK_CELL_M
    )
    hull_fine = bfs_geodesic_length(
        start,
        goal,
        centers,
        radii,
        cell_m=SEAL_CHECK_CELL_M,
        inflation_m=HALF_BEAM_M,
    )
    if plan_fine is None:
        failures.append("no BFS route at planning inflation on the seal grid")
    if hull_fine is None:
        failures.append("no BFS route at hull half-beam on the seal grid")
    if (
        plan_fine is not None
        and hull_fine is not None
        and plan_fine - hull_fine > SEAL_ROUTE_TOLERANCE_M
    ):
        failures.append(
            f"hull route {hull_fine:.2f} m undercuts planning route "
            f"{plan_fine:.2f} m by more than {SEAL_ROUTE_TOLERANCE_M} m: "
            "a sub-tier passage leaked"
        )

    blocked = direct_segment_blocked(start, goal, centers, radii)
    if not blocked:
        failures.append("direct start-goal segment is not blocked")

    straight = float(np.linalg.norm(np.asarray(goal) - np.asarray(start)))
    detour = None if geodesic is None else float(geodesic) / straight
    return AdmissionAudit(
        obstacle_count=int(len(radii)),
        sealed_pair_count=sealed_pairs,
        covered_pair_count=covered_pairs,
        min_open_gap_inflated_m=float(min_open_gap),
        endpoints_clear=bool(endpoints_clear),
        direct_blocked=bool(blocked),
        geodesic_length_m=None if geodesic is None else float(geodesic),
        plan_route_fine_m=None if plan_fine is None else float(plan_fine),
        hull_route_fine_m=None if hull_fine is None else float(hull_fine),
        detour_ratio=detour,
        failures=tuple(failures),
    )


def generate_layout(
    class_name: str,
    index: int,
    *,
    level: int = SUITE_S_DEFAULT_LEVEL,
    salt: str = "",
    arena: SuiteSArena = DEFAULT_ARENA,
    max_attempts: int = MAX_GENERATION_ATTEMPTS,
) -> tuple[SuiteSLayout, AdmissionAudit, int]:
    """Deterministically generate one audited layout.

    Returns ``(layout, audit, candidates_tried)`` where the third value counts
    every raw candidate the audit judged, so callers can report an honest
    admission pass rate. Identical arguments always return identical layouts.
    """
    if class_name not in _BUILDERS:
        valid = ", ".join(sorted(_BUILDERS))
        raise ValueError(f"Unknown Suite S class {class_name!r}; expected one of {valid}.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    difficulty = difficulty_for_level(level)
    builder = _BUILDERS[class_name]
    seed = layout_seed(class_name, index, level, salt)
    rng = np.random.default_rng(seed)
    start = np.zeros(2, dtype=np.float64)

    candidates = 0
    for attempt in range(1, max_attempts + 1):
        built = builder(rng, float(difficulty.bottleneck_m), arena)
        if built is None:
            continue
        goal, centers, radii = built
        candidates += 1
        audit = audit_layout(start, goal, centers, radii, level)
        if not audit.passed:
            continue
        assert audit.geodesic_length_m is not None
        layout = SuiteSLayout(
            class_name=class_name,
            index=int(index),
            level=int(level),
            salt=salt,
            seed=seed,
            start=start,
            goal=goal,
            centers=centers,
            radii=radii,
            geodesic_length=audit.geodesic_length_m,
            direct_blocked=audit.direct_blocked,
            attempts=attempt,
        )
        return layout, audit, candidates

    raise RuntimeError(
        f"Could not generate an admissible {class_name} layout for "
        f"index {index} level {level} in {max_attempts} attempts."
    )


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_checksum(payload: dict) -> str:
    """SHA-256 over the canonical JSON of everything except the checksum."""
    body = {key: value for key, value in payload.items() if key != "checksum_sha256"}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def layout_to_payload(layout: SuiteSLayout, audit: AdmissionAudit) -> dict:
    """Serialize a layout plus its audit into a checksummed JSON payload."""
    payload = {
        "schema": SUITE_S_SCHEMA,
        "schema_version": SUITE_S_SCHEMA_VERSION,
        "class_name": layout.class_name,
        "index": layout.index,
        "level": layout.level,
        "salt": layout.salt,
        "seed": layout.seed,
        "start": [float(v) for v in layout.start],
        "goal": [float(v) for v in layout.goal],
        "centers": [[float(x), float(y)] for x, y in layout.centers],
        "radii": [float(r) for r in layout.radii],
        "geodesic_length_m": float(layout.geodesic_length),
        "direct_blocked": bool(layout.direct_blocked),
        "attempts": int(layout.attempts),
        "audit": {
            "obstacle_count": audit.obstacle_count,
            "sealed_pair_count": audit.sealed_pair_count,
            "covered_pair_count": audit.covered_pair_count,
            "min_open_gap_inflated_m": (
                None
                if math.isinf(audit.min_open_gap_inflated_m)
                else float(audit.min_open_gap_inflated_m)
            ),
            "endpoints_clear": audit.endpoints_clear,
            "plan_route_fine_m": audit.plan_route_fine_m,
            "hull_route_fine_m": audit.hull_route_fine_m,
            "detour_ratio": audit.detour_ratio,
            "hull_beam_m": float(HULL_BEAM_M),
            "obstacle_inflation_m": float(OBSTACLE_INFLATION_M),
        },
    }
    payload["checksum_sha256"] = compute_checksum(payload)
    return payload


def save_layout(layout: SuiteSLayout, audit: AdmissionAudit, path: Path) -> dict:
    payload = layout_to_payload(layout, audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def load_layout(path: Path) -> dict:
    """Load a frozen layout, refusing silently corrupted or edited assets."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SUITE_S_SCHEMA:
        raise ValueError(f"{path} is not a Suite S layout asset")
    expected = payload.get("checksum_sha256")
    actual = compute_checksum(payload)
    if expected != actual:
        raise ValueError(
            f"checksum mismatch for {path}: stored {expected}, recomputed {actual}"
        )
    return payload


def payload_arrays(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(start, goal, centers, radii)`` as float64 arrays."""
    return (
        np.asarray(payload["start"], dtype=np.float64),
        np.asarray(payload["goal"], dtype=np.float64),
        np.asarray(payload["centers"], dtype=np.float64).reshape(-1, 2),
        np.asarray(payload["radii"], dtype=np.float64),
    )


def load_public_layouts(
    class_name: str,
    *,
    level: int = SUITE_S_DEFAULT_LEVEL,
    asset_dir: str | Path = "",
) -> tuple[SuiteSLayout, ...]:
    """Load one class's ten frozen public layouts, checksum-verified.

    This is the runtime injection path used by ``hazard_nav_env.py``. The
    JSON assets are the ground truth -- ``load_layout`` refuses a corrupted
    or hand-edited file via its SHA-256 check -- and the generator is never
    re-run at reset time. The returned tuple is ordered by public index, so
    ``layouts[i]`` is exactly index ``i``. ``asset_dir`` empty resolves to
    the in-repo ``assets/suite_s`` next to this package, which is where the
    frozen assets are committed; a non-empty value overrides it.
    """
    if class_name not in SUITE_S_CLASSES:
        valid = ", ".join(SUITE_S_CLASSES)
        raise ValueError(
            f"Unknown Suite S class {class_name!r}; expected one of {valid}."
        )
    # Resolution order: explicit asset_dir > USVBENCH_ASSETS env (the deploy
    # boxes run this module from the IsaacLab synced copy, where a
    # module-relative path points at a directory that does not exist -- the
    # first Suite S exam died exactly there) > module-relative (local repo).
    if str(asset_dir):
        root = Path(asset_dir)
    else:
        env_root = os.environ.get("USVBENCH_ASSETS", "")
        candidate = Path(env_root) / "suite_s" if env_root else None
        if candidate is not None and candidate.is_dir():
            root = candidate
        else:
            root = Path(__file__).resolve().parents[2] / "assets" / "suite_s"
    layouts: list[SuiteSLayout] = []
    for index in SUITE_S_PUBLIC_INDICES:
        path = root / f"suite_s_{class_name}_L{int(level)}_{index:02d}.json"
        payload = load_layout(path)
        if (
            payload.get("class_name") != class_name
            or int(payload.get("index", -1)) != int(index)
            or int(payload.get("level", -1)) != int(level)
        ):
            raise ValueError(
                f"{path} names ({payload.get('class_name')!r}, "
                f"{payload.get('index')}, L{payload.get('level')}), expected "
                f"({class_name!r}, {index}, L{level})"
            )
        start, goal, centers, radii = payload_arrays(payload)
        layouts.append(
            SuiteSLayout(
                class_name=str(payload["class_name"]),
                index=int(payload["index"]),
                level=int(payload["level"]),
                salt=str(payload["salt"]),
                seed=int(payload["seed"]),
                start=start,
                goal=goal,
                centers=centers,
                radii=radii,
                geodesic_length=float(payload["geodesic_length_m"]),
                direct_blocked=bool(payload["direct_blocked"]),
                attempts=int(payload["attempts"]),
            )
        )
    return tuple(layouts)


def rotation_index(
    env_index: int,
    episode_counter: int,
    *,
    rotation: bool = True,
    layout_count: int = len(SUITE_S_PUBLIC_INDICES),
) -> int:
    """Which frozen layout one env's next episode uses.

    With rotation on, env ``i``'s episode ``k`` uses layout ``(i + k) %
    layout_count``: ten consecutive episodes of any single env cover all ten
    public layouts, and a 128-episode certification at the evaluator's 64
    envs (two episodes per env) lands 12-14 episodes on every layout. With
    rotation off each env is pinned to ``i % layout_count``. Either way the
    index is a pure function of ``(env index, per-env episode count)`` --
    independent of other envs and of policy-dependent reset timing -- so two
    checkpoints certified at the same env count see identical layout
    assignments episode-for-episode, which is stronger pairing than the
    scatter stream's shared-RNG draw order provides.
    """
    if layout_count < 1:
        raise ValueError("layout_count must be at least one")
    if rotation:
        return (int(env_index) + int(episode_counter)) % int(layout_count)
    return int(env_index) % int(layout_count)


def _assert_printable(path: Path) -> None:
    """Refuse control characters (except LF/CR) in a written asset."""
    data = path.read_bytes()
    bad = sorted({byte for byte in data if byte < 32 and byte not in (10, 13)})
    if bad:
        raise ValueError(f"{path} contains control bytes {bad}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, default=SUITE_S_DEFAULT_LEVEL)
    parser.add_argument("--salt", type=str, default="")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "assets" / "suite_s",
    )
    args = parser.parse_args()

    print(
        f"Suite S generation: classes={len(SUITE_S_CLASSES)} "
        f"indices={list(SUITE_S_PUBLIC_INDICES)} level={args.level} "
        f"out={args.out}"
    )
    manifest_rows: list[str] = []
    for class_name in SUITE_S_CLASSES:
        accepted = 0
        candidates = 0
        for index in SUITE_S_PUBLIC_INDICES:
            layout, audit, tried = generate_layout(
                class_name, index, level=args.level, salt=args.salt
            )
            candidates += tried
            accepted += 1
            name = f"suite_s_{class_name}_L{args.level}_{index:02d}.json"
            payload = save_layout(layout, audit, args.out / name)
            _assert_printable(args.out / name)

            # The asset on disk must round-trip and re-verify.
            reloaded = load_layout(args.out / name)
            assert reloaded["checksum_sha256"] == payload["checksum_sha256"]

            # Regeneration from the same key must be bit-identical.
            again, again_audit, _ = generate_layout(
                class_name, index, level=args.level, salt=args.salt
            )
            assert (
                layout_to_payload(again, again_audit)["checksum_sha256"]
                == payload["checksum_sha256"]
            ), f"{name} is not deterministic"

            detour = audit.detour_ratio if audit.detour_ratio is not None else float("nan")
            manifest_rows.append(
                f"{class_name:15s} {index:02d}  n={audit.obstacle_count:2d}  "
                f"geo={layout.geodesic_length:6.2f} m  detour={detour:4.2f}  "
                f"{payload['checksum_sha256'][:12]}"
            )
        rate = accepted / candidates if candidates else 0.0
        print(
            f"  {class_name:15s} accepted {accepted}/{len(SUITE_S_PUBLIC_INDICES)}  "
            f"audit pass rate {accepted}/{candidates} ({100.0 * rate:.0f}%)"
        )

    print("\nMANIFEST (class index n geodesic detour checksum12)")
    for row in manifest_rows:
        print("  " + row)
    print(f"\nPASS: {len(manifest_rows)} frozen layouts generated, verified, deterministic")


__all__ = [
    "AdmissionAudit",
    "DEFAULT_ARENA",
    "SUITE_S_CLASSES",
    "SUITE_S_DEFAULT_LEVEL",
    "SUITE_S_PUBLIC_INDICES",
    "SUITE_S_SCHEMA",
    "SUITE_S_SCHEMA_VERSION",
    "SuiteSArena",
    "SuiteSLayout",
    "audit_layout",
    "compute_checksum",
    "generate_layout",
    "layout_seed",
    "layout_to_payload",
    "load_layout",
    "load_public_layouts",
    "payload_arrays",
    "rotation_index",
    "save_layout",
]


if __name__ == "__main__":
    main()
