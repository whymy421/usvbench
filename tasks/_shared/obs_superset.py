"""Frozen cross-task policy-observation layouts for USVBench.

This module deliberately depends on neither Isaac Lab nor Gymnasium so that
checkpoint adapters and schema tests can import it in a plain Python process.
The v1 index assignments 0..50 are immutable.  New contracts may only claim
channels after index 50; they must never reorder or reinterpret v1 channels.
"""

from __future__ import annotations

from typing import NamedTuple, TypeVar


# The v1 dimension stays public and unchanged for frozen v1 callers.
SUPERSET_DIM = 51
SUPERSET_DIM_V2 = 64

# Frozen v1 layout plus append-only v2 blocks.  Slices use an exclusive stop,
# matching NumPy/Torch.
SLICES = {
    "nav": slice(0, 3),
    "stage_norm": slice(3, 4),
    "lookahead": slice(4, 7),
    "dock": slice(7, 10),
    "rays": slice(10, 46),
    "phase": slice(46, 50),
    "dwell_fraction": slice(50, 51),
    # hazard_nav_env.py:_body_motion_observation normalises surge and sway by
    # SPEED_SCALE_MPS (2 m/s) and yaw rate by yaw_rate_obs_scale_rad_s
    # (1 rad/s in every current cfg), then clamps each channel to [-1, 1].
    "kinematics": slice(51, 54),
    # Per-task channels with no cross-task meaning.  Every use must have a
    # comment in the task environment documenting its native order/semantics.
    "task_specific": slice(54, 58),
    # Must remain zero until a future observation-contract version claims it.
    "reserved": slice(58, 64),
}

# Docking defines speed_norm using a 2 m/s scale.  Reusing that scale is what
# makes the superset speed and surge/sway channels semantically comparable.
SPEED_SCALE_MPS = 2.0

_NAV = [0, 1, 2]
_PATH = [0, 1, 2, 3, 4, 5, 6]
_DOCK = [0, 1, 2, 7, 8, 9]
_RAYS = list(range(10, 46))
_KINEMATICS = list(range(51, 54))
_HAZARD = _NAV + _RAYS
# B1 (v2): + reached latch (slot 3, cross-task "stage complete" semantics) and
# self speed (slot 9, docking's 2 m/s scale). Append-only: v1 untouched.
_HAZARD_V2 = [0, 1, 2, 3, 9] + _RAYS
_HAZARD_V3 = _NAV + _KINEMATICS + _RAYS
_PATH_HAZARD = _PATH + _RAYS
_PATH_HAZARD_KIN = _PATH_HAZARD + _KINEMATICS
_HARBOR = _NAV + _RAYS + list(range(46, 50)) + [7, 8, 50]
_HARBOR_KIN = _NAV + _KINEMATICS + _RAYS + list(range(46, 50)) + [7, 8, 50]
_DOCK_KIN = _DOCK + _KINEMATICS
_DOCK_WALL = _DOCK + _RAYS
_DOCK_WALL_KIN = _DOCK + _KINEMATICS + _RAYS
_STATION_KIN = _NAV + _KINEMATICS
# station_keeping_env.py documents these task-only channels as sea-direction
# dot/cross followed by normalised significant wave height; kinematics follow.
_STATION_WAVE = _NAV + [54, 55, 56] + _KINEMATICS


class UnsupportedNativeLayout(NamedTuple):
    """A registered native observation that cannot be scattered without loss."""

    native_dim: int
    reason: str


_POOLED_HAZARD_UNSUPPORTED = UnsupportedNativeLayout(
    native_dim=15,
    reason=(
        "the native observation contains 9 feasibility-pooled sectors rather "
        "than the frozen 36 individual ray channels; v2 has only four "
        "task-specific slots and reserved slots may not be claimed"
    ),
)


# Every currently registered Gym id is listed explicitly.  Vehicle/current
# variants share their task family's native policy-observation order.  The two
# pooled hazard ids carry an explicit unsupported marker instead of a false
# mapping to the semantically different ray block.
NATIVE_LAYOUTS: dict[str, list[int] | UnsupportedNativeLayout] = {
    # Frozen v1 registrations.
    "Isaac-My-First-Task-Calm-Direct-v1": list(_NAV),
    "Isaac-My-First-Task-Calm-Boat-Direct-v1": list(_NAV),
    "Isaac-USV-BlueBoat-Calm-Direct-v1": list(_NAV),
    "Isaac-USV-StationKeep-Direct-v1": list(_NAV),
    "Isaac-USV-StationKeep-BlueBoat-Direct-v1": list(_NAV),
    "Isaac-USV-StationKeep-BlueBoat-Current-Direct-v1": list(_NAV),
    "Isaac-USV-StationKeep-Boat-Direct-v1": list(_NAV),
    "Isaac-USV-PathFollow-Direct-v1": list(_PATH),
    "Isaac-USV-PathFollow-BlueBoat-Direct-v1": list(_PATH),
    "Isaac-USV-Dock-Direct-v1": list(_DOCK),
    "Isaac-USV-Dock-BlueBoat-Direct-v1": list(_DOCK),
    "Isaac-USV-Dock-BlueBoat-Current-Direct-v1": list(_DOCK),
    "Isaac-USV-HazardNav-Direct-v1": list(_HAZARD),
    "Isaac-USV-HazardNav-Direct-v2": list(_HAZARD_V2),
    "Isaac-USV-PathHazard-Direct-v1": list(_PATH_HAZARD),
    "Isaac-USV-HarborMission-Direct-v1": list(_HARBOR),
    # Hazard v3 lineage. v5/v10 are deliberately unsupported pooled layouts.
    "Isaac-USV-HazardNav-Direct-v3": list(_HAZARD_V3),
    "Isaac-USV-HazardRing-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardNav-Direct-v4": list(_HAZARD_V3),
    "Isaac-USV-HazardNav-Direct-v5": _POOLED_HAZARD_UNSUPPORTED,
    "Isaac-USV-HazardNav-Direct-v6": list(_HAZARD_V3),
    "Isaac-USV-HazardNav-Direct-v7": list(_HAZARD_V3),
    "Isaac-USV-HazardNav-Direct-v8": list(_HAZARD_V3),
    "Isaac-USV-HazardRingSealed-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardCross-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardNav-Direct-v9": list(_HAZARD_V3),
    "Isaac-USV-HazardNav-Direct-v10": _POOLED_HAZARD_UNSUPPORTED,
    "Isaac-USV-HazardNav-Direct-v12": list(_HAZARD_V3),
    "Isaac-USV-HazardBasin-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardCrossDemo-Direct-v1": list(_HAZARD_V3),
    # Mid-episode obstacle appearance: the observation is UNCHANGED (the
    # appeared cylinder enters through the pre-existing ray returns), so
    # the id carries its certified crossing parent layout verbatim and a
    # certified crossing checkpoint loads zero-shot.
    "Isaac-USV-HazardCrossAppear-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardCrossImb-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardRing2-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardFortress-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardFortress2-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardBandFort-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardBandFortSoft-Direct-v1": list(_HAZARD_V3),
    # Same 42-D slot layout. Caveat carried from the env: the Geo/Way variants
    # re-aim the nav slots at the geodesic route (waypoint bearing/distance)
    # rather than the final goal, so projections keep slot POSITIONS but the
    # nav semantics differ; embed_checkpoint across that boundary is a
    # deliberate act, not a free lunch.
    "Isaac-USV-HazardBandFortGeo-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardBandFortWay-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardBandFortWayTax-Direct-v1": list(_HAZARD_V3),
    # Suite D dynamics carriers and the discount-coupling control pair: all
    # plain forced-crossing observations; only rewards/dynamics differ.
    "Isaac-USV-HazardCrossMass-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardCrossDrag-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardCrossThrust-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardCrossTau-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardPbrsGamma-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-HazardPbrsNoGamma-Direct-v1": list(_HAZARD_V3),
    # Contract-mode observations are already in v2 superset order, including
    # the zero-filled task-specific and reserved slices.
    "Isaac-USV-HazardNavC64-Direct-v1": list(range(SUPERSET_DIM_V2)),
    "Isaac-USV-Iceberg-Direct-v1": list(range(SUPERSET_DIM_V2)),
    # Harbor staged curricula append kinematics immediately after nav.
    "Isaac-USV-HarborStage1-Direct-v1": list(_HARBOR_KIN),
    "Isaac-USV-HarborStage2-Direct-v1": list(_HARBOR_KIN),
    "Isaac-USV-HarborStage3-Direct-v1": list(_HARBOR_KIN),
    "Isaac-USV-HarborStage2Warm-Direct-v1": list(_HARBOR_KIN),
    "Isaac-USV-HarborStage2AllRew-Direct-v1": list(_HARBOR_KIN),
    "Isaac-USV-HarborMissionKin-Direct-v1": list(_HARBOR_KIN),
    # Dock phase in isolation: same 49-D staged contract, only the spawn moves.
    "Isaac-USV-HarborDockPhase-Direct-v1": list(_HARBOR_KIN),
    "Isaac-USV-PathHazard-Direct-v2": list(_PATH_HAZARD_KIN),
    # Kinematic and solid-wall docking registrations.
    "Isaac-USV-Dock-BlueBoat-Kin-Direct-v1": list(_DOCK_KIN),
    "Isaac-USV-Dock-BlueBoat-Current-Kin-Direct-v1": list(_DOCK_KIN),
    "Isaac-USV-DockWall-BlueBoat-Direct-v1": list(_DOCK_WALL),
    "Isaac-USV-DockWall-BlueBoat-Kin-Direct-v1": list(_DOCK_WALL_KIN),
    # Kinematic and sea-state station-keeping registrations.
    "Isaac-USV-StationKeep-BlueBoat-Kin-Direct-v1": list(_STATION_KIN),
    "Isaac-USV-StationKeep-BlueBoat-Current-Kin-Direct-v1": list(_STATION_KIN),
    "Isaac-USV-StationKeep-BlueBoat-Wave-Direct-v1": list(_STATION_WAVE),
    # PENDING gym registration (tasks/hazard_nav/PENDING_REGISTRATIONS.md):
    # ramping-current station variant; same 3-D nav observation as the other
    # non-Kin station ids. The drift tests stay green meanwhile because
    # docking's broad f-string pattern shadows orphans of this id shape.
    "Isaac-USV-StationKeep-BlueBoat-RampCurrent-Direct-v1": list(_NAV),
    # Suite S structural-generalization ids: plain v3 hazard observations on
    # frozen layouts; only the reset-time layout source differs.
    "Isaac-USV-SuiteS-SingleRow-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-SuiteS-StaggeredRows-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-SuiteS-DiagonalRow-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-SuiteS-Clusters-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-SuiteS-GapWall-Direct-v1": list(_HAZARD_V3),
    # Wave background variants: the sea is a pure force disturbance (no
    # observation change), so each id carries its certified parent's native
    # layout verbatim and every certified checkpoint loads zero-shot.
    "Isaac-USV-HazardCross-Wave-Direct-v1": list(_HAZARD_V3),
    "Isaac-USV-Iceberg-Wave-Direct-v1": list(range(SUPERSET_DIM_V2)),
    "Isaac-USV-PathFollow-BlueBoat-Wave-Direct-v1": list(_PATH),
    "Isaac-USV-PathHazard-Wave-Direct-v1": list(_PATH_HAZARD),
    "Isaac-USV-Dock-BlueBoat-Wave-Direct-v1": list(_DOCK),
    "Isaac-USV-HarborMission-Wave-Direct-v1": list(_HARBOR),
}


_Array = TypeVar("_Array")


def _supported_layout(gym_id: str) -> list[int]:
    try:
        layout = NATIVE_LAYOUTS[gym_id]
    except KeyError as exc:
        known = ", ".join(sorted(NATIVE_LAYOUTS))
        raise KeyError(f"Unknown USVBench Gym id {gym_id!r}; known ids: {known}") from exc
    if isinstance(layout, UnsupportedNativeLayout):
        raise ValueError(
            f"USVBench Gym id {gym_id!r} is explicitly unsupported by the "
            f"cross-task observation bridge: {layout.reason}"
        )
    return layout


def extract_native(superset_vec: _Array, gym_id: str) -> _Array:
    """Extract ``gym_id``'s native order from a v1 (51-D) or v2 (64-D) value."""

    indices = _supported_layout(gym_id)
    shape = getattr(superset_vec, "shape", None)
    final_dim = shape[-1] if shape is not None and len(shape) else None
    if final_dim not in (SUPERSET_DIM, SUPERSET_DIM_V2):
        raise ValueError(
            f"Expected a NumPy/Torch value with final dimension {SUPERSET_DIM} "
            f"or {SUPERSET_DIM_V2}, got shape {shape!r}"
        )
    if indices and max(indices) >= final_dim:
        raise ValueError(
            f"Gym id {gym_id!r} uses v2 channel {max(indices)}, but the supplied "
            f"value has only {final_dim} channels"
        )
    return superset_vec[..., indices]


def native_to_superset(
    vec: _Array, gym_id: str, dim: int = SUPERSET_DIM_V2
) -> _Array:
    """Scatter a native observation into its frozen superset channel slots."""

    indices = _supported_layout(gym_id)
    shape = getattr(vec, "shape", None)
    if shape is None or len(shape) == 0 or shape[-1] != len(indices):
        raise ValueError(
            f"Expected {gym_id!r} native final dimension {len(indices)}, "
            f"got shape {shape!r}"
        )
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive integer, got {dim!r}")
    if indices and max(indices) >= dim:
        raise ValueError(
            f"Superset dimension {dim} cannot hold channel {max(indices)} "
            f"required by {gym_id!r}"
        )

    if hasattr(vec, "new_zeros"):  # Torch tensor without importing Torch.
        out = vec.new_zeros((*shape[:-1], dim))
    else:
        import numpy as np

        out = np.zeros((*shape[:-1], dim), dtype=getattr(vec, "dtype", None))
    out[..., indices] = vec
    return out


__all__ = [
    "NATIVE_LAYOUTS",
    "SLICES",
    "SPEED_SCALE_MPS",
    "SUPERSET_DIM",
    "SUPERSET_DIM_V2",
    "UnsupportedNativeLayout",
    "extract_native",
    "native_to_superset",
]
