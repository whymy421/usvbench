"""Named vessel windage profiles for the wind suite (D1, J-base).

Values are data, versioned by WINDAGE_PROFILE_VERSION. Geometry for the
BlueBoat comes from the C3 hull-only silhouette computation
(docs/wind/windage_results.json, waterline case "0.0"; docs/wind/WINDAGE.md
table rows 24 and 211). The BlueBoat superstructure/equipment is missing from
the CAD set C3 used, so the profile is PROVISIONAL. Coefficients
CX0=CY0=0.9, CN0=0.1 are C3's transparent comparison assumptions
(docs/wind/WIND_DESIGN.md "Proposed five-level 1:1 ladder"), not fitted data.

The WAM-V entry is a named PLACEHOLDER owned by task D5 (vessel specs). Its
numbers are None, so enabling wind with it fails at construction. C3's
provisional 1:1 reconstruction (A_F 0.6783940 m2, A_L 2.1235695 m2,
z_bar 0.2981497 m, L 5.011322 m; WINDAGE.md row 26) is recorded only as a
note for D5 to confirm or replace.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

WINDAGE_PROFILE_VERSION = 1

STATUS_PROVISIONAL = "provisional"
STATUS_PLACEHOLDER = "placeholder_pending_D5"


@dataclass(frozen=True)
class WindageProfile:
    """Geometry in metres/m2 above the declared waterline; coefficients 1."""

    name: str
    vehicle: str
    status: str
    frontal_area_m2: float | None
    lateral_area_m2: float | None
    length_m: float | None
    centroid_height_m: float | None
    longitudinal_cp_m: float | None
    c_x0: float | None
    c_y0: float | None
    c_n0: float | None
    source: str

    def usable(self) -> bool:
        return self.status != STATUS_PLACEHOLDER

    def as_dict(self) -> dict:
        return asdict(self)


# BlueBoat bounds x = +/-0.5975813150405884 m (windage_results.json), so the
# silhouette length is 2 * 0.5975813150405884 m.
_BLUEBOAT_LENGTH_M = 2.0 * 0.5975813150405884

WINDAGE_PROFILES = {
    "blueboat_hull_only_c3": WindageProfile(
        name="blueboat_hull_only_c3",
        vehicle="blueboat",
        status=STATUS_PROVISIONAL,
        frontal_area_m2=0.04968478224291942,
        lateral_area_m2=0.16220798892378677,
        length_m=_BLUEBOAT_LENGTH_M,
        centroid_height_m=0.07621178065812223,
        # Metadata only (the CN model supplies yaw): +0.0423105 m from the hull
        # midpoint, +0.0483105 m from the source builder's COM (WINDAGE.md:24).
        longitudinal_cp_m=0.042310528307893794 + 0.006,
        c_x0=0.9,
        c_y0=0.9,
        c_n0=0.1,
        source=("C3 20260930_c3_wind windage_results.json blueboat waterline 0.0; "
                "two bare hulls only, superstructure/equipment missing; "
                "coefficients are C3 comparison assumptions"),
    ),
    "wamv_vrx_placeholder_d5": WindageProfile(
        name="wamv_vrx_placeholder_d5",
        vehicle="wamv_vrx",
        status=STATUS_PLACEHOLDER,
        frontal_area_m2=None,
        lateral_area_m2=None,
        length_m=None,
        centroid_height_m=None,
        longitudinal_cp_m=None,
        c_x0=None,
        c_y0=None,
        c_n0=None,
        source=("PLACEHOLDER for task D5. C3 provisional 1:1 VRX-derived values for "
                "D5 to confirm: A_F 0.6783940 m2, A_L 2.1235695 m2, z_bar 0.2981497 m, "
                "L 5.011322 m, x_cp +0.2854508 m (WINDAGE.md row 26); waterline, "
                "mass and coefficients UNVERIFIED"),
    ),
}

# Default profile per registry vehicle name. The legacy registry "wamv"
# (0.8-scale, 100 kg Froude profile) deliberately has NO windage profile.
PROFILE_FOR_VEHICLE = {
    "blueboat": "blueboat_hull_only_c3",
    "wamv_vrx": "wamv_vrx_placeholder_d5",
}


def profile_name_for_vehicle(vehicle: str) -> str:
    try:
        return PROFILE_FOR_VEHICLE[vehicle]
    except KeyError:
        raise ValueError(
            f"no windage profile is declared for vehicle {vehicle!r}; "
            "add a validated profile before enabling wind"
        ) from None


def get_profile(name: str) -> WindageProfile:
    try:
        return WINDAGE_PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown windage profile {name!r}") from None


def require_usable(profile: WindageProfile, vehicle: str) -> WindageProfile:
    """Reject placeholders and profiles declared for another vessel."""
    if profile.vehicle != vehicle:
        raise ValueError(
            f"windage profile {profile.name!r} is declared for {profile.vehicle!r}, "
            f"not for the task vehicle {vehicle!r}"
        )
    if not profile.usable():
        raise ValueError(
            f"windage profile {profile.name!r} is a placeholder pending task D5; "
            "wind cannot be enabled for this vessel yet"
        )
    for key in ("frontal_area_m2", "lateral_area_m2", "length_m", "centroid_height_m",
                "longitudinal_cp_m", "c_x0", "c_y0", "c_n0"):
        value = getattr(profile, key)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"windage profile {profile.name!r} has no finite {key}")
    return profile


__all__ = [
    "WINDAGE_PROFILE_VERSION", "STATUS_PROVISIONAL", "STATUS_PLACEHOLDER",
    "WindageProfile", "WINDAGE_PROFILES", "PROFILE_FOR_VEHICLE",
    "profile_name_for_vehicle", "get_profile", "require_usable",
]
