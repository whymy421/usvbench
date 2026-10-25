"""Journal vehicle record: what an environment needs to SPAWN and SIMULATE a v2 vessel.

Pure Python at import time (no torch, no simulator). Every number comes from
the D5 registry (tasks/_shared/vehicles_v2.py + vehicles_v2_data.py, controller
decision 1); the hydrodynamic profile is built by hydro_profiles.py through
the registry adapters. This module holds NO vessel numbers.

Scope (E2): physics and spawn only -- asset path and hash, frame, mass
properties, calm COM height, windage, the profile's own top speed, the
hydro_v2 profile and the pending-decision flags. Task-contract quantities
(cruise / reference speed, design turning radius, the footprint used for
scoring, validity bands, deadlines) are deliberately NOT here: they live in
D4's journal_cfg_v2.py (task E5), which is also derived from the registry.

Legacy registry names are rejected (see LEGACY_NAME_REASONS): they are not
the journal vessels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

try:  # package import inside tasks/_shared
    from . import vehicles_v2 as _registry
except ImportError:  # path-loaded CPU tests; never import the tasks package
    import vehicles_v2 as _registry


ASSETS_DIRNAME = "assets_v2"   # repository-root folder holding the v2 USD assets

# Legacy tasks/_shared/vehicles.py entries and why the journal rejects them.
LEGACY_NAME_REASONS = MappingProxyType({
    "blueboat": (
        "legacy BlueBoat entry (tasks/_shared/vehicles.py:118-141) spawns blueboat_physics.usd, whose "
        "authored bow is at -X (source axes bow -X, starboard +Y: D5 PARAMS.md frame canonicalization, "
        "blueboat_build_usd.py:3-5,53) while the legacy entry declares bow_body_axis='+x' (vehicles.py:139) "
        "and the legacy envs drive the hull along +X; it also keeps the legacy box hydrostatics and the "
        "explicit 470 N m s/rad roll/pitch damping. Use 'blueboat_v2' (asset rotated to bow +X, origin at "
        "the COM)."),
    "wamv": (
        "legacy 'wamv' entry (tasks/_shared/vehicles.py:94-117) is a placeholder: boat_physics.usdc Froude-"
        "scaled with lambda=0.8 (vehicles.py:116), 100 kg (vehicles.py:98), bow -x (vehicles.py:115); it is "
        "not the VRX WAM-V. Use 'wamv_vrx_v2_opt1' (VRX competition assembly at 1:1, the default) or "
        "another wamv_vrx_v2_* profile."),
    "rov": (
        "legacy 'rov' entry (tasks/_shared/vehicles.py:70-93) is the BlueROV2 articulation, not a surface "
        "vessel; the journal has exactly two vessels (BlueBoat and the VRX WAM-V)."),
})

# Physics defaults of a vessel with their owner-decision status (CONTEXT_R3
# binding decisions; R2 decisions 1 and 6). Values are the defaults this code
# implements; each record lists the ones that apply (field ``pending``, name
# kept for compatibility). G28: entries the owner has decided are prefixed
# "DECIDED (...)"; unprefixed entries are still open (NC 39 lists them).
PENDING_COMMON = (
    "DECIDED (owner 2026-09-30 evening, 1:1 scale): scale 1:1 physical units (R2 decision 1, default)",
    "per-motor thrust allocation with physical limits, radial saturation (R2 decision 6; R3 decision 4)",
    "thrust-line pitch coupling OFF, mixer_mount_z_m = 0 (R3 decision 6)",
    "nominal motor lag 0 s (R3 decision 6)",
    "rotational damping relative to the wave-surface normal rate (R3 decision 3)",
    "horizontal Froude-Krylov force ON with C_a = 0 (R3 decision 7)",
)
PENDING_BLUEBOAT = (
    "DECIDED (owner 2026-10-01 00:0x, A1): roll/pitch damping: journal default zeta = 0.1 per axis; legacy "
    "470 N m s/rad selectable (R3 decision 2)",
    "legacy 17.26 kg mass (historical batteries) vs current 16.9 kg; reverse thrust by the M200 ratio; "
    "yaw request scale arm*(fwd+rev) (D5 FINAL_REPORT owner decision 8)",
)
PENDING_WAMV = (
    "DECIDED (owner 2026-10-01 00:0x, A4): WAM-V profile option 1 = VRX competition assembly with VRX-classic "
    "damping and thrust (R3 decision 5)",
    "DECIDED (owner decision T1.35b, adopted for v2.6 as J1): WAM-V roll/pitch damping: journal default zeta = 0.1 "
    "per axis (the VRX values stay selectable as 'registry') (R3 decision 2)",
    "DECIDED (owner 2026-10-01 10:1x, A5 cross-flow): WAM-V option 1 sway/yaw damping: journal default = "
    "VRX-classic linear terms (40 N s/m, 400 N m s/rad) plus "
    "Sarda 2016 cross-flow sway and strip-theory yaw quadratic terms (controller decision G4-A5 from the G1 "
    "sweep; hydro_profiles swayyaw_damping); VRX classic selectable as the comparison profile "
    "(swayyaw_damping='vrx_classic'); classic per-motor thrust kept (D5 FINAL_REPORT owner decision 4)",
)


def v2_spec_or_raise(name):
    """Registry lookup that explains why a legacy name is rejected."""
    if not isinstance(name, str):
        raise TypeError("vehicle name must be a string, got %r" % (type(name).__name__,))
    if name in LEGACY_NAME_REASONS:
        raise ValueError("vehicle %r is not a journal (v2) vessel: %s" % (name, LEGACY_NAME_REASONS[name]))
    return _registry.get_vehicle_v2(name)


def _frozen_matrix(matrix):
    return tuple(tuple(float(v) for v in row) for row in matrix)


@dataclass(frozen=True)
class JournalVehicle:
    """Immutable spawn/physics record of one v2 vessel (all SI, FLU, origin = COM)."""

    name: str
    registry_schema_version: int
    asset_relpath: str                 # relative to the repository's assets_v2/ folder
    asset_sha256: str
    bow_axis: str                      # always "+x" for v2 assets
    mass_kg: float
    inertia_com_kg_m2: tuple           # full 3x3 tensor about the COM, actor axes
    com_offset_m: tuple                # actor origin -> COM, always (0, 0, 0)
    calm_com_height_above_water_m: float
    windage_kwargs: Mapping            # C3 wind.VesselWindage(**windage_kwargs)
    windage_reference_height_m: float  # lateral centroid above water (U10 conversion height)
    windage_roughness_length_m: float  # neutral log-profile z0 (R3 decision 9)
    top_speed_model_m_s: float         # terminal surge speed of the profile's own laws
    hydro_profile: object              # hydro_profiles.VesselProfileV2 (built via the registry adapters)
    profile_options: Mapping           # options used to build hydro_profile
    profile_status: str
    pending: tuple = field(default_factory=tuple)

    def asset_path(self, assets_dir) -> Path:
        return Path(assets_dir) / self.asset_relpath

    def verify_asset(self, assets_dir) -> Path:
        """Fail closed unless assets_dir/asset_relpath exists with the registry SHA-256.

        Returns the verified path. Raises FileNotFoundError when the file is
        missing and ValueError on any hash mismatch; never returns a path it
        did not hash.
        """
        path = self.asset_path(assets_dir)
        if not path.is_file():
            raise FileNotFoundError("journal asset %s for %s is missing" % (path, self.name))
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != self.asset_sha256:
            raise ValueError("journal asset %s SHA-256 %s differs from the registry value %s for %s"
                             % (path, digest.hexdigest(), self.asset_sha256, self.name))
        return path


def _terminal_speed(linear, quadratic, force):
    if force <= 0:
        return 0.0
    if quadratic == 0:
        return force / linear
    return (-linear + math.sqrt(linear * linear + 4.0 * quadratic * force)) / (2.0 * quadratic)


def resolve_journal_vehicle(name, **options) -> JournalVehicle:
    """Build the journal record for a v2 registry name.

    ``options`` are passed to hydro_profiles.profile_for_vehicle
    (rollpitch_damping, rotational_relative_motion, pitch_coupling,
    froude_krylov_horizontal, swayyaw_damping); defaults implement the
    controller decisions.
    Legacy names are rejected with the reason (LEGACY_NAME_REASONS).
    """
    spec = v2_spec_or_raise(name)
    _registry.validate_vehicle(spec)
    try:
        from .hydro_profiles import profile_for_vehicle
    except ImportError:
        from hydro_profiles import profile_for_vehicle
    profile = profile_for_vehicle(name, **options)
    h, t = profile.hydro, profile.thrusters
    top = _terminal_speed(h.linear_damping[0], h.quadratic_damping[0], sum(t.forward_limit_n))
    if spec.bow_body_axis != "+x" or spec.com_actor_m != (0.0, 0.0, 0.0):
        raise ValueError("v2 record requires a bow +x asset with its origin at the COM")
    pending = PENDING_COMMON + (PENDING_BLUEBOAT if spec.option == "blueboat" else PENDING_WAMV)
    return JournalVehicle(
        name=spec.name, registry_schema_version=spec.schema_version,
        asset_relpath=spec.asset_relpath, asset_sha256=spec.asset_sha256,
        bow_axis=spec.bow_body_axis, mass_kg=spec.mass_kg,
        inertia_com_kg_m2=_frozen_matrix(spec.inertia_com_kg_m2),
        com_offset_m=tuple(float(v) for v in spec.com_actor_m),
        calm_com_height_above_water_m=spec.hydrostatics.com_height_above_water_m,
        windage_kwargs=MappingProxyType(dict(_registry.to_c3_windage_kwargs(spec))),
        windage_reference_height_m=spec.windage.lateral_centroid_height_above_water_m,
        windage_roughness_length_m=spec.windage.roughness_length_m,
        top_speed_model_m_s=top,
        hydro_profile=profile,
        profile_options=MappingProxyType(dict(profile.sources.get("options", {}))),
        profile_status=profile.status,
        pending=pending)


def envelope_spec(vehicle):
    """Registry VesselSpecV2 carrying the journal vehicle's own sway/yaw damping (G4-A5).

    The D7 vessel envelope (conditions/dynamic/vessels.envelope_from_spec_v2)
    takes a VesselSpecV2 and reads its yaw damping for the steady-spin check;
    this is the registry record with the sway and yaw entries of
    ``vehicle.hydro_profile`` (hydro_profiles.swayyaw_profile_spec), so the
    envelope, the env physics and the MPPI model use one rule.
    """
    try:
        from .hydro_profiles import swayyaw_profile_spec
    except ImportError:
        from hydro_profiles import swayyaw_profile_spec
    if not isinstance(vehicle, JournalVehicle):
        raise TypeError("envelope_spec needs a JournalVehicle (resolve_journal_vehicle)")
    return swayyaw_profile_spec(vehicle.hydro_profile)


def verify_asset(vehicle, assets_dir) -> Path:
    """Module-level form of JournalVehicle.verify_asset (fail closed).

    ``vehicle`` is a JournalVehicle or a v2 registry name (legacy names are
    rejected with their reason). Only the registry's asset path and SHA-256
    are needed, so a name is resolved without building the hydro profile.
    """
    if isinstance(vehicle, JournalVehicle):
        return vehicle.verify_asset(assets_dir)
    spec = v2_spec_or_raise(vehicle)
    path = Path(assets_dir) / spec.asset_relpath
    if not path.is_file():
        raise FileNotFoundError("journal asset %s for %s is missing" % (path, spec.name))
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != spec.asset_sha256:
        raise ValueError("journal asset %s SHA-256 %s differs from the registry value %s for %s"
                         % (path, digest.hexdigest(), spec.asset_sha256, spec.name))
    return path


__all__ = ["ASSETS_DIRNAME", "LEGACY_NAME_REASONS", "JournalVehicle", "resolve_journal_vehicle",
           "v2_spec_or_raise", "verify_asset", "envelope_spec"]
