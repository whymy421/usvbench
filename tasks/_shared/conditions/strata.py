"""Scenario ordinals and balanced direction strata for the hydro_v2 layer (E3).

Scenario ordinal. An evaluation grid of ``grid_width`` environments, each
running its own episode sequence 0, 1, 2, ... (per-env quota; C5), is
enumerated as ``ordinal = episode_index * grid_width + env_index``. The
ordinal depends only on the scenario identity (env, episode) and the declared
grid width, never on reset order, controller, dose or timing, so every dose
and every controller of a cell sees the same ordinal for the same scenario.
Consecutive ordinals are consecutive integers, so any complete block of
``K`` ordinals contains each residue mod ``K`` exactly once.

Direction strata (set ``compass8_v1``, PENDING owner confirmation). Eight
relative flow-to angles, measured counter-clockwise from the task reference
axis (the nominal direction of travel supplied by the task at reset):

    index  angle   name                          disturbance comes FROM
    0        0     following                     astern
    1       45     stern_quartering_starboard    starboard quarter
    2       90     beam_starboard                starboard beam
    3      135     bow_quartering_starboard      starboard bow
    4      180     head                          ahead
    5      225     bow_quartering_port           port bow
    6      270     beam_port                     port beam
    7      315     stern_quartering_port         port quarter

(flow-to +90 deg = towards port = coming from starboard.) C7 STRATEGY.md:175
asks for balanced head, beam and following directions; the brief adds
quartering. The compass-8 set contains all four classes with port/starboard
mirror balance; the class shares are head 1/8, bow-quartering 2/8, beam 2/8,
stern-quartering 2/8, following 1/8 (the uniform-circle measure).

Assignment: ``stratum = (ordinal + rotation(seed)) mod 8`` with a seed-keyed
rotation in [0, 8). On a 64-env grid every episode round holds each stratum
exactly 8 times. All directional suites (waves, current, wind, perception
bias) use the SAME stratum for the same scenario, so combined presets are
co-directed (D9 DOSE_DESIGN.md section 9) and single-suite cells share the
strata of every other suite.
"""
from __future__ import annotations

import hashlib
import math

import torch

STRATA_SET = "compass8_v1"
STRATA = (
    ("following", 0.0),
    ("stern_quartering_starboard", 45.0),
    ("beam_starboard", 90.0),
    ("bow_quartering_starboard", 135.0),
    ("head", 180.0),
    ("bow_quartering_port", 225.0),
    ("beam_port", 270.0),
    ("stern_quartering_port", 315.0),
)
STRATA_CLASSES = {
    "following": "following", "head": "head",
    "stern_quartering_starboard": "stern_quartering", "stern_quartering_port": "stern_quartering",
    "bow_quartering_starboard": "bow_quartering", "bow_quartering_port": "bow_quartering",
    "beam_starboard": "beam", "beam_port": "beam",
}
STRATA_COUNT = len(STRATA)
STRATA_RAD = tuple(math.radians(angle) for _, angle in STRATA)


def strata_rotation(seed: int) -> int:
    """Seed-keyed rotation of the stratum sequence, in [0, STRATA_COUNT)."""
    digest = hashlib.blake2b(f"usvbench.strata.{STRATA_SET}|{int(seed)}".encode("ascii"),
                             digest_size=8).digest()
    return int.from_bytes(digest, "little") % STRATA_COUNT


def scenario_ordinals(env_ids, episode_indices, grid_width: int) -> torch.Tensor:
    """int64 CPU ordinals ``episode * grid_width + env``."""
    ids = torch.as_tensor(env_ids, dtype=torch.long).detach().cpu().reshape(-1)
    episodes = torch.as_tensor(episode_indices, dtype=torch.long).detach().cpu().reshape(-1)
    if ids.numel() != episodes.numel():
        raise ValueError("one episode index per env is required")
    if int(grid_width) <= 0:
        raise ValueError("grid width must be positive")
    if bool((episodes < 0).any()) or bool((ids < 0).any()) or bool((ids >= int(grid_width)).any()):
        raise ValueError("ordinals need started episodes and env ids inside the grid")
    return episodes * int(grid_width) + ids


def stratum_indices(ordinals, seed: int) -> torch.Tensor:
    ordinals = torch.as_tensor(ordinals, dtype=torch.long).reshape(-1)
    return (ordinals + strata_rotation(seed)) % STRATA_COUNT


def relative_angles(indices) -> torch.Tensor:
    """float64 relative flow-to angles (rad) of stratum indices."""
    table = torch.tensor(STRATA_RAD, dtype=torch.float64)
    return table[torch.as_tensor(indices, dtype=torch.long).reshape(-1)]


def stratum_name(index: int) -> str:
    return STRATA[int(index)][0]


def describe() -> dict:
    return {"set": STRATA_SET, "count": STRATA_COUNT,
            "angles_deg": [angle for _, angle in STRATA],
            "names": [name for name, _ in STRATA],
            "classes": dict(STRATA_CLASSES),
            "convention": ("relative FLOW-TO angle, counter-clockwise from the task reference "
                           "axis; world azimuth = axis + angle"),
            "assignment": "stratum = (episode * grid_width + env + rotation(seed)) mod 8"}


__all__ = ["STRATA_SET", "STRATA", "STRATA_CLASSES", "STRATA_COUNT", "STRATA_RAD",
           "strata_rotation", "scenario_ordinals", "stratum_indices", "relative_angles",
           "stratum_name", "describe"]
