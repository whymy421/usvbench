# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-path scenario draws for the hazard_nav / harbor_mission families.

WHY A SECOND DRAWS MODULE
-------------------------
``tasks/_shared/scenario_draws.py`` owns the call sites that used to run
against the GLOBAL torch RNG (station keeping, path following, path hazard's
spawn heading).  This file owns a different defect in a different family: the
draws here were never on the global RNG at all.  They came from ONE numpy
``Generator`` per env instance -- ``tasks/hazard_nav/hazard_nav_env.py:118``
and ``tasks/harbor_mission/harbor_mission_env.py:65``, both
``np.random.default_rng(layout_seed)`` with ``layout_seed`` taken from
``cfg.seed`` -- advanced once per reset for whatever batch of envs happened to
be resetting.

That is seed-derived and still not controller-independent, because these two
families terminate episodes EARLY: hazard_nav ends on goal contact or on hull
contact (``hazard_nav_env.py:1452-1456``, ``contact_terminates`` defaults True
at ``hazard_nav_env_cfg.py:232``), and harbor_mission ends on a milestone or a
contact (``harbor_mission_env.py:1133-1141``).  Two controllers therefore reach
reset k at different times, in different groupings, and consume a different
number of rejection-sampler draws, so the same ``(eval seed, env, episode)``
slot is handed a DIFFERENT layout under each controller.  Measured on the
crossing task at one evaluation seed, over three controllers certified for 128
episodes each and compared on the two purely-layout fields the certificate
carries (``route_geodesic_m`` and ``d0_m``): the layouts agreed on only
53.8% / 56.9% / 55.6% of shared ``(env, episode)`` slots, and the multiset of
layouts overlapped on 105 / 114 / 110 of 128.  McNemar and the paired bootstrap
were being run on unpaired data.

Every helper below takes the key-addressed stream instead: a fresh
``np.random.Generator`` for ``(protocol_version, eval_seed, env_index, per-env
episode_index, group)`` from ``ScenarioRNG.numpy_rng``, which is the same door
``tasks/path_hazard/path_hazard_env.py:913-931`` walked through in round 2.
Because the KEY -- not the number of draws taken so far -- decides the stream,
the scenario for a given ``(eval seed, env, episode)`` is the same whether the
previous episode ran to the horizon or died on step three, and the same whether
this env reset alone or with sixty-three others.

WHAT IS AND IS NOT PRESERVED
----------------------------
Preserved: every distribution, every range, every sampler, every attempt
budget, and the ORDER of the calls inside one reset.  The only thing that
changes is the ``Generator`` object handed to the sampler, so "only the source
of the random numbers changed" is checkable by eye against the before/after
lines quoted in each docstring.

Not preserved: the specific numbers.  That is the point; see above.

A "Before (file:NNN)::" block below quotes the PRE-migration source and names
the line it stood on then, which is the convention ``scenario_draws.py`` set
for the round-1 families (its own "Before" citations are likewise
archaeological).  Every citation describing the code as it stands NOW names a
current line.

UNSEEDED FALLBACK
-----------------
``make_scenario_rng`` returns ``None`` when ``cfg.seed is None``
(``scenario_draws.py:112-114``), and every helper here treats ``scenario=None``
as "call the historical line on the env's own ``self._layout_rng``, byte for
byte" -- including the BATCHED numpy calls, which are re-issued in their
original batched form rather than as a per-env loop.

Be aware what that fallback actually is in these two families, because it is
not the same fallback the other families have.  ``hazard_nav_env.py:116-118``
and ``harbor_mission_env.py:63-65`` read::

    isaac_seed = getattr(cfg, "seed", None)
    layout_seed = cfg.layout_seed if isaac_seed is None else int(isaac_seed)
    self._layout_rng = np.random.default_rng(layout_seed)

and ``layout_seed`` defaults to the CONSTANT 0 (``hazard_nav_env_cfg.py:187``,
``harbor_mission_env_cfg.py:181``).  So an unseeded run does not draw fresh
entropy -- it replays one fixed layout sequence every time.  This module does
not change that: an unseeded env keeps the exact behaviour it had, constant
seed included.  Every evaluator in ``scripts/`` sets ``env_cfg.seed`` (e.g.
``scripts/eval_v6_frozen.py:104``), so certification never takes this path.

GROUPS
------
Each primitive owns its own stream, which is property (d) of the protocol
(``scenario_rng.py:53-55``): adding or removing one primitive's draw cannot
shift another primitive's numbers.  The five actuator/dynamics knobs are five
SEPARATE groups rather than five ordered draws off one ``actuator`` stream on
purpose -- ``scripts/eval_imbalance.py`` exists to sweep them, and with a
shared stream, enabling ``drag_scale_choices`` would silently re-draw
``mass_scale`` as well and confound exactly the attribution such a sweep is
run to make.  Group names are hashed as TEXT, not as a positional index
(``scenario_rng.py:56-63``), so introducing these names cannot disturb any
stream that already exists.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# Loaded both as a package member (by the envs) and as a bare top-level module
# (by the CPU test, which cannot import the package without pulling isaaclab),
# so both import spellings have to work -- the idiom scenario_draws.py:71-84
# already uses.
try:
    from .scenario_draws import GROUP_SPAWN
    from .scenario_rng import ScenarioRNG
except ImportError:  # direct execution
    from scenario_draws import GROUP_SPAWN
    from scenario_rng import ScenarioRNG


# The layout/route geometry: start, goal, obstacle field, gates, berth.  Same
# name path_hazard chose for the same primitive (path_hazard_env.py:45), which
# is deliberate -- it is the same kind of object, drawn by rejection from one
# numpy Generator -- and harmless, because streams are keyed by (eval seed, env,
# episode, group) and two different tasks never share a certificate.
GROUP_LAYOUT = "layout"

# The global rotation hazard_nav applies to an accepted layout
# (hazard_nav_env.py:1808-1816).  Kept off GROUP_LAYOUT so the angle does not
# depend on how many draws the layout sampler happened to burn -- which matters
# in practice, because the Suite S branch consumes NO layout draws at all
# (hazard_nav_env.py:1790-1795) while every other layout_mode consumes many.
GROUP_ROTATION = "rotation"

# Appended 2026-08-26 for the mid-episode obstacle appearance axis (sudden
# terrain change; fairness derivation and planner at
# tasks/hazard_nav/hazard_geometry.py:plan_obstacle_appearance).  Its own
# group for the usual reason -- property (d): the appearance time and the
# azimuth-candidate order may not shift the layout, the rotation or any
# actuator knob, and vice versa.  The stream is consumed in a fixed order per
# episode -- first the appearance time t0 at reset, then one ray permutation
# per extra cylinder at the moment of appearance -- so the GENERATOR has to
# stay alive from reset until t0, which is why ``appearance_stream`` below
# returns the generator itself rather than a finished value.  Group names are
# hashed as TEXT (scenario_rng.py:56-63), so appending this name perturbs no
# stream that already exists.
GROUP_APPEARANCE = "appearance"

# One group per actuator/dynamics knob; see GROUPS in the module docstring.
GROUP_THRUST_IMBALANCE = "actuator_thrust_imbalance"
GROUP_MASS_SCALE = "actuator_mass_scale"
GROUP_DRAG_SCALE = "actuator_drag_scale"
GROUP_THRUST_CAP_SCALE = "actuator_thrust_cap_scale"
GROUP_MOTOR_TAU_S = "actuator_motor_tau_s"
# Appended 2026-08-26 for the deck-payload axis (point mass strapped on deck;
# model and distinguishability argument at the payload_* fields in
# tasks/hazard_nav/hazard_nav_env_cfg.py). Group names are hashed as TEXT
# (scenario_rng.py:56-63), so appending this name perturbs no stream that
# already exists.
GROUP_PAYLOAD_MASS_KG = "actuator_payload_mass_kg"

ACTUATOR_GROUPS = (
    GROUP_THRUST_IMBALANCE,
    GROUP_MASS_SCALE,
    GROUP_DRAG_SCALE,
    GROUP_THRUST_CAP_SCALE,
    GROUP_MOTOR_TAU_S,
    GROUP_PAYLOAD_MASS_KG,
)

TWO_PI = 2.0 * math.pi


def episode_layout_rng(
    scenario: ScenarioRNG | None,
    fallback_rng: np.random.Generator,
    env_index: int,
    group: str = GROUP_LAYOUT,
) -> np.random.Generator:
    """The numpy stream this env's CURRENT episode layout is drawn from.

    Seeded env: a fresh ``np.random.Generator`` keyed by (protocol version,
    cfg.seed, env index, this env's episode counter, ``group``).  The env's
    ``_reset_idx`` calls ``ScenarioRNG.reset_idx`` before this, so the counter
    already names the episode about to start.

    Unseeded env (``scenario is None``): the historical single advancing
    generator, byte for byte.

    Callers that need SEVERAL draws from one group must call this ONCE and draw
    them all from the returned object.  Calling it twice for the same (env,
    group) returns two generators on the same key, i.e. the same numbers -- two
    perfectly correlated "independent" draws.
    """
    if scenario is None:
        return fallback_rng
    return scenario.numpy_rng(group, int(env_index))


def actuator_choice_indices(
    scenario: ScenarioRNG | None,
    fallback_rng: np.random.Generator,
    group: str,
    env_indices: Sequence[int],
    choice_count: int,
) -> np.ndarray:
    """One uniform index into a ``*_choices`` tuple per resetting env.

    Before (tasks/hazard_nav/hazard_nav_env.py:1504-1506, and four more
    knobs in the same shape at :1514, :1522, :1530, :1540)::

        choice_indices = self._layout_rng.integers(
            len(self.cfg.thrust_imbalance_choices), size=num_resets
        )

    After: one draw per env off that env's own stream for THIS knob.  The
    distribution is untouched -- ``Generator.integers(n)`` and
    ``Generator.integers(n, size=k)`` are the same uniform on ``[0, n)`` and, on
    one generator, produce the same values element for element -- so only the
    source changed.  The batched call is re-issued verbatim on the unseeded
    path, which is why the fallback stays bit-identical.
    """
    count = len(env_indices)
    if int(choice_count) < 1:
        raise ValueError(f"choice_count must be positive, got {choice_count}")
    if scenario is None:
        return np.asarray(fallback_rng.integers(int(choice_count), size=count))
    return np.asarray(
        [
            int(
                episode_layout_rng(scenario, fallback_rng, index, group)
                .integers(int(choice_count))
            )
            for index in env_indices
        ],
        dtype=np.int64,
    )


def layout_rotation_angle(
    scenario: ScenarioRNG | None,
    fallback_rng: np.random.Generator,
    env_index: int,
) -> float:
    """Global layout rotation for one env, uniform on ``[0, 2*pi)``.

    Before (tasks/hazard_nav/hazard_nav_env.py:1676)::

        goal_angle = float(self._layout_rng.uniform(0.0, 2.0 * math.pi))

    After: the same scalar draw, off the ``rotation`` stream of this
    (env, episode).  ``2.0 * math.pi`` is the same constant the original line
    multiplied by, spelled ``TWO_PI`` here.
    """
    rng = episode_layout_rng(scenario, fallback_rng, env_index, GROUP_ROTATION)
    return float(rng.uniform(0.0, TWO_PI))


def spawn_jitter_offsets(
    scenario: ScenarioRNG | None,
    fallback_rng: np.random.Generator,
    env_indices: Sequence[int],
    jitter_m: float,
) -> np.ndarray:
    """Uniform-on-a-disc spawn offsets, ``(len(env_indices), 2)`` float32.

    Before (tasks/harbor_mission/harbor_mission_env.py:1231-1236)::

        offsets = np.zeros((num_resets, 2), dtype=np.float32)
        if jitter_m > 0.0:
            angles = self._layout_rng.uniform(0.0, 2.0 * math.pi, num_resets)
            radii_j = jitter_m * np.sqrt(
                self._layout_rng.uniform(0.0, 1.0, num_resets)
            )
            offsets[:, 0] = radii_j * np.cos(angles)
            offsets[:, 1] = radii_j * np.sin(angles)

    After: per env, the angle and then the radius off that env's ``spawn_pose``
    stream, in the source order -- both from ONE generator, so the two stay
    independent draws rather than two copies of the same key.  The ``sqrt``
    keeps the sample uniform over the disc, and ``jitter_m <= 0`` still draws
    nothing at all, which is what keeps every config with the default
    ``spawn_phase_jitter_m = 0.0`` (harbor_mission_env_cfg.py:188) consuming
    zero randomness here.
    """
    count = len(env_indices)
    offsets = np.zeros((count, 2), dtype=np.float32)
    if not jitter_m > 0.0:
        return offsets
    if scenario is None:
        angles = fallback_rng.uniform(0.0, TWO_PI, count)
        radii = float(jitter_m) * np.sqrt(fallback_rng.uniform(0.0, 1.0, count))
    else:
        angles = np.empty(count, dtype=np.float64)
        radii = np.empty(count, dtype=np.float64)
        for row, env_index in enumerate(env_indices):
            rng = episode_layout_rng(
                scenario, fallback_rng, env_index, GROUP_SPAWN
            )
            angles[row] = rng.uniform(0.0, TWO_PI)
            radii[row] = float(jitter_m) * math.sqrt(rng.uniform(0.0, 1.0))
    offsets[:, 0] = radii * np.cos(angles)
    offsets[:, 1] = radii * np.sin(angles)
    return offsets


def appearance_stream(
    scenario: ScenarioRNG | None,
    fallback_rng: np.random.Generator,
    env_index: int,
) -> np.random.Generator:
    """The per-(env, episode) stream for the mid-episode appearance primitive.

    Seeded env: the keyed ``appearance`` stream of this (env, episode) --
    identical algebra to ``episode_layout_rng``, different group.  The caller
    draws the appearance time t0 from the returned generator at reset and then
    KEEPS the object, because the azimuth-candidate permutation is drawn from
    the same stream mid-episode; calling this helper again for the same (env,
    episode) would restart the stream and replay the t0 draw.

    Unseeded env (``scenario is None``): a PRIVATE child generator seeded by
    one ``integers`` draw off the env's historical fallback.  This
    deliberately differs from ``episode_layout_rng``'s return-the-fallback
    convention: the appearance stream is consumed MID-EPISODE, and handing
    back the shared advancing generator would let one env's appearance timing
    reorder every other env's later reset draws.  The certified ids are
    untouched either way -- with ``appear_count == 0`` (every pre-existing
    cfg) the env never calls this helper, so the fallback stream is not
    advanced by even that one child-seed draw.
    """
    if scenario is None:
        return np.random.default_rng(int(fallback_rng.integers(2**63)))
    return scenario.numpy_rng(GROUP_APPEARANCE, int(env_index))


def actuator_scenario_entry(group: str, values: Any) -> dict[str, dict[str, Any]]:
    """One ``stamp_scenario`` group block for a knob that was actually drawn.

    Returns ``{group: {short_name: values}}`` where ``short_name`` is the group
    name with its ``actuator_`` prefix removed, so the certificate reads
    ``scenario_hashes["actuator_mass_scale"]`` over a parameter called
    ``mass_scale``.  A knob whose ``*_choices`` tuple is empty draws nothing and
    contributes NO block, so its hash key is simply absent -- which is honest:
    there is no such primitive in that run.
    """
    name = group[len("actuator_"):] if group.startswith("actuator_") else group
    return {group: {name: values}}


__all__ = [
    "ACTUATOR_GROUPS",
    "GROUP_APPEARANCE",
    "GROUP_DRAG_SCALE",
    "GROUP_LAYOUT",
    "GROUP_MASS_SCALE",
    "GROUP_MOTOR_TAU_S",
    "GROUP_PAYLOAD_MASS_KG",
    "GROUP_ROTATION",
    "GROUP_SPAWN",
    "GROUP_THRUST_CAP_SCALE",
    "GROUP_THRUST_IMBALANCE",
    "TWO_PI",
    "actuator_choice_indices",
    "actuator_scenario_entry",
    "appearance_stream",
    "episode_layout_rng",
    "layout_rotation_angle",
    "spawn_jitter_offsets",
]
