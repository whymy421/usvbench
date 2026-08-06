# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-USV-HazardNav-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# B1 wave (append-only): velocity observability + reached latch + swiftness
# term. v1 layouts/champions/certificates stay valid.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v2",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavV2EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# v3 adds body-frame surge, sway, and yaw rate to the v1 goal/ray observation.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v3",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavV3EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Ring siege: same contract as v3, encircled spawn with one tier-width gap.
gym.register(
    id="Isaac-USV-HazardRing-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardRingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# v11 recipe + policy-invariant potential shaping. Append-only: v3 and its
# certified numbers are untouched, so the pair is a clean single-variable test.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v4",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavV3PbrsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardNav-Direct-v5",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavV3FeasEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardNav-Direct-v6",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavV3ThreadEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardNav-Direct-v7",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardNavV3PbrsTermEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardNav-Direct-v8",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardNavV3PbrsShiftEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardRingSealed-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardRingSealedEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Forced crossing: the replacement for Task A's detour hole. Registered as a
# new id rather than a fix to v3, so every certified v1-v8 number keeps
# meaning exactly what it meant when it was measured.
gym.register(
    id="Isaac-USV-HazardCross-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardForcedCrossingEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Open-water tax: the advisor's mechanism, both parameters measured
# (scripts/measure_open_water.py). New id -- v1..v8 keep their exact rewards.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v9",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardOpenWaterTaxEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# v10 = v9 tax + v5 pooling. Owner-approved combination, new id as always.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v10",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardTaxPoolEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Deliberately skip gym id v11: "v11" is the project's colloquial name for
# the historical champion recipe, so registering that number would collide
# with established jargon.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v12",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardSoftLedgerEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Control for the crossing zero-shot: same basin, no bulkhead.
gym.register(
    id="Isaac-USV-HazardBasin-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardOpenBasinEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Same task with the walls drawn. Recording only -- never train or certify on
# this id, so a visual change can never move a reported number.
gym.register(
    id="Isaac-USV-HazardCrossDemo-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossDemoEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Suite D: forced crossing with the frozen per-episode training pack.
gym.register(
    id="Isaac-USV-HazardCrossImb-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossImbalanceEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Owner-approved harder siege: two sealed rings with misaligned exits.
gym.register(
    id="Isaac-USV-HazardRing2-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardDoubleRingEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Fortress geometry makes the tier gap topologically mandatory without changing rewards.
gym.register(
    id="Isaac-USV-HazardFortress-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardFortressEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardFortress2-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardFortress2EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Constructive bands enforce the tier aperture while retaining the canonical ledger.
gym.register(
    id="Isaac-USV-HazardBandFort-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardBandFortEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Soft contact keeps episodes alive while quadratic dwell prices prolonged scraping.
gym.register(
    id="Isaac-USV-HazardBandFortSoft-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardBandFortSoftEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Geodesic progress pays only inward route completion through the fortress gaps.
gym.register(
    id="Isaac-USV-HazardBandFortGeo-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardBandFortGeoEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# First contract-mode id, subject of the smoke acceptance.
gym.register(
    id="Isaac-USV-HazardNavC64-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavC64EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Detour-capability twin of the threading family; first new task born on contract v2.
gym.register(
    id="Isaac-USV-Iceberg-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardIcebergEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Route waypoints couple the fortress policy's bearing channels to geodesic progress.
gym.register(
    id="Isaac-USV-HazardBandFortWay-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardBandFortWayEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Suite D v3 dynamics carriers. Append-only: each samples exactly one frozen
# train pack per episode on the same forced-crossing task.
gym.register(
    id="Isaac-USV-HazardCrossMass-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossMassEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardCrossDrag-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossDragEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardCrossThrust-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossThrustEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardCrossTau-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossTauEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Discount-coupling control: identical carrier configs except pbrs_correct.
gym.register(
    id="Isaac-USV-HazardPbrsGamma-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardPbrsGammaEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardPbrsNoGamma-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardPbrsNoGammaEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
