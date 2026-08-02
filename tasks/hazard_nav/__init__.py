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
