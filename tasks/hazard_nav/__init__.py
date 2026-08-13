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

# v4 keeps v11's 42-D observation and adds sustained reverse-motion shaping.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v4",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavV4EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_v12_cfg.yaml",
    },
)

# Wave variants of v3. Same observation and action spaces as v3, so a v3
# checkpoint evaluates on these unchanged and the score gap is the
# wave-robustness measurement. New ids, so v3 itself stays frozen.
gym.register(
    id="Isaac-USV-HazardNav-Airy-Direct-v3",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavV3AiryEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardNav-Jonswap-Direct-v3",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardNavV3JonswapEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
