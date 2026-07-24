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
