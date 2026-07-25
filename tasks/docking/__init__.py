# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-USV-Dock-Direct-v1",
    entry_point=f"{__name__}.docking_env:DockingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.docking_env_cfg:DockingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-Dock-BlueBoat-Direct-v1",
    entry_point=f"{__name__}.docking_env:DockingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.docking_env_cfg:DockingBlueBoatEnvCfg",
        # Light/agile hull needs lower exploration noise near the hold
        # tolerance (sigma~0.10): -1.9 collapsed into the escape attractor,
        # -2.3 gave a triple-checkpoint 100% golden era.
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_blueboat_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-Dock-BlueBoat-Current-Direct-v1",
    entry_point=f"{__name__}.docking_env:DockingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.docking_env_cfg:DockingBlueBoatCurrentEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_blueboat_cfg.yaml",
    },
)

# C3 x C5 (the fourth double crossing): the berth becomes solid geometry --
# a U-shaped slip the hull must thread and hold inside without contact.
gym.register(
    id="Isaac-USV-DockWall-BlueBoat-Direct-v1",
    entry_point=f"{__name__}.docking_env:DockingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.docking_env_cfg:DockingBlueBoatWallEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_blueboat_cfg.yaml",
    },
)
