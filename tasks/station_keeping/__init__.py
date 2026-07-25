# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-USV-StationKeep-Direct-v1",
    entry_point=f"{__name__}.station_keeping_env:StationKeepingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.station_keeping_env_cfg:StationKeepingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Kinematic-observation variants (v11 block: body-frame surge/sway/yaw rate).
# Append-only: the 3-D ids above and their certified champions are untouched.
gym.register(
    id="Isaac-USV-StationKeep-BlueBoat-Kin-Direct-v1",
    entry_point=f"{__name__}.station_keeping_env:StationKeepingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.station_keeping_env_cfg:StationKeepingBlueBoatKinEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-StationKeep-BlueBoat-Current-Kin-Direct-v1",
    entry_point=f"{__name__}.station_keeping_env:StationKeepingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.station_keeping_env_cfg:"
            "StationKeepingBlueBoatCurrentKinEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-StationKeep-BlueBoat-Direct-v1",
    entry_point=f"{__name__}.station_keeping_env:StationKeepingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.station_keeping_env_cfg:StationKeepingBlueBoatEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-StationKeep-BlueBoat-Current-Direct-v1",
    entry_point=f"{__name__}.station_keeping_env:StationKeepingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.station_keeping_env_cfg:StationKeepingBlueBoatCurrentEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
