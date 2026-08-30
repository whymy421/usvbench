# Copyright (c) 2022-2025, USVBench Contributors.
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-RescueBoat-Direct-v1",
    entry_point=f"{__name__}.rescue_boat_env:RescueBoatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rescue_boat_env_cfg:RescueBoatEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
