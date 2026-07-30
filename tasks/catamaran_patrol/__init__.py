# Copyright (c) 2022-2025, USVBench Contributors.
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Catamaran-Patrol-Direct-v1",
    entry_point=f"{__name__}.catamaran_patrol_env:CatamaranPatrolEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.catamaran_patrol_env_cfg:CatamaranPatrolEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
