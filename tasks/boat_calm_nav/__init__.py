# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# Local NavRL checkouts often contain older packages that register the generic
# reference id, so this collision-free alias is the canonical launcher target.
gym.register(
    id="Isaac-USVBench-Boat-Calm-Direct-v1",
    entry_point=f"{__name__}.my_first_task_env:MyFirstTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_first_task_env_cfg:MyFirstTaskEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
