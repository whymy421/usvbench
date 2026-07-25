# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-USV-HarborMission-Direct-v1",
    entry_point=f"{__name__}.harbor_mission_env:HarborMissionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.harbor_mission_env_cfg:HarborMissionEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Staged curriculum (advisor's plan, v11 recipe: kinematic obs + terminal
# bonuses + terminate-on-milestone/contact + gamma 0.999). Champions chain:
# Stage1 -> warm-start Stage2 -> warm-start Stage3.
for _stage in (1, 2, 3):
    gym.register(
        id=f"Isaac-USV-HarborStage{_stage}-Direct-v1",
        entry_point=f"{__name__}.harbor_mission_env:HarborMissionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.harbor_mission_env_cfg:HarborStage{_stage}EnvCfg"
            ),
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_stage_cfg.yaml",
        },
    )
