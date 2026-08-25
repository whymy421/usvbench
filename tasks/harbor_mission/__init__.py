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

gym.register(
    id="Isaac-USV-HarborStage2Warm-Direct-v1",
    entry_point=f"{__name__}.harbor_mission_env:HarborMissionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.harbor_mission_env_cfg:HarborStage2EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_stage_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HarborStage2AllRew-Direct-v1",
    entry_point=f"{__name__}.harbor_mission_env:HarborMissionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.harbor_mission_env_cfg:HarborStage2AllRewEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_stage_cfg.yaml",
    },
)

# Dock phase in isolation: spawns at the field exit, which is where a phase-2
# hand-off actually starts. Append-only -- every HarborStage/HarborMission
# number keeps its meaning because spawn_phase defaults to 0 everywhere else.
gym.register(
    id="Isaac-USV-HarborDockPhase-Direct-v1",
    entry_point=f"{__name__}.harbor_mission_env:HarborMissionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.harbor_mission_env_cfg:HarborDockPhaseEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_stage_cfg.yaml",
    },
)

# 49-D contract parity so bridged policies keep their velocity channels on the full chain.
gym.register(
    id="Isaac-USV-HarborMissionKin-Direct-v1",
    entry_point=f"{__name__}.harbor_mission_env:HarborMissionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.harbor_mission_env_cfg:HarborMissionKinEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
