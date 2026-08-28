# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-USV-PathHazard-Direct-v1",
    entry_point=f"{__name__}.path_hazard_env:PathHazardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.path_hazard_env_cfg:PathHazardEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# v11 recipe variant (kinematic obs + terminal anchor + outcome termination).
# Append-only: the certified v1 id and its champion are untouched.
gym.register(
    id="Isaac-USV-PathHazard-Direct-v2",
    entry_point=f"{__name__}.path_hazard_env:PathHazardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.path_hazard_env_cfg:PathHazardV2EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_v11_cfg.yaml",
    },
)


# Wave background variants (owner's order: every task family gets one).
# BACKGROUND FORCE ONLY -- unlike the station-keeping Wave id, these add no
# sea-state observation channels: the native layout, reward, and termination
# are byte-identical to the certified parent id, so a certified checkpoint
# loads zero-shot and the id is a pure environmental stress axis.
# Registered append-only; NATIVE_LAYOUTS maps each id to its parent's layout.
gym.register(
    id="Isaac-USV-PathHazard-Wave-Direct-v1",
    entry_point=f"{__name__}.path_hazard_env:PathHazardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.path_hazard_env_cfg:PathHazardWaveEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
