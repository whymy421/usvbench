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

# Ring siege: same contract as v3, encircled spawn with one tier-width gap.
gym.register(
    id="Isaac-USV-HazardRing-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardRingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# v11 recipe + policy-invariant potential shaping. Append-only: v3 and its
# certified numbers are untouched, so the pair is a clean single-variable test.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v4",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavV3PbrsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardNav-Direct-v5",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavV3FeasEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardNav-Direct-v6",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavV3ThreadEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardNav-Direct-v7",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardNavV3PbrsTermEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardNav-Direct-v8",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardNavV3PbrsShiftEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardRingSealed-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardRingSealedEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Forced crossing: the replacement for Task A's detour hole. Registered as a
# new id rather than a fix to v3, so every certified v1-v8 number keeps
# meaning exactly what it meant when it was measured.
gym.register(
    id="Isaac-USV-HazardCross-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardForcedCrossingEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # Baseline-matrix off-policy arms. Isaac Lab's train.py only accepts
        # PPO-family algorithms, so scripts/sac_train.py drives these via
        # --cfg-entry-point; the env itself is byte-identical across arms.
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_td3_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
        # v2: fixes the SAC entropy bomb (min_log_std -20 -> -5). a = that fix
        # alone; b = fix + reward scale 0.01 + gamma 0.99. v1 keys stay so the
        # failed runs remain reproducible exactly as they were trained.
        "skrl_sac_v2a_cfg_entry_point": f"{agents.__name__}:skrl_sac_v2a_cfg.yaml",
        "skrl_sac_v2b_cfg_entry_point": f"{agents.__name__}:skrl_sac_v2b_cfg.yaml",
        # v3: the measured fixes. SAC = tanh-bounded mean (log-prob bomb
        # defused at the source) + sigma floor + scaled returns; TD3 = tanh
        # actor mean, budget raised at launch (undertraining diagnosis).
        "skrl_sac_v3_cfg_entry_point": f"{agents.__name__}:skrl_sac_v3_cfg.yaml",
        "skrl_td3_v3_cfg_entry_point": f"{agents.__name__}:skrl_td3_v3_cfg.yaml",
        # v4: the code-level fix for the same defect v3 patches from the yaml.
        # Generated from the v3 yaml by swapping two strings in the policy
        # block (GaussianMixin -> SquashedGaussianMixin, tanh(ACTIONS) ->
        # ACTIONS); net, critics, memory and the entire agent block are
        # character-identical, so v3-vs-v4 isolates the policy
        # parameterisation. v3's tanh output only bounds the MEAN -- samples
        # are still clipped and the log-prob still omits the change-of-
        # variables Jacobian; v4 squashes inside the model and corrects it.
        # Needs scripts/sac_train.py's SquashedRunner: skrl's Runner._component
        # is a closed whitelist, so yaml alone cannot select the class.
        "skrl_sac_v4_cfg_entry_point": f"{agents.__name__}:skrl_sac_v4_cfg.yaml",
    },
)

# BandFortWay + loiter tax: the measured fix for the fortress training
# collapse (zero-cost absorbing region outside the walls). New id as always;
# the taxless bfway keeps meaning exactly what it meant.
gym.register(
    id="Isaac-USV-HazardBandFortWayTax-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardBandFortWayTaxEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Open-water tax: the advisor's mechanism, both parameters measured
# (scripts/measure_open_water.py). New id -- v1..v8 keep their exact rewards.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v9",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardOpenWaterTaxEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # Second task of the baseline matrix: same off-policy configs as the
        # crossing arm, so an algorithm difference cannot be a config artefact.
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_td3_cfg_entry_point": f"{agents.__name__}:skrl_td3_cfg.yaml",
    },
)

# v10 = v9 tax + v5 pooling. Owner-approved combination, new id as always.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v10",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardTaxPoolEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Deliberately skip gym id v11: "v11" is the project's colloquial name for
# the historical champion recipe, so registering that number would collide
# with established jargon.
gym.register(
    id="Isaac-USV-HazardNav-Direct-v12",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardSoftLedgerEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Control for the crossing zero-shot: same basin, no bulkhead.
gym.register(
    id="Isaac-USV-HazardBasin-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardOpenBasinEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Same task with the walls drawn. Recording only -- never train or certify on
# this id, so a visual change can never move a reported number.
gym.register(
    id="Isaac-USV-HazardCrossDemo-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossDemoEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Mid-episode obstacle appearance (sudden terrain change): the certified
# crossing exam, plus ONE extra cylinder that flips active at a per-episode
# protocol-drawn time, ray-aligned and distance-clamped so it is guaranteed
# visible the instant it exists (fairness derivation in
# hazard_geometry.plan_obstacle_appearance). Append-only as always: the
# crossing id, its layouts, champions and certificates are untouched. The
# dose axis is appearance distance -- rungs 20/15/12/9.5 m via
# eval --set appear_distance_m, never per-rung ids.
gym.register(
    id="Isaac-USV-HazardCrossAppear-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossAppearEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Suite D: forced crossing with the frozen per-episode training pack.
gym.register(
    id="Isaac-USV-HazardCrossImb-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossImbalanceEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Owner-approved harder siege: two sealed rings with misaligned exits.
gym.register(
    id="Isaac-USV-HazardRing2-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardDoubleRingEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Fortress geometry makes the tier gap topologically mandatory without changing rewards.
gym.register(
    id="Isaac-USV-HazardFortress-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardFortressEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardFortress2-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardFortress2EnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Constructive bands enforce the tier aperture while retaining the canonical ledger.
gym.register(
    id="Isaac-USV-HazardBandFort-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardBandFortEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Soft contact keeps episodes alive while quadratic dwell prices prolonged scraping.
gym.register(
    id="Isaac-USV-HazardBandFortSoft-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardBandFortSoftEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Geodesic progress pays only inward route completion through the fortress gaps.
gym.register(
    id="Isaac-USV-HazardBandFortGeo-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardBandFortGeoEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# First contract-mode id, subject of the smoke acceptance.
gym.register(
    id="Isaac-USV-HazardNavC64-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardNavC64EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Detour-capability twin of the threading family; first new task born on contract v2.
gym.register(
    id="Isaac-USV-Iceberg-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hazard_nav_env_cfg:HazardIcebergEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Route waypoints couple the fortress policy's bearing channels to geodesic progress.
gym.register(
    id="Isaac-USV-HazardBandFortWay-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardBandFortWayEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Suite D v3 dynamics carriers. Append-only: each samples exactly one frozen
# train pack per episode on the same forced-crossing task.
gym.register(
    id="Isaac-USV-HazardCrossMass-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossMassEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardCrossDrag-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossDragEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardCrossThrust-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossThrustEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardCrossTau-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardCrossTauEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Discount-coupling control: identical carrier configs except pbrs_correct.
gym.register(
    id="Isaac-USV-HazardPbrsGamma-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardPbrsGammaEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-HazardPbrsNoGamma-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardPbrsNoGammaEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Suite S: structural generalization on frozen, checksum-verified layouts.
# One id per structure class, all on the certified v3 observation/reward
# contract, so crossing / v9 / v12 champions load and run here zero-shot.
# Append-only as always: the scatter ids and their certificates are untouched.
gym.register(
    id="Isaac-USV-SuiteS-SingleRow-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardSuiteSEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-SuiteS-StaggeredRows-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardSuiteSStaggeredRowsEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-SuiteS-DiagonalRow-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardSuiteSDiagonalRowEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-SuiteS-Clusters-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardSuiteSClustersEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-USV-SuiteS-GapWall-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardSuiteSGapWallEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# Wave background variants (owner's order: every task family gets one).
# BACKGROUND FORCE ONLY -- unlike the station-keeping Wave id, these add no
# sea-state observation channels: the native layout, reward, and termination
# are byte-identical to the certified parent id, so a certified checkpoint
# loads zero-shot and the id is a pure environmental stress axis.
# Registered append-only; NATIVE_LAYOUTS maps each id to its parent's layout.
gym.register(
    id="Isaac-USV-HazardCross-Wave-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardForcedCrossingWaveEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-USV-Iceberg-Wave-Direct-v1",
    entry_point=f"{__name__}.hazard_nav_env:HazardNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.hazard_nav_env_cfg:HazardIcebergWaveEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
