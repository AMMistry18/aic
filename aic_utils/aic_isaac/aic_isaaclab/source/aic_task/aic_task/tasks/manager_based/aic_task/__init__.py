# Replace the existing __init__.py at:
#   aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/__init__.py
# Adds AIC-Insert-SC-v0, AIC-Insert-SC2-v0, AIC-Insert-SFP-v0 while keeping
# AIC-Task-v0 available.

import gymnasium as gym

from . import agents

gym.register(
    id="AIC-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_task_env_cfg:AICTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-SC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_env_cfg:AICInsertSCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsertTeacherPPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-SC2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_env_cfg:AICInsertSC2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsertTeacherPPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-SFP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_env_cfg:AICInsertSFPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsertTeacherPPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-Hybrid-SC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_hybrid_env_cfg:AICInsertHybridSCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-Hybrid-SC-Debug-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_hybrid_env_cfg:AICInsertHybridSCDebugEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-Hybrid-SC-DepthDebug-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_hybrid_env_cfg:AICInsertHybridSCDepthDebugEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HybridBootstrapPPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-Hybrid-SC-StrictDebug-v0",
    entry_point=f"{__name__}.kinematic_last_mile_env:KinematicLastMileRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_hybrid_env_cfg:AICInsertHybridSCStrictDebugEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HybridBootstrapPPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-Hybrid-SC-RotDebug-v0",
    entry_point=f"{__name__}.kinematic_last_mile_env:KinematicLastMileRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_hybrid_env_cfg:AICInsertHybridSCRotDebugEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HybridBootstrapPPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-Hybrid-SC-StrictRotDebug-v0",
    entry_point=f"{__name__}.kinematic_last_mile_env:KinematicLastMileRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_hybrid_env_cfg:AICInsertHybridSCStrictRotDebugEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HybridBootstrapPPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-Hybrid-SC2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_hybrid_env_cfg:AICInsertHybridSC2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="AIC-Insert-Hybrid-SFP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aic_insert_hybrid_env_cfg:AICInsertHybridSFPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
