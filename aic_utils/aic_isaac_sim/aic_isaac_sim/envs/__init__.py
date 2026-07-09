"""Gym registration for the GPU-vectorized AIC insertion task."""

import gymnasium as gym

from .. import agents


gym.register(
    id="AIC-LastInch-SFP-Direct-v0",
    entry_point=f"{__name__}.last_inch_env:AICLastInchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.last_inch_env:AICLastInchEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

__all__ = ["AICLastInchEnv", "AICLastInchEnvCfg"]
