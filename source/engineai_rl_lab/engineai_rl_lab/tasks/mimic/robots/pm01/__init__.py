import gymnasium as gym

gym.register(
    id="EngineAI-PM01-Mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pm01_minic:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.pm01_minic:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"engineai_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
