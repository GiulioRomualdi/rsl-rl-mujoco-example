import random
import numpy as np
from omegaconf import OmegaConf
import torch
import yaml
import gymnasium as gym
from RL.runners import OnPolicyRunner
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from rsl_rl_mujoco.env_wrapper import SB3RslVecEnv
from pathlib import Path
from hydra import main

gym.register(
    id='MFG_MS_V8',
    entry_point='Env.MFG_Musculoskelet_V8.mfgenv.mfg_env:MFG_Musculoskelet_V8',
    max_episode_steps=400
)
@main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg = yaml.safe_load(cfg_yaml)
    # import ipdb;ipdb.set_trace()
    # Create environment
    env_id = cfg["env"]["id"]
    num_envs = cfg["env"].get("num_envs", 4)
    # import ipdb;ipdb.set_trace()
    envs = make_vec_env(env_id, n_envs=num_envs,vec_env_cls=SubprocVecEnv)
    
    # envs = [gym.make(env_id) for _ in range(num_envs)]
    env = SB3RslVecEnv(
        envs,
        clip_actions=cfg["env"].get("clip_actions",None),
        device=cfg.get("device", "cpu"),
    )

    
    runner = OnPolicyRunner(
        env,
        cfg["train"],
        log_dir=cfg.get("log_dir", "./logs"),
        device=cfg.get("device", "cpu"),
    )

    # Train
    runner.learn(num_learning_iterations=cfg["train"]["num_learning_iterations"])
      
if __name__ == "__main__":
    main()