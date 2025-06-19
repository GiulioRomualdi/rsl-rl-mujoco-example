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

from Env.MFG_Musculoskelet_V9.mfgenv.ReferTraj_V6 import TrajectoryManager

gym.register(
    id='MFG_MS_V9',
    entry_point='Env.MFG_Musculoskelet_V9.mfgenv.mfg_MSenv:MFG_Musculoskeletal_V9',
    max_episode_steps=400
)
with open("Env\MFG_Musculoskelet_V9\cfg\mujocoenv_default.yaml", "r") as f:
    env_self_cfg = yaml.safe_load(f)
tm_args = env_self_cfg["traj_manager"]
traj_manager = TrajectoryManager(
    data_path=tm_args["data_path"],
    repeat_times=int(tm_args["repeat_times"]),
    sample_frequency=float(tm_args["sample_frequency"]),
    knee_1dof=bool(tm_args["knee_1dof"]),
    enable_mirroring=bool(tm_args["enable_mirroring"]),
    smoothing_sigma=(None if tm_args["smoothing_sigma"] is None else float(tm_args["smoothing_sigma"])),
    splice_overlap=int(tm_args["splice_overlap"]),
    speed_range=(float(tm_args["speed_range"][0]), float(tm_args["speed_range"][1])),
    uniform_length=bool(tm_args["uniform_length"]),
    verbose=bool(tm_args["verbose"])
)
env_cfg = { **env_self_cfg, "traj_manager": traj_manager }



@main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg = yaml.safe_load(cfg_yaml)
    # import ipdb;ipdb.set_trace()
    # Create environment
    env_id = cfg["env"]["id"]
    num_envs = cfg["env"].get("num_envs", 4)
    # import ipdb;ipdb.set_trace()
    envs = make_vec_env(env_id, n_envs=num_envs,vec_env_cls=SubprocVecEnv,env_kwargs={"config": env_cfg})
    # import ipdb;ipdb.set_trace()
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