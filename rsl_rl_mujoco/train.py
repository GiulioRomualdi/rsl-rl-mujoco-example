import os
import psutil
import yaml
import gymnasium as gym
from omegaconf import OmegaConf
from multiprocessing.managers import BaseManager
from multiprocessing import freeze_support
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from rsl_rl_mujoco.env_wrapper import SB3RslVecEnv
from RL.runners import OnPolicyRunner
from hydra import main

from Env.MFG_Musculoskelet_V9.mfgenv.ReferTraj_V6 import TrajectoryManager
from Env.MFG_Musculoskelet_V9.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V9

class TMProxyManager(BaseManager): pass
TMProxyManager.register('TrajectoryManager', TrajectoryManager)

def print_total_mem(prefix=""):
    proc = psutil.Process(os.getpid())
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    print(f"{prefix} → Total RSS: {total/(1024**2):.1f} MB")

@main(version_base=None, config_path="configs", config_name="default")
def hydra_main(cfg):
    cfg = yaml.safe_load(OmegaConf.to_yaml(cfg))
    env_id   = cfg["env"]["id"]
    num_envs = cfg["env"].get("num_envs", 4)

    print_total_mem("Before VecEnv creation")
    envs = make_vec_env(
        MFG_Musculoskeletal_V9,
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "config": env_cfg,
            "traj_manager": shared_tm
        }
    )
    print_total_mem("After VecEnv creation")

    env = SB3RslVecEnv(
        envs,
        clip_actions=cfg["env"].get("clip_actions", None),
        device=cfg.get("device", "cpu"),
    )
    runner = OnPolicyRunner(
        env,
        cfg["train"],
        log_dir=cfg.get("log_dir", "./logs"),
        device=cfg.get("device", "cpu"),
    )
    runner.learn(num_learning_iterations=cfg["train"]["num_learning_iterations"])


if __name__ == "__main__":
    freeze_support()

    manager = TMProxyManager()
    manager.start()

    with open("Env\MFG_Musculoskelet_V9\cfg\mujocoenv_default.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
        
    tm_args = base_cfg["traj_manager"]    # 直接拿，不 pop
    shared_tm = manager.TrajectoryManager(
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

    # 4) **写回** shared_tm 到 config
    base_cfg["traj_manager"] = shared_tm

    # 5) 这份带 proxy 的 config 传给子进程
    global env_cfg
    env_cfg = base_cfg

    # 6) 最后启动 Hydra / 并行环境
    hydra_main()