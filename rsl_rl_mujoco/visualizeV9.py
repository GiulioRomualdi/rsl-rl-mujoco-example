#!/usr/bin/env python3
from collections import defaultdict
import os
import time
import wandb
import yaml
import torch
import argparse
import gymnasium as gym
from pathlib import Path
from RL.modules import ActorCritic,EmpiricalNormalization
from rsl_rl_mujoco.env_wrapper_eval import GymMujocoWrapper
from pathlib import Path

from Env.MFG_Musculoskelet_V9.mfgenv.ReferTraj_V7 import TrajectoryManager
from Env.MFG_Musculoskelet_V9.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V9

class PolicyVisualizer:
    def __init__(self, cfg_path):
        self.load_config(cfg_path)
        self.setup_environment()
        
        self.num_obs=self.env.num_obs
        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs], until=1.0e8).to(self.device)

        self.load_policy()

    def load_config(self, cfg_path):
        """Load visualization configuration"""
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_environment(self):
        with open("Env\MFG_Musculoskelet_V9\cfg\mujocoenv_default.yaml", "r") as f:
            base_cfg = yaml.safe_load(f)
            
        tm_args = base_cfg["traj_manager"]    
        shared_tm =TrajectoryManager(
            data_path=tm_args["data_path"],
            sample_frequency=float(tm_args["sample_frequency"]),
            speed_range=(float(tm_args["speed_range"][0]), float(tm_args["speed_range"][1])),
            verbose=bool(tm_args["verbose"])
        )

        base_cfg["traj_manager"] = shared_tm
        
        self.env = MFG_Musculoskeletal_V9(
            config=base_cfg,
            render_mode="human"
        )
        
        # Wrap for RSL-RL compatibility
        self.env = GymMujocoWrapper(
            self.env,
            device=self.device,
            is_finite_horizon=False
        )

    def load_policy(self):
        """Load trained policy network"""
        # import ipdb;ipdb.set_trace()
        self.policy = ActorCritic(
            num_actions=self.env.num_actions,
            num_actor_obs=self.env.num_obs,
            num_critic_obs=self.env.num_obs,
            actor_hidden_dims=self.cfg["policy"]["hidden_dims"],
            critic_hidden_dims=self.cfg["policy"]["hidden_dims"]
        ).to(self.device)
        
        # Load checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(self.cfg["policy"]["checkpoint"], weights_only=False)

        else:
            checkpoint = torch.load(self.cfg["policy"]["checkpoint"],map_location=torch.device('cpu'))
        # import ipdb;ipdb.set_trace()
        self.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
        self.obs_normalizer.eval()
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()
        print(f"Loaded policy from {self.cfg['policy']['checkpoint']}")

    def run(self):
        """Run visualization loop"""
        print(f"Starting visualization for {self.cfg['visualization']['num_episodes']} episodes...")
        vel_dic={}
        if self.cfg['visualization']["wandb"]:
            wandb.init(project="Test_Env")
        # path="Env/data_test"
        for episode in range(self.cfg["visualization"]["num_episodes"]):
            obs, _ = self.env.reset()
            # data=[obs.squeeze().tolist()]
            obs=self.obs_normalizer(obs)
            episode_reward = 0
            done = False
            while not done:
                with torch.no_grad():
                    actions = self.policy.act_inference(obs)
                
                obs, reward, done, info = self.env.step(actions)
                # data.append(obs.squeeze().tolist())
                obs=self.obs_normalizer(obs)
                episode_reward += reward.item()
                
                # Control playback speed
                time.sleep(1.0 / (self.env.max_episode_length * self.cfg["visualization"]["speedup"]))
            
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description="RSL-RL Policy Visualizer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "visualize.yaml"),
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    visualizer = PolicyVisualizer(args.config)
    visualizer.run()

if __name__ == "__main__":
    main()