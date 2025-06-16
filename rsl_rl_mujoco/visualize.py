#!/usr/bin/env python3
from collections import defaultdict
import os
import time
import numpy as np
import wandb
import yaml
import torch
import argparse
import gymnasium as gym
from pathlib import Path
from RL.modules import ActorCritic,EmpiricalNormalization
from rsl_rl_mujoco.env_wrapper_eval import GymMujocoWrapper
from pathlib import Path
import imageio

gym.register(
    id='MFG_MS_V7',
    entry_point='Env.MFG_Musculoskelet_V7.mfgenv.mfg_env:MFG_Musculoskelet_V7',
    max_episode_steps=1000
)
gym.register(
    id='MFG_MS_V8',
    entry_point='Env.MFG_Musculoskelet_V8.mfgenv.mfg_env:MFG_Musculoskelet_V8',
    max_episode_steps=400
)
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
        
        # Set default device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_environment(self):
        """Create and configure the environment"""
        env_kwargs = {
            "render_mode": self.cfg["env"]["render_mode"],
        }
        

        
        # Create base environment
        self.env = gym.make(self.cfg["env"]["id"], **env_kwargs
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
            speed=self.env.speed()
            finish=not info["terminated_info"]['has_fallen'] and not info["terminated_info"]['site_deviation_exceeded'] 
            print(speed)
            print(info["terminated_info"])

            if speed not in vel_dic.keys():
                vel_dic[speed]=[0.,0.]
            vel_dic[speed][0]+=1.
            if finish:
                vel_dic[speed][1]+=1.
            print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}")
            
            # data=np.array(data).astype(np.float32)
            
            # # import ipdb;ipdb.set_trace()
            # file_path = os.path.join(path ,f"data_{episode:02d}.npz")
            # np.savez_compressed(file_path, **{"joint_position": data})
            if self.cfg['visualization']["wandb"]:
                bin_width = 0.1  # 你可以改成 1.0, 0.2 等
                max_speed = 3.0

                # 初始化桶
                bucket = defaultdict(lambda: [0, 0])  # [count_sum, success_sum]

                # 分桶统计
                for speed, (count, success) in vel_dic.items():
                    bin_index = int(speed // bin_width)
                    bin_key = round(bin_index * bin_width, 2)
                    bucket[bin_key][0] += count
                    bucket[bin_key][1] += success

                # 构造数据
                bins = []
                counts = []
                successes = []
                rates = []

                # 保证x轴顺序
                num_bins = int(max_speed // bin_width)
                for i in range(num_bins):
                    b = round(i * bin_width, 2)
                    c, s = bucket[b]
                    r = s / c if c > 0 else 0.0
                    bins.append(f"{b:.1f}-{b+bin_width:.1f}")
                    counts.append(c)
                    successes.append(s)
                    rates.append(r)
                # Table 格式：每行一个区间
                table_count = wandb.Table(data=list(zip(bins, counts)), columns=["speed_bin", "count"])
                table_success = wandb.Table(data=list(zip(bins, successes)), columns=["speed_bin", "success"])
                table_rate = wandb.Table(data=list(zip(bins, rates)), columns=["speed_bin", "success_rate"])

                # 绘图
                wandb.log({
                    "Speed Bin vs Count": wandb.plot.bar(table_count, "speed_bin", "count", title="Speed Bin vs Count"),
                    "Speed Bin vs Success": wandb.plot.bar(table_success, "speed_bin", "success", title="Speed Bin vs Success"),
                    "Speed Bin vs Success Rate": wandb.plot.bar(table_rate, "speed_bin", "success_rate", title="Speed Bin vs Success Rate")
                })
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