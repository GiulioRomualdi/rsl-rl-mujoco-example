# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import os

import torch
import torch.nn as nn
from torch.distributions import Normal,Categorical
from .normalizer import EmpiricalNormalization
from .actor_critic import ActorCritic
from ..utils import resolve_nn_activation

class Experts(nn.Module):
    def __init__(
        self, 
        num_experts,
        expert_path,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        device="cpu",
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
        ):
        super().__init__()
        self.num_experts=num_experts
        self.expert_path=expert_path
        self.device=device

        self.norm=EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            expert=ActorCritic(
                num_actions=num_actions,
                num_actor_obs=num_actor_obs,
                num_critic_obs=num_critic_obs,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims
                ).to(self.device)
            if torch.cuda.is_available():
                checkpoint = torch.load(os.path.join(self.expert_path, f"expert_{i}_model.pt"), weights_only=False)
            else:
                checkpoint = torch.load(os.path.join(self.expert_path, f"expert_{i}_model.pt"), map_location=torch.device('cpu'))
            
            expert.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"Expert {i} loaded from {self.expert_path}/expert_{i}_model.pt")

            norm=EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
            norm.load_state_dict(checkpoint["obs_norm_state_dict"])
            del checkpoint
            self.experts.append(
                nn.Sequential(
                    norm,
                    expert
                )
            )
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad  = False

    def forward(self, x):
        raise NotImplementedError

class ActorCriticMOE(nn.Module):
    def __init__(
        self,
        num_experts,
        expert_path,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        device="cpu",
        moe_hidden_dims: list[int] = [256, 256],
        moe_activation: str = "relu",
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()
        self.device=device
        self.gating_norm = EmpiricalNormalization(shape=[num_actor_obs], until=1e8).to(self.device)
        activation = resolve_nn_activation(moe_activation)
        layers = [nn.Linear(num_actor_obs, moe_hidden_dims[0]), activation]
        for i in range(len(moe_hidden_dims) - 1):
            layers += [nn.Linear(moe_hidden_dims[i], moe_hidden_dims[i+1]), activation]
        layers += [nn.Linear(moe_hidden_dims[-1], num_experts), nn.Softmax(dim=-1)]
        self.gate = nn.Sequential(*layers).to(self.device)
        self.experts = Experts(
            num_experts=num_experts,
            expert_path=expert_path,
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            device=device,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=moe_activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for idx in range(len(critic_hidden_dims)):
            in_dim = critic_hidden_dims[idx]
            out_dim = 1 if idx == len(critic_hidden_dims)-1 else critic_hidden_dims[idx+1]
            critic_layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self._gate_dist = None
        self._selected_expert_idx = None
        self._last_action = None


    def reset(self, dones=None):
        pass
    
    def update_distribution(self, observations: torch.Tensor):
        # 保存一份原始 obs，后面计算 log_prob 时要用
        self._last_obs = observations

        # Gate forward
        x_norm = self.gating_norm(observations)
        weights = self.gate(x_norm)
        self._gate_dist = torch.distributions.Categorical(weights)
        # 采样专家索引
        self._selected_expert_idx = self._gate_dist.sample()
        # 各 expert 生成动作
        all_actions = []
        for mod in self.experts.experts:
            norm_module, ac_module = mod[0], mod[1]
            x_e = norm_module(observations)
            action_e = ac_module.act(x_e)
            all_actions.append(action_e)
        actions_stack = torch.stack(all_actions, dim=1)
        idx = self._selected_expert_idx.view(-1,1,1).expand(-1,1,actions_stack.size(-1))
        self._last_action = actions_stack.gather(1, idx).squeeze(1)


    def act(self, observations: torch.Tensor) -> torch.Tensor:
        self.update_distribution(observations)
        return self._last_action

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        logp_gate   = self._gate_dist.log_prob(self._selected_expert_idx)

        logp_expert = []
        for i, mod in enumerate(self.experts.experts):
            norm_module, ac_module = mod[0], mod[1]
            # 这里用保留的 last_obs
            x_e = norm_module(self._last_obs)
            ac_module.update_distribution(x_e)
            logp = ac_module.distribution.log_prob(actions).sum(dim=-1)
            logp_expert.append(logp)

        logp_expert = torch.stack(logp_expert, dim=1)
        idx = self._selected_expert_idx.view(-1,1)
        chosen_logp = logp_expert.gather(1, idx).squeeze(1)

        return logp_gate + chosen_logp


    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        x_norm = self.gating_norm(observations)
        weights = self.gate(x_norm)                        # [B, E]
        # 收集各 expert 的 mean
        means = []
        for mod in self.experts.experts:
            norm_module, ac_module = mod[0], mod[1]
            x_e = norm_module(observations)
            mean_e = ac_module.act_inference(x_e)         # [B, A]
            means.append(mean_e)
        means_stack = torch.stack(means, dim=1)            # [B, E, A]
        # 按 gate 权重加权
        weights = weights.unsqueeze(-1)                    # [B, E, 1]
        return (weights * means_stack).sum(dim=1)          # [B, A]

    def evaluate(self, critic_observations: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_observations)
    

if __name__ == "__main__":
    import torch
    # —— 配置示例 —— 
    cfg = {
        "env": {
            "obs_dim": 8,
            "action_dim": 4,
        },
        "policy": {
            "num_experts": 2,
            "expert_path": "model/test_moe",     # 请换成你真实的路径
            "moe_hidden_dims": [64, 32],
            "moe_activation": "relu",
            "actor_hidden_dims": [1024,512,512,256],
            "critic_hidden_dims": [1024,512,512,256],
            "init_noise_std": 0.1,
            "noise_std_type": "scalar",
        },
    }
    device = "cuda"

    # 1) 构造模型
    model = ActorCriticMOE(
        num_actor_obs=cfg["env"]["obs_dim"],
        num_critic_obs=cfg["env"]["obs_dim"],
        num_actions=cfg["env"]["action_dim"],
        num_experts=cfg["policy"]["num_experts"],
        expert_path=cfg["policy"]["expert_path"],
        moe_hidden_dims=cfg["policy"]["moe_hidden_dims"],
        moe_activation=cfg["policy"]["moe_activation"],
        actor_hidden_dims=cfg["policy"]["actor_hidden_dims"],
        critic_hidden_dims=cfg["policy"]["critic_hidden_dims"],
        init_noise_std=cfg["policy"]["init_noise_std"],
        noise_std_type=cfg["policy"]["noise_std_type"],
        device=device,
    )
    model.to(device)

    # 2) 制造一批随机观测
    batch_size = 5
    obs = torch.randn(batch_size, cfg["env"]["obs_dim"], device=device)

    # 3) act() —— 训练模式下采样动作
    actions = model.act(obs)
    print(f"act  output shape: {actions.shape}")  # [B, A]

    # 4) get_actions_log_prob() —— 计算 log prob
    logp = model.get_actions_log_prob(actions)
    print(f"log_prob shape: {logp.shape}")       # [B]

    # 5) act_inference() —— 推理模式下返回 mean action
    means = model.act_inference(obs)
    print(f"act_inference shape: {means.shape}") # [B, A]

    # 6) evaluate() —— 价值函数输出
    values = model.evaluate(obs)
    print(f"evaluate shape: {values.shape}")     # [B, 1]
