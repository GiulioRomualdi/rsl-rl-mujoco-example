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

class Experts(nn.modules):
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
                checkpoint = torch.load(os.path.join(self.expert_path, f"expert_{i}", "model.pth"), weights_only=False)
            else:
                checkpoint = torch.load(os.path.join(self.expert_path, f"expert_{i}", "model.pth"), map_location=torch.device('cpu'))
            
            expert.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"Expert {i} loaded from {self.expert_path}/expert_{i}/model.pth")

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
        # Gate forward
        x_norm = self.gating_norm(observations)
        weights = self.gate(x_norm)
        self._gate_dist = Categorical(weights)
        # 采样专家索引
        self._selected_expert_idx = self._gate_dist.sample()  # [batch]
        # 各 expert 生成动作
        all_actions = []
        for mod in self.experts.experts:
            # mod 是 nn.Sequential(norm, ActorCritic)
            # 先对 obs 做归一化
            norm_module, ac_module = mod[0], mod[1]
            x_e = norm_module(observations)
            # 用 ActorCritic.act 来采样
            action_e = ac_module.act(x_e)
            all_actions.append(action_e)
        # stack 并按索引选取
        actions_stack = torch.stack(all_actions, dim=1)  # [B, E, A]
        idx = self._selected_expert_idx.view(-1,1,1).expand(-1,1,actions_stack.size(-1))
        self._last_action = actions_stack.gather(1, idx).squeeze(1)
        return

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        self.update_distribution(observations)
        return self._last_action

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # gate 部分的 log prob
        logp_gate = self._gate_dist.log_prob(self._selected_expert_idx)
        # expert 部分的 log prob
        logp_expert = []
        for i, mod in enumerate(self.experts.experts):
            norm_module, ac_module = mod[0], mod[1]
            x_e = norm_module(self._last_obs)
            ac_module.update_distribution(x_e)
            # Normal.log_prob 返回 [B, A]，需 sum 到 [B]
            logp = ac_module.distribution.log_prob(actions).sum(dim=-1)
            logp_expert.append(logp)
        logp_expert = torch.stack(logp_expert, dim=1)  # [B, E]
        # 取每个样本对应专家的 logp
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