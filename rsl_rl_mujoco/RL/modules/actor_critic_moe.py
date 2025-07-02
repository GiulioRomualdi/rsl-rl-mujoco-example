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

def nan_hook(module, inputs, outputs, name=None):
    if isinstance(outputs, tuple):
        tensor = outputs[0]
    else:
        tensor = outputs
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[NaN] forward in {name} -> min={tensor.min().item():.3f}, max={tensor.max().item():.3f}")
        import ipdb; ipdb.set_trace()


def grad_hook(grad, name=None):
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print(f"[NaN-grad] in {name} grad -> min={grad.min().item():.3f}, max={grad.max().item():.3f}")
        import ipdb; ipdb.set_trace()
    return grad


class ActorCriticMOE(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_experts,
        expert_path,
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
        self.device = device

        # Gate 网络
        self.gating_norm = EmpiricalNormalization(shape=[num_actor_obs], until=1e8).to(self.device)
        act_fn = resolve_nn_activation(moe_activation)
        layers = [nn.Linear(num_actor_obs, moe_hidden_dims[0]), act_fn]
        for i in range(len(moe_hidden_dims) - 1):
            layers += [nn.Linear(moe_hidden_dims[i], moe_hidden_dims[i+1]), act_fn]
        layers += [nn.Linear(moe_hidden_dims[-1], num_experts), nn.Softmax(dim=-1)]
        self.gate = nn.Sequential(*layers).to(self.device)

        # 小增益初始化 gate
        for m in self.gate:
            if isinstance(m, nn.Linear):
                m.weight.data.mul_(0.1)
                m.bias.data.mul_(0.1)

        # 注册前向钩子监测 NaN/Inf
        for name, module in self.gate.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                module.register_forward_hook(lambda m, inp, out, n=name: nan_hook(m, inp, out, n))
        # 注册反向钩子监测梯度 NaN/Inf
        for name, p in self.gate.named_parameters():
            p.register_hook(lambda grad, n=name: grad_hook(grad, n))

        # 专家容器
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

        # Critic 网络
        activation = resolve_nn_activation(activation)
        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation]
        for idx in range(len(critic_hidden_dims)):
            in_dim = critic_hidden_dims[idx]
            out_dim = 1 if idx == len(critic_hidden_dims)-1 else critic_hidden_dims[idx+1]
            critic_layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers).to(self.device)

        self._gate_dist = None
        self._selected_expert_idx = None
        self._last_obs = None
        self._last_action = None

    @property
    def action_mean(self) -> torch.Tensor:
        # 混合后动作的期望 E[a] = Σ_i w_i μ_i
        x = self._last_obs
        w = self.gate(self.gating_norm(x))             # [B, E]
        means = []
        for mod in self.experts.experts:
            norm_module, ac_module = mod[0], mod[1]
            x_e = norm_module(x)
            mean_i = ac_module.actor(x_e)              # actor 输出即 mean
            means.append(mean_i)
        means = torch.stack(means, dim=1)              # [B, E, A]
        return (w.unsqueeze(-1) * means).sum(dim=1)    # [B, A]

    @property
    def action_std(self) -> torch.Tensor:
        # Var[a] = Σ_i w_i (σ_i^2 + μ_i^2) – (E[a])^2
        x = self._last_obs
        w = self.gate(self.gating_norm(x))
        means, vars_ = [], []
        for mod in self.experts.experts:
            norm_module, ac_module = mod[0], mod[1]
            x_e = norm_module(x)
            ac_module.update_distribution(x_e)
            means.append(ac_module.distribution.mean)
            vars_.append(ac_module.distribution.variance)
        means = torch.stack(means, dim=1)              # [B, E, A]
        vars_ = torch.stack(vars_, dim=1)              # [B, E, A]
        mu_mix = (w.unsqueeze(-1) * means).sum(dim=1)  # [B, A]
        second_moment = (w.unsqueeze(-1) * (vars_ + means**2)).sum(dim=1)
        var_mix = second_moment - mu_mix**2
        return torch.sqrt(var_mix)                     # [B, A]

    @property
    def entropy(self) -> torch.Tensor:
        x = self._last_obs
        w = self.gate(self.gating_norm(x))             # [B, E]

        # 1) 计算各个专家的熵 H_i
        Hs = []
        for mod in self.experts.experts:
            norm_module, ac_module = mod[0], mod[1]
            x_e = norm_module(x)
            ac_module.update_distribution(x_e)
            Hs.append(ac_module.distribution.entropy().sum(dim=-1))
        Hs = torch.stack(Hs, dim=1)  # [B, E]

        # 2) 用 where 屏蔽 w=0 时的 w*log(w)
        w_log_w = torch.where(w > 0, w * w.log(), torch.zeros_like(w))  # [B, E]

        # 3) 最终混合熵
        ent_mix = (w * Hs).sum(dim=1) - w_log_w.sum(dim=1)
        return ent_mix

    def reset(self, dones=None):
        pass
    
    def update_distribution(self, observations: torch.Tensor):
        self._last_obs = observations
        x_norm = self.gating_norm(observations)
        weights = self.gate(x_norm)
        # 运行时再次检查
        if torch.isnan(weights).any() or torch.isinf(weights).any() or (weights < 0).any():
            print("Invalid gate weights detected")
            import ipdb; ipdb.set_trace()
        self._gate_dist = Categorical(weights)
        self._selected_expert_idx = self._gate_dist.sample()
        all_actions = []
        for mod in self.experts.experts:
            norm_module, ac_module = mod[0], mod[1]
            x_e = norm_module(observations)
            action_e = ac_module.act(x_e)
            all_actions.append(action_e)
        actions_stack = torch.stack(all_actions, dim=1)
        idx = self._selected_expert_idx.view(-1,1,1).expand(-1,1,actions_stack.size(-1))
        self._last_action = actions_stack.gather(1, idx).squeeze(1)


    def act(self, observations: torch.Tensor,**kwargs) -> torch.Tensor:
        import ipdb;ipdb.set_trace()
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

    def evaluate(self, critic_observations: torch.Tensor,**kwargs) -> torch.Tensor:
        return self.critic(critic_observations)
    

class ActorCriticMO(nn.Module):
    is_recurrent = False
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_experts,
        expert_path,
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
        actor_hidden_dims = actor_hidden_dims or [256,256,256]
        critic_hidden_dims = critic_hidden_dims or [256,256,256]
        moe_units = moe_hidden_dims or [128,128]

        # 1) 全局状态归一化（供 gate 用）
        self.norm = EmpiricalNormalization(shape=[num_actor_obs], until=1e8).to(device)

        # 2) gate 网络：状态 → moe_units MLP → num_experts logits → Softmax
        act = nn.ReLU if moe_activation=="relu" else nn.ELU
        layers = []
        last_dim = num_actor_obs
        for h in moe_units:
            layers += [ nn.Linear(last_dim, h), act() ]
            last_dim = h
        layers += [ nn.Linear(last_dim, num_experts), nn.Softmax(dim=1) ]
        self.gate = nn.Sequential(*layers)

        # 3) N 个专家
        self.experts = Experts(
            num_experts=num_experts,
            expert_path=expert_path,
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            device=device,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        # 存储中间变量
        self._last_obs = None
        self._last_weights = None
        self._gate_dist: Categorical

    def update_distribution(self, observations: torch.Tensor):
        """
        1) 归一化
        2) 计算 gate 权重
        3) 构造离散分布
        4) 同时更新每个专家内部的 Normal 分布
        """
        self._last_obs = observations
        normed = self.norm(observations)
        self._last_weights = self.gate(normed)                  # [B, num_experts]
        self._gate_dist = Categorical(probs=self._last_weights) # 离散分布

        # 更新每个专家内部的 distribution (Normal)
        for exp in self.experts:
            exp.update_distribution(observations)

    @property
    def action_mean(self) -> torch.Tensor:
        """
        返回混合后（weighted sum）的 mean:
          sum_i w_i * μ_i(s)
        """
        # [B, E, A]
        means = torch.stack([exp.action_mean for exp in self.experts], dim=1)
        # [B, E, 1]
        w = self._last_weights.unsqueeze(-1)
        return (w * means).sum(dim=1)  # [B, A]

    @property
    def action_std(self) -> torch.Tensor:
        """
        返回混合后（weighted sum）的 std
        """
        stds = torch.stack([exp.action_std for exp in self.experts], dim=1)
        w = self._last_weights.unsqueeze(-1)
        return (w * stds).sum(dim=1)

    @property
    def entropy(self) -> torch.Tensor:
        """
        Gate 的 entropy + 各专家加权 entropy
        """
        # gate entropy: [B]
        ent_gate = self._gate_dist.entropy()
        # 各专家 entropy: [B, E]
        ent_exps = torch.stack([exp.entropy for exp in self.experts], dim=1)
        # weighted sum: [B]
        ent_exps = (self._last_weights * ent_exps).sum(dim=1)
        return ent_gate + ent_exps

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        1) 根据 gate 采样一个专家
        2) 再用该专家的 act() 输出连续动作
        """
        self.update_distribution(observations)
        idx = self._gate_dist.sample()  # [B]
        batch_size = observations.shape[0]
        # 每个样本调用对应专家
        actions = []
        for b in range(batch_size):
            expert = self.experts[idx[b].item()]
            a = expert.act(observations[b:b+1], **kwargs)  # [1, A]
            actions.append(a)
        return torch.cat(actions, dim=0)  # [B, A]

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        只对 gate 进行训练：
         - 找到每个动作 a 属于哪个专家最接近
         - 返回 gate 对应 expert_idx 的 log_prob
        """
        # 确保最后一次 update_distribution 过
        assert self._last_obs is not None, "Must call act() before get_actions_log_prob()"

        # 拿各专家的 mean 作为“代表”去测距离
        means = torch.stack([exp.action_mean for exp in self.experts], dim=1)  # [B, E, A]
        # [B, E]
        diffs = torch.sum((actions.unsqueeze(1) - means).abs(), dim=2)
        expert_ids = torch.argmin(diffs, dim=1)  # [B]

        return self._gate_dist.log_prob(expert_ids)  # [B]

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """
        评估时使用：直接选 gate 最高的 expert → 调用其 actor 的 mean
        """
        self.update_distribution(observations)
        idx = torch.argmax(self._last_weights, dim=1)
        batch_size = observations.shape[0]
        actions = []
        for b in range(batch_size):
            expert = self.experts[idx[b].item()]
            # 用 mean 而不是 sample
            a = expert.act_inference(observations[b:b+1])
            actions.append(a)
        return torch.cat(actions, dim=0)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        状态值：用各专家的 V(s) 加权平均
        """
        # 先确保 gate 权重与 experts 分布同步
        self.update_distribution(critic_observations)
        # [B, E, 1]
        values = torch.stack([exp.evaluate(critic_observations) for exp in self.experts], dim=1)  # [B, E, 1]
        w = self._last_weights.unsqueeze(-1)
        return (w * values).sum(dim=1)  # [B, 1] -> [B]
