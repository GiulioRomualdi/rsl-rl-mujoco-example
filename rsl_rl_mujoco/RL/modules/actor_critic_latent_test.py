# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from ..utils import resolve_nn_activation


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        alpha=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
    
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy 
        self.backbone = nn.Sequential(
            nn.Linear(num_actor_obs, actor_hidden_dims[0]),
            activation,
            *sum([[nn.Linear(h_in, h_out), activation]
                  for h_in,h_out in zip(actor_hidden_dims[:-1],actor_hidden_dims[1:])], [])
        )
        self.action_head = nn.Linear(actor_hidden_dims[-1], num_actions)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0) 

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Backbone MLP: {self.backbone}")
        print(f"Critic MLP: {self.critic}")

        # std
        self.action_dim = num_actions
        self.latent_dim = actor_hidden_dims[-1]
        self.log_std = nn.Parameter(
                torch.ones(1, self.action_dim + self.latent_dim) * -2.3,
                requires_grad=True
            )
        self.alpha = alpha
        self._eps = 1e-6                
        self.distribution = None
        MultivariateNormal.set_default_validate_args(False) 

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy()

    def update_distribution(self, observations,step_idx=0):
        """
        根据给定观测更新动作分布；核心与 PolicyLattice 保持一致。
        """
        h = self.backbone(observations)                 # (B, latent_dim)
        a_mean = self.action_head(h)                    # (B, action_dim)

        # 1) 取出 std 向量并拆分
        std = torch.exp(self.log_std)                   # (1, action_dim + latent_dim)
        action_std  = std[:, :self.action_dim]          # (1, A)
        latent_std  = std[:, self.action_dim:]          # (1, L)

        action_var  = action_std.pow(2)                 # (1, A)
        latent_var  = latent_std.pow(2)                 # (1, L)

        W = self.action_head.weight                     # (A, L)
        sigma_mat = (W * latent_var[..., None, :]).matmul(W.t())  # (B, A, A)
        sigma_mat[..., torch.arange(self.action_dim), torch.arange(self.action_dim)] += action_var

        # 3) 保存分布
        self.distribution = MultivariateNormal(a_mean, sigma_mat)

        return self.distribution

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions)

    def act_inference(self, observations):
        actions_mean = self.action_head(self.backbone(observations))
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
