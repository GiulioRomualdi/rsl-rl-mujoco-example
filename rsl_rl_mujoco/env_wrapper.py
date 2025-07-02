import numpy as np
import torch
import gymnasium as gym
from RL.env.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecEnv as SB3VecEnv

class SB3RslVecEnv(VecEnv):
    """
    Wraps a Stable-Baselines3 VecEnv (e.g. SubprocVecEnv) to match rsl_rl.env.VecEnv API.
    """

    def __init__(
        self,
        sb3_vec_env: SB3VecEnv,
        clip_actions: float = None,
        is_finite_horizon: bool = False,
        device: str = "cpu",
    ):
        # underlying SB3 vectorized env
        self.env = sb3_vec_env
        self.num_envs = self.env.num_envs
        self.clip_actions = clip_actions
        self.device = torch.device(device)

        # build a fake cfg object for RSL-RL
        self._cfg = type("Cfg", (), {"is_finite_horizon": is_finite_horizon})

        # gym spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # dims
        self.num_obs     = gym.spaces.flatdim(self.observation_space)
        self.num_actions = gym.spaces.flatdim(self.action_space)
        self.num_privileged_obs = 0

        # episode bookkeeping
        self.max_episode_length = getattr(self.env, "max_episode_steps", None) or 1000
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # placeholder for last observations
        self._obs = None
        self.reset()

    @property
    def cfg(self) -> object:
        return self._cfg

    def get_observations(self):
        obs_tensor = torch.tensor(self._obs, dtype=torch.float32, device=self.device)
        amp_tensor = obs_tensor[:, 6:182]

        return obs_tensor, {"observations": {"policy": obs_tensor,"amp": amp_tensor}}

    def reset(self):
        out = self.env.reset()
        # SB3 VecEnv may return (obs, infos)
        obs = out[0] if isinstance(out, tuple) else out
        # shape (num_envs, *obs_shape)
        self._obs = np.array(obs)
        return self.get_observations()

    def step(self, actions: torch.Tensor):
        # clip if requested
        # import ipdb;ipdb.set_trace()
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        actions_np = actions.cpu().numpy()

        # step through vectorized env
        out = self.env.step(actions_np)
        obs_batch, reward_batch, done_batch, info_batch = out[:4]
        obs_term = obs_batch.copy()
        reset_idxs = np.nonzero(done_batch)[0]
        for i in reset_idxs:
            term_obs = info_batch[i].get("terminal_observation")
            if term_obs is not None:
                obs_term[i] = term_obs
            else:
                pass
        # SB3 automatically resets environments on done and populates info["terminal_observation"]
        self._obs = np.array(obs_batch)
        self._obs_tem=np.array(obs_term)

        # convert to tensors
        obs_tensor     = torch.tensor(self._obs,      dtype=torch.float32, device=self.device)
        obs_term_tensor     = torch.tensor(self._obs_tem,      dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(reward_batch,    dtype=torch.float32, device=self.device)
        dones_tensor   = torch.tensor(done_batch,      dtype=torch.long,    device=self.device)
        # import ipdb;ipdb.set_trace()
        amp_tensor     = obs_term_tensor[:, 6:182]
        extras = {"observations": {"policy": obs_tensor,"amp": amp_tensor}}
        if not self.cfg.is_finite_horizon:
            truncated_flags = [
                info.get("TimeLimit.truncated", False)
                for info in info_batch
            ]
            extras["time_outs"] = torch.tensor(
                truncated_flags, dtype=torch.bool, device=self.device
            )
        return obs_tensor, rewards_tensor, dones_tensor, extras

    def close(self):
        self.env.close()

    def seed(self, seed: int = -1) -> int:
        # SB3 VecEnv supports seed via its seed() method
        try:
            self.env.seed(seed)
        except AttributeError:
            # fallback: reset with seed
            self.env.reset(seed=seed)
        return seed
