from pathlib import Path
import pickle
from typing import List, Union, Tuple, Generator
from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class MotionData:

    joint_positions: Union[torch.Tensor, np.ndarray]
    # joint_velocities: Union[torch.Tensor, np.ndarray]
    # angle_position: Union[torch.Tensor, np.ndarray]
    # angle_velocities: Union[torch.Tensor, np.ndarray]
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        def to_tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)
        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = to_tensor(self.joint_positions)
        # if isinstance(self.joint_velocities, np.ndarray):
        #     self.joint_velocities = to_tensor(self.joint_velocities)
        # if isinstance(self.angle_positions, np.ndarray):
        #     self.angle_positions = to_tensor(self.angle_positions)
        # if isinstance(self.angle_velocities, np.ndarray):
        #     self.angle_velocities = to_tensor(self.angle_velocities)

    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.joint_positions[indices],
                # self.joint_velocities[indices],
                # self.angle_positions[indices],
                # self.angle_velocities[indices],
            ),
            dim=1,
        )

class AMPLoader:

    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        dataset_names: List[str] = None,
        dataset_weights: List[float] = None,
    ) -> None:
        
        self.device = device
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)
        if dataset_names is None:
            dataset_names = [p.stem for p in dataset_path_root.glob("*.pkl")]
        if dataset_weights is None:
            dataset_weights = [1.0] * len(dataset_names)
    
        self.motion_data: List[MotionData] = []
        for dataset_name in dataset_names:
            dataset_path = dataset_path_root / f"{dataset_name}.pkl"
            md = self.load_data(
                dataset_path,
            )
            self.motion_data.append(md)

        # Normalize dataset-level sampling weights
        weights = torch.tensor(dataset_weights, dtype=torch.float32, device=self.device)
        self.dataset_weights = weights / weights.sum()

        # Precompute flat buffers for fast sampling
        obs_list, next_obs_list = [], []
        for data, w in zip(self.motion_data, self.dataset_weights):
            T = len(data)
            idx = torch.arange(T, device=self.device)
            obs = data.get_amp_dataset_obs(idx)
            next_idx = torch.clamp(idx + 1, max=T - 1)
            next_obs = data.get_amp_dataset_obs(next_idx)
            obs_list.append(obs)
            next_obs_list.append(next_obs)

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)

        lengths = [len(d) for d in self.motion_data]
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()

    def load_data(
        self,
        dataset_path: Path,
    ) -> MotionData:
        with open(str(dataset_path), "rb") as f:
            data = pickle.load(f)
            data_new = data[:, 6:182]
        # import ipdb;ipdb.set_trace()
        return MotionData(
            joint_positions=data_new,
            # joint_velocities=data["joint_velocities"],
            # angle_positions=data["angle_positions"],
            # angle_velocities=data["angle_velocities"],
            device=self.device,
        )

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:

        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            yield self.all_obs[idx], self.all_next_obs[idx]

if __name__ == "__main__":
    loader = AMPLoader(
        device="cuda",
        dataset_path_root="Env/data_test",
        dataset_names=None,
        dataset_weights=None,
    )
    print("=== Testing feed_forward_generator ===")
    num_batches = 10
    batch_size = 40000
    gen = loader.feed_forward_generator(num_mini_batch=num_batches, mini_batch_size=batch_size)
    for i, (obs, next_obs) in enumerate(gen):
        # obs, next_obs 的 shape 应该都是 [batch_size, feature_dim]
        print(f"Batch {i}: obs.shape = {obs.shape}, next_obs.shape = {next_obs.shape}")
    print()