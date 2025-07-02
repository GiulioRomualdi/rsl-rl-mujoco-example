'''
ReferenceTrajectories class for MuJoCo musculoskeletal model training.
@author: YAKE
'''
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
from scipy.ndimage import gaussian_filter1d
import logging
from bisect import bisect_left

class TrajectoryManager:
    """
    Load and preprocess all gait segments once, reuse in multiple envs.
    """
    
    # Joint name to index mapping
    JOINT_MAP: Dict[str, int] = {
        'pelvis_tz': 0, 'pelvis_ty': 1, 'pelvis_tx': 2, 'pelvis_tilt': 3, 'pelvis_list': 4, 'pelvis_rotation': 5,
        'hip_flexion_r': 6, 'hip_adduction_r': 7, 'hip_rotation_r': 8,
        'knee_angle_r': 9, 'knee_angle_r_rotation2': 10, 'knee_angle_r_rotation3': 11,
        'ankle_angle_r': 12, 'subtalar_angle_r': 13, 'mtp_angle_r': 14,
        'hip_flexion_l': 15, 'hip_adduction_l': 16, 'hip_rotation_l': 17,
        'knee_angle_l': 18, 'knee_angle_l_rotation2': 19, 'knee_angle_l_rotation3': 20,
        'ankle_angle_l': 21, 'subtalar_angle_l': 22, 'mtp_angle_l': 23,
        'lumbar_extension': 24, 'lumbar_bending': 25, 'lumbar_rotation': 26,
        'arm_flex_r': 27, 'arm_add_r': 28, 'arm_rot_r': 29, 'elbow_flex_r': 30, 'pro_sup_r': 31, 'wrist_flex_r': 32, 'wrist_dev_r': 33,
        'arm_flex_l': 34, 'arm_add_l': 35, 'arm_rot_l': 36, 'elbow_flex_l': 37, 'pro_sup_l': 38, 'wrist_flex_l': 39, 'wrist_dev_l': 40
    }
    
    def __init__(
        self,
        data_path: Union[str, Path],
        repeat_times: int = 10,
        sample_frequency: int = 100,
        knee_1dof: bool = True,
        enable_mirroring: bool = True,
        smoothing_sigma: Optional[float] = None,
        splice_overlap: int = 10,
        speed_range: Tuple[float, float] = (0.3, 3),
        uniform_length: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize and preprocess all gait trajectories.
        Parameters
        ----------
        data_path : Union[str, Path]
            Path to the pickle file containing List[np.ndarray] of raw segments.
        repeat_times : int, default=10
            How many times to tile each trimmed gait cycle.
        sample_frequency : int, default=100
            Sampling rate (Hz) of the raw trajectory data.
        knee_1dof : bool, default=True
            If True, remove extra knee DoFs before mirroring.
        enable_mirroring : bool, default=True
            Whether to append left-right mirrored copies of each segment.
        smoothing_sigma : Optional[float], default=None
            Gaussian smoothing sigma; if None, chosen adaptively.
        splice_overlap : int, default=10
            Number of frames to overlap when smoothing splices.
        speed_range : Tuple[float, float], default=(0.0, 3.0)
            Minimum and maximum forward speeds (m/s) to keep segments.
        uniform_length: bool, default=False
            Whether to uniform all steps.
        verbose : bool, default=False
            If True, set logger to DEBUG level.

        Raises
        ------
        ValueError
            If no trajectories remain after filtering by speed_range.
        """
        # Configure logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
        
        # Initialize attributes
        self.data_path = Path(data_path)
        self.repeat_times = repeat_times
        self.sample_frequency = sample_frequency
        self.smoothing_sigma = smoothing_sigma
        self.splice_overlap = splice_overlap
        self.jnt_name = self.JOINT_MAP.copy()
        self.rng = np.random.default_rng()
        
        # Load and validate raw segments
        raw_segments: List[np.ndarray] = []
        if self.data_path.is_dir():
            pkl_files = sorted(self.data_path.glob("*.pkl"))
            if not pkl_files:
                raise FileNotFoundError(f"No .pkl files found in directory {self.data_path}")
            for pkl in pkl_files:
                raw_segments.extend(self._load_raw_data(pkl))
        else:
            raw_segments = self._load_raw_data(self.data_path)
        self._validate_data(raw_segments)
        
        # Preprocess: remove extra knee DOFs, mirror
        if knee_1dof:
            raw_segments = self._remove_extra_knee_dofs(raw_segments)
        if enable_mirroring:
            raw_segments = self._augment_with_mirroring(raw_segments)
        
        # Classify start foot and compute speeds
        left_ids, right_ids = self._classify_steps(raw_segments)
        speeds = self._calculate_speeds(raw_segments)
        
        trimmed_lengths = [
            int(np.ceil(seg.shape[1] * 2/3)) 
            for seg in raw_segments
        ]
        self.min_cycle_frames = min(trimmed_lengths)
        self.uniform_length = uniform_length
        
        # Prepare storage
        self.qpos_list: List[np.ndarray] = []
        self.qvel_list: List[np.ndarray] = []
        self.foot_starts: List[int] = []       # 1 for left-start, 0 for right-start
        self.speeds_list: List[float] = []
    
        # Process and filter segments
        for idx, step in enumerate(raw_segments):
            sp = speeds[idx]
            if not (speed_range[0] <= sp <= speed_range[1]):
                continue
            # Generate qpos/qvel
            qpos = self._process_step_trim_repeat(step)
            qvel = self._compute_velocity(qpos)
            # Record
            self.qpos_list.append(qpos)
            self.qvel_list.append(qvel)
            self.foot_starts.append(1 if idx in left_ids else 0)
            self.speeds_list.append(sp)

        if not self.qpos_list:
            raise ValueError(f"No trajectories loaded after speed filtering: {speed_range}")
            
        for arr in self.qpos_list + self.qvel_list:
            arr.setflags(write=False)
    
        self.logger.debug(f"Loaded {len(self.qpos_list)} trajectories; foot_starts={self.foot_starts}, speeds={self.speeds_list}")

    def __len__(self) -> int:
        """
        Return the number of available trajectories.
    
        Returns
        -------
        int
            Total count of preprocessed trajectories.
        """
        return len(self.qpos_list)

    def get(self, traj_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve qpos and qvel arrays for a specific trajectory.
    
        Parameters
        ----------
        traj_id : int
            Index of the trajectory to retrieve.
    
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of (qpos, qvel) arrays for the given trajectory.
    
        Raises
        ------
        IndexError
            If traj_id is out of the valid range.
        """
        if not (0 <= traj_id < len(self.qpos_list)):
            raise IndexError(f"traj_id {traj_id} is out of range [0, {len(self.qpos_list)-1}]")
        return self.qpos_list[traj_id], self.qvel_list[traj_id]
    
    def get_foot_start(self, traj_id: int) -> int:
        """
        Get the starting foot indicator for a specific trajectory.
    
        Parameters
        ----------
        traj_id : int
            Index of the trajectory.
    
        Returns
        -------
        int
            1 if left-start, 0 if right-start.
    
        Raises
        ------
        IndexError
            If traj_id is out of the valid range.
        """
        if not (0 <= traj_id < len(self.foot_starts)):
            raise IndexError(f"traj_id {traj_id} is out of range [0, {len(self.foot_starts)-1}]")
        return self.foot_starts[traj_id]
    
    def get_speed(self, traj_id: int) -> float:
        """
        Get the forward speed (m/s) for a specific trajectory.
    
        Parameters
        ----------
        traj_id : int
            Index of the trajectory.
    
        Returns
        -------
        float
            Forward speed of the given trajectory.
    
        Raises
        ------
        IndexError
            If traj_id is out of the valid range.
        """
        if not (0 <= traj_id < len(self.speeds_list)):
            raise IndexError(f"traj_id {traj_id} is out of range [0, {len(self.speeds_list)-1}]")
        return self.speeds_list[traj_id]

    @property
    def all_foot_starts(self) -> List[int]:
        """
        List of starting foot indicators for all trajectories.
    
        Returns
        -------
        List[int]
            1 means left-start, 0 means right-start for each trajectory.
        """
        return list(self.foot_starts)

    @property
    def all_speeds(self) -> List[float]:
        """
        List of speeds (m/s) for all trajectories.
    
        Returns
        -------
        List[float]
        """
        return list(self.speeds_list)

    def _load_raw_data(self, pkl_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Load raw trajectory segments from a pickle file.
    
        Parameters
        ----------
        pkl_path : Union[str, Path]
            Path to the pickle file containing the list of gait segments.
    
        Returns
        -------
        List[np.ndarray]
            List of arrays (dofs × frames) for each gait segment.
    
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        IOError
            If the file cannot be read or unpickled.
        """
        path = Path(pkl_path)
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")
        try:
            with path.open('rb') as f:
                data = pickle.load(f)
            self.logger.debug(f"Loaded raw data from {path}")
            return data
        except Exception as e:
            raise IOError(f"Failed to read pickle file {path}: {e}")
    
    def _validate_data(self, data: List[np.ndarray]) -> None:
        """
        Verify that 'data' is a non-empty list of 2D numpy arrays with sufficient length.
    
        Parameters
        ----------
        data : List[np.ndarray]
            Raw trajectory segments to validate.
    
        Raises
        ------
        ValueError
            If data is not a list, is empty, contains non-numpy elements,
            has arrays that are not 2D, or have fewer than 3 frames.
        """
        if not isinstance(data, list) or not data:
            raise ValueError("Raw trajectory data must be a non-empty list.")
        for idx, step in enumerate(data):
            if not isinstance(step, np.ndarray):
                raise ValueError(f"Segment {idx} is not a numpy array.")
            if step.ndim != 2:
                raise ValueError(f"Segment {idx} must be 2D, but has ndim={step.ndim}.")
            if step.shape[1] < 3:
                raise ValueError(f"Segment {idx} has insufficient frames ({step.shape[1]} < 3).")
    
    # --- Pre-processing: Remove excess knee freedom ---
    def _remove_extra_knee_dofs(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Remove redundant knee DoFs (rotation2/3) from each segment and update joint mapping.
    
        Parameters
        ----------
        data : List[np.ndarray]
            Original gait segments.
    
        Returns
        -------
        List[np.ndarray]
            Processed segments with the extra knee rows deleted.
    
        Side Effects
        ------------
        Updates self.jnt_name to reflect new row indices.
        """
        removal_keys = [
            'knee_angle_r_rotation2', 'knee_angle_r_rotation3',
            'knee_angle_l_rotation2', 'knee_angle_l_rotation3'
        ]
        # gather and sort indices to delete
        removal_indices = sorted(self.jnt_name[k] for k in removal_keys if k in self.jnt_name)
        if not removal_indices:
            self.logger.debug("No extra knee DOFs found; skipping removal.")
            return data
    
        # remove rows by index
        processed = [np.delete(seg, removal_indices, axis=0) for seg in data]
        # update the joint-name → index mapping
        self._update_joint_mapping(removal_indices)
        self.logger.debug(f"Removed knee DOFs at rows {removal_indices}")
        return processed
    
    def _update_joint_mapping(self, removal_indices: List[int]) -> None:
        """
        Rebuild self.jnt_name after removing the specified row indices.
    
        Parameters
        ----------
        removal_indices : List[int]
            Sorted list of row indices that were deleted.
        """
        # use bisect for efficient offset calculation
        new_map: Dict[str, int] = {}
        for joint, old_idx in self.jnt_name.items():
            # skip any joint that was removed
            pos = bisect_left(removal_indices, old_idx)
            if pos < len(removal_indices) and removal_indices[pos] == old_idx:
                continue
            # subtract how many removed indices were less than old_idx
            new_map[joint] = old_idx - pos
    
        self.jnt_name = new_map
        self.logger.debug(f"Updated joint mapping: {self.jnt_name}")
    
    # --- Data enhancement: mirror gait ---
    def _augment_with_mirroring(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Augment the dataset by appending a left–right mirrored copy of each segment.
    
        Parameters
        ----------
        data : List[np.ndarray]
            Original gait segments.
    
        Returns
        -------
        List[np.ndarray]
            Combined list: [seg0, mirror(seg0), seg1, mirror(seg1), …].
        """
        augmented: List[np.ndarray] = []
        for step in data:
            augmented.append(step)
            augmented.append(self._mirror_step(step))
        self.logger.info(f"Mirrored {len(data)} segments → total {len(augmented)}")
        return augmented
    
    def _mirror_step(self, step: np.ndarray) -> np.ndarray:
        """
        Create a mirrored version of one gait segment by swapping left/right joint channels
        and flipping sign of lateral axes, using index-based reordering for efficiency.
    
        Parameters
        ----------
        step : np.ndarray, shape (dofs, frames)
            Original segment.
    
        Returns
        -------
        np.ndarray
            Mirrored segment (same shape).
        """
        dofs, frames = step.shape
        # build index mapping: start with identity
        idx_map = np.arange(dofs, dtype=int)
        # swap each pair
        mirror_pairs = {
            "hip_flexion_r": "hip_flexion_l",
            "hip_adduction_r": "hip_adduction_l",
            "hip_rotation_r": "hip_rotation_l",
            "knee_angle_r": "knee_angle_l",
            "ankle_angle_r": "ankle_angle_l",
            "subtalar_angle_r": "subtalar_angle_l",
            "mtp_angle_r": "mtp_angle_l",
            "arm_flex_r": "arm_flex_l",
            "arm_add_r": "arm_add_l",
            "arm_rot_r": "arm_rot_l",
            "elbow_flex_r": "elbow_flex_l",
            "pro_sup_r": "pro_sup_l",
            "wrist_flex_r": "wrist_flex_l",
            "wrist_dev_r": "wrist_dev_l"
        }
        for r_name, l_name in mirror_pairs.items():
            if r_name in self.jnt_name and l_name in self.jnt_name:
                ir, il = self.jnt_name[r_name], self.jnt_name[l_name]
                idx_map[ir], idx_map[il] = il, ir
    
        # apply reordering
        mirrored = step[idx_map, :].copy()
    
        # flip sign on lateral axes
        sign_flip_keys = ['pelvis_tz', 'pelvis_list', 'pelvis_rotation', 'lumbar_bending', 'lumbar_rotation']
        for key in sign_flip_keys:
            if key in self.jnt_name:
                mirrored[self.jnt_name[key], :] *= -1
    
        return mirrored
    
    # --- Gait classification and selection ---
    def _classify_steps(self, data: List[np.ndarray]) -> Tuple[List[int], List[int]]:
        """
        Split segment indices into left-start and right-start.
    
        Parameters
        ----------
        data : List[np.ndarray]
            Preprocessed gait segments.
    
        Returns
        -------
        left_ids : List[int]
            Indices where the left foot starts first.
        right_ids : List[int]
            Indices where the right foot starts first.
        """
        left_ids = self._determine_left_step_indices(data)
        total = len(data)
        set_left = set(left_ids)
        # right_ids is the complement
        right_ids = [i for i in range(total) if i not in set_left]
    
        self.logger.info(f"Classified {len(left_ids)} left-start and {len(right_ids)} right-start segments")
        return left_ids, right_ids
    
    def _determine_left_step_indices(self, data: List[np.ndarray]) -> List[int]:
        """
        Identify segments that start with the left foot based on initial hip flexion.
    
        Parameters
        ----------
        data : List[np.ndarray]
            Preprocessed gait segments.
    
        Returns
        -------
        List[int]
            Indices of segments where left initial hip flexion > right.
    
        Raises
        ------
        KeyError
            If 'hip_flexion_l' or 'hip_flexion_r' is missing from joint mapping.
        """
        # fetch indices, error early if missing
        if 'hip_flexion_l' not in self.jnt_name or 'hip_flexion_r' not in self.jnt_name:
            raise KeyError("Hip flexion indices missing in joint mapping")
        l_idx = self.jnt_name['hip_flexion_l']
        r_idx = self.jnt_name['hip_flexion_r']
    
        left_ids = []
        for i, seg in enumerate(data):
            # skip if segment has too few DoFs
            if seg.shape[0] <= max(l_idx, r_idx):
                self.logger.warning(f"Segment {i} too short for hip flexion test; skipping")
                continue
            # compare initial frame
            if seg[l_idx, 0] > seg[r_idx, 0]:
                left_ids.append(i)
    
        self.logger.debug(f"Left-start segments: {left_ids}")
        return left_ids
    
    def _calculate_speeds(self, data: List[np.ndarray]) -> List[float]:
        """
        Compute forward speed of each segment via pelvis_tx displacement.
    
        Parameters
        ----------
        data : List[np.ndarray]
            Preprocessed gait segments.
    
        Returns
        -------
        List[float]
            Forward speed (m/s) for each segment.
    
        Raises
        ------
        KeyError
            If 'pelvis_tx' is missing from joint mapping.
        """
        if 'pelvis_tx' not in self.jnt_name:
            raise KeyError("pelvis_tx index missing in joint mapping")
        tx_idx = self.jnt_name['pelvis_tx']
        dt = 1.0 / self.sample_frequency
    
        speeds = []
        for i, seg in enumerate(data):
            frames = seg.shape[1]
            if frames < 2:
                speeds.append(0.0)
                continue
            displacement = float(seg[tx_idx, -1] - seg[tx_idx, 0])
            duration = (frames - 1) * dt
            speeds.append(displacement / duration)
    
        self.logger.debug(f"Calculated segment speeds: {speeds}")
        return speeds
    
    # --- Trajectory processing and splicing ---
    def _process_step_trim_repeat(self, step: np.ndarray) -> np.ndarray:
        """
        Trim a gait segment to ~2/3 cycle, center lateral pelvis, then
        repeat and splice into a full reference trajectory.
    
        Steps
        -----
        1. Trim to ceil(2/3 * original_frames).
        2. Center pelvis lateral (tz) displacement.
        3. Call _repeat_step_with_translation to tile with translation continuity and splice‐smoothing.
        4. Apply a final Gaussian smoothing over the entire trajectory.
    
        Parameters
        ----------
        step : np.ndarray, shape (dofs, frames)
            One raw gait segment.
    
        Returns
        -------
        np.ndarray, shape (dofs, final_frames)
            The processed, repeated, and smoothed trajectory.
        """
        total = step.shape[1]
        cut = int(np.ceil(total * 2/3))
        trimmed = step[:, :cut].copy()
    
        if 'pelvis_tz' in self.jnt_name:
            tz = self.jnt_name['pelvis_tz']
            trimmed[tz, :] -= trimmed[tz, :].mean()
    
        repeated = self._repeat_step_with_translation(trimmed)
    
        sigma = self.smoothing_sigma if self.smoothing_sigma is not None else 3.0
        repeated = gaussian_filter1d(repeated, sigma=sigma, axis=1)
        self.logger.debug(f"Applied full‐trajectory Gaussian smoothing (σ={sigma:.2f})")

        return repeated
  
    def _repeat_step_with_translation(self, arr: np.ndarray) -> np.ndarray:
        """
        Tile a trimmed cycle N times, apply translation offsets for pelvis DOFs,
        smooth only the splice regions, and—if requested—crop to uniform length.
    
        Parameters
        ----------
        arr : np.ndarray, shape (dofs, cycle_frames)
            One trimmed gait cycle.
    
        Returns
        -------
        np.ndarray, shape (dofs, cycle_frames * repeat_times)
            The repeated, splice‐smoothed trajectory, optionally cropped so that
            all trajectories end up the same length.
        """
        N = self.repeat_times
        dofs, L = arr.shape
    
        trans_keys = ['pelvis_tz', 'pelvis_ty', 'pelvis_tx']
        t_idx = [self.jnt_name[k] for k in trans_keys if k in self.jnt_name]
    
        start = arr[t_idx, 0][:, None]    # shape (len(t_idx), 1)
        end   = arr[t_idx, -1][:, None]
        offsets = (end - start) @ np.arange(N)[None, :]  # shape (len(t_idx), N)
    
        tiled = np.tile(arr, (1, N))
        
        for i, idx in enumerate(t_idx):
            tiled[idx, :] += np.repeat(offsets[i], L)
    
        total = L * N
        mask = np.zeros(total, dtype=bool)
        ov = self.splice_overlap
        for c in range(1, N):
            b = c * L
            a = max(0, b - ov)
            d = min(total, b + ov)
            mask[a:d] = True
            
        sigma_splice = self.smoothing_sigma if self.smoothing_sigma is not None else 3.0
        sm = gaussian_filter1d(tiled, sigma=sigma_splice, axis=1)
        tiled[:, mask] = sm[:, mask]
        
        if getattr(self, 'uniform_length', False):
            desired = self.min_cycle_frames * N
            tiled = tiled[:, :desired]

        return tiled
    
    def _compute_velocity(self, pos: np.ndarray, method: str = "center") -> np.ndarray:
        """
        Compute joint velocities from positions via finite differences.
    
        Parameters
        ----------
        pos : np.ndarray, shape (dofs, frames)
            Joint position trajectory.
        method : {'forward', 'backward', 'center'}, default='center'
            Differencing scheme:
            - 'forward': forward difference, last column zero.
            - 'backward': backward difference, first column zero.
            - 'center': central difference with first/last approximated.
    
        Returns
        -------
        np.ndarray, shape (dofs, frames)
            Joint velocities.
    
        Raises
        ------
        ValueError
            If pos is not a 2D array or method is invalid.
        """
        if method not in {"forward", "backward", "center"}:
            raise ValueError("Invalid method for velocity computation.")
        # Validate input array
        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 2:
            raise ValueError(f"pos must be 2D array, got ndim={pos.ndim}")
    
        dt = 1.0 / self.sample_frequency
    
        if method == "center":
            vel = np.gradient(pos, dt, axis=1)
        elif method == "forward":
            diffs = np.diff(pos, axis=1) / dt
            vel = np.pad(diffs, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
        elif method == "backward":
            diffs = np.diff(pos, axis=1) / dt
            vel = np.pad(diffs, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
        else:
            raise ValueError(f"Invalid method '{method}' for velocity computation")
    
        return vel
    
class ReferenceTrajectories:
    """
    Lightweight cursor into trajectories cached by TrajectoryManager.

    Maintains:
      - traj_id: which trajectory in the manager to use
      - _pos:    current frame index within that trajectory
      - hold counter and end flag for hold_phase logic
    """
    def __init__(
        self,
        manager,
        traj_id: Optional[int] = None,
        increment: int = 2,
        max_hold_steps: int = 5,
        random_seed: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        """
        Bind to a TrajectoryManager and initialize pointers.
        
        Parameters
        ----------
        manager : TrajectoryManager
            The shared trajectories holder.
        traj_id : Optional[int], default=None
            Index of the trajectory to start with; if None, chosen randomly.
        increment : int, default=2
            Frames to advance on each next() call.
        max_hold_steps : int, default=5
            Max calls to next(hold_phase=True) before advancing.
        random_seed : Optional[int], default=None
            Seed for reproducible randomness.
        verbose : bool, default=False
            If True, enable DEBUG logging.
        """
        # logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

        # Shared manager and RNG
        self.manager = manager
        self.n_traj = len(manager)
        self.rng = np.random.default_rng(random_seed)
        self.jnt_name = self.manager.jnt_name
        self.sample_frequency = self.manager.sample_frequency

        # stepping config
        self.increment = increment
        self.max_hold_steps = max_hold_steps

        # initialize pointers
        self.set_trajectory(traj_id)

    def set_trajectory(self, traj_id: Optional[int] = None) -> None:
        """
        Switch to a given trajectory by ID, or choose randomly if None.
        Resets frame pointer and hold counter.
        """
        if traj_id is None:
            self.traj_id = int(self.rng.integers(self.n_traj))
        else:
            if not (0 <= traj_id < self.n_traj):
                raise IndexError(f"traj_id {traj_id} out of range [0, {self.n_traj-1}]")
            self.traj_id = traj_id

        # get read-only views (no copy)
        self.qpos, self.qvel = self.manager.get(self.traj_id)
        
        assert not self.qpos.flags.writeable and not self.qvel.flags.writeable, \
            "Underlying trajectory data must be read-only"
        
        self.traj_frames = self.qpos.shape[1]
        # assume cycle length stored or equal to full frames if unknown
        self.step_frames = self.traj_frames // self.manager.repeat_times
        self.speed = self.manager.get_speed(self.traj_id)
        self.foot_start = self.manager.get_foot_start(self.traj_id)  # 1=left, 0=right
    
        # reset state
        self._pos = 0
        self._hold_counter = 0
        self._has_reached_end = False
        
        self.logger.debug(
            f"Switched to trajectory {self.traj_id}: "
            f"frames={self.traj_frames}, cycle={self.step_frames}, "
            f"speed={self.speed:.2f}, foot_start={'left' if self.foot_start else 'right'}"
        )

    def reset(self,
              seed: Optional[int] = None,
              phase: float = 0.0, 
              randomize_traj: bool = True,
              traj_id: Optional[int] = None
              ) -> None:
        """
        Reinitialize this ReferenceTrajectories cursor.

        Parameters
        ----------
        seed : Optional[int]
            If provided, reseed the internal RNG (affects random phase/trajectory choice).
        traj_id : Optional[int]
            If provided, switch to this exact trajectory; otherwise:
              - if phase is None, pick a random trajectory;
              - if phase is not None, keep current trajectory.
        phase : Optional[float]
            If provided, must be in [0,100] and sets the exact gait phase;
            if None, a random phase in [0,100) is chosen.
    
        Raises
        ------
        TypeError
            If 'phase' or 'traj_id' have invalid types.
        ValueError
            If 'phase' is out of [0,100].
        IndexError
            If 'traj_id' is out of range.
        """
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError(f"seed must be int or None, got {type(seed)}")
            self.rng = np.random.default_rng(seed)
        
        if traj_id is not None:
            self.set_trajectory(traj_id)
        elif phase is None:
            self.set_trajectory(None)

        if phase is None:
            phase = float(self.rng.uniform(0.0, 100.0))
        else:
            if not isinstance(phase, (float, int)):
                raise TypeError(f"phase must be float or None, got {type(phase)}")
            phase = float(phase)
            if not (0.0 <= phase <= 100.0):
                raise ValueError("phase must be within [0,100]")

        if self.step_frames > 1:
            idx = int(round(phase / 100.0 * (self.step_frames - 1)))
        else:
            idx = 0
            
        self._pos = idx
        self._hold_counter = 0
        self._has_reached_end = (self._pos >= self.traj_frames - 1)
        self.logger.debug(
            f"reset(): traj_id={self.traj_id}, phase={phase:.2f}% -> frame={self._pos}, "
            f"has_reached_end={self._has_reached_end}"
        )

    def next(self, hold_phase: bool = False) -> None:
        """
        Advance the frame pointer by 'increment', or hold in place up to 'max_hold_steps'.
    
        Parameters
        ----------
        hold_phase : bool
            If True, delay advancement for at most 'max_hold_steps' calls; otherwise reset hold counter.
        """
        if hold_phase:
            self._hold_counter += 1
            if self._hold_counter < self.max_hold_steps:
                self.logger.debug(f"Holding at frame {self._pos} (hold count {self._hold_counter})")
                return
            
            self.logger.debug(f"Reached max hold ({self.max_hold_steps}) at frame {self._pos}, advancing")
            self._hold_counter = 0
        else:
            if self._hold_counter:
                self.logger.debug(f"Hold counter reset from {self._hold_counter} to 0")
            self._hold_counter = 0
        
        if self._has_reached_end:
            return
        
        self._pos += self.increment
        
        if self._pos >= self.traj_frames - 1:
            self._pos = self.traj_frames - 1
            self._has_reached_end = True
            self.logger.debug(f"Reached end of trajectory at frame {self._pos}")
        else:
            self.logger.debug(f"Advanced to frame {self._pos}")
    
    @property
    def phase(self) -> float:
        """
        Current phase percentage [0,100] within one cycle.
        """
        if self.step_frames <= 1:
            return 0.0
        cycle_idx = self._pos % self.step_frames
        return cycle_idx / (self.step_frames - 1) * 100.0

    @property
    def has_reached_end(self) -> bool:
        """
        True if the pointer is at the final frame.
        """
        return self._has_reached_end

    def get_qpos(self) -> np.ndarray:
        """
        Return the qpos vector at the current frame.
        """
        if not hasattr(self, 'qpos'):
            raise RuntimeError("Trajectory not initialized; call set_trajectory() first.")
        return self.qpos[:, self._pos]

    def get_qvel(self) -> np.ndarray:
        """
        Return the qvel vector at the current frame.
        """
        if not hasattr(self, 'qvel'):
            raise RuntimeError("Trajectory not initialized; call set_trajectory() first.")
        return self.qvel[:, self._pos]

    def get_reference_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (qpos, qvel) at the current frame.
        """
        return self.get_qpos(), self.get_qvel()
    
    def get_pelvis_ang(self):
        """
        Extract pelvis generalized coordinates:
        [lateral, vertical, forward translations, tilt, list, rotation] at current frame.

        Returns
        -------
        np.ndarray, shape (6,)
        """
        indices = [self.jnt_name['pelvis_tz'], self.jnt_name['pelvis_ty'], self.jnt_name['pelvis_tx'],
                   self.jnt_name['pelvis_tilt'], self.jnt_name['pelvis_list'], self.jnt_name['pelvis_rotation']]
        return self.qpos[indices, self._pos]
    
    def get_pelvis_angV(self):
        """
        Extract pelvis generalized velocities:
        [translational vx, vy, vz, angular rates tilt_dot, list_dot, rot_dot] at current frame.

        Returns
        -------
        np.ndarray, shape (6,)
        """
        indices = [self.jnt_name['pelvis_tz'], self.jnt_name['pelvis_ty'], self.jnt_name['pelvis_tx'],
                   self.jnt_name['pelvis_tilt'], self.jnt_name['pelvis_list'], self.jnt_name['pelvis_rotation']]
        return self.qvel[indices, self._pos]
    
    def get_torso_ang(self):
        """
        Get torso joint angles [lumbar_extension, lumbar_bending, lumbar_rotation] at current frame.

        Returns
        -------
        np.ndarray, shape (3,)
        """
        indices = [self.jnt_name['lumbar_extension'], self.jnt_name['lumbar_bending'], self.jnt_name['lumbar_rotation']]
        return self.qpos[indices, self._pos]
    
    def get_torso_angV(self):
        """
        Get torso angular velocity [lumbar_extension_dot, lumbar_bending_dot, lumbar_rotation_dot] at current frame.

        Returns
        -------
        np.ndarray, shape (3,)
        """
        indices = [self.jnt_name['lumbar_extension'], self.jnt_name['lumbar_bending'], self.jnt_name['lumbar_rotation']]
        return self.qvel[indices, self._pos]
    
    def get_joint_data(self, joint_group: Union[str, List[str]], data_type: str = "angle") -> np.ndarray:
        """
        Retrieve reference trajectory data (position or velocity) for specified joint(s) at the current frame.
    
        Parameters
        ----------
        joint_group : Union[str, List[str]]
            One of:
              - A group name ('pelvis' or 'torso') to fetch all joints in that group.
              - A single joint name.
              - A list of joint names.
        data_type : str, optional
            Type of data to return:
              - 'angle' (qpos) for joint positions.
              - 'velocity' (qvel) for joint angular velocities.
            Default is 'angle'.
    
        Returns
        -------
        np.ndarray
            1D array of length N (number of joints requested), containing the data at the current frame.
    
        Raises
        ------
        KeyError
            If any requested joint name is not found in self.jnt_name.
        ValueError
            If 'data_type' is not one of ['angle', 'velocity'].
        """
        joint_map = {
            "pelvis": ['pelvis_tz', 'pelvis_ty', 'pelvis_tx', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
            "torso": ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
        }
        # Determine the list of joint names
        if isinstance(joint_group, str):
            if joint_group in joint_map:
                names = joint_map[joint_group]
            else:
                names = [joint_group]
        elif isinstance(joint_group, list):
            names = joint_group
        else:
            raise ValueError("joint_group must be a str or a list of str")
        indices: List[int] = []
        for name in names:
            if name not in self.jnt_name:
                raise ValueError(f"Joint name '{name}' not found in mapping.")
            indices.append(self.jnt_name[name])
        # Extract the corresponding data
        if data_type == "angle":
            if self.qpos is None:
                raise RuntimeError("qpos data is not initialized.")
            return self.qpos[indices, self._pos]
        elif data_type == "velocity":
            if self.qvel is None:
                raise RuntimeError("qvel data is not initialized.")
            return self.qvel[indices, self._pos]
        else:
            raise ValueError("data_type must be 'angle' or 'velocity'.")
    
    