'''
ReferenceTrajectories class for MuJoCo musculoskeletal model training.
@author: YAKE
'''
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import logging

class ReferenceTrajectories:
    """
    Manage cyclic reference trajectories for gait training in MuJoCo environments.

    Responsibilities:
      - Load and validate raw gait data from a pickle file.
      - Optionally remove redundant knee DoFs and augment via left-right mirroring.
      - Classify and select a single gait segment based on starting foot and speed range.
      - Trim to one full cycle, repeat with translation continuity, and smooth the splices.
      - Provide runtime interface to advance the trajectory (`next()`), reset to a phase, and fetch qpos/qvel.
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
            increment: int = 1,
            is_fixed_speed: bool = True,
            is_left_start: bool = False,
            sample_frequency: int = 100,
            knee_1dof: bool = True,
            smoothing_sigma: Optional[float] = None,
            splice_overlap: int = 10,
            max_hold_steps: int = 5,
            speed_range: Tuple[float, float] = (1.2, 1.3),
            enable_mirroring: bool = True,
            random_seed: Optional[int] = None,
            verbose: bool = False
            ) -> None:
        """
        Parameters
        ----------
        data_path : Union[str, Path]
            Path to pickle file containing List[np.ndarray] of shape (dofs, frames).
        repeat_times : int, default=10
            How many times to tile the trimmed gait cycle.
        increment : int, default=1
            Frame step size on each `next()` call.
        is_fixed_speed : bool, default=True
            Whether to filter segments by a fixed speed_range.
        is_left_start : bool, default=False
            Whether to only consider segments that start on the left foot.
        sample_frequency : int, default=100
            Sampling rate (Hz) of the raw trajectory data.
        knee_1dof : bool, default=True
            If True, remove extra knee DoFs before processing.
        smoothing_sigma : Optional[float], default=None
            Standard deviation for Gaussian smoothing; if None, auto‐computed from speed.
        splice_overlap : int, default=10
            Number of frames to overlap when smoothing splices.
        max_hold_steps : int, default=5
            Max consecutive frames to hold when `next(hold_phase=True)`.
        speed_range : Tuple[float, float], default=(1.2, 1.3)
            Allowed [min, max] speed (m/s) for selecting segments.
        enable_mirroring : bool, default=True
            Whether to augment data via left-right mirroring.
        random_seed : Optional[int], default=None
            Seed for reproducible randomness.
        verbose : bool, default=False
            If True, enable DEBUG-level logging.
        """
        # Validate inputs
        if repeat_times < 1 or increment < 1:
            raise ValueError("repeat_times and increment must be >= 1")
        if sample_frequency <= 0:
            raise ValueError("sample_frequency must be positive")
        if max_hold_steps < 1:
            raise ValueError("max_hold_steps must be >= 1")
        if speed_range[0] > speed_range[1]:
            raise ValueError("speed_range must be (low, high) with low <= high")
        
        # configure logger level based on verbosity
        self.logger = logging.getLogger(__name__)
        level = logging.DEBUG if verbose else logging.WARNING
        self.logger.setLevel(level)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            self.logger.addHandler(ch)
        
        # initialize independent RNG for all random operations
        self.rng: np.random.Generator = np.random.default_rng(random_seed)
        
        # Assign parameters
        self.data_path = Path(data_path)
        self.repeat_times = repeat_times
        self.increment = increment
        self.is_fixed_speed = is_fixed_speed
        self.is_left_start = is_left_start
        self.sample_frequency = sample_frequency
        self.knee_1dof = knee_1dof
        self.smoothing_sigma = smoothing_sigma
        self.splice_overlap = splice_overlap
        self.max_hold_steps = max_hold_steps
        self.speed_range = speed_range
        self.enable_mirroring = enable_mirroring
        
        # Internal state
        self.jnt_name = self.JOINT_MAP.copy()
        self.qpos: Optional[np.ndarray] = None
        self.qvel: Optional[np.ndarray] = None
        self.step_frames: int = 0
        self.traj_frames: int = 0
        self.current_speed: float = 0.0
        self._pos: int = 0
        self._hold_counter: int = 0
        self._has_reached_end: bool = False
        
        # Load and prepare data
        self._load_and_process_data()
        self.reset(phase=0.0)
        self.logger.info("Initialized ReferenceTrajectories at phase %.2f%%", self.phase)
    
    # --- Data loading and validation ---
    def _load_and_process_data(self) -> None:
        """
        Load raw gait data, validate its format, optionally remove extra knee DoFs,
        augment via mirroring, classify steps by foot and speed, then select and process
        a single reference trajectory into qpos and qvel.

        Pipeline steps:
          1. _load_raw_data
          2. _validate_data
          3. _remove_extra_knee_dofs (if enabled)
          4. _augment_with_mirroring (if enabled)
          5. _classify_steps
          6. _calculate_speeds
          7. _select_and_process_trajectory

        Raises
        ------
        FileNotFoundError
            If the pickle file is missing.
        ValueError
            If the data list is empty or improperly formatted.
        """
        try:
            raw_data = self._load_raw_data()
            self._validate_data(raw_data)
            if self.knee_1dof:
                raw_data = self._remove_extra_knee_dofs(raw_data)
            if self.enable_mirroring:
                raw_data = self._augment_with_mirroring(raw_data)
            left_steps, right_steps = self._classify_steps(raw_data)
            speeds = self._calculate_speeds(raw_data)
            self._select_and_process_trajectory(raw_data, left_steps, right_steps, speeds)
        except Exception as e:
            self.logger.error("Failed to load/process trajectory data: %s", e)
            raise
    
    def _load_raw_data(self) -> List[np.ndarray]:
        """
        Load the raw list of trajectory segments from a pickle file.

        Returns
        -------
        List[np.ndarray]
            Each element is a 2D array (dofs × frames) for one gait segment.

        Raises
        ------
        FileNotFoundError
            If the data_path does not exist.
        IOError
            If the file cannot be unpickled.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        try:
            with open(self.data_path, 'rb') as f:
                self.logger.debug("Loaded raw data from {self.data_path}.")
                return pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to read pickle file: {e}")
    
    def _validate_data(self, data: List[np.ndarray]):
        """
        Verify that `data` is a nonempty list of 2D numpy arrays.

        Raises
        ------
        ValueError
            If `data` is not a list, is empty, or contains non‐ndarray or non‐2D elements.
        """
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Loaded trajectory data is empty or not a list")
        for step in data:
            if not isinstance(step, np.ndarray):
                raise ValueError("Trajectory data contains non-numpy elements")
            if step.ndim != 2:
                raise ValueError("Each trajectory step must be a 2D numpy array (DOFs x frames)")
    
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
            Processed segments with fewer rows.

        Side Effects
        ------------
        self.jnt_name is updated to reflect new indices.
        """
        removal_keys = ['knee_angle_r_rotation2', 'knee_angle_r_rotation3',
                        'knee_angle_l_rotation2', 'knee_angle_l_rotation3']
        removal_indices = [self.jnt_name[k] for k in removal_keys if k in self.jnt_name]
        if not removal_indices:
            return data
        removal_indices.sort()
        # Delete the corresponding row in each track
        processed_data = [np.delete(step, removal_indices, axis=0) for step in data]
        # Update joint index mapping
        self._update_joint_mapping(removal_indices)
        self.logger.debug("Removed knee DOFs: %s", removal_keys)
        return processed_data
    
    def _update_joint_mapping(self, removal_indices: List[int]):
        """
        Update self.jnt_name after removing DOFs at `removal_indices`.

        Parameters
        ----------
        removal_indices : List[int]
            Sorted list of row indices that were deleted.
        """
        new_map: Dict[str, int] = {}
        for joint, idx in self.jnt_name.items():
            if idx in removal_indices:
                continue
            # Calculate how many deleted indexes are before the current index to determine the offset
            offset = sum(1 for rem in removal_indices if rem < idx)
            new_map[joint] = idx - offset
        self.jnt_name = new_map
        self.logger.debug("Joint mapping updated: %s", self.jnt_name)
    
    # --- Data enhancement: mirror gait ---
    def _augment_with_mirroring(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Augment the dataset by appending a left‐right mirrored copy of each segment.

        Parameters
        ----------
        data : List[np.ndarray]
            Original segments.

        Returns
        -------
        List[np.ndarray]
            Original plus mirrored segments.
        """
        augmented: List[np.ndarray] = []
        for step in data:
            augmented.append(step)
            mirrored = self._mirror_step(step)
            augmented.append(mirrored)
        self.logger.info("Data mirrored: original %d -> augmented %d", len(data), len(mirrored))
        return augmented
    
    def _mirror_step(self, step: np.ndarray) -> np.ndarray:
        """
        Create a mirrored version of one gait segment by swapping left/right joint channels
        and flipping sign of lateral axes.

        Parameters
        ----------
        step : np.ndarray, shape (dofs, frames)
            Original segment.

        Returns
        -------
        np.ndarray
            Mirrored segment.
        """
        mirrored = step.copy()
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
        # Exchange of data for paired joints
        for r, l in mirror_pairs.items():
            if r in self.jnt_name and l in self.jnt_name:
                ir, il = self.jnt_name[r], self.jnt_name[l]
                mirrored[ir, :] = step[il, :]
                mirrored[il, :] = step[ir, :]
        # Flip symbols for a specific axis
        for axis in ['pelvis_tz', 'pelvis_list', 'pelvis_rotation', 'lumbar_bending', 'lumbar_rotation']:
            if axis in self.jnt_name:
                mirrored[self.jnt_name[axis], :] *= -1
        return mirrored
    
    # --- Gait classification and selection ---
    def _classify_steps(self, data: List[np.ndarray]) -> Tuple[List[int], List[int]]:
        """
        Split indices of `data` into left‐start vs right‐start based on initial hip flexion.

        Parameters
        ----------
        data : List[np.ndarray]

        Returns
        -------
        left_ids : List[int]
            Indices where left hip flexion > right at frame 0.
        right_ids : List[int]
            The complement of left_ids.
        """
        left_ids = self._determine_left_step_indices(data)
        right_ids = [i for i in range(len(data)) if i not in left_ids]
        self.logger.info("Identified %d left-start and %d right-start steps.", len(left_ids), len(right_ids))
        return left_ids, right_ids
    
    def _determine_left_step_indices(self, data: List[np.ndarray]) -> List[int]:
        """
        Identify indices where the initial left hip flexion angle exceeds the right.

        Parameters
        ----------
        data : List[np.ndarray]

        Returns
        -------
        List[int]
            Subset of indices for left‐start segments.

        Raises
        ------
        KeyError
            If hip flexion joint indices are missing.
        """
        l_idx = self.jnt_name.get("hip_flexion_l")
        r_idx = self.jnt_name.get("hip_flexion_r")
        if l_idx is None or r_idx is None:
            raise KeyError("Hip flexion indices not found in joint mapping.")
        left_ids: List[int] = []
        for i, step in enumerate(data):
            if step.shape[0] <= max(l_idx, r_idx):
                continue
            if step[l_idx, 0] > step[r_idx, 0]:
                left_ids.append(i)
        return left_ids
    
    def _calculate_speeds(self, data: List[np.ndarray]) -> List[float]:
        """
        Compute forward speed of each segment via pelvis_tx displacement.

        Parameters
        ----------
        data : List[np.ndarray]

        Returns
        -------
        List[float]
            Speed in m/s for each segment.
        """
        speeds: List[float] = []
        idx = self.jnt_name.get("pelvis_tx")
        for step in data:
            frames = step.shape[1]
            if frames < 2:
                speeds.append(0.0)
            else:
                disp = float(step[idx, -1] - step[idx, 0])
                speeds.append(disp / ((frames - 1) / self.sample_frequency))
        self.logger.debug("Calculated speeds for steps: %s", speeds)
        return speeds
    
    def _select_and_process_trajectory(self, 
                                       data: List[np.ndarray], 
                                       left_steps: List[int],
                                       right_steps: List[int], 
                                       speeds: List[float]) -> None:
        """
        Filter candidate segments by start foot and (optionally) speed_range,
        then choose one (random or fallback) and process it into the final trajectory.

        Parameters
        ----------
        data : List[np.ndarray]
            All segments.
        left_steps : List[int]
            Indices of left‐start segments.
        right_steps : List[int]
            Indices of right‐start segments.
        speeds : List[float]
            Computed speeds per segment.

        Raises
        ------
        ValueError
            If no segments at all are available.
        """
        candidates = left_steps.copy() if self.is_left_start else right_steps.copy()
        self.logger.debug("Candidate indices based on foot preference: %s", candidates)
        # If the speed range is defined, the candidates that are not within the range are screened
        if self.is_fixed_speed:
            low, high = self.speed_range
            candidates = [i for i in candidates if low <= speeds[i] <= high]
            self.logger.debug("Candidates after speed filtering (%.2f-%.2f m/s): %s", low, high, candidates)
        if not candidates:
            self.logger.warning("No step matches the given criteria; selecting a random step as fallback.")
            self._fallback_to_random_step(data)
        else:
            idx = self.rng.choice(candidates)
            self.logger.debug("Selected step index %d for reference trajectory.", idx)
            self._process_selected_step(data[idx])
    
    def _fallback_to_random_step(self, data: List[np.ndarray]):
        """
        Randomly select any segment and process it, used when no filtered candidates remain.

        Parameters
        ----------
        data : List[np.ndarray]
        """
        if not data:
            raise ValueError("No trajectory data available for selection.")
        idx = self.rng.integers(len(data))
        self.logger.debug("Fallback: randomly selected step index %d.", idx)
        self._process_selected_step(data[idx])
    
    # --- Trajectory processing and splicing ---
    def _process_selected_step(self, step: np.ndarray) -> None:
        """
        Convert the chosen raw segment into a repeated, translated, and smoothed trajectory:
          1. Trim to ~2/3 cycle.
          2. Center pelvis lateral displacement.
          3. Compute current_speed.
          4. Repeat with translation continuity.
          5. Smooth splices.
          6. Compute qvel by finite‐difference.

        Parameters
        ----------
        step : np.ndarray, shape (dofs, frames)
        """
        # Trim to ~2/3 cycle
        total_frames = step.shape[1]
        cut_frames = int(np.ceil(total_frames * 2 / 3))
        trimmed = step[:, :cut_frames].copy()
        # Center pelvis lateral
        if 'pelvis_tz' in self.jnt_name:
            tz_idx = self.jnt_name['pelvis_tz']
            tz_mean = np.mean(trimmed[tz_idx, :])
            trimmed[tz_idx, :] -= tz_mean
        # Compute current speed
        forward_idx = self.jnt_name.get('pelvis_tx', 2)
        if trimmed.shape[1] > 1:
            disp = trimmed[forward_idx, -1] - trimmed[forward_idx, 0]
            duration = (trimmed.shape[1] - 1) / self.sample_frequency
            self.current_speed = disp / duration if duration > 0 else 0.0
        else:
            self.current_speed = 0.0
        self.step_frames = trimmed.shape[1]
        # Repeat with translation
        repeated = self._repeat_step_with_translation(trimmed)
        # Smooth if requested
        sigma = self.smoothing_sigma
        if sigma is None:
            # Adaptive sigma selection: smaller sigma for faster speeds, larger sigma for slower speeds (range 3 to 10)
            if self.current_speed and self.current_speed > 0:
                sigma = max(3.0, min(10.0, float(np.ceil(3.0 / self.current_speed))))
            else:
                sigma = 3.0
        repeated = gaussian_filter1d(repeated, sigma=sigma, axis=1)
        self.logger.debug("Applied Gaussian smoothing with sigma=%.2f to full trajectory.", sigma)
        # Setting qpos and calculating the corresponding qvel
        self.qpos = repeated
        # import ipdb;ipdb.set_trace()
        self.traj_frames = self.qpos.shape[1]
        self.qvel = self._compute_velocity(self.qpos, method="center")
    
    def _repeat_step_with_translation(self, arr: np.ndarray) -> np.ndarray:
        """
        Repeat a trimmed gait cycle to build a continuous reference trajectory with smooth splices.
    
        This method:
          1. Tiles the input segment `repeat_times` times along the time axis.
          2. Computes per-cycle translation offsets for pelvis DoFs (tz, ty, tx) to maintain continuity.
          3. Applies Gaussian smoothing over an overlap window (`splice_overlap`) at each splice.
    
        Parameters
        ----------
        arr : np.ndarray
            Single gait cycle of shape (dofs, frames).
    
        Returns
        -------
        np.ndarray
            The full repeated trajectory of shape (dofs, frames * repeat_times).
    
        Raises
        ------
        ValueError
            If `arr` is not a 2D array.
        """
        N = self.repeat_times
        dofs, frames = arr.shape
        # Identify translation indices
        trans_keys = ['pelvis_tz', 'pelvis_ty', 'pelvis_tx']
        t_idx = [self.jnt_name[k] for k in trans_keys if k in self.jnt_name]
        # Compute offsets
        start = arr[t_idx, 0][:, None]
        end = arr[t_idx, -1][:, None]
        diff = end - start
        offsets = diff @ np.arange(N)[None, :]
        # Tile and apply offsets
        tiled = np.tile(arr, (1, N))
        for j, idx in enumerate(t_idx):
            tiled[idx] += np.repeat(offsets[j], frames)
        # Construct overlap mask
        total = frames * N
        mask = np.zeros(total, dtype=bool)
        ov = self.splice_overlap
        for i in range(1, N):
            a = max(0, i * frames - ov)
            b = min(i * frames + ov, total)
            mask[a:b] = True
        # Overlap smoothing
        sigma = self.smoothing_sigma if self.smoothing_sigma is not None else 3.0
        smoothed = gaussian_filter1d(tiled, sigma=sigma, axis=1)
        tiled[:, mask] = smoothed[:, mask]

        return tiled
    
    def _compute_velocity(self, pos: np.ndarray, method: str = "center") -> np.ndarray:
        """
        Compute joint velocity via finite difference.

        Parameters
        ----------
        pos : np.ndarray, shape (dofs, frames)
            Trajectory positions.
        method : {'forward','backward','center'}
            Differencing scheme.

        Returns
        -------
        np.ndarray, shape (dofs, frames)
            Joint velocities.

        Raises
        ------
        ValueError
            If `method` is invalid.
        """
        if method not in {"forward", "backward", "center"}:
            raise ValueError("Invalid method for velocity computation.")
        dt = 1.0 / self.sample_frequency
        vel = np.zeros_like(pos)
        if method == "forward":
            if pos.shape[1] > 1:
                vel[:, :-1] = (pos[:, 1:] - pos[:, :-1]) / dt
            vel[:, -1] = 0.0
        elif method == "backward":
            if pos.shape[1] > 1:
                vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / dt
            vel[:, 0] = 0.0
        else:  # "center"
            if pos.shape[1] > 2:
                vel[:, 1:-1] = (pos[:, 2:] - pos[:, :-2]) / (2 * dt)
            if pos.shape[1] > 1:
                vel[:, 0] = vel[:, -1] = 0.0
        return vel
    
    # --- Runtime Control Interface ---
    def reset(self, phase: float = 0.0):
        """
        Reset internal frame pointer to the specified gait phase.

        Parameters
        ----------
        phase : float
            Gait cycle percentage [0,100].

        Raises
        ------
        ValueError
            If phase not in [0,100] or trajectory uninitialized.
        """
        if not (0.0 <= phase <= 100.0):
            raise ValueError("Phase must be between 0 and 100.")
        if self.step_frames is None or self.traj_frames is None:
            raise RuntimeError("Trajectory data is not initialized.")
        idx = int(np.round(phase / 100.0 * (self.step_frames - 1))) if self.step_frames > 1 else 0
        self._pos = idx
        self._has_reached_end = (self._pos >= self.traj_frames - 1)
        self._hold_counter = 0
        self.logger.debug("Trajectory reset to phase %.2f%% (frame index %d).", phase, self._pos)
    
    @property
    def phase(self) -> float:
        """
        Get the current gait phase as a percentage [0,100] of one cycle.
        """
        if self.step_frames is None or self.step_frames <= 1:
            return 0.0
        cycle_idx = self._pos % self.step_frames
        return (cycle_idx / (self.step_frames - 1)) * 100.0
    
    @property
    def has_reached_end(self) -> bool:
        """
        True if the internal frame pointer is at the final trajectory frame.
        """
        return self._has_reached_end
    
    def next(self, hold_phase: bool = False) -> None:
        """
        Advance the trajectory by `increment` frames, or hold if requested.

        Parameters
        ----------
        hold_phase : bool
            If True, delay advancement up to `max_hold_steps` to give agent grace.
        """
        if hold_phase:
            self._hold_counter += 1
            if self._hold_counter < self.max_hold_steps:
                self.logger.debug("Holding phase at frame %d (hold count %d).", self._pos, self._hold_counter)
                return
            else:
                self.logger.debug("Max hold count reached at frame %d, advancing frame.", self._pos)
                self._hold_counter = 0
        else:
            self._hold_counter = 0
        if self._has_reached_end:
            self.logger.warning("Trajectory has already reached the end; cannot advance.")
            return
        next_pos = self._pos + self.increment
        if next_pos >= self.traj_frames:
            self._pos = self.traj_frames - 1
            self._has_reached_end = True
            self.logger.debug("Reached end of trajectory at frame %d.", self._pos)
        else:
            self._pos = next_pos
            if self._pos >= self.traj_frames - 1:
                self._has_reached_end = True
                self.logger.debug("Reached end of trajectory at frame %d.", self._pos)
                
    def set_random_init_state(self, range_start: float = 0, range_end: float = 100, precision: int = 2) -> None:
        """
        Randomly choose an initial phase in [range_start, range_end] and reset to it.

        Parameters
        ----------
        range_start : float
            Lower bound of percent.
        range_end : float
            Upper bound of percent.
        precision : int
            Decimal places for sampling.

        Raises
        ------
        ValueError
            If bounds invalid.
        """
        if not (0 <= range_start <= range_end <= 100):
            raise ValueError("range_start and range_end must be between 0 and 100, and range_start <= range_end.")
        random_phase = round(self.rng.uniform(range_start, range_end), precision)
        self.reset(phase=random_phase)
        self.logger.info("Random initial state set with phase: %.2f", random_phase)
        
    def set_deterministic_init_state(self, phase: float = 0) -> None:
        """
        Reset trajectory to exactly the given phase percentage.

        Parameters
        ----------
        phase : float
            Gait cycle percentage in [0,100].
        """
        self.reset(phase)
        self.logger.info("Deterministic initial state set to phase: %.2f", phase)
    
    # --- Data Access Interface ---
    def get_qpos(self) -> np.ndarray:
        """
        Return the generalized joint positions (qpos) for the current frame.

        Returns
        -------
        np.ndarray
            qpos vector of length dofs.

        Raises
        ------
        RuntimeError
            If trajectory not initialized.
        """
        if self.qpos is None:
            raise RuntimeError("qpos data is not initialized.")
        return self.qpos[:, self._pos]
    
    def get_qvel(self) -> np.ndarray:
        """
        Return the generalized joint velocities (qvel) for the current frame.

        Returns
        -------
        np.ndarray
            qvel vector of length dofs.

        Raises
        ------
        RuntimeError
            If trajectory not initialized.
        """
        if self.qvel is None:
            raise RuntimeError("qvel data is not initialized.")
        return self.qvel[:, self._pos]
    
    def get_reference_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the full current frame's qpos and qvel as a tuple.

        Returns
        -------
        (qpos, qvel) : Tuple[np.ndarray, np.ndarray]
            Each of shape (dofs,).
        """
        if self.qpos is None or self.qvel is None:
            raise RuntimeError("Trajectory data is not initialized.")
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
            If `data_type` is not one of ['angle', 'velocity'].
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
    
    # --- Visualization ---
    def plot_trajectory(self, 
                        joints: List[str],
                        frames: Optional[Tuple[int, int]] = None
                        ) -> None:
        """
        Plot reference joint positions over time for selected joints.
    
        Parameters
        ----------
        joints : List[str]
            Names of joints to plot. If the list is empty, defaults to ['pelvis_tx'].
        frames : Optional[Tuple[int, int]], optional
            A (start_frame, end_frame) pair to restrict the plotted interval.
            If None, the entire trajectory is plotted.
    
        Raises
        ------
        RuntimeError
            If trajectory data (self.qpos) is not initialized.
        """
        if not hasattr(self, 'qpos') or self.qpos is None:
            self.logger.error("qpos has not been initialized and cannot be plotted.")
            return
        
        start, end = (0, self.traj_frames) if frames is None else frames
        t = np.arange(start, end) / self.sample_frequency

        for name in joints:
            idx = self.jnt_name.get(name)
            if idx is None:
                self.logger.warning(f"Unknown joint: {name}, skipped.")
                continue
            data = self.qpos[idx, start:end]
            plt.plot(t, data, label=name)

        plt.xlabel('Time (s)')
        plt.ylabel('Joint value')
        plt.title('Reference Trajectory')
        plt.legend()
        plt.grid(True)
        plt.show()

