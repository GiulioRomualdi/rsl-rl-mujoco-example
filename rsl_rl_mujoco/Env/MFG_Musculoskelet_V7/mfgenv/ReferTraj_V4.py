import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import random
import logging
from typing import List, Optional, Tuple, Dict

# Configure logging for the module
logger = logging.getLogger(__name__)

class ReferenceTrajectories:
    """
    A class for processing reference gait trajectories for musculoskeletal simulation.
    
    Main functionality:
        - Loads gait data from a pickle file.
        - Preprocesses data (e.g. removes extra knee degrees of freedom).
        - Classifies steps into left-start and right-start based on initial hip flexion.
        - Optionally augments the dataset by mirroring each step (swapping left/right joints).
        - Selects a candidate step based on specified conditions (speed range, start side).
        - Trims the selected step to roughly one complete gait cycle (using 2/3 of the frames).
        - Repeats the trimmed step (with translation compensation) to build a continuous reference trajectory.
        - Applies Gaussian smoothing at splice regions.
        - Computes joint velocities using finite differences.
      
    New improvements:
        1. A "hold phase" mechanism in next() that allows the reference to pause (up to a maximum number of frames)
             when the agent falls behind – helping to mitigate early training reward degradation.
        2. Data mirroring: when enabled, each step is mirrored to generate an extra candidate by swapping left/right
             joints and flipping the sign of appropriate joint angles.
         
    Note: This implementation follows a DeepMimic-style fixed reference approach.
    """
    def __init__(self, 
                 data_path: str, 
                 repeat_times: int = 10, 
                 increment: int = 1,
                 is_fixed_speed: bool = True, 
                 is_left_start: bool = False, 
                 verbose: bool = True,
                 sample_frequency: int = 100,
                 knee_1dof: bool = True,
                 filter_enabled: bool = True,
                 speed_range: Optional[Tuple[float, float]] = None,
                 random_seed: Optional[int] = None,
                 smoothing_sigma: Optional[float] = None,
                 splice_overlap: Optional[int] = None,
                 max_hold_steps: int = 5,
                 enable_mirroring: bool = True,
                 **kwargs):
        """
        Initialize the ReferenceTrajectories object.
        
        Parameters:
            data_path (str): Path to the trajectory data file (pickle format).
            repeat_times (int): Number of times to repeat the selected step.
            increment (int): Frame increment for advancing the trajectory.
            is_fixed_speed (bool): Whether to filter candidate steps by a fixed speed range.
            is_left_start (bool): Whether to select steps that start with the left foot.
            verbose (bool): Verbosity flag.
            sample_frequency (int): Sampling frequency (Hz) of the trajectory data.
            knee_1dof (bool): If True, remove extra knee degrees of freedom.
            filter_enabled (bool): If True, apply Gaussian smoothing to the repeated trajectory.
            speed_range (Optional[Tuple[float, float]]): Speed range (low, high) for candidate selection.
            random_seed (Optional[int]): Random seed for reproducibility.
            smoothing_sigma (Optional[float]): Sigma for Gaussian smoothing.
            splice_overlap (Optional[int]): Number of overlapping frames at splice points.
            max_hold_steps (int): Maximum consecutive frames allowed to hold (pause) the phase.
            enable_mirroring (bool): If True, augment the dataset with mirrored copies of each step.
        """
        self.data_path: str = data_path
        self.repeat_times: int = repeat_times
        self.increment: int = increment
        self.is_fixed_speed: bool = is_fixed_speed
        self.is_left_start: bool = is_left_start
        self.verbose: bool = verbose
        self.sample_frequency: int = sample_frequency
        self.knee_1dof: bool = knee_1dof
        self.filter_enabled: bool = filter_enabled
        self.speed_range: Tuple[float, float] = speed_range if speed_range is not None else (1.2, 1.3)
        self.random_seed: Optional[int] = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        self.smoothing_sigma = smoothing_sigma if smoothing_sigma is not None else 3
        self.splice_overlap = splice_overlap if splice_overlap is not None else 10
        self.max_hold_steps = max_hold_steps
        self.enable_mirroring = enable_mirroring
        
        # Internal state variables
        self.qpos: Optional[np.ndarray] = None   # Processed joint positions (dofs x frames)
        self.qvel: Optional[np.ndarray] = None     # Processed joint velocities (dofs x frames)
        self.current_speed: Optional[float] = None  # Speed computed from the selected trajectory
        self.num_dofs: Optional[int] = None         # Number of degrees of freedom in the trajectory
        self.step_frames: Optional[int] = None      # Number of frames in the trimmed step
        self.traj_frames: Optional[int] = None      # Total frames in the repeated trajectory
        self._pos: int = 0                          # Current frame index in the trajectory
        self._has_reached_end: bool = False         # Flag indicating end of trajectory
        
        # Counter for "hold phase" mechanism
        self._hold_counter: int = 0
        
        # Joint mapping (for processed data)
        self.jnt_name: Dict[str, int] = {
            'pelvis_tz': 0, 'pelvis_ty': 1, 'pelvis_tx': 2, 'pelvis_tilt': 3, 'pelvis_list': 4, 'pelvis_rotation': 5,
            'hip_flexion_r': 6, 'hip_adduction_r': 7, 'hip_rotation_r': 8, 
            'knee_angle_r': 9, 'knee_angle_r_rotation2': 10, "knee_angle_r_rotation3": 11,'ankle_angle_r': 12, 'subtalar_angle_r': 13, 'mtp_angle_r': 14,
            'hip_flexion_l': 15, 'hip_adduction_l': 16, 'hip_rotation_l': 17, 
            'knee_angle_l': 18, 'knee_angle_l_rotation2': 19, 'knee_angle_l_rotation3': 20, 'ankle_angle_l': 21, 'subtalar_angle_l': 22, 'mtp_angle_l': 23,
            'lumbar_extension': 24, 'lumbar_bending': 25, 'lumbar_rotation': 26,
            'arm_flex_r': 27, 'arm_add_r': 28, 'arm_rot_r': 29, 'elbow_flex_r': 30, 'pro_sup_r': 31, 'wrist_flex_r': 32, 'wrist_dev_r': 33,
            'arm_flex_l': 34, 'arm_add_l': 35, 'arm_rot_l': 36, 'elbow_flex_l': 37, 'pro_sup_l': 38, 'wrist_flex_l': 39, 'wrist_dev_l': 40,
        }
        
        # Load and process the trajectory data.
        self._load_and_process_data()
        # Reset trajectory to initial phase (default 0%)
        self.reset(phase=0)
        logger.info("ReferenceTrajectories initialized successfully. Current phase: %.2f", self.phase)
    
    def _load_and_process_data(self) -> None:
        """Load raw data, validate it, and process it to create qpos and qvel."""
        try:
            raw_data = self._load_raw_data()
            self._validate_data(raw_data)
            if self.knee_1dof:
                raw_data = self._remove_extra_knee_dofs(raw_data)
            # Augment data by mirroring if enabled
            if self.enable_mirroring:
                raw_data = self._augment_with_mirroring(raw_data)
            left_steps, right_steps = self._classify_steps(raw_data)
            speeds = self._calculate_speeds(raw_data)
            self._select_and_process_trajectory(raw_data, left_steps, right_steps, speeds)
        except Exception as e:
            logger.error("Failed to load and process trajectory data: %s", e)
            raise e
                
    def _load_raw_data(self) -> List[np.ndarray]:
        """
        Load raw trajectory data from the pickle file.
        
        Returns:
            List[np.ndarray]: A list of trajectory steps, each as a 2D numpy array.
        """
        logger.info("Loading trajectory data from %s", self.data_path)
        try:
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError as e:
            logger.error("Data file not found: %s", self.data_path)
            raise e
        except Exception as e:
            logger.error("Error loading data from %s: %s", self.data_path, e)
            raise e
        
    def _validate_data(self, data: List[np.ndarray]) -> None:
        """
        Validate that the loaded data is a non-empty list of numpy arrays.
        
        Raises:
            ValueError: If data is empty or contains non-numpy array elements.
        """
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Loaded data is empty or not a list.")
        for step in data:
            if not isinstance(step, np.ndarray):
                raise ValueError("Data contains non-numpy array elements.")
    
    def _remove_extra_knee_dofs(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Remove extra knee degrees of freedom from each trajectory step.
        
        Parameters:
            data (List[np.ndarray]): A list of trajectory steps, each as a 2D numpy array.
        
        Returns:
            List[np.ndarray]: Processed data with reduced knee DOFs.
        """
        logger.info("Removing extra knee DOFs from trajectory data.")
        removal_keys = ['knee_angle_r_rotation2', 'knee_angle_r_rotation3',
                        'knee_angle_l_rotation2', 'knee_angle_l_rotation3']
        removal_indices = [self.jnt_name[key] for key in removal_keys if key in self.jnt_name]
        removal_indices.sort()
        processed_data = []
        for step in data:
            if step.ndim != 2:
                raise ValueError("Each step in data must be a 2D numpy array.")
            processed_step = np.delete(step, removal_indices, axis=0)
            processed_data.append(processed_step)
        self._update_joint_mapping(removal_indices)
        return processed_data
    
    def _update_joint_mapping(self, removal_indices: List[int]) -> None:
        """
        Update the joint mapping (self.jnt_name) after removing extra DOFs.
        
        Parameters:
            removal_indices (List[int]): Sorted list of indices that have been removed.
        """
        new_mapping = {}
        for joint, orig_index in self.jnt_name.items():
            if orig_index in removal_indices:
                continue
            
            offset = sum(1 for rem in removal_indices if rem < orig_index)
            new_mapping[joint] = orig_index - offset
        self.jnt_name = new_mapping
        logger.info("Updated joint mapping: %s", self.jnt_name)
    
    def _augment_with_mirroring(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Augment the dataset by mirroring each trajectory step.
        For each step, add its mirrored version to the dataset.

        Returns:
            List[np.ndarray]: Augmented data including original and mirrored steps.
        """
        augmented_data = []
        for step in data:
            augmented_data.append(step)
            mirrored_step = self._mirror_step(step)
            augmented_data.append(mirrored_step)
        logger.info("Data augmented with mirroring. Original count: %d, Augmented count: %d", len(data), len(augmented_data))
        return augmented_data
    
    def _mirror_step(self, step: np.ndarray) -> np.ndarray:
        """
        Mirror a given step trajectory.
        This function swaps left and right joint channels and inverts the sign of lateral pelvis components.

        Returns:
            np.ndarray: Mirrored step trajectory.
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
        for key_r, key_l in mirror_pairs.items():
            if key_r in self.jnt_name and key_l in self.jnt_name:
                idx_r = self.jnt_name[key_r]
                idx_l = self.jnt_name[key_l]
                temp = mirrored[idx_r, :].copy()
                mirrored[idx_r, :] = mirrored[idx_l, :]
                mirrored[idx_l, :] = temp
                
        if 'pelvis_tz' in self.jnt_name:
            mirrored[self.jnt_name['pelvis_tz'], :] *= -1
        if 'pelvis_list' in self.jnt_name:
            mirrored[self.jnt_name['pelvis_list'], :] *= -1
        if 'pelvis_rotation' in self.jnt_name:
            mirrored[self.jnt_name['pelvis_rotation'], :] *= -1
        if 'lumbar_bending' in self.jnt_name:
            mirrored[self.jnt_name['lumbar_bending'], :] *= -1
        if 'lumbar_rotation' in self.jnt_name:
            mirrored[self.jnt_name['lumbar_rotation'], :] *= -1
        return mirrored
    
    def _classify_steps(self, data: List[np.ndarray]) -> Tuple[List[int], List[int]]:
        """
        Classify steps into left-start and right-start based on initial hip flexion.
        
        Returns:
            Tuple[List[int], List[int]]: (left_indices, right_indices)
        """
        left_indices = self._determine_left_step_indices(data)
        right_indices = [i for i in range(len(data)) if i not in left_indices]
        logger.info("Found %d left-start steps and %d right-start steps.", len(left_indices), len(right_indices))
        return left_indices, right_indices
    
    def _determine_left_step_indices(self, data: List[np.ndarray]) -> List[int]:
        """
        Determine indices of steps that start with the left foot.
        
        Uses the initial values of 'hip_flexion_l' and 'hip_flexion_r' to classify.
        
        Returns:
            List[int]: Indices of left-start steps.
        """
        r_hip_flex_idx = self.jnt_name.get("hip_flexion_r")
        l_hip_flex_idx = self.jnt_name.get("hip_flexion_l")
        
        if r_hip_flex_idx is None or l_hip_flex_idx is None:
            raise KeyError("Missing hip flexion indices in joint mapping.")
        indices = []
        for i, step in enumerate(data):
            if step.shape[0] <= max(r_hip_flex_idx, l_hip_flex_idx):
                continue
            # If left hip flexion is greater than right hip flexion at the first frame, label as left-start.
            if step[l_hip_flex_idx, 0] > step[r_hip_flex_idx, 0]:
                indices.append(i)
        return indices
    
    def _calculate_speeds(self, data: List[np.ndarray]) -> List[float]:
        """
        Calculate the speed for each step in the trajectory.
        
        Speed is computed based on the displacement in the third DOF (assumed pelvis_tx)
        over the duration of the step.
        
        Returns:
            List[float]: Speed values for each step.
        """
        speeds = []
        forward_idx = self.jnt_name.get("pelvis_tx")
        for step in data:
            if step.shape[1] < 2:
                speeds.append(0.0)
            else:
                displacement = step[forward_idx, -1] - step[forward_idx, 0]
                duration = (step.shape[1] - 1) / self.sample_frequency
                speeds.append(displacement / duration if duration > 0 else 0.0)
        logger.info("Calculated speeds for steps: %s", speeds)
        return speeds
    
    def _select_and_process_trajectory(self,
                                       data: List[np.ndarray],
                                       left_steps: List[int],
                                       right_steps: List[int],
                                       speeds: List[float]) -> None:
        """
        Select a candidate gait segment based on starting foot and (optionally) speed range, then process the trajectory.
        
        The selection process is as follows:
            1. Filter segments by the desired starting foot (left or right).
            2. If is_fixed_speed is True, further filter the segments to only include those with a speed within the specified speed_range.
            3. If no segment satisfies the conditions, fallback to a random selection.
            4. Process the selected segment to generate the final reference trajectory.
    
        Parameters:
            data (List[np.ndarray]): List of gait segments.
            left_steps (List[int]): Indices of left-start gait segments.
            right_steps (List[int]): Indices of right-start gait segments.
            speeds (List[float]): Computed speed for each gait segment.
        """
        if self.is_left_start:
            candidate_indices = left_steps.copy()
        else:
            candidate_indices = right_steps.copy()
        logger.debug("Candidate indices based on starting foot: %s", candidate_indices)
        
        if self.is_fixed_speed:
            low_speed, high_speed = self.speed_range
            candidate_indices = [i for i in candidate_indices if low_speed <= speeds[i] <= high_speed]
        logger.debug("Candidates after speed filtering (range: %.2f - %.2f): %s", low_speed, high_speed, candidate_indices)
        
        if not candidate_indices:
            logger.warning("No valid candidate steps found under specified conditions; performing fallback selection.")
            self._fallback_to_random_step(data)
            return

        selected_index = random.choice(candidate_indices)
        logger.info("Selected step index: %d", selected_index)
        selected_step = data[selected_index]
        self._process_selected_step(selected_step)
        
    def _fallback_to_random_step(self, data: List[np.ndarray]) -> None:
        """
        Fallback: Randomly select any available step from the dataset.
        """
        if not data:
            raise ValueError("No data available for fallback selection.")
        selected_index = random.randint(0, len(data) - 1)
        selected_step = data[selected_index]
        logger.info("Fallback selection: step index %d", selected_index)
        self._process_selected_step(selected_step)
            
    def _process_selected_step(self, step: np.ndarray) -> None:
        """
        Process the selected step:
          - Trim the step to 2/3 of its original length.
          - Recenter the pelvis_tz channel so that its average value is 0.
          - Repeat the trimmed step with translation adjustments.
          - Optionally apply Gaussian smoothing.
          - Compute joint velocities from the repeated trajectory.
        
        Parameters:
            step (np.ndarray): The selected gait segment (2D array, dofs x frames).
        """
        original_frames = step.shape[1]
        cut_frames = int(np.ceil(original_frames * 2 / 3))
        trimmed_step = step[:, :cut_frames]
        
        if 'pelvis_tz' in self.jnt_name:
            idx = self.jnt_name['pelvis_tz']
            avg_val = np.mean(trimmed_step[idx, :])
            trimmed_step[idx, :] -= avg_val
        
        self.current_speed = (trimmed_step[2, -1] - trimmed_step[2, 0]) / ((trimmed_step.shape[1] - 1) / self.sample_frequency)
        self.num_dofs, self.step_frames = trimmed_step.shape

        # Repeat the step with translation adjustments.
        repeated_traj = self._repeat_step_with_translation(trimmed_step, self.repeat_times)
        
        # Apply Gaussian smoothing if enabled.
        if self.filter_enabled:
            sigma = max(min(10, int(np.ceil(3 / self.current_speed))), 3) if self.current_speed > 0 else 3
            repeated_traj = gaussian_filter1d(repeated_traj, sigma=sigma, axis=1)

        self.qpos = repeated_traj
        self.traj_frames = self.qpos.shape[1]
        self.qvel = self._compute_velocity(self.qpos, method="center")
    
    def _repeat_step_with_translation(self, step: np.ndarray, N: int, overlap: Optional[int] = None) -> np.ndarray:
        """
        Repeat the given step N times and adjust the translation DOFs to maintain continuity.
        
        Parameters:
            step (np.ndarray): The original step trajectory (dofs x frames).
            N (int): Number of repetitions.
            overlap (int): Number of overlapping frames for smoothing at splice points.
        
        Returns:
            np.ndarray: The repeated and concatenated trajectory.
        """
        if overlap is None:
            overlap = self.splice_overlap if hasattr(self, 'splice_overlap') else 10
        
        dofs, frames = step.shape
        repeated_traj = np.empty((dofs, frames * N))
    
        # Precompute translation offsets for pelvis DOFs
        pelvis_indices = [self.jnt_name[key] for key in ['pelvis_tz', 'pelvis_ty', 'pelvis_tx'] if key in self.jnt_name]
        offsets = np.zeros((len(pelvis_indices), N))
        for i in range(N):
            for j, idx in enumerate(pelvis_indices):
                offsets[j, i] = i * (step[idx, -1] - step[idx, 0])
    
        # Build repeated trajectory using vectorized addition for translation
        for i in range(N):
            start_idx = i * frames
            end_idx = start_idx + frames
            current_step = step.copy()
            for j, idx in enumerate(pelvis_indices):
                current_step[idx, :] += offsets[j, i]
            repeated_traj[:, start_idx:end_idx] = current_step

        # Smooth the splice regions
        for i in range(1, N):
            left = max(0, i * frames - overlap)
            right = min(repeated_traj.shape[1], i * frames + overlap)
            segment = repeated_traj[:, left:right]
            repeated_traj[:, left:right] = gaussian_filter1d(segment, sigma=self.smoothing_sigma if hasattr(self, 'smoothing_sigma') else 3, axis=1)
        
        return repeated_traj
    
    def _compute_velocity(self, angles: np.ndarray, method: str = "center") -> np.ndarray:
        """
        Compute joint velocities using finite differences.
        
        Parameters:
            angles (np.ndarray): Trajectory positions (dofs x frames).
            method (str): Finite difference method: 'forward', 'backward', or 'center'.
        
        Returns:
            np.ndarray: Computed velocities with the same shape as angles.
        """
        if method not in {"forward", "backward", "center"}:
            raise ValueError("Invalid method. Choose 'forward', 'backward', or 'center'.")
        delta_t = 1 / self.sample_frequency
        velocity = np.zeros_like(angles)
        if method == "forward":
            velocity[:, :-1] = (angles[:, 1:] - angles[:, :-1]) / delta_t
            velocity[:, -1] = 0
        elif method == "backward":
            velocity[:, 1:] = (angles[:, 1:] - angles[:, :-1]) / delta_t
            velocity[:, 0] = 0
        elif method == "center":
            velocity[:, 1:-1] = (angles[:, 2:] - angles[:, :-2]) / (2 * delta_t)
            velocity[:, 0] = 0
            velocity[:, -1] = 0
        return velocity
    
    def reset(self, phase: float = 0) -> None:
        """
        Reset the trajectory to a specified gait phase.
        
        Parameters:
            phase (float): Gait phase in percentage (0 to 100).
        
        Raises:
            ValueError: If phase is out of range or trajectory is not properly initialized.
        """
        if not (0 <= phase <= 100 ):
            raise ValueError("Phase must be between 0 and 100.")
        if self.step_frames is None:
            raise ValueError("Trajectory not initialized properly (step_frames is None).")
        self._pos = int(np.round(phase / 100 * (self.step_frames - 1)))
        self._has_reached_end = (self.traj_frames - self._pos - 1) < 1e-6
        self._hold_counter = 0
        logger.info("Trajectory reset: phase=%.2f, _pos=%d, has_reached_end=%s", phase, self._pos, self._has_reached_end)
    
    @property
    def phase(self) -> float:
        """
        Get the current gait phase as a percentage.
        """
        if self.step_frames is None:
            return 0.0
        return (self._pos % self.step_frames) / (self.step_frames - 1) * 100
    
    @property
    def has_reached_end(self) -> bool:
        """
        Indicates whether the repeated trajectory has reached its end.
        """
        return self._has_reached_end
    
    def next(self, hold_phase: bool = False) -> None:
        """
        Advance to the next frame in the trajectory.
        
        Parameters:
            hold_phase (bool): If True, hold the current phase (do not advance) to give the agent a grace period.
                                If the hold counter exceeds max_hold_steps, the trajectory is advanced anyway.
        """
        if hold_phase:
            self._hold_counter += 1
            if self._hold_counter < self.max_hold_steps:
                logger.info("Holding phase: current hold count %d (< %d)", self._hold_counter, self.max_hold_steps)
                return  # Do not advance
            else:
                logger.info("Hold count reached maximum (%d), forcing advancement.", self.max_hold_steps)
                self._hold_counter = 0
        else:
            self._hold_counter = 0
            
        if self._has_reached_end:
            logger.info("Trajectory has already reached the end. No further advancement.")
            return
        
        next_pos = self._pos + self.increment
        if next_pos >= self.traj_frames:
            # Clamp _pos to the last valid index (traj_frames - 1)
            self._pos = self.traj_frames - 1
            self._has_reached_end = True
            logger.info("Trajectory reached the end.")
        else:
            self._pos = next_pos
            # If _pos now equals the last valid index, mark as reached.
            if self._pos == self.traj_frames - 1:
                self._has_reached_end = True
                logger.info("Trajectory reached the end.")
    
    def get_qpos(self) -> np.ndarray:
        """
        Retrieve the current joint positions from the trajectory.
        
        Returns:
            np.ndarray: Joint positions at the current frame.
        
        Raises:
            ValueError: If trajectory data is not initialized.
        """
        if self.qpos is None:
            raise ValueError("qpos data is not initialized.")
        return self.qpos[:, self._pos]
    
    def get_qvel(self) -> np.ndarray:
        """
        Retrieve the current joint velocities from the trajectory.
        
        Returns:
            np.ndarray: Joint velocities at the current frame.
        
        Raises:
            ValueError: If trajectory data is not initialized.
        """
        if self.qvel is None:
            raise ValueError("qvel data is not initialized.")
        return self.qvel[:, self._pos]
    
    def get_reference_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the current reference joint positions and velocities.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (qpos, qvel) at the current frame.
        """
        return self.get_qpos(), self.get_qvel()
        
    def set_random_init_state(self, range_start: float = 0, range_end: float = 50, precision: int = 2) -> None:
        """
        Set a random initial gait phase within a specified range.
        
        Parameters:
            range_start (float): Lower bound of the gait phase percentage.
            range_end (float): Upper bound of the gait phase percentage.
            precision (int): Number of decimal places for the random phase.
        """
        if not (0 <= range_start <= range_end <= 100):
            raise ValueError("range_start and range_end must be between 0 and 100, and range_start <= range_end.")
        random_phase = round(np.random.uniform(range_start, range_end), precision)
        self.reset(phase=random_phase)
        logger.info("Random initial state set with phase: %.2f", random_phase)
        
    def set_deterministic_init_state(self, phase: float = 0) -> None:
        """
        Set the initial state deterministically to a specified phase.
        
        Parameters:
            phase (float): Gait phase percentage.
        """
        self.reset(phase)
        logger.info("Deterministic initial state set to phase: %.2f", phase)
        
    def get_pelvis_ang(self):
        """Get the generalized joint angle of the pelvis [tz, ty, tx, tilt, list, rotation]"""
        indices = [self.jnt_name['pelvis_tz'], self.jnt_name['pelvis_ty'], self.jnt_name['pelvis_tx'],
                   self.jnt_name['pelvis_tilt'], self.jnt_name['pelvis_list'], self.jnt_name['pelvis_rotation']]
        return self.qpos[indices, self._pos]
    
    def get_pelvis_angV(self):
        """Get the generalized joint angular velocity of the pelvis [tz, ty, tx, tilt, list, rotation]"""
        indices = [self.jnt_name['pelvis_tz'], self.jnt_name['pelvis_ty'], self.jnt_name['pelvis_tx'],
                   self.jnt_name['pelvis_tilt'], self.jnt_name['pelvis_list'], self.jnt_name['pelvis_rotation']]
        return self.qvel[indices, self._pos]
    
    def get_torso_ang(self):
        """Get the joint angle of the torso [lumbar_extension, _bending, _rotation]"""
        indices = [self.jnt_name['lumbar_extension'], self.jnt_name['lumbar_bending'], self.jnt_name['lumbar_rotation']]
        return self.qpos[indices, self._pos]
    
    def get_torso_angV(self):
        """Get the joint angular velocity of the torso [lumbar_extension, _bending, _rotation]"""
        indices = [self.jnt_name['lumbar_extension'], self.jnt_name['lumbar_bending'], self.jnt_name['lumbar_rotation']]
        return self.qvel[indices, self._pos]
    
    def get_joint_data(self, joint_group, data_type="angle"):
        """
        Get generalized joint data (angle or angular velocity) for a specified joint group.
        
        Parameters:
            joint_group (str): The joint group name ('pelvis' or 'torso').
            data_type (str): The type of data to return ('angle' or 'velocity').
            
        Returns:
            np.ndarray: The requested joint data.
            """
        joint_map = {
                "pelvis": ['pelvis_tz', 'pelvis_ty', 'pelvis_tx', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
                "torso": ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
                }

        if joint_group not in joint_map:
            raise ValueError(f"Invalid joint group: {joint_group}. Choose from {list(joint_map.keys())}.")
    
        indices = [self.jnt_name[joint] for joint in joint_map[joint_group]]
    
        if data_type == "angle":
            if self.qpos is None:
                raise ValueError("qpos is not initialized. Ensure data is loaded first.")
            return self.qpos[indices, self._pos]
        elif data_type == "velocity":
            if self.qvel is None:
                raise ValueError("qvel is not initialized. Ensure data is loaded first.")
            return self.qvel[indices, self._pos]
        else:
            raise ValueError(f"Invalid data type: {data_type}. Choose 'angle' or 'velocity'.")
    
    def plot_trajectory(self, dof_names: Optional[List[str]] = None) -> None:
        """
        Plot the trajectory of specified degrees of freedom (DOFs) for debugging purposes.
        
        Parameters:
            dof_names (Optional[List[str]]): List of DOF names to plot. Defaults to ['pelvis_tx'].
        
        Raises:
            ValueError: If any DOF name is not found in the joint mapping.
        """
        if dof_names is None:
            dof_names = ['pelvis_tx']
        for name in dof_names:
            if name not in self.jnt_name:
                raise ValueError(f"DOF {name} not found in joint mapping.")
        if self.qpos is None:
            raise ValueError("No qpos data to plot.")
        
        plt.figure()
        for name in dof_names:
            idx = self.jnt_name[name]
            plt.plot(self.qpos[idx, :], label=name)
        plt.title("Trajectory of DOFs")
        plt.xlabel("Frames")
        plt.ylabel("Position (rad or m)")
        plt.legend()
        plt.show()
            