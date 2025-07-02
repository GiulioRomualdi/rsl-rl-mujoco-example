# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:41:50 2024

@author: YAKE
"""

import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import random
    
class ReferenceTrajectories:
    def __init__(self, 
                 data_path: str, 
                 repeat_times: int = 50, 
                 is_fixed_speed: bool = True, 
                 is_left_start: bool = False, 
                 verbose: bool = True,
                 sample_frequency: int = 100,
                 knee_1dof: bool = True,
                 filter_enabled: bool = True,
                 increment: int = 1):
        """
        Initialize the reference trajectory class with parameters for loading and processing the gait data.

        Parameters:
            data_path (str): Path to the trajectory data file.
            repeat_times (int): Number of times the gait cycle is repeated, must be > 0.
            is_fixed_speed (bool): Whether to use fixed speed for selecting steps.
            is_left_start (bool): Whether to select steps that start with the left foot.
            verbose (bool): Whether to print additional debug information.
            sample_frequency (int): Sampling frequency (Hz) for the trajectory data.
            knee_1dof (bool): If True, remove extra knee DOFs from data (some datasets have more).
            filter_enabled (bool): Whether to apply Gaussian filtering.
            increment (int): The step increment used in `next()`. Default is 1.
        
        Raises:
            TypeError: If data_path is not a string.
            ValueError: If repeat_times <= 0.
            ValueError: If no valid trajectory is available even after fallback.
        """
        if not isinstance(data_path, str):
            raise TypeError("data_path must be a string.")
        if repeat_times <= 0:
            raise ValueError("repeat_times must be a positive integer.")
        if not isinstance(sample_frequency, int) or sample_frequency <= 0:
            raise ValueError("sample_frequency must be a positive integer.")
        if not isinstance(increment, int) or increment <= 0:
            raise ValueError("increment must be a positive integer.")
            
        self.data_path = data_path
        self.repeat_times = repeat_times
        self.is_fixed_speed = is_fixed_speed
        self.is_left_start = is_left_start
        self.verbose = verbose
        self.sample_frequency = sample_frequency
        self.knee_1dof = knee_1dof
        self.filter_enabled = filter_enabled
        self.increment = increment
        
        # Initialize state variables
        self.qpos = None 
        self.qvel = None
        self.current_speed = None
        self.num_dofs = None
        self.step_frames = None
        self.traj_frames = None
        self._pos = 0
        self.phase = 0
        self.has_reached_the_end = False
        
        self.jnt_name = {
            'pelvis_tz': 0, 'pelvis_ty': 1, 'pelvis_tx': 2, 'pelvis_tilt': 3, 'pelvis_list': 4, 'pelvis_rotation': 5,
            'hip_flexion_r': 6, 'hip_adduction_r': 7, 'hip_rotation_r': 8, 
            'knee_angle_r': 9, 'ankle_angle_r': 10, 'subtalar_angle_r': 11, 'mtp_angle_r': 12,
            'hip_flexion_l': 13, 'hip_adduction_l': 14, 'hip_rotation_l': 15, 
            'knee_angle_l': 16, 'ankle_angle_l': 17, 'subtalar_angle_l': 18, 'mtp_angle_l': 19,
            'lumbar_extension': 20, 'lumbar_bending': 21, 'lumbar_rotation': 22,
            'arm_flex_r': 23, 'arm_add_r': 24, 'arm_rot_r': 25, 'elbow_flex_r': 26, 'pro_sup_r': 27, 'wrist_flex_r': 28, 'writs_dev_r': 29,
            'arm_flex_l': 30, 'arm_add_l': 31, 'arm_rot_l': 32, 'elbow_flex_l': 33, 'pro_sup_l': 34, 'wrist_flex_l': 35, 'writs_dev_l': 36
        }
        
        # Load trajectory data
        self.load_data()
        
        if self.qpos is None:
            if self.verbose:
                print("Warning: No suitable step found under the specified condition. Attempting fallback...")
            self.fallback_to_random_step()
            
        if self.qpos is None:
            raise ValueError("No reference trajectory data available after fallback. Check dataset or conditions.")
            
        self.phase = self.get_phase_variable()
        if self.verbose:
            print("Initialization successful. phase=%.2f" % (self.phase))
    
    def load_data(self):
        """
        Load the gait data from the specified file and process it based on conditions.

        This function attempts to:
            1. Load data from pickle file at self.data_path.
            2. Verify data format.
            3. Optionally remove knee DOFs if self.knee_1dof.
            4. Determine valid steps based on conditions.
            5. If valid step found, process it (trim, repeat, smooth) to form qpos and qvel.
            6. If no valid step found, qpos remains None.

        Logs:
            If self.verbose is True, print warnings and info messages.
        """
        try:
            data = self.load_raw_data()
            self.validate_data(data)
            
            if self.knee_1dof:
                data = self.remove_knee_dofs(data)
            
            # Determine left and right steps
            left_steps, right_steps = self.classify_steps(data)
            
            # Calculate the speed for each step
            speeds = self.calculate_speeds(data)
            
            self.select_and_process_trajectory(data, left_steps, right_steps, speeds)
            
        except FileNotFoundError:
            if self.verbose:
               print(f"Error: The file at {self.data_path} could not be found.")
        except Exception as e:
            if self.verbose:
                print(f"Error loading data: {e}")
                
    def load_raw_data(self):
        """Load the raw trajectory data from a pickle file."""
        if self.verbose:
            print(f"Loading data from {self.data_path}...")
        with open(self.data_path, "rb") as f:
            return pickle.load(f)
        
    def validate_data(self, data):
        """Validate the structure and type of loaded data."""
        if not isinstance(data, list) or not all(isinstance(step, np.ndarray) for step in data):
            raise ValueError("Loaded data is not a list of numpy arrays.")
        if len(data) == 0:
            if self.verbose:
                print("No data found in the file.")
            raise ValueError("No data found.")
    
    def remove_knee_dofs(self, data):
        """Remove unnecessary DOFs if knee_1dof is True."""
        if self.verbose:
            print("Removing extra knee DOFs...")
        indices_to_remove = [10, 11, 19, 20]
        return [np.delete(step, indices_to_remove, axis=0) for step in data]
    
    def classify_steps(self, data):
        """
        Classify steps into left-start and right-start based on initial hip flexion angles.
        
        Returns:
            left_steps: Indices of left-start steps.
            right_steps: Indices of right-start steps.
        """
        left_steps = self.determine_left_step_indices(data)
        right_steps = [i for i in range(len(data)) if i not in left_steps]
        if self.verbose:
            print(f"Found {len(left_steps)} left-start steps and {len(right_steps)} right-start steps.")
        return left_steps, right_steps
    
    def calculate_speeds(self, data):
        """
        Calculate the speed of each step in the trajectory dataset.
        
        Returns:
            speeds: A list of speed values for each step.
        """
        speeds = [(step[2, -1] - step[2, 0]) / ((step.shape[1] - 1) / self.sample_frequency) for step in data]
        if self.verbose:
            print(f"Calculated speeds: {speeds}")
        return speeds
    
    def select_and_process_trajectory(self, data, left_steps, right_steps, speeds, speed_range=None):
        """
        Select and process a valid trajectory based on conditions.
        
        Parameters:
            data (list): List of trajectory steps (numpy arrays).
            left_steps (list): Indices of steps starting with the left foot.
            right_steps (list): Indices of steps starting with the right foot.
            speeds (list): List of speeds for all steps.
            speed_range (list[float], optional): A two-element list specifying the [low, high] speed range.
                If None, defaults to [1.2, 1.3].
        
        Modifies:
            self.qpos: Processed trajectory positions.
            self.qvel: Processed trajectory velocities.
        """
        # Interested speed range
        speed_range = speed_range or [1.2, 1.3]
        low_speed, high_speed = speed_range
        
        # Filter candidates based on starting side and speed
        if self.is_left_start:
            candidates = [i for i in left_steps if not self.is_fixed_speed or low_speed <= speeds[i] <= high_speed]
        else:
            candidates = [i for i in right_steps if not self.is_fixed_speed or low_speed <= speeds[i] <= high_speed]
            
        if not candidates:
            if self.verbose:
                print("No valid steps found under the specified conditions.")
            return

        # Randomly select a candidate step
        selected_index = random.choice(candidates)
        selected_step = data[selected_index]
        
        self.process_selected_step(selected_step)

        if self.verbose:
            print(f"Selected step index={selected_index}, traj_frames={self.traj_frames}, num_dofs={self.num_dofs}")
            
    def process_selected_step(self, step):
        """
        Process the selected step to generate the trajectory.
        
        Parameters:
            step (np.ndarray): The selected step to process.
            
        Modifies:
            self.qpos: Processed trajectory positions.
            self.qvel: Processed trajectory velocities.
        """
        # Trim the step to 2/3 length
        original_frames = step.shape[1]
        cut_frames = int(np.ceil(original_frames * 2 / 3))
        step = step[:, :cut_frames]
        
        # Update class variables
        self.current_speed = (step[2, -1] - step[2, 0]) / ((step.shape[1] - 1) / self.sample_frequency)
        self.num_dofs, self.step_frames = step.shape

        # Repeat and smooth trajectory
        trajectory = self.repeat_step_with_translation(step, self.repeat_times)
        if self.filter_enabled:
            sigma = np.max([np.min([10, int(np.ceil(3 / self.current_speed))]), 3])
            trajectory = gaussian_filter1d(trajectory, sigma=sigma, axis=1)

        # Assign processed data
        self.qpos = trajectory
        self.traj_frames = self.qpos.shape[1]
        self.qvel = self.compute_velocity(self.qpos, method="center")
            
    def fallback_to_random_step(self):
        """
        If no suitable step found earlier, randomly pick any step from the dataset for fallback.
        If this also fails, self.qpos will remain None.
        
        If self.verbose is True, print warning messages on failure.
        """
        try:
            data = self.load_raw_data()
            self.validate_data(data)
            
            if self.knee_1dof:
                data = self.remove_knee_dofs(data)

            if not data:
                if self.verbose:
                    print("Warning: No data available for fallback. Leaving qpos as None.")
                return
            
            # Randomly select any step
            selected_index = random.randint(0, len(data)-1)
            selected_step = data[selected_index]
            
            self.process_selected_step(selected_step)
            
            if self.verbose:
                print(f"Fallback succeeded: selected step index={selected_index}, traj_frames={self.traj_frames}, num_dofs={self.num_dofs}")
                
        except FileNotFoundError:
            if self.verbose:
               print(f"Error: The file at {self.data_path} could not be found.")
        except Exception as e:
            if self.verbose:
                print(f"Error loading data: {e}")
     
    def determine_left_step_indices(self, data):
        """
        Determine indices where the step starts with the left foot.
        
        Parameters:
            data (list of np.ndarray): Each element a step (2D np.ndarray), shape=(num_dofs, num_frames).

        Returns:
            indices (list[int]): Indices of steps starting with left foot (left hip flexion > right hip flexion at frame0).

        Notes:
            - Uses self.knee_1dof to determine the correct l_hip_flx_idx.
            - If data is empty or invalid, returns empty list.
        """
        r_hip_fle_idx = self.jnt_name.get("hip_flexion_r")
        l_hip_flx_idx = self.jnt_name.get("hip_flexion_l") if self.knee_1dof else 15
        
        if r_hip_fle_idx is None or l_hip_flx_idx is None:
            raise KeyError("hip_flexion_r or hip_flexion_l not found in jnt_name mapping.")
            
        # Check data validity
        if not isinstance(data, list) or len(data) == 0:
            if self.verbose:
                print("Warning: 'data' is empty or not a list. Cannot determine left step indices.")
            return []
        
        indices = [
            i for i, step in enumerate(data) 
            if step[l_hip_flx_idx,0] > step[r_hip_fle_idx,0]
            ]
        return indices
    
    def repeat_step_with_translation(self, step, N, overlap=10, sigma=3):
        """
        Repeat the given step N times and apply translation adjustments.
        
        Parameters:
            step (np.ndarray): The trajectory to repeat.
            N (int): Number of repetitions.
            overlap (int): Overlap frames for smoothing at splice points.
            sigma (float): Gaussian smoothing standard deviation.
    
        Returns:
            np.ndarray: Repeated and smoothed trajectory.
        
        Note: This method assumes the first three DOFs represent pelvis translations (tz, ty, tx) and the trajectory is in a global frame.
        """
        if not isinstance(step, np.ndarray) or step.ndim != 2:
            raise ValueError("Input step must be a 2D numpy array.")
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer.")

        dofs, frames = step.shape
        if dofs < 3:
            raise ValueError("Input step must have at least 3 DOFs for translation.")

        repeated_trajectory = np.zeros((dofs, frames * N))
        repeated_trajectory[:, :frames] = step

        # Adjust the translation DOFs (first three rows)
        for i in range(1, N):
            start_idx = i * frames
            end_idx = start_idx + frames
            current_step = step.copy()

            # Adjust the translation DOFs (first three rows)
            for j in range(3):  # Only for the first three DOFs
                offset = i * (current_step[j, -1] - current_step[j, 0])   # Use the previous step's last frame
                current_step[j, :] += offset

            repeated_trajectory[:, start_idx:end_idx] = current_step
        
        # Apply smoothing at splice points
        splice_indices = [
            (i * frames - overlap, i * frames + overlap)
            for i in range(1, N)
            ]
        for left, right in splice_indices:
            segment = repeated_trajectory[:, max(0, left):min(right, repeated_trajectory.shape[1])]
            smoothed_segment = gaussian_filter1d(segment, sigma=sigma, axis=1)
            repeated_trajectory[:, max(0, left):min(right, repeated_trajectory.shape[1])] = smoothed_segment

        return repeated_trajectory
    
    def compute_velocity(self, angles, method="center"):
        """
        Compute the velocity using the specified finite difference method ('forward', 'backward', or 'center').
        
        Parameters:
            angles (np.ndarray): Input trajectory data (dofs x frames).
            method (str): Finite difference method ('forward', 'backward', 'center').

        Returns:
            np.ndarray: Computed velocity (same shape as angles).
        """
        if method not in {"forward", "backward", "center"}:
            raise ValueError("Invalid method. Choose 'forward', 'backward', or 'center'.")
        
        if not isinstance(angles, np.ndarray):
            raise ValueError("Input data must be a numpy arrays.")
        
        delta_t = 1 / self.sample_frequency
        velocity = np.zeros_like(angles)
            
        if method == "forward":
            velocity[:, :-1] = (angles[:, 1:] - angles[:, :-1]) / delta_t
            velocity[:, -1] = 0  # Pad the last value with 0
        elif method == "backward":
            velocity[:, 1:] = (angles[:, 1:] - angles[:, :-1]) / delta_t
            velocity[:, 0] = 0  # Pad the first value with 0
        elif method == "center":
            velocity[:, 1:-1] = (angles[:, 2:] - angles[:, :-2]) / (2 * delta_t)
            velocity[:, 0] = 0  # Pad the first value with 0
            velocity[:, -1] = 0  # Pad the last value with 0
        else:
            raise ValueError("Invalid method. Choose 'forward', 'backward', or 'center'.")

        return velocity
    
    def get_qpos(self):
        """Retrieve the current qpos from the trajectory data."""
        if self.qpos is None:
            raise ValueError("qpos is not initialized. Ensure data is loaded first.")
        return self.qpos[:, self._pos]
    
    def get_qvel(self):
        """Retrieve the current qvel from the trajectory data."""
        if self.qvel is None:
            raise ValueError("qvel is not initialized. Ensure data is loaded first.")
        return self.qvel[:, self._pos]
    
    def get_reference_trajectories(self):
        """Return current qpos and qvel for the reference trajectory."""
        qpos = self.get_qpos()
        qvel = self.get_qvel()
        return qpos, qvel
    
    def reset(self, phase=0):
        """
        Reset the trajectory to the specified gait phase.
        
        Parameters:
            phase (float): Gait phase in percentage (0 to 100).
        
        Raises:
            ValueError: If the phase is out of the valid range.
        """
        if not (0 <= phase <= 100 ):
            raise IndexError(f"Phase must be between 0 and 100. Received: {phase}.")
            
        self._pos = int(np.round(phase / 100 * (self.step_frames - 1)))
        self.has_reached_the_end = False
        if (self.traj_frames - self._pos - 1) < 1e-6:
            self.has_reached_the_end = True
            
        if self.verbose:
            print(f"Trajectory reset: phase={phase}, _pos={self._pos}, has_reached_the_end={self.has_reached_the_end}.")
    
    def get_phase_variable(self):
        """Get the gait phase, unit in percentage"""
        phase = (self._pos % self.step_frames) / (self.step_frames - 1) * 100
        assert 0 <= phase <= 100
        return phase

    def next(self):
        """
        Advance to the next position in the trajectory.
        
        Notes:
            - If the end of the trajectory is reached, no further updates are made.
            - The function prints a warning when the trajectory ends if verbose is enabled.
        """
        if self.has_reached_the_end:
            if self.verbose:
                print("Already reached the last pos of the trajectory")
            return
        
        self._pos += self.increment
        self.has_reached_the_end = (self._pos >= self.traj_frames - 1)
        
        if self.verbose and self.has_reached_the_end:
            print("Trajectory has reached the end.")
        
    def set_random_init_state(self, range_start=0, range_end=50, precision=2):
        """
        Set a random initial gait phase within a specified range.

        Parameters:
            range_start (float): The lower bound of the gait phase range (inclusive). Default is 0.
            range_end (float): The upper bound of the gait phase range (inclusive). Default is 100.
            precision (int): The number of decimal places for the random gait phase. Default is 2.
        """
        if not (0 <= range_start <= range_end <= 100):
            raise ValueError("range_start and range_end must be between 0 and 100, and range_start <= range_end.")
        
        gait_cycle_in_percent = round(np.random.uniform(range_start, range_end), precision)
        self.reset(phase=gait_cycle_in_percent)
        
    def set_deterministic_init_state(self, phase=0):
        """
        Set the trajectory to a deterministic initial state.
        
        Parameters:
            phase (float): The initial gait phase as a percentage. Default is 0.
        """
        if not (0 <= phase <= 100 ):
            raise IndexError("Gait phase is out of range [0, 100].")
        self.reset(phase)
        
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
    
    def plot_trajectory(self, dof_names=None):
        """
        Plot the trajectory of specific DOFs over time for debugging.

        Parameters:
            dof_names (list[str], optional): List of DOF names to plot. If None, defaults to ['pelvis_tx'].
        """
        if dof_names is None:
            dof_names = ['pelvis_tx']
   
        for dof_name in dof_names:
            if dof_name not in self.jnt_name:
                raise ValueError(f"DOF {dof_name} not in jnt_name mapping.")
        
        if self.qpos is None:
            raise ValueError("No qpos data to plot.")
        
        plt.figure()
        for dof_name in dof_names:
            dof_idx = self.jnt_name[dof_name]
            plt.plot(self.qpos[dof_idx, :], label=dof_name)
            
        plt.title("Trajectory of DOFs")
        plt.xlabel("Frames")
        plt.ylabel("Position (rad or m)")
        plt.legend()
        plt.show()