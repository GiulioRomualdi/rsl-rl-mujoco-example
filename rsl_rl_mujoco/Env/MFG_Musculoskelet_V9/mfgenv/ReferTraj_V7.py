# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 20:03:59 2025

@author: YAKE
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import logging

logger = logging.getLogger(__name__)

class TrajectoryManager:
    """
    Manages preprocessed 43‐DOF reference trajectories loaded from .npz files.

    Each .npz must contain:
      - 'qpos': ndarray of shape (43, T)
      - 'qvel': ndarray of shape (43, T)

    On initialization, loads all files from a directory, computes:
      - average forward speed (m/s)
      - foot‐start indicator (0=right, 1=left)
    Filters by speed_range, caches results to avoid repeated I/O,
    and provides accessors to retrieve trajectories and metadata.

    Attributes:
        qpos_list (List[np.ndarray]): Loaded qpos arrays.
        qvel_list (List[np.ndarray]): Loaded qvel arrays.
        speeds_list (List[float]): Average speeds for each trajectory.
        foot_starts (List[int]): Foot‐start flags for each trajectory.
        jnt_name (Dict[str,int]): Joint name→index mapping.
        sample_frequency (float): Hz for frame→time conversion.
        speed_range (Tuple[float,float]): Min/max speed filter in m/s.
    """

    # Final 43‐DOF joint names with beta‐DOFs after each mtp joint
    EXPANDED_JOINT_NAMES: List[str] = [
        'pelvis_tz','pelvis_ty','pelvis_tx',
        'pelvis_tilt','pelvis_list','pelvis_rotation',
        'hip_flexion_r','hip_adduction_r','hip_rotation_r',
        'knee_angle_r','ankle_angle_r','subtalar_angle_r','mtp_angle_r',
        'knee_angle_r_beta_translation2',
        'knee_angle_r_beta_translation1',
        'knee_angle_r_beta_rotation1',
        'hip_flexion_l','hip_adduction_l','hip_rotation_l',
        'knee_angle_l','ankle_angle_l','subtalar_angle_l','mtp_angle_l',
        'knee_angle_l_beta_translation2',
        'knee_angle_l_beta_translation1',
        'knee_angle_l_beta_rotation1',
        'lumbar_extension','lumbar_bending','lumbar_rotation',
        'arm_flex_r','arm_add_r','arm_rot_r',
        'elbow_flex_r','pro_sup_r','wrist_flex_r','wrist_dev_r',
        'arm_flex_l','arm_add_l','arm_rot_l',
        'elbow_flex_l','pro_sup_l','wrist_flex_l','wrist_dev_l'
    ]
    # Prebuilt name→index map
    EXPANDED_JOINT_MAP: Dict[str,int] = {
        name: idx for idx, name in enumerate(EXPANDED_JOINT_NAMES)
    }

    # Class‐level cache to avoid reloading the same directory
    _cache: Dict[str, Dict] = {}

    def __init__(
        self,
        data_path: Union[str, Path],
        speed_range: Tuple[float, float] = (0.0, 3.0),
        sample_frequency: float = 100.0,
        verbose: bool = False
    ):
        """
        Initialize TrajectoryManager.

        Args:
            data_path: Directory containing processed .npz trajectory files.
            speed_range: (min, max) forward speeds in m/s to include.
            sample_frequency: Sampling rate in Hz for frame→time conversion.
            verbose: If True, log detailed loading info.

        Raises:
            ValueError: If data_path is invalid or parameters out of range.
            RuntimeError: If no trajectories pass the filter.
        """
        self.path = Path(data_path)
        if not self.path.is_dir():
            raise ValueError(f"Processed trajectory folder not found: {self.path}")

        if len(speed_range) != 2 or speed_range[0] > speed_range[1]:
            raise ValueError("speed_range must be (min, max) with min <= max")
        if sample_frequency <= 0:
            raise ValueError("sample_frequency must be positive")

        self.speed_range = (float(speed_range[0]), float(speed_range[1]))
        self.sample_frequency = float(sample_frequency)
        self.verbose = bool(verbose)

        # fixed joint mapping
        self.jnt_name = TrajectoryManager.EXPANDED_JOINT_MAP

        # load or retrieve from cache
        key = str(self.path.resolve())
        if key in TrajectoryManager._cache:
            cached = TrajectoryManager._cache[key]
            self.qpos_list   = cached['qpos_list']
            self.qvel_list   = cached['qvel_list']
            self.speeds_list = cached['speeds_list']
            self.foot_starts = cached['foot_starts']
        else:
            self._load_and_index()
            TrajectoryManager._cache[key] = {
                'qpos_list':   self.qpos_list,
                'qvel_list':   self.qvel_list,
                'speeds_list': self.speeds_list,
                'foot_starts': self.foot_starts
            }

    def _load_and_index(self) -> None:
        """
        Private: Load all .npz files, compute average speed and foot-start,
        apply speed_range filter, and populate internal lists.
        """
        files = sorted(self.path.glob("*.npz"))
        if not files:
            raise RuntimeError(f"No .npz files found in {self.path}")

        # Indices for computing speed and foot-start
        tx_idx = self.jnt_name['pelvis_tx']
        lhip   = self.jnt_name['hip_flexion_l']
        rhip   = self.jnt_name['hip_flexion_r']

        qpos_list, qvel_list = [], []
        speeds, foots = [], []

        for f in files:
            try:
                data = np.load(f)
                qpos = data['qpos']
                qvel = data['qvel']
            except Exception as e:
                logger.warning(f"Skipping {f.name}: cannot load arrays ({e})")
                continue

            if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape != qvel.shape:
                logger.warning(f"Skipping {f.name}: invalid shapes qpos{qpos.shape} qvel{qvel.shape}")
                continue

            T = qpos.shape[1]
            # compute average forward speed
            disp = float(qpos[tx_idx, -1] - qpos[tx_idx, 0])
            duration = max(T - 1, 1) / self.sample_frequency
            speed = disp / duration

            # determine foot-start: 1 if left hip > right hip at start
            foot = int(qpos[lhip, 0] > qpos[rhip, 0])

            # filter by speed
            if self.speed_range[0] <= speed <= self.speed_range[1]:
                qpos_list.append(qpos)
                qvel_list.append(qvel)
                speeds.append(speed)
                foots.append(foot)
                if self.verbose:
                    logger.info(f"Loaded {f.name}: speed={speed:.2f}, foot={foot}")
            else:
                if self.verbose:
                    logger.debug(f"Filtered out {f.name}: speed={speed:.2f} outside {self.speed_range}")

        if not qpos_list:
            raise RuntimeError("No trajectories passed the speed filter.")

        # make arrays read-only
        for arr in qpos_list + qvel_list:
            arr.setflags(write=False)

        self.qpos_list   = qpos_list
        self.qvel_list   = qvel_list
        self.speeds_list = speeds
        self.foot_starts = foots

    def __len__(self) -> int:
        """Return number of trajectories available."""
        return len(self.qpos_list)
    
    def length(self) -> int:
        return len(self.qpos_list)

    def get(self, traj_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve (qpos, qvel) for a given trajectory index.

        Args:
            traj_id: Index of trajectory in [0, len).

        Returns:
            Tuple of (qpos, qvel), each of shape (43, T).

        Raises:
            IndexError: If traj_id is out of range.
        """
        if not 0 <= traj_id < len(self):
            raise IndexError(f"traj_id {traj_id} out of range")
        return self.qpos_list[traj_id], self.qvel_list[traj_id]

    def get_speed(self, traj_id: int) -> float:
        """
        Get average forward speed for trajectory.

        Args:
            traj_id: Index of trajectory.

        Returns:
            Speed in meters per second.

        Raises:
            IndexError: If traj_id is out of range.
        """
        if not 0 <= traj_id < len(self):
            raise IndexError(f"traj_id {traj_id} out of range")
        return self.speeds_list[traj_id]

    def get_foot_start(self, traj_id: int) -> int:
        """
        Get foot-start indicator for trajectory.

        Args:
            traj_id: Index of trajectory.

        Returns:
            1 if left foot starts first, 0 if right.

        Raises:
            IndexError: If traj_id is out of range.
        """
        if not 0 <= traj_id < len(self):
            raise IndexError(f"traj_id {traj_id} out of range")
        return self.foot_starts[traj_id]

    @property
    def all_speeds(self) -> List[float]:
        """List of average speeds for all trajectories."""
        return list(self.speeds_list)

    @property
    def all_foot_starts(self) -> List[int]:
        """List of foot-start flags for all trajectories."""
        return list(self.foot_starts)

    def get_jnt_map(self) -> Dict[str, int]:
        """
        Get a copy of the joint name → index mapping.

        Returns:
            A dict mapping joint names to their index in qpos/qvel.
        """
        return dict(self.jnt_name)

    def get_sample_frequency(self) -> float:
        """
        Return the sampling frequency used for speed computation.

        Returns:
            Sampling rate in Hz.
        """
        return self.sample_frequency
    
    

class ReferenceTrajectories:
    """
    Per‐environment cursor into a shared set of 43‐DOF trajectories provided
    by TrajectoryManager.

    On reset, selects (or keeps) a trajectory, sets phase, and initializes
    frame pointer. On each step(), returns the reference (qpos, qvel) at
    the current frame, advancing by `increment` frames (with optional holds).

    Attributes
    ----------
    manager : TrajectoryManager
        Shared trajectories provider.
    traj_id : int
        Index of the current trajectory.
    qpos, qvel : np.ndarray
        Arrays of shape (43, T) for the selected trajectory.
    traj_frames : int
        Total number of frames in the trajectory.
    speed : float
        Average forward speed of the current trajectory.
    foot_start : int
        1 if left foot starts first, else 0.
    _pos : int
        Current frame index.
    _hold_counter : int
        Number of consecutive holds at the current frame.
    _has_reached_end : bool
        True once the cursor has reached the final frame.
    """

    def __init__(
        self,
        manager: TrajectoryManager,
        traj_id: Optional[int] = None,
        increment: int = 2,
        max_hold_steps: int = 5,
        random_seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Bind to a TrajectoryManager and initialize internal state.

        Parameters
        ----------
        manager : TrajectoryManager
            Shared trajectory loader.
        traj_id : Optional[int], default=None
            Specific trajectory to select; if None, chosen randomly.
        increment : int, default=1
            Number of frames to advance on each call to step().
        max_hold_steps : int, default=0
            Maximum consecutive holds before advancing.
        random_seed : Optional[int], default=None
            Seed for reproducible random selection.
        verbose : bool, default=False
            Enable debug logging if True.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        self.manager = manager
        self.n_traj = manager.length() if hasattr(manager, "length") else len(manager)
        self.increment = int(increment)
        self.max_hold_steps = int(max_hold_steps)
        self.rng = np.random.default_rng(random_seed)
        self.repeat_times = int(10)

        # fixed joint mapping and sampling rate
        self.jnt_name = manager.get_jnt_map()
        self.sample_frequency = manager.get_sample_frequency()

        # pick and load trajectory
        self.set_trajectory(traj_id)

    def set_trajectory(self, traj_id: Optional[int] = None) -> None:
        """
        Switch to the specified trajectory, or random if None,
        and reset frame pointer and hold counter.

        Parameters
        ----------
        traj_id : Optional[int]
            Index of trajectory to select.
        """
        if traj_id is None:
            self.traj_id = int(self.rng.integers(self.n_traj))
        else:
            if not (0 <= traj_id < self.n_traj):
                raise IndexError(f"traj_id {traj_id} out of range [0, {self.n_traj-1}]")
            self.traj_id = traj_id

        # load qpos and qvel (read‐only views)
        self.qpos, self.qvel = self.manager.get(self.traj_id)
        self.qpos.setflags(write=False)
        self.qvel.setflags(write=False)
        assert not self.qpos.flags.writeable and not self.qvel.flags.writeable, \
            "Underlying trajectory data must be read-only"
            
        self.traj_frames = self.qpos.shape[1]
        self.step_frames = self.traj_frames // self.repeat_times
        self.speed = self.manager.get_speed(self.traj_id)
        self.foot_start = self.manager.get_foot_start(self.traj_id)

        # initialize cursor state
        self._pos = 0
        self._hold_counter = 0
        self._has_reached_end = False

        self.logger.debug(
            f"Switched to trajectory {self.traj_id}: "
            f"frames={self.traj_frames}, cycle={self.step_frames}, "
            f"speed={self.speed:.2f}, foot_start={'left' if self.foot_start else 'right'}"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        phase: Optional[float] = None,
        randomize_traj: bool = True,
        traj_id: Optional[int] = None
        ) -> None:
        """
        Reinitialize this ReferenceTrajectories cursor.

        Parameters
        ----------
        seed : Optional[int]
            If provided, reseed the internal RNG (affects random trajectory/phase).
        phase : Optional[float]
            If provided (0-100), sets exact gait phase; if None, choose random [0,100).
        randomize_traj : bool
            If True and traj_id is None, pick a random trajectory; else keep current or use provided.
        traj_id : Optional[int]
            If provided, switch to this exact trajectory; else behavior depends on randomize_traj.
    
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
        
        if randomize_traj:
            self.set_trajectory(traj_id)
        elif traj_id is not None:
            self.set_trajectory(traj_id)

        if phase is None:
            phase_val = float(self.rng.uniform(0.0, 100.0))
        else:
            if not isinstance(phase, (float, int)):
                raise TypeError(f"phase must be float or None, got {type(phase)}")
            phase_val = float(phase)
            if not (0.0 <= phase_val <= 100.0):
                raise ValueError("phase must be within [0,100]")

        if self.step_frames > 1:
            idx = int(round(phase_val / 100.0 * (self.step_frames - 1)))
        else:
            idx = 0
            
        self._pos = idx
        self._hold_counter = 0
        self._has_reached_end = (self._pos >= self.traj_frames - 1)
        self.logger.debug(
            f"reset(): traj_id={self.traj_id}, phase={phase_val:.2f}% -> frame={self._pos}, "
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
        True if the cursor is at the final frame of the trajectory.
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
        Get (qpos, qvel) for the current frame.
        """
        return self.get_qpos(), self.get_qvel()

    def get_pelvis_ang(self) -> np.ndarray:
        """
        Extract pelvis [tz, ty, tx, tilt, list, rotation] at current frame.
        """
        keys = ['pelvis_tz','pelvis_ty','pelvis_tx',
                'pelvis_tilt','pelvis_list','pelvis_rotation']
        idxs = [self.jnt_name[k] for k in keys]
        return self.qpos[idxs, self._pos]

    def get_pelvis_angV(self) -> np.ndarray:
        """
        Extract pelvis velocities at current frame.
        """
        keys = ['pelvis_tz','pelvis_ty','pelvis_tx',
                'pelvis_tilt','pelvis_list','pelvis_rotation']
        idxs = [self.jnt_name[k] for k in keys]
        return self.qvel[idxs, self._pos]

    def get_torso_ang(self) -> np.ndarray:
        """
        Extract torso [lumbar_extension, lumbar_bending, lumbar_rotation].
        """
        keys = ['lumbar_extension','lumbar_bending','lumbar_rotation']
        idxs = [self.jnt_name[k] for k in keys]
        return self.qpos[idxs, self._pos]

    def get_torso_angV(self) -> np.ndarray:
        """
        Extract torso angular velocities at current frame.
        """
        keys = ['lumbar_extension','lumbar_bending','lumbar_rotation']
        idxs = [self.jnt_name[k] for k in keys]
        return self.qvel[idxs, self._pos]

    def get_joint_data(
        self,
        joint_group: Union[str, List[str]],
        data_type: str = "angle"
        ) -> np.ndarray:
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
