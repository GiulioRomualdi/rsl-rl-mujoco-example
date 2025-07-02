import logging
from pathlib import Path

# -------------------------------------------------------------------
# Module‐level logger configuration (based on ReferTraj_V5.py)
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(ch)

import random
import time
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer as viewer
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, List, Tuple, Union

from . import config as cfg
from . import mujoco_utils as mj_utils
from . import common_utils as c_utils
from . import state as s
from . import termination as t
from . import reward as r
from .ReferTraj_V5 import ReferenceTrajectories as refs

class MFG_Musculoskelet_V7(gym.Env):
    """
    A custom Gymnasium environment for MuJoCo-based musculoskeletal simulation.

    This environment uses position actuators to track a cyclic reference gait
    trajectory, enabling imitation-based reinforcement learning on a biomechanical model.

    Attributes
    ----------
    config : dict
        The merged configuration parameters loaded via 'cfg.load_config'.
    model : mujoco.MjModel
        The MuJoCo model instance.
    data : mujoco.MjData
        The corresponding simulation data.
    ref_traj : ReferenceTrajectories
        Manages the cyclic gait reference trajectory.

    Methods
    -------
    reset(seed=None, options=None, custom_init_state=None)
        Reset the environment, optionally with a specific seed or initial gait phase.
    step(action)
        Apply an action, step the simulation, and return observation, reward, done flags, and info.
    get_obs()
        Construct and return the current observation vector.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50
        }

    EPSILON = 1e-6
    RL_FS = 50
    
    def __init__(self, config_path: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the musculoskeletal simulation environment.

        Parameters
        ----------
        config_path : str or None
            Path to the JSON configuration file; if None, uses the default.
        **kwargs
            Keyword overrides for individual configuration fields.

        Raises
        ------
        FileNotFoundError
            If the specified model or data file cannot be found.
        ValueError
            If critical configuration values are invalid (e.g., sampling frequency not compatible).
        RuntimeError
            For other initialization failures.
        """
        super().__init__()
        
        # Load configuration and apply overrides
        if config_path is not None:
            cfg_file = Path(config_path)
            if not cfg_file.is_file():
                raise FileNotFoundError(f"Config file not found: {cfg_file}")
        self.config: Dict[str, Any] = cfg.load_config(config_path, kwargs)
        
        # Adjust logging level based on verbosity flag
        self.verbose = self.config.get("verbose", False)
        level = logging.DEBUG if self.verbose else logging.WARNING
        logger.setLevel(level)
        
        # Seed setting
        self._initial_seed = self._seed(self.config.get("random_seed", None))
        
        # Validate existence of key files
        c_utils.validate_file(self.config["model_path"], "Model file")
        c_utils.validate_file(self.config["data_path"], "Data file")
        
        # Load MuJoCo model & data
        self.model, self.data = mj_utils.load_mujoco_model(self.config["model_path"])
        self.num_dofs: int = self.model.nv
        self.num_joints: int = self.model.njnt
        self.num_actuators: int = self.model.nu
        self.total_mass: float = self.model.body_mass[1:].sum()
        self.opt_time: float = self.model.opt.timestep
        
        # Extract and validate joint/actuator names
        self.jnt_names: List[str] = [self.model.joint(i).name for i in range(self.num_joints)]
        self.actuator_names: List[str] = [self.model.actuator(i).name for i in range(self.num_actuators)]
        mj_utils.check_invalid_names(self.jnt_names, self.actuator_names)
        self._act_to_qpos_idx: List[int] = [self.jnt_names.index(name) for name in self.actuator_names]
        self.actuator_prms: Dict[str, Any] = mj_utils.parse_actuator_prm_from_xml(self.config["model_path"])
        
        # Initialize reference trajectory manager
        ref_freq: float = self.config.get("ref_traj_sample_frequency", 100)
        if ref_freq % self.RL_FS != 0:
            raise ValueError(f"ref_traj_sample_frequency ({ref_freq}) must be divisible by RL_FS ({self.RL_FS})")
        ref_increment = int(ref_freq // self.RL_FS)
        self.ref_traj = refs(
            data_path = self.config.get("data_path", None),
            repeat_times = self.config.get("ref_traj_repeat_times", 5),
            increment = ref_increment,
            verbose = self.verbose,
            random_seed = self.config.get("random_seed", None),
            )
        self.initialize_ref_traj()
        
        # Load reward parameters
        self.reward_weights: Dict[str, float] = self.config.get("reward_weights", {})
        self.imitation_weights: Dict[str, float] = self.config.get("imitation_weights", {})
        self.smooth_weights: Dict[str, float] = self.config.get("smooth_weights", {})
        self.reward_coefficients: Dict[str, float] = self.config.get("reward_coefficients", {})
        
        # Observation flags
        self.remove_x_pos: bool = self.config.get("remove_x_pos", True)
        self.add_phase: bool = self.config.get("add_phase", False)
        
        # Compute state size and build spaces
        self.state_size: int = s.get_state_size(self)
        self.initialize_spaces()
        
        # Frame-skip calculation 
        self._frame_skip = c_utils.calculate_frameskip(self)
        
        # Initialize action trackers
        self.prev_action: Optional[np.ndarray] = None
        self.current_action: Optional[np.ndarray] = None
        self.prev_qpos: Optional[np.ndarray] = None
        
        # History buffers
        self.short_history_max_len: int = self.config.get("short_history_max_len", 5)
        self.long_history_max_len: int = self.config.get("long_history_max_len", 30)
        self.init_history_buffers()
        
        # Rendering setup
        self.setup_rendering(self.config.get("render_mode"))
        
        logger.debug("Environment initialized successfully.")
    
    def _seed(self, seed: Optional[int] = None) -> int:
        """
        Seed all RNGs (Gym, Python, NumPy) for reproducibility.

        Parameters
        ----------
        seed : int or None
            Desired seed. If None, a random seed is chosen.

        Returns
        -------
        actual_seed : int
            The actual integer seed used.
        """
        from gymnasium.utils import seeding
        self._np_random, actual = seeding.np_random(seed)
        u32 = int(actual) & 0xFFFFFFFF
        random.seed(u32)
        np.random.seed(u32)
        logger.debug("Random seed synchronized: %d", u32)
        return u32
        
    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None, 
              custom_init_state: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to start a new episode.

        Parameters
        ----------
        seed : int or None
            Seed for RNG; if None, previous state is used.
        options : dict or None
            Gymnasium API placeholder (unused).
        custom_init_state : dict or None
            If provided, must contain key 'per' with a float or list of floats
            indicating the gait phase percentage(s) to initialize.

        Returns
        -------
        obs : np.ndarray
            Initial observation after reset.
        info : dict
            Contains at least 'seed': the integer seed used.

        Raises
        ------
        TypeError
            If 'seed' is not an int or None.
        KeyError
            If 'custom_init_state' is provided but lacks 'per'.
        RuntimeError
            If any error occurs during reset.
        """
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError(f"seed must be an int or None, got {type(seed)}")
            self._seed(seed)
            self.ref_traj.rng = np.random.default_rng(seed)
        try:
            if custom_init_state is not None:
                if 'per' not in custom_init_state:
                    raise KeyError("custom_init_state must contain the key 'per' for gait percentage.")
                self.randomize_initial_state(custom_init_state['per'])
            else:
                self.randomize_initial_state()
            
            obs = self.get_obs()
            
            act_obs = np.concatenate([
                np.zeros(self.action_space.shape[0], dtype=np.float32), 
                obs])
            
            self.short_history.clear()
            self.long_history.clear()
            for _ in range(self.short_history_max_len):
                self.short_history.append(act_obs.copy())
            for _ in range(self.long_history_max_len):
                self.long_history.append(act_obs.copy())
            
            self.avg_qvel = self.data.qvel
            self.avg_ctrl = np.zeros((self.num_actuators,))
            
            self.prev_qpos = None
            self.prev_action = None
            self.current_action = None
            
            logger.debug("Environment successfully reset.")
            
            return obs, {}
        except Exception as e:
            raise RuntimeError(f"Environment reset failed: {e}") from e
    
    def get_obs(self) -> np.ndarray:
        """
        Build and return the current observation vector.

        Returns
        -------
        obs : np.ndarray
            Flattened state (and optionally phase) as a float32 array.

        Raises
        ------
        RuntimeError
            If any error occurs while assembling the observation.
        """
        try:
            obs, info = s.get_state(self)
            if self.add_phase:    
                phase = self.ref_traj.phase / 100
                obs = np.concatenate(([phase], obs), axis=0)
            
            return obs.astype(np.float32)
        
        except Exception as e:
            raise RuntimeError(f"Failed to construct observation. Details: {e}") from e
    
    def get_privilege(self) -> np.ndarray:
        """
        Get privileged information, including future states of reference trajectories.
        
        Returns:
            np.ndarray: A flattened observation vector.
        
        Raises:
            RuntimeError: If constructing the privilege fails.
        """
        try:
            pri, info = s.get_state_extend(self)
            if self.add_phase:    
                phase = self.ref_traj.phase / 100
                pri = np.concatenate(([phase], pri), axis=0)
            return pri
        except Exception as e:
            raise RuntimeError(f"Failed to construct privileged information: {e}")
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply an action and advance the simulation by one environment step.

        Parameters
        ----------
        action : np.ndarray
            Control command array of shape '(num_actuators,)'.

        Returns
        -------
        obs : np.ndarray
            Next observation.
        reward : float
            Scalar reward for this transition.
        terminated : bool
            Whether a terminal condition has been reached.
        truncated : bool
            Whether the episode was truncated.
        info : dict
            Additional diagnostics (e.g., histories, termination details).

        Raises
        ------
        RuntimeError
            If any error occurs during action application or simulation.
        """
        try:
            if action is None or action.shape != (self.num_actuators,):
                raise ValueError(f" Invalid action shape: expected ({self.num_actuators},), got {action.shape}")
            
            if not np.isfinite(action).all():
                raise ValueError(" Action contains NaN or Inf values.")
            
            # target_qpos = self.rescale_actions(action)
            target_qpos = action
            self.prev_qpos = self.data.qpos[self._act_to_qpos_idx].copy()
            self.prev_action = self.current_action.copy() if self.current_action is not None else None
            self.current_action = target_qpos.copy()
            
            self.data.ctrl[:] = target_qpos
            for _ in range(self._frame_skip):
                mujoco.mj_step(self.model, self.data)
            
            if self.render_mode == "human":
                self.render()
            
            self.ref_traj.next()
            obs = self.get_obs()
            reward, reward_details = r.compute_reward(self)
            terminated, terminated_details = t.check_termination(self, conditions=["has_fallen", "comY_deviated", "site_deviation_exceeded"])
            truncated = self.ref_traj.has_reached_end
            act_obs = np.concatenate([action, obs])
            self.update_history_buffer(act_obs)
            
            info = {
                "terminated_info": terminated_details,
                "reward_info": reward_details,
                "short_history": self.short_history,
                "long_history": self.long_history,
                }
            
            if self.verbose:
                logger.info("Step completed successfully.")
            return obs, reward, terminated, truncated, info
        
        except Exception as e:
            raise RuntimeError(f"Environment step failed: {e}") from e
    
    def render(self) -> None:
        """
        Render the environment via MuJoCo's passive viewer when in human mode.

        This will launch the viewer if not already running, synchronize frames
        to the target FPS, and close it if render_mode changes.
        """
        if self.render_mode != "human":
            if self.viewer is not None:
                self.close()
            return
        
        if self.viewer is None or not self.viewer.is_running():
            self.viewer = viewer.launch_passive(self.model, self.data)
            
        start_time = time.perf_counter()
        self.viewer.sync()
        elapsed_time = time.perf_counter() - start_time
        time.sleep(max(0, (1 / self.metadata["render_fps"]) - elapsed_time))
    
    def close(self) -> None:
        """
        Close and cleanup the MuJoCo viewer, if open.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
    def __del__(self):
        try:
            if hasattr(self, 'viewer') and self.viewer is not None:
                self.viewer.close()
        except Exception:
            pass
    
    def initialize_ref_traj(self) -> None:
        """
        Prepare and reset the reference trajectory for use.

        Raises
        ------
        ValueError
            If the reference trajectory object is uninitialized or returns invalid data.
        """
        if not hasattr(self, 'ref_traj') or self.ref_traj is None:
            raise ValueError("Reference trajectory object is not initialized.")

        self.ref_traj.reset(phase=0)
        self.init_qpos, self.init_qvel = self.ref_traj.get_reference_trajectories()
        if self.init_qpos is None or self.init_qvel is None:
            raise ValueError("Failed to retrieve reference trajectories.")
        
        self.init_COMY = self.init_qpos[0]
        self.init_qvel = c_utils.convert_ref_traj_qvel(self.init_qvel, self.init_qpos)
        self.init_qpos = c_utils.convert_ref_traj_qpos(self.init_qpos)
    
    def initialize_spaces(self) -> None:
        """
        Define Gym 'observation_space' and 'action_space' based on state and control ranges.

        Raises
        ------
        ValueError
            If computed dimensions or ranges are invalid.
        """
        if self.state_size <= 0:
            raise ValueError(f"Invalid state size: {self.state_size}. Ensure it be a positive value.")
            
        obs_size = self.state_size + 1 if self.add_phase else self.state_size
        state_ubound = np.full(obs_size, np.inf, dtype=np.float32)
        state_lbound = np.full(obs_size, -np.inf, dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low = state_lbound, 
            high = state_ubound,
            shape = (obs_size,), 
            dtype = np.float32
            )
        
        ctrl_range = np.array([self.actuator_prms[name].get("range") for name in self.actuator_names], dtype=np.float32)
        if ctrl_range.ndim != 2 or ctrl_range.shape[1] != 2 or ctrl_range.shape[0] != self.num_actuators:
            raise ValueError(f"Invalid control range shape: expected ({self.num_actuators}, 2), got {ctrl_range.shape}")
        
        self.action_space = spaces.Box(
            low = ctrl_range[:, 0], 
            high = ctrl_range[:, 1], 
            shape = (self.num_actuators,), 
            dtype=np.float32
            )
        
        logger.info("Observation space initialized with shape %s.", self.observation_space.shape)
        
    def init_history_buffers(self) -> None:
        """
        Create 'deque' buffers for action‐observation history.
        """
        self.short_history = deque(maxlen=self.short_history_max_len)
        self.long_history = deque(maxlen=self.long_history_max_len)
    
    def setup_rendering(self, render_mode: str) -> None:
        """
        Configure the environment's render mode and prepare the viewer handle.

        Parameters
        ----------
        render_mode : str
            Target render mode (e.g., 'human').

        Raises
        ------
        ValueError
            If 'render_mode' is not among allowed modes.
        """
        if render_mode and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f" Invalid render mode: {render_mode}. Valid options are {self.metadata['render_modes']}.")
        
        self.render_mode = render_mode
        self.viewer = None
        
    def randomize_initial_state(self, 
                                percent: Optional[Union[float, List[float]]] = None
                                ) -> None:
        """
        Set the initial gait phase, either deterministically or randomly.

        Parameters
        ----------
        percent : float or list of floats or None
            - If float in [0,100]: set that exact phase.
            - If list: choose one element uniformly.
            - If None: random within full [0,100].

        Raises
        ------
        TypeError
            If 'percent' has an invalid type.
        ValueError
            If any specified percentage is outside [0,100].
        RuntimeError
            If underlying reference trajectory reset fails.
        """
        try:
            if percent is None:
                self.ref_traj.set_random_init_state()
                logger.info("Initialized state randomly.")
            elif isinstance(percent, (float, int)):
                if not (0 <= percent <= 100):
                    raise ValueError(f" Percentage {percent} must be in the range [0, 100].")
                self.ref_traj.set_deterministic_init_state(percent)
                logger.info("Initialized state at deterministic percentage %s%%.", percent)
            elif isinstance(percent, list):
                if not all(0 <= p <= 100 for p in percent):
                    raise ValueError(f"All percentages must be in range [0, 100], got: {percent}")
                chosen_percent = np.random.choice(percent)
                self.ref_traj.set_deterministic_init_state(chosen_percent)
                logger.info("Randomly selected %s%% from provided list.", chosen_percent)
            else:
                raise TypeError(f"Percent must be None, a float, or a list of floats. Got {type(percent)}.")
            
            self.reset_to_initial_state()
            
        except Exception as e:
            raise RuntimeError(f"Failed to randomize initial state: {e}")
            
    def reset_to_initial_state(self) -> None:
        """
        Load the reference qpos/qvel into MuJoCo and run a forward pass.

        Raises
        ------
        RuntimeError
            If reference data mismatches model dimensions or simulation fails.
        """
        try:
            self.init_qpos, self.init_qvel = self.ref_traj.get_reference_trajectories()
            if self.init_qpos is None or self.init_qvel is None:
                raise ValueError("Reference trajectory returned invalid states.")
            if len(self.init_qpos) != self.num_dofs:
                raise ValueError(f"The number of DOFs in reference trajectory ({len(self.init_qpos)}) does not match the model ({self.num_dofs}).")
            
            self.init_COMY = -self.init_qpos[0]
            
            self.init_qvel = c_utils.convert_ref_traj_qvel(self.init_qvel, self.init_qpos)
            self.init_qpos = c_utils.convert_ref_traj_qpos(self.init_qpos)
            
            self.data.qpos[:] = self.init_qpos
            self.data.qvel[:] = self.init_qvel
            self.data.ctrl[:] = self.init_qpos[7:]
            
            mujoco.mj_forward(self.model, self.data)
            
            if self.render_mode == "human":
                self.render()
                
            logger.info("Environment reset to initial state successfully.")
        
        except Exception as e:
            raise RuntimeError(f"Failed to reset environment state: {e}")
    
    def rescale_actions(self, action: np.ndarray) -> np.ndarray:
        """
        Map normalized policy outputs [-1,1] to actual joint targets.

        Parameters
        ----------
        action : np.ndarray
            Policy output array, values in [-1,1], length = number of actuators.

        Returns
        -------
        target_qpos : np.ndarray
            Joint set-points scaled to each actuator's control range.

        Raises
        ------
        TypeError
            If 'action' is not a numpy array.
        ValueError
            If its length mismatches the number of actuators.
        """
        if not isinstance(action, np.ndarray):
            raise TypeError("Input action must be a NumPy array.")
        if len(action) != self.num_actuators:
            raise ValueError(f"Action length {len(action)} does not match the number of actuators {self.num_actuators}.")
    
        clipped_action = np.clip(action, -1, 1)
    
        low = np.array([self.actuator_prms.get(name, {}).get("range", [-1, 1])[0] for name in self.actuator_names])
        high = np.array([self.actuator_prms.get(name, {}).get("range", [-1, 1])[1] for name in self.actuator_names])
        
        target_qpos = low + ((clipped_action + 1) / 2) * (high - low)
    
        return target_qpos
    
    def inverse_rescale_actions(self, target_qpos: np.ndarray) -> np.ndarray:
        """
        Map actual joint set-points back into normalized action space [-1,1].

        Parameters
        ----------
        target_qpos : np.ndarray
            Desired joint positions, length = number of actuators.

        Returns
        -------
        action : np.ndarray
            Normalized commands in [-1,1].

        Raises
        ------
        TypeError
            If 'target_qpos' is not a numpy array.
        ValueError
            If its length mismatches the number of actuators or any range is zero.
        """
        if not isinstance(target_qpos, np.ndarray):
            raise TypeError("Input target_qpos must be a NumPy array.")
        
        if len(target_qpos) != self.num_actuators:
            raise ValueError(f"Length of target_qpos ({len(target_qpos)}) does not match the number of actuators ({self.num_actuators}).")
        
        low = np.array([self.actuator_prms.get(name, {}).get("range", [-1, 1])[0] for name in self.actuator_names])
        high = np.array([self.actuator_prms.get(name, {}).get("range", [-1, 1])[1] for name in self.actuator_names])
        denominator = high - low
        if np.any(denominator == 0):
            raise ValueError("One or more actuators have a zero range (high equals low), cannot perform inverse scaling.")
            
        action = 2 * (target_qpos - low) / denominator - 1
        action = np.clip(action, -1, 1)
        return action
    
    def update_history_buffer(self, act_obs: np.ndarray) -> None:
        """
        Append the latest action-observation pair into history deques.

        Parameters
        ----------
        act_obs : np.ndarray
            Concatenated array of '[action, observation]'.

        Raises
        ------
        ValueError
            If its dimension does not match expected length.
        """
        expected_dim = self.observation_space.shape[0] + self.action_space.shape[0]
        if act_obs.shape[0] != expected_dim:
            raise ValueError(f"Mismatch in act_obs dimension. Expected {expected_dim}, but got {act_obs.shape[0]}.")

        self.short_history.append(act_obs)
        self.long_history.append(act_obs)
        
    def get_frameskip(self) -> int:
        """
        Get the number of MuJoCo steps executed per 'step()' call.

        Returns
        -------
        frame_skip : int
            Number of simulator frames per environment frame.
        """
        return self._frame_skip
    
    def set_frame_skip(self, frame_skip: Union[int, float]) -> None:
        """
        Update how many MuJoCo simulation steps run per 'step()' invocation.

        Parameters
        ----------
        frame_skip : int or float
            New frame‐skip value; must be an integer ≥ 1.

        Raises
        ------
        ValueError
            If not an integer ≥ 1, or excessively large values.
        """
        if isinstance(frame_skip, float):
            if frame_skip.is_integer():
                frame_skip = int(frame_skip)
            else:
                raise ValueError(f"Frame skip must be an integer, got {frame_skip}.")

        elif not isinstance(frame_skip, int):
            raise ValueError(f"Frame skip must be an integer, got type {type(frame_skip)}.")

        if frame_skip < 1:
            raise ValueError("Frame skip must be at least 1.")

        if frame_skip > 10:
            logger.warning("Frame skip %d is very high; this may cause instability.", frame_skip)

        self._frame_skip = frame_skip
    
    def get_remove_x_pos(self) -> bool:
        """
        Return the current boolean flag indicating whether to exclude x-directed position.
        """
        return self.remove_x_pos
    
    def set_remove_x_pos(self, remove_x_pos: bool = True) -> None:
        """
        Set whether to exclude the x-directed position from the state.
        
        Args:
            remove_x_pos (bool): If True, exclude x-pos from state. Default True.
        """
        if not isinstance(remove_x_pos, bool):
            raise ValueError(f" remove_x_pos must be a boolean, got {type(remove_x_pos)}.")
        self.remove_x_pos = remove_x_pos
        
    def get_verbose(self) -> bool:
        """
        Return the verbose mode status.
        """
        return self.verbose
    
    def set_verbose(self, verbose: bool) -> None:
        """
        Set the verbose mode status.
        
        Args:
            verbose (bool): Enable or disable verbose mode.
        """
        if not isinstance(verbose, bool):
            raise ValueError(f" verbose must be a bool, got {type(verbose)}.")
        self.verbose = verbose
        if hasattr(self, 'ref_traj'):
            self.ref_traj.verbose = verbose
    
    def get_short_history_length(self) -> int:
        """
        Return the short history buffer length.
        """
        return self.short_history_max_len
    
    def set_short_history_length(self, length: int) -> None:
        """
        Set the short history buffer length.
        
        Args:
            length (int): The new short history length.
        """
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f" Short history length must be a positive integer, got {length}.")
        self.short_history_max_len = length
        self.init_history_buffers()
    
    def get_long_history_length(self) -> int:
        """
        Return the long history buffer length.
        """
        return self.long_history_max_len
    
    def set_long_history_length(self, length: int) -> None:
        """
        Set the long history buffer length.
        
        Args:
            length (int): The new long history length.
        """
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f" Long history length must be a positive integer, got {length}.")
        self.long_history_max_len = length
        self.init_history_buffers()
        
    def set_reward_weights(self, new_weights: Dict[str, Any]) -> None:
        """
        Update the reward weights with the provided values.

        Args:
            new_weights (dict): A dictionary of {key: value} pairs to update in reward_weights.
                                Keys not present in reward_weights will be logged as warnings.

        Raises:
            TypeError: If new_weights is not a dictionary.
        """
        if not isinstance(new_weights, dict):
            raise TypeError(f" new_weights must be a dictionary, got {type(new_weights)}.")

        invalid_keys = [key for key in new_weights if key not in self.reward_weights]
        
        self.reward_weights.update({k: v for k, v in new_weights.items() if k in self.reward_weights})

        if invalid_keys:
            logger.warning("The following keys are invalid and ignored: %s", invalid_keys)
        
    def set_coefficients(self, new_coefficients: Dict[str, Any]) -> None:
        """
        Update the reward coefficients with the provided values.

        Args:
            new_coefficients (dict): A dictionary of {key: value} pairs to update in coefficients.
                                     Keys not present in coefficients will be logged as warnings.

        Raises:
            TypeError: If new_coefficients is not a dictionary.
        """
        if not isinstance(new_coefficients, dict):
            raise TypeError(f" new_coefficients must be a dictionary, got {type(new_coefficients)}.")

        invalid_keys = [key for key in new_coefficients if key not in self.coefficients]
        
        self.coefficients.update({k: v for k, v in new_coefficients.items() if k in self.coefficients})

        if invalid_keys:
            logger.warning("The following coefficient keys are invalid and ignored: %s", invalid_keys)
            
