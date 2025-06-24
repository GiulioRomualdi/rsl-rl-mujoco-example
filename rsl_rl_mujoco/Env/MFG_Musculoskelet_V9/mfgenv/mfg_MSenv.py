import logging
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple
from collections import deque

import numpy as np
from gymnasium import spaces

from .mfg_mujocoenv import MuJoCoEnv
from .ReferTraj_V7 import TrajectoryManager, ReferenceTrajectories
from .common_utils import calculate_frameskip, convert_ref_traj_qpos, convert_ref_traj_qvel
from .state import get_state, get_state_size, compute_ref_site_kinematics, compute_ref_pelvis_kinematics
from .reward import compute_reward
from .termination import check_termination

logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(ch)

class EnvPhase(Enum):
    """Finite‐state machine for imitation → reach‐goal lifecycle."""
    IMITATION = auto()
    REACH_GOAL = auto()

class MFG_Musculoskeletal_V9(MuJoCoEnv):
    """
    Version-9 musculoskeletal environment with reference imitation, reach‐goal,
    smoothness penalties, and optional history buffering.

    Inherits from MuJoCoEnv, adding:
      * a shared ReferenceTrajectories cursor for trajectory imitation
      * a finite‐state 'phase' (IMITATION or REACH_GOAL)
      * motion smoothness and imitation rewards via external modules
      * optional short/long history of (action, obs) pairs

    Attributes:
        traj_manager (TrajectoryManager):
            Shared loader/preprocessor for all reference trajectories.
        ref_traj (ReferenceTrajectories):
            Per-env cursor into the currently selected trajectory.
        enable_reach_goal (bool):
            Whether to enter a reach-goal phase once imitation completes.
        phase (EnvPhase):
            Current phase: IMITATION or REACH_GOAL.
        goal_switch_interval (float):
            Seconds between dynamic goal updates in REACH_GOAL phase.
        goal_pos (np.ndarray):
            [x, y, z] target position for the reach-goal task.
        max_episode_steps (int):
            Maximum number of env.step() calls before truncation.
        step_count (int):
            Number of steps taken since the last reset.
        frame_skip (int):
            Number of MuJoCo simulation steps per environment step, computed dynamically.
        enable_history (bool):
            Whether to record short/long history buffers.
        short_history (deque or None):
            Most recent (action, obs) entries, length = short_history_max_len.
        long_history (deque or None):
            Extended (action, obs) entries, length = long_history_max_len.
        observation_space (spaces.Box):
            Flattened observation space of size `observation_dim`.
    """
    
    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None, **kwargs):
        """
        Initialize the MFG_Musculoskeletal_V9 environment.

        Parameters
        ----------
        config : dict
            -- MuJoCoEnv settings   (passed to super()):
                    model_path          (str): XML model file path (required).
                    default_frame_skip  (int): Physics steps per env.step() (>=1).
                    render_mode         (Optional[str]): "human", "rgb_array", or None.
                    timestep            (Optional[float]): Override integrator timestep.
                    camera              (Optional[int|str]): Offscreen camera ID.

            -- traj_manager        (TrajectoryManager):
                    Shared instance for loading and preprocessing reference trajectories.

            -- High‐level reward weights:
               reward_weights      (dict):
                    'imitation'    (float): Weight for entire imitation term.
                    'smooth'       (float): Weight for entire smoothness term.
                    'goal'         (float): Weight for entire reach‐goal term.

            -- Imitation‐term sub‐weights:
               imitation_weights   (dict):
                    'site_pos'     (float),
                    'site_vel'     (float),
                    'joint_angle'  (float),
                    'joint_angvel' (float).

            -- Smooth‐term sub‐weights:
               smooth_weights      (dict):
                    'grf'          (float),
                    'torque'       (float),
                    'action'       (float).

            -- Per‐term reward coefficients:
               reward_coefficients (dict):
                    'site_pos'     (float),
                    'site_vel'     (float),
                    'joint_angle'  (float),
                    'joint_angvel' (float),
                    'com_speed'    (float),
                    'grf'          (float),
                    'torque'       (float),
                    'action'       (float).

            -- Reach-goal settings:
                    enable_reach_goal   (bool, optional): Enable reach-goal phase. Default False.
                    goal_switch_interval(float, optional): Seconds between goal updates. Default 1.0.


            -- Episode settings:
                    max_episode_steps   (int, optional): Max steps per episode. Default 1000.

            -- Reference‐trajectory cursor:
                    increment            (int, optional): Frames to advance per step (default=1).
                    max_hold_steps       (int, optional): Steps to hold at end of traj before marking done (default=1).
                    traj_seed            (int, optional): Seed for random trajectory selection/phase.
                    rt_verbose           (bool, optional): Whether ReferenceTrajectories prints debug info (default=False).
               
            -- History buffering:
                    enable_history      (bool, optional): Record (action, obs) history. Default False.
                    short_history_max_len(int, optional): Length of short history. Default 5.
                    long_history_max_len (int, optional): Length of long history. Default 30.
        Raises:
            TypeError: If 'config['traj_manager']' is not a TrajectoryManager.
            KeyError:  If any required reward or trajectory keys are missing.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("Initializing MFG_Musculoskeletal_V9")
        
        # Base initialization (physics, action_space, base obs_space, rendering, seeding)
        super().__init__(config, render_mode=render_mode, **kwargs)

        # -------------------------------
        # Shared TrajectoryManager
        # -------------------------------
        from multiprocessing.managers import BaseProxy
        tm = config.get("traj_manager", None)
        if tm is None or not (isinstance(tm, TrajectoryManager) or isinstance(tm, BaseProxy)):
            raise KeyError(
                "Config must contain a 'traj_manager' key with a TrajectoryManager instance or proxy."
            )
        self.traj_manager = tm
        
        # -------------------------------
        # ReferenceTrajectories cursor
        # -------------------------------
        increment = int(config.get("increment", 1))
        max_hold = int(config.get("max_hold_steps", 1))
        rt_seed = config.get("traj_seed", None)
        rt_verbose = bool(config.get("rt_verbose", False))
        self.ref_traj = ReferenceTrajectories(
            manager=self.traj_manager,
            traj_id=None,          # None → choose random trajectory at reset()
            increment=increment,
            max_hold_steps=max_hold,
            random_seed=rt_seed,
            verbose=rt_verbose
        )
        self._update_frame_skip()
        
        # -------------------------------
        # Reward weights and coefficients
        # -------------------------------
        ## High‐level weights for each major term
        rw = config.get("reward_weights", {})
        required_rw = {"imitation", "smooth", "goal"}
        if not required_rw.issubset(rw):
            raise KeyError(f"Config['reward_weights'] must contain keys: {required_rw}")
        self.reward_weights = {k: float(rw[k]) for k in required_rw}
        
        ## Sub‐weights for the imitation term
        iw = config.get("imitation_weights", {})
        expected_i = {"site_pos", "site_vel", "joint_angle", "joint_angvel"}
        if not expected_i.issubset(iw):
            raise KeyError(f"Config['imitation_weights'] must contain keys: {expected_i}")
        self.imitation_weights = {k: float(iw[k]) for k in expected_i}

        ## Sub‐weights for the smoothness term
        sw = config.get("smooth_weights", {})
        expected_s = {"grf", "torque", "action"}
        if not expected_s.issubset(sw):
            raise KeyError(f"Config['smooth_weights'] must contain keys: {expected_s}")
        self.smooth_weights = {k: float(sw[k]) for k in expected_s}

        ## Per‐term reward coefficients
        rc = config.get("reward_coefficients", {})
        expected_rc = {"site_pos", "site_vel", "joint_angle", "joint_angvel",
                       "com_speed", "grf", "torque", "action"}
        if not expected_rc.issubset(rc):
            raise KeyError(f"Config['reward_coefficients'] must contain keys: {expected_rc}")
        self.reward_coefficients = {k: float(rc[k]) for k in expected_rc}

        # -------------------------------
        # Reach‐goal FSM settings
        # -------------------------------
        self.enable_reach_goal = bool(config.get("enable_reach_goal", False))
        self.phase = EnvPhase.IMITATION
        self.goal_switch_interval = float(config.get("goal_switch_interval", 1.0))
        self.goal_pos = np.zeros(3, dtype=np.float32)
        self._next_goal_update = 0.0  # simulation time (s) to next goal recalculation
        
        # -------------------------------
        # Internal state flags & counters
        # -------------------------------
        self.step_count = 0
        self.max_episode_steps = int(config.get("max_episode_steps", 1000))
        self._prev_action = np.zeros((self.num_actuators,), dtype=np.float32)
        self.relative_pelvis = bool(config.get("relative_pelvis", False))
        
        # -------------------------------
        # History recording toggle
        # -------------------------------
        self.enable_history = bool(config.get("enable_history", False))
        if self.enable_history:
            self.short_history_max_len = int(config.get("short_history_max_len", 5))
            self.long_history_max_len  = int(config.get("long_history_max_len", 30))
            self._init_history_buffers()
            self.logger.debug(
                "History enabled: short_max=%d, long_max=%d",
                self.short_history_max_len,
                self.long_history_max_len
            )
        else:
            self.short_history = None
            self.long_history  = None

        # -------------------------------
        # Observation_space
        # -------------------------------
        base_dim = get_state_size(self)
        self._obs_dim = base_dim + 2
        low  = -np.inf * np.ones(self._obs_dim, dtype=np.float32)
        high = +np.inf * np.ones(self._obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )
        
        self.logger.debug("MSenv initiallization finished.")
        
    @property
    def observation_dim(self) -> int:
        """Total flattened observation dimension."""
        return self._obs_dim
    
    def _init_history_buffers(self) -> None:
        """Create deques for short and long history of (action, obs) combos."""
        self.short_history = deque(maxlen=self.short_history_max_len)
        self.long_history  = deque(maxlen=self.long_history_max_len)
        
    def _update_history_buffer(self, action: np.ndarray, obs: np.ndarray) -> None:
        """Append the concatenated (action, obs) to history deques."""
        combo = np.concatenate([action.astype(np.float32),
                                obs.astype(np.float32)], axis=0)
        self.short_history.append(combo)
        self.long_history.append(combo)
        
    def _get_obs(self) -> Tuple[np.ndarray, dict]:
        """
        Construct the full observation vector.

        Observation layout:
          [ progress, cycle_phase, *base_state ]

        - progress (float ∈ [0,1]): 
            Normalized progress along the reference trajectory.
        - cycle_phase (float ∈ [0,1]):
            Phase within one cycle for periodic motions. For non-periodic trajectories, always 0.
        - base_state (np.ndarray):
            The state vector, returned by 'get_state(self)'.

        Returns
        -------
        obs : np.ndarray, shape=(2 + base_dim,)
            Concatenated observation array.
        """
        # Fraction through trajectory
        try:
            pos     = float(self.ref_traj._pos)
            frames  = float(self.ref_traj.traj_frames)
            progress = pos / frames if frames > 0 else 0.0
        except Exception as e:
            raise RuntimeError(f"Error computing progress: {e}")
        
        # Normalized cycle percent, or 0 if non‐periodic
        try:
            cycle_phase = float(self.ref_traj.phase) / 100.0
        except Exception:
            cycle_phase = 0.0
        
        # base_state from external module
        try:
            base_state, components = get_state(self)
        except Exception as e:
            raise RuntimeError(f"get_state(self) failed: {e}")
        base_state = base_state.astype(np.float32, copy=False)
        
        B = base_state.size
        obs = np.empty(2 + B, dtype=np.float32)
        obs[0]        = np.float32(progress)
        obs[1]        = np.float32(cycle_phase)
        obs[2:]       = base_state  # bulk copy
        
        if obs.shape[0] != self._obs_dim:
            raise AssertionError(f"Obs shape mismatch: expected {self._obs_dim}, got {obs.shape[0]}")
    
        return obs, components

    def _update_frame_skip(self) -> None:
        """Recompute self.frame_skip from the current trajectory settings."""
        # calculate_frameskip will raise if something is invalid
        new_fs = calculate_frameskip(self)
        # log if it actually changed
        if hasattr(self, "frame_skip") and new_fs != self.frame_skip:
            self.logger.debug(
                "Frame skip changed from %d → %d", self.frame_skip, new_fs
            )
        self.frame_skip = new_fs
    
    def _update_traj_init_info(self) -> None:
        """
        Prepare and reset the reference trajectory for use.

        Raises
        ------
        ValueError
            If the reference trajectory object is uninitialized or returns invalid data.
        """
        if not hasattr(self, 'ref_traj') or self.ref_traj is None:
            raise ValueError("Reference trajectory object is not initialized.")

        ref_qpos, ref_qvel = self.ref_traj.get_reference_trajectories()
        if ref_qpos is None or ref_qvel is None:
            raise ValueError("Failed to retrieve reference trajectories.")
        
        self._init_qvel = convert_ref_traj_qvel(ref_qvel, ref_qpos)
        self._init_qpos = convert_ref_traj_qpos(ref_qpos)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        traj_id: Optional[int] = None,
        frame: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment state, reference cursor, phase, frame_skip, and history.

        This override does the following in addition to MuJoCoEnv.reset():
          1. Reset the ReferenceTrajectories cursor to a (possibly new) trajectory.
          2. Reset the FSM phase back to IMITATION.
          3. Clear step counter, goal position, and next‐goal timer.
          4. Recompute frame_skip to match the new trajectory parameters.
          5. Reinitialize history buffers if 'enable_history' is True.
          6. Populate 'info' with traj_id, progress, phase, and empty histories.

        Parameters
        ----------
        seed : Optional[int]
            Random seed to pass through to the base reset.
        options : Optional[Dict[str, Any]]
            Additional options for the base reset (unused here).

        Returns
        -------
        obs : np.ndarray
            Initial observation after reset, as returned by '_get_obs()'.
        info : Dict[str, Any]
            Diagnostic info including:
              - 'traj_id'      : int
              - 'progress'     : float, initial progress (0.0)
              - 'phase'        : str, 'IMITATION'
              - 'short_history': list (empty if enabled)
              - 'long_history' : list (empty if enabled)
        """
        if seed is not None:
            self._seed(seed)
        
        self.ref_traj.reset(
            seed=seed,
            phase=None,
            randomize_traj=(traj_id is None),
            traj_id=traj_id
            )
        if frame is not None:
            self.ref_traj._pos = frame
            self.ref_traj._has_reached_end = (frame >= self.ref_traj.traj_frames - 1)
        self._update_traj_init_info()
        
        reset_opts = {} if options is None else options.copy()
        obs, info = super().reset(options=reset_opts)

        self.phase = EnvPhase.IMITATION
        self.step_count = 0
        self.goal_pos = np.zeros(3, dtype=np.float32)
        self._next_goal_update = 0.0

        self._update_frame_skip()

        # Clear history if enabled
        if self.enable_history:
            self._init_history_buffers()
            info["short_history"] = []
            info["long_history"]  = []

        info.update({
            "traj_id": self.ref_traj.traj_id,
            "progress": float(self.ref_traj._pos) / float(self.ref_traj.traj_frames),
            "phase": self.phase.name
        })

        self.logger.debug("Environment reset complete: traj_id=%s", info["traj_id"])
        return obs, info

    def step(
        self,
        action: np.ndarray
        ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one time-step and compute task-specific reward and termination.

        Steps performed:
          1. Delegate physics, rendering, buffer-recording to MuJoCoEnv.step().
          2. Advance the reference-trajectory cursor.
          3. Update reach-goal phase and possibly sample a new goal.
          4. Compute imitation, smoothness, and goal-reaching rewards.
          5. Check environment termination and truncation conditions.
          6. Assemble 'info' dict with diagnostics.

        Parameters
        ----------
        action : np.ndarray, shape=(num_actuators,)
            Control inputs for each actuator.

        Returns
        -------
        obs : np.ndarray
            Next observation as built by '_get_obs()'.
        reward : float
            Scalar task reward combining all terms.
        terminated : bool
            True if a terminal condition (e.g., fall) is met.
        truncated : bool
            True if the episode is truncated (e.g., ref_traj finished or max steps reached).
        info : dict
            Diagnostics, including per-term reward breakdown and termination details.
        """
        obs, _, _, _, info = super().step(action)
        
        comp = info["obs_components"]
        sim_joint = comp["joint"]
        sim_pel   = comp["pelvis"]
        if sim_joint is None or sim_pel is None:
            raise KeyError("info['obs_components'] must contain keys 'joint' and 'pelvis'")
        
        self.ref_traj.next()
        self.step_count += 1
        sim_time = self.step_count * self.opt_time * self.frame_skip
        
        ref_kin = compute_ref_site_kinematics(self)
        ref_pel = compute_ref_pelvis_kinematics(self, use_free_joint=True)
        
        comp_trim = {
            "joint":   sim_joint,
            "pelvis":  sim_pel,
            "ref_kin": ref_kin,
            "ref_pel": ref_pel
        }
        
        # FSM transition: IMITATION → REACH_GOAL
        if self.enable_reach_goal:
            if (self.phase is EnvPhase.IMITATION and
                self.ref_traj.has_reached_end):
                self.phase = EnvPhase.REACH_GOAL
                self._next_goal_update = sim_time + self.goal_switch_interval
            if (self.phase is EnvPhase.REACH_GOAL and
                sim_time >= self._next_goal_update):
                self.goal_pos = self._sample_goal_position()
                self._next_goal_update += self.goal_switch_interval
        
        total_reward, reward_info = compute_reward(self, comp_trim)
        terminated, term_info = check_termination(self, obs_components=comp_trim, conditions=["has_fallen", "site_deviation_exceeded"])
        
        at_max = (self.step_count >= self.max_episode_steps)
        if self.enable_reach_goal:
            truncated = at_max
        else:
            truncated = self.ref_traj.has_reached_end or at_max

        if self.enable_history:
            self._update_history_buffer(action, obs)
            
        # Build info dict
        info.update({
            "reward_info": reward_info,
            "terminated_info": term_info,
            "phase": self.phase.name,
            "traj_id": self.ref_traj.traj_id,
            "progress": self.ref_traj._pos / self.ref_traj.traj_frames,
        })
        if self.enable_history:
            info["short_history"] = list(self.short_history)
            info["long_history"]  = list(self.long_history)
            
        self.logger.debug(
            "Step %d | reward=%.4f | term=%s | trunc=%s",
            self.step_count, total_reward, terminated, truncated
        )

        return obs, total_reward, terminated, truncated, info
    
    def _sample_goal_position(self) -> np.ndarray:
        pass
    
    @property
    def pelvis_heading(self) -> np.ndarray:
        """
        Compute the horizontal heading direction of the pelvis.

        Returns
        -------
        heading : np.ndarray, shape=(3,)
            Unit vector in world coordinates representing the pelvis heading.

        Raises
        ------
        ValueError
            If the pelvis body ID cannot be found or the rotation matrix
            is malformed (e.g., not a 3x3 matrix).

        Notes
        -----
        - World coordinate axes:
          X: forward (anterior), Y: leftward, Z: upward.
        - The pelvis local frame at initialization aligns its X-axis
          with the world X-axis, Y-axis with the world Z-axis, and
          Z-axis with the world Y-axis.
        - This method ignores any tilt (rotation about Z) and list
          (rotation about X) to extract only the yaw rotation.
        """
        try:
            pelvis_body = self.model.body('pelvis')
            R_flat = self.data.xmat[pelvis_body.id]
        except Exception as e:
            raise ValueError(f"Failed to retrieve pelvis rotation: {e}")

        R = np.asarray(R_flat, dtype=float)
        if R.size != 9:
            raise ValueError(f"Invalid rotation matrix size: expected 9 elements, got {R.size}")
        R = R.reshape(3, 3)

        x_axis = R[:, 0].copy()
        x_axis[2] = 0.0
        norm = np.linalg.norm(x_axis)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=float)

        return (x_axis / norm).astype(float)

    def close(self) -> None:
        """Close renderers and clean up resources."""
        self.logger.debug("Closing environment")
        super().close()
        self.logger.debug("Environment closed successfully")
        