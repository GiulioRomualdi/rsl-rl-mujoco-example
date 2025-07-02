import os
import time
import logging
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import mujoco
import mujoco.viewer as mujviewer
from gymnasium import spaces

from .mfg_baseenv import BaseEnv
from .mujoco_utils import (
    load_mujoco_model,
    check_invalid_names,
    parse_actuator_prm_from_xml
)

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(ch)

class MuJoCoEnv(BaseEnv):
    """
    A MuJoCo-based Gymnasium environment.

    Attributes
    ----------
    model : mujoco.MjModel
        Compiled MuJoCo model with only the actuators you intend to control.
    data : mujoco.MjData
        Simulation state corresponding to 'model'.
    action_space : spaces.Box
        Continuous controls matching 'model.actuator[i].ctrlrange'.
    observation_space : spaces.Box
        Concatenated [qpos, qvel] with ±inf bounds by default.
    frame_skip : int
        Number of physics steps per call to 'step()'.
    auto_render : bool
        If True, automatically calls 'render()' after 'reset()' and 'step()'.
    num_dofs : int
        Number of generalized coordinates (model.nv).
    num_actuators : int
        Number of actuators (model.nu).
    jnt_names : List[str]
        Names of all joints.
    actuator_names : List[str]
        Names of all actuators.
    """

    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None, **kwargs):
        """
        Initialize the MuJoCo environment.

        Args
        ----
        config : dict
            Required keys:
              - 'model_path' (str): Path to MJCF XML file.
              - 'default_frame_skip' (int): Physics steps per env.step().
            Optional keys:
              - 'render_mode' (str or None): One of BaseEnv.metadata["render_modes"].
              - 'timestep' (float): Override the XML integrator timestep.
              - 'camera' (int or str): Camera index or name for offscreen.
              - 'use_muscle' (bool): If True, keep group 1+3 actuators; else 1+2.
              - 'auto_render' (bool): If True, call render() automatically.

        Raises
        ------
        KeyError
            If required config keys are missing.
        ValueError
            If 'default_frame_skip' < 1.
        RuntimeError
            If model loading or compilation fails.
        """
        logger.debug("Initializing MuJoCoEnv")
        # ---------------------------------------------------------------
        # Validate config keys
        # ---------------------------------------------------------------
        if "model_path" not in config:
            raise KeyError("Config must contain 'model_path'.")
        if "default_frame_skip" not in config:
            raise KeyError("Config must contain 'default_frame_skip'.")
            
        model_path = config["model_path"]
        default_fs = int(config["default_frame_skip"])
        if default_fs < 1:
            raise ValueError(f"default_frame_skip must be >= 1, got {default_fs}.")
        
        rm = render_mode if render_mode is not None else config.get("render_mode", None)
        timestep    = config.get("timestep", None)
        camera      = config.get("camera", 0)
        use_muscle  = bool(config.get("use_muscle", False))
        self.auto_render = bool(config.get("auto_render", False))
        if rm is None:
            self.auto_render = False

        # ---------------------------------------------------------------
        # Initialize BaseEnv (sets render_mode, metadata, seed stub)
        # ---------------------------------------------------------------
        super().__init__(render_mode=rm)

        # ---------------------------------------------------------------
        # Load & compile model/data, filtering actuators by use_muscle
        # ---------------------------------------------------------------
        try:
            self.model, self.data = load_mujoco_model(model_path, use_muscle)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {e}")
            
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Override integrator timestep if provided
        if timestep is not None:
            self.model.opt.timestep = float(timestep)
        self.opt_time = float(self.model.opt.timestep)

        # ---------------------------------------------------------------
        # Extract model dimensions & names
        # ---------------------------------------------------------------
        self.num_dofs       = self.model.nv
        self.num_joints     = self.model.njnt
        self.num_actuators  = self.model.nu
        self.total_mass     = float(self.model.body_mass[1:].sum())

        self.jnt_names      = [self.model.joint(i).name for i in range(self.num_joints)]
        self.actuator_names = [self.model.actuator(i).name for i in range(self.num_actuators)]
        check_invalid_names(self.jnt_names, self.actuator_names, use_muscle)
        
        all_prms = parse_actuator_prm_from_xml(model_path)
        self.actuator_prms = {nm: all_prms[nm] for nm in self.actuator_names if nm in all_prms}
        
        self._ensure_joint_site_metadata()
        
        # ---------------------------------------------------------------
        # Interaction flags & frame skip
        # ---------------------------------------------------------------
        self.frame_skip     = default_fs
        self.paused         = False
        self.follow         = False
        self.disable_reset  = False
        self.camera         = camera
        
        # ---------------------------------------------------------------
        # Build action_space from ctrlrange of each actuator
        # ---------------------------------------------------------------
        lows  = [self.model.actuator(i).ctrlrange[0] for i in range(self.num_actuators)]
        highs = [self.model.actuator(i).ctrlrange[1] for i in range(self.num_actuators)]
        self.action_space = spaces.Box(
            low = np.array(lows,  dtype=np.float32),
            high= np.array(highs, dtype=np.float32),
            dtype=np.float32
        )

        # ---------------------------------------------------------------
        # Build observation_space (qpos + qvel)
        # ---------------------------------------------------------------
        inf = np.inf
        pos_low  = -inf * np.ones(self._init_qpos.shape, dtype=np.float32)
        pos_high = +inf * np.ones(self._init_qpos.shape, dtype=np.float32)
        vel_low  = -inf * np.ones(self._init_qvel.shape, dtype=np.float32)
        vel_high = +inf * np.ones(self._init_qvel.shape, dtype=np.float32)
        obs_low  = np.concatenate([pos_low, vel_low], axis=0)
        obs_high = np.concatenate([pos_high, vel_high], axis=0)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ---------------------------------------------------------------
        # Prepare history buffers for qpos, qvel, ctrl
        # ---------------------------------------------------------------
        self.qpos_history = []
        self.qvel_history = []
        self.ctrl_history = []
        
        logger.debug("MuJoCoEnv initialized: num_dofs=%d, num_actuators=%d, frame_skip=%d",
                     self.num_dofs, self.num_actuators, self.frame_skip)

    def create_viewer(self) -> None:
        """
        Instantiate the MuJoCo viewer or offscreen renderer based on 'self.render_mode'.
        Called by BaseEnv.render() on first invocation.

        - "human": launches a passive GLFW viewer and registers 'key_callback'.
        - "rgb_array": creates an offscreen Renderer and primes the first frame.
        - None: ensures no viewer or renderer is active.

        Raises
        ------
        RuntimeError
            If 'render_mode' is not one of {None, "human", "rgb_array"} or
            if creation of the viewer/renderer fails.
        """
        if getattr(self, "viewer", None) is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        if getattr(self, "renderer", None) is not None:
            self.renderer = None
            
        if self.render_mode == "human":
            try:
                self.viewer = mujviewer.launch_passive(
                    self.model,
                    self.data,
                    key_callback=self.key_callback
                )
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = 1
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.distance  = 5      
                self.viewer.cam.azimuth   = 135.0
                self.viewer.cam.elevation = -20.0
            except Exception as e:
                raise RuntimeError(f"Failed to launch human-mode viewer: {e}")
        elif self.render_mode == "rgb_array":
            try:
                self.renderer = mujoco.Renderer(self.model)
                # Prime the scene so the first render() call works immediately
                mujoco.mj_forward(self.model, self.data)
                self.renderer.update_scene(self.data)
            except Exception as e:
                raise RuntimeError(f"Failed to create offscreen renderer: {e}")
        elif self.render_mode is None:
            # No visualization requested
            self.viewer = None
            self.renderer = None
        else:
            raise RuntimeError(f"Unsupported render_mode: {self.render_mode}")

    def key_callback(self, keycode: int) -> None:
        """
        Handle keyboard events in the human-mode viewer.

        Supported keys:
          - Space (" "): Toggle pause/resume of stepping.
          - "R":         Reset the environment (if 'disable_reset=False').
          - "F":         Toggle camera follow mode.
          - "T":         Toggle fast-forward (skip render sleep).
          - "M":         Toggle 'disable_reset' flag.
          - "D":         Print the first three qpos/qvel for debugging.
          - "S":         Save an RGB screenshot (only in "rgb_array" mode).
          - "G":         Trigger history plotting (qpos, qvel, ctrl).

        Notes
        -----
        This callback is only active when 'render_mode=="human"'.
        """
        try:
            c = chr(keycode)
        except Exception:
            return

        if c == " ":
            self.paused = not self.paused
            print(f"[MuJoCoEnv] Paused = {self.paused}")

        elif c.upper() == "R":
            if not self.disable_reset:
                print("[MuJoCoEnv] Resetting environment.")
                self.reset()
            else:
                print("[MuJoCoEnv] Reset is disabled.")

        elif c.upper() == "F":
            self.follow = not self.follow
            print(f"[MuJoCoEnv] Follow = {self.follow}")

        elif c.upper() == "M":
            self.disable_reset = not self.disable_reset
            print(f"[MuJoCoEnv] Disable reset = {self.disable_reset}")

        elif c.upper() == "D":
            qpos_sample = self.data.qpos[:3] if self.num_dofs >= 3 else self.data.qpos
            qvel_sample = self.data.qvel[:3] if self.num_dofs >= 3 else self.data.qvel
            print(f"[MuJoCoEnv] qpos[:3]={qpos_sample}, qvel[:3]={qvel_sample}")

        elif c.upper() == "S" and self.render_mode == "rgb_array":
            img = self.renderer.render()
            Image.fromarray(img).save("screenshot.png")
            print("[MuJoCoEnv] Screenshot saved to screenshot.png")

        elif c.upper() == "G":
            self._plot_history()
    
    def _plot_history(self):
        """
        Plot and save joint position, velocity, and actuator control histories.

        This method creates a 3-panel figure showing:
          1. The first two joint positions over time.
          2. The first two joint velocities over time.
          3. The first two actuator commands over time.

        The plot is saved as a timestamped PNG in the 'history_plots/' folder,
        which is created if it does not already exist.

        Raises
        ------
        RuntimeError
            If no history data is available when attempting to plot.
        """
        if not (self.qpos_history and self.qvel_history and self.ctrl_history):
            print("[MuJoCoEnv] No history to plot.")
            return

        # Convert lists to arrays
        qpos_arr = np.stack(self.qpos_history, axis=0)
        qvel_arr = np.stack(self.qvel_history, axis=0)
        ctrl_arr = np.stack(self.ctrl_history, axis=0)
        steps = np.arange(qpos_arr.shape[0], dtype=int)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Simulation History", fontsize=16)

        # Plot first two qpos dimensions
        for i in range(min(2, self.num_dofs)):
            axes[0].plot(steps, qpos_arr[:, i], label=f"qpos[{i}]")
        axes[0].set_ylabel("qpos")
        axes[0].legend(loc="upper right")
        axes[0].grid(True)
        
        # Plot first two qvel dimensions
        for i in range(min(2, self.num_dofs)):
            axes[1].plot(steps, qvel_arr[:, i], label=f"qvel[{i}]")
        axes[1].set_ylabel("qvel")
        axes[1].legend(loc="upper right")
        axes[1].grid(True)

        # Plot first two ctrl dimensions
        for j in range(min(2, self.num_actuators)):
            axes[2].plot(steps, ctrl_arr[:, j], label=f"ctrl[{j}]")
        axes[2].set_ylabel("ctrl")
        axes[2].set_xlabel("Step")
        axes[2].legend(loc="upper right")
        axes[2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        folder = "history_plots"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"history_{timestamp}.png"    
        save_path = os.path.join(folder, filename)
        
        plt.savefig(save_path)
        plt.close(fig)
        print("[MuJoCoEnv] History plot saved to 'history.png'.")

    def _get_simulation_state(self):
        """
        Provide the current MjData to the offscreen renderer.
        Called by BaseEnv.render() in "rgb_array" mode.
        """
        return self.data

    def _get_current_camera(self):
        """
        Provide the camera index or name for offscreen rendering.
        Called by BaseEnv.render() in "rgb_array" mode.
        """
        cam = self.camera
        if cam >= self.model.ncam or cam < -1:
            return -1
        return cam
    
    def _clear_history(self) -> None:
        """
        Clear all stored simulation history buffers.
        """
        self.qpos_history.clear()
        self.qvel_history.clear()
        self.ctrl_history.clear()
        logger.debug("Simulation history buffers have been cleared.")
    
    def _cleanup_plots(
            self,
            folder: str = "history_plots",
            max_age_days: int = 3
            ) -> None:
        """
        Remove old plot files to prevent disk bloat.

        Scans the specified `folder` for files and deletes any whose
        modification time is older than `max_age_days`. If the folder
        does not exist, the method exits silently.

        Args:
            folder (str): Directory where historical plots are saved.
                Defaults to "history_plots".
            max_age_days (int): Files older than this many days will
                be deleted. Defaults to 7 days.

        Returns:
            None
        """
        if not os.path.isdir(folder):
            logger.debug("No plot directory found at '%s'; skipping cleanup.", folder)
            return

        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        removed_files = 0

        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            try:
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_mtime < cutoff_time:
                    os.remove(filepath)
                    removed_files += 1
                    logger.debug("Removed old plot file: %s", filepath)
            except Exception as err:
                logger.warning("Could not remove file '%s': %s", filepath, err)

        logger.debug(
            "Cleanup complete: %d file(s) older than %d day(s) removed from '%s'.",
            removed_files, max_age_days, folder
        )
    
    def reset(
            self, 
            *, 
            seed: Optional[int] = None, 
            options: Dict[str, Any] = None
            ) -> Tuple[np.ndarray, dict]:
        """
        Reset the MuJoCo simulation to its initial state (or a custom state).

        This implements the Gymnasium API:
          - Uses 'seed' to reseed RNGs.
          - Accepts an 'options' dict; if it contains "custom_init_state",
            that dict may supply "qpos" and/or "qvel" arrays to override
            the factory defaults.
          - Clears internal history buffers.
          - Primes the offscreen renderer if in "rgb_array" mode.
          - Auto‐renders the first frame if enabled.

        Args:
            seed: Optional integer seed to make resets reproducible.
            options: Optional dict. May contain:
                "custom_init_state": {
                    "qpos": np.ndarray of length nv,
                    "qvel": np.ndarray of length nv
                }

        Returns:
            obs (np.ndarray): The initial observation, concatenated [qpos, qvel].
            info (dict): Empty dict (override in subclasses if needed).

        Raises:
            ValueError: If a provided custom_init_state vector has incorrect length.
        """
        logger.debug("Resetting environment (seed=%s)", seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Handle options and custom init
        options = options or {}
        custom = options.get("custom_init_state", None)
        
        # Reseed RNGs for reproducibility
        if seed is not None:
            self._seed(seed)

        # Restore qpos/qvel
        if custom is not None:
            if "qpos" in custom:
                qpos = np.asarray(custom["qpos"], dtype=np.float64)
                if qpos.shape != self._init_qpos.shape:
                    raise ValueError(f"custom qpos shape {qpos.shape} != {self._init_qpos.shape}")
                self.data.qpos[:] = qpos
            else:
                self.data.qpos[:] = self._init_qpos

            if "qvel" in custom:
                qvel = np.asarray(custom["qvel"], dtype=np.float64)
                if qvel.shape != self._init_qvel.shape:
                    raise ValueError(f"custom qvel shape {qvel.shape} != {self._init_qvel.shape}")
                self.data.qvel[:] = qvel
            else:
                self.data.qvel[:] = self._init_qvel
        else:
            self.data.qpos[:] = self._init_qpos.copy()
            self.data.qvel[:] = self._init_qvel.copy()
        self.data.ctrl[:] = 0.0

        # Forward pass to update geometry, sensors, etc.
        mujoco.mj_forward(self.model, self.data)
        
        # Clear any stored history
        self._clear_history()

        # Update offscreen renderer if needed
        if self.render_mode == "rgb_array" and self.renderer:
            self.renderer.update_scene(self.data)

        # Visualize this reset state
        if self.render_mode is not None and self.auto_render:
            _ = self.render()

        # Return dummy placeholders; subclasses must override to return real obs and info
        obs, components = self._get_obs()
        info = {"obs_components": components}
        logger.debug("Reset done, obs shape=%s", obs.shape)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance the simulation by one environment step using the given action.

        This method implements the Gymnasium step API:
          1. Validate the action’s shape and bounds against 'self.action_space'.
          2. Apply the action to 'self.data.ctrl' and perform 'self.frame_skip' physics steps.
          3. Record qpos, qvel, and ctrl into history buffers.
          4. Update camera (human mode) and renderer (rgb_array mode) as needed.
          5. Auto‐render the frame if 'self.auto_render' is True.
          6. Construct the next observation, then return
             (obs, reward, terminated, truncated, info).

        Args
        ----
        action : np.ndarray of shape (num_actuators,)
            Continuous control inputs for each actuator.

        Returns
        -------
        obs        : np.ndarray
            Concatenated [qpos, qvel] after the step.
        reward     : float
            Placeholder reward (always 0.0 here; override in subclasses).
        terminated : bool
            Episode termination flag (always False here; override in subclasses).
        truncated  : bool
            Episode truncation flag (always False here; override in subclasses).
        info       : dict
            Auxiliary information (empty by default; override in subclasses).

        Raises
        ------
        ValueError
            If 'action' has incorrect shape or contains out‐of‐bounds values.
        """
        logger.debug("Starting step")
        while self.paused:
            if self.render_mode is not None and self.auto_render:
                _ = self.render()
                time.sleep(1/self.metadata["render_fps"])
            
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.num_actuators,):
            raise ValueError(f"Action shape mismatch: expected ({self.num_actuators},), got {action.shape}")
        # if not self.action_space.contains(action):
        #     print(self.action_space)
        #     raise ValueError(f"Action values {action} out of bounds for {self.action_space}")
            
        # Physics stepping
        if not self.paused:
            self.data.ctrl[:] = action                
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
            
        self.qpos_history.append(self.data.qpos.copy())
        self.qvel_history.append(self.data.qvel.copy())
        self.ctrl_history.append(action.copy())

        # Human‐mode camera follow
        if self.render_mode == "human" and self.follow and self.viewer:
            self.viewer.cam.lookat = self.data.qpos[:3]

        # Offscreen rendering update
        if self.render_mode == "rgb_array" and self.renderer:
            self.renderer.update_scene(self.data)

        # Auto‐render if enabled
        if self.render_mode is not None and self.auto_render:
            _ = self.render()

        # Build observation
        obs, components = self._get_obs()
        if not self.observation_space.contains(obs):
            raise ValueError(f"Observation {obs} out of bounds for {self.observation_space}")
        
        # Return placeholders for reward/termination/info
        reward     = 0.0
        terminated = False
        truncated  = False
        info       = {"obs_components": components}
        logger.debug("Step completed: obs_shape=%s", obs.shape)
        logger.debug("History lengths: qpos=%d, qvel=%d, ctrl=%d",
                     len(self.qpos_history), len(self.qvel_history), len(self.ctrl_history))
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """
        Unify functions that construct observation vectors for easy subclass coverage.
        """
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        return np.concatenate([qpos, qvel], axis=0).astype(np.float32), {}
    
    def _ensure_joint_site_metadata(self):
        """
        Populate self._joint_sites and self._pelvis_gyro exactly once,
        mirroring the logic in get_joint_kinematics().
        """
        if hasattr(self, "_joint_sites") and hasattr(self, "_pelvis_gyro"):
            return

        model = self.model
        temp: Dict[int, Dict[str, Any]] = {}
        for si in range(model.nsensor):
            if model.sensor_objtype[si] != mujoco.mjtObj.mjOBJ_SITE:
                continue
            sid, adr, dim, st = (
                model.sensor_objid[si],
                model.sensor_adr[si],
                model.sensor_dim[si],
                model.sensor_type[si],
            )
            if sid not in temp:
                name = model.site(sid).name
                if isinstance(name, bytes):
                    name = name.decode()
                temp[sid] = {"name": name, "vel": None, "gyr": None}
            if st == mujoco.mjtSensor.mjSENS_VELOCIMETER:
                temp[sid]["vel"] = (adr, dim)
            elif st == mujoco.mjtSensor.mjSENS_GYRO:
                temp[sid]["gyr"] = (adr, dim)

        try:
            pelvis_sid = model.site("pelvis_sensor").id
        except Exception:
            pelvis_sid = None

        joint_sites: List[Tuple[int, str, Tuple[int,int], Tuple[int,int]]] = []
        pelvis_gyro: Tuple[int,int] = (None, 0)

        for sid, info in temp.items():
            if sid == pelvis_sid:
                pelvis_gyro = info["gyr"] or (None, 0)
            elif info["vel"] is not None:
                gyr = info["gyr"] or (None, 0)
                joint_sites.append((sid, info["name"], info["vel"], gyr))

        if not joint_sites:
            raise ValueError("No joint-site sensors found in the model.")

        self._joint_sites = joint_sites
        self._pelvis_gyro = pelvis_gyro

    def close(self):
        """
        Close MuJoCo viewer/renderer and release resources.
        """
        logger.debug("Closing MuJoCoEnv and cleaning up resources")
        self._clear_history()
        self._cleanup_plots()
        if hasattr(self, "viewer") and self.viewer is not None:
            try: self.viewer.close()
            except: pass
            self.viewer = None
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer = None
        super().close()
        logger.debug("MuJoCoEnv closed successfully")
