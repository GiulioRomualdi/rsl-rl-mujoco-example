import gymnasium as gym
import numpy as np
import random
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)


class BaseEnv(gym.Env):
    """
    The most basic abstract environment class. Defines the interface and common utilities
    (seeding, render mode handling, key callbacks, etc.) that all derived environments should inherit.

    Attributes
    ----------
    metadata : dict
        Contains supported render modes and target render FPS.
    render_mode : str or None
        Current render mode ("human" or "rgb_array"), or None if rendering is disabled.
    viewer : Any
        Placeholder for a rendering viewer handle (to be created by subclasses if needed).
    renderer : Any
        Placeholder for an offscreen renderer handle (for "rgb_array" mode).
    _np_random : np.random.Generator
        NumPy RNG instance for reproducibility.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50
    }

    def __init__(self, render_mode: str = None):
        """
        Initialize the BaseEnv. Sets up rendering attributes and RNG placeholders.
        Subclasses should call super().__init__(render_mode) at the beginning of their __init__.

        Parameters
        ----------
        render_mode : str or None
            One of BaseEnv.metadata["render_modes"] or None to disable rendering.
        """
        super().__init__()

        # Validate and store render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. "
                f"Valid options are {self.metadata['render_modes']} or None."
            )
        self.render_mode = render_mode

        # Placeholders for viewer (human) and renderer (rgb_array)
        self.viewer = None
        self.renderer = None

        # Placeholder for NumPy RNG; will be set via self._seed()
        self._np_random = None

    def _seed(self, seed: int = None) -> int:
        """
        Seed all random number generators (Gym, Python, NumPy) for reproducibility.

        Parameters
        ----------
        seed : int or None
            Desired seed. If None, a random seed is chosen via Gym's seeding utility.

        Returns
        -------
        actual_seed : int
            The actual seed used for RNG synchronization.

        Notes
        -----
        - Uses gymnasium's seeding.np_random to generate a reproducible seed for NumPy,
          then applies it to Python's 'random' and NumPy's global RNG.
        - Subclasses should call self._seed(seed) at the start of their reset() implementation.
        """
        from gymnasium.utils import seeding
        self._np_random, actual = seeding.np_random(seed)
        # Convert to 32-bit unsigned int for Python's random seed
        u32 = int(actual) & 0xFFFFFFFF
        random.seed(u32)
        np.random.seed(u32)
        logger.debug(f"BaseEnv RNG seeded with {u32}")
        return u32

    def seed(self, seed: int = None):
        """
        Public alias for setting the RNG seed. Some Gym-based code expects seed().

        Parameters
        ----------
        seed : int or None
            Desired seed.

        Returns
        -------
        [actual_seed] : list[int]
            Returns the actual seed in a one-element list, per Gym convention.
        """
        actual = self._seed(seed)
        return [actual]

    def create_viewer(self):
        """
        Abstract hook for creating the viewer (for "human" mode) or renderer (for "rgb_array" mode).
        Subclasses should override this method, instantiate self.viewer and/or self.renderer,
        and bind key callbacks if desired.
        """
        raise NotImplementedError("create_viewer() must be implemented by the subclass if using rendering.")

    def key_callback(self, keycode):
        """
        Stub for keyboard callback. Subclasses may override to handle keys like:
        - space (toggle pause)
        - 'R' (reset)
        - 'F' (toggle camera follow), etc.

        Parameters
        ----------
        keycode : int
            The integer ASCII/keycode pressed in the viewer window.
        """
        pass

    def render(self):
        """
        Renders the environment in "human" or "rgb_array" mode, depending on render_mode.

        - If render_mode is None: do nothing (no-op), closing any existing viewer/renderer.
        - If render_mode is "human": ensure self.viewer exists, then sync() it and sleep
          for the remaining interval to maintain target FPS.
        - If render_mode is "rgb_array": ensure self.renderer exists, update scene, and return pixels.

        Returns
        -------
        pixels : np.ndarray or None
            If render_mode == "rgb_array", returns the rendered image array (H, W, 3). Otherwise, None.
        """
        if self.render_mode is None:
            # No rendering requested; close any existing viewer/renderer
            if self.viewer is not None or self.renderer is not None:
                self.close()
            return None

        # First‐time creation of viewer/renderer
        if self.viewer is None and self.renderer is None:
            self.create_viewer()

        # HUMAN mode: sync and compensate for rendering time
        if self.render_mode == "human":
            if self.viewer is None:
                raise RuntimeError("Viewer was not created, but render_mode='human' was requested.")

            import time
            start_time = time.perf_counter()
            # Synchronize the viewer to display the current state
            self.viewer.sync()
            # Compute elapsed time and sleep the remainder to achieve target frame interval
            elapsed = time.perf_counter() - start_time
            target_interval = 1.0 / self.metadata["render_fps"]
            to_sleep = target_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            return None

        # RGB_ARRAY mode: update scene and return pixel buffer
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                raise RuntimeError("Renderer was not created, but render_mode='rgb_array' was requested.")

            sim_state = self._get_simulation_state()
            cam = self._get_current_camera()
            # Update the renderer's scene graph with current simulation state
            self.renderer.update_scene(sim_state, camera=cam)
            pixels = self.renderer.render()
            return pixels

        else:
            # Unreachable due to render_mode validation in __init__
            return None

    def _get_simulation_state(self):
        """
        Abstract helper for rgb_array mode. Subclasses that implement an offscreen renderer
        should override this to return the relevant simulation state (e.g., MuJoCo mj_data).
        """
        raise NotImplementedError("_get_simulation_state() must be implemented by the subclass for rgb_array.")

    def _get_current_camera(self):
        """
        Abstract helper for rgb_array mode. Subclasses should override this to select which camera
        to use when rendering offscreen. Return an integer index or camera name, per renderer API.
        """
        return 0  # default camera index

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment and return the initial observation.

        This method follows the Gymnasium API:
          - 'seed': an optional integer seed for reproducibility.
          - 'options': an optional dict for environment-specific reset options.

        Subclasses MUST override this method to:
          1. Reseed RNGs if 'seed' is not None.
          2. Restore their internal state to the episode start.
          3. Call any necessary simulator “forward” methods.
          4. Return (obs, info).

        Args:
            seed: Optional integer seed for all RNGs.
            options: Optional dict of reset parameters.

        Returns:
            obs: np.ndarray
                The initial observation.
            info: dict
                A dict of auxiliary reset info.

        Raises:
            NotImplementedError: If a subclass does not implement this.
        """
        raise NotImplementedError("'reset()' must be implemented by the subclass.")

    def step(self, action):
        """
        Execute one step of the environment's dynamics. Must be overridden by subclasses.

        Parameters
        ----------
        action : np.ndarray
            Action provided by the agent (shape and type defined by derived class).

        Returns
        -------
        obs : np.ndarray
            Next observation.
        reward : float
            Reward for taking the given action.
        terminated : bool
            Whether a terminal condition has been reached.
        truncated : bool
            Whether the episode was truncated (e.g., max steps reached).
        info : dict
            Additional diagnostic information (e.g., debug data, reward breakdown).

        Raises
        ------
        NotImplementedError
            This method must be implemented by the subclass.
        """
        raise NotImplementedError("step() must be implemented by the subclass.")

    def close(self):
        """
        Close and cleanup any rendering or other persistent resources. If a viewer or renderer
        exists, close them. Subclasses can override to clean up simulator or file handles, but
        should always call super().close() to clear self.viewer/self.renderer.
        """
        # Close human viewer if it exists
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            finally:
                self.viewer = None

        # Close offscreen renderer if it exists
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            finally:
                self.renderer = None
