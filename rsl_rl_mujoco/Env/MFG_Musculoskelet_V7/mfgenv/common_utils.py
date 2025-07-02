"""
Common utilities for musculoskeletal simulation environment.
@author: YAKE
"""

import os
import numpy as np
import time
from pathlib import Path
import mujoco
import mujoco.viewer as viewer
from scipy.spatial.transform import Rotation as R
import logging
from typing import Any, List, Tuple, Union, Optional, Dict
import quaternion

# Module-level logger configuration.
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
def validate_file(file_path: Union[str, Path], description: str) -> None:
    """
    Ensure that a given path exists and refers to a regular file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the file to validate. May be a string or Path object.
    description : str
        Human-readable description of the file, used in error messages.

    Raises
    ------
    TypeError
        If 'file_path' is not a str or Path, or if 'description' is not a str.
    FileNotFoundError
        If the path does not exist or is not a file.
    """
    if not isinstance(description, str):
        raise TypeError(f"description must be a str, got {type(description).__name__}")
    if not isinstance(file_path, (str, Path)):
        raise TypeError(f"file_path must be a str or Path, got {type(file_path).__name__}")

    p = Path(file_path).expanduser().resolve()
    logger.debug("Validating %s at path: %s", description, p)

    if not p.exists():
        raise FileNotFoundError(f"{description} does not exist at: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"{description} is not a regular file at: {p}")

def remove_x_position(data: np.ndarray,
                      joint_names: Union[List[str], Tuple[str, ...]],
                      remove_x_pos: bool = True) -> np.ndarray:
    """
    Optionally remove the 'pelvis_tx' entry from a vector of joint positions.

    Parameters
    ----------
    data : np.ndarray
        1D array of joint position values, one per joint.
    joint_names : list of str or tuple of str
        Names corresponding to each entry in 'data'.
    remove_x_pos : bool, optional
        If True and 'pelvis_tx' is found in 'joint_names', remove that entry;
        otherwise return 'data' unchanged. Default is True.

    Returns
    -------
    np.ndarray
        A new 1D array with the same joint values except that the
        'pelvis_tx' element has been removed if requested.

    Raises
    ------
    TypeError
        If 'data' is not a numpy ndarray, or if 'joint_names' is not a list/tuple of str.
    ValueError
        If 'data' length does not match 'joint_names', or if removing would
        leave an empty array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a numpy ndarray, got {type(data).__name__}")
    if not isinstance(joint_names, (list, tuple)) or not all(isinstance(n, str) for n in joint_names):
        raise TypeError("joint_names must be a list or tuple of strings")

    if data.ndim < 1 or data.shape[0] != len(joint_names):
        raise ValueError(
            f"Length mismatch: data has length {data.shape[0]}, "
            f"but joint_names has {len(joint_names)} entries"
        )

    if not remove_x_pos:
        return data

    try:
        idx = joint_names.index('pelvis_tx')
    except ValueError:
        return data

    if data.shape[0] <= 1:
        raise ValueError("Cannot remove 'pelvis_tx': resulting array would be empty")

    return np.delete(data, idx)
        
def add_noise(data: np.ndarray,
              noise_std: float = 2e-3,
              noise_type: str = 'gaussian',
              rng: np.random.Generator = None) -> np.ndarray:
    """
    Add random noise to an array to improve robustness.

    Parameters
    ----------
    data : np.ndarray
        Input array of arbitrary shape.
    noise_std : float, optional
        Noise scale parameter. For 'gaussian', this is the standard deviation;
        for 'uniform', this is the half‐range. Must be non‐negative.
        Default is 2e-3.
    noise_type : {'gaussian', 'uniform'}, optional
        Type of noise distribution to use:
        - 'gaussian': samples from N(0, noise_std^2)
        - 'uniform': samples from U(-noise_std, noise_std)
        Default is 'gaussian'.
    rng : numpy.random.Generator, optional
        NumPy random number generator for reproducibility. If None, a new Generator
        will be created via 'default_rng()'.

    Returns
    -------
    np.ndarray
        A new array with the same shape as 'data', with noise added.

    Raises
    ------
    TypeError
        If 'data' is not a numpy ndarray, or if 'noise_std' is not a float,
        or if 'rng' is provided but is not a numpy.random.Generator.
    ValueError
        If 'noise_std' is negative, or if 'noise_type' is not one of
        'gaussian' or 'uniform'.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a numpy ndarray, got {type(data).__name__}")
    if not isinstance(noise_std, (float, int)):
        raise TypeError(f"noise_std must be a float or int, got {type(noise_std).__name__}")
    noise_std = float(noise_std)
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")

    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")

    noise_type = noise_type.lower()
    if noise_type == 'gaussian':
        noise = rng.normal(loc=0.0, scale=noise_std, size=data.shape)
    elif noise_type == 'uniform':
        noise = rng.uniform(low=-noise_std, high=noise_std, size=data.shape)
    else:
        raise ValueError(f"noise_type must be 'gaussian' or 'uniform', got '{noise_type}'")

    return data + noise

def calculate_frameskip(env: Any, tolerance: float = 1e-8) -> int:
    """
    Compute how many MuJoCo simulation steps to take per environment step.

    The frame skip is given by:

        frame_skip = ref_traj.increment / (opt_time * ref_traj.sample_frequency)

    Parameters
    ----------
    env : Any
        An environment instance providing:
        - 'opt_time' (float): simulation integrator timestep (must be > 0).
        - 'ref_traj.increment' (int or float): number of trajectory frames to advance per step (must be > 0).
        - 'ref_traj.sample_frequency' (int or float): reference trajectory sampling rate in Hz (must be > 0).
    tolerance : float, optional
        Maximum allowable deviation from integer when checking the result 
        (default is 1e-8, to account for floating-point error).

    Returns
    -------
    frame_skip : int
        The integer number of MuJoCo steps to skip per environment step.

    Raises
    ------
    AttributeError
        If 'opt_time', 'ref_traj.increment', or 'ref_traj.sample_frequency' is missing.
    ValueError
        If any of the three parameters is non-positive, or if the computed
        frame_skip differs from the nearest integer by more than 'tolerance',
        or if the resulting frame_skip is less than 1.
    """
    try:
        increment = env.ref_traj.increment
        opt_time = env.opt_time
        sample_freq = env.ref_traj.sample_frequency
    except AttributeError as e:
        raise AttributeError(f"Missing required attribute: {e}")

    if not (isinstance(increment, (int, float)) and increment > 0):
        raise ValueError(f"ref_traj.increment must be > 0, got {increment}")
    if not (isinstance(opt_time, (int, float)) and opt_time > 0):
        raise ValueError(f"opt_time must be > 0, got {opt_time}")
    if not (isinstance(sample_freq, (int, float)) and sample_freq > 0):
        raise ValueError(f"ref_traj.sample_frequency must be > 0, got {sample_freq}")

    raw_skip = increment / (opt_time * sample_freq)
    nearest = round(raw_skip)
    if abs(raw_skip - nearest) > tolerance:
        raise ValueError(
            f"Computed frame_skip = {raw_skip:.6f}, which differs from integer {nearest} "
            f"by more than tolerance {tolerance}. Adjust opt_time, increment, or sample_frequency."
        )

    frame_skip = int(nearest)
    if frame_skip < 1:
        raise ValueError(f"Calculated frame_skip is {frame_skip}, but must be at least 1.")

    return frame_skip

def playback_ref_traj(env: Any, 
                      timestep: int = 1000, 
                      fps: int = 50, 
                      delay: int = 3, 
                      speed_factor: float = 1.0, 
                      start_current: bool = True, 
                      verbose: bool = False) -> None:
    """
    Render the reference gait trajectory in the MuJoCo viewer for debugging and analysis.

    Parameters
    ----------
    env : Any
        A Gymnasium environment instance with:
        - 'model': mujoco.MjModel
        - 'data':  mujoco.MjData
        - 'ref_traj': object providing
            * '.phase' property,
            * '.has_reached_end' property,
            * '.get_reference_trajectories()' -> (qpos, qvel),
            * '.next()' method,
            * '.randomize_initial_state(percent: float)' method.
    timestep : int, optional
        Maximum number of frames to render (default is 1000).
    fps : int, optional
        Target frames per second for timing (must be > 0, default is 60).
    delay : float, optional
        Seconds to pause before and after playback (default is 3.0).
    speed_factor : float, optional
        Playback speed multiplier; effective interval = 1/(fps * speed_factor)
        (must be > 0, default is 1.0).
    start_current : bool, optional
        If True, begin from 'env.ref_traj.phase'; if False, reset to 0% before playback
        (default is True).
    verbose : bool, optional
        If True, log INFO/WARNING messages about playback progress and timing
        (default is False).

    Raises
    ------
    ValueError
        If 'fps' <= 0 or 'speed_factor' <= 0.
    RuntimeError
        If the MuJoCo viewer cannot be launched or 'env.ref_traj' is missing/invalid.
    """
    if fps <= 0 or speed_factor <= 0:
        raise ValueError("fps and speed_factor must be positive values.")
    frame_time = 1 / (fps * speed_factor)
    
    ref = env.ref_traj
    data = env.data
    model = env.model
    get_rt = ref.get_reference_trajectories
    step_rt = ref.next
    has_end = lambda: ref.has_reached_end
    conv_pos = convert_ref_traj_qpos
    conv_vel = convert_ref_traj_qvel
    mj_kin = mujoco.mj_kinematics

    initial_phase = ref.phase
    if not start_current:
        env.randomize_initial_state(0.0)
        if verbose:
            logger.debug("Starting playback from the first frame of the reference trajectory.")
    elif verbose:
        logger.debug("Starting playback from the current phase: %.2f%%.", initial_phase)
    
    try:
        playback_viewer = viewer.launch_passive(model, data)
    except Exception as e:
        raise RuntimeError(f"Failed to launch MuJoCo viewer: {e}")
    
    try:
        if delay > 0:
            time.sleep(delay)
        if verbose:
            logger.debug("Starting reference trajectory playback...( %d ) frames", timestep)

        for frame_idx in range(timestep):
            if has_end():
                if verbose:
                    logger.info("Frame %d: early stopping for traj end", frame_idx)
                break

            start_t = time.perf_counter()
            try:
                qpos, qvel = get_rt()
                qpos = conv_pos(qpos)
                qvel = conv_vel(qvel, qpos)
                if qpos.shape[0] != model.nq:
                    raise ValueError(f"qpos 长度 {qpos.shape[0]} != 模型 DoF {model.nq}")
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                mj_kin(model, data)
                playback_viewer.sync()

            except KeyboardInterrupt:
                logger.warning("User interruption Playback")
                break
            except Exception as exc:
                logger.error("Frame %d-rendering error: %s", frame_idx, exc)
                break

            elapsed = time.perf_counter() - start_t
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            elif verbose:
                logger.warning("Frame %d takes %.4fs, exceeding the target %.4fs", frame_idx, elapsed, frame_time)
            step_rt()
    finally:
        if delay > 0:
            time.sleep(delay)
        playback_viewer.close()
        env.randomize_initial_state(initial_phase)
        if verbose:
            logger.info("Playback ends and restores to original phase %.2f%%", initial_phase)

def compute_site_kinematics(ref_qpos: np.ndarray,
                            ref_qvel: np.ndarray,
                            sim_model: mujoco.MjModel,
                            relative_to_pelvis: bool = False
                            ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute global (or pelvis-relative) positions and linear velocities for all MuJoCo sites
    whose names contain “sensor”, given a reference trajectory.
    
    Parameters
    ----------
    ref_qpos : np.ndarray
        Reference generalized positions of length equal to `sim_model.nq`.
    ref_qvel : np.ndarray
        Reference generalized velocities of length equal to `sim_model.nv`.
    sim_model : mujoco.MjModel
        A loaded MuJoCo model instance used to create a temporary MjData.
    relative_to_pelvis : bool, default=False
        If True, subtract the pelvis translation and velocity so that returned
        positions and velocities are expressed in the pelvis‐centered frame.
    
    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        A mapping from each site name containing “sensor” to a dict with:
          - 'pos': 3-element array of the site’s world (or pelvis-relative) position.
          - 'vel': 3-element array of the site’s world (or pelvis-relative) linear velocity.
    
    Raises
    ------
    ValueError
        If input lengths do not match model dims, conversion routines fail, or data is invalid.
    """
    # Creating A Temporary MuJoCo Data Object
    sim_data = mujoco.MjData(sim_model)

    # Convert the reference trajectory positions and velocities to the simulation coordinate space.
    try:
        d_qpos = convert_ref_traj_qpos(ref_qpos)
        d_qvel = convert_ref_traj_qvel(ref_qvel, ref_qpos)
    except Exception as e:
        raise ValueError(f"Error converting reference trajectory: {e}")
    # Validate the dimensions against the model's expectations.
    if d_qpos.shape[0] != sim_model.nq or d_qvel.shape[0] != sim_model.nv:
        raise ValueError(f"Dimension mismatch: expected qpos={sim_model.nq}, qvel={sim_model.nv}, "
                         f"got qpos={d_qpos.shape[0]}, qvel={d_qvel.shape[0]}")
        
    # Assign converted values to the temporary simulation data.
    sim_data.qpos[:] = d_qpos
    sim_data.qvel[:] = d_qvel
    mujoco.mj_forward(sim_model, sim_data)
    
    if relative_to_pelvis:
        if sim_data.qpos.shape[0] < 3 or sim_data.qvel.shape[0] < 3:
            raise ValueError("Insufficient qpos/qvel for floating-base pelvis state")
        pelvis_pos = sim_data.qpos[:3].copy()
        pelvis_vel = sim_data.qvel[:3].copy()
    else:
        pelvis_pos = np.zeros(3)
        pelvis_vel = np.zeros(3)

    site_kinematics: Dict[str, Dict[str, np.ndarray]] = {}

    for i in range(sim_model.nsite):
        name = sim_model.site(i).name
        if "sensor" not in name.lower():
            continue
        try:
            pos = sim_data.site_xpos[i].copy()
        except Exception as e:
            logger.error("Failed to read site_xpos for %s: %s", name, e)
            raise ValueError(f"Could not retrieve position for site '{name}'")
        # subtract pelvis if requested
        if relative_to_pelvis:
            pos -= pelvis_pos
        site_kinematics[name] = {"pos": pos}
    
    # Extract site velocities via velocimeter sensors
    for i in range(sim_model.nsensor):
       if sim_model.sensor_type[i] != mujoco.mjtSensor.mjSENS_VELOCIMETER:
           continue
       if sim_model.sensor_objtype[i] != mujoco.mjtObj.mjOBJ_SITE:
           continue

       site_id = sim_model.sensor_objid[i]
       site_name = sim_model.site(site_id).name
       if isinstance(site_name, bytes):
           site_name = site_name.decode('utf-8')
       if site_name not in site_kinematics:
           continue

       sensor_reading = np.copy(sim_data.sensordata[i * 3:(i + 1) * 3])
       xmat = np.copy(sim_data.site_xmat[site_id])
       if xmat.size != 9:
           raise ValueError(f"Invalid site rotation matrix for {site_name}.")
       R_site_to_global = xmat.reshape(3, 3)
       vel_global = R_site_to_global @ sensor_reading
       if relative_to_pelvis:
           vel_global -= pelvis_vel

       site_kinematics[site_name]['vel'] = vel_global
     
    # For any sensor site without a velocity sensor, assign a zero velocity.
    epsilon = 1e-6
    for site_name, kin_data in site_kinematics.items():
        if 'vel' not in kin_data:
            kin_data['vel'] = np.zeros(3)
        if relative_to_pelvis:
            kin_data['pos'][np.abs(kin_data['pos']) < epsilon] = 0
            kin_data['vel'][np.abs(kin_data['vel']) < epsilon] = 0

    return site_kinematics

def get_penalty(x: np.ndarray,
                prev_x: Optional[np.ndarray],
                smoothness_weight: float = 0.005,
                clip_range: Optional[Tuple[float, float]] = None,
                enable_clip: bool = False) -> float:
    """
    Compute a smoothness penalty for the change between two vectors.

    Parameters
    ----------
    x : np.ndarray
        Current vector.
    prev_x : np.ndarray or None
        Previous vector of the same shape as 'x', or None to indicate no previous step.
    smoothness_weight : float, optional
        Weight factor (>= 0) applied to the squared norm of the difference.
        Default is 0.005.
    clip_range : tuple of two floats, optional
        If provided, the penalty will be clipped to the interval
        [clip_range[0], clip_range[1]]. If None and 'enable_clip' is True,
        defaults to (0.0, 0.1).
    enable_clip : bool, optional
        If True, apply clipping using 'clip_range'. Default is False.

    Returns
    -------
    float
        The computed (and possibly clipped) penalty.

    Raises
    ------
    TypeError
        If 'x' or 'prev_x' is not convertible to a NumPy array, or if
        'smoothness_weight' is not a non-negative float.
    ValueError
        If 'prev_x' is not None but has a different shape from 'x', or if
        'smoothness_weight' is negative, or if 'clip_range' is invalid,
        or if clipping is enabled but would produce an empty interval.
    """
    if prev_x is None:
        return 0.0

    try:
        x_arr = np.asarray(x, dtype=float)
        prev_arr = np.asarray(prev_x, dtype=float)
    except Exception as e:
        raise TypeError(f"Inputs must be array‐like: {e}")
    if x_arr.shape != prev_arr.shape:
        raise ValueError(f"Shape mismatch: x has shape {x_arr.shape}, prev_x has {prev_arr.shape}")
    if not isinstance(smoothness_weight, (int, float)) or smoothness_weight < 0:
        raise ValueError(f"smoothness_weight must be a non-negative float, got {smoothness_weight}")

    diff = x_arr - prev_arr
    penalty = smoothness_weight * float(np.dot(diff, diff))

    if enable_clip:
        if clip_range is None:
            min_val, max_val = 0.0, 0.1
        else:
            if (not isinstance(clip_range, tuple) or
                len(clip_range) != 2 or
                not all(isinstance(v, (int, float)) for v in clip_range)):
                raise ValueError(f"clip_range must be a tuple of two numbers, got {clip_range}")
            min_val, max_val = float(clip_range[0]), float(clip_range[1])
        if min_val > max_val:
            raise ValueError(f"clip_range lower bound {min_val} exceeds upper bound {max_val}")
        penalty = float(np.clip(penalty, min_val, max_val))

    return penalty

def compute_global_quaternion(tilt: float, list_angle: float, rotation: float) -> quaternion.quaternion:
    """
    Convert local pelvis Euler angles (tilt, list, rotation) into a global orientation quaternion.

    These are composed on top of an initial world-frame pelvis orientation q0 = [√2/2, √2/2, 0, 0].

    Parameters
    ----------
    tilt : float
        Pelvis rotation (radians) about its local z-axis.
    list_angle : float
        Pelvis rotation (radians) about its local x-axis.
    rotation : float
        Pelvis rotation (radians) about its local y-axis.

    Returns
    -------
    quaternion.quaternion
        The resulting global quaternion in [w, x, y, z] format.

    Raises
    ------
    TypeError
        If any of the inputs is not a real number.
    """
    for name, val in (("tilt", tilt), ("list_angle", list_angle), ("rotation", rotation)):
        if not isinstance(val, (int, float, np.floating, np.integer)):
            raise TypeError(f"{name} must be a real number, got {type(val).__name__}")

    q0 = quaternion.quaternion(np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0)

    qz = quaternion.quaternion(np.cos(tilt / 2), 0.0, 0.0, np.sin(tilt / 2))
    qx = quaternion.quaternion(np.cos(list_angle / 2), np.sin(list_angle / 2), 0.0, 0.0)
    qy = quaternion.quaternion(np.cos(rotation / 2), 0.0, np.sin(rotation / 2), 0.0)

    q_inc = qz * qx * qy

    q_global = q0 * q_inc

    logger.debug(
        "Computed global quaternion: [w=%.4f, x=%.4f, y=%.4f, z=%.4f]",
        q_global.w, q_global.x, q_global.y, q_global.z
    )
    return q_global

def compute_local_angles(q_global: quaternion.quaternion) -> Tuple[float, float, float]:
    """
    Extract the pelvis-local Z-X-Y Euler angles (tilt, list, rotation) from a global orientation quaternion.

    Parameters
    ----------
    q_global : quaternion.quaternion
        Global orientation quaternion in [w, x, y, z] format.

    Returns
    -------
    tilt : float
        Pelvis rotation (radians) about its local z-axis.
    list_angle : float
        Pelvis rotation (radians) about its local x-axis.
    rotation : float
        Pelvis rotation (radians) about its local y-axis.

    Raises
    ------
    TypeError
        If 'q_global' is not a 'quaternion.quaternion'.
    RuntimeError
        If the rotation matrix contains invalid values (e.g., due to numerical instability).
    
    Notes
    -----
    - This extraction can suffer from gimbal lock when 'list_angle' is near ±π/2.
    - Ensure that 'q_global' was produced by 'compute_global_quaternion' to maintain consistency.
    """
    if not isinstance(q_global, quaternion.quaternion):
        raise TypeError(f"q_global must be a quaternion.quaternion, got {type(q_global).__name__}")
        
    q0 = quaternion.quaternion(np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0)
    q_inc = q0.inverse() * q_global

    R_mat = quaternion.as_rotation_matrix(q_inc)
    # R = R_z(tilt) * R_x(list) * R_y(rotation) =
    # [c1c3-s1s2s3, -s1c2, c1s3+s1s2c3]
    # [s1c3+c1s2s3,  c1c2, s1s3-c1s2c3]
    # [      -c2s3,    s2,        c2c3]
    
    list_angle = np.arcsin(R_mat[2,1])
    rotation = np.arctan2(-R_mat[2,0], R_mat[2,2])
    tilt = np.arctan2(-R_mat[0,1], R_mat[1,1])
    
    logger.debug("Recovered local angles: tilt=%f, list=%f, rotation=%f", tilt, list_angle, rotation)
    return tilt, list_angle, rotation
    
def convert_ref_traj_qpos(ref_qpos_raw: np.ndarray) -> np.ndarray:
    """
    Convert a 6-dimensional pelvis state (3 translations + 3 Euler angles)
    into a 7-dimensional state (3 translations + 4 quaternion components),
    and append the remaining DOFs unchanged.

    The input vector is assumed to be:
        [tz, ty, tx, tilt, list, rotation, ...other joints...]

    The output becomes:
        [tx, -tz, ty+0.95,  qw, qx, qy, qz, ...other joints...]

    Parameters
    ----------
    ref_qpos_raw : np.ndarray
        1D array of length >= 6 containing the reference pelvis state
        followed by additional joint angles.

    Returns
    -------
    np.ndarray
        1D array of length = len(ref_qpos_raw) + 1, where the first 7 entries
        encode the pelvis as translation+quaternion, and the rest are copied
        from ref_qpos_raw[6:].

    Raises
    ------
    TypeError
        If ref_qpos_raw is not array-like or cannot be converted to 1D float array.
    ValueError
        If ref_qpos_raw has fewer than 6 elements.
    """
    try:
        temp = np.asarray(ref_qpos_raw, dtype=float).ravel()
    except Exception as e:
        raise TypeError(f"ref_qpos_raw must be array-like of numbers: {e}")
    if temp.size < 6:
        raise ValueError(f"Expected at least 6 elements, got {temp.size}")

    pelvis_translation = np.array([temp[2], -temp[0], 0.95 + temp[1]])
    logger.debug("Converted pelvis translation: %s", pelvis_translation)

    # Convert pelvis rotation: use compute_global_quaternion to get the global quaternion
    pelvis_q = compute_global_quaternion(temp[3], temp[4], temp[5])
    pelvis_quat = np.array([pelvis_q.w, pelvis_q.x, pelvis_q.y, pelvis_q.z])
    logger.debug("Converted pelvis quaternion: [%f, %f, %f, %f]", pelvis_q.w, pelvis_q.x, pelvis_q.y, pelvis_q.z)

    # Concatenate the new pelvis state with the rest of the joint states
    new_ref_qpos = np.concatenate((pelvis_translation, pelvis_quat, temp[6:]))
    logger.debug("New qpos dimension: %d", new_ref_qpos.shape[0])
    return new_ref_qpos    
    
def inverse_convert_ref_traj_qpos(d_qpos: np.ndarray) -> np.ndarray:
    """
    Convert a freejoint-format state vector (3 translations + 4-component quaternion + other DOFs)
    back to the reference trajectory format (3 translations + 3 Euler angles + other DOFs).

    This is the inverse of 'convert_ref_traj_qpos', reversing:
      - Translation reordering and offset:
          [tx, -tz, ty + 0.95] → [tz, ty, tx]
      - Quaternion back to Z-X-Y Euler angles.

    Parameters
    ----------
    d_qpos : np.ndarray
        State vector in freejoint format, length >= 7.
        The first 3 elements are the pelvis translation [tx, -tz, ty+0.95],
        and the next 4 are the quaternion [w, x, y, z].

    Returns
    -------
    np.ndarray
        Recovered reference trajectory state vector:
        [tz, ty, tx, tilt, list, rotation, ...other DOFs...], length = len(d_qpos) - 1.

    Raises
    ------
    TypeError
        If 'd_qpos' is not array-like of real numbers.
    ValueError
        If 'd_qpos' has fewer than 7 elements.
    RuntimeError
        If quaternion-to-Euler conversion fails.
    """
    try:
        arr = np.asarray(d_qpos, dtype=float).ravel()
    except Exception as e:
        raise TypeError(f"d_qpos must be array-like of numbers: {e}")
    if arr.size < 7:
        raise ValueError(f"d_qpos must have at least 7 elements, got {arr.size}")
        
    pelvis_trans = arr[0:3]
    w, x, y, z = arr[3:7]

    orig_translation = np.array([-pelvis_trans[1], pelvis_trans[2] - 0.95, pelvis_trans[0]])
    logger.debug("Recovered original pelvis translation: %s", orig_translation)
    
    q = quaternion.quaternion(w, x, y, z)
    try:
        tilt, list_angle, rotation = compute_local_angles(q)
    except Exception as e:
        raise RuntimeError(f"Failed to recover Euler angles from quaternion: {e}")
    logger.debug(
        "Recovered Euler angles: tilt=%.6f, list=%.6f, rotation=%.6f",
        tilt, list_angle, rotation
    )
    
    orig_pelvis_state = np.concatenate((orig_translation, np.array([tilt, list_angle, rotation])))
    orig_ref_qpos = np.concatenate((orig_pelvis_state, d_qpos[7:]))
    logger.debug("Recovered ref qpos dimension: %d", orig_ref_qpos.shape[0])
    
    return orig_ref_qpos

def euler_rates_to_axis_angle(tilt: float, 
                              list_angle: float, 
                              rotation: float,
                              tilt_dot: float, 
                              list_dot: float, 
                              rotation_dot: float,
                              dt: float = 0.001) -> np.ndarray:
    """
    Convert local pelvis Euler angle rates (tilṫ, lisṫ, rotatioṅ) into a global angular velocity vector.

    Parameters
    ----------
    tilt : float
        Current pelvis tilt angle (radians), rotation about local Z-axis.
    list_angle : float
        Current pelvis list angle (radians), rotation about local X-axis.
    rotation : float
        Current pelvis rotation angle (radians), rotation about local Y-axis.
    tilt_dot : float
        Time derivative of tilt (radians/s).
    list_dot : float
        Time derivative of list_angle (radians/s).
    rotation_dot : float
        Time derivative of rotation (radians/s).
    dt : float, default=0.001
        Time step used for finite-difference (seconds).

    Returns
    -------
    np.ndarray
        Global angular velocity vector [ωx, ωy, ωz] in rad/s.

    Raises
    ------
    ValueError
        If 'dt' is not positive.
    RuntimeError
        If quaternion operations fail (e.g., invalid quaternion).

    Notes
    -----
    - Assumes 'compute_global_quaternion' uses the same conventions for (tilt, list, rotation).
    - For small dt, this approximates the true angular velocity; accuracy depends on dt magnitude.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    try:
        q0 = compute_global_quaternion(tilt, list_angle, rotation)
    except Exception as e:
        raise RuntimeError(f"Failed to compute initial quaternion: {e}")
    
    # Integrate Euler angles using simple Euler integration:
    tilt_new     = tilt      + tilt_dot      * dt
    list_new     = list_angle+ list_dot      * dt
    rotation_new = rotation  + rotation_dot  * dt
    
    try:
        q1 = compute_global_quaternion(tilt_new, list_new, rotation_new)
    except Exception as e:
        raise RuntimeError(f"Failed to compute updated quaternion: {e}")
    
    q_delta = q0.inverse() * q1
    
    try:
        rot_vec = quaternion.as_rotation_vector(q_delta)
    except Exception as e:
        raise RuntimeError(f"Failed to extract rotation vector: {e}")
    # Estimate the global angular velocity as the rotation vector divided by dt.
    ang_vel = rot_vec / dt
    logger.debug("Computed angular velocity: %s", ang_vel)
    return ang_vel

def convert_ref_traj_qvel(ref_qvel_raw: np.ndarray, 
                          ref_qpos_raw: np.ndarray, 
                          dt: float = 0.001) -> np.ndarray:
    """
    Convert the pelvis portion of a reference trajectory velocity vector from
    local Euler-rate form into global velocity (linear + axis-angle angular velocity).

    Parameters
    ----------
    ref_qvel_raw : np.ndarray
        1D array with length >= 6: [v_z, v_y, v_x, tilt_dot, list_dot, rotation_dot, ...].
    ref_qpos_raw : np.ndarray
        1D array with length >= 6: [tz, ty, tx, tilt, list, rotation, ...].
    dt : float, default=0.001
        Timestep (s) used for finite-difference conversion of Euler rates to angular velocity.

    Returns
    -------
    np.ndarray
        1D array of the same length as 'ref_qvel_raw'. The first six elements are:
        [v_x, -v_z, v_y, ω_x, ω_y, ω_z], and the rest are 'ref_qvel_raw[6:]'.

    Raises
    ------
    TypeError
        If inputs are not convertible to 1D float arrays.
    ValueError
        If inputs have fewer than 6 elements or if 'dt' is non-positive.
    RuntimeError
        If conversion to angular velocity fails.
    """
    try:
        vel = np.asarray(ref_qvel_raw, dtype=float).ravel()
        pos = np.asarray(ref_qpos_raw, dtype=float).ravel()
    except Exception as e:
        raise TypeError(f"Inputs must be array-like of numbers: {e}")
    if vel.size < 6 or pos.size < 6:
        raise ValueError(f"Both ref_qvel_raw and ref_qpos_raw must have at least 6 elements; "
                         f"got {vel.size} and {pos.size}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
        
    v_z, v_y, v_x = vel[0], vel[1], vel[2]
    tilt_dot, list_dot, rotation_dot = vel[3], vel[4], vel[5]
    tilt, list_angle, rotation = pos[3], pos[4], pos[5]
    
    lin_vel = np.array([v_x, -v_z, v_y], dtype=float)
    logger.debug("Converted linear velocity: %s", lin_vel)
    try:
        ang_vel = euler_rates_to_axis_angle(
            tilt, list_angle, rotation,
            tilt_dot, list_dot, rotation_dot,
            dt
        )
    except Exception as e:
        raise RuntimeError(f"Failed to convert Euler rates to angular velocity: {e}")
    logger.debug("Converted angular velocity: %s", ang_vel)
    pelvis_vel = np.concatenate((lin_vel, ang_vel))
    rest = vel[6:]
    return np.concatenate((pelvis_vel, rest))

def inverse_convert_ref_traj_qvel(d_qvel: np.ndarray, 
                                  ref_qpos_raw: np.ndarray, 
                                  dt: float = 0.001) -> np.ndarray:
    """
    Inversely convert a velocity vector from global (linear + axis-angle angular) form
    back into the reference trajectory format of (local linear + Euler-rate angular).

    Parameters
    ----------
    d_qvel : np.ndarray
        1D array with length >= 6: [v_x, v_y, v_z, ω_x, ω_y, ω_z, ...].
    ref_qpos_raw : np.ndarray
        1D array with length >= 6: [tz, ty, tx, tilt, list, rotation, ...].
    dt : float, default=0.001
        Timestep (seconds) used when converting angular velocity back to Euler-rate.

    Returns
    -------
    np.ndarray
        1D array matching 'd_qvel' length, where the first six elements are:
        [v_z, v_y, v_x, tilt_dot, list_dot, rotation_dot], and the tail is 'd_qvel[6:]'.

    Raises
    ------
    TypeError
        If inputs are not array-like of numbers.
    ValueError
        If input arrays have fewer than 6 elements or 'dt' is non-positive.
    RuntimeError
        If quaternion operations fail (e.g., invalid rotation vector).

    Notes
    -----
    - Assumes 'compute_global_quaternion' and 'compute_local_angles' use consistent conventions.
    - Accuracy depends on small 'dt'; for large angular velocities or dt, integration error may grow.
    """
    try:
        vel = np.asarray(d_qvel, dtype=float).ravel()
        pos = np.asarray(ref_qpos_raw, dtype=float).ravel()
    except Exception as e:
        raise TypeError(f"Inputs must be array-like numeric: {e}")
    if vel.size < 6 or pos.size < 6:
        raise ValueError(f"Both d_qvel and ref_qpos_raw must have at least 6 elements; "
                         f"got {vel.size} and {pos.size}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
        
    d_lin = vel[0:3]
    original_lin = np.array([-d_lin[1], d_lin[2], d_lin[0]])
    
    tilt, list_angle, rotation = pos[3:6]
    
    q0 = compute_global_quaternion(tilt, list_angle, rotation)
    ang_vel = vel[3:6]
    delta_rot = ang_vel * dt
    delta_q = quaternion.from_rotation_vector(delta_rot)    
    q_new = q0 * delta_q
    
    tilt_new, list_new, rotation_new = compute_local_angles(q_new)
    
    euler_rates = np.array([
        (tilt_new - tilt) / dt,
        (list_new - list_angle) / dt,
        (rotation_new - rotation) / dt,
    ])
    
    pelvis_ref_qvel = np.concatenate((original_lin, euler_rates))
    
    new_ref_qvel = np.concatenate((pelvis_ref_qvel, vel[6:]))
    return new_ref_qvel

def get_ref_ee_pos(ref_qpos: np.ndarray, ee: str = 'rightfoot') -> np.ndarray:
    """
    Compute the world‐frame position of a specified end effector from a reference QPOS vector.

    Parameters
    ----------
    ref_qpos : np.ndarray
        Reference trajectory generalized positions. Must be a 1D array of length 41 (full DOFs)
        or 37 (with extra knee DOFs removed). The first six entries correspond to pelvis
        translation (z, y, x) and Euler angles (tilt, list, rotation), followed by joint angles.
    ee : str, optional
        Identifier for the desired end effector. Case‐insensitive aliases supported:
          - 'rightfoot' or 'rf'
          - 'leftfoot'  or 'lf'
          - 'righthand' or 'rh'
          - 'lefthand'  or 'lh'
          - 'head'      or 'h'
        Defaults to 'rightfoot'.

    Returns
    -------
    np.ndarray
        A 3‐element array containing the [x, y, z] coordinates of the specified end effector in world frame.

    Raises
    ------
    TypeError
        If `ref_qpos` is not a numpy array.
    ValueError
        If `ref_qpos` does not have length 41 or 37, or if `ee` is not one of the supported identifiers.
    """
    ee_mapping = {
        'rightfoot': 'rightfoot',
        'rf': 'rightfoot',
        'leftfoot': 'leftfoot',
        'lf': 'leftfoot',
        'righthand': 'righthand',
        'rh': 'righthand',
        'lefthand': 'lefthand',
        'lh': 'lefthand',
        'head': 'head',
        'h': 'head'
    }
    ee = ee_mapping.get(ee.lower(), None)
    if ee is None or ee not in ['rightfoot', 'leftfoot', 'righthand', 'lefthand', 'head']:
        raise ValueError("ee must be one of 'rightfoot', 'leftfoot', 'righthand', 'lefthand', or 'head'.")
    
    if not isinstance(ref_qpos, np.ndarray):
        raise TypeError("[ERROR] ref qpos should be a numpy array.")
    if ref_qpos.shape[0] not in [41, 37]:
        raise ValueError("[ERROR] ref qpos has an invalid shape. Expected 41 or 37 elements.")
    
    # Remove extra knee DOFs if present
    if ref_qpos.shape[0] == 41:
        ref_qpos = np.delete(ref_qpos, [10, 11, 19, 20])
    
    pelvis_tz, pelvis_ty, pelvis_tx = ref_qpos[0:3]
    pelvis_tilt, pelvis_list, pelvis_rotation = ref_qpos[3:6]
    com = np.array([pelvis_tx, pelvis_ty, pelvis_tz])
    
    if ee == 'rightfoot':
        hip_flexion, hip_adduction, hip_rotation = ref_qpos[6:9]
        knee_angle = ref_qpos[9]
        ankle_angle = ref_qpos[10]
        subtalar_angle = ref_qpos[11]

        V = [
            np.array([-0.056276, -0.07849, 0.07726]),
            np.array([-4.6e-07, -0.404425, -0.00126526]),
            np.array([-0.01, -0.4, 0]),
            np.array([-0.04877, -0.04195, 0.00792]),
            np.array([0.1788, -0.002, 0.00108]),
        ]
        
        rotations = [
            R.from_quat([0.7071067811865475, 0.0, 0.0, 0.7071067811865475]),
            R.from_euler('ZXY', [pelvis_tilt, pelvis_list, pelvis_rotation]),
            R.from_euler('ZXY', [hip_flexion, hip_adduction, hip_rotation]),
            R.from_rotvec(knee_angle * np.array([0.0, -0.0707131, -0.997497]) / np.linalg.norm([0.0, -0.0707131, -0.997497])),
            R.from_rotvec(ankle_angle * np.array([-0.105014, -0.174022, 0.979126]) / np.linalg.norm([-0.105014, -0.174022, 0.979126])),
            R.from_rotvec(subtalar_angle * np.array([0.78718, 0.604747, -0.120949]) / np.linalg.norm([0.78718, 0.604747, -0.120949])),
        ]
        
    elif ee == 'leftfoot':
        hip_flexion, hip_adduction, hip_rotation = ref_qpos[13:16]
        knee_angle = ref_qpos[16]
        ankle_angle = ref_qpos[17]
        subtalar_angle = ref_qpos[18]

        V = [
            np.array([-0.056276, -0.07849, -0.07726]),
            np.array([-4.6e-07, -0.404425, 0.00126526]),
            np.array([-0.01, -0.4, 0]),
            np.array([-0.04877, -0.04195, -0.00792]),
            np.array([0.1788, -0.002, -0.00108]),
        ]

        rotations = [
            R.from_quat([0.7071067811865475, 0.0, 0.0, 0.7071067811865475]),
            R.from_euler('ZXY', [pelvis_tilt, pelvis_list, pelvis_rotation]),
            R.from_euler('ZXY', [hip_flexion, -hip_adduction, -hip_rotation]),
            R.from_rotvec(knee_angle * np.array([0.0, 0.0707131, -0.997497]) / np.linalg.norm([0.0, 0.0707131, -0.997497])),
            R.from_rotvec(ankle_angle * np.array([0.105014, 0.174022, 0.979126]) / np.linalg.norm([0.105014, 0.174022, 0.979126])),
            R.from_rotvec(subtalar_angle * np.array([-0.78718, -0.604747, -0.120949]) / np.linalg.norm([-0.78718, -0.604747, -0.120949])),
        ]
    
    elif ee == 'righthand':
        lumbar_extension, lumbar_bending, lumbar_rotation = ref_qpos[20:23]
        arm_flex, arm_add, arm_rot = ref_qpos[23:26]
        elbow_flex = ref_qpos[26]
        pro_sup = ref_qpos[27]

        V = [
            np.array([-0.1007, 0.0815, 0]),
            np.array([0.003155, 0.3715, 0.17]),
            np.array([0.013144, -0.286273, -0.009595]),
            np.array([-0.006727, -0.013007, 0.026083]),
            np.array([-0.008797, -0.235841, 0.01361]),
        ]

        rotations = [
            R.from_quat([0.7071067811865475, 0.0, 0.0, 0.7071067811865475]),
            R.from_euler('ZXY', [pelvis_tilt, pelvis_list, pelvis_rotation]),
            R.from_euler('ZXY', [lumbar_extension, lumbar_bending, lumbar_rotation]),
            R.from_euler('ZXY', [arm_flex, arm_add, arm_rot]),
            R.from_rotvec(elbow_flex * np.array([0.226047, 0.022269, 0.973862]) / np.linalg.norm([0.226047, 0.022269, 0.973862])),
            R.from_rotvec(pro_sup * np.array([0.056398, 0.998406, 0.001952]) / np.linalg.norm([0.056398, 0.998406, 0.001952])),
        ]
    
    elif ee == 'lefthand':
        lumbar_extension, lumbar_bending, lumbar_rotation = ref_qpos[20:23]
        arm_flex, arm_add, arm_rot = ref_qpos[30:33]
        elbow_flex = ref_qpos[33]
        pro_sup = ref_qpos[34]

        V = [
            np.array([-0.1007, 0.0815, 0]),
            np.array([0.003155, 0.3715, -0.17]),
            np.array([0.013144, -0.286273, 0.009595]),
            np.array([-0.006727, -0.013007, -0.026083]),
            np.array([-0.008797, -0.235841, -0.01361]),
        ]

        rotations = [
            R.from_quat([0.7071067811865475, 0.0, 0.0, 0.7071067811865475]),
            R.from_euler('ZXY', [pelvis_tilt, pelvis_list, pelvis_rotation]),
            R.from_euler('ZXY', [lumbar_extension, lumbar_bending, lumbar_rotation]),
            R.from_euler('ZXY', [arm_flex, -arm_add, -arm_rot]),
            R.from_rotvec(elbow_flex * np.array([-0.226047, -0.022269, 0.973862]) / np.linalg.norm([-0.226047, -0.022269, 0.973862])),
            R.from_rotvec(pro_sup * np.array([-0.056398, -0.998406, 0.001952]) / np.linalg.norm([-0.056398, -0.998406, 0.001952])),
        ]

    elif ee == 'head':
        lumbar_extension, lumbar_bending, lumbar_rotation = ref_qpos[20:23]
        
        V = [
            np.array([-0.1007, 0.0815, 0]),
            np.array([0.01, 0.66, 0]),
        ]

        rotations = [
            R.from_quat([0.7071067811865475, 0.0, 0.0, 0.7071067811865475]),
            R.from_euler('ZXY', [pelvis_tilt, pelvis_list, pelvis_rotation]),
            R.from_euler('ZXY', [lumbar_extension, lumbar_bending, lumbar_rotation]),
        ]
    
    ee_pos_in_pelvis = com.copy()
    R_combined = np.eye(3)
    for i in range(len(V)):
        R_combined = R_combined @ rotations[i+1].as_matrix()
        ee_pos_in_pelvis += R_combined @ V[i]
    
    ee_pos_in_ground = rotations[0].as_matrix() @ ee_pos_in_pelvis + np.array([0, 0, 0.95])  # add z-offset
    
    if ee_pos_in_ground.shape[0] != 3:
        raise ValueError(f"Invalid end-effector position shape: expected (3,), got {ee_pos_in_ground.shape}.")
    
    return ee_pos_in_ground

