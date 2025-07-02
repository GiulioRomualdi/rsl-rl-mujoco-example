"""
Common utilities for musculoskeletal simulation environment.
@author: YAKE
"""

import os
import math
import numpy as np
import time
from pathlib import Path
import mujoco
import mujoco.viewer as viewer
from scipy.spatial.transform import Rotation as R
import logging
from typing import Any, List, Tuple, Union, Optional, Dict, Sequence
import quaternion

# Module-level logger configuration.
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
def validate_file(
        file_path: Union[str, Path], 
        description: str
        ) -> None:
    """
    Ensure that `file_path` exists and is a regular file.

    Parameters
    ----------
    file_path : str or os.PathLike
        Path to validate.
    description : str
        Human-readable name of the file, for error messages.

    Raises
    ------
    TypeError
        If `file_path` is not a str/PathLike or `description` is not a str.
    FileNotFoundError
        If the path does not exist or is not a regular file.
    """
    if not isinstance(description, str):
        raise TypeError(f"description must be a str, got {type(description).__name__}")
    if not isinstance(file_path, (str, os.PathLike)):
        raise TypeError(f"file_path must be str or PathLike, got {type(file_path).__name__}")

    p = Path(file_path).expanduser().resolve()
    logger.debug("Validating %s at %s", description, p)

    if not p.exists():
        raise FileNotFoundError(f"{description} does not exist at: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"{description} is not a regular file at: {p}")

def orient6_to_mat(o6: np.ndarray) -> np.ndarray:
    """
    Convert a 6D orientation vector into a 3×3 rotation matrix.

    The first three elements are the first column, the next three the second;
    the third column is computed via cross product, and all columns are
    orthonormalized.

    Parameters
    ----------
    o6 : np.ndarray, shape (6,)
        6D orientation representation.

    Returns
    -------
    np.ndarray, shape (3,3)
        Orthonormal rotation matrix.

    Raises
    ------
    ValueError
        If input is not length 6 or contains zero-length vectors.
    """
    o6 = np.asarray(o6, dtype=np.float64)
    if o6.ndim != 1 or o6.size != 6:
        raise ValueError(f"orient6_to_mat requires a 6-element vector, got shape {o6.shape}")
    
    r1 = o6[0:3].copy()
    r2 = o6[3:6].copy()

    n1 = np.linalg.norm(r1)
    r1 = r1 / (n1 if n1>1e-8 else 1.0)
    
    r3 = np.cross(r1, r2)
    n3 = np.linalg.norm(r3)
    r3 = r3 / (n3 if n3>1e-8 else 1.0)
    r2 = np.cross(r3, r1)
    return np.stack([r1, r2, r3], axis=0)  # shape (3,3)

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion (w, x, y, z) to a 3×3 rotation matrix.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Quaternion in (w, x, y, z) order.

    Returns
    -------
    np.ndarray, shape (3,3)
        Rotation matrix.

    Raises
    ------
    ValueError
        If input is not length 4 or has near-zero norm.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or q.size != 4:
        raise ValueError(f"quat_to_mat requires a 4-element vector, got shape {q.shape}")

    w, x, y, z = q
    norm = math.hypot(w, x, y, z)
    if norm < 1e-12:
        raise ValueError("Quaternion has near-zero norm")

    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    ww = w*w; xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z

    return np.array([
        [ ww + xx - yy - zz,  2*(xy - wz),      2*(xz + wy)    ],
        [ 2*(xy + wz),        ww - xx + yy - zz,2*(yz - wx)    ],
        [ 2*(xz - wy),        2*(yz + wx),      ww - xx - yy + zz]
    ], dtype=np.float64)

def rotation_geodesic(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute the geodesic angle between two rotation matrices.

    Given R = R1ᵀ R2, the rotation angle θ satisfies:
        trace(R) = 1 + 2 cos θ
        θ = arccos((trace(R)-1)/2)

    Parameters
    ----------
    R1, R2 : np.ndarray, shape (3,3)
        Rotation matrices.

    Returns
    -------
    float
        Geodesic distance in radians ∈ [0, π].

    Raises
    ------
    ValueError
        If inputs are not 3×3 matrices.
    """
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    if R1.shape != (3,3) or R2.shape != (3,3):
        raise ValueError(f"rotation_geodesic requires two 3×3 matrices, got {R1.shape}, {R2.shape}")

    R = R1.T @ R2
    tr = (np.trace(R) - 1.0) / 2.0
    tr = max(-1.0, min(1.0, tr))
    return math.acos(tr)
        
def add_noise(
        data: np.ndarray,
        noise_std: float = 2e-3,
        noise_type: str = 'gaussian',
        rng: np.random.Generator = None
        ) -> np.ndarray:
    """
    Add random noise to an array, elementwise.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    noise_std : float, optional
        Scale parameter (σ for Gaussian, half-range for uniform). Must be ≥ 0.
    noise_type : {'gaussian','uniform'}, optional
        Distribution of noise:
          - 'gaussian': N(0, σ²)
          - 'uniform': U(-σ, σ)
    rng : np.random.Generator, optional
        Random number generator. If None, a new default_rng() is used.

    Returns
    -------
    np.ndarray
        Noisy array of the same shape and dtype as `data`.

    Raises
    ------
    TypeError
        If `data` is not a numpy.ndarray or `rng` is not a Generator.
    ValueError
        If `noise_std` < 0 or `noise_type` invalid.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be np.ndarray, got {type(data).__name__}")
    if not isinstance(noise_std, (int, float)):
        raise TypeError(f"noise_std must be numeric, got {type(noise_std).__name__}")
    noise_std = float(noise_std)
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")

    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be numpy.random.Generator, got {type(rng).__name__}")

    nt = noise_type.lower()
    if nt == 'gaussian':
        noise = rng.standard_normal(size=data.shape) * noise_std
    elif nt == 'uniform':
        noise = rng.uniform(-noise_std, noise_std, size=data.shape)
    else:
        raise ValueError(f"noise_type must be 'gaussian' or 'uniform', got '{noise_type}'")

    # Preserve input dtype, avoid unnecessary copy if possible
    return (data + noise).astype(data.dtype, copy=False)

def calculate_frameskip(
        env: Any, 
        tolerance: float = 1e-8
        ) -> int:
    """
    Determine the integer number of MuJoCo steps to take per environment step.

    The frame skip is computed as:

        raw_skip = ref_traj.increment / (opt_time * ref_traj.sample_frequency)

    and must be an integer within a specified tolerance.

    Parameters
    ----------
    env : Any
        Environment providing:
          - env.opt_time : float > 0
              Simulation integrator timestep (seconds).
          - env.ref_traj.increment : int or float > 0
              Reference‐trajectory frame increment per env.step().
          - env.ref_traj.sample_frequency : int or float > 0
              Reference trajectory sampling rate (Hz).
    tolerance : float, default=1e-8
        Maximum allowed deviation of `raw_skip` from the nearest integer
        to account for floating‐point imprecision.

    Returns
    -------
    frame_skip : int
        Number of physics steps per environment step (≥1).

    Raises
    ------
    AttributeError
        If any required attribute is missing.
    ValueError
        If any parameter is non‐positive, if `raw_skip` differs from its
        nearest integer by more than `tolerance`, or if the resulting
        frame_skip < 1.
    """
    try:
        opt_time = float(env.opt_time)
        increment = float(env.ref_traj.increment)
        sample_freq = float(env.ref_traj.sample_frequency)
    except AttributeError as e:
        raise AttributeError(f"Missing attribute for frameskip calc: {e}")
    except (TypeError, ValueError):
        raise ValueError(
            "opt_time, ref_traj.increment, and sample_frequency "
            "must be numeric."
        )

    if opt_time <= 0.0:
        raise ValueError(f"opt_time must be positive, got {opt_time}")
    if increment <= 0.0:
        raise ValueError(f"ref_traj.increment must be positive, got {increment}")
    if sample_freq <= 0.0:
        raise ValueError(f"ref_traj.sample_frequency must be positive, got {sample_freq}")

    raw_skip = increment / (opt_time * sample_freq)
    nearest = int(round(raw_skip))
    
    if not math.isfinite(raw_skip):
        raise ValueError(f"Computed raw_skip is not finite: {raw_skip}")
    if abs(raw_skip - nearest) > tolerance:
        raise ValueError(
            f"frame_skip {raw_skip:.8f} differs from integer {nearest} "
            f"by more than tolerance {tolerance}"
        )
    if nearest < 1:
        raise ValueError(f"Calculated frame_skip is {nearest}, must be ≥1")

    return nearest

def playback_ref_traj(
        env: Any, 
        timestep: int = 500, 
        fps: int = 50, 
        delay: float = 3.0, 
        speed_factor: float = 0.5, 
        start_current: bool = True, 
        verbose: bool = False
        ) -> None:
    """
    Render the reference gait trajectory in a passive MuJoCo viewer.

    Parameters
    ----------
    env : Any
        Gymnasium/MuJoCo environment providing:
          - model : mujoco.MjModel
          - ref_traj : ReferenceTrajectories
          - optionally env._ref_data : mujoco.MjData
    timestep : int, default=500
        Maximum number of frames to render.
    fps : int, default=50
        Target frames per second (must be > 0).
    delay : float, default=3.0
        Seconds to pause before and after playback.
    speed_factor : float, default=0.5
        Playback speed multiplier (must be > 0).
    start_current : bool, default=True
        If True, start from the current ref_traj phase; else from phase 0.
    verbose : bool, default=False
        If True, log progress and warnings.

    Raises
    ------
    ValueError
        If fps or speed_factor ≤ 0.
    RuntimeError
        If the viewer fails to launch.
    """
    if fps <= 0 or speed_factor <= 0:
        raise ValueError("fps and speed_factor must be positive.")
    frame_time = 1.0 / (fps * speed_factor)
    
    model = env.model
    ref   = env.ref_traj
    
    if not hasattr(env, "_ref_data"):
        env._ref_data = mujoco.MjData(model)
    play_data = env._ref_data
    
    get_rt   = ref.get_reference_trajectories
    step_rt  = ref.next
    has_end = lambda: ref.has_reached_end
    
    orig_phase  = ref.phase
    orig_traj_id = ref.traj_id
    
    if not start_current:
        ref.reset(traj_id=orig_traj_id, phase=0.0)
        if verbose:
            logger.debug("Playback: starting from phase 0%% of traj %d", orig_traj_id)
    elif verbose:
        logger.debug(
            "Playback: starting from phase %.2f%% of traj %d",
            orig_phase, orig_traj_id
        )
    
    try:
        my_viewer = viewer.launch_passive(model, play_data)
    except Exception as e:
        raise RuntimeError(f"Failed to launch MuJoCo viewer: {e}")
    
    try:
        pelvis_id = model.body("pelvis").id
        my_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        my_viewer.cam.trackbodyid = pelvis_id
        my_viewer.cam.fixedcamid = -1
        my_viewer.cam.distance  = 5      
        my_viewer.cam.azimuth   = 135.0
        my_viewer.cam.elevation = -20.0  
    except Exception:
        if verbose:
            logger.warning("Viewer does not support automatic body tracking.")
    
    if delay > 0:
        time.sleep(delay)   
    try:
        for frame_idx in range(timestep):
            if has_end():
                if verbose:
                    logger.info(
                        "Playback: reached end of trajectory at frame %d", frame_idx
                    )
                break

            t0 = time.perf_counter()
            try:
                qpos_ref, qvel_ref = get_rt()
                qpos = convert_ref_traj_qpos(qpos_ref)
                qvel = convert_ref_traj_qvel(qvel_ref, qpos_ref)
                
                if qpos.shape[0] != model.nq or qvel.shape[0] != model.nv:
                    raise ValueError(
                        f"Converted qpos length {qpos.shape[0]} != model.nq {model.nq}"
                    )
                
                play_data.qpos[:] = qpos
                play_data.qvel[:] = qvel
                mujoco.mj_forward(model, play_data)
                my_viewer.sync()

            except KeyboardInterrupt:
                if verbose:
                    logger.info("Playback interrupted by user at frame %d", frame_idx)
                break
            except Exception as err:
                logger.error("Error at frame %d: %s", frame_idx, err)
                break

            elapsed = time.perf_counter() - t0
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            elif verbose:
                logger.warning(
                    "Frame %d took %.4fs (target %.4fs)",
                    frame_idx, elapsed, frame_time
                )

            step_rt()
    
    finally:
        if delay > 0:
            time.sleep(delay)
            
        try:
            my_viewer.close()
        except Exception:
            pass
        ref.reset(traj_id=orig_traj_id, phase=orig_phase)
        if verbose:
            logger.info(
                "Playback finished; restored phase %.2f%% of traj %d",
                orig_phase, orig_traj_id
            )
    
def get_penalty(
        x: np.ndarray,
        prev_x: Optional[np.ndarray],
        smoothness_weight: float = 0.005,
        clip_range: Optional[Tuple[float, float]] = None,
        enable_clip: bool = False
        ) -> float:
    """
    Compute a smoothness penalty based on the squared norm of the difference
    between the current vector and the previous vector.

    penalty = smoothness_weight * ||x - prev_x||²

    Optionally, the penalty can be clipped to a specified range.

    Parameters
    ----------
    x : array_like
        Current vector.
    prev_x : array_like or None
        Previous vector of the same shape, or None to indicate this is the first
        step (in which case penalty = 0.0).
    smoothness_weight : float, default=0.005
        Non-negative scaling factor applied to the squared norm of the difference.
    clip_range : tuple of two floats, optional
        (min, max) interval to which the penalty will be clipped if enable_clip
        is True. If None, defaults to (0.0, 0.1).
    enable_clip : bool, default=False
        If True, clip the penalty into the interval specified by clip_range.

    Returns
    -------
    float
        The computed (and possibly clipped) penalty.

    Raises
    ------
    ValueError
        If prev_x is not None and shapes of x and prev_x differ, or if
        smoothness_weight is negative, or if clip_range is invalid when
        clipping is enabled.
    """
    if prev_x is None:
        return 0.0

    x_arr = np.asarray(x, dtype=np.float64)
    prev_arr = np.asarray(prev_x, dtype=np.float64)
    
    if x_arr.shape != prev_arr.shape:
        raise ValueError(
            f"Shape mismatch: x has shape {x_arr.shape}, prev_x has {prev_arr.shape}"
        )
        
    if not isinstance(smoothness_weight, (int, float)) or smoothness_weight < 0.0:
        raise ValueError(
            f"smoothness_weight must be a non-negative number, got {smoothness_weight}"
        )
    
    w = float(smoothness_weight)

    diff = x_arr - prev_arr
    sq_norm = float((diff * diff).sum())
    penalty = w * sq_norm

    if enable_clip:
        if clip_range is None:
            lo, hi = 0.0, 0.1
        else:
            if (
                not isinstance(clip_range, tuple)
                or len(clip_range) != 2
                or not all(isinstance(v, (int, float)) for v in clip_range)
            ):
                raise ValueError(
                    f"clip_range must be a tuple of two numbers, got {clip_range}"
                )
            lo, hi = float(clip_range[0]), float(clip_range[1])
        if lo > hi:
            raise ValueError(
                f"clip_range lower bound {lo} exceeds upper bound {hi}"
            )
        penalty = penalty if penalty >= lo else lo
        penalty = penalty if penalty <= hi else hi

    return penalty

# -----------------------------------
# Euler angle <--> Quaternion Modular 
# -----------------------------------

_INV_SQRT2 = math.sqrt(2.0) / 2.0
_Q0_INV = quaternion.quaternion(_INV_SQRT2, -_INV_SQRT2, 0.0, 0.0)

def compute_global_quaternion(
    tilt: Union[int, float],
    list_angle: Union[int, float],
    rotation: Union[int, float]
    ) -> quaternion.quaternion:
    """
    Convert local pelvis Euler angles (tilt, list, rotation) into a global
    orientation quaternion, using minimal arithmetic and no intermediate arrays.

    The local rotations are applied in Z–X–Y order on top of an initial
    pelvis orientation q0 = [√2/2, √2/2, 0, 0].

    Steps (all in radians):
      1. half‐angles: ht = tilt/2, hl = list_angle/2, hr = rotation/2
      2. compute sin/cos for each half‐angle
      3. form qz = (cz, 0, 0, sz), qx = (cx, sx, 0, 0), qy = (cy, 0, sy, 0)
      4. multiply qz * qx → (w1,x1,y1,z1)
      5. multiply that result by qy → (w2,x2,y2,z2)
      6. left‐multiply by q0 = (a,a,0,0) where a = √2/2, yielding final (w,x,y,z)

    This avoids creating temporary quaternion objects for qz, qx, qy, and
    uses only Python floats and math.* calls, for maximum speed.

    Parameters
    ----------
    tilt : float
        Rotation about the local Z-axis (radians).
    list_angle : float
        Rotation about the local X-axis (radians).
    rotation : float
        Rotation about the local Y-axis (radians).

    Returns
    -------
    quaternion.quaternion
        The resulting global quaternion in (w, x, y, z) format.

    Raises
    ------
    TypeError
        If any input is not a real number.
    """
    # Validate inputs
    for name, val in (("tilt", tilt), ("list_angle", list_angle), ("rotation", rotation)):
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a real number, got {type(val).__name__}")

    # Half‐angles
    ht = tilt       * 0.5
    hl = list_angle * 0.5
    hr = rotation   * 0.5

    # Sin/cos once each
    cz, sz = math.cos(ht), math.sin(ht)
    cx, sx = math.cos(hl), math.sin(hl)
    cy, sy = math.cos(hr), math.sin(hr)

    # qz * qx
    w1 = cz*cx
    x1 = cz*sx
    y1 = sz*sx
    z1 = sz*cx

    # (qz*qx) * qy
    w2 = w1*cy - y1*sy
    x2 = x1*cy - z1*sy
    y2 = w1*sy + y1*cy
    z2 = x1*sy + z1*cy

    # q0 * (previous result), where q0 = (a,a,0,0)
    a  = _INV_SQRT2
    w  = a*(w2 - x2)
    x  = a*(x2 + w2)
    y  = a*(y2 - z2)
    z  = a*(z2 + y2)

    return quaternion.quaternion(w, x, y, z)

def compute_local_angles(
    q_global: quaternion.quaternion
    ) -> Tuple[float, float, float]:
    """
    Extract pelvis-local Z–X–Y Euler angles (tilt, list, rotation) from a global quaternion.

    This assumes the global orientation was built as:
        q_global = q0 * qz(tilt) * qx(list_angle) * qy(rotation),
    where q0 = [√2/2, √2/2, 0, 0]. We recover the increments q_inc = q0⁻¹ * q_global,
    then extract Z–X–Y Euler angles directly from q_inc.

    Parameters
    ----------
    q_global : quaternion.quaternion
        Global orientation quaternion (w, x, y, z).

    Returns
    -------
    tilt       : float
        Rotation about local Z-axis (radians).
    list_angle : float
        Rotation about local X-axis (radians).
    rotation   : float
        Rotation about local Y-axis (radians).

    Raises
    ------
    TypeError
        If `q_global` is not a quaternion.quaternion.
    """
    if not isinstance(q_global, quaternion.quaternion):
        raise TypeError(f"q_global must be quaternion.quaternion, got {type(q_global).__name__}")

    # Multiply q_inc = q0⁻¹ * q_global
    w0, x0, y0, z0 = _Q0_INV.w, _Q0_INV.x, _Q0_INV.y, _Q0_INV.z
    wg, xg, yg, zg = q_global.w, q_global.x, q_global.y, q_global.z

    # Quaternion multiplication (no temporaries)
    w = w0*wg - x0*xg - y0*yg - z0*zg
    x = w0*xg + x0*wg + y0*zg - z0*yg
    y = w0*yg - x0*zg + y0*wg + z0*xg
    z = w0*zg + x0*yg - y0*xg + z0*wg

    # Precompute squares
    xx = x*x
    yy = y*y
    zz = z*z

    # Extract rotation-matrix elements for Z–X–Y sequence:
    # R01 =  2*(x*y - w*z)
    # R11 =  1 - 2*(x² + z²)
    # R20 =  2*(x*z - w*y)
    # R22 =  1 - 2*(x² + y²)
    # R21 =  2*(y*z + w*x)
    r01 = 2.0 * (x*y - w*z)
    r11 = 1.0 - 2.0 * (xx + zz)
    r20 = 2.0 * (x*z - w*y)
    r22 = 1.0 - 2.0 * (xx + yy)
    r21 = 2.0 * (y*z + w*x)

    # Clamp for numerical safety
    if r21 > 1.0:
        r21 = 1.0
    elif r21 < -1.0:
        r21 = -1.0

    # Recover angles
    list_angle = math.asin(r21)                  # X-axis rotation
    rotation   = math.atan2(-r20, r22)           # Y-axis rotation
    tilt       = math.atan2(-r01, r11)           # Z-axis rotation

    return tilt, list_angle, rotation

# --------------------------------------
# Ref-traj format <--> Free-joint format 
# --------------------------------------
def convert_ref_traj_qpos(
    ref_qpos_raw: Union[Sequence[float], np.ndarray]
    ) -> np.ndarray:
    """
    Convert a 6-dim pelvis reference state into a 7-dim translation+quaternion,
    then append the remaining joint angles unchanged.

    Input format (length ≥ 6):
        [tz, ty, tx, tilt, list_angle, rotation, ...other joints...]

    Output format (length = len(ref_qpos_raw) + 1):
        [tx, -tz, ty+0.95, qw, qx, qy, qz, ...other joints...]

    Parameters
    ----------
    ref_qpos_raw : sequence of float or np.ndarray
        1D array-like with at least 6 elements:
        - indices 0–2: pelvis [tz, ty, tx]
        - indices 3–5: Euler angles [tilt, list, rotation]
        - indices ≥6: other joint coordinates

    Returns
    -------
    np.ndarray
        1D array of length `len(ref_qpos_raw) + 1`, where the first 7 entries
        are the pelvis (translation + quaternion) and the rest are copied
        from ref_qpos_raw[6:].

    Raises
    ------
    TypeError
        If ref_qpos_raw cannot be converted to a 1D numeric array.
    ValueError
        If ref_qpos_raw has fewer than 6 elements.
    """
    # Convert input to 1D float64 array
    try:
        tmp = np.asarray(ref_qpos_raw, dtype=np.float64).ravel()
    except Exception as e:
        raise TypeError(f"ref_qpos_raw must be array-like of numbers: {e}")
    if tmp.size < 6:
        raise ValueError(f"ref_qpos_raw requires at least 6 elements, got {tmp.size}")

    # Determine how many "other joints" follow
    n_remain = tmp.size - 6

    # Allocate output: 7 pelvis dims + remaining joints
    out = np.empty(7 + n_remain, dtype=np.float64)

    # 1) Pelvis translation: [x, y, z] = [tx, -tz, ty + 0.95]
    out[0] = tmp[2]          # tx
    out[1] = -tmp[0]         # -tz
    out[2] = tmp[1] + 0.95   # ty + 0.95

    # 2) Pelvis orientation quaternion
    q = compute_global_quaternion(tmp[3], tmp[4], tmp[5])
    out[3] = q.w
    out[4] = q.x
    out[5] = q.y
    out[6] = q.z

    # 3) Copy remaining joint angles (if any)
    if n_remain > 0:
        out[7:] = tmp[6:]

    logger.debug("convert_ref_traj_qpos → output length %d", out.size)
    return out

def inverse_convert_ref_traj_qpos(
    d_qpos: Union[Sequence[float], np.ndarray]
    ) -> np.ndarray:
    """
    Invert a freejoint-format reference pose back to the original 6-DOF pelvis
    + rest-of-joints format.

    This reverses `convert_ref_traj_qpos`, mapping:
      - Translations: [tx, -tz, ty+0.95] → [tz, ty, tx]
      - Orientation: 4-component quaternion → Z–X–Y Euler angles
      - Other DOFs passed through unchanged.

    Parameters
    ----------
    d_qpos : array_like, shape (>=7,)
        Freejoint-format state, where:
          - d_qpos[0:3] are [tx, -tz, ty+0.95],
          - d_qpos[3:7] are quaternion (w, x, y, z),
          - d_qpos[7:] are the remaining joint DOFs.

    Returns
    -------
    np.ndarray, shape (len(d_qpos)-1,)
        Recovered reference trajectory state:
        [tz, ty, tx, tilt, list_angle, rotation, ...other DOFs...].

    Raises
    ------
    TypeError
        If `d_qpos` cannot be converted to a 1D numeric array.
    ValueError
        If `d_qpos` has fewer than 7 elements.
    RuntimeError
        If quaternion-to-Euler conversion fails.
    """
    # Convert to 1D float array
    try:
        arr = np.asarray(d_qpos, dtype=np.float64).ravel()
    except Exception as e:
        raise TypeError(f"d_qpos must be array-like of numbers: {e}")
    n = arr.size
    if n < 7:
        raise ValueError(f"d_qpos must have at least 7 elements, got {n}")

    # Preallocate output: one fewer element
    out = np.empty(n - 1, dtype=arr.dtype)

    # 1) Recover pelvis translation:
    #    arr[0]=tx, arr[1]=-tz, arr[2]=ty+0.95
    out[0] = -arr[1]        # tz
    out[1] = arr[2] - 0.95  # ty
    out[2] = arr[0]         # tx

    # 2) Recover local Euler angles from quaternion
    w, x, y, z = arr[3], arr[4], arr[5], arr[6]
    q = quaternion.quaternion(w, x, y, z)
    try:
        tilt, list_angle, rotation = compute_local_angles(q)
    except Exception as e:
        raise RuntimeError(f"Euler recovery failed: {e}")
    out[3] = tilt
    out[4] = list_angle
    out[5] = rotation

    # 3) Copy remaining DOFs (if any)
    if n > 7:
        out[6:] = arr[7:]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "inverse_convert_ref_traj_qpos → recovered length %d", out.size
        )
    return out


def euler_rates_to_axis_angle(
    tilt: Union[int, float],
    list_angle: Union[int, float],
    rotation: Union[int, float],
    tilt_rate: Union[int, float],
    list_rate: Union[int, float],
    rotation_rate: Union[int, float],
    dt: float = 0.001
    ) -> np.ndarray:
    """
    Convert Z–X–Y Euler angle rates into the global angular velocity vector ω,
    by small‐step quaternion integration.

    Steps:
      1. Compute current global quaternion q0 = Q(tilt, list_angle, rotation).
      2. Advance each Euler angle by rate*dt, get q1.
      3. Compute delta quaternion qDelta = q0.inverse() * q1.
      4. For small dt, rotation vector ≈ 2 * (qDelta.x, qDelta.y, qDelta.z).
      5. ω = rotation_vector / dt.

    Parameters
    ----------
    tilt : float
        Current pelvis tilt (radians) about local Z.
    list_angle : float
        Current pelvis list (radians) about local X.
    rotation : float
        Current pelvis rotation (radians) about local Y.
    tilt_rate : float
        Time derivative of tilt (rad/s).
    list_rate : float
        Time derivative of list (rad/s).
    rotation_rate : float
        Time derivative of rotation (rad/s).
    dt : float, default=0.001
        Time step over which rates are applied (seconds). Must be > 0.

    Returns
    -------
    np.ndarray, shape (3,)
        Global angular velocity [ω_x, ω_y, ω_z] in rad/s.

    Raises
    ------
    ValueError
        If dt ≤ 0.
    TypeError
        If any input is not a numeric type.
    """
    # -- Validate --
    if not isinstance(dt, (int, float)) or dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    for name, val in (
        ("tilt", tilt), ("list_angle", list_angle), ("rotation", rotation),
        ("tilt_rate", tilt_rate), ("list_rate", list_rate), ("rotation_rate", rotation_rate)
    ):
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a real number, got {type(val).__name__}")

    # -- Current quaternion --
    q0 = compute_global_quaternion(tilt, list_angle, rotation)

    # -- Next quaternion after Euler‐step dt --
    t1 = tilt      + tilt_rate      * dt
    l1 = list_angle+ list_rate      * dt
    r1 = rotation  + rotation_rate  * dt
    q1 = compute_global_quaternion(t1, l1, r1)

    # -- Delta quaternion in body/world frame --
    qd = q0.inverse() * q1

    # -- For small dt, rotation_vector ≈ 2 * qd.vec
    vx, vy, vz = qd.x, qd.y, qd.z
    omega = np.array([vx, vy, vz], dtype=float) * (2.0 / dt)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "euler_rates_to_axis_angle: ω=[%.4f, %.4f, %.4f]",
            omega[0], omega[1], omega[2]
        )
    return omega

def euler_rates_to_axis_angle1(tilt: float, 
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

def convert_ref_traj_qvel(
    ref_qvel_raw: Union[Sequence[float], np.ndarray],
    ref_qpos_raw: Union[Sequence[float], np.ndarray],
    dt: float = 0.001
    ) -> np.ndarray:
    """
    Convert a reference trajectory’s pelvis velocity from local Euler‐rate form
    into global linear + angular velocity, then append the remaining DOFs unchanged.

    Input:
      ref_qvel_raw (length ≥ 6):
        [v_z, v_y, v_x, tilt_dot, list_dot, rotation_dot, ...]
      ref_qpos_raw (length ≥ 6):
        [tz, ty, tx, tilt, list, rotation, ...]

    Output (same length as ref_qvel_raw):
      [v_x, -v_z, v_y, ω_x, ω_y, ω_z, ...other DOFs...]

    The angular part ω = [ω_x, ω_y, ω_z] is computed by
    `euler_rates_to_axis_angle(tilt, list, rotation,
                               tilt_dot, list_dot, rotation_dot, dt)`.

    Parameters
    ----------
    ref_qvel_raw : array_like
        1D array of length ≥ 6: pelvis velocities + Euler rates + other DOFs.
    ref_qpos_raw : array_like
        1D array of length ≥ 6: pelvis positions + Euler angles + other DOFs.
    dt : float, default=0.001
        Time step (s) used in the Euler→axis‐angle conversion. Must be > 0.

    Returns
    -------
    np.ndarray
        1D array of same length as `ref_qvel_raw`, with the first 6 entries
        transformed to [global linear vel, global angular vel] and the rest copied.

    Raises
    ------
    TypeError
        If inputs cannot be converted to 1D numeric arrays.
    ValueError
        If either array has fewer than 6 elements or if `dt` ≤ 0.
    RuntimeError
        If the Euler‐rate→angular‐velocity conversion fails.
    """
    # Convert and validate inputs
    try:
        vel = np.asarray(ref_qvel_raw, dtype=np.float64).ravel()
        pos = np.asarray(ref_qpos_raw, dtype=np.float64).ravel()
    except Exception as e:
        raise TypeError(f"Inputs must be array-like of numbers: {e}")

    if vel.size < 6 or pos.size < 6:
        raise ValueError(f"Both arrays must have at least 6 elements; got {vel.size} and {pos.size}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    n = vel.size
    out = np.empty(n, dtype=np.float64)

    # 1) Global linear velocity: [v_x, -v_z, v_y]
    #    raw ordering: [v_z, v_y, v_x]
    out[0] = vel[2]
    out[1] = -vel[0]
    out[2] = vel[1]

    # 2) Global angular velocity via quaternion small‐step
    try:
        ang_vel = euler_rates_to_axis_angle(
            pos[3], pos[4], pos[5],
            vel[3], vel[4], vel[5],
            dt
        )
    except Exception as e:
        raise RuntimeError(f"Euler‐rate to angular velocity conversion failed: {e}")

    out[3:6] = ang_vel

    # 3) Copy remaining DOFs (if any)
    if n > 6:
        out[6:] = vel[6:]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "convert_ref_traj_qvel → linear=%s, angular=%s, total_length=%d",
            out[:3], out[3:6], n
        )
    return out

def inverse_convert_ref_traj_qvel(
    d_qvel: Union[Sequence[float], np.ndarray],
    ref_qpos_raw: Union[Sequence[float], np.ndarray],
    dt: float = 0.001
    ) -> np.ndarray:
    """
    Inversely convert a global pelvis+angular‐velocity vector back into
    the reference‐trajectory format of local linear velocities + Euler‐rate angular velocities.

    The forward mapping was:
      ref_qvel_raw = [v_z, v_y, v_x, tilt_dot, list_dot, rotation_dot, ...]
      d_qvel = [v_x, -v_z, v_y, ω_x, ω_y, ω_z, ...]
      where ω = euler_rates_to_axis_angle(...).

    This function inverts that:
      - Recover [v_z, v_y, v_x] from [v_x, -v_z, v_y].
      - Recover Euler‐rates by small‐step quaternion integration:
          q0 = Q(tilt, list_angle, rotation)
          Δq = from_rotation_vector(ω * dt)
          q1 = q0 * Δq
          (tilt1, list1, rot1) = euler angles of q1
          rates = (tilt1 − tilt)/dt, etc.

    Parameters
    ----------
    d_qvel : array_like, shape (>=6,)
        Global velocity from `convert_ref_traj_qvel`:
        [v_x, v_y, v_z, ω_x, ω_y, ω_z, ...].
    ref_qpos_raw : array_like, shape (>=6,)
        The original reference positions:
        [tz, ty, tx, tilt, list_angle, rotation, ...].
    dt : float, default=0.001
        Time step (s) matching the forward conversion.

    Returns
    -------
    np.ndarray, shape same as d_qvel
        Recovered reference velocities:
        [v_z, v_y, v_x, tilt_dot, list_dot, rotation_dot, ...].

    Raises
    ------
    TypeError
        If inputs cannot be converted to numeric arrays.
    ValueError
        If either array has fewer than 6 elements or `dt` <= 0.
    RuntimeError
        If quaternion‐to‐Euler conversion fails.
    """
    # --- Convert and validate inputs ---
    try:
        arr_vel = np.asarray(d_qvel, dtype=float).ravel()
        arr_pos = np.asarray(ref_qpos_raw, dtype=float).ravel()
    except Exception as e:
        raise TypeError(f"Inputs must be array-like of numbers: {e}")
    if arr_vel.size < 6 or arr_pos.size < 6:
        raise ValueError(f"Both vectors must have ≥6 elements; got {arr_vel.size}, {arr_pos.size}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Prepare output array
    out = np.empty_like(arr_vel)

    # --- 1) Invert linear velocity mapping ---
    # forward: [v_z, v_y, v_x] → [v_x, -v_z, v_y]
    # so inverse:
    #   orig_vz = -d_qvel[1]
    #   orig_vy =  d_qvel[2]
    #   orig_vx =  d_qvel[0]
    out[0] = -arr_vel[1]   # v_z
    out[1] =  arr_vel[2]   # v_y
    out[2] =  arr_vel[0]   # v_x

    # --- 2) Invert angular rates via quaternion small‐step ---
    tilt, list_ang, rotation = arr_pos[3], arr_pos[4], arr_pos[5]
    omega = arr_vel[3:6]  # global angular velocity [ω_x, ω_y, ω_z]

    # Current orientation
    q0 = compute_global_quaternion(tilt, list_ang, rotation)
    # Small‐step delta quaternion from ω * dt
    try:
        delta_q = quaternion.from_rotation_vector(omega * dt)
    except Exception as e:
        raise RuntimeError(f"Failed to form delta quaternion: {e}")
    # Advance orientation
    q1 = q0 * delta_q

    # Extract new local Euler angles
    try:
        tilt1, list1, rot1 = compute_local_angles(q1)
    except Exception as e:
        raise RuntimeError(f"Failed to recover Euler angles: {e}")

    # Recover rates by finite difference
    out[3] = (tilt1 - tilt) / dt
    out[4] = (list1 - list_ang) / dt
    out[5] = (rot1 - rotation) / dt

    # --- 3) Copy remaining DOFs ---
    if arr_vel.size > 6:
        out[6:] = arr_vel[6:]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "inverse_convert_ref_traj_qvel → linear=[%.4f,%.4f,%.4f], rates=[%.4f,%.4f,%.4f]",
            out[0], out[1], out[2], out[3], out[4], out[5]
        )
    return out

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

def export_reference_observations(env: Any, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    manager = env.traj_manager
    total_traj = len(manager)
    
    import pickle
    from tqdm import tqdm
    
    for traj_id in tqdm(range(total_traj), desc="Processing trajectories"):
        env.ref_traj.reset(traj_id=traj_id, phase=0)
        traj_len = env.ref_traj.traj_frames

        even_obs = []
        odd_obs = []

        for frame_idx in range(traj_len):
            env.ref_traj._pos = frame_idx
            env.update_traj_init_info()
            env.reset_to_initial_state()

            obs = env.get_obs()
            if frame_idx % 2 == 0:
                even_obs.append(obs.copy())
            else:
                odd_obs.append(obs.copy())

        even_array = np.stack(even_obs)
        odd_array = np.stack(odd_obs)

        with open(os.path.join(save_dir, f"traj_{traj_id}_even.pkl"), "wb") as f:
            pickle.dump(even_array, f)
        with open(os.path.join(save_dir, f"traj_{traj_id}_odd.pkl"), "wb") as f:
            pickle.dump(odd_array, f)

        print(f"Saved traj {traj_id}: even {even_array.shape}, odd {odd_array.shape}")