"""
State-related utilities for musculoskeletal simulation environment.
Handles state extraction, normalization, and statistics collection.
@author: YAKE
"""

import numpy as np
import mujoco
from .common_utils import (
    convert_ref_traj_qpos,
    convert_ref_traj_qvel,
    inverse_convert_ref_traj_qpos, 
    inverse_convert_ref_traj_qvel
    )
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def compute_grf(body_id: int,
                geom_ids: List[int],
                env: Any
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ground reaction force (GRF) for a specified foot, with optional pelvis‐aligned frame.
    
    The pelvis-aligned frame is defined as follows:
      - X axis: env.pelvis_heading (pelvis local X projected onto XY plane)
      - Y axis: world Z axis (vertical up)
      - Z axis: cross(X, Y)

    Parameters
    ----------
    body_id : int
        MuJoCo body ID of the foot (e.g., calcaneus).
    geom_ids : np.ndarray, shape=(G,)
        Array of geom IDs belonging to that foot.
    env : Any
        Must have:
            - data: with attributes
                * xpos (Nbody×3), contact.geom1/geom2 (ncon,), contact.pos (ncon×3), contact.frame (ncon×9), ncon (int)
            - model: with body() and optionally pelvis body
            - relative_pelvis: bool
            - pelvis_heading: (3,) unit vector in world XY-plane

    Returns
    -------
    pos_local : (3,) weighted-avg contact point in chosen frame
    force_local : (3,) net contact force in chosen frame
    torque_local : (3,) net contact torque about body origin
    """
    data = env.data
    model = env.model
    
    # Reference point and origin offset
    ref_pos = data.xpos[body_id]
    if ref_pos.shape != (3,):
        raise ValueError(f"Expected data.xpos[{body_id}].shape==(3,), got {ref_pos.shape}")
    if env.relative_pelvis:
        pid = model.body("pelvis").id
        origin_offset = data.xpos[pid]
    else:
        origin_offset = np.zeros(3, dtype=float)
    
    ncon    = data.ncon
    geom1   = data.contact.geom1[:ncon]
    geom2   = data.contact.geom2[:ncon]
    pos_all = data.contact.pos[:ncon]
    frame9  = data.contact.frame[:ncon]
    try:
        frame_mats = frame9.reshape(-1,3,3)
    except ValueError:
        raise ValueError("Malformed data.contact.frame; expected ncon×9 array")
    
    mask = np.isin(geom1, geom_ids) | np.isin(geom2, geom_ids)
    if not np.any(mask):
        return np.zeros(3), np.zeros(3), np.zeros(3)
    
    total_force  = np.zeros(3, dtype=float)
    total_torque = np.zeros(3, dtype=float)
    weighted_pos = np.zeros(3, dtype=float)
    tmp = np.zeros(6, dtype=float)
    
    for i in np.nonzero(mask)[0]:
        try:
            mujoco.mj_contactForce(model, data, i, tmp)
        except RuntimeError:
            continue
        f_world = frame_mats[i].T.dot(tmp[:3])
        p_world = pos_all[i]
        total_force  += f_world
        total_torque += np.cross(p_world - ref_pos, f_world)
        weighted_pos += p_world * f_world

    if np.allclose(total_force, 0.0):
        return np.zeros(3), np.zeros(3), np.zeros(3)

    avg_pos_world = weighted_pos / np.where(total_force!=0, total_force, 1.0)
    pos_rel_world = avg_pos_world - origin_offset

    if env.relative_pelvis:
        xh = env.pelvis_heading
        yh = np.array([0.0,0.0,1.0], dtype=float)
        zh = np.cross(xh, yh)
        zn = np.linalg.norm(zh)
        if zn < 1e-6:
            zh = np.array([0.0,1.0,0.0], dtype=float)
        else:
            zh /= zn
        xh = np.cross(yh, zh)
        xh /= np.linalg.norm(xh)
        R = np.stack((xh, yh, zh), axis=1)
        Rw2l = R.T
        return (Rw2l.dot(pos_rel_world),
                Rw2l.dot(total_force),
                Rw2l.dot(total_torque))
    else:
        return pos_rel_world, total_force, total_torque

def get_GRF_info(env: Any) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """
    Compute GRF information for both feet and concatenate into an 18-dimensional vector.
    
    Caches foot body IDs and geom sets on first call as:
      env._grf_foot_ids  = {'right': int, 'left': int}
      env._grf_geom_sets = {'right': np.ndarray, 'left': np.ndarray}

    Returns
    -------
    grf_concat : np.ndarray, shape=(18,)
        [r_pos, r_force, r_torque, l_pos, l_force, l_torque], each 3D.
    info : dict
        {
          'right': {'GRF_pos': ..., 'GRF_force': ..., 'GRF_torque': ...},
          'left':  {'GRF_pos': ..., 'GRF_force': ..., 'GRF_torque': ...}
        }
    """
    # early-out if no contacts
    if env.data.ncon == 0:
        zeros = np.zeros(3, dtype=float)
        return np.zeros(18, dtype=float), {
            'right': {'GRF_pos': zeros, 'GRF_force': zeros, 'GRF_torque': zeros},
            'left':  {'GRF_pos': zeros, 'GRF_force': zeros, 'GRF_torque': zeros}
        }
    
    # cache foot IDs & geom sets
    if not hasattr(env, "_grf_foot_ids"):
        try:
            r_id = env.model.body('calcn_r').id
            l_id = env.model.body('calcn_l').id
            r_geoms = np.array([env.model.geom(n).id for n in 
                                ['C_r_foot1','C_r_foot3','C_r_foot4','C_r_bofoot1','C_r_bofoot2']], int)
            l_geoms = np.array([env.model.geom(n).id for n in 
                                ['C_l_foot1','C_l_foot3','C_l_foot4','C_l_bofoot1','C_l_bofoot2']], int)
        except Exception as e:
            raise ValueError(f"Failed to retrieve foot IDs or geoms: {e}")
        env._grf_foot_ids  = {'right': r_id, 'left': l_id}
        env._grf_geom_sets = {'right': r_geoms, 'left': l_geoms}
    
    # compute for each side
    rv = compute_grf(env._grf_foot_ids['right'],
                     env._grf_geom_sets['right'], env)
    lv = compute_grf(env._grf_foot_ids['left'],
                     env._grf_geom_sets['left'], env)
    
    # concatenate and build info dict
    grf_vec = np.hstack((rv[0], rv[1], rv[2], lv[0], lv[1], lv[2]))
    info = {
        'right': {'GRF_pos': rv[0], 'GRF_force': rv[1], 'GRF_torque': rv[2]},
        'left':  {'GRF_pos': lv[0], 'GRF_force': lv[1], 'GRF_torque': lv[2]}
    }
    return grf_vec, info

def normalize_grf(
        grf_info: np.ndarray,
        total_mass: float = 75.337,
        gravity: float = 9.81,
        torque_scale: float = 100.0,
        clip_force: bool = False,
        clip_torque: bool = True
    ) -> np.ndarray:
    """
    Normalize ground reaction forces and torques while leaving contact positions unchanged.

    This function splits the 18-element GRF vector into right and left components,
    normalizes force by body weight (mass * gravity), and normalizes torque by a
    configurable scale. Positions are returned unchanged.

    Parameters
    ----------
    grf_info : np.ndarray, shape (18,)
        GRF vector in the format:
        [r_pos(3), r_force(3), r_torque(3), l_pos(3), l_force(3), l_torque(3)].
    total_mass : float, default=75.337
        Total mass (kg) used to normalize force components.
    gravity : float, default=9.81
        Gravitational acceleration (m/s^2) used in force normalization.
    torque_scale : float, default=100.0
        Maximum expected torque magnitude; used to scale torque components.
    clip_force : bool, default=False
        If True, clip normalized force components to [-1, 1] after scaling.
    clip_torque : bool, default=True
        If True, clip normalized torque components to [-1, 1] after scaling.

    Returns
    -------
    np.ndarray, shape (18,)
        Normalized GRF vector with same ordering as input:
        - Position channels (0:3, 9:12) unchanged.
        - Force channels (3:6, 12:15) divided by (total_mass * gravity).
        - Torque channels (6:9, 15:18) divided by torque_scale.

    Raises
    ------
    ValueError
        If 'grf_info' is not a 1D array of length 18, or if 'total_mass' <= 0,
        or if 'torque_scale' <= 0.
    """
    grf = np.asarray(grf_info, dtype=np.float64)
    if grf.ndim != 1 or grf.shape[0] != 18:
        raise ValueError(f"Expected 1D GRF array of length 18, got shape {grf.shape}")
    if total_mass <= 0:
        raise ValueError(f"total_mass must be positive, got {total_mass}")
    if torque_scale <= 0:
        raise ValueError(f"torque_scale must be positive, got {torque_scale}")

    # Copy to avoid in-place modification
    norm = grf.copy()

    # Compute normalization factors
    weight = total_mass * gravity
    # Indices
    r_force_idx = slice(3, 6)
    r_torque_idx = slice(6, 9)
    l_force_idx = slice(12, 15)
    l_torque_idx = slice(15, 18)

    # Normalize forces
    with np.errstate(divide='ignore', invalid='ignore'):
        norm[r_force_idx] /= weight
        norm[l_force_idx] /= weight
    if clip_force:
        norm[r_force_idx] = np.clip(norm[r_force_idx], -3.0, 3.0)
        norm[l_force_idx] = np.clip(norm[l_force_idx], -3.0, 3.0)

    # Normalize torques
    norm[r_torque_idx] /= torque_scale
    norm[l_torque_idx] /= torque_scale
    if clip_torque:
        norm[r_torque_idx] = np.clip(norm[r_torque_idx], -2.0, 2.0)
        norm[l_torque_idx] = np.clip(norm[l_torque_idx], -2.0, 2.0)

    # Replace any NaN or inf with zero
    return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

def compute_foot_contacts(
    grf_comp: Dict[str, Dict[str, np.ndarray]],
    threshold: float
    ) -> np.ndarray:
    """
    Compute binary contact flags for left/right foot from GRF.

    Parameters
    ----------
    grf_comp : dict
        Output of get_GRF_info()['right']['GRF_force'] and ['left']['GRF_force']
    threshold : float
        Force magnitude above which the foot is considered in contact.

    Returns
    -------
    contacts : np.ndarray, shape=(2,)
        [right_contact, left_contact], each 0.0 or 1.0.
    """
    rf = grf_comp['right']['GRF_force']
    lf = grf_comp['left']['GRF_force']
    right_flag = 1.0 if np.linalg.norm(rf) > threshold else 0.0
    left_flag  = 1.0 if np.linalg.norm(lf) > threshold else 0.0
    return np.array([right_flag, left_flag], dtype=float)

def compute_ref_pelvis_kinematics(
    env: Any,
    use_free_joint: bool = True
    ) -> Dict[str, np.ndarray]:
    """
    Extract pelvis pose & velocity directly from the reference trajectory.

    Parameters
    ----------
    env : Any
        Must provide:
          - env.ref_traj.get_extend_refer() -> (ref_qpos, ref_qvel)
    use_free_joint : bool, default=True
        If False, assumes ref_qpos/ref_qvel are already in "ref-traj" format:
            pos    = ref_qpos[0:3]
            orient = ref_qpos[3:6]
            linvel = ref_qvel[0:3]
            angvel = ref_qvel[3:6]
        If True, first converts to MuJoCo free-joint format via
        `convert_ref_traj_qpos` / `convert_ref_traj_qvel`, then slices:
            pos    = d_qpos[0:3]
            orient = d_qpos[3:7]      # quaternion
            linvel = d_qvel[0:3]
            angvel = d_qvel[3:6]

    Returns
    -------
    kin : Dict[str, np.ndarray]
        {
          "pos"    : (3,)   translation [m],
          "orient" : (3,) or (4,)  minimal rotation or quaternion,
          "linvel" : (3,)   linear velocity [m/s],
          "angvel" : (3,)   angular velocity [rad/s]
        }

    Raises
    ------
    ValueError
        If reference arrays are too short or conversion fails.
    """
    try:
        ref_qpos, ref_qvel = env.ref_traj.get_extend_refer()
    except Exception as e:
        raise ValueError(f"Failed to get reference qpos/qvel: {e}")

    for name, arr in (("ref_qpos", ref_qpos), ("ref_qvel", ref_qvel)):
        if not isinstance(arr, np.ndarray) or arr.ndim != 1 or arr.size < 6:
            raise ValueError(f"{name} must be 1D length>=6, got shape {getattr(arr, 'shape', None)}")
        
    if not use_free_joint:
        return {
            "pos":    ref_qpos[0:3].astype(np.float32).copy(),
            "orient": ref_qpos[3:6].astype(np.float32).copy(),
            "linvel": ref_qvel[0:3].astype(np.float32).copy(),
            "angvel": ref_qvel[3:6].astype(np.float32).copy(),
        }
    
    try:
        d_qpos = convert_ref_traj_qpos(ref_qpos)
        d_qvel = convert_ref_traj_qvel(ref_qvel, ref_qpos)
    except Exception as e:
        raise ValueError(f"Reference conversion error: {e}")
        
    if d_qpos.ndim != 1 or d_qpos.size < 7:
        raise ValueError(f"Converted qpos must length>=7, got {d_qpos.size}")
    if d_qvel.ndim != 1 or d_qvel.size < 6:
        raise ValueError(f"Converted qvel must length>=6, got {d_qvel.size}")
    
    return {
        "pos":    d_qpos[0:3].astype(np.float32).copy(),
        "orient": d_qpos[3:7].astype(np.float32).copy(),
        "linvel": d_qvel[0:3].astype(np.float32).copy(),
        "angvel": d_qvel[3:6].astype(np.float32).copy(),
    }

def get_pelvis_kinematics(
        env: Any,
        use_free_joint: bool = True
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Retrieve pelvis pose & velocity from the live simulation state.

    Parameters
    ----------
    env : Any
        Must provide:
          - env.data.qpos (length>=7), env.data.qvel (length>=6)
    use_free_joint : bool, default=True
        If True, return raw MuJoCo free-joint (13D) format:
            state = [tx,ty,tz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
            comp = {"pos":state[0:3], "orient":state[3:7], ...}
        If False, return reference-trajectory (12D) format:
            state = [tz,ty,tx, rot_x,rot_y,rot_z, vz,vy,vx, ang_x,ang_y,ang_z]
            comp = {"pos":state[0:3], "orient":state[3:6], ...}

    Returns
    -------
    state : np.ndarray
        13D (free-joint) or 12D (ref-traj) pelvis state.
    comp : Dict[str, np.ndarray]
        {
          "pos"    : (3,),
          "orient" : (4,) or (3,),
          "linvel" : (3,),
          "angvel" : (3,)
        }

    Raises
    ------
    ValueError
        If env.data.qpos/qvel are too short or conversions fail.
    """
    data = env.data
    
    qpos = getattr(data, "qpos", None)
    qvel = getattr(data, "qvel", None)
    if qpos is None or qvel is None:
        raise ValueError("env.data.qpos and qvel must be present")
    if qpos.shape[0] < 7 or qvel.shape[0] < 6:
        raise ValueError(f"Expected len(qpos)>=7 and len(qvel)>=6, got "
                         f"{qpos.shape[0]}, {qvel.shape[0]}")
        
    if use_free_joint:
        state = np.empty(13, dtype=np.float64)
        # --- raw free-joint MuJoCo output ---
        state[0:3]   = qpos[0:3]
        state[3:7]   = qpos[3:7]
        state[7:10]  = qvel[0:3]
        state[10:13] = qvel[3:6] 

        comp = {
            'pos':     state[0:3],
            'orient':  state[3:7],
            'linvel':  state[7:10],
            'angvel':  state[10:13],
        }
        return state, comp
    
    try:
        ref_qpos = inverse_convert_ref_traj_qpos(qpos)
        ref_qvel = inverse_convert_ref_traj_qvel(qvel, ref_qpos)
    except Exception as e:
        raise ValueError(f"Simulation→ref-traj conversion failed: {e}")

    state = np.empty(12, dtype=np.float64)
    state[0:3]  = ref_qpos[0:3]
    state[3:6]  = ref_qpos[3:6]
    state[6:9]  = ref_qvel[0:3]
    state[9:12] = ref_qvel[3:6]

    comp = {
        'pos':     state[0:3],
        'orient':  state[3:6],
        'linvel':  state[6:9],
        'angvel':  state[9:12],
    }
    return state, comp

def compute_ref_site_kinematics(env: Any) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute reference‐trajectory site kinematics by placing the model at the current
    ref qpos/qvel, running a forward pass, and extracting each joint‐site’s global
    or pelvis‐relative position, linear velocity, orientation (6D), and angular
    velocity.
    
    Parameters
    ----------
    env : Any
        Environment instance. Must provide:
          - env.model : mujoco.MjModel
          - env.relative_pelvis : bool

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        {
          "pos":    {site_name: np.ndarray(3,)},
          "linvel":    {site_name: np.ndarray(3,)},
          "orient": {site_name: np.ndarray(6,)},
          "angvel": {site_name: np.ndarray(3,)}
        }
    """
    model = env.model
    joint_sites = env._joint_sites
    gadr_pel, gdim_pel = env._pelvis_gyro
    
    data = mujoco.MjData(model)
    
    ref_qpos, ref_qvel = env.ref_traj.get_extend_refer()
    try:
        d_qpos = convert_ref_traj_qpos(ref_qpos) 
        d_qvel = convert_ref_traj_qvel(ref_qvel, ref_qpos)
    except Exception as e:
        raise ValueError(f"Reference conversion error: {e}")
    
    if d_qpos.shape[0] != model.nq or d_qvel.shape[0] != model.nv:
        raise ValueError(
            f"Expected qpos size {model.nq}, qvel size {model.nv}; "
            f"got {d_qpos.shape[0]}, {d_qvel.shape[0]}"
        )
    data.qpos[:] = d_qpos
    data.qvel[:] = d_qvel
    mujoco.mj_forward(model, data)
    
    if env.relative_pelvis:
        pel_bid = model.body("pelvis").id
        pel_sid = model.site("pelvis_sensor").id
        origin = data.xpos[pel_bid].copy()
        vpl_w  = data.qvel[0:3].copy()
        if gadr_pel is not None and gdim_pel >= 3:
            mat_p   = data.site_xmat[pel_sid].reshape(3,3)
            raw_pg  = data.sensordata[gadr_pel:gadr_pel+gdim_pel]
            vpa_w   = mat_p.dot(raw_pg)
        else:
            vpa_w   = data.qvel[3:6].copy()
        
        mat_pel = data.xmat[pel_bid].reshape(3,3)
        xh = mat_pel[:,0].copy()
        xh[2] = 0.0
        norm = np.linalg.norm(xh)
        xh = xh/norm if norm>1e-8 else np.array([1.0,0.0,0.0])
        
        yh = np.array([0.0, 0.0, 1.0]) 
        zh = np.cross(xh, yh)
        zn = np.linalg.norm(zh)
        zh = zh/zn if zn>1e-8 else np.array([0.0,1.0,0.0])
        xh = np.cross(yh, zh)
        xh /= np.linalg.norm(xh)
        Rb    = np.stack((xh, yh, zh), axis=1)
        R_w2b = Rb.T
    else:
        origin = np.zeros(3)
        vpl_w  = np.zeros(3)
        vpa_w  = np.zeros(3)
        R_w2b  = np.eye(3)
    
    xpos = data.site_xpos
    xmat = data.site_xmat.reshape(-1,3,3)
    sd   = data.sensordata
    pos_dict:    Dict[str, np.ndarray] = {}
    vel_dict:    Dict[str, np.ndarray] = {}
    orient_dict: Dict[str, np.ndarray] = {}
    angvel_dict: Dict[str, np.ndarray] = {}
    
    for sid, name, (vadr,vdim), (gadr,gdim) in joint_sites:
        mat = xmat[sid]
        pw  = xpos[sid]
        vw  = mat.dot(sd[vadr:vadr+vdim])
        ww  = mat.dot(sd[gadr:gadr+gdim]) if (gadr is not None and gdim>=3) else np.zeros(3)
        
        pos    = R_w2b.dot(pw - origin).astype(np.float32)
        linvel = R_w2b.dot(vw - vpl_w).astype(np.float32)
        angvel = R_w2b.dot(ww - vpa_w).astype(np.float32)
        orient6= R_w2b.dot(mat).ravel()[:6].astype(np.float32)

        pos_dict[name]    = pos
        vel_dict[name]    = linvel
        orient_dict[name] = orient6
        angvel_dict[name] = angvel

    return {
        "pos":    pos_dict,
        "linvel": vel_dict,
        "orient": orient_dict,
        "angvel": angvel_dict
    }

def get_site_kinematics(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Retrieve non-pelvis joint-site origin positions, orientations(6D),
    linear velocities, and angular velocities, plus qpos/qvel
    (excluding pelvis DOFs), optionally in a yaw-only pelvis frame.

    Parameters
    ----------
    env : Any
        MuJoCo environment with:
          - model, data
          - data.site_xpos (nsite×3), data.site_xmat (nsite×9)
          - data.sensordata, model.sensor_* arrays
          - env.relative_pelvis : bool
          - env.pelvis_heading  : np.ndarray shape (3,)

    Returns
    -------
    joint_state : np.ndarray
        [ all pos(3×N), all ori(9×N), all lin_vel(3×N), all ang_vel(3×N),
          qpos[7:], qvel[6:] ]

    components : dict
        {
          'joint_space_pos'   : {site_name: np.ndarray(3,)},
          'joint_orientation' : {site_name: np.ndarray(3,2)},
          'joint_lin_vel'     : {site_name: np.ndarray(3,)},
          'joint_ang_vel'     : {site_name: np.ndarray(3,)},
          'joint_qpos'        : np.ndarray,  # qpos[7:]
          'joint_qvel'        : np.ndarray   # qvel[6:]
        }

    Raises
    ------
    ValueError
        If no joint-site velocimeter sensors found, or if qpos/qvel too short.
    """
    model = env.model
    data  = env.data
    eps   = 1e-8

    joint_sites = env._joint_sites
    pelvis_gyro = env._pelvis_gyro
    N = len(joint_sites)

    vpl_w = np.zeros(3)  # vel_pelvis_linear_inWorldFrame
    vpa_w = np.zeros(3)
    origin = np.zeros(3)
    R_w2b = np.eye(3)

    if env.relative_pelvis:
        pelvis_bid = model.body("pelvis").id
        pelvis_sid = model.site("pelvis_sensor").id
        origin     = data.xpos[pelvis_bid].copy()
        qvel       = data.qvel
        vpl_w      = qvel[0:3].copy()

        gadr, gdim = pelvis_gyro
        if gadr is not None and gdim >= 3:
            mat_p  = data.site_xmat[pelvis_sid].reshape(3,3)
            raw_pg = data.sensordata[gadr:gadr+gdim]
            vpa_w   = mat_p.dot(raw_pg)
        else:
            vpa_w   = qvel[3:6].copy()
            
        xh = env.pelvis_heading
        yh = np.array([0.0, 0.0, 1.0])
        zh = np.cross(xh, yh)
        zn = np.linalg.norm(zh)
        if zn < eps:
            zh = np.array([0.0, 1.0, 0.0])
        else:
            zh /= zn
        xh = np.cross(yh, zh); xh /= np.linalg.norm(xh)
        Rb    = np.stack((xh, yh, zh), axis=1)  # local→world
        R_w2b = Rb.T
        
    # preallocate arrays
    pos_arr     = np.empty((N, 3))
    ori_arr     = np.empty((N, 6))
    vel_lin_arr = np.empty((N, 3))
    vel_ang_arr = np.empty((N, 3))
    names       = [None] * N
    
    # fill kinematics
    for i, (sid, name, (vadr, vdim), (gadr, gdim)) in enumerate(joint_sites):
        names[i] = name

        pw  = data.site_xpos[sid]
        mat = data.site_xmat[sid].reshape(3, 3)
        raw_v = data.sensordata[vadr:vadr+vdim]
        vw    = mat.dot(raw_v)
        if gadr is not None and gdim >= 3:
            raw_w = data.sensordata[gadr:gadr+gdim]
            ww    = mat.dot(raw_w)
        else:
            ww    = np.zeros(3)

        v_rel = vw - vpl_w
        w_rel = ww - vpa_w
        vel_lin_arr[i] = R_w2b.dot(v_rel)
        vel_ang_arr[i] = R_w2b.dot(w_rel)
        
        pos_arr[i] = R_w2b.dot(pw - origin)
        ori_arr[i] = R_w2b.dot(mat).ravel()[:6]
        
        pos_arr[i][np.abs(pos_arr[i]) < eps]       = 0.0
        vel_lin_arr[i][np.abs(vel_lin_arr[i]) < eps] = 0.0
        vel_ang_arr[i][np.abs(vel_ang_arr[i]) < eps] = 0.0

    qpos = data.qpos; qvel = data.qvel
    if qpos.size <= 7 or qvel.size <= 6:
        raise ValueError("qpos/qvel too short to exclude pelvis DOFs.")
    qpj = qpos[7:].copy()
    qvj = qvel[6:].copy()

    total_len   = (3 + 6 + 3 + 3) * N + qpj.size + qvj.size
    joint_state = np.empty(total_len, dtype=float)
    off = 0
    joint_state[off:off + 3*N] = pos_arr.ravel()
    off += 3*N
    joint_state[off:off + 6*N] = ori_arr.ravel()
    off += 6*N
    joint_state[off:off + 3*N] = vel_lin_arr.ravel()
    off += 3*N
    joint_state[off:off + 3*N] = vel_ang_arr.ravel()
    off += 3*N
    joint_state[off:off + qpj.size] = qpj
    off += qpj.size
    joint_state[off:off + qvj.size] = qvj

    components: Dict[str, Any] = {
        'pos'   : {},
        'orient' : {},
        'linvel'     : {},
        'angvel'     : {},
        'joint_qpos'        : qpj,
        'joint_qvel'        : qvj
    }
    for i, nm in enumerate(names):
        components['pos'][nm]   = pos_arr[i].copy()
        components['orient'][nm] = ori_arr[i].copy()
        components['linvel'][nm]     = vel_lin_arr[i].copy()
        components['angvel'][nm]     = vel_ang_arr[i].copy()

    return joint_state, components

def get_traj_info(
        env: Any, 
        horizon: Optional[Union[int, List[int]]] = None
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Retrieve future reference-trajectory joint positions and velocities.

    Parameters
    ----------
    env : Any
        Environment instance with a `ref_traj` attribute of type `ReferenceTrajectories`.
        Must have, after reset():
          - env.ref_traj.qpos : 2D np.ndarray, shape (n_dofs, traj_frames)
          - env.ref_traj.qvel : 2D np.ndarray, shape (n_dofs, traj_frames)
          - env.ref_traj._pos : int, current frame index
          - env.ref_traj.increment : int, frame step per call
          - env.ref_traj.traj_frames : int, total number of frames in trajectory

    horizon : int or list[int], optional
        One or more frame-offsets ahead of the current frame:
          - If None, defaults to [1].
          - If int, fetches that many increments ahead.
          - If list of ints, fetches each listed offset.
        Offsets must be ≥ 0.

    Returns
    -------
    future_state : np.ndarray, shape=((qpos + qvel) * m,)

    components : dict
        {
          'future_qpos': np.ndarray, shape=(qpos, m),
          'future_qvel': np.ndarray, shape=(qvel, m)
        }

    Raises
    ------
    ValueError
        If ref_traj missing or mis-shaped, or horizon invalid.
    """
    if not hasattr(env, "ref_traj"):
        raise ValueError("Environment has no attribute 'ref_traj'.")
    ref = env.ref_traj
    try:
        qpos_all = ref.qpos
        qvel_all = ref.qvel
        current  = int(ref._pos)
        incr     = int(ref.increment)
        total    = int(ref.traj_frames)
    except Exception as e:
        raise ValueError(f"ref_traj is not properly set up: {e}")

    if qpos_all.ndim != 2 or qvel_all.ndim != 2:
        raise ValueError(f"qpos and qvel must be 2D arrays; got {qpos_all.ndim}D and {qvel_all.ndim}D")

    n_qpos, N1 = qpos_all.shape
    n_qvel, N2 = qvel_all.shape
    if N1 != total or N2 != total:
        raise ValueError(f"qpos/qvel second dimension must equal traj_frames ({total}); got {N1}, {N2}")

    if horizon is None:
        offsets = [1]
    elif isinstance(horizon, int):
        offsets = [horizon]
    elif isinstance(horizon, list) and all(isinstance(x, int) for x in horizon):
        offsets = horizon
    else:
        raise ValueError("`horizon` must be None, int, or list of ints.")
    if any(off < 0 for off in offsets):
        raise ValueError("All horizon offsets must be non-negative.")

    offs_arr = np.array(offsets, dtype=int)
    idxs = (current + offs_arr * incr) % total
    m = idxs.size
    future_qpos = np.take(qpos_all, idxs, axis=1)
    future_qvel = np.take(qvel_all, idxs, axis=1)

    len_pos = n_qpos * m
    len_vel = n_qvel * m
    future_state = np.empty(len_pos + len_vel, dtype=qpos_all.dtype)
    future_state[:len_pos]       = future_qpos.ravel(order='C')
    future_state[len_pos:]       = future_qvel.ravel(order='C')

    components = {
        'future_qpos': future_qpos,
        'future_qvel': future_qvel
    }
    return future_state, components

def get_COM_kinematics(env: any) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute the overall Center-of-Mass (COM) position and velocity of the model.

    Parameters
    ----------
    env : Any
        MuJoCo environment with:
          - env.data.subtree_com : (n_bodies,3) array
          - env.data.cvel        : (n_bodies,6) array
          - env.model.body_mass  : (n_bodies,) array
          - env.data.xpos        : (n_bodies,3) array
          - env.data.qvel        : (ndof,) array
          - env.relative_pelvis  : bool
          - env.pelvis_heading   : property returning unit (x,y,z) pelvis X‐axis proj

    Returns
    -------
    com_state : np.ndarray, shape=(6,) 
        [com_pos_x, com_pos_y, com_pos_z, com_vel_x, com_vel_y, com_vel_z]
        Or if relative_pelvis=True, the same in the pelvis frame.
    components : dict
        {
          'com_pos': np.ndarray, shape=(3,),
          'com_vel': np.ndarray, shape=(3,)
        }

    Raises
    ------
    ValueError
        On missing attributes or unexpected shapes.
    """
    model = env.model
    data = env.data
    
    # Retrieve COM position from subtree_com[0]
    try:
        com_pos_world = data.subtree_com[0]
    except Exception as e:
        raise ValueError(f"Failed to read data.subtree_com[0]: {e}")
    if com_pos_world.shape != (3,):
        raise ValueError(f"data.subtree_com[0] must be shape (3,), got {com_pos_world.shape}")
    
    if not hasattr(env, "_com_masses"):
        masses = np.asarray(model.body_mass[1:], dtype=float)
        if masses.ndim != 1 or masses.size == 0:
            raise ValueError(f"Unexpected model.body_mass shape: {model.body_mass.shape}")
        env._com_masses = masses
    else:
        masses = env._com_masses
        
    try:
        cvel = data.cvel[1:, 3:6]
    except Exception as e:
        raise ValueError(f"Failed to read data.cvel: {e}")
    if cvel.shape[0] != masses.size or cvel.shape[1] != 3:
        raise ValueError(f"data.cvel[1:,3:6] shape {cvel.shape} mismatches masses {masses.shape}")
    total_mass = masses.sum()  # recomputed each call
    if total_mass <= 0:
        raise ValueError(f"Nonpositive total mass: {total_mass}")
    com_vel_world = (cvel * masses[:, None]).sum(axis=0) / total_mass
    
    if getattr(env, "relative_pelvis", False):
        pelvis_id = model.body("pelvis").id
        origin    = data.xpos[pelvis_id]
        vpl_w     = data.qvel[:3]

        xh = env.pelvis_heading
        yh = np.array([0.0, 0.0, 1.0], dtype=float)
        zh = np.cross(xh, yh)
        zn = np.linalg.norm(zh)
        if zn < 1e-8:
            zh = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            zh /= zn
        xh = np.cross(yh, zh); xh /= np.linalg.norm(xh)
        R_basis = np.stack([xh, yh, zh], axis=1)
        R_w2b   = R_basis.T

        pos_rel_world = com_pos_world - origin
        vel_rel_world = com_vel_world - vpl_w
        com_pos = R_w2b.dot(pos_rel_world)
        com_vel = R_w2b.dot(vel_rel_world)
    else:
        com_pos = com_pos_world.copy()
        com_vel = com_vel_world.copy()

    com_state  = np.concatenate([com_pos, com_vel])
    components = {
        "com_pos": com_pos.copy(),
        "com_vel": com_vel.copy()
    }
    return com_state, components
    
def get_state(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build the full observation state by concatenating:
      - pelvis kinematics
      - joint kinematics
      - COM kinematics
      - ground reaction forces
      - binary foot-contact flags
      - future reference trajectory
      

    Parameters
    ----------
    env : Any
        
    Returns
    -------
    state : np.ndarray, shape=(M,)
        Concatenation of all sub-states and contacts.
    components : dict
        {
          'pelvis': dict,         # from get_pelvis_kinematics
          'joint': dict,          # from get_joint_kinematics
          'com': dict,            # from get_COM_kinematics
          'grf': dict,            # from get_GRF_info
          'foot_contacts': np.ndarray(shape=(2,))
          'traj': dict,           # from get_traj_info
        }

    Raises
    ------
    ValueError
        If any sub-state extraction or concatenation fails.
    """
    # Extract sub-states
    try:
        pelvis_state, pelvis_comp = get_pelvis_kinematics(env, use_free_joint=True)
    except Exception as e:
        raise ValueError(f"get_pelvis_kinematics failed: {e}")
    
    try:
        joint_state, joint_comp = get_site_kinematics(env)
    except Exception as e:
        raise ValueError(f"get_joint_kinematics failed: {e}")
        
    try:
        com_state, com_comp = get_COM_kinematics(env)
    except Exception as e:
        raise ValueError(f"get_joint_kinematics failed: {e}")
    
    try:
        grf_state, grf_comp = get_GRF_info(env)
    except Exception as e:
        raise ValueError(f"get_GRF_info failed: {e}")
        
    thresh = getattr(env, 'contact_threshold', 1e-2)
    try:
        foot_contacts = compute_foot_contacts(grf_comp, thresh)
    except Exception as e:
        raise ValueError(f"compute_foot_contacts failed: {e}")
    
    try:
        future_state, future_comp = get_traj_info(env)
    except Exception as e:
        raise ValueError(f"get_traj_info failed: {e}")
    
    # Precompute segment lengths
    L_p   = pelvis_state.size
    L_j   = joint_state.size
    L_com = com_state.size
    L_g   = grf_state.size
    L_ct  = foot_contacts.size  # ==2
    L_t   = future_state.size
    total_len = L_p + L_j + L_com + L_g + L_ct + L_t
    
    state = np.empty(total_len, dtype=pelvis_state.dtype)
    idx = 0
    state[idx:idx + L_p] = pelvis_state
    idx += L_p
    state[idx:idx + L_j] = joint_state
    idx += L_j
    state[idx:idx + L_com] = com_state
    idx += L_com
    state[idx:idx + L_g] = grf_state
    idx += L_g
    state[idx:idx + L_ct] = foot_contacts
    idx += L_ct
    state[idx:idx + L_t] = future_state
    
    
    components = {
        'pelvis':        pelvis_comp,
        'joint':         joint_comp,
        'com':           com_comp,
        'grf':           grf_comp,
        'foot_contacts': foot_contacts,
        'traj':          future_comp
    }

    return state, components

def get_state_size(env: Any) -> int:
    """
    Compute the dimensionality of the full state vector.

    Parameters
    ----------
    env : Any
        MuJoCo environment instance.

    Returns
    -------
    size : int
        Number of elements in the state returned by get_state().
    """
    state, _ = get_state(env)
    return state.size  

def get_leaf_kinematics(
        env: Any,
        site_names: List[str] = None
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract kinematics for a subset of leaf sites (e.g. feet, hands, head).

    Parameters
    ----------
    env : Any
    site_names : list[str], optional
        Names of the sensor sites to extract. Defaults to:
        ['calcn_r_sensor', 'calcn_l_sensor',
         'hand_r_sensor', 'hand_l_sensor',
         'head_sensor'].

    Returns
    -------
    term_state : np.ndarray, shape=(M,)
        Flattened vector concatenating for each site in `site_names`:
          [pos(3), ori(6), lin_vel(3), ang_vel(3)]
        in the **same order** as `site_names`.
    components : dict
        {
          'pos':      {name: np.ndarray(3,)},
          'orientation': {name: np.ndarray(6,)},
          'lin_vel':  {name: np.ndarray(3,)},
          'ang_vel':  {name: np.ndarray(3,)}
        }

    Raises
    ------
    ValueError
        If `get_joint_kinematics` fails, or any of the requested `site_names`
        is not present in the returned components.
    """
    if site_names is None:
        site_names = [
            'calcn_r_sensor',
            'calcn_l_sensor',
            'hand_r_sensor',
            'hand_l_sensor',
            'head_sensor'
        ]

    try:
        _, joint_comp = get_site_kinematics(env)
    except Exception as e:
        raise ValueError(f"get_site_kinematics failed: {e}")

    pos_map = joint_comp.get('joint_space_pos', {})
    ori_map = joint_comp.get('joint_orientation', {})
    lin_map = joint_comp.get('joint_lin_vel', {})
    ang_map = joint_comp.get('joint_ang_vel', {})

    missing = [n for n in site_names if n not in pos_map]
    if missing:
        raise ValueError(f"Requested terminal sites not found: {missing}")

    dims_per = 15
    N = len(site_names)
    term_state = np.empty(dims_per * N, dtype=float)

    comps = {
        'pos':         {},
        'orientation': {},
        'lin_vel':     {},
        'ang_vel':     {}
    }
    
    offset = 0
    for name in site_names:
        p = pos_map[name]           # shape (3,)
        o = ori_map[name]           # shape (6,)
        v = lin_map[name]           # shape (3,)
        w = ang_map[name]           # shape (3,)

        term_state[offset:offset+3]      = p
        term_state[offset+3:offset+9]    = o
        term_state[offset+9:offset+12]   = v
        term_state[offset+12:offset+15]  = w
        offset += dims_per

        comps['pos'][name]         = p.copy()
        comps['orientation'][name] = o.copy()
        comps['lin_vel'][name]     = v.copy()
        comps['ang_vel'][name]     = w.copy()

    return term_state, comps

def get_actuator_info(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pass

def get_muscle_info(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pass

def get_energy_info(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pass

def get_terrain_info(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pass

def get_state_extend(env: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pass

def get_state_extend_size(env: Any) -> int:
    pass
    

# foot slip velocity
