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
        MuJoCo body ID of the foot (e.g. calcaneus).
    geom_ids : List[int]
        List of geom IDs belonging to that foot.
    env : Any
        Environment with attributes:
          - data (MjData): simulation state, includes .xpos, .ncon, .contact
          - model (MjModel): for mj_contactForce
          - relative_pelvis (bool): whether to express in pelvis frame
          - pelvis_heading (np.ndarray, shape=(3,)): x-axis of pelvis frame

    Returns
    -------
    pos_local : np.ndarray, shape=(3,)
        Weighted‐average contact point in chosen frame.
    force_local : np.ndarray, shape=(3,)
        Net contact force in chosen frame.
    torque_local : np.ndarray, shape=(3,)
        Net contact torque about body reference in chosen frame.
    """
    data = env.data
    model = env.model
    
    # Origin offset
    ref_pos = data.xpos[body_id]
    
    if env.relative_pelvis:
        pid = model.body('pelvis').id
        origin = data.xpos[pid]
    else:
        origin = np.zeros(3, dtype=np.float64)
    
    ncon = data.ncon
    try:
        frames = data.contact.frame[:ncon].reshape(ncon, 3, 3)
    except Exception as e:
        raise ValueError(f"Invalid contact frame shape: {e}")
        
    g1 = data.contact.geom1[:ncon]
    g2 = data.contact.geom2[:ncon]
    pos_all = data.contact.pos[:ncon]
    foot_geoms = set(geom_ids)

    mask = np.array([i in foot_geoms for i in g1], bool) | \
           np.array([i in foot_geoms for i in g2], bool)
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return np.zeros(3), np.zeros(3), np.zeros(3)
    
    total_force  = np.zeros(3, dtype=float)
    total_torque = np.zeros(3, dtype=float)
    weighted_pos = np.zeros(3, dtype=float)
    tmp = np.zeros(6, dtype=float)
    
    forces = []
    positions = []
    tmp = np.zeros(6, dtype=np.float64)
    for i in idxs:
        try:
            mujoco.mj_contactForce(model, data, i, tmp)
        except RuntimeError:
            continue
        f_w = frames[i].T.dot(tmp[:3])  # world-frame force
        forces.append(f_w)
        positions.append(pos_all[i])

    if not forces:
        return np.zeros(3), np.zeros(3), np.zeros(3)
    
    forces_arr = np.stack(forces, axis=0)      # (K,3)
    pos_arr    = np.stack(positions, axis=0)   # (K,3)

    total_force  = forces_arr.sum(axis=0)                     # (3,)
    weighted_pos = (forces_arr * pos_arr).sum(axis=0)         # (3,)
    r_vecs       = pos_arr - ref_pos                          # (K,3)
    torques      = np.cross(r_vecs, forces_arr)               # (K,3)
    total_torque = torques.sum(axis=0)                        # (3,)

    if np.allclose(total_force, 0.0):
        return np.zeros(3), np.zeros(3), np.zeros(3)

    avg_contact = weighted_pos / total_force                  # (3,)
    contact_rel = avg_contact - origin                        # (3,)

    if env.relative_pelvis:
        xh = env.pelvis_heading.astype(np.float64)
        yh = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        zh = np.cross(xh, yh)
        zn = np.linalg.norm(zh)
        zh = zh / (zn if zn>1e-8 else 1.0)
        xh = np.cross(yh, zh)
        xh /= np.linalg.norm(xh)
        R = np.stack([xh, yh, zh], axis=1)
        Rw2l = R.T
        pos_local    = Rw2l.dot(contact_rel)
        force_local  = Rw2l.dot(total_force)
        torque_local = Rw2l.dot(total_torque)
    else:
        pos_local, force_local, torque_local = contact_rel, total_force, total_torque

    return pos_local, force_local, torque_local

def get_GRF_info(env: Any) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """
    Compute ground reaction forces (GRF) for both feet and return a concatenated vector.

    This function does not reshape or copy more than necessary and only caches the
    foot body IDs and their geom ID lists on first call.

    Parameters
    ----------
    env : Any
        Environment instance providing:
          - env.data.ncon : int, number of contacts
          - env.model.body(name).id : int, body IDs for 'calcn_r' and 'calcn_l'
          - env.model.geom(name).id : int, geom IDs under each foot
          - compute_grf(body_id, geom_ids, env) : function returning (pos, force, torque)

    Returns
    -------
    grf_vec : np.ndarray, shape=(18,)
        [r_pos(3), r_force(3), r_torque(3), l_pos(3), l_force(3), l_torque(3)]
    info : dict
        {
          'right': {'GRF_pos': np.ndarray(3,), 'GRF_force': np.ndarray(3,), 'GRF_torque': np.ndarray(3,)},
          'left':  {...}
        }

    Notes
    -----
    - If there are no contacts (env.data.ncon == 0), returns zeros.
    - Caches `env._grf_foot_cfg = {'right': (body_id, geom_ids), 'left': (...)}` on first call.
    """
    # early-out if no contacts
    if env.data.ncon == 0:
        zeros = np.zeros(3, dtype=np.float64)
        grf_vec = np.zeros(18, dtype=np.float64)
        return grf_vec, {
            'right': {'GRF_pos': zeros, 'GRF_force': zeros, 'GRF_torque': zeros},
            'left':  {'GRF_pos': zeros, 'GRF_force': zeros, 'GRF_torque': zeros}
        }
    
    # cache foot IDs & geom sets
    if not hasattr(env, "_grf_foot_cfg"):
        try:
            r_body = env.model.body('calcn_r').id
            l_body = env.model.body('calcn_l').id
            # adjust these names to match your model's foot‐geom names
            right_geoms = [
                env.model.geom(n).id for n in
                ['C_r_foot1','C_r_foot3','C_r_foot4','C_r_bofoot1','C_r_bofoot2']
            ]
            left_geoms = [
                env.model.geom(n).id for n in
                ['C_l_foot1','C_l_foot3','C_l_foot4','C_l_bofoot1','C_l_bofoot2']
            ]
        except Exception as e:
            raise ValueError(f"Failed to retrieve foot bodies or geoms: {e}")
        env._grf_foot_cfg = {
            'right': (r_body, right_geoms),
            'left':  (l_body,  left_geoms)
        }
    
    cfg = env._grf_foot_cfg
    grf_vec = np.empty(18, dtype=np.float64)
    info: Dict[str, Dict[str, np.ndarray]] = {'right': {}, 'left': {}}

    # fill in for each foot
    for idx, side in enumerate(('right', 'left')):
        body_id, geom_ids = cfg[side]
        pos, force, torque = compute_grf(body_id, geom_ids, env)
        base = idx * 9
        grf_vec[base:base+3]   = pos
        grf_vec[base+3:base+6] = force
        grf_vec[base+6:base+9] = torque
        info[side] = {
            'GRF_pos':    pos,
            'GRF_force':  force,
            'GRF_torque': torque
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
          - env.ref_traj.get_reference_trajectories() -> (ref_qpos, ref_qvel)
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
        ref_qpos, ref_qvel = env.ref_traj.get_reference_trajectories()
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
        raise ValueError(
            f"Expected len(qpos)>=7 and len(qvel)>=6, got "
            f"{qpos.shape[0]} and {qvel.shape[0]}"
        )
        
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

def compute_ref_site_kinematics(
        env: Any,
        include_orient_angvel: bool = False
        ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute reference‐trajectory kinematics for all non‐pelvis joint sites.

    This function:
      1. Retrieves the current 37→43‐DOF reference qpos/qvel from env.ref_traj.
      2. Converts them into MuJoCo qpos/qvel via `convert_ref_traj_*`.
      3. Reuses a single MjData buffer to run mj_forward.
      4. If `env.relative_pelvis` is True, computes a pelvis‐aligned frame (R_w2b),
         origin offset, and pelvis linear/angular base velocities.
      5. For each site in `env._joint_sites`, extracts:
         - world‐frame site position (`data.site_xpos`),
         - linear velocity via `site_xmat @ sensordata[vadr:vadr+vdim]`,
         - angular velocity via `site_xmat @ sensordata[gadr:gadr+gdim]`,
         - a 6D orientation matrix via the first two columns of `site_xmat`.
      6. Transforms all vectors into the chosen frame and casts to float32.

    Parameters
    ----------
    env : Any
        Must provide:
          - env.model           : mujoco.MjModel
          - env.ref_traj        : ReferenceTrajectories with get_reference_trajectories()
          - env._joint_sites    : List of tuples
                (site_id, site_name, (vadr,vdim), (gadr,gdim))
          - env._pelvis_gyro    : (gadr_pel, gdim_pel) for pelvis gyro sensor
          - env.relative_pelvis : bool
    include_orient_angvel : bool
        If False, won't return orient and angvel.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        {
          "pos":    {site_name: np.ndarray(3,)},
          "linvel": {site_name: np.ndarray(3,)},
          "orient": {site_name: np.ndarray(6,)},
          "angvel": {site_name: np.ndarray(3,)}
        }

    Raises
    ------
    ValueError
        If reference conversion fails or qpos/qvel length mismatches model.
    """
    model = env.model
    
    if not hasattr(env, "_ref_data"):
        env._ref_data = mujoco.MjData(model)
    data = env._ref_data
    
    joint_sites = env._joint_sites
    
    ref_qpos, ref_qvel = env.ref_traj.get_reference_trajectories()
    try:
        d_qpos = convert_ref_traj_qpos(ref_qpos) 
        d_qvel = convert_ref_traj_qvel(ref_qvel, ref_qpos)
    except Exception as e:
        raise ValueError(f"Reference conversion error: {e}")
    
    if d_qpos.size != model.nq or d_qvel.size != model.nv:
        raise ValueError(
            f"Expected qpos size={model.nq}, qvel size={model.nv}; "
            f"got {d_qpos.size}, {d_qvel.size}"
        )
        
    data.qpos[:] = d_qpos
    data.qvel[:] = d_qvel
    mujoco.mj_forward(model, data)
    
    if env.relative_pelvis:
        pel_bid = model.body("pelvis").id
        pel_sid = model.site("pelvis_sensor").id
        
        origin = data.xpos[pel_bid].copy()
        vpl_w  = data.qvel[0:3].copy()
        gadr_pel, gdim_pel = env._pelvis_gyro
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
        xh = xh / (norm if norm>1e-8 else 1.0)
        yh = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        zh = np.cross(xh, yh)
        zn = np.linalg.norm(zh)
        zh = zh / (zn if zn>1e-8 else 1.0)
        xh = np.cross(yh, zh)
        xh /= np.linalg.norm(xh)
        Rb   = np.stack((xh, yh, zh), axis=1)
        R_w2b = Rb.T
    else:
        origin = np.zeros(3, dtype=np.float64)
        vpl_w  = np.zeros(3, dtype=np.float64)
        vpa_w  = np.zeros(3, dtype=np.float64)
        R_w2b  = np.eye(3, dtype=np.float64)
    
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
        if gadr is not None and gdim >= 3:
            ww = mat.dot(sd[gadr:gadr+gdim])
        else:
            ww = np.zeros(3, dtype=np.float64)
        
        pos    = R_w2b.dot(pw - origin).astype(np.float32)
        linvel = R_w2b.dot(vw - vpl_w).astype(np.float32)
        angvel = R_w2b.dot(ww - vpa_w).astype(np.float32)
        orient6= R_w2b.dot(mat).ravel()[:6].astype(np.float32)

        pos_dict[name]    = pos
        vel_dict[name]    = linvel
        orient_dict[name] = orient6
        angvel_dict[name] = angvel
    
    result = {
        "pos":    pos_dict,
        "linvel": vel_dict,
    }

    if include_orient_angvel:
        result["orient"] = orient_dict
        result["angvel"] = angvel_dict

    return result

def get_site_kinematics(
        env: Any,
        include_orient_angvel: bool = False
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Retrieve kinematics for all non‐pelvis joint sites and the remaining qpos/qvel.

    This function:
      1. Builds a pelvis‐aligned frame (optional yaw only).
      2. Preallocates buffers for site positions, orientations (6D), linear and angular velocities.
      3. Vectorizes the MuJoCo site arrays (`site_xpos`, `site_xmat`) once outside the loop.
      4. For each site:
         - Computes world‐frame pw, vw, ww via matrix‐vector products.
         - Transforms into pelvis/local frame and zeroes near‐zero noise.
      5. Packs all site kinematics and the tail of qpos/qvel into a 1D state vector.
      6. Builds a components dict of per‐site views and joint qpos/qvel.

    Parameters
    ----------
    env : Any
        Must provide:
          - env.model           : mujoco.MjModel
          - env.data            : mujoco.MjData
          - env._joint_sites    : List of (site_id, name, (vadr,vdim), (gadr,gdim))
          - env._pelvis_gyro    : Tuple[gadr_pel, gdim_pel]
          - env.relative_pelvis : bool
          - env.pelvis_heading  : np.ndarray shape (3,)
    include_orient_angvel : bool
        If False, won't return orient and angvel.
        
    Returns
    -------
    joint_state : np.ndarray, shape=(M,)
        Concatenated [pos(3N), orient6(6N), linvel(3N), angvel(3N), qpos_tail, qvel_tail].
    components : dict
        {
          'pos'   : {name: ndarray(3,)},
          'orient': {name: ndarray(6,)},
          'linvel': {name: ndarray(3,)},
          'angvel': {name: ndarray(3,)},
          'joint_qpos': ndarray,
          'joint_qvel': ndarray
        }

    Raises
    ------
    ValueError
        If qpos/qvel lengths are insufficient or sensor data misaligned.
    """
    model = env.model
    data  = env.data
    eps   = 1e-8

    joint_sites = env._joint_sites
    gadr_pel, gdim_pel = env._pelvis_gyro
    N = len(joint_sites)

    if env.relative_pelvis:
        pelvis_bid = model.body("pelvis").id
        pelvis_sid = model.site("pelvis_sensor").id
        origin     = data.xpos[pelvis_bid].copy()
        qvel       = data.qvel
        vpl_w      = qvel[0:3].copy()
        
        if gadr_pel is not None and gdim_pel >= 3:
            mat_p = data.site_xmat[pelvis_sid].reshape(3,3)
            raw_pg = data.sensordata[gadr_pel:gadr_pel+gdim_pel]
            vpa_w = mat_p.dot(raw_pg)
        else:
            vpa_w = data.qvel[3:6].copy()
            
        xh = env.pelvis_heading
        yh = np.array([0.0,0.0,1.0], dtype=np.float64)
        zh = np.cross(xh, yh)
        zn = np.linalg.norm(zh)
        zh = zh / (zn if zn>eps else 1.0)
        xh = np.cross(yh, zh)
        xh /= np.linalg.norm(xh)
        Rb    = np.stack((xh, yh, zh), axis=1)     # body→world
        R_w2b = Rb.T
    else:
        origin = np.zeros(3, dtype=np.float64)
        vpl_w  = np.zeros(3, dtype=np.float64)
        vpa_w  = np.zeros(3, dtype=np.float64)
        R_w2b  = np.eye(3, dtype=np.float64)
    
    site_xpos = data.site_xpos
    site_xmat = data.site_xmat.reshape(-1,3,3)
    sensordata = data.sensordata
    
    # preallocate arrays
    pos_arr     = np.empty((N, 3), dtype=np.float32)
    orient_arr  = np.empty((N, 6), dtype=np.float32)
    linvel_arr  = np.empty((N, 3), dtype=np.float32)
    angvel_arr  = np.empty((N, 3), dtype=np.float32)
    names       = [None]*N
    
    # fill kinematics
    for i, (sid, name, (vadr, vdim), (gadr, gdim)) in enumerate(joint_sites):
        names[i] = name

        mat = site_xmat[sid]
        pw  = site_xpos[sid]
        vw  = mat.dot(sensordata[vadr:vadr+vdim])
        if gadr is not None and gdim>=3:
            ww = mat.dot(sensordata[gadr:gadr+gdim])
        else:
            ww = np.zeros(3, dtype=np.float64)

        p_loc = R_w2b.dot(pw - origin)
        v_loc = R_w2b.dot(vw - vpl_w)
        w_loc = R_w2b.dot(ww - vpa_w)
        
        ori6 = R_w2b.dot(mat).ravel()[:6]
        
        p_loc[np.abs(p_loc) < eps] = 0.0
        v_loc[np.abs(v_loc) < eps] = 0.0
        w_loc[np.abs(w_loc) < eps] = 0.0
        ori6[np.abs(ori6) < eps] = 0.0
        
        pos_arr[i]    = p_loc.astype(np.float32)
        linvel_arr[i] = v_loc.astype(np.float32)
        angvel_arr[i] = w_loc.astype(np.float32)
        orient_arr[i] = ori6.astype(np.float32)

    qpos = data.qpos; qvel = data.qvel
    if qpos.size <= 7 or qvel.size <= 6:
        raise ValueError("qpos/qvel too short to exclude pelvis DOFs.")
    qpj = qpos[7:].astype(np.float32, copy=True)
    qvj = qvel[6:].astype(np.float32, copy=True)

    if include_orient_angvel:
        L = 3*N + 6*N + 3*N + 3*N
    else:
        L = 3*N + 3*N

    joint_state = np.empty(L + qpj.size + qvj.size, dtype=np.float32)
    off = 0
    joint_state[off:off+3*N] = pos_arr.ravel()
    off += 3*N
    if include_orient_angvel:
        joint_state[off:off+6*N] = orient_arr.ravel()
        off += 6*N
    joint_state[off:off+3*N] = linvel_arr.ravel()
    off += 3*N
    if include_orient_angvel:
        joint_state[off:off+3*N] = angvel_arr.ravel()
        off += 3*N
    joint_state[off:off+qpj.size] = qpj
    off += qpj.size
    joint_state[off:off+qvj.size] = qvj

    # --- Prepare components dict ---
    components: Dict[str, Any] = {
        'pos':        {name: pos_arr[i]   for i, name in enumerate(names)},
        'linvel':     {name: linvel_arr[i] for i, name in enumerate(names)},
        'joint_qpos': qpj,
        'joint_qvel': qvj
    }
    if include_orient_angvel:
        components['orient'] = {name: orient_arr[i] for i, name in enumerate(names)}
        components['angvel'] = {name: angvel_arr[i]  for i, name in enumerate(names)}

    return joint_state, components

def get_traj_info(
        env: Any, 
        horizon: Optional[Union[int, List[int]]] = None,
        remove_root: bool = False,
        center_root: bool = False
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Retrieve future reference-trajectory joint positions and velocities.

    Given the current frame in env.ref_traj, this returns the qpos and qvel
    at one or more future offsets, packed both as a flat state vector and
    as separate arrays for positions and velocities.

    Parameters
    ----------
    env : Any
        Environment with attribute `ref_traj` of type ReferenceTrajectories,
        which after reset() must provide:
          - ref_traj.qpos          : ndarray, shape (n_dofs, traj_frames)
          - ref_traj.qvel          : ndarray, shape (n_dofs, traj_frames)
          - ref_traj._pos          : int, current frame index
          - ref_traj.increment     : int, frames to advance per offset unit
          - ref_traj.traj_frames   : int, total frames in the trajectory
    horizon : int or list[int], optional
        Frame offsets ahead of the current frame.  
        None -> [1], int -> [horizon], list -> itself.
    remove_root : bool, default=False
        If True, **drop** the first three DOFs (pelvis x,y,z) from each future qpos/qvel.
    center_root : bool, default=False
        If True, **subtract** the current-frame pelvis position from all future qpos
        (making root motion relative to the current root). Overrides remove_root.

    Returns
    -------
    future_state : np.ndarray, shape=(n_qpos*m + n_qvel*m,)
        Concatenation of future_qpos.ravel() then future_qvel.ravel().
    components : dict
        {
          'future_qpos': ndarray, shape (n_qpos', m),
          'future_qvel': ndarray, shape (n_qvel', m)
        }
        where m = number of offsets, and
        n_qpos' = n_qpos    if not remove_root
               = n_qpos-3  if remove_root
               = n_qpos    if center_root (we keep dims but zero-center)
        similarly for n_qvel'.

    Raises
    ------
    ValueError
        If ref_traj is missing or misconfigured, or if horizon is invalid.
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

    T1 = qpos_all.shape[1]
    T2 = qvel_all.shape[1]
    if T1 != total or T2 != total:
        raise ValueError(f"Expected traj_frames={total}, got qpos.shape[1]={T1}, qvel.shape[1]={T2}")

    if horizon is None:
        offsets = np.array([2], dtype=int)
    elif isinstance(horizon, int):
        offsets = np.array([horizon], dtype=int)
    elif isinstance(horizon, (list, tuple)):
        if not all(isinstance(x, int) and x >= 0 for x in horizon):
            raise ValueError("All horizon offsets must be non-negative integers")
        offsets = np.array(horizon, dtype=int)
    else:
        raise ValueError("`horizon` must be None, int, or list/tuple of ints")
    
    m = offsets.size
    idxs = (current + offsets * incr) % total  # shape (m,)

    stacked = np.vstack((qpos_all, qvel_all))
    future_both = np.take(stacked, idxs, axis=1)
    n_qpos = qpos_all.shape[0]
    future_qpos = future_both[:n_qpos, :]
    future_qvel = future_both[n_qpos:, :]

    if center_root:
        root0 = qpos_all[0:3, current][:, None]  # (3,1)
        future_qpos[0:3, :] -= root0
    elif remove_root:
        future_qpos = future_qpos[3:, :]

    n_qpos2, _ = future_qpos.shape
    n_qvel2, _ = future_qvel.shape
    len_pos = n_qpos2 * m
    len_vel = n_qvel2 * m
    future_state = np.empty(len_pos + len_vel, dtype=qpos_all.dtype)
    future_state[:len_pos]        = future_qpos.ravel(order='C')
    future_state[len_pos:]        = future_qvel.ravel(order='C')

    components = {
        'future_qpos': future_qpos,
        'future_qvel': future_qvel
    }
    return future_state, components

def get_COM_kinematics(
        env: any
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute the model’s center-of-mass (COM) position and velocity.

    This function weights each body’s subtree COM and spatial velocity
    by its mass to yield the overall COM kinematics. Optionally expresses
    results in a pelvis-aligned frame.

    Parameters
    ----------
    env : Any
        Must provide:
          - env.model           : mujoco.MjModel
          - env.data            : mujoco.MjData with fields:
                * subtree_com    : ndarray, shape (n_bodies, 3)
                * cvel           : ndarray, shape (n_bodies, 6),
                                    columns [ωx,ωy,ωz, vx,vy,vz]
                * xpos           : ndarray, shape (n_bodies, 3)
          - env.relative_pelvis : bool
          - env.pelvis_heading  : ndarray, shape (3,)

    Returns
    -------
    com_state : ndarray, shape (6,)
        [com_pos_x, com_pos_y, com_pos_z, com_vel_x, com_vel_y, com_vel_z].
        If `env.relative_pelvis` is True, values are in the pelvis frame.

    components : dict
        {
          "com_pos": ndarray shape (3,),
          "com_vel": ndarray shape (3,)
        }

    Raises
    ------
    ValueError
        If required data are missing or have unexpected shapes, or total mass ≤ 0.
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
    
    masses = getattr(env, "_com_masses", None)
    if masses is None:
        body_mass = np.asarray(model.body_mass, dtype=np.float64)
        if body_mass.ndim != 1 or body_mass.size < 2:
            raise ValueError(f"Unexpected model.body_mass shape: {model.body_mass.shape}")
        # skip world body at index 0
        masses = body_mass[1:]
        env._com_masses = masses
    total_mass = float(masses.sum())
    if total_mass <= 0:
        raise ValueError(f"Nonpositive total mass: {total_mass}")
        
    try:
        cvel = data.cvel[1:, 3:6]
    except Exception as e:
        raise ValueError(f"Failed to read data.cvel: {e}")
    if cvel.shape[0] != masses.size or cvel.shape[1] != 3:
        raise ValueError(f"data.cvel[1:,3:6] shape {cvel.shape} mismatches masses {masses.shape}")
    
    com_vel_world = (cvel * masses[:, None]).sum(axis=0) / total_mass
    
    if getattr(env, "relative_pelvis", False):
        pid    = model.body("pelvis").id
        origin = data.xpos[pid]
        vpl_w  = data.qvel[0:3] 

        xh = env.pelvis_heading.astype(np.float64)
        yh = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        zh = np.cross(xh, yh)
        zn = np.linalg.norm(zh)
        zh = zh / (zn if zn>1e-8 else 1.0)
        xh = np.cross(yh, zh)
        xh /= np.linalg.norm(xh)
        R_basis = np.stack([xh, yh, zh], axis=1)
        R_w2b   = R_basis.T

        pos_local = R_w2b.dot(com_pos_world - origin)
        vel_local = R_w2b.dot(com_vel_world - vpl_w)
    else:
        pos_local = com_pos_world.copy()
        vel_local = com_vel_world.copy()

    com_state = np.empty(6, dtype=np.float64)
    com_state[0:3] = pos_local
    com_state[3:6] = vel_local

    components = {
        "com_pos": pos_local.copy(),
        "com_vel": vel_local.copy()
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
        Must provide the following functions/attributes:
          - get_pelvis_kinematics(env, use_free_joint=True)
          - get_site_kinematics(env)
          - get_COM_kinematics(env)
          - get_GRF_info(env)
          - compute_foot_contacts(grf_comp, threshold)
          - get_traj_info(env)
          - env.contact_threshold : float (optional)

    Returns
    -------
    state : np.ndarray, shape=(M,)
        Flat vector concatenating all sub-states in the order above.
    components : dict
        {
          'pelvis'        : dict,      # from get_pelvis_kinematics
          'joint'         : dict,      # from get_site_kinematics
          'com'           : dict,      # from get_COM_kinematics
          'grf'           : dict,      # from get_GRF_info
          'foot_contacts' : np.ndarray,# shape=(2,)
          'traj'          : dict       # from get_traj_info
        }

    Raises
    ------
    ValueError
        If any sub-state extraction fails.
    """
    gp = get_pelvis_kinematics
    gs = get_site_kinematics
    gc = get_COM_kinematics
    gg = get_GRF_info
    cf = compute_foot_contacts
    gt = get_traj_info

    # 1) Pelvis
    try:
        pelvis_state, pelvis_comp = gp(env, use_free_joint=True)
        # pelvis_state = pelvis_state[2:]  # remove x, y
    except Exception as e:
        raise ValueError(f"get_pelvis_kinematics failed: {e}")

    # 2) Joint sites
    try:
        joint_state, joint_comp = gs(env, include_orient_angvel=False)
    except Exception as e:
        raise ValueError(f"get_site_kinematics failed: {e}")

    # # 3) COM
    # try:
    #     com_state, com_comp = gc(env)
    #     # com_state = com_state[2:]  # remove x, y
    # except Exception as e:
    #     raise ValueError(f"get_COM_kinematics failed: {e}")

    # 4) GRF
    try:
        grf_state, grf_comp = gg(env)
    except Exception as e:
        raise ValueError(f"get_GRF_info failed: {e}")

    # 5) Foot contacts
    threshold = getattr(env, "contact_threshold", 1e-2)
    try:
        foot_contacts = cf(grf_comp, threshold)
    except Exception as e:
        raise ValueError(f"compute_foot_contacts failed: {e}")

    # 6) Future trajectory
    try:
        future_state, future_comp = gt(env, horizon=[2], remove_root=True)
    except Exception as e:
        raise ValueError(f"get_traj_info failed: {e}")

    # --- Concatenate into one flat state vector ---
    sub_states: List[np.ndarray] = [
        pelvis_state,
        joint_state,
        # com_state,
        grf_state,
        foot_contacts,
        future_state
    ]

    # Compute total length
    total_len = sum(s.size for s in sub_states)

    # Preallocate
    state = np.empty(total_len, dtype=pelvis_state.dtype)

    # Fill in order
    idx = 0
    for arr in sub_states:
        length = arr.size
        state[idx : idx + length] = arr.ravel(order='C')
        idx += length

    # Package components
    components: Dict[str, Any] = {
        'pelvis':        pelvis_comp,
        'joint':         joint_comp,
        # 'com':           com_comp,
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
